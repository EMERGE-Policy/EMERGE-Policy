"""Thin adapter around the official VGGT package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from external_model_server.localization_types import (
    VGGTAlignedCamera,
    VGGTAlignment,
    VGGTResult,
    VGGTViewInput,
    VGGTViewPrediction,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


class VGGTBackend:
    """Load VGGT, run inference, align outputs, and save them."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self._model: Any | None = None
        self._device: str | None = None

    def infer(
        self,
        views: list[VGGTViewInput],
        *,
        reference_view: str,
    ) -> VGGTResult:
        return self.infer_batch([(views, reference_view)])[0]

    def infer_batch(
        self,
        scenes: Sequence[tuple[list[VGGTViewInput], str]],
    ) -> list[VGGTResult]:
        """Run compatible calibrated scenes in one VGGT forward pass."""
        if not scenes:
            raise ValueError("VGGTBackend.infer_batch requires at least one scene")
        for views, reference_view in scenes:
            self._validate_views(views, reference_view=reference_view)

        view_counts = {len(views) for views, _reference_view in scenes}
        image_shapes = {
            tuple(int(value) for value in np.asarray(views[0].rgb).shape)
            for views, _reference_view in scenes
        }
        if len(view_counts) != 1 or len(image_shapes) != 1:
            raise ValueError(
                "VGGT batched scenes must share one view count and RGB shape"
            )

        device = str(self.config.get("device", "cuda")).strip().lower()
        if device != "cuda":
            raise RuntimeError(f"VGGTBackend currently requires device='cuda', got {device!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("VGGTBackend requires CUDA but torch.cuda.is_available() is False")

        precision = str(self.config.get("precision", "fp16")).strip().lower()
        if precision == "fp16":
            dtype = torch.float16
        elif precision == "bf16":
            major, _minor = torch.cuda.get_device_capability()
            if major < 8:
                raise RuntimeError("VGGTBackend precision='bf16' requires an Ampere-or-newer CUDA GPU")
            dtype = torch.bfloat16
        else:
            raise RuntimeError(f"Unsupported VGGT precision: {precision!r}")

        model = self._load_model(device)
        raw_images = torch.stack(
            [
                torch.stack(
                    [
                        torch.from_numpy(
                            np.array(view.rgb, dtype=np.uint8, order="C", copy=True)
                        )
                        .to(device=device, dtype=torch.float32)
                        .permute(2, 0, 1)
                        / 255.0
                        for view in views
                    ],
                    dim=0,
                )
                for views, _reference_view in scenes
            ],
            dim=0,
        )
        source_h, source_w = int(raw_images.shape[-2]), int(raw_images.shape[-1])
        processed_rgb = (
            (raw_images.clamp(0.0, 1.0) * 255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 1, 3, 4, 2)
            .cpu()
            .numpy()
        )

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=dtype):
                predictions = model(raw_images)

        pred_extrinsics, pred_intrinsics = pose_encoding_to_extri_intri(
            predictions["pose_enc"],
            image_size_hw=raw_images.shape[-2:],
        )
        pred_extrinsics = pred_extrinsics.detach().cpu().numpy()
        pred_intrinsics = pred_intrinsics.detach().cpu().numpy()
        depth = self._normalize_spatial_batch(
            predictions["depth"],
            source_h=source_h,
            source_w=source_w,
            name="depth",
        )
        depth_conf = predictions.get("depth_conf")
        if depth_conf is not None:
            depth_conf = self._normalize_spatial_batch(
                depth_conf,
                source_h=source_h,
                source_w=source_w,
                name="depth_conf",
            )

        return [
            self._build_result(
                views=views,
                reference_view=reference_view,
                processed_rgb=processed_rgb[index],
                pred_extrinsics=pred_extrinsics[index],
                pred_intrinsics=pred_intrinsics[index],
                depth=depth[index],
                depth_conf=None if depth_conf is None else depth_conf[index],
            )
            for index, (views, reference_view) in enumerate(scenes)
        ]

    @staticmethod
    def _normalize_spatial_batch(
        value: Any,
        *,
        source_h: int,
        source_w: int,
        name: str,
    ) -> np.ndarray:
        array = value.detach().cpu().numpy()
        if array.ndim == 5 and array.shape[-1] == 1:
            array = array[..., 0]
        elif array.ndim == 5 and array.shape[2] == 1:
            array = array[:, :, 0]
        if array.ndim != 4:
            raise RuntimeError(
                f"VGGT {name} must have shape BxSxHxW (optionally with one channel), got {array.shape}"
            )
        return array[:, :, :source_h, :source_w]

    def _build_result(
        self,
        *,
        views: list[VGGTViewInput],
        reference_view: str,
        processed_rgb: np.ndarray,
        pred_extrinsics: np.ndarray,
        pred_intrinsics: np.ndarray,
        depth: np.ndarray,
        depth_conf: np.ndarray | None,
    ) -> VGGTResult:
        observed_intrinsics = [
            np.asarray(view.intrinsics, dtype=np.float64).copy()
            for view in views
        ]
        observed_centers = np.stack(
            [
                np.asarray(view.T_world_camera, dtype=np.float64)[:3, 3]
                for view in views
            ],
            axis=0,
        )
        predicted_centers = np.stack(
            [
                self._camera_center_from_extrinsic(extrinsic)
                for extrinsic in pred_extrinsics
            ],
            axis=0,
        )
        alignment = self._calibrate_depth_scale(
            predicted_centers=predicted_centers,
            observed_centers=observed_centers,
        )
        self._validate_alignment_quality(alignment)

        cameras: list[VGGTAlignedCamera] = []
        view_predictions: list[VGGTViewPrediction] = []
        for idx, view in enumerate(views):
            intrinsics_observed = observed_intrinsics[idx]
            cameras.append(
                VGGTAlignedCamera(
                    name=view.name,
                    intrinsics_pred=np.asarray(pred_intrinsics[idx], dtype=np.float64),
                    intrinsics_observed=intrinsics_observed,
                    extrinsics_pred=np.asarray(pred_extrinsics[idx], dtype=np.float64),
                    T_world_camera_observed=np.asarray(view.T_world_camera, dtype=np.float64),
                )
            )
            view_depth_m = (
                np.asarray(depth[idx]).astype(np.float64, copy=False) * alignment.depth_scale
            ).astype(np.float32)
            point_map_world = self._unproject_depth_to_world(
                view_depth_m,
                intrinsics=intrinsics_observed,
                T_world_camera=np.asarray(view.T_world_camera, dtype=np.float64),
            )
            view_depth_conf = (
                None
                if depth_conf is None
                else np.asarray(depth_conf[idx]).astype(np.float32, copy=False)
            )
            view_predictions.append(
                VGGTViewPrediction(
                    name=view.name,
                    rgb=np.ascontiguousarray(processed_rgb[idx]),
                    point_map_world=point_map_world,
                    depth_m=view_depth_m,
                    depth_conf=view_depth_conf,
                    point_conf=view_depth_conf,
                )
            )

        reference = next(view for view in view_predictions if view.name == reference_view)
        return VGGTResult(
            reference_view=reference_view,
            point_map_world=reference.point_map_world,
            depth_m=reference.depth_m,
            depth_conf=reference.depth_conf,
            point_conf=reference.point_conf,
            cameras=cameras,
            alignment=alignment,
            views=view_predictions,
        )

    @staticmethod
    def _validate_views(
        views: list[VGGTViewInput],
        *,
        reference_view: str,
    ) -> None:
        if len(views) < 2:
            raise ValueError(
                "metric VGGT reconstruction requires at least two calibrated views"
            )
        names = [str(view.name).strip() for view in views]
        if any(not name for name in names):
            raise ValueError("VGGT view names must be non-empty")
        if len(set(names)) != len(names):
            raise ValueError(f"VGGT view names must be unique, got {names}")
        if reference_view not in names:
            raise ValueError(
                f"reference_view {reference_view!r} is not present in views {names}"
            )

        image_shapes: set[tuple[int, ...]] = set()
        for view in views:
            rgb = np.asarray(view.rgb)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(
                    f"view {view.name!r} RGB must have shape HxWx3, got {rgb.shape}"
                )
            height, width = int(rgb.shape[0]), int(rgb.shape[1])
            if height % 14 != 0 or width % 14 != 0:
                raise ValueError(
                    "VGGT requires every RGB view height and width to be divisible "
                    f"by patch size 14; view {view.name!r} has "
                    f"height={height}, width={width}"
                )
            image_shapes.add(tuple(int(value) for value in rgb.shape))

            intrinsics = np.asarray(view.intrinsics, dtype=np.float64)
            if intrinsics.shape != (3, 3) or not np.all(
                np.isfinite(intrinsics)
            ):
                raise ValueError(
                    f"view {view.name!r} intrinsics must be a finite 3x3 matrix"
                )
            if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
                raise ValueError(
                    f"view {view.name!r} intrinsics must have positive focal lengths"
                )

            transform = np.asarray(view.T_world_camera, dtype=np.float64)
            if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
                raise ValueError(
                    f"view {view.name!r} T_world_camera must be a finite 4x4 matrix"
                )
            if not np.allclose(
                transform[3],
                [0.0, 0.0, 0.0, 1.0],
                atol=1e-8,
            ):
                raise ValueError(
                    f"view {view.name!r} T_world_camera has an invalid last row"
                )
            rotation = transform[:3, :3]
            if not np.allclose(
                rotation.T @ rotation,
                np.eye(3),
                atol=1e-5,
            ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
                raise ValueError(
                    f"view {view.name!r} T_world_camera rotation is not orthonormal"
                )

        if len(image_shapes) != 1:
            raise ValueError(
                f"all VGGT RGB views must share one shape, got {sorted(image_shapes)}"
            )

    @staticmethod
    def _camera_center_from_extrinsic(extrinsic: np.ndarray) -> np.ndarray:
        extrinsic = np.asarray(extrinsic, dtype=np.float64)
        if extrinsic.shape != (3, 4):
            raise ValueError(
                f"VGGT extrinsic must have shape 3x4, got {extrinsic.shape}"
            )
        rotation = extrinsic[:, :3]
        translation = extrinsic[:, 3]
        return -(rotation.T @ translation)

    @staticmethod
    def _calibrate_depth_scale(
        *,
        predicted_centers: np.ndarray,
        observed_centers: np.ndarray,
    ) -> VGGTAlignment:
        predicted_centers = np.asarray(predicted_centers, dtype=np.float64)
        observed_centers = np.asarray(observed_centers, dtype=np.float64)
        if (
            predicted_centers.shape != observed_centers.shape
            or predicted_centers.ndim != 2
            or predicted_centers.shape[1] != 3
        ):
            raise ValueError(
                "predicted and observed camera centers must both have shape Nx3"
            )

        scale_samples: list[float] = []
        for left in range(len(predicted_centers)):
            for right in range(left + 1, len(predicted_centers)):
                predicted_baseline = float(
                    np.linalg.norm(
                        predicted_centers[left] - predicted_centers[right]
                    )
                )
                observed_baseline = float(
                    np.linalg.norm(
                        observed_centers[left] - observed_centers[right]
                    )
                )
                if predicted_baseline <= 1e-8 or observed_baseline <= 1e-8:
                    continue
                scale_samples.append(observed_baseline / predicted_baseline)
        if not scale_samples:
            raise RuntimeError(
                "cannot calibrate VGGT depth scale from degenerate camera baselines"
            )

        samples = np.asarray(scale_samples, dtype=np.float64)
        depth_scale = float(np.median(samples))
        if not np.isfinite(depth_scale) or depth_scale <= 0.0:
            raise RuntimeError(
                f"VGGT produced an invalid depth scale: {depth_scale}"
            )
        relative_mad = float(
            np.median(np.abs(samples - depth_scale)) / depth_scale
        )

        predicted_mean = predicted_centers.mean(axis=0)
        observed_mean = observed_centers.mean(axis=0)
        predicted_centered = predicted_centers - predicted_mean
        observed_centered = observed_centers - observed_mean
        covariance = predicted_centered.T @ observed_centered
        left_vectors, _singular_values, right_vectors_t = np.linalg.svd(
            covariance
        )
        correction = np.eye(3, dtype=np.float64)
        if np.linalg.det(right_vectors_t.T @ left_vectors.T) < 0.0:
            correction[-1, -1] = -1.0
        rotation = (
            right_vectors_t.T @ correction @ left_vectors.T
        )
        translation = observed_mean - depth_scale * (
            rotation @ predicted_mean
        )
        aligned_centers = (
            depth_scale * (predicted_centers @ rotation.T) + translation
        )
        errors = np.linalg.norm(
            aligned_centers - observed_centers,
            axis=1,
        )
        return VGGTAlignment(
            method="known_camera_depth_unprojection",
            depth_scale=depth_scale,
            baseline_pair_count=len(scale_samples),
            baseline_scale_median=depth_scale,
            baseline_scale_relative_mad=relative_mad,
            rms_camera_center_error_m=float(
                np.sqrt(np.mean(errors * errors))
            ),
            max_camera_center_error_m=float(np.max(errors)),
        )

    def _validate_alignment_quality(self, alignment: VGGTAlignment) -> None:
        max_relative_mad = float(
            self.config.get("max_baseline_scale_relative_mad", 0.25)
        )
        if alignment.baseline_scale_relative_mad > max_relative_mad:
            raise RuntimeError(
                "VGGT camera baselines disagree on metric depth scale: "
                f"relative_mad={alignment.baseline_scale_relative_mad:.4f}, "
                f"limit={max_relative_mad:.4f}"
            )
        max_center_rms = float(
            self.config.get("max_camera_center_rms_error_m", 0.10)
        )
        if alignment.rms_camera_center_error_m > max_center_rms:
            raise RuntimeError(
                "VGGT predicted camera geometry is inconsistent with calibration: "
                f"rms={alignment.rms_camera_center_error_m:.4f}m, "
                f"limit={max_center_rms:.4f}m"
            )

    @staticmethod
    def _unproject_depth_to_world(
        depth_m: np.ndarray,
        *,
        intrinsics: np.ndarray,
        T_world_camera: np.ndarray,
    ) -> np.ndarray:
        depth_m = np.asarray(depth_m, dtype=np.float64)
        if depth_m.ndim != 2:
            raise ValueError(
                f"metric depth must have shape HxW, got {depth_m.shape}"
            )
        intrinsics = np.asarray(intrinsics, dtype=np.float64)
        transform = np.asarray(T_world_camera, dtype=np.float64)
        height, width = depth_m.shape
        pixel_y, pixel_x = np.indices((height, width), dtype=np.float64)
        camera_x = (
            (pixel_x - intrinsics[0, 2])
            / intrinsics[0, 0]
            * depth_m
        )
        camera_y = (
            (pixel_y - intrinsics[1, 2])
            / intrinsics[1, 1]
            * depth_m
        )
        points_camera = np.stack(
            [camera_x, camera_y, depth_m],
            axis=-1,
        )
        points_world = (
            points_camera @ transform[:3, :3].T
            + transform[:3, 3]
        )
        invalid = ~np.isfinite(depth_m) | (depth_m <= 0.0)
        points_world[invalid] = np.nan
        return points_world.astype(np.float32)

    def save_result(
        self,
        result: VGGTResult,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (_REPO_ROOT / output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        views: list[dict[str, Any]] = []
        for view in result.views:
            point_map_path = output_dir / f"{view.name}_point_map_world.npy"
            depth_path = output_dir / f"{view.name}_depth_m.npy"
            np.save(point_map_path, view.point_map_world)
            np.save(depth_path, view.depth_m)
            entry: dict[str, Any] = {
                "name": view.name,
                "point_map_world_path": point_map_path.name,
                "depth_m_path": depth_path.name,
            }
            if view.depth_conf is not None:
                path = output_dir / f"{view.name}_depth_conf.npy"
                np.save(path, view.depth_conf)
                entry["depth_conf_path"] = path.name

            ply_path = output_dir / f"{view.name}_pointcloud_world.ply"
            self._write_colored_pointcloud_ply(
                ply_path,
                view.point_map_world,
                view.rgb,
            )
            entry["pointcloud_world_ply"] = ply_path.name
            views.append(entry)

        merged_points_parts: list[np.ndarray] = []
        merged_colors_parts: list[np.ndarray] = []
        for view in result.views:
            points = view.point_map_world.reshape(-1, 3)
            colors = view.rgb.reshape(-1, 3)
            finite = np.isfinite(points).all(axis=1)
            merged_points_parts.append(points[finite])
            merged_colors_parts.append(colors[finite])
        merged_points_np = np.concatenate(merged_points_parts, axis=0)
        merged_colors_np = np.concatenate(merged_colors_parts, axis=0)
        merged_npy_path = output_dir / "merged_pointcloud_world.npy"
        np.save(merged_npy_path, merged_points_np)
        merged_ply_path = output_dir / "merged_pointcloud_world.ply"
        self._write_colored_pointcloud_ply(
            merged_ply_path,
            merged_points_np,
            merged_colors_np,
        )

        cameras = []
        for camera in result.cameras:
            camera_path = output_dir / f"{camera.name}_camera.json"
            camera_path.write_text(
                json.dumps(
                    {
                        "name": camera.name,
                        "intrinsics_pred": camera.intrinsics_pred.tolist(),
                        "intrinsics_observed": (
                            camera.intrinsics_observed.tolist()
                        ),
                        "extrinsics_pred": camera.extrinsics_pred.tolist(),
                        "T_world_camera_observed": (
                            camera.T_world_camera_observed.tolist()
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            cameras.append(
                {"name": camera.name, "camera_path": camera_path.name}
            )

        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "reference_view": result.reference_view,
                    "merged_pointcloud_world_path": merged_npy_path.name,
                    "merged_pointcloud_world_ply": merged_ply_path.name,
                    "alignment": result.alignment.to_dict(),
                    "views": views,
                    "cameras": cameras,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "output_dir": str(output_dir),
            "summary_path": str(summary_path),
            "reference_view": result.reference_view,
        }

    @staticmethod
    def _write_colored_pointcloud_ply(
        path: Path,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        """Write an ASCII PLY with xyz + rgb, dropping non-finite points."""
        points = np.asarray(points)
        colors = np.asarray(colors, dtype=np.uint8)
        if points.shape != colors.shape or points.ndim < 2 or points.shape[-1] != 3:
            raise ValueError(
                f"point cloud points and colors must have matching (..., 3) shapes, "
                f"got {points.shape} and {colors.shape}"
            )
        points = points.reshape(-1, 3)
        colors = colors.reshape(-1, 3)
        finite = np.isfinite(points).all(axis=1)
        points = points[finite].astype(np.float32)
        colors = colors[finite]
        with path.open("w", encoding="ascii") as handle:
            handle.write("ply\n")
            handle.write("format ascii 1.0\n")
            handle.write(f"element vertex {len(points)}\n")
            handle.write("property float x\n")
            handle.write("property float y\n")
            handle.write("property float z\n")
            handle.write("property uchar red\n")
            handle.write("property uchar green\n")
            handle.write("property uchar blue\n")
            handle.write("end_header\n")
            for point, color in zip(points, colors, strict=True):
                handle.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
                )

    def _load_model(self, device: str) -> Any:
        if self._model is not None and self._device == device:
            return self._model

        model_path = Path(str(self.config["model_path"])).expanduser()
        if not model_path.is_absolute():
            model_path = (_REPO_ROOT / model_path).resolve()

        model = VGGT()
        state_dict = torch.load(str(model_path), map_location=device)
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        if isinstance(state_dict, dict):
            state_dict = {
                key[7:] if key.startswith("module.") else key: value
                for key, value in state_dict.items()
            }
        model.load_state_dict(state_dict)
        model.eval()
        self._model = model.to(device)
        self._device = device
        return self._model

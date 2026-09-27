"""Read the current calibrated camera observation from the workspace."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class CameraView:
    name: str
    image_path: Path
    rgb: np.ndarray
    depth_path: Path | None
    depth_m: np.ndarray | None
    intrinsics: np.ndarray
    T_world_camera: np.ndarray

    def image_data_url(self) -> str:
        encoded = base64.b64encode(self.image_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"


@dataclass(frozen=True, slots=True)
class CameraObservation:
    reference_view: str
    coordinate_frame: str
    views: tuple[CameraView, ...]

    @property
    def has_rgbd(self) -> bool:
        return bool(self.views) and all(view.depth_m is not None for view in self.views)


class ObservationStore:
    """Keep one camera snapshot for the duration of a sub-agent run."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.manifest_path = (
            self.workspace / "artifacts/observations/observation.json"
        )
        self._current: CameraObservation | None = None

    def reset(self) -> None:
        self._current = None

    def load(self) -> CameraObservation:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("camera observation manifest must be a JSON object")
        raw_views = payload.get("views")
        if not isinstance(raw_views, list) or len(raw_views) < 2:
            raise ValueError("object localization requires at least two calibrated camera views")
        views = []
        requires_depth = bool(payload.get("rgbd"))
        names: set[str] = set()
        for item in raw_views:
            if not isinstance(item, dict):
                raise ValueError("each camera view must be an object")
            name = str(item.get("name") or "").strip()
            if not name:
                raise ValueError("each camera view requires a non-empty name")
            if name in names:
                raise ValueError(f"camera view names must be unique, got {name!r} twice")
            names.add(name)
            image_path = self.workspace / item["image_path"]
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            depth_path, depth_m = _load_depth(
                self.workspace,
                item,
                rgb_shape=rgb.shape[:2],
                required=requires_depth,
                view_name=name,
            )
            intrinsics = np.asarray(item.get("intrinsics"), dtype=np.float64)
            transform = np.asarray(item.get("T_world_camera"), dtype=np.float64)
            _validate_calibration(name, intrinsics, transform)
            views.append(
                CameraView(
                    name=name,
                    image_path=image_path,
                    rgb=np.ascontiguousarray(rgb),
                    depth_path=depth_path,
                    depth_m=depth_m,
                    intrinsics=intrinsics,
                    T_world_camera=transform,
                )
            )
        reference_view = str(payload.get("reference_view") or "").strip()
        if reference_view not in names:
            raise ValueError(
                f"reference_view {reference_view!r} is not one of the published camera views"
            )
        coordinate_frame = str(payload.get("coordinate_frame") or "").strip()
        if not coordinate_frame:
            raise ValueError("camera observation requires a non-empty coordinate_frame")
        self._current = CameraObservation(
            reference_view=reference_view,
            coordinate_frame=coordinate_frame,
            views=tuple(views),
        )
        return self._current

    def current(self) -> CameraObservation:
        if self._current is None:
            raise RuntimeError(
                "No scene observation is loaded. Call observe_scene before "
                "segment_candidates."
            )
        return self._current


def _load_depth(
    workspace: Path,
    item: dict,
    *,
    rgb_shape: tuple[int, int],
    required: bool,
    view_name: str,
) -> tuple[Path | None, np.ndarray | None]:
    raw_path = item.get("depth_path")
    if not raw_path:
        if required:
            raise ValueError(f"camera {view_name!r} is missing required depth_path")
        return None, None
    if item.get("depth_unit") != "meter":
        raise ValueError(f"camera {view_name!r} depth_unit must be 'meter'")
    depth_path = workspace / str(raw_path)
    try:
        depth = np.load(depth_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"camera {view_name!r} depth array is unreadable: {depth_path}"
        ) from exc
    if depth.dtype.kind != "f":
        raise ValueError(f"camera {view_name!r} depth must use a floating dtype")
    if depth.shape != rgb_shape:
        raise ValueError(
            f"camera {view_name!r} RGB/depth shapes differ: {rgb_shape} vs {depth.shape}"
        )
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        raise ValueError(f"camera {view_name!r} depth has no finite positive samples")
    return depth_path, np.ascontiguousarray(depth.astype(np.float32, copy=False))


def _validate_calibration(
    name: str,
    intrinsics: np.ndarray,
    transform: np.ndarray,
) -> None:
    """Reject absent or fabricated geometry before invoking heavyweight models."""
    if intrinsics.shape != (3, 3) or not np.all(np.isfinite(intrinsics)):
        raise ValueError(f"camera {name!r} requires finite 3x3 intrinsics")
    if intrinsics[0, 0] <= 0.0 or intrinsics[1, 1] <= 0.0:
        raise ValueError(f"camera {name!r} has invalid focal lengths")
    if not np.allclose(intrinsics[2], [0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"camera {name!r} has an invalid pinhole intrinsics matrix")
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"camera {name!r} requires a finite 4x4 T_world_camera")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"camera {name!r} has an invalid homogeneous transform")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-4
    ):
        raise ValueError(f"camera {name!r} rotation is not orthonormal")

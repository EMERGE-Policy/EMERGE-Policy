from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from external_model_server.localization_types import TargetMaskResult, VGGTResult


def resolve_excluded_views(
    config: dict[str, Any],
    *,
    available_views: list[str],
) -> set[str]:
    """Resolve the optional point-cloud view filter with strict validation."""
    raw_filter = config.get("view_filter") or {}
    if not isinstance(raw_filter, dict):
        raise ValueError("pointcloud.view_filter must be an object")

    enabled = raw_filter.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("pointcloud.view_filter.enabled must be a boolean")
    if not enabled:
        return set()

    raw_excluded = raw_filter.get("excluded_views", [])
    if not isinstance(raw_excluded, list):
        raise ValueError(
            "pointcloud.view_filter.excluded_views must be a list"
        )
    excluded: set[str] = set()
    for index, raw_name in enumerate(raw_excluded):
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError(
                "pointcloud.view_filter.excluded_views"
                f"[{index}] must be a non-empty string"
            )
        excluded.add(raw_name.strip())

    available = set(available_views)
    unknown = sorted(excluded - available)
    if unknown:
        raise ValueError(
            "pointcloud.view_filter.excluded_views contains unknown views "
            f"{unknown}; available={sorted(available)}"
        )
    if available and excluded == available:
        raise ValueError(
            "pointcloud.view_filter cannot exclude every available view"
        )
    return excluded


def merge_masked_pointclouds(
    geometry: VGGTResult,
    masks: dict[str, TargetMaskResult],
    *,
    target_objects: list[str],
    point_conf_threshold: float = 0.0,
    excluded_views: set[str] | None = None,
    view_center_tolerance_m: float | None = None,
    ray_consensus_tolerance_m: float | None = None,
    min_consistent_views: int = 1,
) -> dict[str, dict[str, Any]]:
    excluded = set(excluded_views or ())
    cameras = {camera.name: camera for camera in geometry.cameras}
    merged: dict[str, dict[str, Any]] = {}
    for object_key in target_objects:
        candidates: list[dict[str, Any]] = []

        for view in geometry.views:
            if view.name in excluded:
                continue
            mask_view = masks.get(view.name)
            if mask_view is None:
                continue
            target = next(
                (item for item in mask_view.targets if item.object_key == object_key),
                None,
            )
            if target is None:
                continue
            if not target.found or target.mask is None:
                continue
            mask = np.array(target.mask, dtype=bool, copy=True)
            if view.point_conf is not None and point_conf_threshold > 0.0:
                mask &= np.asarray(view.point_conf, dtype=np.float32) >= point_conf_threshold
            points = np.asarray(view.point_map_world, dtype=np.float32)[mask]
            finite = np.isfinite(points).all(axis=1)
            points = points[finite]
            if points.shape[0] == 0:
                continue
            colors = np.asarray(view.rgb, dtype=np.uint8)[mask][finite]
            candidates.append(
                {
                    "name": view.name,
                    "confidence": float(target.confidence),
                    "center": np.median(points, axis=0),
                    "points_world": points,
                    "colors": colors,
                    "ray": _bbox_center_ray(
                        target.bbox_2d,
                        cameras.get(view.name),
                    ),
                }
            )

        selected, triangulated_center, ray_residuals = (
            _select_bbox_ray_consensus(
                candidates,
                tolerance_m=ray_consensus_tolerance_m,
                min_views=min_consistent_views,
            )
        )
        selection_method = "calibrated_bbox_ray_consensus"
        if triangulated_center is None:
            selected = _select_consistent_views(
                candidates,
                tolerance_m=view_center_tolerance_m,
                min_views=min_consistent_views,
            )
            selection_method = "vggt_depth_center_consensus"

        aligned = [
            _align_candidate_to_center(item, triangulated_center)
            for item in selected
        ]
        selected_names = {item["name"] for item in selected}
        rejected_names = [
            item["name"]
            for item in candidates
            if item["name"] not in selected_names
        ]
        views_payload = {
            item["name"]: {
                "points_world": item["points_world"],
                "colors": item["colors"],
            }
            for item in aligned
        }
        points_by_view = [item["points_world"] for item in aligned]
        colors_by_view = [item["colors"] for item in aligned]

        if points_by_view:
            points_world = np.concatenate(points_by_view, axis=0)
            colors = np.concatenate(colors_by_view, axis=0)
        else:
            points_world = np.zeros((0, 3), dtype=np.float32)
            colors = np.zeros((0, 3), dtype=np.uint8)

        merged[object_key] = {
            "points_world": points_world,
            "colors": colors,
            "source_views": [item["name"] for item in aligned],
            "views": views_payload,
            "excluded_views": sorted(excluded),
            "view_selection": {
                "method": selection_method,
                "candidate_views": [item["name"] for item in candidates],
                "selected_views": [item["name"] for item in aligned],
                "rejected_views": rejected_names,
                "raw_depth_centers_m": {
                    item["name"]: np.asarray(
                        item["center"],
                        dtype=float,
                    ).tolist()
                    for item in candidates
                },
                "triangulated_center_m": (
                    np.asarray(triangulated_center, dtype=float).tolist()
                    if triangulated_center is not None
                    else None
                ),
                "ray_residuals_m": ray_residuals,
                "ray_consensus_tolerance_m": ray_consensus_tolerance_m,
                "depth_center_tolerance_m": view_center_tolerance_m,
                "min_consistent_views": min_consistent_views,
            },
        }
    return merged


def _select_consistent_views(
    candidates: list[dict[str, Any]],
    *,
    tolerance_m: float | None,
    min_views: int,
) -> list[dict[str, Any]]:
    if tolerance_m is None:
        return candidates

    best_group: tuple[int, ...] = ()
    best_confidence = -1.0
    for group_size in range(len(candidates), min_views - 1, -1):
        for group in combinations(range(len(candidates)), group_size):
            centers = [
                np.asarray(candidates[index]["center"], dtype=np.float64)
                for index in group
            ]
            if any(
                np.linalg.norm(left - right) > tolerance_m
                for left, right in combinations(centers, 2)
            ):
                continue
            confidence = sum(
                float(candidates[index]["confidence"])
                for index in group
            )
            if confidence > best_confidence:
                best_group = group
                best_confidence = confidence
        if best_group:
            break

    return [candidates[index] for index in best_group]


def _bbox_center_ray(
    bbox_2d: list[int] | None,
    camera: Any | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if bbox_2d is None or camera is None:
        return None

    bbox = np.asarray(bbox_2d, dtype=np.float64)
    pixel = np.asarray(
        [
            0.5 * (bbox[0] + bbox[2]),
            0.5 * (bbox[1] + bbox[3]),
            1.0,
        ],
        dtype=np.float64,
    )
    intrinsics = np.asarray(camera.intrinsics_observed, dtype=np.float64)
    transform = np.asarray(camera.T_world_camera_observed, dtype=np.float64)
    direction_camera = np.linalg.solve(intrinsics, pixel)
    direction_world = transform[:3, :3] @ direction_camera
    direction_world /= np.linalg.norm(direction_world)
    return transform[:3, 3], direction_world


def _select_bbox_ray_consensus(
    candidates: list[dict[str, Any]],
    *,
    tolerance_m: float | None,
    min_views: int,
) -> tuple[list[dict[str, Any]], np.ndarray | None, dict[str, float]]:
    ray_candidates = [item for item in candidates if item["ray"] is not None]
    if tolerance_m is None or len(ray_candidates) < max(2, min_views):
        return [], None, {}

    best_indices: tuple[int, ...] = ()
    best_score: tuple[int, float, float] | None = None
    for left, right in combinations(range(len(ray_candidates)), 2):
        center = _triangulate_rays(
            [ray_candidates[left]["ray"], ray_candidates[right]["ray"]]
        )
        if center is None:
            continue
        residuals = [
            _ray_residual(center, item["ray"])
            for item in ray_candidates
        ]
        inliers = tuple(
            index
            for index, residual in enumerate(residuals)
            if residual <= tolerance_m
            and _point_is_in_front(center, ray_candidates[index]["ray"])
        )
        if len(inliers) < min_views:
            continue
        score = (
            len(inliers),
            sum(ray_candidates[index]["confidence"] for index in inliers),
            -sum(residuals[index] for index in inliers),
        )
        if best_score is None or score > best_score:
            best_indices = inliers
            best_score = score

    if not best_indices:
        return [], None, {}

    center = _triangulate_rays(
        [ray_candidates[index]["ray"] for index in best_indices]
    )
    if center is None:
        return [], None, {}
    residuals = {
        item["name"]: float(_ray_residual(center, item["ray"]))
        for item in ray_candidates
    }
    selected = [ray_candidates[index] for index in best_indices]
    return selected, center, residuals


def _triangulate_rays(
    rays: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray | None:
    lhs = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for origin, direction in rays:
        projector = np.eye(3) - np.outer(direction, direction)
        lhs += projector
        rhs += projector @ origin
    if np.linalg.cond(lhs) > 1e8:
        return None
    center = np.linalg.solve(lhs, rhs)
    return center if np.all(np.isfinite(center)) else None


def _ray_residual(
    point: np.ndarray,
    ray: tuple[np.ndarray, np.ndarray],
) -> float:
    origin, direction = ray
    projector = np.eye(3) - np.outer(direction, direction)
    return float(np.linalg.norm(projector @ (point - origin)))


def _point_is_in_front(
    point: np.ndarray,
    ray: tuple[np.ndarray, np.ndarray],
) -> bool:
    origin, direction = ray
    return float(np.dot(point - origin, direction)) > 0.0


def _align_candidate_to_center(
    candidate: dict[str, Any],
    center: np.ndarray | None,
) -> dict[str, Any]:
    if center is None:
        return candidate
    points = np.asarray(candidate["points_world"], dtype=np.float32)
    aligned_points = points - candidate["center"] + center
    return {**candidate, "points_world": aligned_points.astype(np.float32)}


def save_result(
    masked_pointclouds: dict[str, dict[str, Any]],
    output_dir: str | Path,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict[str, Any]] = []
    for object_key, pointcloud in masked_pointclouds.items():
        points_world = np.asarray(pointcloud.get("points_world", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32)
        colors = np.asarray(pointcloud.get("colors", np.zeros((0, 3), dtype=np.uint8)), dtype=np.uint8)
        source_views = list(pointcloud.get("source_views", []))
        excluded_views = list(pointcloud.get("excluded_views", []))

        points_path = output_dir / f"{object_key}_points_world.npy"
        colors_path = output_dir / f"{object_key}_colors.npy"
        ply_path = output_dir / f"{object_key}_pointcloud_world.ply"

        np.save(points_path, points_world)
        np.save(colors_path, colors)
        _write_pointcloud_ply(points_world, colors, ply_path)

        summary.append(
            {
                "object_key": object_key,
                "point_count": int(points_world.shape[0]),
                "source_views": source_views,
                "excluded_views": excluded_views,
                "points_world_path": points_path.name,
                "colors_path": colors_path.name,
                "pointcloud_world_ply": ply_path.name,
            }
        )

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps({"objects": summary}, indent=2),
        encoding="utf-8",
    )
    return {
        "output_dir": str(output_dir),
        "summary_path": str(summary_path),
        "objects": summary,
    }


def _write_pointcloud_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    colors = colors[valid]
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

"""Metric object geometry from aligned RGB-D views and verified SAM masks."""

from __future__ import annotations

from typing import Any

import numpy as np

from Emerge.subagents.object_location.tools.observation import CameraObservation


def backproject_masked_depth(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    T_world_camera: np.ndarray,
    *,
    min_depth_m: float = 0.05,
    max_depth_m: float = 10.0,
) -> np.ndarray:
    """Back-project selected depth pixels into the manifest coordinate frame."""
    depth = np.asarray(depth_m, dtype=np.float64)
    selected = np.asarray(mask, dtype=bool)
    K = np.asarray(intrinsics, dtype=np.float64)
    transform = np.asarray(T_world_camera, dtype=np.float64)
    if depth.ndim != 2 or selected.shape != depth.shape:
        raise ValueError(
            f"depth and mask must have the same HW shape, got {depth.shape} and {selected.shape}"
        )
    if K.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("RGB-D backprojection requires 3x3 intrinsics and a 4x4 transform")
    if not 0.0 <= min_depth_m < max_depth_m:
        raise ValueError("depth limits must satisfy 0 <= min_depth_m < max_depth_m")

    rows, cols = np.nonzero(selected)
    z = depth[rows, cols]
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    rows, cols, z = rows[valid], cols[valid], z[valid]
    if z.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    points_camera = np.column_stack(
        (
            (cols.astype(np.float64) - cx) * z / fx,
            (rows.astype(np.float64) - cy) * z / fy,
            z,
        )
    )
    return points_camera @ transform[:3, :3].T + transform[:3, 3]


def estimate_rgbd_object_locations(
    observation: CameraObservation,
    masks_payload: dict[str, Any],
    targets: list[dict[str, str]],
    *,
    min_points: int,
    min_consistent_views: int,
    min_depth_m: float = 0.05,
    max_depth_m: float = 10.0,
    outlier_quantile: float = 0.01,
) -> list[dict[str, Any]]:
    """Estimate robust visible-surface boxes directly from calibrated RGB-D."""
    if not observation.has_rgbd:
        raise RuntimeError("RGB-D geometry was selected but the observation has no depth")
    if min_points < 1 or min_consistent_views < 1:
        raise ValueError("min_points and min_consistent_views must be positive")
    if not 0.0 <= outlier_quantile < 0.5:
        raise ValueError("outlier_quantile must be in [0, 0.5)")

    views_by_name = {view.name: view for view in observation.views}
    mask_views = {
        str(view["view_name"]): view for view in masks_payload.get("views", [])
    }
    results = []
    for target in targets:
        object_key = str(target["object_key"])
        visible_views: list[str] = []
        confidences: list[float] = []
        per_view_points: list[np.ndarray] = []
        for view_name, mask_view in mask_views.items():
            view = views_by_name.get(view_name)
            if view is None or view.depth_m is None:
                continue
            mask_target = next(
                (
                    item
                    for item in mask_view.get("targets", [])
                    if str(item.get("object_key")) == object_key
                ),
                None,
            )
            if (
                mask_target is None
                or not mask_target.get("found")
                or mask_target.get("mask") is None
            ):
                continue
            points = backproject_masked_depth(
                view.depth_m,
                np.asarray(mask_target["mask"], dtype=bool),
                view.intrinsics,
                view.T_world_camera,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            )
            if points.size == 0:
                continue
            visible_views.append(view_name)
            confidences.append(float(mask_target.get("confidence", 0.0)))
            per_view_points.append(points)

        base = {
            "name": object_key,
            "sam_prompt": str(target["prompt"]),
            "sam3_confidence": max(confidences, default=0.0),
            "visible_views": visible_views,
            "view_selection": {
                "candidate_views": list(mask_views),
                "selected_views": visible_views,
                "excluded_views": [name for name in mask_views if name not in visible_views],
            },
            "num_points": int(sum(len(points) for points in per_view_points)),
        }
        if len(visible_views) < min_consistent_views:
            results.append(
                {**base, "found": False, "failure_reason": "insufficient_consistent_views"}
            )
            continue
        points = np.concatenate(per_view_points, axis=0)
        points = _trim_axis_outliers(points, outlier_quantile)
        if len(points) < min_points:
            results.append(
                {**base, "found": False, "failure_reason": "insufficient_depth_points"}
            )
            continue

        lower = points.min(axis=0)
        upper = points.max(axis=0)
        center = 0.5 * (lower + upper)
        extent = upper - lower
        results.append(
            {
                **base,
                "found": True,
                "frame": observation.coordinate_frame,
                "position_type": "visible_surface_bbox_center",
                "position_m": center.astype(float).tolist(),
                "size_m": extent.astype(float).tolist(),
                "rotation_matrix": np.eye(3).tolist(),
                "rpy_rad": [0.0, 0.0, 0.0],
                "bbox_world_min_m": lower.astype(float).tolist(),
                "bbox_world_max_m": upper.astype(float).tolist(),
                "num_inliers": int(len(points)),
                "pose_inlier_ratio": float(len(points) / max(1, base["num_points"])),
            }
        )
    return results


def _trim_axis_outliers(points: np.ndarray, quantile: float) -> np.ndarray:
    if quantile <= 0.0 or len(points) < 20:
        return points
    lower = np.quantile(points, quantile, axis=0)
    upper = np.quantile(points, 1.0 - quantile, axis=0)
    keep = np.all((points >= lower) & (points <= upper), axis=1)
    return points[keep]

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from Emerge.subagents.object_location.tools.location.object_pose_estimator import (
    ObjectPoseEstimator,
)
from external_model_server.localization_types import (
    ObjectPoseEstimate,
    TargetMaskResult,
    VGGTResult,
)


def refine_object_pose_with_multiview_bboxes(
    pose: ObjectPoseEstimate,
    *,
    geometry: VGGTResult,
    masks: dict[str, TargetMaskResult],
    config: dict[str, Any] | None = None,
) -> ObjectPoseEstimate:
    """Refine point-cloud center and extent against calibrated SAM bboxes."""
    settings = dict(config or {})
    views = _collect_valid_views(
        geometry=geometry,
        masks=masks,
        object_key=pose.object_key,
        padding_pixels=max(0, int(settings.get("padding_pixels", 0))),
    )
    if len(views) < 2:
        return pose

    center0 = np.asarray(pose.center, dtype=np.float64)
    extent0 = np.maximum(
        np.asarray(pose.extent, dtype=np.float64),
        0.005,
    )
    rotation = np.asarray(pose.rotation_matrix, dtype=np.float64)

    initial_centers = [center0]
    triangulated = _triangulate_bbox_centers(views)
    if triangulated is not None and np.linalg.norm(triangulated - center0) <= 0.5:
        initial_centers.append(triangulated)

    best_center = center0
    best_extent = extent0
    best_loss = _objective(
        center=best_center,
        extent=best_extent,
        rotation=rotation,
        center_prior=center0,
        extent_prior=extent0,
        views=views,
    )
    for initial_center in initial_centers:
        center, extent, loss = _coordinate_descent(
            center=initial_center,
            extent=extent0,
            rotation=rotation,
            center_prior=center0,
            extent_prior=extent0,
            views=views,
        )
        if loss < best_loss:
            best_center = center
            best_extent = extent
            best_loss = loss

    if np.allclose(best_center, center0) and np.allclose(
        best_extent,
        extent0,
    ):
        return pose

    corners = ObjectPoseEstimator._box_corners_from_pose(
        center=best_center,
        rotation_matrix=rotation,
        extent=best_extent,
    )
    return ObjectPoseEstimate(
        object_key=pose.object_key,
        frame=pose.frame,
        center=best_center,
        rotation_matrix=rotation,
        rpy=np.asarray(pose.rpy, dtype=np.float64),
        extent=best_extent,
        bbox_3d_corners=corners,
        num_points=pose.num_points,
        num_inliers=pose.num_inliers,
        pose_inlier_ratio=pose.pose_inlier_ratio,
    )


def _coordinate_descent(
    *,
    center: np.ndarray,
    extent: np.ndarray,
    rotation: np.ndarray,
    center_prior: np.ndarray,
    extent_prior: np.ndarray,
    views: list[dict[str, np.ndarray | list[int]]],
) -> tuple[np.ndarray, np.ndarray, float]:
    center = np.asarray(center, dtype=np.float64).copy()
    extent = np.asarray(extent, dtype=np.float64).copy()
    center_step = np.maximum(extent_prior * 0.5, 0.04)
    extent_step = np.maximum(extent_prior * 0.25, 0.015)
    best_loss = _objective(
        center=center,
        extent=extent,
        rotation=rotation,
        center_prior=center_prior,
        extent_prior=extent_prior,
        views=views,
    )

    for _level in range(7):
        for _iteration in range(12):
            candidates: list[tuple[np.ndarray, np.ndarray]] = []
            for axis in range(3):
                for direction in (-1.0, 1.0):
                    candidate_center = center.copy()
                    candidate_center[axis] += direction * center_step[axis]
                    candidates.append((candidate_center, extent.copy()))
            for axis in range(3):
                for direction in (-1.0, 1.0):
                    candidate_extent = extent.copy()
                    candidate_extent[axis] = np.clip(
                        candidate_extent[axis]
                        + direction * extent_step[axis],
                        0.005,
                        0.6,
                    )
                    candidates.append((center.copy(), candidate_extent))

            improved = False
            for candidate_center, candidate_extent in candidates:
                loss = _objective(
                    center=candidate_center,
                    extent=candidate_extent,
                    rotation=rotation,
                    center_prior=center_prior,
                    extent_prior=extent_prior,
                    views=views,
                )
                if loss + 1e-10 < best_loss:
                    center = candidate_center
                    extent = candidate_extent
                    best_loss = loss
                    improved = True
            if not improved:
                break
        center_step *= 0.5
        extent_step *= 0.5

    return center, extent, best_loss


def _objective(
    *,
    center: np.ndarray,
    extent: np.ndarray,
    rotation: np.ndarray,
    center_prior: np.ndarray,
    extent_prior: np.ndarray,
    views: Iterable[dict[str, np.ndarray | list[int]]],
) -> float:
    reprojection = _bbox_alignment_loss(
        center=center,
        extent=extent,
        rotation=rotation,
        views=views,
    )
    if not np.isfinite(reprojection):
        return float("inf")
    center_regularizer = float(
        np.sum(((center - center_prior) / 0.25) ** 2)
    )
    extent_regularizer = float(
        np.sum(
            np.log(
                np.maximum(extent, 0.005)
                / np.maximum(extent_prior, 0.005)
            )
            ** 2
        )
    )
    return reprojection + 0.01 * center_regularizer + 0.02 * extent_regularizer


def _collect_valid_views(
    *,
    geometry: VGGTResult,
    masks: dict[str, TargetMaskResult],
    object_key: str,
    padding_pixels: int,
) -> list[dict[str, np.ndarray | list[int]]]:
    views: list[dict[str, np.ndarray | list[int]]] = []
    camera_by_name = {camera.name: camera for camera in geometry.cameras}
    for view in geometry.views:
        mask_view = masks.get(view.name)
        camera = camera_by_name.get(view.name)
        if mask_view is None or camera is None:
            continue
        target = next(
            (
                item
                for item in mask_view.targets
                if item.object_key == object_key
            ),
            None,
        )
        if target is None or not target.found or target.bbox_2d is None:
            continue
        views.append(
            {
                "bbox_2d": _padded_bbox(
                    target.bbox_2d,
                    image_shape=mask_view.image_shape,
                    padding_pixels=padding_pixels,
                ),
                "intrinsics": np.asarray(
                    camera.intrinsics_observed,
                    dtype=np.float64,
                ),
                "T_world_camera": np.asarray(
                    camera.T_world_camera_observed,
                    dtype=np.float64,
                ),
            }
        )
    return views


def _padded_bbox(
    bbox: list[int],
    *,
    image_shape: tuple[int, int],
    padding_pixels: int,
) -> list[int]:
    height, width = [int(value) for value in image_shape]
    x0, y0, x1, y1 = [int(value) for value in bbox]
    padding = max(0, int(padding_pixels))
    return [
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width - 1, x1 + padding),
        min(height - 1, y1 + padding),
    ]


def _triangulate_bbox_centers(
    views: Iterable[dict[str, np.ndarray | list[int]]],
) -> np.ndarray | None:
    lhs = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    count = 0
    for view in views:
        bbox = np.asarray(view["bbox_2d"], dtype=np.float64)
        pixel = np.array(
            [
                0.5 * (bbox[0] + bbox[2]),
                0.5 * (bbox[1] + bbox[3]),
                1.0,
            ],
            dtype=np.float64,
        )
        intrinsics = np.asarray(view["intrinsics"], dtype=np.float64)
        transform = np.asarray(view["T_world_camera"], dtype=np.float64)
        direction_camera = np.linalg.solve(intrinsics, pixel)
        direction_world = transform[:3, :3] @ direction_camera
        norm = float(np.linalg.norm(direction_world))
        if norm <= 1e-10:
            continue
        direction_world /= norm
        origin = transform[:3, 3]
        projector = np.eye(3) - np.outer(direction_world, direction_world)
        lhs += projector
        rhs += projector @ origin
        count += 1
    if count < 2 or np.linalg.cond(lhs) > 1e8:
        return None
    center = np.linalg.solve(lhs, rhs)
    return center if np.all(np.isfinite(center)) else None


def _bbox_alignment_loss(
    *,
    center: np.ndarray,
    extent: np.ndarray,
    rotation: np.ndarray,
    views: Iterable[dict[str, np.ndarray | list[int]]],
) -> float:
    corners = ObjectPoseEstimator._box_corners_from_pose(
        center=center,
        rotation_matrix=rotation,
        extent=extent,
    )
    residuals: list[float] = []
    for view in views:
        projected = _project_world_corners_to_pixels(
            corners,
            intrinsics=np.asarray(view["intrinsics"], dtype=np.float64),
            T_world_camera=np.asarray(
                view["T_world_camera"],
                dtype=np.float64,
            ),
        )
        if projected is None:
            return float("inf")
        residuals.append(
            _bbox_residual(
                _bbox_from_projected_corners(projected),
                [int(value) for value in view["bbox_2d"]],
            )
        )
    return float(np.mean(residuals)) if residuals else float("inf")


def _bbox_residual(
    bbox_projected: list[int],
    bbox_reference: list[int],
) -> float:
    px0, py0, px1, py1 = [float(value) for value in bbox_projected]
    rx0, ry0, rx1, ry1 = [float(value) for value in bbox_reference]
    projected_center_x = 0.5 * (px0 + px1)
    projected_center_y = 0.5 * (py0 + py1)
    reference_center_x = 0.5 * (rx0 + rx1)
    reference_center_y = 0.5 * (ry0 + ry1)
    projected_width = max(px1 - px0, 1.0)
    projected_height = max(py1 - py0, 1.0)
    reference_width = max(rx1 - rx0, 1.0)
    reference_height = max(ry1 - ry0, 1.0)
    delta_x = (
        projected_center_x - reference_center_x
    ) / reference_width
    delta_y = (
        projected_center_y - reference_center_y
    ) / reference_height
    delta_width = np.log(projected_width / reference_width)
    delta_height = np.log(projected_height / reference_height)
    return float(
        delta_x * delta_x
        + delta_y * delta_y
        + delta_width * delta_width
        + delta_height * delta_height
    )


def _project_world_corners_to_pixels(
    corners_world: np.ndarray,
    *,
    intrinsics: np.ndarray,
    T_world_camera: np.ndarray,
) -> np.ndarray | None:
    homogeneous = np.concatenate(
        [
            np.asarray(corners_world, dtype=np.float64),
            np.ones((len(corners_world), 1), dtype=np.float64),
        ],
        axis=1,
    )
    points_camera = (
        np.linalg.inv(T_world_camera) @ homogeneous.T
    ).T[:, :3]
    depth = points_camera[:, 2]
    if np.any(depth <= 1e-8):
        return None
    projected = (intrinsics @ points_camera.T).T
    return projected[:, :2] / projected[:, 2:3]


def _bbox_from_projected_corners(corners_px: np.ndarray) -> list[int]:
    return [
        int(np.floor(np.min(corners_px[:, 0]))),
        int(np.floor(np.min(corners_px[:, 1]))),
        int(np.ceil(np.max(corners_px[:, 0]))),
        int(np.ceil(np.max(corners_px[:, 1]))),
    ]

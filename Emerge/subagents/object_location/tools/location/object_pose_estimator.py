"""Lightweight pointcloud-to-3D-bbox estimation for masked objects."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d

from external_model_server.localization_types import ObjectPoseEstimate


class ObjectPoseEstimator:
    """Estimate a world-frame 3D bounding box from a merged object pointcloud."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def estimate(self, object_key: str, points_world: np.ndarray) -> ObjectPoseEstimate:
        points_world = np.asarray(points_world, dtype=np.float32)
        if points_world.ndim != 2 or points_world.shape[1] != 3:
            raise ValueError(
                f"ObjectPoseEstimator expects points with shape Nx3, got {points_world.shape!r}"
            )

        total_points = int(points_world.shape[0])
        filtered_points = self._finite_points(points_world)
        min_points = int(self.config.get("min_points", 80))
        if filtered_points.shape[0] < min_points:
            raise RuntimeError(
                f"ObjectPoseEstimator requires at least {min_points} filtered points for {object_key}, "
                f"got {filtered_points.shape[0]}"
            )

        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(filtered_points.astype(np.float64, copy=False))

        voxel_size_m = float(self.config.get("voxel_size_m", 0.003))
        if voxel_size_m > 0.0:
            pointcloud = pointcloud.voxel_down_sample(voxel_size=voxel_size_m)
        pointcloud = self._remove_outliers(pointcloud)
        pointcloud = self._keep_largest_cluster(pointcloud)
        inlier_points = np.asarray(pointcloud.points, dtype=np.float64)
        if inlier_points.shape[0] < min_points:
            raise RuntimeError(
                f"ObjectPoseEstimator requires at least {min_points} inlier points for {object_key}, "
                f"got {inlier_points.shape[0]}"
            )

        obb = pointcloud.get_oriented_bounding_box()
        center = np.asarray(obb.center, dtype=np.float64)
        raw_rotation = np.asarray(obb.R, dtype=np.float64)
        raw_extent = np.asarray(obb.extent, dtype=np.float64)
        rotation, extent = self._canonicalize_obb(
            raw_rotation,
            raw_extent,
        )
        if bool(self.config.get("gravity_align_upright", True)):
            tilt_deg = float(
                np.rad2deg(
                    np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))
                )
            )
            max_tilt_deg = float(
                self.config.get("upright_snap_max_tilt_deg", 35.0)
            )
            if tilt_deg <= max_tilt_deg:
                center, rotation, extent = (
                    self._gravity_aligned_bounding_box(inlier_points)
                )
        rpy = self._rotation_matrix_to_rpy(rotation)
        bbox_3d_corners = self._box_corners_from_pose(
            center=center,
            rotation_matrix=rotation,
            extent=extent,
        )

        pose_inlier_ratio = (
            float(inlier_points.shape[0]) / float(total_points)
            if total_points > 0
            else 0.0
        )
        pose_inlier_ratio = float(max(0.0, min(1.0, pose_inlier_ratio)))
        # Keep the explicit box corners consistent with the final, possibly
        # gravity-aligned pose fields consumed downstream.
        return ObjectPoseEstimate(
            object_key=object_key,
            frame="world",
            center=center,
            rotation_matrix=rotation,
            rpy=rpy,
            extent=extent,
            bbox_3d_corners=bbox_3d_corners,
            num_points=total_points,
            num_inliers=int(inlier_points.shape[0]),
            pose_inlier_ratio=pose_inlier_ratio,
        )

    @staticmethod
    def save_result(
        object_poses: dict[str, ObjectPoseEstimate],
        output_dir: str | Path,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        objects: dict[str, dict[str, Any]] = {}
        for object_key, pose in object_poses.items():
            objects[object_key] = {
                "object_key": pose.object_key,
                "frame": pose.frame,
                "center": np.asarray(pose.center, dtype=np.float64).tolist(),
                "rotation_matrix": np.asarray(pose.rotation_matrix, dtype=np.float64).tolist(),
                "rpy": np.asarray(pose.rpy, dtype=np.float64).tolist(),
                "extent": np.asarray(pose.extent, dtype=np.float64).tolist(),
                "bbox_3d_corners": np.asarray(pose.bbox_3d_corners, dtype=np.float64).tolist(),
                "num_points": int(pose.num_points),
                "num_inliers": int(pose.num_inliers),
                "pose_inlier_ratio": float(pose.pose_inlier_ratio),
            }

        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps({"objects": objects}, indent=2),
            encoding="utf-8",
        )
        return {
            "output_dir": str(output_dir),
            "summary_path": str(summary_path),
            "objects": objects,
        }

    @staticmethod
    def _finite_points(points_world: np.ndarray) -> np.ndarray:
        mask = np.isfinite(points_world).all(axis=1)
        return points_world[mask]

    def _remove_outliers(self, pointcloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        num_points_before = len(pointcloud.points)

        statistical_nb_neighbors = int(self.config.get("statistical_nb_neighbors", 20))
        statistical_std_ratio = float(self.config.get("statistical_std_ratio", 2.0))
        if num_points_before >= statistical_nb_neighbors and statistical_nb_neighbors > 1:
            pointcloud, _ = pointcloud.remove_statistical_outlier(
                nb_neighbors=statistical_nb_neighbors,
                std_ratio=statistical_std_ratio,
            )

        radius_outlier_nb_points = int(self.config.get("radius_outlier_nb_points", 8))
        radius_outlier_radius_m = float(self.config.get("radius_outlier_radius_m", 0.01))
        if len(pointcloud.points) >= radius_outlier_nb_points and radius_outlier_nb_points > 1:
            pointcloud, _ = pointcloud.remove_radius_outlier(
                nb_points=radius_outlier_nb_points,
                radius=radius_outlier_radius_m,
            )

        return pointcloud

    def _keep_largest_cluster(self, pointcloud: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        min_cluster_points = int(self.config.get("cluster_min_points", 20))
        cluster_eps_m = float(self.config.get("cluster_eps_m", 0.015))
        if len(pointcloud.points) < min_cluster_points or min_cluster_points <= 1 or cluster_eps_m <= 0.0:
            return pointcloud

        labels = np.asarray(
            pointcloud.cluster_dbscan(
                eps=cluster_eps_m,
                min_points=min_cluster_points,
                print_progress=False,
            ),
            dtype=np.int32,
        )
        if labels.size == 0:
            return pointcloud

        valid_labels = labels[labels >= 0]
        if valid_labels.size == 0:
            return pointcloud

        unique_labels, counts = np.unique(valid_labels, return_counts=True)
        largest_label = int(unique_labels[int(np.argmax(counts))])
        indices = np.flatnonzero(labels == largest_label)
        if indices.size == 0:
            return pointcloud
        return pointcloud.select_by_index(indices.tolist())

    @staticmethod
    def _gravity_aligned_bounding_box(
        points_world: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit a yaw-only 3D box while taking the vertical span from points."""
        import cv2

        points = np.asarray(points_world, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or points.shape[0] < 3
            or not np.all(np.isfinite(points))
        ):
            raise ValueError(
                "gravity-aligned bounding box requires at least three finite 3D points"
            )

        (center_x, center_y), (width, height), angle_deg = cv2.minAreaRect(
            points[:, :2].astype(np.float32)
        )
        width = float(width)
        height = float(height)
        if width <= 1e-8 or height <= 1e-8:
            raise RuntimeError(
                "cannot fit gravity-aligned bounding box to degenerate XY points"
            )
        if height > width:
            width, height = height, width
            angle_deg += 90.0

        yaw = float(np.deg2rad(angle_deg))
        x_axis = np.asarray(
            [np.cos(yaw), np.sin(yaw), 0.0],
            dtype=np.float64,
        )
        if x_axis[0] < 0.0 or (
            abs(x_axis[0]) <= 1e-8 and x_axis[1] < 0.0
        ):
            x_axis *= -1.0
        z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        y_axis = np.cross(z_axis, x_axis)
        rotation = np.stack([x_axis, y_axis, z_axis], axis=1)

        z_min = float(np.min(points[:, 2]))
        z_max = float(np.max(points[:, 2]))
        center = np.asarray(
            [float(center_x), float(center_y), 0.5 * (z_min + z_max)],
            dtype=np.float64,
        )
        extent = np.asarray(
            [width, height, z_max - z_min],
            dtype=np.float64,
        )
        return center, rotation, extent

    @staticmethod
    def _canonicalize_obb(rotation: np.ndarray, extent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rotation = np.asarray(rotation, dtype=np.float64)
        extent = np.asarray(extent, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must have shape 3x3, got {rotation.shape!r}")
        if extent.shape != (3,):
            raise ValueError(f"extent must have shape (3,), got {extent.shape!r}")

        z_idx = int(np.argmax(np.abs(rotation[2, :])))
        remaining = [idx for idx in range(3) if idx != z_idx]
        x_idx = max(remaining, key=lambda idx: abs(rotation[0, idx]))
        y_idx = next(idx for idx in remaining if idx != x_idx)

        z_axis = rotation[:, z_idx].copy()
        if z_axis[2] < 0.0:
            z_axis *= -1.0

        x_axis = rotation[:, x_idx].copy()
        x_axis = x_axis - z_axis * float(np.dot(z_axis, x_axis))
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm <= 1e-8:
            x_axis = rotation[:, y_idx].copy()
            x_axis = x_axis - z_axis * float(np.dot(z_axis, x_axis))
            x_norm = float(np.linalg.norm(x_axis))
            x_idx, y_idx = y_idx, x_idx
        if x_norm <= 1e-8:
            raise RuntimeError("ObjectPoseEstimator failed to derive a stable horizontal axis from OBB")
        x_axis /= x_norm
        if x_axis[0] < 0.0 or (abs(x_axis[0]) <= 1e-8 and x_axis[1] < 0.0):
            x_axis *= -1.0

        y_axis = np.cross(z_axis, x_axis)
        y_norm = float(np.linalg.norm(y_axis))
        if y_norm <= 1e-8:
            raise RuntimeError("ObjectPoseEstimator failed to derive a stable orthogonal axis from OBB")
        y_axis /= y_norm

        rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
        if np.linalg.det(rotation) < 0.0:
            rotation[:, 1] *= -1.0

        return rotation, np.array([extent[x_idx], extent[y_idx], extent[z_idx]], dtype=np.float64)

    @staticmethod
    def _rotation_matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must have shape 3x3, got {rotation.shape!r}")

        sy = float(np.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0]))
        singular = sy < 1e-8
        if not singular:
            roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
            pitch = float(np.arctan2(-rotation[2, 0], sy))
            yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        else:
            roll = float(np.arctan2(-rotation[1, 2], rotation[1, 1]))
            pitch = float(np.arctan2(-rotation[2, 0], sy))
            yaw = 0.0
        return np.array([roll, pitch, yaw], dtype=np.float64)

    @staticmethod
    def _box_corners_from_pose(
        *,
        center: np.ndarray,
        rotation_matrix: np.ndarray,
        extent: np.ndarray,
    ) -> np.ndarray:
        center = np.asarray(center, dtype=np.float64)
        rotation_matrix = np.asarray(rotation_matrix, dtype=np.float64)
        extent = np.asarray(extent, dtype=np.float64)
        if center.shape != (3,):
            raise ValueError(f"center must have shape (3,), got {center.shape!r}")
        if rotation_matrix.shape != (3, 3):
            raise ValueError(f"rotation_matrix must have shape 3x3, got {rotation_matrix.shape!r}")
        if extent.shape != (3,):
            raise ValueError(f"extent must have shape (3,), got {extent.shape!r}")

        half = extent / 2.0
        local_corners = np.array(
            [
                [-half[0], -half[1], -half[2]],
                [half[0], -half[1], -half[2]],
                [half[0], half[1], -half[2]],
                [-half[0], half[1], -half[2]],
                [-half[0], -half[1], half[2]],
                [half[0], -half[1], half[2]],
                [half[0], half[1], half[2]],
                [-half[0], half[1], half[2]],
            ],
            dtype=np.float64,
        )
        return (local_corners @ rotation_matrix.T) + center.reshape(1, 3)

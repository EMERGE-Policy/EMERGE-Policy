"""Scene-adaptive five-camera rig for VGGT reconstruction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraPose:
    """One world-frame MuJoCo camera pose."""

    position_m: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]


class VggtCameraRig:
    """Place five overlapping views around the active manipulation region."""

    _LAYOUT = {
        "perception_oblique_1": (0.0, 46.0),
        "perception_oblique_2": (48.0, 40.0),
        "perception_oblique_3": (-48.0, 40.0),
        "perception_side_1": (-70.0, 25.0),
        "perception_side_2": (70.0, 25.0),
    }
    _WORKSPACE_SIZE_ATTRIBUTES = {
        "main_table": "table_full_size",
        "kitchen_table": "kitchen_table_full_size",
        "study_table": "study_table_full_size",
        "living_room_table": "living_room_table_full_size",
        "coffee_table": "coffee_table_full_size",
    }
    _REGION_PADDING_M = 0.10
    _SCENE_HEIGHT_M = 0.32
    _LOOK_HEIGHT_M = 0.14
    _FRAMING_MARGIN = 1.08
    _MIN_DISTANCE_M = 0.62
    _DISTANCE_SCALE = 0.60

    def plan(
        self,
        environment: Any,
        camera_settings: dict[str, dict[str, Any]],
    ) -> dict[str, CameraPose]:
        """Calculate fixed episode poses from scene geometry and camera FOV."""
        if set(camera_settings) != set(self._LAYOUT):
            expected = ", ".join(self._LAYOUT)
            raise ValueError(f"automatic VGGT rig requires these cameras: {expected}")

        bounds_min, bounds_max, surface_z = self._observation_bounds(environment)
        look_at = np.array(
            [
                0.5 * (bounds_min[0] + bounds_max[0]),
                0.5 * (bounds_min[1] + bounds_max[1]),
                surface_z + self._LOOK_HEIGHT_M,
            ],
            dtype=np.float64,
        )

        poses: dict[str, CameraPose] = {}
        for name, (azimuth_deg, elevation_deg) in self._LAYOUT.items():
            settings = camera_settings[name]
            direction = self._view_direction(azimuth_deg, elevation_deg)
            fitted_distance = self._fit_distance(
                bounds_min=bounds_min,
                bounds_max=bounds_max,
                look_at=look_at,
                camera_direction=direction,
                fovy_deg=float(settings.get("fovy", 62.0)),
                aspect=float(settings["width"]) / float(settings["height"]),
            )
            distance = max(
                self._MIN_DISTANCE_M,
                fitted_distance * self._DISTANCE_SCALE,
            )
            position = look_at + direction * distance
            quaternion = self._look_at_quaternion(position, look_at)
            poses[name] = CameraPose(
                position_m=tuple(position.astype(float)),
                quaternion_wxyz=tuple(quaternion.astype(float)),
            )
        return poses

    def _observation_bounds(
        self,
        environment: Any,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        workspace_name = str(environment.workspace_name)
        workspace_offset = np.asarray(
            environment.workspace_offset,
            dtype=np.float64,
        )
        surface_z = float(workspace_offset[2])

        rectangles = [
            np.asarray(rectangle, dtype=np.float64)
            for region in environment.parsed_problem["regions"].values()
            if region.get("target") == workspace_name
            for rectangle in region.get("ranges", ())
        ]
        workspace_bounds = self._workspace_xy_bounds(
            environment,
            workspace_name,
            workspace_offset,
        )
        if rectangles:
            ranges = np.stack(rectangles)
            xy_min = np.array([ranges[:, 0].min(), ranges[:, 1].min()])
            xy_max = np.array([ranges[:, 2].max(), ranges[:, 3].max()])
            xy_min += workspace_offset[:2] - self._REGION_PADDING_M
            xy_max += workspace_offset[:2] + self._REGION_PADDING_M
        elif workspace_bounds is not None:
            xy_min, xy_max = workspace_bounds
        else:
            raise ValueError(
                f"workspace {workspace_name!r} has no observable placement region"
            )

        if workspace_bounds is not None:
            workspace_min, workspace_max = workspace_bounds
            xy_min = np.maximum(xy_min, workspace_min)
            xy_max = np.minimum(xy_max, workspace_max)

        bounds_min = np.array([xy_min[0], xy_min[1], surface_z], dtype=np.float64)
        bounds_max = np.array(
            [xy_max[0], xy_max[1], surface_z + self._SCENE_HEIGHT_M],
            dtype=np.float64,
        )
        return bounds_min, bounds_max, surface_z

    def _workspace_xy_bounds(
        self,
        environment: Any,
        workspace_name: str,
        workspace_offset: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        size_attribute = self._WORKSPACE_SIZE_ATTRIBUTES.get(workspace_name)
        if size_attribute is None:
            return None
        size = np.asarray(getattr(environment, size_attribute), dtype=np.float64)
        half_size = 0.5 * size[:2]
        return workspace_offset[:2] - half_size, workspace_offset[:2] + half_size

    def _fit_distance(
        self,
        *,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        look_at: np.ndarray,
        camera_direction: np.ndarray,
        fovy_deg: float,
        aspect: float,
    ) -> float:
        camera_forward = -camera_direction
        camera_right = self._normalize(
            np.cross(camera_forward, np.array([0.0, 0.0, 1.0]))
        )
        camera_up = np.cross(camera_right, camera_forward)
        tan_vertical = math.tan(math.radians(fovy_deg) / 2.0)
        tan_horizontal = tan_vertical * aspect

        required_distance = self._MIN_DISTANCE_M
        for corner in self._box_corners(bounds_min, bounds_max):
            relative = corner - look_at
            forward_offset = float(np.dot(relative, camera_forward))
            horizontal = abs(float(np.dot(relative, camera_right)))
            vertical = abs(float(np.dot(relative, camera_up)))
            required_distance = max(
                required_distance,
                self._FRAMING_MARGIN * horizontal / tan_horizontal - forward_offset,
                self._FRAMING_MARGIN * vertical / tan_vertical - forward_offset,
            )
        return required_distance

    @staticmethod
    def _view_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
        azimuth = math.radians(azimuth_deg)
        elevation = math.radians(elevation_deg)
        return np.array(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ],
            dtype=np.float64,
        )

    @classmethod
    def _look_at_quaternion(
        cls,
        position: np.ndarray,
        target: np.ndarray,
    ) -> np.ndarray:
        optical_forward = cls._normalize(target - position)
        camera_right = cls._normalize(
            np.cross(optical_forward, np.array([0.0, 0.0, 1.0]))
        )
        camera_up = np.cross(camera_right, optical_forward)
        rotation = np.column_stack((camera_right, camera_up, -optical_forward))
        return cls._rotation_to_quaternion_wxyz(rotation)

    @staticmethod
    def _rotation_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
        else:
            diagonal = np.diag(rotation)
            index = int(np.argmax(diagonal))
            next_index = (index + 1) % 3
            last_index = (index + 2) % 3
            scale = math.sqrt(
                1.0
                + rotation[index, index]
                - rotation[next_index, next_index]
                - rotation[last_index, last_index]
            ) * 2.0
            quaternion = np.empty(4, dtype=np.float64)
            quaternion[index + 1] = 0.25 * scale
            quaternion[0] = (
                rotation[last_index, next_index]
                - rotation[next_index, last_index]
            ) / scale
            quaternion[next_index + 1] = (
                rotation[next_index, index] + rotation[index, next_index]
            ) / scale
            quaternion[last_index + 1] = (
                rotation[last_index, index] + rotation[index, last_index]
            ) / scale
        quaternion /= np.linalg.norm(quaternion)
        return quaternion if quaternion[0] >= 0.0 else -quaternion

    @staticmethod
    def _box_corners(minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                [x, y, z]
                for x in (minimum[0], maximum[0])
                for y in (minimum[1], maximum[1])
                for z in (minimum[2], maximum[2])
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        return vector / np.linalg.norm(vector)

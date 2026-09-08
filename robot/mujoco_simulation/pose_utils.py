"""Pure pose and OSC conversion helpers."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class PoseUtils:
    """Stateless pose utilities shared by the MuJoCo action and VLA paths."""

    @staticmethod
    def vector(value: Any, *, name: str, length: int = 3) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size != length or not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain exactly {length} finite numbers")
        return array

    @staticmethod
    def normalize_quaternion(quaternion_xyzw: Any) -> np.ndarray:
        quaternion = PoseUtils.vector(
            quaternion_xyzw, name="quaternion", length=4
        )
        norm = float(np.linalg.norm(quaternion))
        if norm <= 1e-12:
            raise ValueError("quaternion norm must be positive")
        quaternion = quaternion / norm
        if quaternion[3] < 0.0:
            quaternion = -quaternion
        return quaternion

    @staticmethod
    def quaternion_conjugate(quaternion_xyzw: Any) -> np.ndarray:
        quaternion = PoseUtils.normalize_quaternion(quaternion_xyzw)
        return np.array(
            [-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]],
            dtype=np.float64,
        )

    @staticmethod
    def quaternion_multiply(left_xyzw: Any, right_xyzw: Any) -> np.ndarray:
        left = PoseUtils.normalize_quaternion(left_xyzw)
        right = PoseUtils.normalize_quaternion(right_xyzw)
        lx, ly, lz, lw = left
        rx, ry, rz, rw = right
        return PoseUtils.normalize_quaternion(
            np.array(
                [
                    lw * rx + lx * rw + ly * rz - lz * ry,
                    lw * ry - lx * rz + ly * rw + lz * rx,
                    lw * rz + lx * ry - ly * rx + lz * rw,
                    lw * rw - lx * rx - ly * ry - lz * rz,
                ],
                dtype=np.float64,
            )
        )

    @staticmethod
    def quaternion_to_axis_angle(quaternion_xyzw: Any) -> np.ndarray:
        quaternion = PoseUtils.normalize_quaternion(quaternion_xyzw)
        xyz = quaternion[:3]
        sin_half = float(np.linalg.norm(xyz))
        if sin_half <= 1e-10:
            return np.zeros(3, dtype=np.float64)
        angle = 2.0 * math.atan2(sin_half, float(quaternion[3]))
        return xyz / sin_half * angle

    @staticmethod
    def axis_angle_to_quaternion(axis_angle: Any) -> np.ndarray:
        vector = PoseUtils.vector(axis_angle, name="axis_angle")
        angle = float(np.linalg.norm(vector))
        if angle <= 1e-10:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        axis = vector / angle
        half = 0.5 * angle
        return np.concatenate((axis * math.sin(half), [math.cos(half)]))

    @staticmethod
    def quaternion_error(current_xyzw: Any, target_xyzw: Any) -> np.ndarray:
        error = PoseUtils.quaternion_multiply(
            target_xyzw, PoseUtils.quaternion_conjugate(current_xyzw)
        )
        return PoseUtils.quaternion_to_axis_angle(error)

    @staticmethod
    def euler_to_quaternion(euler_rpy: Any) -> np.ndarray:
        roll, pitch, yaw = PoseUtils.vector(euler_rpy, name="orientation_euler")
        cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
        cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
        cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
        return PoseUtils.normalize_quaternion(
            np.array(
                [
                    sr * cp * cy - cr * sp * sy,
                    cr * sp * cy + sr * cp * sy,
                    cr * cp * sy - sr * sp * cy,
                    cr * cp * cy + sr * sp * sy,
                ]
            )
        )

    @staticmethod
    def resolve_orientation(params: dict[str, Any]) -> np.ndarray | None:
        if "orientation_quat" in params:
            return PoseUtils.normalize_quaternion(params["orientation_quat"])
        if "orientation_euler" in params:
            return PoseUtils.euler_to_quaternion(params["orientation_euler"])
        return None

    @staticmethod
    def osc_action(
        position_error: Any,
        rotation_error: Any,
        gripper_signal: float,
        *,
        position_scale: float,
        rotation_scale: float,
        position_gain: float,
        rotation_gain: float,
    ) -> np.ndarray:
        if position_scale <= 0.0 or rotation_scale <= 0.0:
            raise ValueError("OSC scales must be positive")
        position = (
            PoseUtils.vector(position_error, name="position_error")
            * float(position_gain)
            / float(position_scale)
        )
        rotation = (
            PoseUtils.vector(rotation_error, name="rotation_error")
            * float(rotation_gain)
            / float(rotation_scale)
        )
        return np.concatenate(
            (
                np.clip(position, -1.0, 1.0),
                np.clip(rotation, -1.0, 1.0),
                [float(np.clip(gripper_signal, -1.0, 1.0))],
            )
        ).astype(np.float32)

    @staticmethod
    def rotate_vector(vector: Any, axis: Any, angle_rad: float) -> np.ndarray:
        value = PoseUtils.vector(vector, name="vector")
        axis_value = PoseUtils.vector(axis, name="axis")
        norm = float(np.linalg.norm(axis_value))
        if norm <= 1e-12:
            raise ValueError("axis must be non-zero")
        unit = axis_value / norm
        return (
            value * math.cos(angle_rad)
            + np.cross(unit, value) * math.sin(angle_rad)
            + unit * np.dot(unit, value) * (1.0 - math.cos(angle_rad))
        )

    @staticmethod
    def vector_dict(vector: Any) -> dict[str, float]:
        value = PoseUtils.vector(vector, name="vector")
        return dict(zip(("x", "y", "z"), value.astype(float), strict=True))

    @staticmethod
    def rpy_dict(euler_rpy: Any) -> dict[str, float]:
        value = PoseUtils.vector(euler_rpy, name="euler_rpy")
        return dict(
            zip(("roll", "pitch", "yaw"), value.astype(float), strict=True)
        )

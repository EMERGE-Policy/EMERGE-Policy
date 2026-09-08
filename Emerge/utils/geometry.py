"""Shared geometry helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def quaternion_wxyz_to_rpy(quaternion: Any) -> np.ndarray:
    """Convert a WXYZ quaternion to XYZ roll, pitch, and yaw in radians."""
    value = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if value.size != 4 or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must contain exactly four finite values")

    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("quaternion norm must be positive")
    w, x, y, z = value / norm

    roll = np.arctan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return np.array([roll, pitch, yaw], dtype=np.float64)


def quaternion_xyzw_to_rpy(quaternion: Any) -> np.ndarray:
    """Convert an XYZW quaternion to XYZ roll, pitch, and yaw in radians."""
    value = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if value.size != 4 or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must contain exactly four finite values")
    x, y, z, w = value
    return quaternion_wxyz_to_rpy([w, x, y, z])

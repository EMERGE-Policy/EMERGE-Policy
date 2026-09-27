"""Deterministic pre-contact geometry for measured object bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_STANDOFF_M = 0.12
MIN_STANDOFF_M = 0.08
MAX_STANDOFF_M = 0.15


@dataclass(frozen=True, slots=True)
class PrecontactPose:
    """A clearance pose derived from measured geometry and current robot state."""

    side: str
    position_m: tuple[float, float, float]
    orientation_quat: tuple[float, float, float, float]
    surface_point_m: tuple[float, float, float]
    standoff_m: float


def compute_precontact_pose(
    robot_state: dict[str, Any],
    bbox_min_m: Any,
    bbox_max_m: Any,
    *,
    side: str | None = None,
    standoff_m: float = DEFAULT_STANDOFF_M,
) -> PrecontactPose:
    """Return a reachable-side candidate outside a measured axis-aligned box.

    The ray from the box centre toward the selected current end effector picks
    the near surface.  The target is then offset farther toward that end
    effector by ``standoff_m``.  This avoids treating the object centre as an
    end-effector target and preserves the live wrist orientation.
    """

    lower = _vector(bbox_min_m, 3, "bbox_world_min_m")
    upper = _vector(bbox_max_m, 3, "bbox_world_max_m")
    if np.any(upper <= lower):
        raise ValueError("bbox_world_max_m must be greater than bbox_world_min_m")

    standoff = float(standoff_m)
    if not np.isfinite(standoff) or not MIN_STANDOFF_M <= standoff <= MAX_STANDOFF_M:
        raise ValueError(
            f"standoff_m must be between {MIN_STANDOFF_M:.2f} and "
            f"{MAX_STANDOFF_M:.2f} metres"
        )

    center = (lower + upper) / 2.0
    requested_side = None if side is None else str(side).strip().lower()
    if requested_side in {None, "", "auto"}:
        candidates = ("left", "right")
        selected = min(
            candidates,
            key=lambda candidate: float(
                np.linalg.norm(_eef_position(robot_state, candidate) - center)
            ),
        )
    elif requested_side in {"left", "right"}:
        selected = requested_side
    else:
        raise ValueError("side must be 'left', 'right', or 'auto'")

    eef_position = _eef_position(robot_state, selected)
    direction = eef_position - center
    distance = float(np.linalg.norm(direction))
    if distance < 1e-9:
        raise ValueError("selected end effector is at the bounding-box centre")
    unit = direction / distance

    half_extent = (upper - lower) / 2.0
    active = np.abs(unit) > 1e-12
    ray_scales = half_extent[active] / np.abs(unit[active])
    surface = center + unit * float(np.min(ray_scales))
    target = surface + unit * standoff
    orientation = _eef_orientation(robot_state, selected)

    return PrecontactPose(
        side=selected,
        position_m=tuple(float(value) for value in target),
        orientation_quat=tuple(float(value) for value in orientation),
        surface_point_m=tuple(float(value) for value in surface),
        standoff_m=standoff,
    )


def _eef_position(robot_state: dict[str, Any], side: str) -> np.ndarray:
    try:
        value = robot_state["arms"][side]["eef_position"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"latest R1Pro state does not contain the {side} end-effector position"
        ) from exc
    return _vector(value, 3, f"{side} end-effector position")


def _eef_orientation(robot_state: dict[str, Any], side: str) -> np.ndarray:
    try:
        value = robot_state["arms"][side]["eef_orientation_xyzw"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"latest R1Pro state does not contain the {side} end-effector orientation"
        ) from exc
    quaternion = _vector(value, 4, f"{side} end-effector orientation")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError(f"{side} end-effector orientation must be non-zero")
    return quaternion / norm


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {length} finite values")
    return vector

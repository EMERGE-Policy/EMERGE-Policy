"""R1Pro-only conversion of LightNav waypoints to official 23-D actions."""

from __future__ import annotations

import math
from collections.abc import Sequence


ACTION_HZ = 30
LIGHTNAV_HZ = 4
BASE_MAX_VELOCITY = (0.75, 0.75, 1.0)  # m/s, m/s, rad/s


def frame_action_steps(frame_index: int) -> int:
    """Use exactly 30 evaluator actions for each four LightNav frames."""
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    return ((frame_index + 1) * ACTION_HZ // LIGHTNAV_HZ) - (frame_index * ACTION_HZ // LIGHTNAV_HZ)


def waypoint_action(
    waypoint: Sequence[float],
    hold_action: Sequence[float],
    *,
    action_steps: int,
    max_normalized_speed: float = 0.6,
) -> tuple[list[float], dict[str, object]]:
    """Map local displacement over ``action_steps`` to normalized base velocity.

    The R1Pro base accepts local x/y/yaw velocities in the first three action
    slots. The other 20 slots retain the observed trunk, arm and gripper pose.
    Saturation is recorded rather than pretending the full waypoint was reached.
    """
    if len(waypoint) != 3 or len(hold_action) != 23:
        raise ValueError("LightNav waypoint must be 3-D and R1Pro hold action 23-D")
    if action_steps < 1:
        raise ValueError("action_steps must be positive")
    if not math.isfinite(max_normalized_speed) or not 0 < max_normalized_speed <= 1:
        raise ValueError("max_normalized_speed must be in (0, 1]")
    values = [float(value) for value in waypoint]
    hold = [float(value) for value in hold_action]
    if not all(math.isfinite(value) for value in values + hold):
        raise ValueError("waypoint and hold action must contain only finite values")
    dt = action_steps / ACTION_HZ
    unclipped = [value / (dt * limit) for value, limit in zip(values, BASE_MAX_VELOCITY, strict=True)]
    normalized = [max(-max_normalized_speed, min(max_normalized_speed, value)) for value in unclipped]
    action = normalized + hold[3:]
    return action, {
        "requested_displacement": values,
        "unclipped_normalized_base": unclipped,
        "applied_normalized_base": normalized,
        "clipped": any(abs(before - after) > 1e-9 for before, after in zip(unclipped, normalized, strict=True)),
        "command_duration_s": dt,
    }

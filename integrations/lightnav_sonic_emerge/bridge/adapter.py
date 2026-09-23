"""Pure conversion from a LightNav local waypoint to a SONIC planner command.

LightNav waypoints are robot-local ``[forward_m, lateral_m, yaw_rad]`` with
positive lateral motion to the robot's left and positive yaw counterclockwise.
SONIC's kinematic planner instead consumes *world-frame* unit movement and
facing vectors in its MuJoCo Z-up frame.  The adapter is intentionally small
and dependency-free so its geometry can be tested without CUDA, LightNav, or
SONIC installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Real
from typing import Any, Sequence


SONIC_IDLE_MODE = 0
SONIC_SLOW_WALK_MODE = 1
_EPSILON = 1e-9


@dataclass(frozen=True)
class AdapterLimits:
    """Conservative initial limits for a G1 SONIC planner command."""

    control_period_s: float = 0.25
    max_forward_mps: float = 0.25
    max_lateral_mps: float = 0.15
    max_planar_mps: float = 0.25
    max_yaw_rate_rps: float = 0.35
    position_tolerance_m: float = 0.12
    yaw_tolerance_rad: float = 0.087

    def __post_init__(self) -> None:
        positive = {
            "control_period_s": self.control_period_s,
            "max_forward_mps": self.max_forward_mps,
            "max_lateral_mps": self.max_lateral_mps,
            "max_planar_mps": self.max_planar_mps,
            "max_yaw_rate_rps": self.max_yaw_rate_rps,
            "position_tolerance_m": self.position_tolerance_m,
            "yaw_tolerance_rad": self.yaw_tolerance_rad,
        }
        invalid = [name for name, value in positive.items() if not math.isfinite(value) or value <= 0.0]
        if invalid:
            raise ValueError(f"adapter limits must be finite and positive: {', '.join(invalid)}")


@dataclass(frozen=True)
class SonicPlannerCommand:
    """A standard SONIC ``MovementState``-compatible planner command."""

    mode: int
    movement_direction: tuple[float, float, float]
    facing_direction: tuple[float, float, float]
    movement_speed: float
    height: float = -1.0
    reason: str = ""

    @property
    def is_idle(self) -> bool:
        return self.mode == SONIC_IDLE_MODE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite_scalar(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class LightNav2SonicAdapter:
    """Convert LightNav's first local waypoint into a safe SONIC command.

    The adapter sends no packets itself.  Callers should pass the returned
    :class:`SonicPlannerCommand` to this integration's ``sonic_zmq`` module.
    Invalid model output intentionally maps to locomotion IDLE rather than
    propagating stale motion.
    """

    def __init__(self, limits: AdapterLimits | None = None) -> None:
        self.limits = limits or AdapterLimits()

    def idle(self, robot_yaw_rad: float | None = None, *, reason: str) -> SonicPlannerCommand:
        """Return an explicit SONIC locomotion-IDLE command.

        A unit facing vector is still supplied because SONIC validates it even
        when the requested locomotion mode is IDLE.  It follows the measured
        current heading when available and otherwise uses the canonical +X
        world forward direction.
        """

        yaw = 0.0
        if robot_yaw_rad is not None:
            try:
                yaw = _finite_scalar(robot_yaw_rad, "robot_yaw_rad")
            except ValueError:
                reason = f"{reason}; invalid robot yaw"
        return SonicPlannerCommand(
            mode=SONIC_IDLE_MODE,
            movement_direction=(0.0, 0.0, 0.0),
            facing_direction=(math.cos(yaw), math.sin(yaw), 0.0),
            movement_speed=-1.0,
            height=-1.0,
            reason=reason,
        )

    def convert(self, waypoint: Sequence[float], robot_yaw_rad: float) -> SonicPlannerCommand:
        """Convert one valid local waypoint.

        ``robot_yaw_rad`` is the current yaw in SONIC's world frame.  The
        first two waypoint values become a clipped world direction and scalar
        speed.  The third value remains an independent desired-facing delta.
        """

        if isinstance(waypoint, (str, bytes)) or len(waypoint) != 3:
            raise ValueError("LightNav waypoint must have exactly three values")
        forward_m = _finite_scalar(waypoint[0], "waypoint forward_m")
        lateral_m = _finite_scalar(waypoint[1], "waypoint lateral_m")
        delta_yaw_rad = _finite_scalar(waypoint[2], "waypoint delta_yaw_rad")
        robot_yaw = _finite_scalar(robot_yaw_rad, "robot_yaw_rad")

        translation_m = math.hypot(forward_m, lateral_m)
        if (
            translation_m < self.limits.position_tolerance_m
            and abs(delta_yaw_rad) < self.limits.yaw_tolerance_rad
        ):
            return self.idle(robot_yaw, reason="first waypoint is within stop tolerances")

        applied_delta_yaw = _clip(
            delta_yaw_rad,
            -self.limits.max_yaw_rate_rps * self.limits.control_period_s,
            self.limits.max_yaw_rate_rps * self.limits.control_period_s,
        )
        target_yaw = _wrap_angle(robot_yaw + applied_delta_yaw)
        facing = (math.cos(target_yaw), math.sin(target_yaw), 0.0)

        # A valid yaw-only action deliberately retains a zero movement vector.
        # In SONIC ``movement_speed == 0`` means *stationary*, whereas ``-1``
        # asks the selected locomotion mode for its default speed.  The latter
        # is required for the planner to generate a turn-in-place motion from
        # the changed facing direction.
        if translation_m < self.limits.position_tolerance_m:
            return SonicPlannerCommand(
                mode=SONIC_SLOW_WALK_MODE,
                movement_direction=(0.0, 0.0, 0.0),
                facing_direction=facing,
                movement_speed=-1.0,
                height=-1.0,
                reason="yaw-only waypoint",
            )

        vx_body = _clip(
            forward_m / self.limits.control_period_s,
            -self.limits.max_forward_mps,
            self.limits.max_forward_mps,
        )
        vy_body = _clip(
            lateral_m / self.limits.control_period_s,
            -self.limits.max_lateral_mps,
            self.limits.max_lateral_mps,
        )
        speed = math.hypot(vx_body, vy_body)
        if speed > self.limits.max_planar_mps:
            scale = self.limits.max_planar_mps / speed
            vx_body *= scale
            vy_body *= scale
            speed = self.limits.max_planar_mps

        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        world_vx = cos_yaw * vx_body - sin_yaw * vy_body
        world_vy = sin_yaw * vx_body + cos_yaw * vy_body
        world_speed = math.hypot(world_vx, world_vy)
        if world_speed < _EPSILON:
            # This path is defensive: a non-zero translation should not become
            # zero after finite clipping, but an explicit stop is safer than a
            # malformed direction should it happen.
            return self.idle(robot_yaw, reason="clipped waypoint has zero planar velocity")

        return SonicPlannerCommand(
            mode=SONIC_SLOW_WALK_MODE,
            movement_direction=(world_vx / world_speed, world_vy / world_speed, 0.0),
            facing_direction=facing,
            movement_speed=speed,
            height=-1.0,
            reason="tracked first LightNav waypoint",
        )

    def convert_or_idle(self, waypoint: Sequence[float], robot_yaw_rad: float) -> SonicPlannerCommand:
        """Fail closed for untrusted LightNav output or stale pose data."""

        try:
            return self.convert(waypoint, robot_yaw_rad)
        except (TypeError, ValueError, OverflowError) as exc:
            return self.idle(robot_yaw_rad, reason=f"invalid LightNav waypoint: {exc}")

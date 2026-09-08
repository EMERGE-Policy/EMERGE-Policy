"""Rule-based actions implemented as OSC_POSE control sequences."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from robot.mujoco_simulation.pose_utils import PoseUtils


class _ActionInterrupted(Exception):
    pass


class MujocoActionController:
    """Translate Emerge rule actions into direct LIBERO action7 steps."""

    def __init__(
        self,
        environment: Any,
        config: dict[str, Any] | None = None,
        *,
        vla_executor: Any | None = None,
    ) -> None:
        self._environment = environment
        self._vla = vla_executor
        self.config = dict(config or {})
        self.position_scale = float(self.config.get("position_scale", 0.05))
        self.rotation_scale = float(self.config.get("rotation_scale", 0.5))
        self.position_gain = float(self.config.get("position_gain", 1.0))
        self.rotation_gain = float(self.config.get("rotation_gain", 1.0))
        self.position_tolerance = float(self.config.get("position_tolerance_m", 0.005))
        self.rotation_tolerance = float(self.config.get("rotation_tolerance_rad", 0.05))
        self.max_steps = max(1, int(self.config.get("max_steps", 160)))
        self.gripper_max_opening = float(self.config.get("gripper_max_opening_m", 0.08))
        # Last explicitly commanded gripper signal (+1 close .. -1 open).
        # Rule moves reuse this instead of re-deriving a slack signal from
        # gripper qpos, which collapses toward ~0 while a grasped object holds
        # the fingers apart and would otherwise drop the object mid-move.
        self._gripper_command: float | None = None
        self._cancel_check: Callable[[], str | None] | None = None

    def _active_gripper_signal(self) -> float:
        if self._gripper_command is not None:
            return self._gripper_command
        return self._environment.current_gripper_signal()

    def execute(
        self,
        action_type: str,
        params: dict[str, Any],
        *,
        cancel_check: Callable[[], str | None] | None = None,
    ) -> str:
        handlers = {
            "move_to_pose": self._move_to_pose,
            "move_linear": self._move_linear,
            "set_gripper": self._set_gripper,
            "follow_arc": self._follow_arc,
            "vla_execute": self._vla_execute,
        }
        handler = handlers.get(action_type)
        if handler is None:
            return f"Unknown action type: {action_type!r}"
        self._cancel_check = cancel_check
        try:
            self._check_interrupted()
            return handler(dict(params or {}))
        except _ActionInterrupted as exc:
            return f"Interrupted: {exc}"
        except (TypeError, ValueError, RuntimeError) as exc:
            return f"Failed: {exc}"
        finally:
            self._cancel_check = None

    def _check_interrupted(self) -> None:
        if self._cancel_check is None:
            return
        reason = self._cancel_check()
        if reason:
            raise _ActionInterrupted(reason)

    def _vla_execute(self, params: dict[str, Any]) -> str:
        if self._vla is None:
            return "Failed: LIBERO VLA executor is not initialized"
        instruction = str(
            params.get("instruction", params.get("prompt", ""))
        ).strip()
        if not instruction:
            return "Failed: vla_execute requires instruction or prompt"
        if "step" not in params:
            return "Failed: vla_execute requires step"
        step = int(params["step"])
        result = self._vla.execute(
            instruction,
            step=step,
            cancel_check=self._cancel_check,
        )
        if result.last_gripper_command is not None:
            self._gripper_command = float(result.last_gripper_command)
        if result.reason == "interrupted":
            return f"Interrupted: {result.error_message or 'visual monitor requested a stop'}"
        if result.success:
            return f"VLA execution finished: {result.reason}, steps={result.total_steps}."
        detail = f" ({result.error_message})" if result.error_message else ""
        return (
            f"Failed: VLA execution {result.reason} "
            f"after {result.total_steps} steps{detail}."
        )

    def _move_to_pose(self, params: dict[str, Any]) -> str:
        if "position_m" not in params:
            raise ValueError("move_to_pose requires position_m")
        target_position = PoseUtils.vector(params["position_m"], name="position_m")
        target_orientation = PoseUtils.resolve_orientation(params)
        if target_orientation is None:
            raise ValueError("move_to_pose requires orientation_quat or orientation_euler")
        converged, steps = self._servo_pose(
            target_position,
            target_orientation,
            max_steps=int(params.get("steps", self.max_steps)),
        )
        if not converged:
            return f"Failed: move_to_pose did not converge after {steps} steps"
        return (
            "End-effector moved to pose at "
            f"({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}) m."
        )

    def _move_linear(self, params: dict[str, Any]) -> str:
        start_position, start_orientation = self._environment.get_eef_pose()
        if "position_m" in params:
            target_position = PoseUtils.vector(params["position_m"], name="position_m")
        elif "delta_m" in params:
            target_position = start_position + PoseUtils.vector(params["delta_m"], name="delta_m")
        else:
            raise ValueError("move_linear requires position_m or delta_m")
        target_orientation = PoseUtils.resolve_orientation(params)
        if target_orientation is None:
            target_orientation = start_orientation
        steps = max(1, int(params.get("steps", self.config.get("move_linear_steps", 80))))
        gripper = self._active_gripper_signal()
        for index in range(steps):
            self._check_interrupted()
            alpha = float(index + 1) / float(steps)
            waypoint = start_position * (1.0 - alpha) + target_position * alpha
            current_position, current_orientation = self._environment.get_eef_pose()
            action = self._pose_action(
                waypoint - current_position,
                PoseUtils.quaternion_error(current_orientation, target_orientation),
                gripper,
            )
            self._environment.step(action)
        converged, settle_steps = self._servo_pose(
            target_position,
            target_orientation,
            max_steps=int(params.get("settle_steps", 40)),
        )
        if not converged:
            return f"Failed: move_linear did not converge after {steps + settle_steps} steps"
        return (
            "End-effector moved linearly to "
            f"({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}) m."
        )

    def _set_gripper(self, params: dict[str, Any]) -> str:
        if "opening_m" in params:
            opening = float(params["opening_m"])
        elif "width_m" in params:
            opening = float(params["width_m"])
        elif str(params.get("command", "")).strip().lower() == "open":
            opening = self.gripper_max_opening
        elif str(params.get("command", "")).strip().lower() == "close":
            opening = 0.0
        else:
            raise ValueError("set_gripper requires opening_m, width_m, or command")
        if not 0.0 <= opening <= self.gripper_max_opening:
            raise ValueError(
                f"gripper opening must be between 0 and {self.gripper_max_opening:.4f} m"
            )
        signal = self._opening_to_signal(opening)
        steps = max(1, int(params.get("steps", self.config.get("gripper_steps", 30))))
        for _ in range(steps):
            self._check_interrupted()
            self._environment.step(np.asarray([0.0] * 6 + [signal], dtype=np.float32))
        self._gripper_command = float(signal)
        return f"Set gripper opening command to {opening:.4f} m."

    def _follow_arc(self, params: dict[str, Any]) -> str:
        center = PoseUtils.vector(params.get("center"), name="center")
        axis = PoseUtils.vector(params.get("axis"), name="axis")
        if "radius_m" not in params:
            raise ValueError("follow_arc requires radius_m")
        radius = float(params["radius_m"])
        if radius <= 0.0:
            raise ValueError("follow_arc.radius_m must be positive")
        angle = self._resolve_angle(params)
        orientation_mode = str(params.get("orientation_mode", "fixed")).strip().lower()
        if orientation_mode not in {"fixed", "co_rotate"}:
            raise ValueError("orientation_mode must be fixed or co_rotate")
        steps = max(1, int(params.get("steps", self.config.get("follow_arc_steps", 100))))
        start_position, start_orientation = self._environment.get_eef_pose()
        requested_orientation = PoseUtils.resolve_orientation(params)
        if requested_orientation is not None:
            start_orientation = requested_orientation
        start_vector = start_position - center
        norm = float(np.linalg.norm(start_vector))
        if norm <= 1e-10:
            raise ValueError("end-effector cannot start at the arc center")
        start_vector = start_vector / norm * radius
        axis_unit = axis / np.linalg.norm(axis)
        gripper = self._active_gripper_signal()
        for index in range(steps):
            self._check_interrupted()
            step_angle = angle * float(index + 1) / float(steps)
            target_position = center + PoseUtils.rotate_vector(start_vector, axis_unit, step_angle)
            if orientation_mode == "co_rotate":
                delta_quaternion = PoseUtils.axis_angle_to_quaternion(axis_unit * step_angle)
                target_orientation = PoseUtils.quaternion_multiply(
                    delta_quaternion, start_orientation
                )
            else:
                target_orientation = start_orientation
            current_position, current_orientation = self._environment.get_eef_pose()
            action = self._pose_action(
                target_position - current_position,
                PoseUtils.quaternion_error(current_orientation, target_orientation),
                gripper,
            )
            self._environment.step(action)
        return (
            "Followed arc around center "
            f"({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) m "
            f"for {math.degrees(angle):.2f} deg."
        )

    def _servo_pose(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        *,
        max_steps: int,
    ) -> tuple[bool, int]:
        gripper = self._active_gripper_signal()
        for index in range(max(1, max_steps)):
            self._check_interrupted()
            current_position, current_orientation = self._environment.get_eef_pose()
            position_error = target_position - current_position
            rotation_error = PoseUtils.quaternion_error(
                current_orientation, target_orientation
            )
            if (
                float(np.linalg.norm(position_error)) <= self.position_tolerance
                and float(np.linalg.norm(rotation_error)) <= self.rotation_tolerance
            ):
                return True, index
            self._environment.step(
                self._pose_action(position_error, rotation_error, gripper)
            )
        return False, max(1, max_steps)

    def _pose_action(
        self,
        position_error: np.ndarray,
        rotation_error: np.ndarray,
        gripper_signal: float,
    ) -> np.ndarray:
        return PoseUtils.osc_action(
            position_error,
            rotation_error,
            gripper_signal,
            position_scale=self.position_scale,
            rotation_scale=self.rotation_scale,
            position_gain=self.position_gain,
            rotation_gain=self.rotation_gain,
        )

    def _opening_to_signal(self, opening: float) -> float:
        if self.gripper_max_opening <= 0.0:
            return -1.0
        return float(np.clip(1.0 - 2.0 * opening / self.gripper_max_opening, -1.0, 1.0))

    def _resolve_angle(self, params: dict[str, Any]) -> float:
        if "angle_rad" in params:
            return float(params["angle_rad"])
        if "angle_deg" in params:
            return math.radians(float(params["angle_deg"]))
        if "angle" in params:
            unit = str(params.get("angle_unit", "degrees")).strip().lower()
            value = float(params["angle"])
            return value if unit in {"rad", "radian", "radians"} else math.radians(value)
        raise ValueError("follow_arc requires angle_rad, angle_deg, or angle")

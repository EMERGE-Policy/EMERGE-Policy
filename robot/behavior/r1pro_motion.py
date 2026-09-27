"""Plan bounded R1Pro joint commands for the BEHAVIOR action multiplexer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot.behavior.r1pro_kinematics import IKSolution, R1ProKinematics

ARM_ACTION_SLICE = {"left": slice(7, 14), "right": slice(15, 22)}
GRIPPER_ACTION_INDEX = {"left": 14, "right": 22}
MAX_STATE_MODEL_POSITION_ERROR_M = 0.05


@dataclass(frozen=True, slots=True)
class MotionPlan:
    actions: tuple[tuple[float, ...], ...]
    final_position_error_m: float
    description: str


def plan_move_to_pose(
    kinematics: R1ProKinematics,
    robot_state: dict[str, Any],
    hold_action: list[float],
    *,
    side: str,
    position_m: list[float],
    orientation_quat: list[float] | None = None,
    max_joint_step_rad: float = 0.03,
    max_steps: int = 240,
) -> MotionPlan:
    current, trunk = _joint_state(robot_state, side)
    _validate_model_state(kinematics, robot_state, side, current, trunk)
    target_orientation = _target_orientation(robot_state, side, orientation_quat)
    solution = kinematics.solve(
        side,
        position_m,
        trunk,
        current,
        target_orientation_xyzw=target_orientation,
    )
    _require_solution(solution, side)
    target = np.asarray(solution.arm_qpos, dtype=np.float64)
    actions = _interpolate_arm_actions(
        hold_action,
        side=side,
        start=current,
        target=target,
        max_joint_step_rad=max_joint_step_rad,
        max_steps=max_steps,
    )
    return MotionPlan(
        actions=tuple(tuple(action) for action in actions),
        final_position_error_m=solution.position_error_m,
        description=f"{side} arm joint-space approach to {list(map(float, position_m))}",
    )


def plan_move_linear(
    kinematics: R1ProKinematics,
    robot_state: dict[str, Any],
    hold_action: list[float],
    *,
    side: str,
    position_m: list[float],
    orientation_quat: list[float] | None = None,
    max_cartesian_step_m: float = 0.01,
    max_joint_step_rad: float = 0.03,
    max_steps: int = 240,
) -> MotionPlan:
    current, trunk = _joint_state(robot_state, side)
    _validate_model_state(kinematics, robot_state, side, current, trunk)
    target_orientation = _target_orientation(robot_state, side, orientation_quat)
    start_position = np.asarray(robot_state["arms"][side]["eef_position"], dtype=np.float64)
    target_position = _vector(position_m, 3, "position_m")
    distance = float(np.linalg.norm(target_position - start_position))
    waypoint_count = max(1, int(math.ceil(distance / float(max_cartesian_step_m))))
    actions: list[list[float]] = []
    seed = current
    final_error = math.inf
    for index in range(1, waypoint_count + 1):
        fraction = index / waypoint_count
        waypoint = start_position + fraction * (target_position - start_position)
        solution = kinematics.solve(
            side,
            waypoint,
            trunk,
            seed,
            target_orientation_xyzw=target_orientation,
        )
        _require_solution(solution, side)
        target = np.asarray(solution.arm_qpos, dtype=np.float64)
        remaining = max_steps - len(actions)
        if remaining < 1:
            raise ValueError(f"linear motion requires more than {max_steps} simulator steps")
        segment = _interpolate_arm_actions(
            hold_action,
            side=side,
            start=seed,
            target=target,
            max_joint_step_rad=max_joint_step_rad,
            max_steps=remaining,
        )
        actions.extend(segment)
        seed = target
        final_error = solution.position_error_m
    return MotionPlan(
        actions=tuple(tuple(action) for action in actions),
        final_position_error_m=final_error,
        description=f"{side} arm Cartesian line to {target_position.tolist()}",
    )


def plan_set_gripper(
    hold_action: list[float],
    *,
    side: str,
    command: str | None = None,
    opening_m: float | None = None,
    steps: int = 15,
) -> MotionPlan:
    _side(side)
    if (command is None) == (opening_m is None):
        raise ValueError("set_gripper requires exactly one of command or opening_m")
    if command is not None:
        normalized_command = str(command).strip().lower()
        if normalized_command not in {"open", "close"}:
            raise ValueError("set_gripper command must be 'open' or 'close'")
        target = 1.0 if normalized_command == "open" else -1.0
    else:
        width = float(opening_m)
        if not math.isfinite(width) or not 0.0 <= width <= 0.1:
            raise ValueError("set_gripper opening_m must be between 0.0 and 0.1")
        target = 2.0 * width / 0.1 - 1.0
    if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 120:
        raise ValueError("set_gripper steps must be an integer between 1 and 120")
    index = GRIPPER_ACTION_INDEX[side]
    start = float(hold_action[index])
    actions = []
    for fraction in np.linspace(1.0 / steps, 1.0, steps):
        action = list(map(float, hold_action))
        action[index] = start + float(fraction) * (target - start)
        actions.append(tuple(action))
    return MotionPlan(
        actions=tuple(actions),
        final_position_error_m=0.0,
        description=f"{side} gripper command {target:.3f}",
    )


def _joint_state(robot_state: dict[str, Any], side: str) -> tuple[np.ndarray, np.ndarray]:
    _side(side)
    try:
        arm = _vector(robot_state["arms"][side]["qpos"], 7, f"{side} arm qpos")
        trunk = _vector(robot_state["trunk"]["qpos"], 4, "trunk qpos")
    except (KeyError, TypeError) as exc:
        raise ValueError("latest R1Pro state does not contain arm and trunk qpos") from exc
    return arm, trunk


def _target_orientation(
    robot_state: dict[str, Any], side: str, requested: list[float] | None
) -> np.ndarray:
    if requested is not None:
        return _normalized_quaternion(requested, "orientation_quat")
    try:
        current = robot_state["arms"][side]["eef_orientation_xyzw"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"latest R1Pro state does not contain the {side} end-effector orientation"
        ) from exc
    return _normalized_quaternion(current, f"{side} end-effector orientation")


def _normalized_quaternion(value: Any, name: str) -> np.ndarray:
    quaternion = _vector(value, 4, name)
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError(f"{name} must be non-zero")
    return quaternion / norm


def _interpolate_arm_actions(
    hold_action: list[float],
    *,
    side: str,
    start: np.ndarray,
    target: np.ndarray,
    max_joint_step_rad: float,
    max_steps: int,
) -> list[list[float]]:
    if max_joint_step_rad <= 0.0:
        raise ValueError("max_joint_step_rad must be positive")
    count = max(1, int(math.ceil(float(np.max(np.abs(target - start))) / max_joint_step_rad)))
    if count > max_steps:
        raise ValueError(f"motion requires {count} simulator steps, limit is {max_steps}")
    result = []
    action_slice = ARM_ACTION_SLICE[_side(side)]
    for fraction in np.linspace(1.0 / count, 1.0, count):
        action = list(map(float, hold_action))
        action[action_slice] = (start + float(fraction) * (target - start)).tolist()
        result.append(action)
    return result


def _validate_model_state(
    kinematics: R1ProKinematics,
    robot_state: dict[str, Any],
    side: str,
    arm: np.ndarray,
    trunk: np.ndarray,
) -> None:
    try:
        observed = _vector(
            robot_state["arms"][side]["eef_position"],
            3,
            f"{side} end-effector position",
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"latest R1Pro state does not contain the {side} end-effector position"
        ) from exc
    predicted = kinematics.forward(side, trunk, arm)[:3, 3]
    error = float(np.linalg.norm(predicted - observed))
    if error > MAX_STATE_MODEL_POSITION_ERROR_M:
        raise ValueError(
            f"{side} R1Pro kinematics does not match the simulator state: "
            f"FK error is {error:.4f} m (limit {MAX_STATE_MODEL_POSITION_ERROR_M:.4f} m)"
        )


def _require_solution(solution: IKSolution, side: str) -> None:
    if not solution.success:
        raise ValueError(
            f"{side} arm IK failed: position_error={solution.position_error_m:.4f} m, "
            f"orientation_error={solution.orientation_error_rad}, {solution.message}"
        )


def _side(side: str) -> str:
    value = str(side).strip().lower()
    if value not in ARM_ACTION_SLICE:
        raise ValueError("side must be 'left' or 'right'")
    return value


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {length} finite values")
    return vector

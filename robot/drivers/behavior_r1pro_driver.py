"""Native EMERGE driver for the BEHAVIOR R1Pro evaluation bridge."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from robot.behavior import contracts
from robot.behavior.approach import compute_precontact_pose
from robot.behavior.gateway import (
    EPISODE_FINISHED,
    EPISODE_RESET,
    ERROR,
    EXECUTING,
    INTERRUPTED,
    NO_PROGRESS,
    STEP_COMPLETED,
    TERMINAL_EXECUTION_STATES,
    TIMEOUT,
    B1KProxyGateway,
)
from robot.behavior.r1pro_kinematics import R1ProKinematics
from robot.behavior.r1pro_motion import (
    plan_move_linear,
    plan_move_to_pose,
    plan_set_gripper,
)
from robot.drivers.base_driver import BaseDriver, CancelCheck

_SUCCESS_STATES = frozenset({STEP_COMPLETED, EPISODE_FINISHED, EPISODE_RESET})
_OK_SYNC_STATES = frozenset({STEP_COMPLETED, EPISODE_FINISHED})


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class BehaviorR1ProDriver(BaseDriver):
    """Expose bounded BEHAVIOR R1Pro execution through the EMERGE action queue."""

    ROBOT_ID = "behavior_r1pro"

    def __init__(
        self,
        gui: bool = False,
        workspace: str | Path = "~/.Emerge/workspace",
        upstream_url: str = "ws://127.0.0.1:8011",
        listen_host: str = "127.0.0.1",
        listen_port: int = 8000,
        upstream_open_timeout_s: float = 10.0,
        required_instruction: str = "",
        max_action_steps: int = 4000,
        status_path: str | Path | None = None,
        observation_publish_interval: int = 5,
        camera_calibration: dict[str, Any] | None = None,
        r1pro_urdf_path: str | Path | None = None,
        primitive_max_steps: int = 240,
        profile_path: str | Path | None = None,
        execution_timeout_s: float = 1800.0,
        progress_timeout_s: float = 300.0,
        state_sync_timeout_s: float = 120.0,
        expected_task: str = "",
        expected_instance_id: int | None = None,
        **_kwargs: Any,
    ) -> None:
        del gui
        config = contracts.validate_driver_config(
            {
                "workspace": str(workspace),
                "upstream_url": upstream_url,
                "listen_host": listen_host,
                "listen_port": listen_port,
                "upstream_open_timeout_s": upstream_open_timeout_s,
                "required_instruction": required_instruction,
                "max_action_steps": max_action_steps,
                "status_path": status_path,
                "observation_publish_interval": observation_publish_interval,
                "camera_calibration": camera_calibration,
                "r1pro_urdf_path": str(r1pro_urdf_path) if r1pro_urdf_path else None,
                "primitive_max_steps": primitive_max_steps,
                "profile_path": profile_path,
                "execution_timeout_s": execution_timeout_s,
                "progress_timeout_s": progress_timeout_s,
                "state_sync_timeout_s": state_sync_timeout_s,
                "expected_task": expected_task,
                "expected_instance_id": expected_instance_id,
            }
        )
        self.workspace = Path(config["workspace"]).expanduser().resolve()
        self.required_instruction = config["required_instruction"]
        self.max_action_steps = config["max_action_steps"]
        self._expected_task = config["expected_task"]
        self._expected_instance_id = config["expected_instance_id"]
        self._execution_timeout_s = config["execution_timeout_s"]
        self._progress_timeout_s = config["progress_timeout_s"]
        self._state_sync_timeout_s = config["state_sync_timeout_s"]
        self._primitive_max_steps = config["primitive_max_steps"]
        self._kinematics = (
            R1ProKinematics(config["r1pro_urdf_path"])
            if config["r1pro_urdf_path"]
            else None
        )
        configured_status = config["status_path"]
        self.status_path = (
            Path(configured_status).expanduser().resolve()
            if configured_status
            else self.workspace / "BEHAVIOR_STATUS.json"
        )
        default_profile = Path(__file__).resolve().parents[1] / "profiles/behavior_r1pro.md"
        configured_profile = (
            Path(config["profile_path"]).expanduser()
            if config["profile_path"]
            else default_profile
        )
        if not configured_profile.is_absolute():
            configured_profile = Path(__file__).resolve().parents[2] / configured_profile
        self._profile_path = configured_profile.resolve()
        self._last_sync_ok: bool | None = None
        self._last_applied_steps: int | None = None
        self._last_verdict: str | None = None
        self._gateway_action_steps = 0
        self._gateway_execution_count = 0
        self._gateway = B1KProxyGateway(
            upstream_url=config["upstream_url"],
            listen_host=config["listen_host"],
            listen_port=config["listen_port"],
            upstream_open_timeout_s=config["upstream_open_timeout_s"],
            controlled=True,
            observation_workspace=self.workspace,
            observation_publish_interval=config["observation_publish_interval"],
            camera_calibration=config["camera_calibration"],
        )

    # ------------------------------------------------------------------ #
    # BaseDriver
    # ------------------------------------------------------------------ #

    def get_profile_path(self) -> Path:
        return self._profile_path

    def load_scene(self, scene: dict[str, dict]) -> None:
        del scene
        self._gateway.start()

    def execute_action(
        self,
        action_type: str,
        params: dict,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        if action_type == "gateway_status":
            state = self._gateway.snapshot()
            return (
                "Gateway status: "
                f"{state['status']}, active_connections={state['active_connections']}, "
                f"requests_forwarded={state['requests_forwarded']}, "
                f"control_mode={state['control_mode']}, "
                f"base_motion_enabled={state['base_motion_enabled']}."
            )
        supported = {
            "vla_execute",
            "hold_position",
            "move_to_pose",
            "move_linear",
            "set_gripper",
            "set_control_mode",
        }
        if action_type not in supported:
            return f"Unknown action type: {action_type!r}."

        if action_type == "set_control_mode":
            if not isinstance(params, dict):
                return f"{contracts.FAILED_PREFIX} set_control_mode parameters must be an object"
            unknown = sorted(set(params) - {"mode"})
            if unknown:
                return (
                    f"{contracts.FAILED_PREFIX} set_control_mode received unknown parameters: "
                    + ", ".join(unknown)
                )
            mode = params.get("mode")
            if not isinstance(mode, str) or not mode.strip():
                return (
                    f"{contracts.FAILED_PREFIX} set_control_mode requires a non-empty mode"
                )
            try:
                changed = self._gateway.set_control_mode(mode)
            except (RuntimeError, ValueError) as exc:
                return f"{contracts.FAILED_PREFIX} {exc}"
            state = self._gateway.snapshot()
            current = state["control_mode"]
            base = "enabled" if state["base_motion_enabled"] else "locked"
            transition = "changed" if changed else "already active"
            reset_note = (
                "OpenPI will replan before the next VLA action."
                if changed
                else "The control boundary is unchanged."
            )
            return (
                f"Control mode {transition}: {current}; base motion is {base}. "
                f"{reset_note}"
            )

        try:
            if action_type == "vla_execute":
                instruction, step = contracts.validate_action(
                    params,
                    required_instruction=self.required_instruction,
                    max_steps=self.max_action_steps,
                )
            elif action_type == "hold_position":
                if not isinstance(params, dict):
                    raise contracts.ActionValidationError(
                        "hold_position parameters must be an object"
                    )
                unknown = sorted(set(params) - {"step"})
                if unknown:
                    raise contracts.ActionValidationError(
                        "hold_position received unknown parameters: " + ", ".join(unknown)
                    )
                step = contracts.validate_step(
                    params.get("step", 1), max_steps=self.max_action_steps
                )
                instruction = "hold_position"
            else:
                if not isinstance(params, dict):
                    raise contracts.ActionValidationError(
                        f"{action_type} parameters must be an object"
                    )
                step = 0
                instruction = action_type
        except contracts.ActionValidationError as exc:
            return f"{contracts.FAILED_PREFIX} {exc}"

        try:
            problem = self._readiness_problem()
        except contracts.EvaluationValidationError as exc:
            return f"{contracts.FAILED_PREFIX} official evaluator status is unusable: {exc}"
        if problem:
            return f"{contracts.FAILED_PREFIX} {problem}"

        evaluation = self._read_evaluation()
        baseline_steps = _int_or_none((evaluation or {}).get("steps"))

        try:
            if action_type == "vla_execute":
                execution_id = self._gateway.begin_execution(
                    instruction,
                    step,
                    baseline_evaluator_steps=baseline_steps,
                )
            elif action_type == "hold_position":
                execution_id = self._gateway.begin_local_execution(
                    instruction,
                    step,
                    self._gateway.current_hold_action(),
                    baseline_evaluator_steps=baseline_steps,
                )
            else:
                if self._kinematics is None:
                    raise ValueError(
                        "R1Pro motion primitives are unavailable because r1pro_urdf_path is not configured"
                    )
                state = self._gateway.current_robot_state()
                hold = self._gateway.current_hold_action()
                if action_type == "move_to_pose":
                    unknown = sorted(
                        set(params)
                        - {
                            "side",
                            "position_m",
                            "orientation_quat",
                            "bbox_world_min_m",
                            "bbox_world_max_m",
                            "standoff_m",
                        }
                    )
                    if unknown:
                        raise ValueError("move_to_pose received unknown parameters: " + ", ".join(unknown))
                    has_position = "position_m" in params
                    has_bbox = (
                        "bbox_world_min_m" in params or "bbox_world_max_m" in params
                    )
                    if has_position == has_bbox:
                        raise ValueError(
                            "move_to_pose requires exactly one of position_m or a "
                            "bbox_world_min_m/bbox_world_max_m pair"
                        )
                    if has_position and "standoff_m" in params:
                        raise ValueError(
                            "standoff_m is valid only with bbox pre-contact geometry"
                        )
                    side = params.get("side")
                    position = params.get("position_m")
                    orientation = params.get("orientation_quat")
                    if has_bbox:
                        if "bbox_world_min_m" not in params or "bbox_world_max_m" not in params:
                            raise ValueError(
                                "bbox pre-contact requires bbox_world_min_m and bbox_world_max_m"
                            )
                        if orientation is not None:
                            raise ValueError(
                                "bbox pre-contact preserves the current wrist orientation; "
                                "do not provide orientation_quat"
                            )
                        precontact = compute_precontact_pose(
                            state,
                            params["bbox_world_min_m"],
                            params["bbox_world_max_m"],
                            side=side,
                            standoff_m=params.get("standoff_m", 0.12),
                        )
                        side = precontact.side
                        position = list(precontact.position_m)
                        orientation = list(precontact.orientation_quat)
                    plan = plan_move_to_pose(
                        self._kinematics,
                        state,
                        hold,
                        side=side,
                        position_m=position,
                        orientation_quat=orientation,
                        max_steps=self._primitive_max_steps,
                    )
                elif action_type == "move_linear":
                    unknown = sorted(
                        set(params) - {"side", "position_m", "delta_m", "orientation_quat"}
                    )
                    if unknown:
                        raise ValueError("move_linear received unknown parameters: " + ", ".join(unknown))
                    plan = plan_move_linear(
                        self._kinematics,
                        state,
                        hold,
                        side=params.get("side"),
                        position_m=params.get("position_m"),
                        delta_m=params.get("delta_m"),
                        orientation_quat=params.get("orientation_quat"),
                        max_steps=self._primitive_max_steps,
                    )
                else:
                    unknown = sorted(set(params) - {"side", "command", "opening_m", "steps"})
                    if unknown:
                        raise ValueError("set_gripper received unknown parameters: " + ", ".join(unknown))
                    plan = plan_set_gripper(
                        hold,
                        side=params.get("side"),
                        command=params.get("command"),
                        opening_m=params.get("opening_m"),
                        steps=params.get("steps", 15),
                    )
                execution_id = self._gateway.begin_local_trajectory(
                    plan.description,
                    [list(action) for action in plan.actions],
                    baseline_evaluator_steps=baseline_steps,
                )
            result = self._gateway.wait_for_execution(
                execution_id,
                cancel_check=cancel_check,
                progress_timeout_s=self._progress_timeout_s,
                total_timeout_s=self._execution_timeout_s,
            )
        except (RuntimeError, ValueError) as exc:
            return f"{contracts.FAILED_PREFIX} {exc}"

        completed = int(result.get("completed_steps") or 0)
        self._gateway_action_steps += completed
        self._gateway_execution_count += 1
        status = str(result.get("status") or ERROR)
        detail = str(result.get("detail") or "")

        if status not in TERMINAL_EXECUTION_STATES:
            self._last_verdict = "unexpected_status"
            return f"{contracts.FAILED_PREFIX} unexpected robot execution status {status!r}"
        if status == INTERRUPTED:
            self._last_sync_ok = False
            self._last_verdict = "interrupted"
            return (
                f"{contracts.INTERRUPTED_PREFIX} {detail or 'action interrupted'} "
                f"after {completed} step(s)."
            )
        if status in {ERROR, TIMEOUT, NO_PROGRESS}:
            self._last_sync_ok = False
            self._last_verdict = status
            return (
                f"{contracts.FAILED_PREFIX} {action_type} did not complete "
                f"({status}) after {completed} step(s): {detail}"
            )

        synced, applied_steps = self._synchronize(result, cancel_check=cancel_check)
        self._last_sync_ok = synced
        self._last_applied_steps = applied_steps
        self._last_verdict = status

        if not synced:
            return (
                f"{contracts.FAILED_PREFIX} {completed} robot action(s) were issued but the "
                "official evaluator state could not be confirmed afterwards; "
                "the simulator may not have applied them."
            )
        if status in _SUCCESS_STATES:
            applied = f", applied_steps={applied_steps}" if applied_steps is not None else ""
            labels = {
                "vla_execute": "VLA execution",
                "hold_position": "Hold execution",
                "move_to_pose": "Move-to-pose execution",
                "move_linear": "Linear execution",
                "set_gripper": "Gripper execution",
            }
            label = labels[action_type]
            return f"{label} finished: {status}, steps={completed}{applied}."
        return f"{contracts.FAILED_PREFIX} unexpected robot execution status {status!r}"

    def get_runtime_state(self) -> dict[str, Any]:
        state = self._gateway.snapshot()
        try:
            evaluation = self._read_evaluation()
        except contracts.EvaluationValidationError as exc:
            evaluation = {"unusable": f"{type(exc).__name__}: {exc}"}
        execution = state.get("execution")
        if isinstance(execution, dict) and execution.get("status") == EXECUTING:
            status = "executing"
        elif not state.get("evaluator_connected"):
            status = "waiting_for_evaluator"
        else:
            status = "idle"
        payload = {key: value for key, value in state.items() if key != "status"}
        return {
            "robots": {
                self.ROBOT_ID: {
                    "mode": "controlled_pi05_gateway",
                    "effective_prompt": self.required_instruction,
                    **payload,
                    "status": status,
                    "gateway_status": state.get("status"),
                    "max_action_steps": self.max_action_steps,
                    "state_synced": self._last_sync_ok,
                    "applied_steps": self._last_applied_steps,
                    "last_verdict": self._last_verdict,
                    "runtime_progress": {
                        "source": "emerge_gateway_diagnostic",
                        "gateway_action_steps": self._gateway_action_steps,
                        "execution_count": self._gateway_execution_count,
                        "official_steps_available": _int_or_none(
                            (evaluation or {}).get("steps")
                        )
                        is not None,
                    },
                    "execution_timeout_s": self._execution_timeout_s,
                    "progress_timeout_s": self._progress_timeout_s,
                    "state_sync_timeout_s": self._state_sync_timeout_s,
                    "evaluation": evaluation,
                    "evaluation_status_path": str(self.status_path),
                }
            }
        }

    def close(self) -> None:
        self._gateway.stop()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _read_evaluation(self) -> dict[str, Any] | None:
        """Read and validate the official evaluator status.

        Returns ``None`` only when the evaluator has not published yet.  A
        present-but-unusable file raises, because silently treating it as
        "not finished" would let the Agent keep issuing actions against a
        broken evaluation.
        """
        try:
            raw = self.status_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise contracts.EvaluationValidationError(f"cannot read {self.status_path}: {exc}") from exc
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise contracts.EvaluationValidationError(
                f"{self.status_path} is not valid JSON: {exc}"
            ) from exc
        evaluation = contracts.validate_evaluation(
            payload,
            expected_task=self._expected_task,
            expected_instance_id=self._expected_instance_id,
        )
        if self._expected_instance_id is None:
            observed = _int_or_none(evaluation.get("instance_id"))
            if observed is not None:
                self._expected_instance_id = observed
        return evaluation

    def _readiness_problem(self) -> str | None:
        """Reject an action that cannot produce a meaningful episode step."""
        state = self._gateway.snapshot()
        if not state.get("evaluator_connected"):
            return "no BEHAVIOR evaluator is connected to the gateway yet"
        if not state.get("upstream_connected"):
            return "the OpenPI policy upstream is not connected"
        evaluation = self._read_evaluation()
        if evaluation is not None and contracts.episode_terminated(evaluation):
            return (
                "the official episode already finished "
                f"(phase={evaluation.get('phase')!r}, "
                f"success={evaluation.get('success')!r}, "
                f"termination_reason={evaluation.get('termination_reason')!r})"
            )
        return None

    def _synchronize(
        self,
        result: dict[str, Any],
        *,
        cancel_check: CancelCheck | None,
    ) -> tuple[bool, int | None]:
        """Confirm that the executed actions reached the simulator.

        The primary signal is the next observation frame, which the evaluator
        only produces after applying the action.  The evaluator step counter is
        used as a fallback for the final action of an episode, where no further
        frame is sent.
        """
        baseline = _int_or_none(result.get("sync_observation_baseline"))
        if baseline is None:
            baseline = self._gateway.observation_count
        if self._gateway.wait_for_observation_after(
            baseline,
            timeout_s=self._state_sync_timeout_s,
            cancel_check=cancel_check,
        ):
            self._gateway.publish_observation()
            return True, self._applied_steps_from_status()

        # BEHAVIOR 3.9.3 writes the official result after its websocket closes.
        # Give the runner a short window to translate that result into the
        # episode status before declaring the final action unsynchronized.
        evaluation = None
        status = str(result.get("status") or "")
        status_deadline = time.monotonic() + min(2.0, self._state_sync_timeout_s)
        while True:
            try:
                evaluation = self._read_evaluation()
            except contracts.EvaluationValidationError:
                return False, None
            if contracts.episode_terminated(evaluation):
                break
            if status != EPISODE_FINISHED or time.monotonic() >= status_deadline:
                break
            time.sleep(0.05)
        if evaluation is None:
            return False, None
        applied = _int_or_none(evaluation.get("steps"))
        if contracts.episode_terminated(evaluation):
            return True, applied
        baseline_steps = _int_or_none(result.get("baseline_evaluator_steps"))
        completed = _int_or_none(result.get("completed_steps")) or 0
        if applied is not None and baseline_steps is not None and applied >= baseline_steps + completed:
            return True, applied
        return False, applied

    def _applied_steps_from_status(self) -> int | None:
        try:
            evaluation = self._read_evaluation()
        except contracts.EvaluationValidationError:
            return None
        if evaluation is None:
            return None
        return _int_or_none(evaluation.get("steps"))

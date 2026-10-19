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
    plan_adjust_posture,
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
        navigation_only: bool = False,
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
                "navigation_only": navigation_only,
            }
        )
        self.workspace = Path(config["workspace"]).expanduser().resolve()
        self.required_instruction = config["required_instruction"]
        self._navigation_only = config["navigation_only"]
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
        self._last_navigation_result: dict[str, Any] | None = None
        self._navigation_results: list[dict[str, Any]] = []
        self._posture_results: list[dict[str, Any]] = []
        self._navigation_client: Any | None = None
        self._navigation_episode_id: int | None = None
        self._navigation_server_url: str | None = None
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
                f"requests_forwarded={state['requests_forwarded']}."
            )
        supported = {
            "vla_execute",
            "navigate_vln",
            "adjust_posture",
            "hold_position",
            "move_to_pose",
            "move_linear",
            "set_gripper",
        }
        if action_type not in supported:
            return f"Unknown action type: {action_type!r}."

        if self._navigation_only and action_type not in {"navigate_vln", "adjust_posture", "gateway_status"}:
            return f"{contracts.FAILED_PREFIX} {action_type} is disabled in this navigation-only experiment"

        if self._navigation_only and self._remaining_steps() < 1:
            return f"{contracts.FAILED_PREFIX} episode action budget is exhausted"

        if action_type == "navigate_vln":
            return self._execute_navigation(params, cancel_check=cancel_check)

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
        posture_targets: dict[str, float] = {}
        approach_report: dict[str, Any] | None = None

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
                            "steps",
                            "settle_steps",
                            "coarse",
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
                        steps=params.get("steps"),
                        settle_steps=params.get("settle_steps", 0),
                        coarse=params.get("coarse", False),
                        max_steps=self._primitive_max_steps,
                    )
                    approach_report = plan.approach_report
                    if has_bbox and approach_report is not None:
                        endpoint = approach_report["planned_reachable_position_m"]
                        if all(
                            lower <= value <= upper
                            for lower, value, upper in zip(
                                params["bbox_world_min_m"], endpoint, params["bbox_world_max_m"], strict=True
                            )
                        ):
                            raise ValueError("coarse approach endpoint is inside the measured object box")
                elif action_type == "move_linear":
                    unknown = sorted(
                        set(params) - {"side", "position_m", "delta_m", "orientation_quat", "steps", "settle_steps"}
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
                        steps=params.get("steps"),
                        settle_steps=params.get("settle_steps", 0),
                        max_steps=self._primitive_max_steps,
                    )
                elif action_type == "adjust_posture":
                    unknown = sorted(set(params) - {"joint_deltas_rad", "max_steps"})
                    if unknown:
                        raise ValueError("adjust_posture received unknown parameters: " + ", ".join(unknown))
                    available = min(self._primitive_max_steps, self._remaining_steps())
                    budget = contracts.validate_step(params.get("max_steps", available), max_steps=available)
                    plan = plan_adjust_posture(
                        self._kinematics, state, hold,
                        joint_deltas_rad=params.get("joint_deltas_rad"), max_steps=budget,
                    )
                    measured = self._kinematics.body_state(state)["joint_positions_rad"]
                    posture_targets = {
                        name: float(measured[name] + float(delta))
                        for name, delta in params["joint_deltas_rad"].items()
                    }
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
        if action_type == "adjust_posture" and status not in _SUCCESS_STATES:
            self._record_failed_posture(params, posture_targets, completed, status, detail)

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
            if action_type == "adjust_posture":
                self._record_failed_posture(
                    params, posture_targets, completed, "unconfirmed",
                    "post-action observation could not be synchronized",
                )
            return (
                f"{contracts.FAILED_PREFIX} {completed} robot action(s) were issued but the "
                "official evaluator state could not be confirmed afterwards; "
                "the simulator may not have applied them."
            )
        if status in _SUCCESS_STATES:
            if action_type == "adjust_posture":
                return self._posture_completed(params, completed, status, posture_targets)
            applied = f", applied_steps={applied_steps}" if applied_steps is not None else ""
            labels = {
                "vla_execute": "VLA execution",
                "hold_position": "Hold execution",
                "move_to_pose": "Move-to-pose execution",
                "move_linear": "Linear execution",
                "set_gripper": "Gripper execution",
            }
            label = labels[action_type]
            report = (
                f" Coarse approach plan: {json.dumps(approach_report, ensure_ascii=False)}"
                if approach_report is not None else ""
            )
            return f"{label} finished: {status}, steps={completed}{applied}.{report}"
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
                    "remaining_action_steps": self._remaining_steps(),
                    "body_posture": self._body_posture(),
                    "state_synced": self._last_sync_ok,
                    "applied_steps": self._last_applied_steps,
                    "last_verdict": self._last_verdict,
                    "runtime_progress": {
                        "source": "emerge_gateway_diagnostic",
                        "gateway_action_steps": self._gateway_action_steps,
                        "execution_count": self._gateway_execution_count,
                        **({"navigation_calls": self._navigation_results} if self._navigation_results else {}),
                        **({"posture_calls": self._posture_results} if self._posture_results else {}),
                        **({"last_navigation": self._last_navigation_result}
                           if self._last_navigation_result is not None else {}),
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
        try:
            self._close_navigation_client()
        finally:
            self._gateway.stop()

    def _close_navigation_client(self) -> None:
        client, self._navigation_client = self._navigation_client, None
        self._navigation_episode_id = None
        self._navigation_server_url = None
        if client is not None:
            client.close()

    def _remaining_steps(self) -> int:
        return max(0, self.max_action_steps - self._gateway_action_steps) if self._navigation_only else self.max_action_steps

    def _body_posture(self) -> dict[str, Any]:
        if self._kinematics is None:
            return {"available": False, "reason": "R1Pro URDF is not configured"}
        try:
            return {"available": True, **self._kinematics.body_state(self._gateway.current_robot_state())}
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return {"available": False, "reason": str(exc)}

    def _record_failed_posture(
        self, params: dict[str, Any], targets: dict[str, float], completed: int,
        status: str, detail: str,
    ) -> None:
        record = {
            "posture_id": len(self._posture_results) + 1,
            "requested_joint_deltas_rad": params["joint_deltas_rad"],
            "target_joint_positions_rad": targets,
            "completed_steps": completed,
            "status": status,
            "error": detail or status,
            "observation_confirmed": False,
            "collision_checked": False,
        }
        self._posture_results.append(record)
        with (self.workspace / "lightnav_trace.jsonl").open("a", encoding="utf-8") as trace:
            trace.write(json.dumps({"event": "posture_failed", **record}, ensure_ascii=False) + "\n")

    def _posture_completed(
        self, params: dict[str, Any], completed: int, status: str, targets: dict[str, float]
    ) -> str:
        """Publish the actual multi-camera observation, not just command targets."""
        from integrations.lightnav_behavior.probe import load_head_frame

        record: dict[str, Any] = {
            "posture_id": len(self._posture_results) + 1,
            "requested_joint_deltas_rad": params["joint_deltas_rad"],
            "completed_steps": completed,
            "target_joint_positions_rad": targets,
            "status": status,
            "collision_checked": False,
            "observation_confirmed": False,
        }
        try:
            manifest = self._gateway.publish_observation()
            if manifest is None:
                raise RuntimeError("no synchronized multi-camera observation after posture motion")
            record["observation_manifest"] = str(manifest)
            record["actual_body_posture"] = self._body_posture()
            if not record["actual_body_posture"].get("available"):
                raise RuntimeError("measured body posture could not be reconstructed")
            measured = record["actual_body_posture"]["joint_positions_rad"]
            error = max(abs(float(measured[name]) - target) for name, target in targets.items())
            record["max_joint_tracking_error_rad"] = error
            record["pose_reached"] = error <= 0.05
            frame = load_head_frame(manifest)
            frame_dir = self.workspace / "lightnav_frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            indices = [int(path.stem.split("_")[-1]) for path in frame_dir.glob("frame_*.jpg")
                       if path.stem.split("_")[-1].isdigit()]
            frame_index = max(indices, default=-1) + 1
            image_path = frame_dir / f"frame_{frame_index:05d}.jpg"
            image_path.write_bytes(frame.jpeg)
            record.update({"observation_seq": frame.observation_seq, "episode_id": frame.episode_id,
                           "image_path": str(image_path), "frame_index": frame_index,
                           "observation_confirmed": True})
            if not record["pose_reached"]:
                record["error"] = "actual posture differs from the requested target by more than 0.05 rad"
        except (OSError, RuntimeError, ValueError) as exc:
            record["error"] = str(exc)
        self._posture_results.append(record)
        with (self.workspace / "lightnav_trace.jsonl").open("a", encoding="utf-8") as trace:
            trace.write(json.dumps({"event": "posture_completed", **record}, ensure_ascii=False) + "\n")
        prefix = contracts.FAILED_PREFIX if record.get("error") else "Posture execution finished:"
        return prefix + " " + json.dumps(record, ensure_ascii=False)

    @property
    def last_navigation_result(self) -> dict[str, Any] | None:
        return dict(self._last_navigation_result) if self._last_navigation_result else None

    def _execute_navigation(
        self, params: dict[str, Any], *, cancel_check: CancelCheck | None
    ) -> str:
        """Run LightNav against fresh head frames through the controlled gateway."""
        from integrations.lightnav_behavior.closed_loop import run_closed_loop
        from integrations.lightnav_behavior.probe import load_head_frame
        from integrations.lightnav_behavior.r1pro_adapter import frame_action_steps, waypoint_action
        from integrations.lightnav_sonic_emerge.bridge.lightnav_ws import LightNavWsClient

        if not isinstance(params, dict):
            return f"{contracts.FAILED_PREFIX} navigate_vln parameters must be an object"
        unknown = sorted(set(params) - {
            "instruction", "max_steps", "server_url", "trace_path", "max_wall_time_s",
            "max_normalized_speed",
        })
        if unknown:
            return f"{contracts.FAILED_PREFIX} navigate_vln unknown parameters: {', '.join(unknown)}"
        instruction = params.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            return f"{contracts.FAILED_PREFIX} navigate_vln requires a navigation instruction"
        start_steps = self._gateway_action_steps
        try:
            max_steps = contracts.validate_step(params.get("max_steps"), max_steps=self._remaining_steps())
            max_wall_time_s = float(params.get("max_wall_time_s", 1800))
            max_normalized_speed = float(params.get("max_normalized_speed", 0.6))
            server_url = str(params.get("server_url", "ws://127.0.0.1:8051"))
            trace_path = Path(params.get("trace_path", self.workspace / "lightnav_trace.jsonl")).expanduser().resolve()
            if not trace_path.is_relative_to(self.workspace):
                raise ValueError("navigation trace must be inside the episode workspace")
            if not 0 < max_wall_time_s <= self._execution_timeout_s:
                raise ValueError("max_wall_time_s exceeds the driver's execution timeout")
            if not 0 < max_normalized_speed <= 1:
                raise ValueError("max_normalized_speed must be in (0, 1]")
            problem = self._readiness_problem()
            if problem:
                raise RuntimeError(problem)
            last_observation_count = 0

            def frame_source():
                nonlocal last_observation_count
                if not self._gateway.wait_for_observation_after(
                    last_observation_count,
                    timeout_s=self._state_sync_timeout_s,
                    cancel_check=cancel_check,
                ):
                    raise TimeoutError("no fresh head-camera observation after navigation action")
                manifest = self._gateway.publish_observation()
                if manifest is None:
                    observation = self._gateway.snapshot().get("observation") or {}
                    detail = observation.get("last_error") or observation.get("last_warning")
                    raise RuntimeError(
                        "could not publish the current head-camera observation"
                        + (f": {detail}" if detail else "")
                    )
                frame = load_head_frame(manifest)
                last_observation_count = self._gateway.observation_count
                return frame

            def execute_waypoint(waypoint: tuple[float, float, float], requested: int) -> dict[str, Any]:
                hold = self._gateway.current_hold_action()
                action, action_info = waypoint_action(
                    waypoint, hold, action_steps=requested,
                    max_normalized_speed=max_normalized_speed,
                )
                execution_id = self._gateway.begin_local_execution(
                    "navigate_vln", requested, action,
                    baseline_evaluator_steps=None,
                )
                outcome = self._gateway.wait_for_execution(
                    execution_id, cancel_check=cancel_check,
                    progress_timeout_s=self._progress_timeout_s,
                    total_timeout_s=self._execution_timeout_s,
                )
                completed = int(outcome.get("completed_steps") or 0)
                self._gateway_action_steps += completed
                self._gateway_execution_count += 1
                return {"status": outcome.get("status"), "completed_steps": completed,
                        "action": action_info}

            initial_frame = frame_source()
            continued = self._navigation_client is not None
            if (self._navigation_episode_id != initial_frame.episode_id
                    or self._navigation_server_url != server_url):
                self._close_navigation_client()
                continued = False
            if self._navigation_client is None:
                self._navigation_client = LightNavWsClient(
                    server_url, client_id=f"emerge-behavior-active-{self._expected_task or 'task'}",
                    request_timeout_s=30.0,
                )
                self._navigation_client.connect()
                self._navigation_episode_id = initial_frame.episode_id
                self._navigation_server_url = server_url
            first = [initial_frame]

            def next_frame():
                return first.pop() if first else frame_source()

            result = run_closed_loop(
                instruction=instruction.strip(), max_steps=max_steps,
                frame_source=next_frame, execute_waypoint=execute_waypoint,
                steps_per_frame=frame_action_steps, client=self._navigation_client,
                trace_path=trace_path, max_wall_time_s=max_wall_time_s,
                append_trace=True, call_id=len(self._navigation_results) + 1,
            )
            result.update({"history_continued": continued, "instruction": instruction.strip(),
                           "remaining_action_steps": self._remaining_steps(),
                           "observation_manifest": str(self.workspace / "artifacts/observations/observation.json")})
            self._last_navigation_result = result
            self._navigation_results.append(result)
            if result.get("error") or result.get("finish_reason") not in {"step_budget", "model_stop_candidate"}:
                self._close_navigation_client()
                return f"{contracts.FAILED_PREFIX} Navigation " + json.dumps(result, ensure_ascii=False)
            return "Navigation segment finished: " + json.dumps(result, ensure_ascii=False)
        except (contracts.ActionValidationError, RuntimeError, ValueError, TimeoutError, OSError) as exc:
            self._last_navigation_result = {
                "finish_reason": "error", "error": str(exc),
                "instruction": instruction.strip(),
                "call_id": len(self._navigation_results) + 1,
                "nav_steps": self._gateway_action_steps - start_steps,
            }
            self._navigation_results.append(self._last_navigation_result)
            self._close_navigation_client()
            return f"{contracts.FAILED_PREFIX} navigate_vln: {exc}"

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

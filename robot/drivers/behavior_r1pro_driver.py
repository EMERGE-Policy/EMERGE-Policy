"""Native EMERGE driver for the BEHAVIOR R1Pro evaluation bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from robot.behavior import contracts
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
    OFFICIAL_PROMPT = contracts.OFFICIAL_PROMPT

    def __init__(
        self,
        gui: bool = False,
        workspace: str | Path = "~/.Emerge/workspace",
        upstream_url: str = "ws://127.0.0.1:8001",
        listen_host: str = "127.0.0.1",
        listen_port: int = 8000,
        upstream_open_timeout_s: float = 10.0,
        required_instruction: str = OFFICIAL_PROMPT,
        max_action_steps: int = 4000,
        status_path: str | Path | None = None,
        observation_publish_interval: int = 5,
        profile_path: str | Path | None = None,
        execution_timeout_s: float = 1800.0,
        progress_timeout_s: float = 300.0,
        state_sync_timeout_s: float = 120.0,
        expected_task: str = contracts.TASK_NAME,
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
        self._gateway = B1KProxyGateway(
            upstream_url=config["upstream_url"],
            listen_host=config["listen_host"],
            listen_port=config["listen_port"],
            upstream_open_timeout_s=config["upstream_open_timeout_s"],
            controlled=True,
            observation_workspace=self.workspace,
            observation_publish_interval=config["observation_publish_interval"],
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
        if action_type != "vla_execute":
            return f"Unknown action type: {action_type!r}."

        try:
            instruction, step = contracts.validate_action(
                params,
                required_instruction=self.required_instruction,
                max_steps=self.max_action_steps,
            )
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
            execution_id = self._gateway.begin_execution(
                instruction,
                step,
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
        status = str(result.get("status") or ERROR)
        detail = str(result.get("detail") or "")

        if status not in TERMINAL_EXECUTION_STATES:
            self._last_verdict = "unexpected_status"
            return f"{contracts.FAILED_PREFIX} unexpected VLA execution status {status!r}"
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
                f"{contracts.FAILED_PREFIX} VLA execution did not complete "
                f"({status}) after {completed} step(s): {detail}"
            )

        synced, applied_steps = self._synchronize(result, cancel_check=cancel_check)
        self._last_sync_ok = synced
        self._last_applied_steps = applied_steps
        self._last_verdict = status

        if not synced:
            return (
                f"{contracts.FAILED_PREFIX} {completed} policy action(s) were forwarded but the "
                "official evaluator state could not be confirmed afterwards; "
                "the simulator may not have applied them."
            )
        if status in _SUCCESS_STATES:
            applied = f", applied_steps={applied_steps}" if applied_steps is not None else ""
            return f"VLA execution finished: {status}, steps={completed}{applied}."
        return f"{contracts.FAILED_PREFIX} unexpected VLA execution status {status!r}"

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

        try:
            evaluation = self._read_evaluation()
        except contracts.EvaluationValidationError:
            return False, None
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

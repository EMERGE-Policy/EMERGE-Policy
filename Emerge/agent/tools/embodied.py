"""Embodied action tool for executing robot actions."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    from loguru import logger
except ImportError:  # pragma: no cover - fallback for lightweight test envs
    logger = logging.getLogger(__name__)

from Emerge.base import Tool
from Emerge.utils.action_queue import (
    action_timestamp,
    append_action,
    cancel_actions,
    empty_action_document,
    normalize_action_document,
    parse_action_markdown,
    pending_action_type,
    update_action_document,
)

if TYPE_CHECKING:
    from Emerge.agent.visual_monitor import VisualInterruptCoordinator


_BASE_RESULT_TIMEOUT_S = 60.0
_VLA_PER_STEP_TIMEOUT_S = 15.0
_WAM_INFERENCE_TIMEOUT_S = 180.0
_WAM_PER_STEP_TIMEOUT_S = 2.0


class EmbodiedActionTool(Tool):
    """Validate embodied actions and dispatch them through the workspace queue."""

    @property
    def name(self) -> str:
        return "execute_robot_action"

    @property
    def description(self) -> str:
        backend = os.environ.get("EMERGE_POLICY_BACKEND", "").strip().lower()
        if backend == "vla":
            policy_guidance = (
                "Use vla_execute for the current visually sensitive contact phase, "
                "with a phase-local instruction that omits future subgoals. "
            )
        elif backend == "wam":
            policy_guidance = (
                "Use wam_execute for the current visually sensitive contact phase. "
                "phase_instruction must describe exactly the next unfinished "
                "visual-contact phase, not the complete mission or a later phase. "
                "The evaluator preserves the full task separately and locks the "
                "configured conditioning mode. "
            )
        else:
            policy_guidance = (
                "Use the active model-policy backend for the current visually "
                "sensitive contact phase. VLA uses instruction; WAM keeps task and "
                "phase instructions separate. "
            )
        return (
            "Execute a physical action on the robot. "
            "Use explicit geometry-driven motion primitives for coarse approach and clear-space transport. "
            "Choose concrete poses, line segments, arc geometry, and gripper openings from the latest ROBOT_STATE.md or a successful object_location result. "
            f"{policy_guidance}"
            "Every tool call MUST include a non-empty `parameters` object. "
            "Do not call this tool with only `action_type` and `reasoning`."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        backend = os.environ.get("EMERGE_POLICY_BACKEND", "").strip().lower()
        if backend == "vla":
            policy_actions = (
                "- 'vla_execute': Execute the current phase's natural-language "
                "instruction using the VLA policy"
            )
            policy_examples = (
                "- vla_execute: {instruction: 'pick up the red block', step: 40}\n"
                "For vla_execute, provide one phase-local instruction and a positive "
                "step budget."
            )
        elif backend == "wam":
            policy_actions = (
                "- 'wam_execute': Execute a bounded Cosmos Policy WAM action chunk "
                "using the evaluator-locked conditioning"
            )
            policy_examples = (
                "- wam_execute grasp: {phase_instruction: 'grasp and lift the red block', step: 48}\n"
                "- wam_execute placement: {phase_instruction: 'place the held red block in the basket and release it', step: 60}\n"
                "For wam_execute, provide exactly one current phase_instruction and a "
                "positive step budget. The evaluator supplies task_instruction."
            )
        else:
            policy_actions = (
                "- 'vla_execute': Execute a phase-local instruction with VLA\n"
                "- 'wam_execute': Execute a bounded Cosmos Policy WAM action chunk"
            )
            policy_examples = (
                "- vla_execute: {instruction: 'pick up the red block', step: 40}\n"
                "- wam_execute: {phase_instruction: 'grasp and lift the red block', step: 48}"
            )
        return {
            "type": "object",
            "properties": {
                "action_type": {
                    "type": "string",
                    "description": (
                        "The type of action to execute. Supported actions:\n"
                        "- 'move_to_pose': Move end-effector to target position and orientation\n"
                        "- 'move_linear': Move end-effector linearly (straight line) to target or by delta\n"
                        "- 'set_gripper': Set gripper opening (open/close command or specific width in meters)\n"
                        "- 'follow_arc': Move end-effector along an arc (circular motion)\n"
                        f"{policy_actions}"
                    ),
                },
                "parameters": {
                    "type": "object",
                    "description": (
                        "The parameters for the action. "
                        "Vector parameters prefer list form like [x,y,z], but keyed objects like {x,y,z} are accepted. "
                        "Simulation end-effector targets use metres; Panda reach is approximately 0.85 m from the current robot base pose in ROBOT_STATE.md, and negative x/y values can be valid. "
                        "Examples:\n"
                        "- move_to_pose: {position_m: [x,y,z], orientation_euler: [r,p,y]} or {orientation_quat: [x,y,z,w]}\n"
                        "- move_linear: {position_m: [x,y,z]} or {delta_m: [dx,dy,dz]}, optionally with orientation\n"
                        "- set_gripper: {opening_m: 0.08} or {command: 'open'} or {command: 'close'}\n"
                        "- follow_arc: {center: [x,y,z], axis: [x,y,z], radius_m: 0.1, angle_deg: 90}\n"
                        f"{policy_examples}\n"
                        "For move_to_pose, ALWAYS provide position_m and either orientation_euler or orientation_quat. "
                        "For move_linear, ALWAYS provide either position_m or delta_m. "
                        "For follow_arc, ALWAYS provide center, axis, radius_m, and one of angle_deg or angle_rad. "
                        "A model-policy action completes when its step budget is used or "
                        "the goal is reached earlier. Re-read ROBOT_STATE.md before "
                        "choosing the next action."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": "The reasoning behind choosing this action.",
                },
            },
            "required": ["action_type", "parameters", "reasoning"],
        }

    def __init__(
        self,
        workspace: Path,
        visual_interrupts: VisualInterruptCoordinator | None = None,
    ):
        self.workspace = workspace
        self.visual_interrupts = visual_interrupts
        self.active_action_ids: set[str] = set()
        self.on_event = None

    async def execute(
        self,
        action_type: str,
        parameters: dict[str, Any],
        reasoning: str,
    ) -> str:
        """Validate and enqueue an action in the active workspace."""
        embodied_file = self.workspace / "EMBODIED.md"
        action_file = self.workspace / "ACTION.md"

        if not embodied_file.exists():
            return f"Error: {embodied_file.name} not found for the target robot. Cannot dispatch action."

        backend = os.environ.get("EMERGE_POLICY_BACKEND", "").strip().lower()
        requested_backend = {
            "vla_execute": "vla",
            "wam_execute": "wam",
        }.get(action_type)
        if backend in {"vla", "wam"} and requested_backend not in {None, backend}:
            return f"Error: {action_type} is disabled by EMERGE_POLICY_BACKEND={backend}"
        parameters = self._effective_parameters(action_type, parameters)
        logger.info("Dispatching action: {} {}", action_type, parameters)
        accepted = self._accept_action(action_type, parameters, action_file)
        if isinstance(accepted, str):
            return accepted
        dispatch_message, action_id = accepted
        self.active_action_ids.add(action_id)
        if self.on_event:
            self.on_event("action.updated", {"action_id": action_id, "action_type": action_type, "status": "pending"})
        return await self._wait_for_action_result(
            action_file,
            action_id=action_id,
            dispatch_message=dispatch_message,
            timeout_s=self._result_timeout(action_type, parameters),
        )

    @staticmethod
    def _effective_parameters(
        action_type: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Record evaluator-locked WAM conditioning in the action queue."""
        effective = dict(parameters)
        if action_type != "wam_execute":
            return effective
        mode = os.environ.get("EMERGE_WAM_CONDITIONING_MODE", "").strip()
        task = os.environ.get("EMERGE_WAM_TASK_INSTRUCTION", "").strip()
        if mode not in {"task", "phase", "task_with_phase"}:
            return effective
        effective["conditioning_mode"] = mode
        if task:
            effective["task_instruction"] = task
        if mode == "task":
            effective.pop("instruction", None)
            effective.pop("prompt", None)
            effective.pop("phase_instruction", None)
            if task:
                effective["conditioning_instruction"] = task
        else:
            phase = str(effective.get("phase_instruction", "")).strip()
            if phase:
                effective["conditioning_instruction"] = phase
        return effective

    @staticmethod
    def _result_timeout(action_type: str, parameters: dict[str, Any]) -> float:
        """Backstop wait budget, scaled by the action's step count."""
        if action_type == "vla_execute":
            try:
                step = max(0, int(parameters.get("step", 0)))
            except (TypeError, ValueError):
                step = 0
            return _BASE_RESULT_TIMEOUT_S + step * _VLA_PER_STEP_TIMEOUT_S
        if action_type == "wam_execute":
            try:
                step = max(0, int(parameters.get("step", 0)))
            except (TypeError, ValueError):
                step = 0
            query_count = max(1, (step + 15) // 16)
            return (
                _BASE_RESULT_TIMEOUT_S
                + query_count * _WAM_INFERENCE_TIMEOUT_S
                + step * _WAM_PER_STEP_TIMEOUT_S
            )
        return _BASE_RESULT_TIMEOUT_S

    @staticmethod
    def _accept_action(action_type: str, parameters: dict[str, Any], action_file: Path) -> str | tuple[str, str]:
        """Write validated action to ACTION.md."""
        def append(document):
            existing = pending_action_type(document)
            if existing:
                raise ValueError(f"Action '{existing}' is still pending/running; wait before dispatching")
            return append_action(document, action_type=action_type, parameters=parameters)
        try:
            document = update_action_document(action_file, append)
        except ValueError as exc:
            return f"Error: {exc}"
        action_id = str(document["actions"][-1]["id"])
        return f"Action '{action_type}' validated and dispatched to hardware.", action_id

    async def _wait_for_action_result(
        self,
        action_file: Path,
        *,
        action_id: str,
        dispatch_message: str,
        timeout_s: float = _BASE_RESULT_TIMEOUT_S,
        poll_interval_s: float = 0.5,
    ) -> str:
        result_task = asyncio.create_task(
            self._poll_action_result(
                action_file,
                action_id=action_id,
                dispatch_message=dispatch_message,
                timeout_s=timeout_s,
                poll_interval_s=poll_interval_s,
            )
        )
        if self.visual_interrupts is None:
            return await result_task

        interrupt_task = asyncio.create_task(self.visual_interrupts.wait_for(action_id))
        try:
            done, _ = await asyncio.wait(
                (result_task, interrupt_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if result_task in done:
                return result_task.result()

            signal = interrupt_task.result()
            self._request_action_cancel(
                action_file,
                action_id,
                reason=f"visual monitor completed {signal.step_id}",
            )
            result = await result_task
            evidence = json.dumps(signal.evidence, ensure_ascii=False)
            return (
                f"{result} The visual monitor already verified the current subgoal "
                "as achieved. Do not call task_verification again; update PLAN.md "
                f"and continue. Visual monitor evidence: {evidence}"
            )
        finally:
            for task in (result_task, interrupt_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(result_task, interrupt_task, return_exceptions=True)

    async def _poll_action_result(
        self,
        action_file: Path,
        *,
        action_id: str,
        dispatch_message: str,
        timeout_s: float,
        poll_interval_s: float,
    ) -> str:
        deadline = time.monotonic() + timeout_s
        last_status = "pending"
        while time.monotonic() < deadline:
            await asyncio.sleep(poll_interval_s)
            document = self._load_action_document(action_file)
            if document is None:
                continue
            for action in document.get("actions", []):
                if str(action.get("id")) != action_id:
                    continue
                status = str(action.get("status") or "pending").strip().lower()
                if self.on_event and status != last_status:
                    self.on_event("action.updated", {"action_id": action_id, **action})
                last_status = status
                if status in {"pending", "running"}:
                    break
                result = str(action.get("result", "")).strip()
                if status == "failed":
                    return f"Error: Robot action failed. {result}"
                if result:
                    return f"{dispatch_message} Execution {status}. Result: {result}"
                return f"{dispatch_message} Execution {status}."
        return (
            f"Error: {dispatch_message} Still pending after {timeout_s:.0f}s; the action "
            "remains queued in ACTION.md and the watchdog may be stalled. Do not "
            "re-dispatch or sleep; inspect the watchdog process."
        )

    @staticmethod
    def _request_action_cancel(
        action_file: Path,
        action_id: str,
        *,
        reason: str,
    ) -> None:
        def mark(document):
            for action in document.get("actions", []):
                if str(action.get("id")) == action_id and action.get("status") in {"pending", "running"}:
                    action["cancel_requested"] = True
                    action["cancel_reason"] = reason
                    action["cancel_requested_at"] = action_timestamp()
        update_action_document(action_file, mark)

    async def cancel_active(self, reason: str, timeout: float) -> dict:
        return await cancel_actions(
            self.workspace / "ACTION.md", reason, timeout, sorted(self.active_action_ids),
        )

    @staticmethod
    def _load_action_document(action_file: Path) -> dict[str, Any] | None:
        if not action_file.exists():
            return empty_action_document()
        content = action_file.read_text(encoding="utf-8").strip()
        if not content:
            return empty_action_document()
        payload = parse_action_markdown(content)
        if payload is None:
            return None
        return normalize_action_document(payload)

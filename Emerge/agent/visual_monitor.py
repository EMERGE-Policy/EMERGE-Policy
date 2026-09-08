"""Always-on visual monitor running beside the main agent."""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from Emerge.agent.plan_state import read_current_subgoal
from Emerge.base import ToolRegistry
from Emerge.providers.base import LLMProvider
from Emerge.subagents import ImageContent, SubagentTask, TextContent
from Emerge.subagents.context import SubagentContextBuilder
from Emerge.subagents.skills import SkillRegistry
from Emerge.subagents.task_verification.tools import SubmitVerificationTool
from Emerge.utils.action_queue import (
    first_active_action,
    normalize_action_document,
    parse_action_markdown,
)


_VISUAL_MONITOR_PROMPT = """
You are a fast visual interrupt monitor for an embodied agent. The task includes
one current reference-camera image. Decide whether the stated plan subgoal and
its full done criterion are visibly achieved, then call
`submit_task_verification` exactly once. Use only visible evidence and mark a
condition uncertain when this image cannot prove it. Do not plan robot motion
or call any other tool.
""".strip()


@dataclass(frozen=True, slots=True)
class VisualInterruptSignal:
    plan_revision: str
    step_id: str
    action_id: str
    observation_revision: int
    evidence: dict[str, Any]


class VisualInterruptCoordinator:
    """Deliver monitor-thread signals to the main asyncio loop."""

    def __init__(self) -> None:
        self._queue: queue.Queue[VisualInterruptSignal] = queue.Queue()

    def bind(self) -> None:
        self._queue = queue.Queue()

    def publish_from_thread(self, signal: VisualInterruptSignal) -> None:
        self._queue.put_nowait(signal)

    async def wait_for(self, action_id: str) -> VisualInterruptSignal:
        while True:
            try:
                signal = self._queue.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            if signal.action_id == action_id:
                return signal


class VisualMonitor:
    """Read PLAN.md and new camera snapshots until the subgoal is achieved."""

    def __init__(
        self,
        *,
        workspace: Path,
        provider: LLMProvider,
        model: str,
        poll_interval_seconds: float,
        verification_timeout_seconds: float,
        confirmations: int,
    ) -> None:
        self.workspace = workspace
        self.provider = provider
        self.model = model
        self.poll_interval_seconds = poll_interval_seconds
        self.verification_timeout_seconds = verification_timeout_seconds
        self.confirmations = confirmations
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._verification_future: concurrent.futures.Future | None = None
        self._verification_requests: queue.Queue[
            tuple[SubagentTask, concurrent.futures.Future]
        ] = queue.Queue()
        self._verification_task: asyncio.Task | None = None
        self._submission_tool = SubmitVerificationTool()
        self._tools = ToolRegistry()
        self._tools.register(self._submission_tool)
        self._context_builder = SubagentContextBuilder(
            agent_name="visual_monitor",
            system_prompt=_VISUAL_MONITOR_PROMPT,
            skills=SkillRegistry(),
        )
        self._publish: Callable[[VisualInterruptSignal], None] | None = None
        self._last_input: tuple[str, str, int] | None = None
        self._candidate: tuple[str, str] | None = None
        self._candidate_count = 0
        self._sent: tuple[str, str] | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        publish: Callable[[VisualInterruptSignal], None],
        main_loop: asyncio.AbstractEventLoop,
    ) -> None:
        if self.is_running:
            return
        self._publish = publish
        self._main_loop = main_loop
        self._verification_requests = queue.Queue()
        self._verification_task = main_loop.create_task(self._verification_loop())
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="visual-monitor",
            daemon=True,
        )
        self._thread.start()
        logger.info("Visual monitor thread started")

    def stop(self) -> None:
        self._stop.set()
        if self._verification_future is not None:
            self._verification_future.cancel()
        if self._verification_task is not None:
            self._verification_task.cancel()
        if self._thread is not None:
            self._thread.join()
        self._thread = None
        self._main_loop = None
        self._verification_future = None
        self._verification_task = None
        logger.info("Visual monitor thread stopped")

    async def _verification_loop(self) -> None:
        while not self._stop.is_set():
            try:
                task, future = self._verification_requests.get_nowait()
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            if future.cancelled():
                continue
            try:
                result = await asyncio.wait_for(
                    self._verify(task),
                    timeout=task.timeout,
                )
            except asyncio.CancelledError:
                future.cancel()
                raise
            except Exception as exc:
                future.set_exception(exc)
            else:
                if not future.done():
                    future.set_result(result)

    async def _verify(self, task: SubagentTask) -> dict[str, Any] | None:
        self._submission_tool.reset()
        context = self._context_builder.build(task)
        started = time.perf_counter()
        response = await self.provider.chat_with_retry(
            messages=context.messages,
            tools=self._tools.get_definitions(),
            model=self.model,
            tool_choice="required",
        )
        logger.info(
            "Visual monitor LLM timing | model={} elapsed={:.3f}s",
            self.model,
            time.perf_counter() - started,
        )
        if response.finish_reason == "error":
            return None
        for call in response.tool_calls:
            if call.name == self._submission_tool.name:
                await self._tools.execute(call.name, call.arguments)
                break
        return self._submission_tool.last_result

    def _thread_main(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except concurrent.futures.CancelledError:
                if not self._stop.is_set():
                    logger.exception("Visual monitor verification was cancelled")
            except Exception:
                logger.exception("Visual monitor poll failed")
            self._stop.wait(self.poll_interval_seconds)

    def _poll_once(self) -> None:
        plan_file = self.workspace / "PLAN.md"
        action_file = self.workspace / "ACTION.md"
        observation_file = self.workspace / "artifacts/observations/observation.json"
        if not plan_file.exists() or not action_file.exists() or not observation_file.exists():
            return

        current = read_current_subgoal(plan_file)
        action = self._read_pending_action(action_file)
        if current is None or action is None:
            return

        observation = json.loads(observation_file.read_text(encoding="utf-8"))
        observation_revision = int(
            observation.get("revision", observation_file.stat().st_mtime_ns)
        )
        action_id = str(action["id"])
        input_key = (current.plan_revision, action_id, observation_revision)
        if input_key == self._last_input:
            return
        self._last_input = input_key

        reference_name = str(observation["reference_view"])
        reference = next(
            view
            for view in observation["views"]
            if str(view["name"]) == reference_name
        )
        image_path = self.workspace / str(reference["image_path"])
        encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")

        task = SubagentTask(
            content=(
                TextContent(
                    "Verify whether the current plan subgoal is already achieved.\n"
                    f"Subgoal: {current.subgoal}\n"
                    f"Done criterion: {current.done_criterion}"
                ),
                ImageContent(
                    f"data:image/png;base64,{encoded_image}",
                    detail="high",
                ),
            ),
            timeout=self.verification_timeout_seconds,
        )
        if self._main_loop is None:
            return
        self._verification_future = concurrent.futures.Future()
        self._verification_requests.put_nowait((task, self._verification_future))
        try:
            result = self._verification_future.result()
        finally:
            self._verification_future = None
        output = result if isinstance(result, dict) else {}
        achieved = output.get("outcome") == "achieved"
        candidate = (current.plan_revision, action_id)
        if not achieved:
            self._candidate = None
            self._candidate_count = 0
            return

        if self._candidate == candidate:
            self._candidate_count += 1
        else:
            self._candidate = candidate
            self._candidate_count = 1

        if self._candidate_count < self.confirmations or self._sent == candidate:
            return
        self._sent = candidate
        if self._publish is not None and not self._stop.is_set():
            self._publish(
                VisualInterruptSignal(
                    plan_revision=current.plan_revision,
                    step_id=current.step_id,
                    action_id=action_id,
                    observation_revision=observation_revision,
                    evidence=output,
                )
            )

    @staticmethod
    def _read_pending_action(action_file: Path) -> dict[str, Any] | None:
        payload = parse_action_markdown(action_file.read_text(encoding="utf-8"))
        if payload is None:
            return None
        document = normalize_action_document(payload)
        if document is None:
            return None
        pending = first_active_action(document)
        return pending[1] if pending is not None else None

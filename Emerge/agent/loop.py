"""Agent loop: the core processing engine."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable
from uuid import uuid4

from loguru import logger

from Emerge.agent.context import ContextBuilder
from Emerge.agent.memory import MemoryConsolidator
from Emerge.agent.tools.delegate import DelegateSubagentTool
from Emerge.agent.tools.embodied import EmbodiedActionTool
from Emerge.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from Emerge.agent.tools.message import MessageTool
from Emerge.agent.tools.scene_graph import SceneGraphQueryTool
from Emerge.agent.tools.shell import ExecTool
from Emerge.agent.tools.update_plan import UpdatePlanTool
from Emerge.agent.visual_monitor import (
    VisualInterruptCoordinator,
    VisualMonitor,
)
from Emerge.base import ToolRegistry
from Emerge.bus.events import InboundMessage, OutboundMessage
from Emerge.bus.queue import MessageBus
from Emerge.providers.base import LLMProvider
from Emerge.session.manager import Session, SessionManager
from Emerge.subagents import SubagentRegistry
from Emerge.subagents.object_location import (
    build_object_location_subagent,
)
from Emerge.subagents.task_verification import (
    build_task_verification_subagent,
)

if TYPE_CHECKING:
    from Emerge.config.schema import (
        ExecToolConfig,
        ObjectLocationSubagentConfig,
        TaskVerificationSubagentConfig,
        VisualMonitorConfig,
    )


class AgentLoop:
    """
    The agent loop is the core processing engine.

    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _TOOL_RESULT_MAX_CHARS = 16_000

    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        context_window_tokens: int = 65_536,
        exec_config: ExecToolConfig | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        object_location_subagent_config: (
            ObjectLocationSubagentConfig | None
        ) = None,
        task_verification_subagent_config: (
            TaskVerificationSubagentConfig | None
        ) = None,
        visual_monitor_config: VisualMonitorConfig | None = None,
        model_services_config=None,
    ):
        from Emerge.config.schema import (
            ExecToolConfig,
            ModelServicesConfig,
            ObjectLocationSubagentConfig,
            TaskVerificationSubagentConfig,
            VisualMonitorConfig,
        )
        from external_model_server.model_service.discovery import ServiceDiscovery
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = context_window_tokens
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagent_registry = SubagentRegistry()
        object_location_config = (
            object_location_subagent_config
            or ObjectLocationSubagentConfig()
        )
        self.subagent_registry.register(
            build_object_location_subagent(
                provider=provider,
                workspace=workspace,
                model=object_location_config.model or self.model,
                config=object_location_config.model_dump(),
                discovery=ServiceDiscovery(
                    (model_services_config or ModelServicesConfig()).discovery_config()
                ),
            )
        )
        task_verification_config = (
            task_verification_subagent_config
            or TaskVerificationSubagentConfig()
        )
        self.subagent_registry.register(
            build_task_verification_subagent(
                provider=provider,
                workspace=workspace,
                model=task_verification_config.model or self.model,
                config=task_verification_config.model_dump(),
            )
        )
        monitor_config = visual_monitor_config or VisualMonitorConfig()
        self.visual_interrupts = VisualInterruptCoordinator()
        self.visual_monitor = None
        if monitor_config.enabled:
            self.visual_monitor = VisualMonitor(
                workspace=workspace,
                provider=provider,
                model=task_verification_config.model or self.model,
                poll_interval_seconds=monitor_config.poll_interval_seconds,
                verification_timeout_seconds=(
                    monitor_config.verification_timeout_seconds
                ),
                confirmations=monitor_config.confirmations,
            )

        self.on_event = None
        self.stream_output = False
        self.run_id = None
        self.last_run_info = {"iterations": 0, "usage": {}, "finish_reason": "stop"}
        self._running = False
        self._active_tasks: dict[str, list[asyncio.Task]] = {}  # session_key -> tasks
        self._processing_lock = asyncio.Lock()
        self.memory_consolidator = MemoryConsolidator(
            workspace=workspace,
            provider=provider,
            model=self.model,
            sessions=self.sessions,
            context_window_tokens=context_window_tokens,
            build_messages=self.context.build_messages,
            get_tool_definitions=self.tools.get_definitions,
        )
        self._register_default_tools()
        # Load env variables
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path=self.workspace / ".env")
        except Exception:
            logger.warning("Failed to load .env file, ignore using env variables")

    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        protected_paths = {self.workspace / "PLAN.md"}
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(
                cls(
                    workspace=self.workspace,
                    allowed_dir=allowed_dir,
                    protected_paths=protected_paths,
                )
            )
        self.tools.register(UpdatePlanTool(workspace=self.workspace))
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        self.tools.register(MessageTool())
        self.tools.register(DelegateSubagentTool(self.subagent_registry))

        action_tool = EmbodiedActionTool(
            workspace=self.workspace,
            visual_interrupts=(
                self.visual_interrupts if self.visual_monitor is not None else None
            ),
        )
        self.tools.register(action_tool)
        self.tools.register(SceneGraphQueryTool(workspace=self.workspace))

    def _emit(self, kind: str, data: dict) -> None:
        if self.on_event is not None:
            self.on_event(kind, data)

    async def cancel_actions(self, reason: str, timeout: float) -> dict:
        tool = self.tools.get("execute_robot_action")
        return await tool.cancel_active(reason, timeout) if tool else {"acknowledged": True, "action_ids": []}

    async def aclose(self) -> None:
        self.stop()
        self._stop_visual_monitor()
        await self.provider.aclose()

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """Remove <think>…</think> blocks that some models embed in content."""
        if not text:
            return None
        return re.sub(r"<think>[\s\S]*?</think>", "", text).strip() or None

    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """Format tool calls as concise hint, e.g. 'web_search("query")'."""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls if tc.name != "message")

    async def _run_agent_loop(
        self,
        initial_messages: list[dict],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict]]:
        """Run the agent iteration loop."""
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        while iteration < self.max_iterations:
            iteration += 1

            tool_defs = self.tools.get_definitions()

            self.last_run_info["iterations"] = iteration
            llm_started = time.perf_counter()
            self._emit("llm.started", {"iteration": iteration, "model": self.model})
            async def delta(text):
                self._emit("assistant.delta", {"message_id": f"{self.run_id}:{iteration}", "text": text})
            kwargs = dict(messages=messages, tools=tool_defs, model=self.model)
            if self.stream_output:
                response = await self.provider.chat_with_stream(on_delta=delta, **kwargs)
            else:
                response = await self.provider.chat_with_retry(**kwargs)
            self.last_run_info["finish_reason"] = response.finish_reason
            for key, value in response.usage.items():
                self.last_run_info["usage"][key] = self.last_run_info["usage"].get(key, 0) + value
            self._emit("llm.completed", {"iteration": iteration, "usage": response.usage,
                                       "duration_ms": round((time.perf_counter() - llm_started) * 1000),
                                       "finish_reason": response.finish_reason})
            if response.finish_reason != "error":
                self._emit("assistant.message", {"message_id": f"{self.run_id}:{iteration}",
                                                 "text": self._strip_think(response.content) or "",
                                                 "final": not response.has_tool_calls})
            logger.info(
                "Main LLM timing | iteration={} model={} elapsed={:.3f}s",
                iteration,
                self.model,
                time.perf_counter() - llm_started,
            )

            if response.has_tool_calls:
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    tool_hint = self._tool_hint(response.tool_calls)
                    if tool_hint:
                        await on_progress(tool_hint, tool_hint=True)

                tool_call_dicts = [
                    tc.to_openai_tool_call()
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    started = time.perf_counter()
                    self._emit("tool.started", {"tool_call_id": tool_call.id, "tool": tool_call.name,
                                               "arguments": tool_call.arguments})
                    action_tool = self.tools.get("execute_robot_action")
                    if tool_call.name == "execute_robot_action" and action_tool:
                        action_tool.on_event = self._emit
                    try:
                        result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    except asyncio.CancelledError:
                        self._emit("tool.cancelled", {"tool_call_id": tool_call.id, "tool": tool_call.name})
                        raise
                    failed = isinstance(result, str) and result.startswith("Error")
                    self._emit("tool.failed" if failed else "tool.completed",
                               {"tool_call_id": tool_call.id, "tool": tool_call.name,
                                "duration_ms": round((time.perf_counter() - started) * 1000),
                                "result": str(result)[:16000]})
                    if tool_call.name == "update_plan" and not failed:
                        from Emerge.runtime.snapshots import plan_snapshot
                        self._emit("plan.updated", plan_snapshot(self.workspace))
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                clean = self._strip_think(response.content)
                # Don't persist error responses to session history — they can
                # poison the context and cause permanent 400 loops (#1303).
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        if final_content is None and iteration >= self.max_iterations:
            self.last_run_info["finish_reason"] = "max_iterations"
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    async def run(self) -> None:
        """Run the agent loop, dispatching messages as tasks to stay responsive to /stop."""
        self._running = True
        self._start_visual_monitor()
        logger.info("Agent loop started")
        try:
            while self._running:
                try:
                    msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                cmd = msg.content.strip().lower()
                if cmd == "/stop":
                    await self._handle_stop(msg)
                elif cmd == "/restart":
                    await self._handle_restart(msg)
                else:
                    task = asyncio.create_task(self._dispatch(msg))
                    self._active_tasks.setdefault(msg.session_key, []).append(task)
                    task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)
        finally:
            self._stop_visual_monitor()

    def _start_visual_monitor(self) -> bool:
        if self.visual_monitor is None or self.visual_monitor.is_running:
            return False
        self.visual_interrupts.bind()
        self.visual_monitor.start(
            self.visual_interrupts.publish_from_thread,
            asyncio.get_running_loop(),
        )
        return True

    def _stop_visual_monitor(self) -> None:
        if self.visual_monitor is not None and self.visual_monitor.is_running:
            self.visual_monitor.stop()

    async def _handle_stop(self, msg: InboundMessage) -> None:
        """Cancel all active tasks for the session."""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        content = f"Stopped {cancelled} task(s)." if cancelled else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    async def _handle_restart(self, msg: InboundMessage) -> None:
        """Restart the process in-place via os.execv."""
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content="Restarting...",
        ))

        async def _do_restart():
            await asyncio.sleep(1)
            os.execv(sys.executable, [sys.executable] + sys.argv)

        asyncio.create_task(_do_restart())

    async def _dispatch(self, msg: InboundMessage) -> None:
        """Process a message under the global lock."""
        async with self._processing_lock:
            try:
                response = await self._process_message(msg)
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
        logger.info("Agent loop stopping")

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> OutboundMessage | None:
        """Process a single inbound message and return the response."""
        # System messages: parse origin from chat_id ("channel:chat_id")
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            history = session.get_history(max_messages=0)
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )

            async def _system_progress(content: str, *, tool_hint: bool = True) -> None:
                meta = dict(msg.metadata or {})
                meta["_progress"] = True
                meta["_tool_hint"] = tool_hint
                await self.bus.publish_outbound(OutboundMessage(
                    channel=channel, chat_id=chat_id, content=content, metadata=meta,
                ))

            message_tool = self.tools.get("message")
            if isinstance(message_tool, MessageTool):
                message_tool.set_progress_callback(_system_progress)
            try:
                final_content, _, all_msgs = await self._run_agent_loop(
                    messages, on_progress=_system_progress,
                )
            finally:
                if isinstance(message_tool, MessageTool):
                    message_tool.set_progress_callback(None)
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            await self.memory_consolidator.maybe_consolidate_by_tokens(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        history = session.get_history(max_messages=0)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        async def _bus_progress(content: str, *, tool_hint: bool = True) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        progress_callback = on_progress or _bus_progress
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_progress_callback(progress_callback)
        try:
            final_content, _, all_msgs = await self._run_agent_loop(
                initial_messages, on_progress=progress_callback,
            )
        except asyncio.CancelledError:
            # Preserve completed tool evidence and close unanswered calls without inventing results.
            answered = {m["tool_call_id"] for m in initial_messages if m["role"] == "tool"}
            unanswered = [call for m in initial_messages for call in m.get("tool_calls", [])
                          if call["id"] not in answered]
            for call in unanswered:
                self.context.add_tool_result(
                    initial_messages, call["id"], call["function"]["name"],
                    "Run interrupted before a result was recorded. Execution is unconfirmed; "
                    "inspect the environment before retrying.",
                )
            self._save_turn(session, initial_messages, 1 + len(history))
            session.add_message("assistant", "Run interrupted. Re-observe the environment before continuing.",
                                run_id=self.run_id, interrupted=True)
            self.sessions.save(session)
            raise
        finally:
            if isinstance(message_tool, MessageTool):
                message_tool.set_progress_callback(None)

        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)
        await self.memory_consolidator.maybe_consolidate_by_tokens(session)

        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """Save new-turn messages into session, truncating large tool results."""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue  # skip empty assistant messages — they poison session context
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    # Strip the runtime-context prefix, keep only the user text.
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
            entry.setdefault("timestamp", datetime.now().isoformat())
            entry.setdefault("message_id", uuid4().hex)
            if self.run_id:
                entry.setdefault("run_id", self.run_id)
            session.messages.append(entry)
        session.updated_at = datetime.now()

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> str:
        """Process a message directly (for CLI or programmatic usage)."""
        self.last_run_info = {"iterations": 0, "usage": {}, "finish_reason": "stop"}
        started_monitor = self._start_visual_monitor()
        async def report(content, *, tool_hint=False):
            self._emit("agent.progress", {"text": content, "tool_hint": tool_hint})
            if on_progress:
                await on_progress(content, tool_hint=tool_hint)
        try:
            msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
            response = await self._process_message(msg, session_key=session_key, on_progress=report)
            return response.content if response else ""
        finally:
            if started_monitor:
                self._stop_visual_monitor()

"""Single execution authority for human and machine clients."""
from __future__ import annotations

import asyncio
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Callable

from loguru import logger

from Emerge.runtime.configuration import load_runtime_config
from Emerge.runtime.protocol import RunError, RunEvent, RunRequest, RunResult, utc_now
from Emerge.runtime.storage import RunArtifacts, WorkspaceLease, redact


def build_agent(config):
    from Emerge.agent.loop import AgentLoop
    from Emerge.bus.queue import MessageBus
    from Emerge.providers.factory import create_provider
    from Emerge.utils.helpers import sync_workspace_templates

    sync_workspace_templates(config.workspace_path, silent=True)
    return AgentLoop(
        bus=MessageBus(), provider=create_provider(config), workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_iterations=config.agents.defaults.max_tool_iterations,
        context_window_tokens=config.agents.defaults.context_window_tokens,
        exec_config=config.tools.exec, restrict_to_workspace=config.tools.restrict_to_workspace,
        object_location_subagent_config=config.subagents.object_location,
        task_verification_subagent_config=config.subagents.task_verification,
        visual_monitor_config=config.visual_monitor,
        model_services_config=config.model_services,
    )


class AgentRuntime:
    """Owns the workspace until execution and physical-action cleanup finish.

    Event consumers are synchronous; they must not block or raise.
    Execution errors become RunResult. Artifact I/O errors propagate to the client.
    """

    def __init__(self, agent_factory=build_agent):
        self.agent_factory = agent_factory

    async def run(
        self, request: RunRequest, *, output_dir: Path | None = None,
        on_event: Callable[[RunEvent], None] | None = None,
        cancel: asyncio.Event | None = None,
    ) -> RunResult:
        cancel = cancel or asyncio.Event()
        started_at, started = utc_now(), time.monotonic()
        agent = None
        seq = 0
        status, reason, reply, error, cancellation = "failed", "runtime_error", "", None, None
        with ExitStack() as resources, logger.contextualize(run_id=request.run_id):
            artifacts = RunArtifacts(output_dir, request) if output_dir else None
            if artifacts:
                resources.callback(artifacts.close)
                sink = logger.add(
                    artifacts.directory / "runtime.log",
                    filter=lambda record: record["extra"].get("run_id") == request.run_id,
                    backtrace=False, diagnose=False,
                )
                resources.callback(logger.remove, sink)

            def emit(kind, data):
                nonlocal seq
                seq += 1
                event = RunEvent(run_id=request.run_id, session_id=request.session_id,
                                 seq=seq, type=kind, data=redact(data))
                if artifacts:
                    artifacts.event(event)
                if on_event:
                    on_event(event)

            try:
                emit("run.started", {"message": request.message, "state": "starting"})
                config = load_runtime_config(request)
                workspace = config.workspace_path.resolve()
                resources.enter_context(WorkspaceLease(workspace))
                if cancel.is_set():
                    status, reason = "cancelled", "cancel_requested"
                else:
                    agent = self.agent_factory(config)
                    agent.on_event = emit
                    agent.stream_output = request.stream
                    agent.run_id = request.run_id
                    emit("run.configured", {"model": agent.model, "workspace": str(workspace),
                                           "context_window_tokens": config.agents.defaults.context_window_tokens})
                    session = agent.sessions.get_or_create(request.session_id)
                    session.metadata.setdefault("title", request.message.strip().splitlines()[0][:80])
                    session.metadata.update(model=agent.model, last_run_id=request.run_id)
                    agent.sessions.save(session)
                    emit("run.state", {"state": "running"})
                    work = asyncio.create_task(agent.process_direct(request.message, request.session_id))
                    stopped = asyncio.create_task(cancel.wait())
                    try:
                        done, _ = await asyncio.wait(
                            (work, stopped), timeout=request.timeout_s,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        if work in done:
                            reply = work.result()
                            reason = agent.last_run_info["finish_reason"]
                            status = "failed" if reason in {"error", "max_iterations", "length", "content_filter"} else "completed"
                            if status == "failed":
                                error = RunError(code=reason, message=reply or reason)
                        else:
                            reason = "cancel_requested" if stopped in done else "timeout"
                            status = "cancelled" if stopped in done else "timed_out"
                            emit("run.state", {"state": "cancelling", "reason": reason})
                    finally:
                        work.cancel()
                        stopped.cancel()
                        await asyncio.gather(work, stopped, return_exceptions=True)
            except asyncio.CancelledError:
                status, reason = "cancelled", "cancel_requested"
            except Exception as exc:
                status, reason = "failed", "runtime_error"
                error = RunError(code=type(exc).__name__, message=str(exc))
                logger.exception("Run failed")
            finally:
                if agent:
                    try:
                        cancellation = await agent.cancel_actions(
                            "run_finished" if status == "completed" else reason, request.cancel_timeout_s,
                        )
                        if not cancellation["acknowledged"]:
                            status, reason = "failed", "cancellation_unconfirmed"
                            error = RunError(code=reason, message="Execution endpoint did not confirm stopping")
                        elif status == "completed" and cancellation.get("requested_action_ids"):
                            status, reason = "failed", "unfinished_action"
                            error = RunError(code=reason, message="Agent returned with unfinished actions; they were stopped")
                    except Exception as exc:
                        status, reason = "failed", "cancellation_unconfirmed"
                        error = RunError(code=reason, message=str(exc))
                    finally:
                        try:
                            await agent.aclose()
                        except Exception as exc:
                            status = "failed"
                            if reason != "cancellation_unconfirmed":
                                reason = "cleanup_error"
                                error = RunError(code=type(exc).__name__, message=str(exc))

            info = agent.last_run_info if agent else {}
            result = RunResult(
                run_id=request.run_id, session_id=request.session_id,
                run_status=status, finish_reason=reason, assistant_text=reply,
                model=agent.model if agent else request.model, iterations=info.get("iterations", 0),
                usage=info.get("usage", {}), started_at=started_at,
                duration_ms=round((time.monotonic() - started) * 1000),
                error=error, cancellation=cancellation,
            )
            emit("run.finished", result.model_dump(mode="json"))
            if artifacts:
                artifacts.finish(result)
            return result

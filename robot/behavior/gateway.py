"""Websocket gateway between BEHAVIOR and the OpenPI B1K server.

Observation and OpenPI action payloads are forwarded unchanged. An optional
observation publisher decodes a side copy for EMERGE's visual tools. Local
actions (including LightNav base commands) use the same execution gate.

The gateway is also the execution gate: in controlled mode it only lets a
bounded number of policy actions reach the evaluator, and it reports back what
actually happened.  Three details make that boundary trustworthy:

* an execution is owned by the evaluator connection that serves it, so a
  reconnect cannot be credited with another episode's replies,
* replies that were already in flight when an execution ended are drained
  before a new execution starts counting, and
* a disconnect is classified by which side closed, so an upstream crash is no
  longer reported as a finished episode.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import http
import json
import logging
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:
    from .observation import ObservationPublisher
except ImportError:  # Support direct execution of this module for diagnostics.
    from observation import ObservationPublisher

logger = logging.getLogger(__name__)

#: Execution lifecycle states.
EXECUTING = "executing"
STEP_COMPLETED = "step_completed"
INTERRUPTED = "interrupted"
EPISODE_FINISHED = "episode_finished"
EPISODE_RESET = "episode_reset"
ERROR = "error"
TIMEOUT = "timeout"
NO_PROGRESS = "no_progress"

TERMINAL_EXECUTION_STATES = frozenset(
    {STEP_COMPLETED, INTERRUPTED, EPISODE_FINISHED, EPISODE_RESET, ERROR, TIMEOUT, NO_PROGRESS}
)


@dataclass
class GatewayStats:
    status: str = "stopped"
    listen_host: str = ""
    listen_port: int = 0
    upstream_url: str = ""
    active_connections: int = 0
    total_connections: int = 0
    requests_forwarded: int = 0
    responses_forwarded: int = 0
    local_actions_generated: int = 0
    evaluator_resets_forwarded: int = 0
    upstream_cache_invalidations_sent: int = 0
    last_error: str | None = None
    started_at_monotonic: float | None = None


@dataclass
class ExecutionState:
    execution_id: int
    instruction: str
    requested_steps: int
    mode: str = "vla"
    local_action_count: int = 0
    completed_steps: int = 0
    status: str = EXECUTING
    detail: str | None = None
    #: Evaluator connection that owns this execution.  Adopted by the first
    #: connection that actually forwards a request for it.
    connection_serial: int | None = None
    #: Requests forwarded for this execution and replies recorded for it.  The
    #: difference is the number of replies still in flight when it retired.
    requests_sent: int = 0
    responses_recorded: int = 0
    #: Observation count at the moment the budget was exhausted.  The next
    #: frame that arrives after this value has been produced by an evaluator
    #: that already applied every action of this execution.
    sync_observation_baseline: int | None = None
    #: How many evaluator steps had been published when the execution started.
    baseline_evaluator_steps: int | None = None
    #: Replies still owed to this execution when it retired.  Diagnostic only;
    #: the gateway keeps the running total in ``_orphaned_responses``.
    pending_responses_at_retire: int = 0
    started_at_monotonic: float = field(default_factory=time.monotonic)


class B1KProxyGateway:
    """Relay the official BEHAVIOR websocket protocol with controlled action gating."""

    def __init__(
        self,
        *,
        upstream_url: str = "ws://127.0.0.1:8011",
        listen_host: str = "127.0.0.1",
        listen_port: int = 8000,
        upstream_open_timeout_s: float = 10.0,
        controlled: bool = False,
        observation_workspace: str | Path | None = None,
        observation_publish_interval: int = 5,
        camera_calibration: dict[str, Any] | None = None,
    ) -> None:
        self.upstream_url = str(upstream_url)
        self.listen_host = str(listen_host)
        self.listen_port = int(listen_port)
        self.upstream_open_timeout_s = float(upstream_open_timeout_s)
        self.controlled = bool(controlled)
        self._observation_publisher = (
            ObservationPublisher(
                observation_workspace,
                publish_interval=observation_publish_interval,
                camera_calibration=camera_calibration,
            )
            if observation_workspace is not None
            else None
        )
        self._stats = GatewayStats(
            listen_host=self.listen_host,
            listen_port=self.listen_port,
            upstream_url=self.upstream_url,
        )
        self._stats_lock = threading.Lock()
        self._execution_condition = threading.Condition()
        self._execution: ExecutionState | None = None
        self._local_action_sequence: list[list[float]] = []
        self._execution_serial = 0
        self._orphaned_responses = 0
        self._episode_id = 0
        self._connection_serial = 0
        self._active_connection_serial: int | None = None
        self._last_disconnect: str | None = None
        self._evaluator_connected = False
        self._upstream_connected = False
        self._upstream_needs_action_cache_invalidation = False
        self._upstream_supports_action_cache_invalidation = False
        self._stopped_cleanly = True
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._startup_error: BaseException | None = None

    # ------------------------------------------------------------------ #
    # Inspection
    # ------------------------------------------------------------------ #

    @property
    def observation_count(self) -> int:
        """Monotonic count of decoded observation frames, or 0 when disabled."""
        if self._observation_publisher is None:
            return 0
        return self._observation_publisher.observation_count

    def snapshot(self) -> dict[str, Any]:
        with self._stats_lock:
            result = asdict(self._stats)
        started = result.pop("started_at_monotonic")
        result["uptime_s"] = max(0.0, time.monotonic() - started) if started else 0.0
        with self._execution_condition:
            result.update(
                {
                    "controlled": self.controlled,
                    "episode_id": self._episode_id,
                    "connection_serial": self._connection_serial,
                    "active_connection_serial": self._active_connection_serial,
                    "last_disconnect": self._last_disconnect,
                    "orphaned_responses": self._orphaned_responses,
                    "evaluator_connected": self._evaluator_connected,
                    "upstream_connected": self._upstream_connected,
                    "upstream_needs_action_cache_invalidation": self._upstream_needs_action_cache_invalidation,
                    "upstream_supports_action_cache_invalidation": self._upstream_supports_action_cache_invalidation,
                    "stopped_cleanly": self._stopped_cleanly,
                    "execution": asdict(self._execution) if self._execution else None,
                }
            )
        if self._observation_publisher is not None:
            result["observation"] = self._observation_publisher.snapshot()
        return result

    def read_evaluation_steps(self) -> int | None:
        """Evaluator step baseline recorded for the newest execution, if any."""
        with self._execution_condition:
            execution = self._execution
        if execution is None:
            return None
        return execution.baseline_evaluator_steps

    # ------------------------------------------------------------------ #
    # Execution control
    # ------------------------------------------------------------------ #

    def begin_execution(
        self,
        instruction: str,
        requested_steps: int,
        *,
        baseline_evaluator_steps: int | None = None,
    ) -> int:
        """Allow a bounded number of policy actions through a controlled gateway."""
        if not self.controlled:
            raise RuntimeError("Execution control is disabled for this gateway")
        if requested_steps < 1:
            raise ValueError("requested_steps must be positive")
        with self._execution_condition:
            if self._execution is not None and self._execution.status == EXECUTING:
                raise RuntimeError("Another VLA execution is already active")
            self._execution_serial += 1
            self._execution = ExecutionState(
                execution_id=self._execution_serial,
                instruction=str(instruction),
                requested_steps=int(requested_steps),
                baseline_evaluator_steps=baseline_evaluator_steps,
            )
            self._execution_condition.notify_all()
            return self._execution_serial

    def begin_local_execution(
        self,
        label: str,
        requested_steps: int,
        action: list[float],
        *,
        baseline_evaluator_steps: int | None = None,
    ) -> int:
        """Execute a fixed official R1Pro action without querying OpenPI."""
        if requested_steps < 1:
            raise ValueError("requested_steps must be positive")
        return self.begin_local_trajectory(
            label,
            [action for _ in range(int(requested_steps))],
            baseline_evaluator_steps=baseline_evaluator_steps,
        )

    def begin_local_trajectory(
        self,
        label: str,
        actions: list[list[float]],
        *,
        baseline_evaluator_steps: int | None = None,
    ) -> int:
        """Execute an explicit sequence of official 23-D R1Pro actions."""
        if not actions:
            raise ValueError("local R1Pro trajectory must contain at least one action")
        normalized_actions = [_validate_local_action(action) for action in actions]
        with self._execution_condition:
            if self._execution is not None and self._execution.status == EXECUTING:
                raise RuntimeError("Another robot execution is already active")
            if self._orphaned_responses:
                raise RuntimeError(
                    "cannot start a local action while an OpenPI reply is still in flight"
                )
            self._execution_serial += 1
            self._execution = ExecutionState(
                execution_id=self._execution_serial,
                instruction=str(label),
                requested_steps=len(normalized_actions),
                mode="local",
                local_action_count=len(normalized_actions),
                baseline_evaluator_steps=baseline_evaluator_steps,
            )
            self._local_action_sequence = normalized_actions
            self._execution_condition.notify_all()
            return self._execution_serial

    def current_hold_action(self) -> list[float]:
        publisher = self._observation_publisher
        if publisher is None:
            raise RuntimeError("observation publishing is disabled")
        return publisher.current_hold_action()

    def current_robot_state(self) -> dict[str, Any]:
        publisher = self._observation_publisher
        if publisher is None:
            raise RuntimeError("observation publishing is disabled")
        return publisher.current_robot_state()

    def wait_for_execution(
        self,
        execution_id: int,
        *,
        cancel_check: Any | None = None,
        poll_interval_s: float = 0.1,
        progress_timeout_s: float | None = None,
        total_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Wait for one execution while honoring cancellation and time budgets.

        ``progress_timeout_s`` bounds the time between two confirmed replies
        (and the time to the first reply), so a wedged evaluator cannot hold the
        controller forever.  ``total_timeout_s`` bounds the whole execution.
        """
        started = time.monotonic()
        last_progress = started
        last_completed = 0
        while True:
            if cancel_check is not None:
                reason = cancel_check()
                if reason:
                    self.cancel_execution(execution_id, str(reason))
            now = time.monotonic()
            with self._execution_condition:
                execution = self._execution
                if execution is None or execution.execution_id != execution_id:
                    raise RuntimeError(f"Unknown execution id: {execution_id}")
                if execution.completed_steps != last_completed:
                    last_completed = execution.completed_steps
                    last_progress = now
                if execution.status != EXECUTING:
                    return asdict(execution)
                if total_timeout_s is not None and now - started > total_timeout_s:
                    self._retire_locked(
                        execution,
                        TIMEOUT,
                        f"execution exceeded {total_timeout_s:.0f}s "
                        f"after {execution.completed_steps} step(s)",
                    )
                    return asdict(execution)
                if progress_timeout_s is not None and now - last_progress > progress_timeout_s:
                    self._retire_locked(
                        execution,
                        NO_PROGRESS,
                        f"no policy reply for {progress_timeout_s:.0f}s "
                        f"after {execution.completed_steps} step(s)",
                    )
                    return asdict(execution)
                self._execution_condition.wait(timeout=poll_interval_s)

    def cancel_execution(self, execution_id: int, reason: str) -> bool:
        with self._execution_condition:
            execution = self._execution
            if (
                execution is None
                or execution.execution_id != execution_id
                or execution.status != EXECUTING
            ):
                return False
            self._retire_locked(execution, INTERRUPTED, str(reason or "action interrupted"))
            return True

    # ------------------------------------------------------------------ #
    # Observation synchronization
    # ------------------------------------------------------------------ #

    def wait_for_observation_after(
        self,
        baseline: int | None,
        *,
        timeout_s: float,
        cancel_check: Any | None = None,
        poll_interval_s: float = 0.05,
    ) -> bool:
        """Wait until a frame newer than ``baseline`` has been decoded.

        The evaluator applies an action and only then sends the next
        observation, so a newer frame is direct evidence that the executed
        actions reached the simulator.
        """
        publisher = self._observation_publisher
        if publisher is None or baseline is None:
            return False
        deadline = time.monotonic() + timeout_s
        while True:
            if cancel_check is not None and cancel_check():
                return False
            if publisher.observation_count > baseline:
                return True
            with self._execution_condition:
                connected = self._evaluator_connected
            if not connected:
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(poll_interval_s)

    def publish_observation(self) -> Path | None:
        """Force the newest decoded frame to be published immediately."""
        if self._observation_publisher is None:
            return None
        return self._observation_publisher.publish_latest()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self, *, timeout_s: float = 10.0) -> None:
        """Start the gateway in a daemon thread and wait until its socket binds."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._ready.clear()
        self._startup_error = None
        self._stopped_cleanly = True
        self._thread = threading.Thread(
            target=self._thread_main,
            name="behavior-b1k-pi05-gateway",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise TimeoutError("Timed out while starting the BEHAVIOR B1K gateway")
        if self._startup_error is not None:
            raise RuntimeError("Failed to start the BEHAVIOR B1K gateway") from self._startup_error

    def stop(self, *, timeout_s: float = 10.0) -> None:
        self._finish_active_execution(ERROR, "gateway stopped")
        loop = self._loop
        shutdown_event = self._shutdown_event
        if loop is not None and shutdown_event is not None and loop.is_running():
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(shutdown_event.set)
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout_s)
        self._stopped_cleanly = thread is None or not thread.is_alive()
        self._thread = None
        with self._stats_lock:
            self._stats.status = "stopped" if self._stopped_cleanly else "stop_timeout"
            if not self._stopped_cleanly:
                self._stats.last_error = (
                    f"gateway thread still alive after {timeout_s:.0f}s; shutdown is not confirmed"
                )

    def serve_forever(self) -> None:
        """Run the gateway in the current thread until interrupted."""
        try:
            asyncio.run(self._run())
        except KeyboardInterrupt:
            logger.info("Gateway interrupted")

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except BaseException as exc:
            self._startup_error = exc
            with self._stats_lock:
                self._stats.status = "failed"
                self._stats.last_error = f"{type(exc).__name__}: {exc}"
            logger.exception("BEHAVIOR B1K gateway stopped with an error")
        finally:
            self._ready.set()

    async def _run(self) -> None:
        from websockets.asyncio.server import serve

        self._loop = asyncio.get_running_loop()
        self._shutdown_event = asyncio.Event()
        try:
            async with serve(
                self._handle_connection,
                self.listen_host,
                self.listen_port,
                compression=None,
                max_size=None,
                process_request=_health_check,
            ) as server:
                sockets = list(server.sockets or [])
                if sockets:
                    self.listen_port = int(sockets[0].getsockname()[1])
                with self._stats_lock:
                    self._stats.status = "ready"
                    self._stats.listen_port = self.listen_port
                    self._stats.started_at_monotonic = time.monotonic()
                    self._stats.last_error = None
                self._ready.set()
                logger.info(
                    "BEHAVIOR B1K gateway ready on ws://%s:%d -> %s",
                    self.listen_host,
                    self.listen_port,
                    self.upstream_url,
                )
                await self._shutdown_event.wait()
        finally:
            self._loop = None
            self._shutdown_event = None

    # ------------------------------------------------------------------ #
    # Connection handling
    # ------------------------------------------------------------------ #

    async def _handle_connection(self, downstream: Any) -> None:
        import websockets
        from websockets.asyncio.client import connect

        with self._execution_condition:
            duplicate = self.controlled and self._evaluator_connected
            if not duplicate:
                self._connection_serial += 1
                connection_serial = self._connection_serial
                self._active_connection_serial = connection_serial
                self._evaluator_connected = True
                self._last_disconnect = None
                self._execution_condition.notify_all()
            else:
                connection_serial = self._connection_serial
        if duplicate:
            await downstream.close(code=1013, reason="A BEHAVIOR evaluator is already connected")
            return
        with self._stats_lock:
            self._stats.active_connections += 1
            self._stats.total_connections += 1
        try:
            async with connect(
                self.upstream_url,
                compression=None,
                max_size=None,
                open_timeout=self.upstream_open_timeout_s,
            ) as upstream:
                # The first upstream frame is OpenPI policy metadata.  BEHAVIOR
                # expects it immediately after connecting.
                metadata = await upstream.recv()
                with self._execution_condition:
                    self._upstream_supports_action_cache_invalidation = (
                        _supports_action_cache_invalidation(metadata)
                    )
                    self._upstream_connected = True
                    self._execution_condition.notify_all()
                await downstream.send(metadata)

                to_upstream = asyncio.create_task(
                    self._relay(
                        downstream,
                        upstream,
                        request_direction=True,
                        connection_serial=connection_serial,
                    )
                )
                to_downstream = asyncio.create_task(
                    self._relay(
                        upstream,
                        downstream,
                        request_direction=False,
                        connection_serial=connection_serial,
                    )
                )
                done, pending = await asyncio.wait(
                    {to_upstream, to_downstream},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                for task in pending:
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                for task in done:
                    error = task.exception()
                    if error is not None:
                        raise error
        except websockets.ConnectionClosed:
            pass
        except Exception as exc:
            self._note_disconnect("proxy_error", f"{type(exc).__name__}: {exc}")
            with self._stats_lock:
                self._stats.last_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Failed to proxy a BEHAVIOR policy connection")
            with contextlib.suppress(Exception):
                await downstream.close(code=1013, reason="OpenPI upstream unavailable")
            self._finish_active_execution(ERROR, f"{type(exc).__name__}: {exc}")
        finally:
            with self._stats_lock:
                self._stats.active_connections = max(0, self._stats.active_connections - 1)
            with self._execution_condition:
                self._evaluator_connected = False
                self._upstream_connected = False
                self._upstream_supports_action_cache_invalidation = False
                if self._active_connection_serial == connection_serial:
                    self._active_connection_serial = None
                # Replies in flight for the retired connection can never arrive.
                self._orphaned_responses = 0
                self._execution_condition.notify_all()
            status, detail = self._disconnect_verdict(connection_serial)
            self._retire_for_connection(connection_serial, status, detail)

    def _disconnect_verdict(self, connection_serial: int) -> tuple[str, str]:
        """Turn a closed connection into an execution verdict.

        An evaluator that closes itself is an expected episode end.  An
        upstream that dies, or an evaluator that vanishes without a single
        applied action, is an infrastructure failure and must not be reported
        as a finished episode.
        """
        reason = self._last_disconnect or "unknown"
        if reason.startswith("upstream"):
            return ERROR, f"OpenPI upstream disconnected ({reason})"
        if reason.startswith("proxy_error"):
            return ERROR, f"gateway proxy error ({reason})"
        with self._execution_condition:
            execution = self._execution
            applied = execution.completed_steps if execution is not None else 0
            owns = execution is not None and (
                execution.connection_serial in (None, connection_serial)
            )
        if owns and applied == 0:
            return ERROR, "BEHAVIOR evaluator disconnected before any action was applied"
        return EPISODE_FINISHED, "BEHAVIOR evaluator disconnected"

    def _note_disconnect(self, side: str, detail: str) -> None:
        """Record why this connection ended; the first cause wins.

        Tearing down one relay also closes the other side, so a later note is a
        consequence of the first one.  Letting it overwrite the root cause would
        report a crashed policy upstream as a finished episode.
        """
        with self._execution_condition:
            if self._last_disconnect is None:
                self._last_disconnect = f"{side}: {detail}"
            self._execution_condition.notify_all()

    # ------------------------------------------------------------------ #
    # Relaying
    # ------------------------------------------------------------------ #

    async def _relay(
        self,
        source: Any,
        destination: Any,
        *,
        request_direction: bool,
        connection_serial: int,
    ) -> None:
        import websockets

        direction = "evaluator" if request_direction else "upstream"
        try:
            async for message in source:
                is_reset = request_direction and _is_reset_message(message)
                if is_reset:
                    self._record_reset()
                if request_direction and not is_reset and self._observation_publisher is not None:
                    self._observation_publisher.capture(message)
                dispatch = None
                if request_direction and not is_reset and self.controlled:
                    dispatch = await self._await_request_permission(source, connection_serial)
                    if dispatch is False:
                        return
                if request_direction and not is_reset and dispatch is not None:
                    mode, local_action = dispatch
                    if mode == "local":
                        await source.send(_pack_action_response(local_action))
                        with self._execution_condition:
                            self._upstream_needs_action_cache_invalidation = True
                            self._execution_condition.notify_all()
                        with self._stats_lock:
                            self._stats.local_actions_generated += 1
                            self._stats.responses_forwarded += 1
                        execution_finished = self._record_action_response(connection_serial)
                        if execution_finished and self._observation_publisher is not None:
                            self._observation_publisher.publish_latest()
                        continue
                    if self._consume_action_cache_invalidation_needed():
                        await destination.send(_pack_action_cache_invalidation_request())
                        with self._stats_lock:
                            self._stats.upstream_cache_invalidations_sent += 1
                await destination.send(message)
                with self._stats_lock:
                    if request_direction:
                        self._stats.requests_forwarded += 1
                        if is_reset:
                            self._stats.evaluator_resets_forwarded += 1
                    else:
                        self._stats.responses_forwarded += 1
                if not request_direction:
                    execution_finished = self._record_action_response(connection_serial)
                    if execution_finished and self._observation_publisher is not None:
                        self._observation_publisher.publish_latest()
        except websockets.ConnectionClosed as exc:
            self._note_disconnect(direction, _close_description(exc))
            raise
        # websockets 16 ends the async iterator without raising when the peer
        # closes, so a normal loop exit is itself the disconnect signal.  The
        # only other way out of this coroutine is the explicit ``return`` above.
        self._note_disconnect(direction, "stream ended")

    async def _await_request_permission(
        self, downstream: Any, connection_serial: int
    ) -> tuple[str, list[float] | None] | bool:
        while True:
            reserved = self._reserve_request_slot(connection_serial)
            if isinstance(reserved, tuple):
                return reserved
            with self._execution_condition:
                if not self._evaluator_connected:
                    return False
            if getattr(downstream, "close_code", None) is not None:
                return False
            if self._shutdown_event is not None and self._shutdown_event.is_set():
                return False
            await asyncio.sleep(0.05)

    def _reserve_request_slot(
        self, connection_serial: int
    ) -> tuple[str, list[float] | None] | bool | None:
        """Atomically confirm an active execution and count a forwarded request.

        Returns the execution mode and optional local action when the request
        may run, ``None`` when there is no active execution, and ``False`` when
        the active execution belongs to a different evaluator connection.
        """
        with self._execution_condition:
            execution = self._execution
            if execution is None or execution.status != EXECUTING:
                return None
            if execution.connection_serial is None:
                execution.connection_serial = connection_serial
            elif execution.connection_serial != connection_serial:
                return False
            local_action = None
            if execution.mode == "local":
                if execution.requests_sent >= len(self._local_action_sequence):
                    raise RuntimeError("local R1Pro trajectory was exhausted unexpectedly")
                local_action = self._local_action_sequence[execution.requests_sent]
            execution.requests_sent += 1
            self._execution_condition.notify_all()
            return execution.mode, local_action

    def _record_action_response(self, connection_serial: int) -> bool:
        if not self.controlled:
            return False
        with self._execution_condition:
            if self._orphaned_responses > 0:
                # A reply owed to an execution that already retired.  Crediting
                # it here would silently shorten the next action's budget.
                self._orphaned_responses -= 1
                self._execution_condition.notify_all()
                return False
            execution = self._execution
            if execution is None or execution.status != EXECUTING:
                return False
            if execution.connection_serial not in (None, connection_serial):
                return False
            execution.completed_steps += 1
            execution.responses_recorded += 1
            if execution.completed_steps >= execution.requested_steps:
                if self._observation_publisher is not None:
                    execution.sync_observation_baseline = (
                        self._observation_publisher.observation_count
                    )
                self._retire_locked(
                    execution, STEP_COMPLETED, "requested action budget exhausted"
                )
            self._execution_condition.notify_all()
            return execution.status != EXECUTING

    def _record_reset(self) -> None:
        with self._execution_condition:
            self._episode_id += 1
            # This same evaluator reset message is forwarded to OpenPI.
            self._upstream_needs_action_cache_invalidation = False
            execution = self._execution
            if (
                execution is not None
                and execution.status == EXECUTING
                and execution.completed_steps > 0
            ):
                self._retire_locked(
                    execution, EPISODE_RESET, "BEHAVIOR evaluator reset the policy"
                )
            self._execution_condition.notify_all()
        if self._observation_publisher is not None:
            self._observation_publisher.note_reset()

    def _consume_action_cache_invalidation_needed(self) -> bool:
        with self._execution_condition:
            if not self._upstream_needs_action_cache_invalidation:
                return False
            if not self._upstream_supports_action_cache_invalidation:
                raise RuntimeError(
                    "OpenPI upstream does not advertise invalidate_action_cache. "
                    "Restart the updated champion server before handing off from local motion; "
                    "refusing to reset task/stage state."
                )
            self._upstream_needs_action_cache_invalidation = False
            self._execution_condition.notify_all()
            return True

    # ------------------------------------------------------------------ #
    # Retirement bookkeeping
    # ------------------------------------------------------------------ #

    def _retire_locked(self, execution: ExecutionState, status: str, detail: str) -> None:
        """Move an execution to a terminal state and account for its orphans."""
        execution.status = status
        execution.detail = detail
        execution.pending_responses_at_retire = max(
            0, execution.requests_sent - execution.responses_recorded
        )
        self._orphaned_responses += execution.pending_responses_at_retire
        self._execution_condition.notify_all()

    def _retire_for_connection(self, connection_serial: int, status: str, detail: str) -> None:
        with self._execution_condition:
            execution = self._execution
            if execution is None or execution.status != EXECUTING:
                return
            if execution.connection_serial not in (None, connection_serial):
                return
            self._retire_locked(execution, status, detail)

    def _finish_active_execution(self, status: str, detail: str) -> None:
        with self._execution_condition:
            execution = self._execution
            if execution is None or execution.status != EXECUTING:
                return
            self._retire_locked(execution, status, detail)


def _close_description(exc: Any) -> str:
    for attribute in ("rcvd", "sent"):
        frame = getattr(exc, attribute, None)
        code = getattr(frame, "code", None)
        if code is not None:
            reason = getattr(frame, "reason", "") or ""
            return f"code={code}{f' reason={reason}' if reason else ''}"
    code = getattr(exc, "code", None)
    return f"code={code}" if code is not None else "closed"


def _is_reset_message(message: Any) -> bool:
    if not isinstance(message, (bytes, bytearray, memoryview)) or len(message) > 128:
        return False
    try:
        import msgpack

        payload = msgpack.unpackb(message, raw=False)
    except Exception:
        return False
    return isinstance(payload, dict) and payload.get("reset") is True


def _pack_numpy_array(value: Any) -> dict[bytes, Any]:
    import numpy as np

    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"cannot pack {type(value).__name__}")


def _pack_action_response(action: list[float] | None) -> bytes:
    if action is None:
        raise RuntimeError("local execution has no action payload")
    import msgpack
    import numpy as np

    array = np.asarray(action, dtype=np.float32)[None, :]
    return msgpack.packb({"action": array}, default=_pack_numpy_array)


def _supports_action_cache_invalidation(metadata: Any) -> bool:
    """Read a side copy of metadata; never rewrite the evaluator's handshake."""
    if not isinstance(metadata, (bytes, bytearray, memoryview)):
        return False
    import msgpack

    try:
        payload = msgpack.unpackb(metadata, raw=False)
    except (TypeError, ValueError, msgpack.UnpackException):
        return False
    if not isinstance(payload, dict):
        return False
    capabilities = payload.get("emerge_capabilities")
    return isinstance(capabilities, dict) and capabilities.get("invalidate_action_cache") is True


def _pack_action_cache_invalidation_request() -> bytes:
    import msgpack

    return msgpack.packb({"invalidate_action_cache": True})


def _validate_local_action(action: list[float]) -> list[float]:
    if len(action) != 23:
        raise ValueError(f"local R1Pro action must have 23 values, got {len(action)}")
    try:
        normalized = [float(value) for value in action]
    except (TypeError, ValueError) as exc:
        raise ValueError("local R1Pro action must contain numbers") from exc
    if not all(math.isfinite(value) for value in normalized):
        raise ValueError("local R1Pro action must contain finite values")
    return normalized


def _health_check(connection: Any, request: Any) -> Any | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-url", default="ws://127.0.0.1:8011")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8000)
    parser.add_argument("--upstream-open-timeout", type=float, default=10.0)
    parser.add_argument("--controlled", action="store_true")
    parser.add_argument("--observation-workspace", type=Path)
    parser.add_argument("--observation-publish-interval", type=int, default=5)
    parser.add_argument(
        "--camera-calibration-config",
        type=Path,
        help="JSON file containing either camera_calibration or the calibration object itself.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    camera_calibration = None
    if args.camera_calibration_config is not None:
        config_path = args.camera_calibration_config.expanduser().resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"camera calibration config must be an object: {config_path}")
        camera_calibration = payload.get("camera_calibration", payload)
        if not isinstance(camera_calibration, dict):
            raise ValueError(f"camera_calibration must be an object: {config_path}")
    gateway = B1KProxyGateway(
        upstream_url=args.upstream_url,
        listen_host=args.listen_host,
        listen_port=args.listen_port,
        upstream_open_timeout_s=args.upstream_open_timeout,
        controlled=args.controlled,
        observation_workspace=args.observation_workspace,
        observation_publish_interval=args.observation_publish_interval,
        camera_calibration=camera_calibration,
    )
    gateway.serve_forever()


if __name__ == "__main__":
    main()

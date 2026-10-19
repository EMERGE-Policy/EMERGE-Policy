"""End-to-end checks for the real Controller, action queue and driver.

The websocket transport itself is covered by ``test_gateway``.  Here the point
is the EMERGE side: a queued ``vla_execute`` must move through
``pending -> running -> completed/failed/cancelled`` exactly once, and
``ROBOT_STATE.md`` must reflect the driver's synchronized state when it does.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import msgpack
import numpy as np

from scripts.Behavior_eval.checks.test_driver import _FakeGateway
from robot.drivers.behavior_r1pro_driver import BehaviorR1ProDriver
from scripts.Behavior_eval.runner_common import (
    prepare_workspace,
    robot_state_section,
)
from robot.behavior.contracts import EVALUATION_STATUS_SCHEMA
from Emerge.utils.action_queue import (
    append_action,
    normalize_action_document,
    parse_action_markdown,
    update_action_document,
)
from robot.controller import _poll_once

TASK_NAME = "turning_on_radio"
OFFICIAL_PROMPT = "Turn on the radio receiver that's on the table in the living room."


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _pack_array(value):
    return {
        b"__ndarray__": True,
        b"data": value.tobytes(),
        b"dtype": value.dtype.str,
        b"shape": value.shape,
    }


def _camera_message(seed: int = 0) -> bytes:
    image = np.full((4, 5, 3), seed % 255, dtype=np.uint8)
    return msgpack.packb(
        {
            "robot::robot:zed_link:Camera:0::rgb": image,
            "robot::robot:left_realsense_link:Camera:0::rgb": image,
            "robot::robot:right_realsense_link:Camera:0::rgb": image,
        },
        default=_pack_array,
    )


class _AutoReplyOpenPIServer:
    """Fake policy server that echoes one action per observation."""

    def __init__(self) -> None:
        self.port = _free_port()
        self.ready = threading.Event()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stop_event: asyncio.Event | None = None
        self.thread = threading.Thread(target=self._thread_main, daemon=True)

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(5):
            raise TimeoutError("fake OpenPI server did not start")

    def stop(self) -> None:
        if self.loop is not None and self.stop_event is not None:
            self.loop.call_soon_threadsafe(self.stop_event.set)
        self.thread.join(5)

    def _thread_main(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        from websockets.asyncio.server import serve

        self.loop = asyncio.get_running_loop()
        self.stop_event = asyncio.Event()
        async with serve(self._handler, "127.0.0.1", self.port, compression=None, max_size=None):
            self.ready.set()
            await self.stop_event.wait()

    @staticmethod
    async def _handler(websocket) -> None:
        await websocket.send(b"openpi-metadata")
        async for message in websocket:
            try:
                payload = msgpack.unpackb(message, raw=False)
            except Exception:
                payload = None
            if isinstance(payload, dict) and payload.get("reset") is True:
                continue
            await websocket.send(b"action:" + message)


class ControllerQueueIntegrationTest(unittest.TestCase):
    """The Controller must drive a real gateway to a synchronized result."""

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.workspace = Path(self.temporary.name) / "workspace"
        prepare_workspace(self.workspace, task_name=TASK_NAME, instruction=OFFICIAL_PROMPT)
        self.action_file = self.workspace / "ACTION.md"
        self.state_file = self.workspace / "ROBOT_STATE.md"
        self.status_path = self.workspace / "BEHAVIOR_STATUS.json"
        self.upstream = _AutoReplyOpenPIServer()
        self.upstream.start()
        self.driver = BehaviorR1ProDriver(
            workspace=self.workspace,
            upstream_url=f"ws://127.0.0.1:{self.upstream.port}",
            listen_host="127.0.0.1",
            listen_port=0,
            status_path=self.status_path,
            max_action_steps=100,
            progress_timeout_s=10.0,
            execution_timeout_s=20.0,
            state_sync_timeout_s=3.0,
            required_instruction=OFFICIAL_PROMPT,
            expected_task=TASK_NAME,
        )
        self.driver.load_scene({})
        self.gateway_port = self.driver._gateway.listen_port
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._evaluator: threading.Thread | None = None

    def tearDown(self) -> None:
        self._stop.set()
        if self._evaluator is not None:
            self._evaluator.join(timeout=5)
        self.driver.close()
        self.upstream.stop()
        self.temporary.cleanup()

    # -- helpers -------------------------------------------------------- #

    def _write_status(self, **overrides: Any) -> None:
        payload = {
            "schema_version": EVALUATION_STATUS_SCHEMA,
            "task": TASK_NAME,
            "instance_id": 301,
            "rollout_id": 0,
            "phase": "running",
            "steps": 0,
            "success": False,
            "done": False,
            "terminated": False,
            "truncated": False,
        }
        payload.update(overrides)
        self.status_path.write_text(json.dumps(payload), encoding="utf-8")

    def _start_evaluator(self, *, finish_after: int | None) -> None:
        self._evaluator = threading.Thread(
            target=self._evaluator_loop, args=(finish_after,), daemon=True
        )
        self._evaluator.start()
        if not self._ready.wait(10):
            raise TimeoutError("the fake evaluator did not connect")

    def _evaluator_loop(self, finish_after: int | None) -> None:
        from websockets.sync.client import connect

        with connect(
            f"ws://127.0.0.1:{self.gateway_port}", compression=None, max_size=None
        ) as client:
            client.recv()
            client.send(msgpack.packb({"reset": True}))
            self._write_status(steps=0)
            self._ready.set()
            steps = 0
            while not self._stop.is_set():
                client.send(_camera_message(steps))
                if self._recv_action(client) is None:
                    break
                steps += 1
                self._write_status(steps=steps)
                if finish_after is not None and steps >= finish_after:
                    break
            if finish_after is not None:
                self._write_status(
                    steps=steps,
                    phase="completed",
                    done=True,
                    termination_reason="truncated",
                )

    def _recv_action(self, client) -> bytes | None:
        """Receive one action, staying responsive to the teardown flag."""
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if self._stop.is_set():
                return None
            try:
                return client.recv(timeout=0.2)
            except TimeoutError:
                continue
        return None

    def _enqueue(self, *, steps: int, instruction: str = OFFICIAL_PROMPT) -> str:
        document = update_action_document(
            self.action_file,
            lambda current: append_action(
                current,
                action_type="vla_execute",
                parameters={"instruction": instruction, "step": steps},
            ),
        )
        return str(document["actions"][-1]["id"])

    def _find(self, action_id: str) -> dict[str, Any] | None:
        try:
            payload = parse_action_markdown(self.action_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        document = normalize_action_document(payload) if payload else None
        if document is None:
            return None
        for action in document["actions"]:
            if action["id"] == action_id:
                return dict(action)
        return None

    def _drive_until_terminal(self, action_id: str, *, timeout_s: float = 30.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            _poll_once(self.driver, self.action_file, self.state_file)
            action = self._find(action_id)
            if action is not None and action["status"] in {"completed", "failed", "cancelled"}:
                return action
            time.sleep(0.02)
        raise AssertionError(f"action {action_id} never reached a terminal state")

    # -- tests ---------------------------------------------------------- #

    def test_obsolete_control_mode_action_is_rejected_through_queue(self) -> None:
        document = update_action_document(
            self.action_file,
            lambda current: append_action(
                current,
                action_type="set_control_mode",
                parameters={"mode": "manipulation"},
            ),
        )
        action_id = str(document["actions"][-1]["id"])

        _poll_once(self.driver, self.action_file, self.state_file)

        action = self._find(action_id)
        self.assertEqual(action["status"], "failed", action)
        self.assertIn("Unknown action type", action["result"])
        robot = robot_state_section(self.workspace)
        self.assertNotIn("control_mode", robot)
        self.assertNotIn("base_motion_enabled", robot)

    def test_action_completes_and_publishes_synchronized_state(self) -> None:
        self._start_evaluator(finish_after=None)
        action_id = self._enqueue(steps=2)
        action = self._drive_until_terminal(action_id)

        self.assertEqual(action["status"], "completed")
        self.assertIn("step_completed", action["result"])
        robot = robot_state_section(self.workspace)
        self.assertTrue(robot["state_synced"], robot)
        self.assertIsNotNone(robot["applied_steps"])
        self.assertGreaterEqual(int(robot["applied_steps"]), 2)
        self.assertTrue(
            (self.workspace / "artifacts/observations/observation.json").is_file()
        )

    def test_episode_that_ends_with_the_last_action_is_not_a_failure(self) -> None:
        # The evaluator stops exactly at the action budget, so no further
        # observation frame can confirm the final action.
        self._start_evaluator(finish_after=2)
        action_id = self._enqueue(steps=2)
        action = self._drive_until_terminal(action_id)

        self.assertEqual(action["status"], "completed", action)
        status = json.loads(self.status_path.read_text(encoding="utf-8"))
        self.assertTrue(status["done"])
        self.assertTrue(robot_state_section(self.workspace)["state_synced"])

    def test_rejected_instruction_marks_the_queue_entry_failed(self) -> None:
        self._start_evaluator(finish_after=None)
        action_id = self._enqueue(steps=2, instruction="turn on the radio")
        action = self._drive_until_terminal(action_id)

        self.assertEqual(action["status"], "failed")
        self.assertIn("Failed:", action["result"])
        self.assertIn(OFFICIAL_PROMPT, action["result"])


class ControllerCancellationIntegrationTest(unittest.TestCase):
    """A cancelled action must not be reported as a completed one."""

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.workspace = Path(self.temporary.name) / "workspace"
        prepare_workspace(self.workspace, task_name=TASK_NAME, instruction=OFFICIAL_PROMPT)
        self.action_file = self.workspace / "ACTION.md"
        self.state_file = self.workspace / "ROBOT_STATE.md"
        self.driver = BehaviorR1ProDriver(
            workspace=self.workspace,
            listen_port=0,
            required_instruction=OFFICIAL_PROMPT,
            expected_task=TASK_NAME,
        )
        self.gateway = _CancelAwareGateway()
        self.driver._gateway = self.gateway

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _enqueue(self, *, steps: int) -> str:
        document = update_action_document(
            self.action_file,
            lambda current: append_action(
                current,
                action_type="vla_execute",
                parameters={"instruction": OFFICIAL_PROMPT, "step": steps},
            ),
        )
        return str(document["actions"][-1]["id"])

    def _find(self, action_id: str) -> dict[str, Any] | None:
        try:
            payload = parse_action_markdown(self.action_file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        document = normalize_action_document(payload) if payload else None
        if document is None:
            return None
        for action in document["actions"]:
            if action["id"] == action_id:
                return dict(action)
        return None

    def test_cancel_request_reaches_the_driver_and_cancels_the_queue_entry(self) -> None:
        action_id = self._enqueue(steps=50)
        worker = threading.Thread(
            target=_poll_once, args=(self.driver, self.action_file, self.state_file), daemon=True
        )
        worker.start()

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            action = self._find(action_id)
            if action is not None and action["status"] == "running":
                break
            time.sleep(0.02)
        else:
            raise AssertionError("the action never started running")

        def mark(document):
            for item in document["actions"]:
                if item["id"] == action_id:
                    item["cancel_requested"] = True
                    item["cancel_reason"] = "test cancellation"

        update_action_document(self.action_file, mark)
        worker.join(timeout=15)
        self.assertFalse(worker.is_alive(), "the Controller did not observe the cancellation")

        action = self._find(action_id)
        self.assertEqual(action["status"], "cancelled", action)
        self.assertTrue(action["result"].startswith("Interrupted:"), action["result"])
        self.assertFalse(action["cancel_requested"] is None)


class _CancelAwareGateway(_FakeGateway):
    """Fake gateway that honors the Controller's cancellation hook."""

    def wait_for_execution(self, execution_id: int, *, cancel_check=None, **_kwargs: Any):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if cancel_check is not None:
                reason = cancel_check()
                if reason:
                    self.execution.update(
                        status="interrupted",
                        completed_steps=0,
                        detail=str(reason),
                    )
                    return dict(self.execution)
            time.sleep(0.02)
        raise AssertionError("the Controller never asked the driver to cancel")


if __name__ == "__main__":
    unittest.main()

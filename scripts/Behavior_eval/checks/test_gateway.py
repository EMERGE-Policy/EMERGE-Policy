from __future__ import annotations

import asyncio
import contextlib
import json
import socket
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib import request as urllib_request

import msgpack
import numpy as np
from PIL import Image

from robot.behavior.gateway import B1KProxyGateway, _supports_action_cache_invalidation
from robot.behavior.observation import ObservationPublisher


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class CacheCapabilityTest(unittest.TestCase):
    def test_capability_requires_explicit_boolean_true_in_valid_metadata(self) -> None:
        metadata = {"emerge_capabilities": {"invalidate_action_cache": True}}
        self.assertTrue(_supports_action_cache_invalidation(msgpack.packb(metadata)))
        for value in (b"openpi-metadata", b"", "text", msgpack.packb([]),
                      msgpack.packb({}),
                      msgpack.packb({"emerge_capabilities": ["invalidate_action_cache"]}),
                      msgpack.packb({"emerge_capabilities": {"invalidate_action_cache": 1}}),
                      msgpack.packb({"emerge_capabilities": {"invalidate_action_cache": False}})):
            with self.subTest(value=value):
                self.assertFalse(_supports_action_cache_invalidation(value))


class _FakeOpenPIServer:
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
        async with serve(
            self._handler,
            "127.0.0.1",
            self.port,
            compression=None,
            max_size=None,
        ):
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
            if message == b"reset" or (isinstance(payload, dict) and payload.get("reset") is True):
                continue
            await websocket.send(b"action:" + message)


class B1KProxyGatewayTest(unittest.TestCase):
    def setUp(self) -> None:
        self.upstream = _FakeOpenPIServer()
        self.upstream.start()
        self.gateway = B1KProxyGateway(
            upstream_url=f"ws://127.0.0.1:{self.upstream.port}",
            listen_host="127.0.0.1",
            listen_port=0,
        )
        self.gateway.start()

    def tearDown(self) -> None:
        self.gateway.stop()
        self.upstream.stop()

    def test_relays_metadata_observation_and_action_without_changes(self) -> None:
        from websockets.sync.client import connect

        with connect(
            f"ws://127.0.0.1:{self.gateway.listen_port}",
            compression=None,
            max_size=None,
        ) as client:
            self.assertEqual(client.recv(), b"openpi-metadata")
            client.send(b"observation-bytes")
            self.assertEqual(client.recv(), b"action:observation-bytes")
            client.send(b"reset")

        state = self.gateway.snapshot()
        self.assertEqual(state["status"], "ready")
        self.assertEqual(state["total_connections"], 1)
        self.assertEqual(state["requests_forwarded"], 2)
        self.assertEqual(state["responses_forwarded"], 1)

    def test_health_endpoint_is_available(self) -> None:
        opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
        with opener.open(
            f"http://127.0.0.1:{self.gateway.listen_port}/healthz", timeout=2
        ) as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(response.read(), b"OK\n")

    def test_observation_side_copy_does_not_change_wire_payload(self) -> None:
        from websockets.sync.client import connect

        def pack_array(value):
            return {
                b"__ndarray__": True,
                b"data": value.tobytes(),
                b"dtype": value.dtype.str,
                b"shape": value.shape,
            }

        self.gateway.stop()
        with TemporaryDirectory() as temporary:
            self.gateway = B1KProxyGateway(
                upstream_url=f"ws://127.0.0.1:{self.upstream.port}",
                listen_host="127.0.0.1",
                listen_port=0,
                observation_workspace=temporary,
                observation_publish_interval=1,
            )
            self.gateway.start()
            image = np.zeros((4, 5, 3), dtype=np.uint8)
            depth = np.ones((4, 5, 1), dtype=np.float32)
            message = msgpack.packb(
                {
                    "robot::robot:zed_link:Camera:0::rgb": image,
                    "robot::robot:left_realsense_link:Camera:0::rgb": image,
                    "robot::robot:right_realsense_link:Camera:0::rgb": image,
                    "robot::robot:zed_link:Camera:0::depth_linear": depth,
                    "robot::robot:left_realsense_link:Camera:0::depth_linear": depth,
                    "robot::robot:right_realsense_link:Camera:0::depth_linear": depth,
                },
                default=pack_array,
            )
            with connect(
                f"ws://127.0.0.1:{self.gateway.listen_port}",
                compression=None,
                max_size=None,
            ) as client:
                self.assertEqual(client.recv(), b"openpi-metadata")
                client.send(message)
                self.assertEqual(client.recv(), b"action:" + message)
            self.assertTrue((Path(temporary) / "artifacts/observations/observation.json").is_file())
            manifest = json.loads(
                (Path(temporary) / "artifacts/observations/observation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(manifest["rgbd"])

    def test_controlled_gateway_waits_for_execution_and_honors_budget(self) -> None:
        from websockets.sync.client import connect

        self.gateway.stop()
        self.gateway = B1KProxyGateway(
            upstream_url=f"ws://127.0.0.1:{self.upstream.port}",
            listen_host="127.0.0.1",
            listen_port=0,
            controlled=True,
        )
        self.gateway.start()

        with connect(
            f"ws://127.0.0.1:{self.gateway.listen_port}",
            compression=None,
            max_size=None,
        ) as client:
            self.assertEqual(client.recv(), b"openpi-metadata")
            client.send(msgpack.packb({"reset": True}))
            client.send(b"observation-1")
            with self.assertRaises(TimeoutError):
                client.recv(timeout=0.1)

            execution_id = self.gateway.begin_execution("turn on the radio", 2)
            self.assertEqual(client.recv(timeout=1), b"action:observation-1")
            client.send(b"observation-2")
            self.assertEqual(client.recv(timeout=1), b"action:observation-2")
            result = self.gateway.wait_for_execution(execution_id)
            self.assertEqual(result["status"], "step_completed")
            self.assertEqual(result["completed_steps"], 2)

            client.send(b"observation-3")
            with self.assertRaises(TimeoutError):
                client.recv(timeout=0.1)

    def test_controlled_gateway_returns_when_evaluator_disconnects(self) -> None:
        from websockets.sync.client import connect

        self.gateway.stop()
        self.gateway = B1KProxyGateway(
            upstream_url=f"ws://127.0.0.1:{self.upstream.port}",
            listen_host="127.0.0.1",
            listen_port=0,
            controlled=True,
        )
        self.gateway.start()
        execution_id = self.gateway.begin_execution("turn on the radio", 5)

        with connect(
            f"ws://127.0.0.1:{self.gateway.listen_port}",
            compression=None,
            max_size=None,
        ) as client:
            self.assertEqual(client.recv(), b"openpi-metadata")
            client.send(b"observation")
            self.assertEqual(client.recv(timeout=1), b"action:observation")

        deadline = time.monotonic() + 2
        while self.gateway.snapshot()["evaluator_connected"] and time.monotonic() < deadline:
            time.sleep(0.01)
        result = self.gateway.wait_for_execution(execution_id)
        self.assertEqual(result["status"], "episode_finished")
        self.assertEqual(result["completed_steps"], 1)

    def test_cancel_execution_unblocks_waiter(self) -> None:
        self.gateway.stop()
        self.gateway = B1KProxyGateway(
            upstream_url=f"ws://127.0.0.1:{self.upstream.port}",
            listen_host="127.0.0.1",
            listen_port=0,
            controlled=True,
        )
        self.gateway.start()
        execution_id = self.gateway.begin_execution("turn on the radio", 5)
        self.assertTrue(self.gateway.cancel_execution(execution_id, "test cancellation"))
        result = self.gateway.wait_for_execution(execution_id)
        self.assertEqual(result["status"], "interrupted")
        self.assertEqual(result["detail"], "test cancellation")


class ObservationPublisherTest(unittest.TestCase):
    def test_publishes_three_camera_views_in_emerge_manifest(self) -> None:
        def pack_array(value):
            return {
                b"__ndarray__": True,
                b"data": value.tobytes(),
                b"dtype": value.dtype.str,
                b"shape": value.shape,
            }

        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            publisher = ObservationPublisher(workspace, publish_interval=1)
            image = np.arange(4 * 5 * 4, dtype=np.uint8).reshape(4, 5, 4)
            payload = {
                "robot::robot:zed_link:Camera:0::rgb": image,
                "robot::robot:left_realsense_link:Camera:0::rgb": image + 1,
                "robot::robot:right_realsense_link:Camera:0::rgb": image + 2,
                "robot::cam_rel_poses": np.tile(
                    np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32),
                    3,
                )[None, :],
                "robot::proprio": np.zeros((1, 61), dtype=np.float32),
            }
            message = msgpack.packb(payload, default=pack_array)
            publisher.capture(message)

            manifest = json.loads(publisher.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["reference_view"], "head")
            self.assertFalse(manifest["calibrated"])
            self.assertTrue(manifest["extrinsics_calibrated"])
            self.assertEqual(manifest["robot_state"]["proprio_dim"], 61)
            self.assertEqual(
                [view["name"] for view in manifest["views"]],
                ["head", "left_wrist", "right_wrist"],
            )
            for view in manifest["views"]:
                image_path = workspace / view["image_path"]
                self.assertTrue(image_path.is_file())
                with Image.open(image_path) as saved:
                    self.assertEqual(saved.size, (5, 4))


def _pack_array(value):
    return {
        b"__ndarray__": True,
        b"data": value.tobytes(),
        b"dtype": value.dtype.str,
        b"shape": value.shape,
    }


def _camera_message(seed: int = 0) -> bytes:
    image = np.full((4, 5, 3), seed, dtype=np.uint8)
    return msgpack.packb(
        {
            "robot::robot:zed_link:Camera:0::rgb": image,
            "robot::robot:left_realsense_link:Camera:0::rgb": image,
            "robot::robot:right_realsense_link:Camera:0::rgb": image,
        },
        default=_pack_array,
    )


def _r1pro_message() -> bytes:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    proprio = np.zeros((1, 61), dtype=np.float32)
    proprio[0, 3:10] = np.arange(7, dtype=np.float32) / 10
    proprio[0, 24:26] = 0.05
    proprio[0, 28:35] = np.arange(7, dtype=np.float32) / -10
    proprio[0, 49:51] = 0.05
    proprio[0, 53:57] = [1.0, -1.0, -0.5, 0.0]
    poses = np.tile(
        np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32), 3
    )[None, :]
    return msgpack.packb(
        {
            "robot::robot:zed_link:Camera:0::rgb": image,
            "robot::robot:left_realsense_link:Camera:0::rgb": image,
            "robot::robot:right_realsense_link:Camera:0::rgb": image,
            "robot::cam_rel_poses": poses,
            "robot::proprio": proprio,
        },
        default=_pack_array,
    )


def _unpack_array(value):
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    return value


class _ControllableOpenPIServer:
    """Fake OpenPI server whose replies are released by the test on demand."""

    def __init__(self) -> None:
        self.port = _free_port()
        self.ready = threading.Event()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.stop_event: asyncio.Event | None = None
        self.received: list[bytes] = []
        self.metadata = b"openpi-metadata"
        self._websocket = None
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._thread_main, daemon=True)

    def start(self) -> None:
        self.thread.start()
        if not self.ready.wait(5):
            raise TimeoutError("controllable OpenPI server did not start")

    def stop(self) -> None:
        if self.loop is not None and self.stop_event is not None:
            with contextlib.suppress(RuntimeError):
                self.loop.call_soon_threadsafe(self.stop_event.set)
        self.thread.join(5)

    def _thread_main(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        from websockets.asyncio.server import serve

        self.loop = asyncio.get_running_loop()
        self.stop_event = asyncio.Event()
        async with serve(
            self._handler, "127.0.0.1", self.port, compression=None, max_size=None
        ):
            self.ready.set()
            await self.stop_event.wait()

    async def _handler(self, websocket) -> None:
        self._websocket = websocket
        await websocket.send(self.metadata)
        async for message in websocket:
            with self._lock:
                self.received.append(bytes(message))

    def request_count(self) -> int:
        with self._lock:
            return len(self.received)

    def wait_for_requests(self, count: int, timeout_s: float = 3.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.request_count() >= count:
                return
            time.sleep(0.01)
        raise TimeoutError(f"upstream received {self.request_count()} of {count} requests")

    def send(self, payload: bytes) -> None:
        websocket = self._websocket
        loop = self.loop
        if websocket is None or loop is None:
            raise RuntimeError("the evaluator has not connected to the fake upstream yet")
        asyncio.run_coroutine_threadsafe(websocket.send(payload), loop).result(timeout=3)

    def drop(self) -> None:
        """Close the upstream side abruptly, as a crashed policy server would."""
        websocket = self._websocket
        loop = self.loop
        if websocket is None or loop is None:
            return
        asyncio.run_coroutine_threadsafe(websocket.close(), loop).result(timeout=3)


class ControlledExecutionOwnershipTest(unittest.TestCase):
    """An execution may only be credited by the replies it actually earned."""

    def setUp(self) -> None:
        self.upstream = _ControllableOpenPIServer()
        self.upstream.start()
        self.gateway = B1KProxyGateway(
            upstream_url=f"ws://127.0.0.1:{self.upstream.port}",
            listen_host="127.0.0.1",
            listen_port=0,
            controlled=True,
        )
        self.gateway.start()

    def tearDown(self) -> None:
        self.gateway.stop()
        self.upstream.stop()

    def _connect(self):
        from websockets.sync.client import connect

        return connect(
            f"ws://127.0.0.1:{self.gateway.listen_port}",
            compression=None,
            max_size=None,
        )

    def test_full_body_action_and_chunk_are_forwarded_byte_for_byte(self) -> None:
        from websockets.sync.client import connect

        execution_id = self.gateway.begin_execution("turn on the radio", 1)
        action = np.arange(1, 24, dtype=np.float32)[None, :]
        chunk = np.arange(1, 47, dtype=np.float32).reshape(2, 23)
        response = msgpack.packb(
            {"action": action, "action_chunk": chunk, "metadata": "preserved"},
            default=_pack_array,
        )

        with connect(
            f"ws://127.0.0.1:{self.gateway.listen_port}",
            compression=None,
            max_size=None,
        ) as client:
            self.assertEqual(client.recv(), b"openpi-metadata")
            client.send(b"observation")
            self.upstream.wait_for_requests(1)
            self.assertEqual(self.upstream.received, [b"observation"])
            self.upstream.send(response)
            self.assertEqual(client.recv(timeout=2), response)

        result = self.gateway.wait_for_execution(execution_id, progress_timeout_s=2)
        self.assertEqual(result["status"], "step_completed")
        state = self.gateway.snapshot()
        self.assertNotIn("control_mode", state)
        self.assertNotIn("base_motion_enabled", state)
        self.assertEqual(state["upstream_cache_invalidations_sent"], 0)
        self.assertEqual(state["evaluator_resets_forwarded"], 0)

    def test_late_reply_from_a_cancelled_execution_is_not_credited_to_the_next_one(self) -> None:
        execution_a = self.gateway.begin_execution("turn on the radio", 5)
        with self._connect() as client:
            self.assertEqual(client.recv(), b"openpi-metadata")

            client.send(b"observation-1")
            self.upstream.wait_for_requests(1)
            self.upstream.send(b"action:observation-1")
            self.assertEqual(client.recv(timeout=2), b"action:observation-1")

            client.send(b"observation-2")
            self.upstream.wait_for_requests(2)
            self.assertTrue(self.gateway.cancel_execution(execution_a, "test cancellation"))
            result_a = self.gateway.wait_for_execution(execution_a)
            self.assertEqual(result_a["status"], "interrupted")
            # observation-2 was forwarded but never answered, so one reply is owed.
            self.assertEqual(result_a["pending_responses_at_retire"], 1)

            execution_b = self.gateway.begin_execution("turn on the radio", 1)
            client.send(b"observation-3")
            self.upstream.wait_for_requests(3)

            self.upstream.send(b"action:observation-2")
            self.assertEqual(client.recv(timeout=2), b"action:observation-2")
            deadline = time.monotonic() + 2
            while (
                self.gateway.snapshot()["orphaned_responses"] and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            state = self.gateway.snapshot()
            self.assertEqual(state["orphaned_responses"], 0)
            # The stale reply must NOT have consumed execution B's single step.
            self.assertEqual(state["execution"]["execution_id"], execution_b)
            self.assertEqual(state["execution"]["status"], "executing")
            self.assertEqual(state["execution"]["completed_steps"], 0)

            self.upstream.send(b"action:observation-3")
            self.assertEqual(client.recv(timeout=2), b"action:observation-3")
            result_b = self.gateway.wait_for_execution(execution_b, progress_timeout_s=3)
            self.assertEqual(result_b["status"], "step_completed")
            self.assertEqual(result_b["completed_steps"], 1)

    def test_upstream_failure_is_not_reported_as_a_finished_episode(self) -> None:
        execution_id = self.gateway.begin_execution("turn on the radio", 5)
        with self._connect() as client:
            self.assertEqual(client.recv(), b"openpi-metadata")
            client.send(b"observation-1")
            self.upstream.wait_for_requests(1)
            self.upstream.drop()
            result = self.gateway.wait_for_execution(execution_id, progress_timeout_s=5)
            with contextlib.suppress(Exception):
                client.recv(timeout=2)

        self.assertEqual(result["status"], "error")
        self.assertIn("upstream", str(result["detail"]).lower())
        self.assertEqual(result["completed_steps"], 0)

    def test_evaluator_that_vanishes_before_any_step_is_an_error(self) -> None:
        execution_id = self.gateway.begin_execution("turn on the radio", 5)
        with self._connect() as client:
            self.assertEqual(client.recv(), b"openpi-metadata")
        result = self.gateway.wait_for_execution(execution_id, progress_timeout_s=5)
        self.assertEqual(result["status"], "error")
        self.assertIn("before any action", str(result["detail"]))

    def test_duplicate_evaluator_connection_is_rejected(self) -> None:
        import websockets
        from websockets.sync.client import connect

        with connect(
            f"ws://127.0.0.1:{self.gateway.listen_port}", compression=None, max_size=None
        ) as first:
            self.assertEqual(first.recv(), b"openpi-metadata")
            with connect(
                f"ws://127.0.0.1:{self.gateway.listen_port}",
                compression=None,
                max_size=None,
            ) as second:
                with self.assertRaises(websockets.ConnectionClosed) as caught:
                    second.recv(timeout=2)
                self.assertEqual(caught.exception.rcvd.code, 1013)

    def test_progress_timeout_retires_a_wedged_execution(self) -> None:
        execution_id = self.gateway.begin_execution("turn on the radio", 5)
        result = self.gateway.wait_for_execution(
            execution_id, poll_interval_s=0.02, progress_timeout_s=0.3
        )
        self.assertEqual(result["status"], "no_progress")
        self.assertEqual(result["completed_steps"], 0)

    def test_total_timeout_retires_a_long_execution(self) -> None:
        execution_id = self.gateway.begin_execution("turn on the radio", 5)
        result = self.gateway.wait_for_execution(
            execution_id, poll_interval_s=0.02, total_timeout_s=0.3, progress_timeout_s=30
        )
        self.assertEqual(result["status"], "timeout")

    def test_reset_interrupts_the_active_execution_and_bumps_the_episode(self) -> None:
        execution_id = self.gateway.begin_execution("turn on the radio", 5)
        with self._connect() as client:
            self.assertEqual(client.recv(), b"openpi-metadata")
            client.send(b"observation-1")
            self.upstream.wait_for_requests(1)
            self.upstream.send(b"action:observation-1")
            self.assertEqual(client.recv(timeout=2), b"action:observation-1")
            client.send(msgpack.packb({"reset": True}))

            result = self.gateway.wait_for_execution(execution_id, progress_timeout_s=3)
            self.assertEqual(result["status"], "episode_reset")
            self.assertEqual(result["completed_steps"], 1)
        self.assertEqual(self.gateway.snapshot()["episode_id"], 1)

    def test_stop_reports_whether_shutdown_was_confirmed(self) -> None:
        self.gateway.stop(timeout_s=5)
        state = self.gateway.snapshot()
        self.assertTrue(state["stopped_cleanly"])
        self.assertEqual(state["status"], "stopped")

    def test_local_hold_action_bypasses_openpi_and_uses_official_action_shape(self) -> None:
        from websockets.sync.client import connect

        self.gateway.stop()
        with TemporaryDirectory() as temporary:
            self.gateway = B1KProxyGateway(
                upstream_url=f"ws://127.0.0.1:{self.upstream.port}",
                listen_host="127.0.0.1",
                listen_port=0,
                controlled=True,
                observation_workspace=temporary,
                observation_publish_interval=1,
            )
            self.gateway.start()
            with connect(
                f"ws://127.0.0.1:{self.gateway.listen_port}",
                compression=None,
                max_size=None,
            ) as client:
                self.assertEqual(client.recv(), b"openpi-metadata")
                client.send(_r1pro_message())
                deadline = time.monotonic() + 2
                while self.gateway.observation_count < 1 and time.monotonic() < deadline:
                    time.sleep(0.01)
                action = self.gateway.current_hold_action()
                execution_id = self.gateway.begin_local_execution("hold_position", 1, action)
                response = msgpack.unpackb(
                    client.recv(timeout=2), raw=False, object_hook=_unpack_array
                )
                result = self.gateway.wait_for_execution(execution_id, progress_timeout_s=2)

            self.assertEqual(response["action"].shape, (1, 23))
            np.testing.assert_allclose(response["action"][0], action)
            self.assertEqual(self.upstream.request_count(), 0)
            self.assertEqual(result["status"], "step_completed")
            self.assertEqual(result["completed_steps"], 1)
            self.assertEqual(self.gateway.snapshot()["local_actions_generated"], 1)

    def test_vln_handoff_invalidates_only_actions_then_preserves_consecutive_vla(self) -> None:
        from websockets.sync.client import connect

        self.gateway.stop()
        self.upstream.metadata = msgpack.packb({
            "emerge_capabilities": {"invalidate_action_cache": True}, "model": "test-champion",
        })
        with TemporaryDirectory() as temporary:
            self.gateway = B1KProxyGateway(
                upstream_url=f"ws://127.0.0.1:{self.upstream.port}",
                listen_host="127.0.0.1",
                listen_port=0,
                controlled=True,
                observation_workspace=temporary,
                observation_publish_interval=1,
            )
            self.gateway.start()
            with connect(
                f"ws://127.0.0.1:{self.gateway.listen_port}",
                compression=None,
                max_size=None,
            ) as client:
                self.assertEqual(client.recv(), self.upstream.metadata)
                client.send(_r1pro_message())
                deadline = time.monotonic() + 2
                while self.gateway.observation_count < 1 and time.monotonic() < deadline:
                    time.sleep(0.01)
                local_id = self.gateway.begin_local_execution(
                    "navigate_vln", 1, self.gateway.current_hold_action()
                )
                client.recv(timeout=2)
                self.assertEqual(
                    self.gateway.wait_for_execution(local_id, progress_timeout_s=2)["status"],
                    "step_completed",
                )
                self.assertTrue(self.gateway.snapshot()["upstream_needs_action_cache_invalidation"])

                vla_id = self.gateway.begin_execution("turn on the radio", 1)
                client.send(_r1pro_message())
                self.upstream.wait_for_requests(2)
                control = msgpack.unpackb(self.upstream.received[0], raw=False)
                self.assertEqual(control, {"invalidate_action_cache": True})
                response = msgpack.packb(
                    {"action": np.arange(1, 24, dtype=np.float32)[None, :]}, default=_pack_array
                )
                self.upstream.send(response)
                self.assertEqual(client.recv(timeout=2), response)
                result = self.gateway.wait_for_execution(vla_id, progress_timeout_s=2)
                self.assertEqual(result["status"], "step_completed")

                next_vla = self.gateway.begin_execution("turn on the radio", 1)
                client.send(_r1pro_message())
                self.upstream.wait_for_requests(3)
                self.assertEqual(self.upstream.received[2], _r1pro_message())
                self.upstream.send(response)
                self.assertEqual(client.recv(timeout=2), response)
                result = self.gateway.wait_for_execution(next_vla, progress_timeout_s=2)

            self.assertEqual(result["status"], "step_completed")
            self.assertFalse(self.gateway.snapshot()["upstream_needs_action_cache_invalidation"])
            self.assertEqual(self.gateway.snapshot()["upstream_cache_invalidations_sent"], 1)
            self.assertEqual(self.gateway.snapshot()["evaluator_resets_forwarded"], 0)

    def test_legacy_upstream_cannot_silently_reset_stages_after_local_motion(self) -> None:
        from websockets.exceptions import ConnectionClosed

        local_id = self.gateway.begin_local_execution("navigate_vln", 1, [0.0] * 23)
        with self._connect() as client:
            self.assertEqual(client.recv(), b"openpi-metadata")
            client.send(b"initial-observation")
            client.recv(timeout=2)
            self.gateway.wait_for_execution(local_id, progress_timeout_s=2)
            self.assertEqual(self.upstream.request_count(), 0)

            vla_id = self.gateway.begin_execution("turn on the radio", 1)
            client.send(b"post-navigation-observation")
            with self.assertRaises(ConnectionClosed):
                client.recv(timeout=3)
            result = self.gateway.wait_for_execution(vla_id, progress_timeout_s=2)
        self.assertEqual(result["status"], "error")
        self.assertIn("invalidate_action_cache", result["detail"])
        self.assertEqual(self.upstream.received, [])
        self.assertEqual(self.gateway.snapshot()["upstream_cache_invalidations_sent"], 0)

    def test_official_episode_reset_clears_pending_invalidation_without_extra_command(self) -> None:
        local_id = self.gateway.begin_local_execution("navigate_vln", 1, [0.0] * 23)
        with self._connect() as client:
            client.recv()
            client.send(b"initial-observation")
            client.recv(timeout=2)
            self.gateway.wait_for_execution(local_id, progress_timeout_s=2)
            client.send(msgpack.packb({"reset": True}))
            self.upstream.wait_for_requests(1)
            self.assertEqual(msgpack.unpackb(self.upstream.received[0], raw=False), {"reset": True})
            vla_id = self.gateway.begin_execution("turn on the radio", 1)
            client.send(b"new-episode-observation")
            self.upstream.wait_for_requests(2)
            self.assertEqual(self.upstream.received[1], b"new-episode-observation")
            self.upstream.send(b"new-action")
            self.assertEqual(client.recv(timeout=2), b"new-action")
            self.gateway.wait_for_execution(vla_id, progress_timeout_s=2)
        state = self.gateway.snapshot()
        self.assertEqual(state["evaluator_resets_forwarded"], 1)
        self.assertEqual(state["upstream_cache_invalidations_sent"], 0)
        self.assertFalse(state["upstream_needs_action_cache_invalidation"])

    def test_local_trajectory_returns_each_action_in_order(self) -> None:
        from websockets.sync.client import connect

        trajectory = [[float(index)] * 23 for index in range(3)]
        execution_id = self.gateway.begin_local_trajectory("test trajectory", trajectory)
        received = []
        with self._connect() as client:
            self.assertEqual(client.recv(), b"openpi-metadata")
            for index in range(3):
                client.send(f"observation-{index}".encode())
                response = msgpack.unpackb(
                    client.recv(timeout=2), raw=False, object_hook=_unpack_array
                )
                received.append(response["action"])

        result = self.gateway.wait_for_execution(execution_id, progress_timeout_s=2)
        self.assertEqual(result["completed_steps"], 3)
        for index, action in enumerate(received):
            np.testing.assert_allclose(action, np.asarray(trajectory[index])[None, :])


class ObservationSynchronizationTest(unittest.TestCase):
    """The gateway must expose the frame that proves an action was applied."""

    def setUp(self) -> None:
        self.upstream = _ControllableOpenPIServer()
        self.upstream.start()
        self.temporary = TemporaryDirectory()
        self.gateway = B1KProxyGateway(
            upstream_url=f"ws://127.0.0.1:{self.upstream.port}",
            listen_host="127.0.0.1",
            listen_port=0,
            observation_workspace=self.temporary.name,
            observation_publish_interval=100,
        )
        self.gateway.start()

    def tearDown(self) -> None:
        self.gateway.stop()
        self.upstream.stop()
        self.temporary.cleanup()

    def test_wait_for_observation_after_only_returns_for_a_newer_frame(self) -> None:
        from websockets.sync.client import connect

        with connect(
            f"ws://127.0.0.1:{self.gateway.listen_port}", compression=None, max_size=None
        ) as client:
            self.assertEqual(client.recv(), b"openpi-metadata")
            baseline = self.gateway.observation_count
            self.assertFalse(
                self.gateway.wait_for_observation_after(baseline, timeout_s=0.3)
            )
            client.send(_camera_message(seed=7))
            self.upstream.wait_for_requests(1)
            self.assertTrue(
                self.gateway.wait_for_observation_after(baseline, timeout_s=3)
            )
            self.assertFalse(
                self.gateway.wait_for_observation_after(
                    self.gateway.observation_count, timeout_s=0.3
                )
            )

    def test_forced_publish_makes_the_frame_visible_without_waiting_for_the_interval(
        self,
    ) -> None:
        from websockets.sync.client import connect

        manifest = Path(self.temporary.name) / "artifacts/observations/observation.json"
        with connect(
            f"ws://127.0.0.1:{self.gateway.listen_port}", compression=None, max_size=None
        ) as client:
            self.assertEqual(client.recv(), b"openpi-metadata")
            client.send(_camera_message(seed=3))
            self.upstream.wait_for_requests(1)
            self.assertTrue(manifest.is_file())
            first_revision = json.loads(manifest.read_text(encoding="utf-8"))["revision"]

            client.send(_camera_message(seed=9))
            self.upstream.wait_for_requests(2)
            # The publish interval is 100 frames, so nothing new is published yet.
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8"))["revision"],
                first_revision,
            )
            self.assertIsNotNone(self.gateway.publish_observation())
            self.assertEqual(
                json.loads(manifest.read_text(encoding="utf-8"))["revision"],
                first_revision + 1,
            )


if __name__ == "__main__":
    unittest.main()

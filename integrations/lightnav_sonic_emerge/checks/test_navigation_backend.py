from __future__ import annotations

from types import SimpleNamespace
import tempfile
import unittest

from integrations.lightnav_sonic_emerge.navigation_backend import (
    NavigationBackend,
    NavigationConfig,
)
from integrations.lightnav_sonic_emerge.observation_writer import EgoObservationWriter


class _SequenceSource:
    def __init__(self, values):
        self.values = list(values)
        self.closed = False

    def latest(self, _max_age):
        if len(self.values) > 1:
            return self.values.pop(0)
        return self.values[0]

    def close(self):
        self.closed = True


class _Publisher:
    def __init__(self):
        self.activated = False
        self.commands = []
        self.idles = []
        self.closed = False

    def activate_planner(self):
        self.activated = True

    def send_planner(self, command):
        self.commands.append(command)

    def send_idle(self, command):
        self.idles.append(command)

    def close(self):
        self.closed = True


class _Client:
    def __init__(self, replies):
        self.replies = list(replies)
        self.connected = False
        self.closed = False

    def connect(self):
        self.connected = True

    def predict(self, _jpeg, _instruction):
        if len(self.replies) > 1:
            return self.replies.pop(0)
        return self.replies[0]

    def close(self):
        self.closed = True


class _Adapter:
    def convert_or_idle(self, waypoint, yaw):
        return SimpleNamespace(mode=1, waypoint=waypoint, yaw=yaw)

    def idle(self, yaw, *, reason):
        return SimpleNamespace(mode=0, yaw=yaw, reason=reason)


class _Writer:
    def __init__(self):
        self.images = []

    def write_jpeg(self, jpeg):
        self.images.append(jpeg)


def _reply(*, stop=False, waypoint=(0.2, 0.0, 0.0)):
    return SimpleNamespace(stop=stop, first_waypoint=waypoint)


class NavigationBackendTest(unittest.TestCase):
    def _backend(self, camera, yaw, publisher, client, writer=None, **config_overrides):
        values = dict(
            lightnav_server="ws://test:8051",
            control_hz=1000.0,
            max_stale_ticks=2,
        )
        values.update(config_overrides)
        return NavigationBackend(
            NavigationConfig(**values),
            writer or _Writer(),
            camera_factory=lambda *_: camera,
            yaw_factory=lambda *_: yaw,
            publisher_factory=lambda *_: publisher,
            client_factory=lambda *_: client,
            adapter_factory=lambda *_: _Adapter(),
            sleep=lambda _seconds: None,
        )

    def test_lightnav_stop_completes_and_finally_sends_idle(self) -> None:
        camera = _SequenceSource([b"jpeg"])
        yaw = _SequenceSource([0.4])
        publisher = _Publisher()
        client = _Client([_reply(stop=True)])
        writer = _Writer()
        result = self._backend(camera, yaw, publisher, client, writer).run(
            instruction="go to the table", max_steps=5
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.reason, "lightnav_stop")
        self.assertTrue(result.lightnav_stop)
        self.assertEqual(writer.images, [b"jpeg"])
        self.assertGreaterEqual(len(publisher.idles), 1)
        self.assertTrue(camera.closed and yaw.closed and publisher.closed and client.closed)

    def test_budget_exhaustion_is_bounded(self) -> None:
        publisher = _Publisher()
        result = self._backend(
            _SequenceSource([b"jpeg"]),
            _SequenceSource([0.0]),
            publisher,
            _Client([_reply()]),
        ).run(instruction="walk forward", max_steps=3)

        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(result.ticks, 3)
        self.assertEqual(result.command_steps, 3)
        self.assertEqual(len(publisher.commands), 3)
        self.assertGreaterEqual(len(publisher.idles), 1)

    def test_cancel_check_stops_before_a_command(self) -> None:
        publisher = _Publisher()
        result = self._backend(
            _SequenceSource([b"jpeg"]),
            _SequenceSource([0.0]),
            publisher,
            _Client([_reply()]),
        ).run(
            instruction="walk forward",
            max_steps=3,
            cancel_check=lambda: "user stop",
        )

        self.assertEqual(result.status, "cancelled")
        self.assertEqual(result.reason, "user stop")
        self.assertEqual(result.ticks, 0)
        self.assertFalse(publisher.commands)
        self.assertGreaterEqual(len(publisher.idles), 1)

    def test_repeated_stale_input_fails_closed(self) -> None:
        publisher = _Publisher()
        result = self._backend(
            _SequenceSource([None]),
            _SequenceSource([0.0]),
            publisher,
            _Client([_reply()]),
        ).run(instruction="walk forward", max_steps=10)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.reason, "stale_input")
        self.assertEqual(result.ticks, 2)
        self.assertFalse(publisher.commands)
        self.assertGreaterEqual(len(publisher.idles), 2)

    def test_release_file_created_only_after_valid_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            release = f"{directory}/release"
            backend = self._backend(
                _SequenceSource([b"jpeg"]),
                _SequenceSource([0.0]),
                _Publisher(),
                _Client([_reply(stop=True)]),
                elastic_band_release_file=release,
            )
            backend.run(instruction="walk forward", max_steps=1)
            self.assertTrue(backend.config.elastic_band_release_file.exists())

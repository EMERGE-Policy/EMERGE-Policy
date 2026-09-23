"""SONIC ZMQManager planner wire protocol.

The packet format matches the checked-in SONIC deployment implementation:
``[topic][1280-byte JSON header][little-endian payload]``.  This module uses
only the upstream-standard ``mode``, ``movement``, ``facing``, ``speed``, and
``height`` fields.  It intentionally does not use any local, uncommitted
``body_relative`` protocol extension.
"""

from __future__ import annotations

import json
import math
import struct
import time
from typing import Any, Sequence

from .adapter import SONIC_IDLE_MODE, SonicPlannerCommand


HEADER_SIZE = 1280


def _finite_vector(values: Sequence[float], name: str, *, allow_zero: bool) -> tuple[float, float, float]:
    if isinstance(values, (str, bytes)) or len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain finite values")
    if not allow_zero and math.sqrt(sum(value * value for value in result)) < 1e-8:
        raise ValueError(f"{name} must be non-zero")
    return result  # type: ignore[return-value]


def _header(fields: list[dict[str, Any]]) -> bytes:
    raw = json.dumps(
        {"v": 1, "endian": "le", "count": 1, "fields": fields},
        separators=(",", ":"),
    ).encode("utf-8")
    if len(raw) > HEADER_SIZE:
        raise ValueError(f"SONIC ZMQ header exceeds {HEADER_SIZE} bytes")
    return raw.ljust(HEADER_SIZE, b"\x00")


def build_command_packet(*, start: bool, stop: bool, planner: bool) -> bytes:
    """Build the SONIC ``command`` packet used to open planner mode.

    The bridge calls this only with ``stop=False``.  SONIC's command-level stop
    terminates deployment and is not a normal navigation halt.
    """

    fields = [
        {"name": "start", "dtype": "u8", "shape": [1]},
        {"name": "stop", "dtype": "u8", "shape": [1]},
        {"name": "planner", "dtype": "u8", "shape": [1]},
    ]
    return b"command" + _header(fields) + struct.pack("BBB", int(start), int(stop), int(planner))


def build_planner_packet(command: SonicPlannerCommand) -> bytes:
    """Encode one standard world-frame SONIC planner command."""

    movement = _finite_vector(command.movement_direction, "movement_direction", allow_zero=True)
    facing = _finite_vector(command.facing_direction, "facing_direction", allow_zero=False)
    speed = float(command.movement_speed)
    height = float(command.height)
    if not math.isfinite(speed) or not math.isfinite(height):
        raise ValueError("movement_speed and height must be finite")
    fields = [
        {"name": "mode", "dtype": "i32", "shape": [1]},
        {"name": "movement", "dtype": "f32", "shape": [3]},
        {"name": "facing", "dtype": "f32", "shape": [3]},
        {"name": "speed", "dtype": "f32", "shape": [1]},
        {"name": "height", "dtype": "f32", "shape": [1]},
    ]
    payload = b"".join(
        (
            struct.pack("<i", int(command.mode)),
            struct.pack("<fff", *movement),
            struct.pack("<fff", *facing),
            struct.pack("<f", speed),
            struct.pack("<f", height),
        )
    )
    return b"planner" + _header(fields) + payload


class SonicZmqPublisher:
    """A single-owner PUB socket for SONIC's ``ZMQManager`` input.

    Start the deployment with ``--input-type zmq_manager`` and configure it to
    connect to this publisher's bound endpoint.  The adapter owns this socket;
    do not run it alongside another publisher bound to the same port.
    """

    def __init__(self, bind_endpoint: str = "tcp://*:5556") -> None:
        try:
            import zmq
        except ImportError as exc:  # pragma: no cover - deployment-only dependency
            raise RuntimeError("pyzmq is required for SONIC ZMQ publishing") from exc
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(bind_endpoint)
        self.bind_endpoint = bind_endpoint
        self._activated = False

    @property
    def activated(self) -> bool:
        return self._activated

    def activate_planner(self, *, warmup_s: float = 0.5, repeats: int = 3) -> None:
        """Allow subscribers to connect, then request SONIC planner mode."""

        time.sleep(max(0.0, warmup_s))
        packet = build_command_packet(start=True, stop=False, planner=True)
        for _ in range(max(1, int(repeats))):
            self._socket.send(packet)
        self._activated = True

    def send_planner(self, command: SonicPlannerCommand) -> None:
        if not self._activated:
            raise RuntimeError("SONIC planner is inactive; call activate_planner first")
        self._socket.send(build_planner_packet(command))

    def send_idle(self, command: SonicPlannerCommand) -> None:
        """Send locomotion IDLE, never a deployment-level stop command."""

        if command.mode != SONIC_IDLE_MODE:
            raise ValueError("send_idle requires a SONIC IDLE command")
        self.send_planner(command)

    def close(self) -> None:
        self._socket.close()
        self._context.term()

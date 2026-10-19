"""Small synchronous client for the published LightNav WebSocket protocol."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import math
from numbers import Real
from typing import Any, Mapping, Sequence


class LightNavProtocolError(RuntimeError):
    """A server reply cannot safely be used as a navigation command."""


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise LightNavProtocolError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LightNavProtocolError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class LightNavReply:
    """Validated inference result from one ``next`` request."""

    seq: int
    waypoints: tuple[tuple[float, float, float], ...]
    stop: bool
    visible: bool | None = None
    latency_ms: float | None = None
    raw_text: str | None = None

    @property
    def first_waypoint(self) -> tuple[float, float, float]:
        if not self.waypoints:
            raise LightNavProtocolError("LightNav response has no waypoint")
        return self.waypoints[0]


def _response_data(payload: Mapping[str, Any], expected_action: str) -> Mapping[str, Any]:
    action = payload.get("action")
    if action != expected_action:
        raise LightNavProtocolError(f"expected response action {expected_action!r}, got {action!r}")
    data = payload.get("data")
    if not isinstance(data, Mapping):
        raise LightNavProtocolError("LightNav response data must be an object")
    rc = data.get("rc")
    if not _is_int(rc):
        raise LightNavProtocolError("LightNav response rc must be an integer")
    if rc != 0:
        message = data.get("msg", "server returned an error")
        raise LightNavProtocolError(f"LightNav server rc={rc}: {message}")
    return data


def parse_next_response(payload: Mapping[str, Any], *, expected_seq: int | None = None) -> LightNavReply:
    """Validate a successful LightNav ``next`` response.

    This intentionally rejects partial, non-finite, or out-of-order output so
    callers can issue SONIC IDLE instead of reusing a previous command.
    """

    data = _response_data(payload, "next")
    seq = data.get("seq")
    if not _is_int(seq) or seq < 0:
        raise LightNavProtocolError("LightNav response seq must be a non-negative integer")
    if expected_seq is not None and seq != expected_seq:
        raise LightNavProtocolError(f"LightNav response sequence mismatch: expected {expected_seq}, got {seq}")
    stop = data.get("stop")
    if not isinstance(stop, bool):
        raise LightNavProtocolError("LightNav response stop must be boolean")
    visible = data.get("visible")
    if visible is not None and not isinstance(visible, bool):
        raise LightNavProtocolError("LightNav response visible must be boolean or null")
    actions = data.get("actions")
    if not isinstance(actions, Mapping):
        raise LightNavProtocolError("LightNav response actions must be an object")
    raw_waypoints = actions.get("actions")
    if isinstance(raw_waypoints, (str, bytes)) or not isinstance(raw_waypoints, Sequence):
        raise LightNavProtocolError("LightNav actions.actions must be a waypoint sequence")
    if not raw_waypoints:
        raise LightNavProtocolError("LightNav response actions.actions is empty")
    waypoints: list[tuple[float, float, float]] = []
    for row_index, row in enumerate(raw_waypoints):
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence) or len(row) != 3:
            raise LightNavProtocolError(f"waypoint {row_index} must have exactly three values")
        waypoints.append(
            (
                _finite_float(row[0], f"waypoint {row_index} forward"),
                _finite_float(row[1], f"waypoint {row_index} lateral"),
                _finite_float(row[2], f"waypoint {row_index} yaw"),
            )
        )
    latency = data.get("latency_ms")
    latency_ms = None if latency is None else _finite_float(latency, "latency_ms")
    raw_text = data.get("raw_text")
    if raw_text is not None and not isinstance(raw_text, str):
        raise LightNavProtocolError("raw_text must be text when present")
    return LightNavReply(
        seq=seq,
        waypoints=tuple(waypoints),
        stop=stop,
        visible=visible,
        latency_ms=latency_ms,
        raw_text=raw_text,
    )


class LightNavWsClient:
    """One ordered LightNav WebSocket session.

    The client uses raw JPEG bytes from the robot camera.  It does not need
    torch or the LightNav Python inference package, allowing it to run in the
    SONIC environment while the model remains on a separate GPU host.
    """

    def __init__(
        self,
        server_url: str,
        *,
        client_id: str = "lightnav-sonic-g1",
        request_timeout_s: float | None = None,
    ) -> None:
        if request_timeout_s is not None:
            if isinstance(request_timeout_s, bool):
                raise ValueError("request_timeout_s must be positive and finite")
            try:
                request_timeout_s = float(request_timeout_s)
            except (TypeError, ValueError) as exc:
                raise ValueError("request_timeout_s must be positive and finite") from exc
            if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
                raise ValueError("request_timeout_s must be positive and finite")
        self.server_url = server_url
        self.client_id = client_id
        self.request_timeout_s = request_timeout_s
        self._socket: Any | None = None
        self._seq = 0

    def connect(self) -> None:
        if self._socket is not None:
            return
        try:
            from websockets.sync.client import connect
        except ImportError as exc:  # pragma: no cover - deployment-only dependency
            raise RuntimeError("websockets>=12 is required for the LightNav client") from exc
        options = {"open_timeout": self.request_timeout_s} if self.request_timeout_s is not None else {}
        self._socket = connect(self.server_url, max_size=64 * 1024 * 1024, **options)
        try:
            self._send_expect_ok("login", {"clientId": self.client_id})
            self._send_expect_ok("reset", {})
        except Exception:
            self.close()
            raise
        self._seq = 0

    def _send_expect_ok(self, action: str, data: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._socket is None:
            raise RuntimeError("LightNav WebSocket is not connected")
        self._socket.send(json.dumps({"action": action, "data": dict(data)}, separators=(",", ":")))
        raw = self._socket.recv(timeout=self.request_timeout_s)
        if not isinstance(raw, str):
            raise LightNavProtocolError("LightNav server sent a binary WebSocket frame")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LightNavProtocolError("LightNav server sent invalid JSON") from exc
        if not isinstance(payload, Mapping):
            raise LightNavProtocolError("LightNav server JSON response must be an object")
        return _response_data(payload, action)

    def predict(self, jpeg_bytes: bytes, instruction: str) -> LightNavReply:
        """Submit one RGB frame and return a validated prediction."""

        if not isinstance(jpeg_bytes, bytes) or not jpeg_bytes:
            raise ValueError("camera frame must be non-empty JPEG bytes")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be non-empty text")
        if self._socket is None:
            self.connect()
        sequence = self._seq
        self._seq += 1
        data = self._send_expect_ok(
            "next",
            {
                "seq": sequence,
                "image": base64.b64encode(jpeg_bytes).decode("ascii"),
                "instruction": instruction,
            },
        )
        return parse_next_response({"action": "next", "data": data}, expected_seq=sequence)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self) -> "LightNavWsClient":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

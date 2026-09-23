"""Run the LightNav → SONIC navigation bridge against SONIC Sim2Sim.

The process is intentionally a thin, standalone robot client:

* subscribes to the upstream MuJoCo image publisher (default port 5555),
* subscribes to the upstream SONIC ``g1_debug`` feedback topic for base yaw,
* calls a remote ``lightnav-serve`` WebSocket server, and
* owns the sole SONIC planner PUB socket (default port 5556).

It never imports or edits SONIC.  ``--enable-motion`` is deliberately required
before planner activation or packet transmission.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from dataclasses import asdict
import json
import math
from numbers import Real
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .adapter import AdapterLimits, LightNav2SonicAdapter
from .lightnav_ws import LightNavWsClient
from .sonic_zmq import SonicZmqPublisher


def extract_jpeg_from_camera_message(payload: Mapping[str, Any], camera_name: str) -> bytes:
    """Extract JPEG bytes from SONIC's camera-publisher msgpack payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("camera payload must be an object")
    candidate = payload.get(camera_name)
    if candidate is None:
        images = payload.get("images")
        if isinstance(images, Mapping):
            candidate = images.get(camera_name)
    if isinstance(candidate, bytes):
        if not candidate:
            raise ValueError(f"camera {camera_name!r} published empty bytes")
        return candidate
    if not isinstance(candidate, str) or not candidate:
        raise ValueError(f"camera payload does not contain non-empty {camera_name!r} JPEG")
    try:
        return base64.b64decode(candidate, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"camera {camera_name!r} JPEG is not valid base64") from exc


def yaw_from_wxyz(quaternion: Sequence[float]) -> float:
    """Return SONIC/MuJoCo world yaw from a finite wxyz quaternion."""

    if isinstance(quaternion, (str, bytes)) or len(quaternion) != 4:
        raise ValueError("base_quat must contain four wxyz values")
    values: list[float] = []
    for value in quaternion:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("base_quat values must be finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("base_quat values must be finite numbers")
        values.append(number)
    w, x, y, z = values
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-8:
        raise ValueError("base_quat has zero norm")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class SonicCameraSubscriber:
    """Conflated subscriber for SONIC's msgpack camera publisher."""

    def __init__(self, endpoint: str, camera_name: str) -> None:
        try:
            import msgpack
            import zmq
        except ImportError as exc:  # pragma: no cover - deployment-only dependency
            raise RuntimeError("runner requires pyzmq and msgpack") from exc
        self._msgpack = msgpack
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.connect(endpoint)
        self._camera_name = camera_name
        self._jpeg: bytes | None = None
        self._received_at: float | None = None

    def poll(self) -> None:
        while True:
            try:
                raw = self._socket.recv(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                return
            payload = self._msgpack.unpackb(raw, raw=False, strict_map_key=False)
            try:
                self._jpeg = extract_jpeg_from_camera_message(payload, self._camera_name)
                self._received_at = time.monotonic()
            except (TypeError, ValueError):
                # A malformed camera frame is discarded; its existing value is
                # allowed to expire naturally and then produces SONIC IDLE.
                continue

    def latest(self, max_age_s: float) -> bytes | None:
        self.poll()
        if self._jpeg is None or self._received_at is None:
            return None
        if time.monotonic() - self._received_at > max_age_s:
            return None
        return self._jpeg

    def close(self) -> None:
        self._socket.close()
        self._context.term()


class SonicYawSubscriber:
    """Conflated subscriber for SONIC's topic-prefixed msgpack feedback."""

    def __init__(self, endpoint: str, topic: str) -> None:
        try:
            import msgpack
            import zmq
        except ImportError as exc:  # pragma: no cover - deployment-only dependency
            raise RuntimeError("runner requires pyzmq and msgpack") from exc
        self._msgpack = msgpack
        self._zmq = zmq
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._topic = topic.encode("utf-8")
        self._socket.setsockopt(zmq.SUBSCRIBE, self._topic)
        self._socket.connect(endpoint)
        self._yaw: float | None = None
        self._received_at: float | None = None

    def poll(self) -> None:
        while True:
            try:
                raw = self._socket.recv(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                return
            if not raw.startswith(self._topic):
                continue
            payload = self._msgpack.unpackb(raw[len(self._topic) :], raw=False, strict_map_key=False)
            if not isinstance(payload, Mapping):
                continue
            try:
                self._yaw = yaw_from_wxyz(payload.get("base_quat"))
                self._received_at = time.monotonic()
            except (TypeError, ValueError):
                continue

    def latest(self, max_age_s: float) -> float | None:
        self.poll()
        if self._yaw is None or self._received_at is None:
            return None
        if time.monotonic() - self._received_at > max_age_s:
            return None
        return self._yaw

    def close(self) -> None:
        self._socket.close()
        self._context.term()


def _command_log(command: Any, *, waypoint: Sequence[float] | None, robot_yaw: float | None) -> str:
    payload = {
        "waypoint": None if waypoint is None else list(waypoint),
        "robot_yaw_rad": robot_yaw,
        "sonic_command": asdict(command),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _safe_idle(publisher: SonicZmqPublisher | None, adapter: LightNav2SonicAdapter, yaw: float | None, reason: str) -> None:
    if publisher is None or not publisher.activated:
        return
    command = adapter.idle(yaw, reason=reason)
    try:
        publisher.send_idle(command)
    except Exception as exc:  # pragma: no cover - physical transport failure
        print(f"failed to send SONIC IDLE after {reason}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--lightnav-server", help="LightNav WebSocket endpoint, e.g. ws://gpu-host:8051")
    source.add_argument("--mock-waypoint", nargs=3, type=float, metavar=("FORWARD_M", "LATERAL_M", "YAW_RAD"))
    parser.add_argument("--instruction", default="walk forward", help="language instruction for LightNav")
    parser.add_argument("--client-id", default="lightnav-sonic-g1")
    parser.add_argument("--camera-endpoint", default="tcp://127.0.0.1:5555")
    parser.add_argument("--camera-name", default="ego_view")
    parser.add_argument("--sonic-feedback-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--sonic-feedback-topic", default="g1_debug")
    parser.add_argument("--sonic-bind", default="tcp://*:5556")
    parser.add_argument("--control-hz", type=float, default=4.0)
    parser.add_argument("--input-timeout-s", type=float, default=0.75)
    parser.add_argument("--max-steps", type=int, default=0, help="0 runs until LightNav stop or Ctrl-C")
    parser.add_argument("--enable-motion", action="store_true", help="explicitly activate SONIC planner and publish commands")
    parser.add_argument(
        "--elastic-band-release-file",
        type=Path,
        help=(
            "create this bridge-local signal after SONIC yaw feedback is live; "
            "use it with room_sim --elastic-band-release-file"
        ),
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if not math.isfinite(args.control_hz) or args.control_hz <= 0.0:
        raise ValueError("--control-hz must be positive")
    if not math.isfinite(args.input_timeout_s) or args.input_timeout_s <= 0.0:
        raise ValueError("--input-timeout-s must be positive")
    if args.max_steps < 0:
        raise ValueError("--max-steps cannot be negative")
    if args.mock_waypoint is not None and args.enable_motion and args.max_steps == 0:
        raise ValueError("mock motion requires a finite --max-steps safety limit")

    adapter = LightNav2SonicAdapter(AdapterLimits(control_period_s=1.0 / args.control_hz))
    camera = SonicCameraSubscriber(args.camera_endpoint, args.camera_name)
    yaw_source = SonicYawSubscriber(args.sonic_feedback_endpoint, args.sonic_feedback_topic)
    publisher: SonicZmqPublisher | None = None
    client: LightNavWsClient | None = None
    elastic_band_release_file = args.elastic_band_release_file
    if elastic_band_release_file is not None:
        elastic_band_release_file = elastic_band_release_file.resolve()
        if elastic_band_release_file.exists():
            raise FileExistsError(
                f"elastic-band release file already exists: {elastic_band_release_file}; "
                "remove this stale signal before starting the runner"
            )
    period_s = 1.0 / args.control_hz
    step = 0
    try:
        if args.enable_motion:
            publisher = SonicZmqPublisher(args.sonic_bind)
            print(f"SONIC planner publisher bound at {args.sonic_bind}; waiting for subscriber")
            publisher.activate_planner()
            print("SONIC planner mode activated; normal stop sends locomotion IDLE only")
        if args.lightnav_server:
            client = LightNavWsClient(args.lightnav_server, client_id=args.client_id)
            client.connect()
            print(f"connected to LightNav server {args.lightnav_server}")

        while args.max_steps == 0 or step < args.max_steps:
            tick_started = time.monotonic()
            jpeg = camera.latest(args.input_timeout_s)
            yaw = yaw_source.latest(args.input_timeout_s)
            if jpeg is None or yaw is None:
                missing = "camera" if jpeg is None else "robot yaw feedback"
                _safe_idle(publisher, adapter, yaw, f"stale or missing {missing}")
                print(f"idle: stale or missing {missing}")
            else:
                try:
                    if elastic_band_release_file is not None:
                        elastic_band_release_file.touch(exist_ok=False)
                        print(f"released room ElasticBand via {elastic_band_release_file}")
                        elastic_band_release_file = None
                    if args.mock_waypoint is not None:
                        waypoint = tuple(args.mock_waypoint)
                        stop = False
                    else:
                        assert client is not None
                        reply = client.predict(jpeg, args.instruction)
                        waypoint = reply.first_waypoint
                        stop = reply.stop
                    if stop:
                        command = adapter.idle(yaw, reason="LightNav stop=true")
                        _safe_idle(publisher, adapter, yaw, command.reason)
                        print(_command_log(command, waypoint=waypoint, robot_yaw=yaw))
                        return 0
                    command = adapter.convert_or_idle(waypoint, yaw)
                    if publisher is not None and publisher.activated:
                        publisher.send_planner(command)
                    print(_command_log(command, waypoint=waypoint, robot_yaw=yaw))
                    step += 1
                # Any recoverable bridge failure must degrade to a planner-level
                # IDLE packet. KeyboardInterrupt is deliberately not caught, so
                # normal shutdown still reaches the finally block below.
                except Exception as exc:
                    _safe_idle(publisher, adapter, yaw, f"bridge failure: {exc}")
                    print(f"idle: bridge failure: {exc}")

            remaining = period_s - (time.monotonic() - tick_started)
            if remaining > 0.0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("interrupted; sending SONIC locomotion IDLE")
        return 130
    finally:
        _safe_idle(publisher, adapter, yaw_source.latest(args.input_timeout_s), "runner shutdown")
        if client is not None:
            client.close()
        if publisher is not None:
            publisher.close()
        camera.close()
        yaw_source.close()
    return 0


def main() -> int:
    return run(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

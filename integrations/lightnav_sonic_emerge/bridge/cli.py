"""Offline LightNav waypoint to SONIC command inspector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .adapter import LightNav2SonicAdapter
from .lightnav_ws import LightNavProtocolError, parse_next_response


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--waypoint", nargs=3, type=float, metavar=("FORWARD_M", "LATERAL_M", "YAW_RAD"))
    source.add_argument("--response-json", type=Path, help="saved complete LightNav WebSocket response")
    parser.add_argument("--robot-yaw", type=float, required=True, help="current SONIC world yaw in radians")
    return parser


def _load_response(path: Path) -> Mapping[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read response JSON: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"response JSON is invalid: {exc.msg}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("response JSON must be an object")
    return raw


def main() -> int:
    args = build_parser().parse_args()
    adapter = LightNav2SonicAdapter()
    waypoint: tuple[float, float, float] | None = None
    if args.waypoint is not None:
        waypoint = tuple(args.waypoint)
    else:
        try:
            response = parse_next_response(_load_response(args.response_json))
            if response.stop:
                command = adapter.idle(args.robot_yaw, reason="LightNav stop=true")
                print(json.dumps(command.to_dict(), ensure_ascii=False, indent=2))
                return 0
            waypoint = response.first_waypoint
        except (LightNavProtocolError, ValueError) as exc:
            command = adapter.idle(args.robot_yaw, reason=f"invalid LightNav response: {exc}")
            print(json.dumps(command.to_dict(), ensure_ascii=False, indent=2))
            return 2
    command = adapter.convert_or_idle(waypoint, args.robot_yaw)
    print(json.dumps(command.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Probe LightNav-0 with a published BEHAVIOR head-camera observation.

This entrypoint only reads an existing observation and requests a waypoint. It
never starts the evaluator or sends actions to the R1Pro gateway.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any

from PIL import Image

from integrations.lightnav_sonic_emerge.bridge.lightnav_ws import LightNavWsClient


@dataclass(frozen=True)
class HeadFrame:
    jpeg: bytes
    width: int
    height: int
    image_path: Path
    revision: int | None
    observation_seq: int | None
    episode_id: int | None


def load_head_frame(manifest_path: Path) -> HeadFrame:
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    if manifest_path.name != "observation.json" or manifest_path.parent.name != "observations":
        raise ValueError("expected a BEHAVIOR artifacts/observations/observation.json path")
    workspace = manifest_path.parents[2]
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("observation manifest must be an object")
    views = manifest.get("views")
    if not isinstance(views, list):
        raise ValueError("observation manifest has no views")
    head = next((view for view in views if isinstance(view, dict) and view.get("name") == "head"), None)
    if head is None:
        raise ValueError("observation manifest has no head camera")
    relative_path = head.get("image_path")
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("head view has no image_path")
    image_path = (workspace / relative_path).resolve()
    if not image_path.is_relative_to(workspace):
        raise ValueError("head image is outside the observation workspace")
    image_path = image_path.resolve(strict=True)

    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=90)
    if (head.get("width"), head.get("height")) != (width, height):
        raise ValueError("head image dimensions differ from the observation manifest")
    return HeadFrame(
        jpeg=buffer.getvalue(),
        width=width,
        height=height,
        image_path=image_path,
        revision=manifest.get("revision"),
        observation_seq=manifest.get("observation_seq"),
        episode_id=manifest.get("episode_id"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--server", default="ws://127.0.0.1:8051")
    parser.add_argument("--client-id", default="emerge-behavior-probe")
    parser.add_argument("--dry-run", action="store_true", help="Validate the image without calling LightNav")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.instruction.strip():
        raise ValueError("instruction must be non-empty")
    frame = load_head_frame(args.manifest)
    result: dict[str, Any] = {
        "image_path": str(frame.image_path),
        "width": frame.width,
        "height": frame.height,
        "revision": frame.revision,
        "observation_seq": frame.observation_seq,
        "episode_id": frame.episode_id,
        "instruction": args.instruction.strip(),
    }
    if not args.dry_run:
        with LightNavWsClient(args.server, client_id=args.client_id) as client:
            reply = client.predict(frame.jpeg, args.instruction.strip())
        result.update(
            server=args.server,
            first_waypoint=list(reply.first_waypoint),
            stop=reply.stop,
            latency_ms=reply.latency_ms,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

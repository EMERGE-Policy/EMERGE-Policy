"""Read-only helpers shared by BEHAVIOR shadow experiments."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import time
from typing import Any

from Emerge.base import Tool
from Emerge.utils.helpers import sync_workspace_templates
from scripts.Behavior_eval.runner_common import atomic_write_json, read_json


MANIFEST = Path("artifacts/observations/observation.json")
STATUS = "SHADOW_STATUS.json"


class WatchObservationTool(Tool):
    """Let the Agent wait for a new published frame when it chooses to."""

    def __init__(self, workspace: Path):
        self.workspace = workspace

    @property
    def name(self) -> str:
        return "watch_observation"

    @property
    def description(self) -> str:
        return "Wait for a newer camera observation or episode end; does not control the robot."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "after_revision": {"type": "integer", "minimum": 0},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": 60},
            },
            "required": ["after_revision"],
        }

    async def execute(self, after_revision: int, timeout_s: int = 30) -> str:
        deadline = time.monotonic() + timeout_s
        while True:
            manifest = read_json(self.workspace / MANIFEST)
            status = read_json(self.workspace / STATUS)
            revision = int(manifest.get("revision") or 0)
            if revision > after_revision or status.get("phase") in {"completed", "error"}:
                return json.dumps({
                    "revision": revision,
                    "observation_seq": manifest.get("observation_seq"),
                    "manifest": str(MANIFEST),
                    "phase": status.get("phase", "running"),
                    "official_result": status.get("official_result"),
                })
            if time.monotonic() >= deadline:
                return json.dumps({"revision": revision, "phase": "waiting", "timeout": True})
            await asyncio.sleep(0.5)


def gateway_command(*, upstream_port: int, gateway_port: int, workspace: Path,
                    camera_config: Path) -> list[str]:
    """Intentionally omit --controlled: all champion actions pass untouched."""
    return [
        sys.executable, "-m", "robot.behavior.gateway",
        "--upstream-url", f"ws://127.0.0.1:{upstream_port}",
        "--listen-host", "127.0.0.1", "--listen-port", str(gateway_port),
        "--observation-workspace", str(workspace),
        "--observation-publish-interval", "5",
        "--camera-calibration-config", str(camera_config),
    ]


def prepare_observer_workspace(workspace: Path, *, task_name: str, instruction: str) -> None:
    workspace.mkdir(parents=True)
    sync_workspace_templates(workspace, silent=True)
    (workspace / "AGENTS.md").write_text(
        "# Read-only EMERGE observer\n\n"
        "You are observing a BEHAVIOR episode driven entirely by an independent champion policy. "
        "You cannot send robot actions. Decide on your own whether and when the "
        "available read-only tools would help you understand the scene. "
        "Do not infer benchmark success from your own analysis; report only the official result.\n",
        encoding="utf-8",
    )
    (workspace / "EMBODIED.md").write_text(
        "# BEHAVIOR passive observation\n\n"
        f"Task: {task_name}\nInstruction: {instruction}\n\n"
        f"The latest calibrated RGB-D manifest is at `{MANIFEST}`. "
        f"The episode state is at `{STATUS}`. "
        "Use `watch_observation` to wait for new frames if useful. "
        "Coordinates in the manifest are in `robot_base`; +X forward, +Y left, +Z up. "
        "No action queue or robot control exists in this workspace.\n",
        encoding="utf-8",
    )
    atomic_write_json(workspace / STATUS, {"phase": "running", "task": task_name})

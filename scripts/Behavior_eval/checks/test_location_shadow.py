"""Offline safety checks for the passive location-observer experiment."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from Emerge.base import ToolRegistry
from scripts.Behavior_eval.run_location_shadow import (
    READ_ONLY_TOOLS,
    WatchObservationTool,
    build_observer,
    gateway_command,
)
from scripts.Behavior_eval.runner_common import atomic_write_json


class LocationShadowTests(unittest.TestCase):
    def test_gateway_is_passive(self):
        command = gateway_command(
            upstream_port=8011, gateway_port=8000,
            workspace=Path("/tmp/observer"), camera_config=Path("/tmp/camera.json"),
        )
        self.assertNotIn("--controlled", command)
        self.assertEqual(command[command.index("--upstream-url") + 1], "ws://127.0.0.1:8011")

    def test_agent_cannot_act_or_run_shell(self):
        tools = ToolRegistry()
        for name in ("read_file", "list_dir", "delegate_subagent", "update_plan",
                     "execute_robot_action", "exec", "write_file", "edit_file"):
            tools._tools[name] = object()
        agents = SimpleNamespace(names=("object_location", "task_verification"))
        agents.unregister = lambda name: setattr(agents, "names", ("object_location",))
        fake = SimpleNamespace(tools=tools, subagent_registry=agents)
        config = SimpleNamespace(
            tools=SimpleNamespace(restrict_to_workspace=False),
            visual_monitor=SimpleNamespace(enabled=True),
            workspace_path=Path("/tmp/observer"),
        )
        with patch("scripts.Behavior_eval.run_location_shadow.build_agent", return_value=fake):
            observer = build_observer(config)
        self.assertEqual(set(observer.tools.tool_names), READ_ONLY_TOOLS)
        self.assertEqual(observer.subagent_registry.names, ("object_location",))
        self.assertTrue(config.tools.restrict_to_workspace)
        self.assertFalse(config.visual_monitor.enabled)

    def test_watch_only_reads_published_revision_and_status(self):
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            atomic_write_json(workspace / "artifacts/observations/observation.json", {
                "revision": 3, "observation_seq": 15,
            })
            atomic_write_json(workspace / "SHADOW_STATUS.json", {"phase": "running"})
            answer = asyncio.run(WatchObservationTool(workspace).execute(after_revision=2))
            self.assertIn('"revision": 3', answer)
            self.assertIn('"observation_seq": 15', answer)


if __name__ == "__main__":
    unittest.main()

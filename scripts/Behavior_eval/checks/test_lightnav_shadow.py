"""Offline safety checks for the passive LightNav observer entrypoint."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from Emerge.base import ToolRegistry
from Emerge.agent.skills import SkillsLoader
from scripts.Behavior_eval.run_lightnav_shadow import (
    PREVIEW_TOOLS,
    build_preview_observer,
    gateway_command,
    install_preview_skill,
    parse_args,
)


class LightNavShadowTests(unittest.TestCase):
    def test_gateway_does_not_enable_control(self):
        command = gateway_command(
            upstream_port=8011, gateway_port=8000,
            workspace=Path("/tmp/preview"), camera_config=Path("/tmp/camera.json"),
        )
        self.assertNotIn("--controlled", command)

    def test_observer_has_no_robot_or_shell_action(self):
        tools = ToolRegistry()
        for name in ("read_file", "list_dir", "delegate_subagent", "update_plan",
                     "execute_robot_action", "exec", "write_file", "edit_file"):
            tools._tools[name] = object()
        specialists = SimpleNamespace(names=("object_location", "task_verification"))
        specialists.unregister = lambda name: setattr(
            specialists, "names", tuple(item for item in specialists.names if item != name)
        )
        fake = SimpleNamespace(tools=tools, subagent_registry=specialists)
        config = SimpleNamespace(
            workspace_path=Path("/tmp/preview"),
            tools=SimpleNamespace(restrict_to_workspace=False),
            visual_monitor=SimpleNamespace(enabled=True),
        )
        created = []
        with patch("scripts.Behavior_eval.run_lightnav_shadow.build_agent", return_value=fake):
            observer = build_preview_observer(config, server_url="ws://127.0.0.1:8051", tools=created)
        self.assertEqual(set(observer.tools.tool_names), PREVIEW_TOOLS)
        self.assertNotIn("execute_robot_action", observer.tools.tool_names)
        self.assertNotIn("exec", observer.tools.tool_names)
        self.assertNotIn("delegate_subagent", observer.tools.tool_names)
        self.assertEqual(observer.subagent_registry.names, ())
        self.assertTrue(config.tools.restrict_to_workspace)
        self.assertFalse(config.visual_monitor.enabled)
        self.assertEqual(len(created), 1)

    def test_defaults_do_not_overwrite_previous_experiment(self):
        args = parse_args([])
        self.assertEqual(args.instance_index, 0)
        self.assertIsNone(args.output_dir)
        self.assertEqual(args.lightnav_server, "ws://127.0.0.1:8051")

    def test_skill_is_visible_only_in_the_episode_workspace(self):
        with TemporaryDirectory() as temp:
            workspace = Path(temp)
            installed = install_preview_skill(workspace)
            self.assertEqual(installed, workspace / "skills/lightnav-behavior-preview/SKILL.md")
            self.assertIn("lightnav-behavior-preview", {
                skill["name"] for skill in SkillsLoader(workspace).list_skills()
            })


if __name__ == "__main__":
    unittest.main()

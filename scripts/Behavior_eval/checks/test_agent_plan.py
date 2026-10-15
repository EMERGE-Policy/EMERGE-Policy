"""Checks that BEHAVIOR uses EMERGE's generic manipulation skills."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from scripts.Behavior_eval.eval_behavior_agent import (
    _agent_message,
    _evaluation_instructions,
    _feedback_message,
)
from Emerge.agent.tools.embodied import EmbodiedActionTool
from Emerge.agent.tools.filesystem import ReadFileTool
from Emerge.agent.skills import SkillsLoader
from scripts.Behavior_eval.runner_common import prepare_workspace


ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "robot/profiles/behavior_r1pro.md"
GENERIC_VLA = ROOT / "Emerge/skills/vla/SKILL.md"
GENERIC_ATTACH = ROOT / "Emerge/skills/attach/SKILL.md"
TASK_GUIDANCE = ROOT / "scripts/Behavior_eval/tasks/turning_on_radio.md"
TEST_TASK = "turning_on_radio"
TEST_PROMPT = "Turn on the radio receiver that's on the table in the living room."


class BehaviorAgentPlanTest(unittest.TestCase):
    def test_restricted_read_can_load_builtin_skills_but_not_other_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            workspace.mkdir()
            outside = root / "outside.txt"
            outside.write_text("private", encoding="utf-8")
            tool = ReadFileTool(
                workspace=workspace,
                allowed_dir=workspace,
                additional_allowed_dirs=(GENERIC_VLA.parents[1],),
            )

            skill = asyncio.run(tool.execute(str(GENERIC_VLA)))
            denied = asyncio.run(tool.execute(str(outside)))

            self.assertIn("active embodiment profile is authoritative", skill)
            self.assertIn("outside allowed directory", denied)

    def test_workspace_does_not_shadow_generic_skills(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = prepare_workspace(
                Path(temporary), task_name="example_task", instruction="do the task"
            )
            loader = SkillsLoader(workspace)
            self.assertFalse((workspace / "skills/vla/SKILL.md").exists())
            self.assertEqual(loader.load_skill("vla"), GENERIC_VLA.read_text(encoding="utf-8"))
            self.assertEqual(
                loader.load_skill("attach"), GENERIC_ATTACH.read_text(encoding="utf-8")
            )

    def test_robot_profile_matches_the_native_driver_actions(self) -> None:
        profile = PROFILE.read_text(encoding="utf-8")
        for action in (
            "vla_execute",
            "move_to_pose",
            "move_linear",
            "set_gripper",
            "hold_position",
            "set_control_mode",
        ):
            with self.subTest(action=action):
                self.assertIn(f"`{action}`", profile)
        self.assertIn("Switching back to the OpenPI policy resets its action chunk", profile)
        self.assertIn("preserve the current wrist orientation", profile)
        self.assertIn("sole authoritative success signal", profile)
        self.assertIn("generic EMERGE skill", profile)
        self.assertNotIn("Required task route", profile)
        self.assertNotIn("Turning on the radio normally", profile)

    def test_runner_prompt_keeps_protocol_and_requires_the_task_contract(self) -> None:
        rules = _evaluation_instructions(TEST_TASK, TEST_PROMPT)
        self.assertIn("calibrated", rules)
        self.assertIn("object_location", rules)
        self.assertIn("move_to_pose", rules)
        self.assertIn("task-specific execution contract", rules)
        self.assertIn("set_control_mode", rules)
        self.assertIn("base_motion_enabled == false", rules)
        self.assertNotIn("Follow the phased route", rules)
        initial = _agent_message(
            6002, task_name=TEST_TASK, task_instruction=TEST_PROMPT
        )
        feedback = _feedback_message(
            {"success": False}, 6002, task_instruction=TEST_PROMPT
        )
        self.assertIn("generic skills", initial)
        self.assertIn(TEST_TASK, initial)
        self.assertNotIn("localizing the button", initial)
        self.assertIn("Resume the current plan", feedback)

    def test_task_contract_delays_button_localization_until_after_reorientation(self) -> None:
        guidance = TASK_GUIDANCE.read_text(encoding="utf-8")
        phases = (
            "Locate the whole radio with the head RGB-D view",
            "Approach the radio body",
            "Grasp and reorient the radio",
            "Verify the new pose",
            "Relocalize the exposed button",
            "Press and hold with the other arm",
        )
        positions = [guidance.index(phase) for phase in phases]
        self.assertEqual(positions, sorted(positions))
        normalized = " ".join(guidance.split())
        self.assertIn("do not localize or press the button", normalized)
        self.assertIn("changes from red to green", normalized)
        self.assertIn("four verification views", normalized)
        self.assertIn('{"mode": "manipulation"}', normalized)

    def test_generic_vla_defers_fixed_prompt_semantics_to_the_embodiment(self) -> None:
        skill = GENERIC_VLA.read_text(encoding="utf-8")
        self.assertIn("active embodiment profile is authoritative", skill)
        self.assertIn("effective_prompt", skill)

    def test_agent_wait_budget_covers_generated_primitive_trajectories(self) -> None:
        self.assertGreaterEqual(
            EmbodiedActionTool._result_timeout("move_to_pose", {}),
            600.0,
        )
        self.assertGreater(
            EmbodiedActionTool._result_timeout("set_gripper", {"steps": 15}),
            60.0,
        )


if __name__ == "__main__":
    unittest.main()

"""Checks that BEHAVIOR uses EMERGE's generic manipulation skills."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from scripts.Behavior_eval.eval_behavior_agent import (
    EVALUATION_INSTRUCTIONS,
    _agent_message,
    _feedback_message,
)
from Emerge.agent.tools.embodied import EmbodiedActionTool
from Emerge.agent.skills import SkillsLoader
from scripts.Behavior_eval.runner_common import prepare_workspace


ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "robot/profiles/behavior_r1pro.md"
GENERIC_VLA = ROOT / "Emerge/skills/vla/SKILL.md"
GENERIC_ATTACH = ROOT / "Emerge/skills/attach/SKILL.md"


class BehaviorAgentPlanTest(unittest.TestCase):
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
        ):
            with self.subTest(action=action):
                self.assertIn(f"`{action}`", profile)
        self.assertIn("Switching back to π0.5 resets its action chunk", profile)
        self.assertIn("preserve the current wrist orientation", profile)
        self.assertIn("sole authoritative success signal", profile)
        self.assertIn("generic EMERGE skill", profile)
        self.assertNotIn("Required task route", profile)
        self.assertNotIn("Turning on the radio normally", profile)

    def test_runner_prompt_keeps_protocol_without_prescribing_a_route(self) -> None:
        self.assertIn("calibrated", EVALUATION_INSTRUCTIONS)
        self.assertIn("object_location", EVALUATION_INSTRUCTIONS)
        self.assertIn("move_to_pose", EVALUATION_INSTRUCTIONS)
        self.assertIn("task route is preselected", EVALUATION_INSTRUCTIONS)
        self.assertNotIn("Follow the phased route", EVALUATION_INSTRUCTIONS)
        initial = _agent_message(6002)
        feedback = _feedback_message({"success": False}, 6002)
        self.assertIn("generic skills", initial)
        self.assertIn("does not prescribe a task-specific route", initial)
        self.assertIn("Resume the current plan", feedback)

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

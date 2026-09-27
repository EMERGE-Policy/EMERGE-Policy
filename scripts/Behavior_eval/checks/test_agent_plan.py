"""Checks that the BEHAVIOR Agent receives the complete phased control route."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.Behavior_eval.eval_behavior_agent import (
    EVALUATION_INSTRUCTIONS,
    _agent_message,
    _feedback_message,
)
from Emerge.agent.tools.embodied import EmbodiedActionTool


ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / "robot/profiles/behavior_r1pro.md"
SKILL = ROOT / "scripts/Behavior_eval/workspace_skill/SKILL.md"


class BehaviorAgentPlanTest(unittest.TestCase):
    def test_workspace_skill_contains_every_control_phase(self) -> None:
        skill = SKILL.read_text(encoding="utf-8")
        for requirement in (
            "Coarse navigation",
            "Calibrated localization",
            "Measured pre-contact approach",
            "Fine contact",
            "Verify or recover",
            '"agent_name": "object_location"',
            "frame: robot_base",
            "bbox_world_min_m",
            "standoff_m: 0.12",
            "move_to_pose",
            "move_linear",
            "task_verification",
        ):
            with self.subTest(requirement=requirement):
                self.assertIn(requirement, skill)
        self.assertNotIn("Do not use `move_to_pose`", skill)

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

    def test_runner_prompt_does_not_restore_the_old_vla_only_route(self) -> None:
        self.assertIn("calibrated object localization", EVALUATION_INSTRUCTIONS)
        self.assertIn("move_to_pose", EVALUATION_INSTRUCTIONS)
        initial = _agent_message(6002)
        feedback = _feedback_message({"success": False}, 6002)
        self.assertIn("phased route", initial)
        self.assertIn("Recover from lack of progress", initial)
        self.assertIn("Resume the correct phase", feedback)

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

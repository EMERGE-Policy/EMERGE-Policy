"""Checks for R1Pro primitive trajectory generation."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from robot.behavior.r1pro_kinematics import R1ProKinematics
from robot.behavior.r1pro_motion import (
    plan_adjust_posture,
    plan_move_linear,
    plan_move_to_pose,
    plan_set_gripper,
)

URDF = (
    Path(__file__).resolve().parents[3]
    / "third_party/behavior-1k/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf"
)
TRUNK = np.asarray([1.025, -1.45, -0.47, 0.0], dtype=np.float64)


def _state(kinematics: R1ProKinematics) -> dict:
    arms = {}
    for side in ("left", "right"):
        pose = kinematics.forward(side, TRUNK, np.zeros(7))
        arms[side] = {
            "qpos": [0.0] * 7,
            "eef_position": pose[:3, 3].tolist(),
            "eef_orientation_xyzw": Rotation.from_matrix(pose[:3, :3]).as_quat().tolist(),
        }
    return {"trunk": {"qpos": TRUNK.tolist()}, "arms": arms}


@unittest.skipUnless(URDF.is_file(), f"official R1Pro URDF not found at {URDF}")
class R1ProMotionPlanningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kinematics = R1ProKinematics(URDF)
        cls.state = _state(cls.kinematics)
        cls.hold = [0.0] * 23
        cls.hold[3:7] = TRUNK.tolist()
        cls.hold[14] = 1.0
        cls.hold[22] = 1.0

    def test_move_to_pose_is_bounded_and_reaches_the_requested_position(self) -> None:
        known_qpos = np.asarray([-0.25, 0.45, 0.2, -0.7, 0.25, 0.1, -0.2])
        target_pose = self.kinematics.forward("left", TRUNK, known_qpos)
        target = target_pose[:3, 3]
        target_orientation = Rotation.from_matrix(target_pose[:3, :3]).as_quat()

        plan = plan_move_to_pose(
            self.kinematics,
            self.state,
            self.hold,
            side="left",
            position_m=target.tolist(),
            orientation_quat=target_orientation.tolist(),
            max_joint_step_rad=0.03,
        )

        actions = np.asarray(plan.actions)
        self.assertGreater(len(actions), 1)
        self.assertTrue(np.allclose(actions[:, 0:3], 0.0))
        deltas = np.diff(np.vstack([np.zeros((1, 7)), actions[:, 7:14]]), axis=0)
        self.assertLessEqual(float(np.max(np.abs(deltas))), 0.030001)
        final_pose = self.kinematics.forward("left", TRUNK, actions[-1, 7:14])
        self.assertLess(float(np.linalg.norm(final_pose[:3, 3] - target)), 0.01)
        orientation_error = Rotation.from_matrix(
            final_pose[:3, :3] @ target_pose[:3, :3].T
        ).magnitude()
        self.assertLess(float(orientation_error), 0.15)

    def test_move_linear_tracks_intermediate_cartesian_waypoints(self) -> None:
        start = np.asarray(self.state["arms"]["right"]["eef_position"])
        target = start + np.asarray([0.015, 0.0, 0.0])

        plan = plan_move_linear(
            self.kinematics,
            self.state,
            self.hold,
            side="right",
            position_m=target.tolist(),
            max_cartesian_step_m=0.005,
        )

        self.assertGreaterEqual(len(plan.actions), 3)
        final_pose = self.kinematics.forward("right", TRUNK, plan.actions[-1][15:22])
        self.assertLess(float(np.linalg.norm(final_pose[:3, 3] - target)), 0.01)

    def test_move_linear_accepts_a_relative_displacement(self) -> None:
        start = np.asarray(self.state["arms"]["right"]["eef_position"])
        delta = np.asarray([0.01, 0.0, 0.0])

        plan = plan_move_linear(
            self.kinematics,
            self.state,
            self.hold,
            side="right",
            delta_m=delta.tolist(),
            max_cartesian_step_m=0.005,
        )

        final_pose = self.kinematics.forward("right", TRUNK, plan.actions[-1][15:22])
        self.assertLess(float(np.linalg.norm(final_pose[:3, 3] - (start + delta))), 0.01)

    def test_move_linear_requires_exactly_one_target_form(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one"):
            plan_move_linear(
                self.kinematics,
                self.state,
                self.hold,
                side="right",
            )
        with self.assertRaisesRegex(ValueError, "exactly one"):
            plan_move_linear(
                self.kinematics,
                self.state,
                self.hold,
                side="right",
                position_m=[0.1, 0.2, 0.3],
                delta_m=[0.0, 0.0, 0.1],
            )

    def test_pose_steps_slow_motion_and_settle_holds_the_final_command(self) -> None:
        target_pose = self.kinematics.forward(
            "left", TRUNK, [-0.25, 0.45, 0.2, -0.7, 0.25, 0.1, -0.2]
        )
        kwargs = dict(
            side="left", position_m=target_pose[:3, 3].tolist(),
            orientation_quat=Rotation.from_matrix(target_pose[:3, :3]).as_quat().tolist(),
        )
        original = plan_move_to_pose(self.kinematics, self.state, self.hold, **kwargs)
        count = len(original.actions) + 13
        timed = plan_move_to_pose(
            self.kinematics, self.state, self.hold, **kwargs,
            steps=count, settle_steps=7, max_steps=count + 7,
        )
        actions = np.asarray(timed.actions)
        self.assertEqual(len(actions), count + 7)
        np.testing.assert_allclose(actions[count - 1:], np.tile(actions[-1], (8, 1)))
        np.testing.assert_allclose(actions[-1], original.actions[-1])
        deltas = np.diff(np.vstack([np.zeros((1, 7)), actions[:, 7:14]]), axis=0)
        self.assertLessEqual(float(np.max(np.abs(deltas))), 0.030001)
        unchanged = [*range(7), *range(14, 23)]
        np.testing.assert_allclose(actions[:, unchanged], np.tile(np.asarray(self.hold)[unchanged], (len(actions), 1)))

    def test_linear_timing_preserves_all_original_waypoints(self) -> None:
        kwargs = dict(side="right", delta_m=[0.015, 0.0, 0.0], max_cartesian_step_m=0.005)
        original = plan_move_linear(self.kinematics, self.state, self.hold, **kwargs)
        timed = plan_move_linear(
            self.kinematics, self.state, self.hold, **kwargs, steps=30, settle_steps=20,
        )
        self.assertEqual(len(timed.actions), 50)
        np.testing.assert_allclose(timed.actions[29:], np.tile(timed.actions[-1], (21, 1)))
        offset = 0
        for waypoint in original.actions:
            while offset < 30 and not np.allclose(timed.actions[offset], waypoint):
                offset += 1
            self.assertLess(offset, 30, "timing must not skip a Cartesian waypoint")
            offset += 1
        deltas = np.diff(np.vstack([self.hold, timed.actions]), axis=0)
        self.assertLessEqual(float(np.max(np.abs(deltas))), 0.030001)

    def test_short_steps_cannot_speed_up_motion_beyond_joint_limits(self) -> None:
        pose = self.kinematics.forward("right", TRUNK, [-0.1, 0.2, 0.1, -0.3, 0.1, 0.05, -0.1])
        for planner, kwargs in (
            (plan_move_to_pose, {"position_m": pose[:3, 3].tolist(), "orientation_quat": Rotation.from_matrix(pose[:3, :3]).as_quat().tolist()}),
            (plan_move_linear, {"delta_m": [0.015, 0.0, 0.0]}),
        ):
            with self.subTest(planner=planner.__name__):
                original = planner(self.kinematics, self.state, self.hold, side="right", **kwargs)
                timed = planner(self.kinematics, self.state, self.hold, side="right", **kwargs, steps=1)
                np.testing.assert_allclose(timed.actions, original.actions)

    def test_invalid_timing_is_rejected_before_ik(self) -> None:
        invalid = [
            *({"steps": value} for value in (True, 0, -1, 1.5, "40", float("nan"))),
            *({"settle_steps": value} for value in (True, -1, 1.5, "20", None)),
            {"steps": 241}, {"settle_steps": 240}, {"steps": 230, "settle_steps": 20},
        ]
        with patch.object(self.kinematics, "solve") as solve:
            for planner, target in (
                (plan_move_to_pose, {"position_m": [0.1, 0.2, 0.3]}),
                (plan_move_linear, {"delta_m": [0.01, 0.0, 0.0]}),
            ):
                for timing in invalid:
                    with self.subTest(planner=planner.__name__, timing=timing), self.assertRaises(ValueError):
                        planner(self.kinematics, self.state, self.hold, side="left", **target, **timing)
            solve.assert_not_called()

    def test_settling_reserves_budget_for_the_safe_motion(self) -> None:
        kwargs = dict(side="right", delta_m=[0.015, 0.0, 0.0])
        original = plan_move_linear(self.kinematics, self.state, self.hold, **kwargs)
        with self.assertRaisesRegex(ValueError, "steps|limit"):
            plan_move_linear(
                self.kinematics, self.state, self.hold, **kwargs,
                steps=1, settle_steps=1, max_steps=len(original.actions),
            )

    def test_timing_does_not_bypass_an_unreachable_ik_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "IK failed"):
            plan_move_to_pose(
                self.kinematics, self.state, self.hold,
                side="left", position_m=[100.0, 100.0, 100.0], steps=40,
            )

    def test_coarse_plan_reports_its_fk_endpoint_and_keeps_other_joints_held(self) -> None:
        requested = [1.5, 0.3, 0.6]
        plan = plan_move_to_pose(
            self.kinematics, self.state, self.hold,
            side="left", position_m=requested, coarse=True,
        )
        report = plan.approach_report
        self.assertIsNotNone(report)
        self.assertEqual(report["requested_position_m"], requested)
        endpoint = self.kinematics.forward("left", TRUNK, plan.actions[-1][7:14])[:3, 3]
        np.testing.assert_allclose(report["planned_reachable_position_m"], endpoint)
        self.assertAlmostEqual(report["remaining_distance_m"], np.linalg.norm(endpoint - requested))
        self.assertGreater(report["remaining_distance_m"], 0.01)
        self.assertFalse(report["collision_checked"])
        actions = np.asarray(plan.actions)
        differences = np.diff(np.vstack([self.hold, actions]), axis=0)
        self.assertLessEqual(np.max(np.abs(differences)), 0.030001)
        unchanged = [*range(7), *range(14, 23)]
        np.testing.assert_allclose(actions[:, unchanged], np.tile(np.asarray(self.hold)[unchanged], (len(actions), 1)))
        self.assertLessEqual(len(actions), 240)

    def test_exact_remains_default_and_coarse_flag_requires_a_boolean(self) -> None:
        for invalid in (1, "true", None):
            with self.subTest(coarse=invalid), self.assertRaisesRegex(ValueError, "boolean"):
                plan_move_to_pose(self.kinematics, self.state, self.hold, side="left", position_m=[1.5, 0.3, 0.6], coarse=invalid)
        for options in ({}, {"coarse": False}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "IK failed"):
                plan_move_to_pose(self.kinematics, self.state, self.hold, side="left", position_m=[1.5, 0.3, 0.6], **options)

    def test_coarse_does_not_bypass_fk_consistency_or_motion_budget(self) -> None:
        state = _state(self.kinematics)
        state["arms"]["left"]["eef_position"][0] += 0.2
        with self.assertRaisesRegex(ValueError, "does not match"):
            plan_move_to_pose(self.kinematics, state, self.hold, side="left", position_m=[1.5, 0.3, 0.6], coarse=True)
        with self.assertRaisesRegex(ValueError, "limit"):
            plan_move_to_pose(self.kinematics, self.state, self.hold, side="left", position_m=[1.5, 0.3, 0.6], coarse=True, max_steps=1)

    def test_set_gripper_changes_only_the_selected_gripper(self) -> None:
        plan = plan_set_gripper(self.hold, side="left", command="close", steps=5)
        actions = np.asarray(plan.actions)

        self.assertEqual(actions.shape, (5, 23))
        self.assertAlmostEqual(actions[-1, 14], -1.0)
        expected = np.tile(np.asarray(self.hold), (5, 1))
        expected[:, 14] = actions[:, 14]
        self.assertTrue(np.allclose(actions, expected))

    def test_motion_is_rejected_when_fk_disagrees_with_observed_state(self) -> None:
        inconsistent = _state(self.kinematics)
        inconsistent["arms"]["left"]["eef_position"][0] += 0.2
        target = self.kinematics.forward("left", TRUNK, np.zeros(7))[:3, 3]

        with self.assertRaisesRegex(ValueError, "does not match the simulator state"):
            plan_move_to_pose(
                self.kinematics,
                inconsistent,
                self.hold,
                side="left",
                position_m=target.tolist(),
            )

    def test_posture_holds_base_grippers_and_unrequested_joints(self) -> None:
        hold = self.hold.copy()
        hold[:3] = [0.3, -0.2, 0.1]
        plan = plan_adjust_posture(self.kinematics, self.state, hold,
                                   joint_deltas_rad={"torso_joint4": 0.1, "left_arm_joint1": 0.15})
        actions = np.asarray(plan.actions)
        np.testing.assert_allclose(actions[:, :3], 0.0)
        np.testing.assert_allclose(actions[-1, 6:8], [0.1, 0.15])
        unchanged = [3, 4, 5, *range(8, 23)]
        np.testing.assert_allclose(actions[:, unchanged], np.tile(np.asarray(hold)[unchanged], (len(actions), 1)))
        differences = np.diff(np.vstack([self.hold, actions]), axis=0)
        self.assertLessEqual(float(np.max(np.abs(differences[:, 3:]))), 0.010001)
        np.testing.assert_allclose(actions[-10:], np.tile(actions[-1], (10, 1)))

    def test_posture_rejects_invalid_deltas_and_insufficient_budget(self) -> None:
        for changes in ({}, {"camera_yaw": 0.1}, {"torso_joint4": True},
                        {"torso_joint4": float("nan")}, {"torso_joint4": 0.3},
                        {"torso_joint4": 0.0}, {1: 0.1}, {"torso_joint4": []}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                plan_adjust_posture(self.kinematics, self.state, self.hold, joint_deltas_rad=changes)
        with self.assertRaisesRegex(ValueError, "budget"):
            plan_adjust_posture(self.kinematics, self.state, self.hold,
                                joint_deltas_rad={"torso_joint4": 0.1}, max_steps=15)
        state = _state(self.kinematics)
        upper = self.kinematics.posture_joint_limits()["torso_joint4"][1]
        state["trunk"]["qpos"][3] = upper - 0.01
        # Keep measured end-effector geometry consistent with this start pose.
        for side in ("left", "right"):
            state["arms"][side]["eef_position"] = self.kinematics.forward(side, state["trunk"]["qpos"], [0.0] * 7)[:3, 3].tolist()
        with self.assertRaisesRegex(ValueError, "URDF limits"):
            plan_adjust_posture(self.kinematics, state, self.hold, joint_deltas_rad={"torso_joint4": 0.1})


if __name__ == "__main__":
    unittest.main()

"""Checks for R1Pro primitive trajectory generation."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot.behavior.r1pro_kinematics import R1ProKinematics
from robot.behavior.r1pro_motion import (
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


if __name__ == "__main__":
    unittest.main()

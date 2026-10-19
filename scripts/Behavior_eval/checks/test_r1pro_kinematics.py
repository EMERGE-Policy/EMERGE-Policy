"""Checks for the standalone R1Pro URDF kinematics used by BEHAVIOR actions."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from robot.behavior.r1pro_kinematics import R1ProKinematics


URDF = (
    Path(__file__).resolve().parents[3]
    / "third_party/behavior-1k/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf"
)
TRUNK = np.asarray([1.025, -1.45, -0.47, 0.0], dtype=np.float64)


@unittest.skipUnless(URDF.is_file(), f"official R1Pro URDF not found at {URDF}")
class R1ProKinematicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.kinematics = R1ProKinematics(URDF)

    def test_forward_kinematics_returns_valid_eef_poses_for_both_arms(self) -> None:
        for side in ("left", "right"):
            with self.subTest(side=side):
                pose = self.kinematics.forward(side, TRUNK, np.zeros(7))
                self.assertEqual(pose.shape, (4, 4))
                np.testing.assert_allclose(pose[3], [0.0, 0.0, 0.0, 1.0])
                np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-8)
                self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1.0)

    def test_inverse_kinematics_recovers_a_reachable_position_within_limits(self) -> None:
        expected = np.asarray([-0.4, 0.8, 0.3, -1.0, 0.4, 0.2, -0.3])
        target = self.kinematics.forward("left", TRUNK, expected)[:3, 3]

        solution = self.kinematics.solve("left", target, TRUNK, np.zeros(7))

        lower, upper = self.kinematics.arm_limits("left")
        self.assertTrue(solution.success, solution)
        self.assertLess(solution.position_error_m, 0.01)
        self.assertTrue(np.all(np.asarray(solution.arm_qpos) >= lower))
        self.assertTrue(np.all(np.asarray(solution.arm_qpos) <= upper))

    def test_unreachable_position_is_rejected(self) -> None:
        solution = self.kinematics.solve(
            "right", [10.0, 10.0, 10.0], TRUNK, np.zeros(7), max_evaluations=30
        )
        self.assertFalse(solution.success)
        self.assertGreater(solution.position_error_m, 1.0)

    def test_coarse_approach_accepts_a_pose_rejected_by_exact_orientation_constraints(self) -> None:
        seed = np.zeros(7)
        target = self.kinematics.forward(
            "right", TRUNK, [-0.1, 0.2, 0.1, -0.3, 0.1, 0.05, -0.1]
        )[:3, 3]
        orientation = Rotation.from_matrix(self.kinematics.forward("right", TRUNK, seed)[:3, :3]).as_quat()
        exact = self.kinematics.solve("right", target, TRUNK, seed, target_orientation_xyzw=orientation)
        coarse = self.kinematics.solve_approach("right", target, TRUNK, seed, target_orientation_xyzw=orientation)
        self.assertFalse(exact.success)
        self.assertTrue(coarse.success, coarse)
        lower, upper = self.kinematics.arm_limits("right")
        self.assertTrue(np.all(np.asarray(coarse.arm_qpos) >= lower))
        self.assertTrue(np.all(np.asarray(coarse.arm_qpos) <= upper))
        endpoint = self.kinematics.forward("right", TRUNK, coarse.arm_qpos)[:3, 3]
        self.assertAlmostEqual(coarse.position_error_m, np.linalg.norm(endpoint - target))

    def test_coarse_approach_can_stop_short_of_an_unreachable_target(self) -> None:
        target = [1.5, 0.3, 0.6]
        seed = np.zeros(7)
        exact = self.kinematics.solve("left", target, TRUNK, seed)
        coarse = self.kinematics.solve_approach("left", target, TRUNK, seed)
        self.assertFalse(exact.success)
        self.assertTrue(coarse.success, coarse)
        self.assertGreater(coarse.position_error_m, 0.01)
        self.assertLess(coarse.position_error_m, np.linalg.norm(self.kinematics.forward("left", TRUNK, seed)[:3, 3] - target))

    def test_coarse_does_not_accept_optimizer_failure_or_no_progress(self) -> None:
        from types import SimpleNamespace
        frozen = SimpleNamespace(success=True, x=np.zeros(7), nfev=1, message="no movement")
        with patch("scipy.optimize.least_squares", return_value=frozen):
            result = self.kinematics.solve_approach("left", [2, 0, 1], TRUNK, np.zeros(7))
            self.assertFalse(result.success)
            self.assertIn("no useful approach progress", result.message)
        frozen.success = False
        with patch("scipy.optimize.least_squares", return_value=frozen):
            target = self.kinematics.forward("left", TRUNK, np.zeros(7))[:3, 3]
            self.assertFalse(self.kinematics.solve_approach("left", target, TRUNK, np.zeros(7)).success)

    def test_coarse_approach_rejects_non_finite_targets(self) -> None:
        for target in ([float("nan"), 0, 0], [0, float("inf"), 0]):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.kinematics.solve_approach("left", target, TRUNK, np.zeros(7))

    def test_invalid_side_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "left.*right"):
            self.kinematics.forward("middle", TRUNK, np.zeros(7))

    def test_body_links_follow_torso_and_arm_while_base_stays_fixed(self) -> None:
        state = {"trunk": {"qpos": TRUNK.tolist()},
                 "arms": {side: {"qpos": [0.0] * 7} for side in ("left", "right")}}
        before = self.kinematics.body_state(state)
        state["trunk"]["qpos"][3] += 0.1
        turned = self.kinematics.body_state(state)
        np.testing.assert_allclose(turned["link_poses"]["base_link"], np.eye(4))
        self.assertFalse(np.allclose(before["link_poses"]["zed_link"], turned["link_poses"]["zed_link"]))
        state["arms"]["left"]["qpos"][0] += 0.1
        arm_moved = self.kinematics.body_state(state)
        self.assertFalse(np.allclose(turned["link_poses"]["left_realsense_link"], arm_moved["link_poses"]["left_realsense_link"]))
        np.testing.assert_allclose(turned["link_poses"]["zed_link"], arm_moved["link_poses"]["zed_link"])
        self.assertEqual(len(before["joint_limits_rad"]), 18)
        self.assertFalse(before["collision_checked"])
        self.assertNotIn("wheel_motor_link1", before["link_poses"])


if __name__ == "__main__":
    unittest.main()

"""Checks for the standalone R1Pro URDF kinematics used by BEHAVIOR actions."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

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

    def test_invalid_side_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "left.*right"):
            self.kinematics.forward("middle", TRUNK, np.zeros(7))


if __name__ == "__main__":
    unittest.main()

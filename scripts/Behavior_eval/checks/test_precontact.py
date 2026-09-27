"""Checks for measured bounding-box pre-contact geometry."""

from __future__ import annotations

import unittest

import numpy as np

from robot.behavior.approach import compute_precontact_pose


def _robot_state() -> dict:
    return {
        "arms": {
            "left": {
                "eef_position": [0.30, 0.35, 1.05],
                "eef_orientation_xyzw": [0.0, 0.0, 0.0, 2.0],
            },
            "right": {
                "eef_position": [0.35, -0.25, 0.95],
                "eef_orientation_xyzw": [0.0, 0.70710678, 0.0, 0.70710678],
            },
        }
    }


class PrecontactGeometryTest(unittest.TestCase):
    def test_auto_selects_nearer_arm_and_stays_outside_the_box(self) -> None:
        result = compute_precontact_pose(
            _robot_state(),
            [0.48, -0.12, 0.82],
            [0.62, 0.02, 0.96],
            standoff_m=0.12,
        )

        self.assertEqual(result.side, "right")
        np.testing.assert_allclose(
            result.orientation_quat,
            [0.0, 0.70710678, 0.0, 0.70710678],
            atol=1e-7,
        )
        surface = np.asarray(result.surface_point_m)
        target = np.asarray(result.position_m)
        self.assertAlmostEqual(float(np.linalg.norm(target - surface)), 0.12)
        inside = np.all(target >= [0.48, -0.12, 0.82]) and np.all(
            target <= [0.62, 0.02, 0.96]
        )
        self.assertFalse(inside)

    def test_explicit_side_is_respected_and_orientation_is_normalized(self) -> None:
        result = compute_precontact_pose(
            _robot_state(),
            [0.48, -0.12, 0.82],
            [0.62, 0.02, 0.96],
            side="left",
        )

        self.assertEqual(result.side, "left")
        self.assertEqual(result.orientation_quat, (0.0, 0.0, 0.0, 1.0))

    def test_invalid_bbox_and_standoff_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than"):
            compute_precontact_pose(_robot_state(), [1, 0, 0], [0, 1, 1])
        with self.assertRaisesRegex(ValueError, "between"):
            compute_precontact_pose(
                _robot_state(), [0, 0, 0], [1, 1, 1], standoff_m=0.3
            )


if __name__ == "__main__":
    unittest.main()

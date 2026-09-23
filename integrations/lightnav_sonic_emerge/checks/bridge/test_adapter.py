import math
import unittest

from integrations.lightnav_sonic_emerge.bridge.adapter import (
    SONIC_IDLE_MODE,
    SONIC_SLOW_WALK_MODE,
    LightNav2SonicAdapter,
)


class LightNav2SonicAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = LightNav2SonicAdapter()

    def test_forward_at_zero_yaw_is_world_x(self) -> None:
        command = self.adapter.convert((0.25, 0.0, 0.0), 0.0)
        self.assertEqual(command.mode, SONIC_SLOW_WALK_MODE)
        self.assertAlmostEqual(command.movement_direction[0], 1.0)
        self.assertAlmostEqual(command.movement_direction[1], 0.0)
        self.assertAlmostEqual(command.movement_speed, 0.25)

    def test_positive_lateral_is_world_left(self) -> None:
        command = self.adapter.convert((0.0, 0.20, 0.0), 0.0)
        self.assertAlmostEqual(command.movement_direction[0], 0.0)
        self.assertAlmostEqual(command.movement_direction[1], 1.0)
        self.assertAlmostEqual(command.movement_speed, 0.15)

    def test_robot_yaw_rotates_body_motion_into_world_frame(self) -> None:
        command = self.adapter.convert((0.25, 0.0, 0.0), math.pi / 2.0)
        self.assertAlmostEqual(command.movement_direction[0], 0.0, places=6)
        self.assertAlmostEqual(command.movement_direction[1], 1.0, places=6)

    def test_delta_yaw_updates_facing_and_is_clipped(self) -> None:
        command = self.adapter.convert((0.25, 0.0, 1.0), 0.0)
        expected = 0.35 * 0.25
        self.assertAlmostEqual(command.facing_direction[0], math.cos(expected), places=6)
        self.assertAlmostEqual(command.facing_direction[1], math.sin(expected), places=6)

    def test_planar_speed_is_clipped_after_component_clipping(self) -> None:
        command = self.adapter.convert((0.25, 0.25, 0.0), 0.0)
        self.assertAlmostEqual(command.movement_speed, 0.25)
        self.assertAlmostEqual(math.hypot(*command.movement_direction[:2]), 1.0)

    def test_near_waypoint_becomes_idle(self) -> None:
        command = self.adapter.convert((0.05, 0.02, 0.02), 0.3)
        self.assertEqual(command.mode, SONIC_IDLE_MODE)
        self.assertEqual(command.movement_speed, -1.0)
        self.assertEqual(command.movement_direction, (0.0, 0.0, 0.0))

    def test_yaw_only_waypoint_keeps_heading_request(self) -> None:
        command = self.adapter.convert((0.0, 0.0, 0.2), 0.0)
        self.assertEqual(command.mode, SONIC_SLOW_WALK_MODE)
        self.assertEqual(command.movement_direction, (0.0, 0.0, 0.0))
        self.assertEqual(command.movement_speed, -1.0)
        self.assertGreater(command.facing_direction[1], 0.0)

    def test_invalid_output_fails_closed(self) -> None:
        command = self.adapter.convert_or_idle((float("nan"), 0.0, 0.0), 0.0)
        self.assertEqual(command.mode, SONIC_IDLE_MODE)
        self.assertIn("invalid LightNav waypoint", command.reason)


if __name__ == "__main__":
    unittest.main()

import math
import unittest

from integrations.lightnav_sonic_emerge.bridge.adapter import SONIC_IDLE_MODE, SONIC_SLOW_WALK_MODE
from integrations.lightnav_sonic_emerge.bridge.teleop import TeleopState, command_from_state


class TeleopCommandTests(unittest.TestCase):
    def test_forward_at_zero_yaw_is_world_x(self) -> None:
        state = TeleopState(forward=1.0, speed_mps=0.18)
        command = command_from_state(state, 0.0, heading_tolerance_rad=0.05)
        self.assertEqual(command.mode, SONIC_SLOW_WALK_MODE)
        self.assertEqual(command.movement_direction, (1.0, 0.0, 0.0))
        self.assertAlmostEqual(command.movement_speed, 0.18)

    def test_local_left_rotates_into_world_frame(self) -> None:
        state = TeleopState(lateral=1.0, speed_mps=0.18)
        command = command_from_state(state, math.pi / 2.0, heading_tolerance_rad=0.05)
        self.assertAlmostEqual(command.movement_direction[0], -1.0, places=6)
        self.assertAlmostEqual(command.movement_direction[1], 0.0, places=6)

    def test_turn_in_place_uses_default_planner_speed(self) -> None:
        state = TeleopState(target_yaw=math.pi / 2.0)
        command = command_from_state(state, 0.0, heading_tolerance_rad=0.05)
        self.assertEqual(command.mode, SONIC_SLOW_WALK_MODE)
        self.assertEqual(command.movement_direction, (0.0, 0.0, 0.0))
        self.assertEqual(command.movement_speed, -1.0)
        self.assertAlmostEqual(command.facing_direction[1], 1.0)

    def test_reached_heading_is_idle(self) -> None:
        state = TeleopState(target_yaw=0.02)
        command = command_from_state(state, 0.0, heading_tolerance_rad=0.05)
        self.assertEqual(command.mode, SONIC_IDLE_MODE)


if __name__ == "__main__":
    unittest.main()

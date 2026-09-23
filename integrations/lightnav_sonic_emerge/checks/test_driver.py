from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from integrations.lightnav_sonic_emerge.driver import SonicLightNavDriver
from integrations.lightnav_sonic_emerge.navigation_backend import NavigationResult


class _Backend:
    def __init__(self, result):
        self.result = result

    def run(self, *, instruction, max_steps, cancel_check, state_callback):
        state_callback(
            {
                "status": "running",
                "instruction": instruction,
                "ticks": 1,
                "command_steps": 1,
                "max_steps": max_steps,
                "last_yaw_rad": 0.5,
                "last_waypoint": [0.2, 0.0, 0.0],
                "last_error": None,
            }
        )
        return self.result


class _TestDriver(SonicLightNavDriver):
    def __init__(self, result, **kwargs):
        self._test_result = result
        super().__init__(**kwargs)

    def _make_backend(self, _config):
        return _Backend(self._test_result)


class SonicLightNavDriverTest(unittest.TestCase):
    def _driver(self, result, directory):
        return _TestDriver(
            result,
            workspace=directory,
            elastic_band_release_file=None,
        )

    def test_navigation_result_updates_runtime_without_fake_xy(self) -> None:
        result = NavigationResult(
            status="completed",
            reason="lightnav_stop",
            ticks=4,
            command_steps=3,
            elapsed_s=1.0,
            last_waypoint=(0.1, 0.0, 0.0),
            last_yaw_rad=0.5,
            lightnav_stop=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            driver = self._driver(result, directory)
            driver.load_scene({})
            message = driver.execute_action(
                "navigate_vln",
                {"instruction": "go to the sofa", "max_steps": 8},
            )
            state = driver.get_runtime_state()["robots"][driver.ROBOT_ID]

        self.assertIn("semantic goal still requires", message)
        self.assertEqual(state["nav_state"]["status"], "completed")
        self.assertEqual(state["robot_pose"], {"frame": "sonic_world", "yaw": 0.5})
        self.assertNotIn("x", state["robot_pose"])
        self.assertNotIn("y", state["robot_pose"])

    def test_budget_exhaustion_is_action_completion_not_goal_success(self) -> None:
        result = NavigationResult(
            status="budget_exhausted",
            reason="max_steps",
            ticks=8,
            command_steps=8,
            elapsed_s=2.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            message = self._driver(result, directory).execute_action(
                "navigate_vln", {"instruction": "go to the sofa", "max_steps": 8}
            )
        self.assertTrue(message.startswith("Navigation action completed its step budget"))
        self.assertIn("not automatically verified", message)

    def test_rejects_unknown_and_invalid_actions(self) -> None:
        result = NavigationResult("completed", "test", 0, 0, 0.0)
        with tempfile.TemporaryDirectory() as directory:
            driver = self._driver(result, directory)
            self.assertTrue(driver.execute_action("move_to_pose", {}).startswith("Unknown"))
            self.assertTrue(
                driver.execute_action("navigate_vln", {}).startswith("Failed:")
            )
            self.assertTrue(
                driver.execute_action(
                    "navigate_vln",
                    {"instruction": "go", "max_steps": 9999},
                ).startswith("Failed:")
            )

    def test_profile_exists(self) -> None:
        result = NavigationResult("completed", "test", 0, 0, 0.0)
        with tempfile.TemporaryDirectory() as directory:
            profile = self._driver(result, directory).get_profile_path()
        self.assertTrue(profile.is_file())
        self.assertIn("navigate_vln", profile.read_text(encoding="utf-8"))

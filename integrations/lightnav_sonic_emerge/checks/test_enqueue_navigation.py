from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from Emerge.utils.action_queue import parse_action_markdown
from integrations.lightnav_sonic_emerge.enqueue_navigation import enqueue


class EnqueueNavigationTest(unittest.TestCase):
    def test_enqueues_expected_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            action_id = enqueue(
                directory,
                instruction="  move to the sofa  ",
                max_steps=20,
                control_hz=4.0,
            )
            payload = parse_action_markdown(
                (Path(directory) / "ACTION.md").read_text(encoding="utf-8")
            )

        action = payload["actions"][0]
        self.assertEqual(action["id"], action_id)
        self.assertEqual(action["action_type"], "navigate_vln")
        self.assertEqual(action["parameters"]["instruction"], "move to the sofa")
        self.assertEqual(action["status"], "pending")

    def test_rejects_second_active_action(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            enqueue(
                directory,
                instruction="move forward",
                max_steps=10,
                control_hz=4.0,
            )
            with self.assertRaises(RuntimeError):
                enqueue(
                    directory,
                    instruction="turn around",
                    max_steps=10,
                    control_hz=4.0,
                )

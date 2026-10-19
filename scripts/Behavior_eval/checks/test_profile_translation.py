"""Keep the human-readable R1Pro translation aligned with its runtime profile."""

from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ENGLISH = ROOT / "robot/profiles/behavior_r1pro.md"
CHINESE = ROOT / "robot/profiles/behavior_r1pro.zh-CN.md"
SOURCE_HASH = re.compile(r"^英文原文 SHA-256：([0-9a-f]{64})$", re.MULTILINE)
ACTION_HEADINGS = re.compile(r"^### `([^`]+)`$", re.MULTILINE)
JSON_EXAMPLES = re.compile(r"```json\n(.*?)\n```", re.DOTALL)


class R1ProProfileTranslationTest(unittest.TestCase):
    def test_chinese_copy_matches_current_english_source(self) -> None:
        english = ENGLISH.read_bytes()
        chinese = CHINESE.read_text(encoding="utf-8")
        match = SOURCE_HASH.search(chinese)
        self.assertIsNotNone(match, "Chinese profile needs the English source SHA-256")
        self.assertEqual(
            match.group(1),
            hashlib.sha256(english).hexdigest(),
            "English profile changed: update the Chinese translation and its source hash",
        )

    def test_action_names_and_json_examples_stay_identical(self) -> None:
        english = ENGLISH.read_text(encoding="utf-8")
        chinese = CHINESE.read_text(encoding="utf-8")
        self.assertEqual(ACTION_HEADINGS.findall(chinese), ACTION_HEADINGS.findall(english))
        self.assertEqual(
            [json.loads(example) for example in JSON_EXAMPLES.findall(chinese)],
            [json.loads(example) for example in JSON_EXAMPLES.findall(english)],
        )

    def test_profile_has_no_default_hand_preference(self) -> None:
        english = ENGLISH.read_text(encoding="utf-8")
        chinese = CHINESE.read_text(encoding="utf-8")
        self.assertIn("neither hand\n   has default priority", english)
        self.assertIn("左右手均无默认优先级", chinese)
        self.assertNotIn("should explicitly select the left", english)
        self.assertNotIn("先准备左手", chinese)
        self.assertNotIn("显式选择左手", chinese)


if __name__ == "__main__":
    unittest.main()

"""The BEHAVIOR workspace skill must shadow the builtin manipulation skill.

The builtin ``Emerge/skills/vla/SKILL.md`` requires a phase-local instruction,
while this checkpoint accepts only its exact frozen task sentence.  If the
workspace skill fails to shadow it, the Agent is told to do the opposite of what
the driver enforces.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from Emerge.agent.context import ContextBuilder
from Emerge.agent.skills import BUILTIN_SKILLS_DIR, SkillsLoader
from robot.behavior.contracts import OFFICIAL_PROMPT
from scripts.Behavior_eval.runner_common import install_workspace_skill

BUILTIN_VLA = BUILTIN_SKILLS_DIR / "vla" / "SKILL.md"


class WorkspaceSkillInstallationTest(unittest.TestCase):
    def test_install_writes_a_vla_skill_into_the_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            destination = install_workspace_skill(workspace)
            self.assertEqual(destination, workspace / "skills/vla/SKILL.md")
            self.assertTrue(destination.is_file())

    def test_missing_source_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                install_workspace_skill(Path(temporary), source=Path(temporary) / "nope.md")

    def test_workspace_skill_shadows_the_builtin(self) -> None:
        self.assertTrue(BUILTIN_VLA.is_file(), "the builtin vla skill must exist for this test")
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            install_workspace_skill(workspace)
            loader = SkillsLoader(workspace)

            loaded = loader.load_skill("vla")
            self.assertIsNotNone(loaded)
            self.assertIn("BEHAVIOR-1K Turning On Radio", loaded)
            self.assertIn(OFFICIAL_PROMPT, loaded)
            # The conflicting generic rule must be gone.
            self.assertNotIn("Phase-Local Instructions", loaded)

            vla_entries = [s for s in loader.list_skills(filter_unavailable=False) if s["name"] == "vla"]
            self.assertEqual(len(vla_entries), 1)
            self.assertEqual(vla_entries[0]["source"], "workspace")

    def test_workspace_skill_is_a_prompt_always_skill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            install_workspace_skill(workspace)
            loader = SkillsLoader(workspace)
            self.assertIn("vla", loader.get_always_skills())

    def test_system_prompt_contains_the_active_behavior_skill(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            install_workspace_skill(workspace)
            prompt = ContextBuilder(workspace).build_system_prompt()
            self.assertIn("BEHAVIOR-1K Turning On Radio", prompt)
            self.assertIn(OFFICIAL_PROMPT, prompt)
            self.assertNotIn("Phase-Local Instructions", prompt)

    def test_skill_frontmatter_is_parseable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            install_workspace_skill(workspace)
            metadata = SkillsLoader(workspace).get_skill_metadata("vla")
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata["name"], "vla")
            self.assertIn("BEHAVIOR-1K", metadata["description"])


if __name__ == "__main__":
    unittest.main()

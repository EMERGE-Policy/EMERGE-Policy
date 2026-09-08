"""Local skill registration for one sub-agent instance."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Skill:
    """A named SKILL.md file owned by one sub-agent."""

    name: str
    path: Path

    def read(self) -> str:
        """Read the skill instructions from disk."""
        return self.path.read_text(encoding="utf-8")


class SkillRegistry:
    """Explicit collection of skills visible to one sub-agent."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, name: str, path: Path) -> Skill:
        """Register one skill file under a unique name."""
        if not name.strip():
            raise ValueError("Skill name cannot be empty")
        if name in self._skills:
            raise ValueError(f"Skill '{name}' is already registered")

        skill_path = path.resolve()
        if not skill_path.is_file():
            raise FileNotFoundError(f"Skill file does not exist: {skill_path}")

        skill = Skill(name=name, path=skill_path)
        self._skills[name] = skill
        return skill

    def register_directory(self, root: Path) -> None:
        """Register each immediate ``<name>/SKILL.md`` below ``root``."""
        for directory in sorted(root.iterdir()):
            skill_file = directory / "SKILL.md"
            if directory.is_dir() and skill_file.is_file():
                self.register(directory.name, skill_file)

    def get(self, name: str) -> Skill:
        """Return a registered skill by name."""
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' is not registered")
        return self._skills[name]

    def render(self) -> str:
        """Render all registered instructions for the sub-agent context."""
        sections = []
        for skill in self._skills.values():
            body = self._instruction_body(skill.read())
            sections.append(f"## Skill: {skill.name}\n\n{body}")
        return "\n\n---\n\n".join(sections)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered skill names in registration order."""
        return tuple(self._skills)

    def __len__(self) -> int:
        return len(self._skills)

    @staticmethod
    def _instruction_body(content: str) -> str:
        """Remove optional YAML frontmatter before injecting instructions."""
        if not content.startswith("---"):
            return content.strip()

        parts = content.split("---", 2)
        return parts[2].strip() if len(parts) == 3 else content.strip()

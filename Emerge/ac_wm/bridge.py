"""Selected control sequence passed from AC-WM to the action dispatcher."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SelectedAction:
    candidate_id: str
    skill_name: str
    actions: tuple[tuple[float, ...], ...]
    metadata: dict[str, Any]

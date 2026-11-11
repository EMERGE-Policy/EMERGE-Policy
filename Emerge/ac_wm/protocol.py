"""Serializable contracts between the planner, skills, Cosmos3, and VLM judge."""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

# Candidate IDs become artifact file and directory names; keep them to one safe
# path component so planner-supplied IDs cannot escape the decision directory.
_CANDIDATE_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def is_valid_candidate_id(candidate_id: str) -> bool:
    return bool(_CANDIDATE_ID_PATTERN.fullmatch(candidate_id)) and ".." not in candidate_id


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    candidate_id: str
    skill_name: str
    actions: tuple[tuple[float, ...], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.skill_name.strip():
            raise ValueError("candidate_id and skill_name are required")
        if not is_valid_candidate_id(self.candidate_id):
            raise ValueError(
                f"candidate_id must be 1-128 characters of letters, digits, '.', '_' or '-' "
                f"and must not contain '..': {self.candidate_id!r}"
            )
        if not self.actions:
            raise ValueError("actions cannot be empty")
        width = len(self.actions[0])
        if width == 0 or any(len(row) != width for row in self.actions):
            raise ValueError("actions must be a non-empty rectangular matrix")
        if any(not math.isfinite(float(v)) for row in self.actions for v in row):
            raise ValueError("actions must contain only finite values")


@dataclass(frozen=True, slots=True)
class RolloutRequest:
    task: str
    observation_path: str
    candidates: tuple[ActionCandidate, ...]
    domain_name: str
    view_point: str | None = None
    output_dir: str = ""
    planning_context: Mapping[str, Any] = field(default_factory=dict)
    generation_errors: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.task.strip() or not self.observation_path.strip():
            raise ValueError("task and observation_path are required")
        if not self.candidates:
            raise ValueError("at least one candidate is required")
        ids = [c.candidate_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate_id values must be unique within a planning request")


@dataclass(frozen=True, slots=True)
class RolloutResult:
    candidate_id: str
    status: str
    video_path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate_id: str
    score: float
    rationale: str
    rollout: RolloutResult

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.score)) or not 0.0 <= self.score <= 1.0:
            raise ValueError("score must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_candidate_id: str | None
    evaluations: tuple[CandidateEvaluation, ...]
    status: str
    reason: str
    selection_method: str = "vlm_unique_top_score"
    tie_break: Mapping[str, Any] = field(default_factory=dict)

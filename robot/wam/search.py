"""Deterministic local search primitives for WAM candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True, eq=False)
class ScoredActionCandidate:
    """One candidate action chunk and its model score."""

    index: int
    seed: int
    actions: Any
    score: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("candidate index must be non-negative")
        if not math.isfinite(float(self.score)):
            raise ValueError("candidate score must be finite")


def select_best_candidate(
    candidates: Iterable[ScoredActionCandidate],
) -> ScoredActionCandidate:
    """Choose the highest score, preferring the first candidate on ties."""
    values = list(candidates)
    if not values:
        raise ValueError("at least one scored candidate is required")
    return max(values, key=lambda candidate: candidate.score)


def search_metadata(
    candidates: Iterable[ScoredActionCandidate],
    selected: ScoredActionCandidate,
    *,
    score_mode: str,
) -> dict[str, Any]:
    values = list(candidates)
    if not any(candidate is selected for candidate in values):
        raise ValueError("selected candidate must be present in candidates")
    return {
        "strategy": "best_of_n",
        "score_mode": str(score_mode),
        "num_candidates": len(values),
        "selected_index": selected.index,
        "selected_seed": selected.seed,
        "selected_score": float(selected.score),
        "candidate_seeds": [candidate.seed for candidate in values],
        "candidate_scores": [float(candidate.score) for candidate in values],
    }

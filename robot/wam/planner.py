"""Emerge-side candidate planning for the WAM client."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .protocol import validate_candidate_request
from .search import ScoredActionCandidate, search_metadata, select_best_candidate


@dataclass(frozen=True, slots=True)
class WAMPlan:
    """The selected action chunk and optional local search diagnostics."""

    actions: np.ndarray
    search_decision: dict[str, Any] | None = None


class CosmosWAMPlanner:
    """Request all candidates remotely and select one locally."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        candidate_request = validate_candidate_request(dict(config or {}))
        self.num_candidates = candidate_request["num_candidates"]
        self.score_mode = candidate_request["score_mode"]

    def plan(
        self,
        client: Any,
        observation: dict[str, Any],
        task_instruction: str,
        *,
        phase_instruction: str | None = None,
        conditioning_mode: str = "task",
        seed: int,
    ) -> WAMPlan:
        response = client.infer(
            **observation,
            task_instruction=task_instruction,
            phase_instruction=phase_instruction,
            conditioning_mode=conditioning_mode,
            seed=seed,
            num_candidates=self.num_candidates,
            score_mode=self.score_mode,
        )
        raw_candidates = response.get("candidates")
        if raw_candidates is None:
            if self.num_candidates != 1 or self.score_mode != "none":
                raise ValueError("Cosmos WAM response omitted requested candidates")
            return WAMPlan(np.asarray(response.get("actions"), dtype=np.float32))
        if not isinstance(raw_candidates, list) or len(raw_candidates) != self.num_candidates:
            raise ValueError("Cosmos WAM candidate count does not match local configuration")
        if response.get("score_mode", self.score_mode) != self.score_mode:
            raise ValueError("Cosmos WAM score mode does not match local configuration")
        if self.num_candidates == 1 and self.score_mode == "none":
            return WAMPlan(np.asarray(raw_candidates[0]["actions"], dtype=np.float32))

        candidates = [
            ScoredActionCandidate(
                index=int(candidate["index"]),
                seed=int(candidate["seed"]),
                actions=np.asarray(candidate["actions"], dtype=np.float32),
                score=float(candidate["score"]),
            )
            for candidate in raw_candidates
        ]
        selected = select_best_candidate(candidates)
        decision = search_metadata(candidates, selected, score_mode=self.score_mode)
        if isinstance(response.get("conditioning"), dict):
            decision["conditioning"] = dict(response["conditioning"])
        return WAMPlan(np.asarray(selected.actions, dtype=np.float32), decision)

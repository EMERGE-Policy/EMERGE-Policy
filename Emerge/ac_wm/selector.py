"""Candidate rollout and fair joint VLM selection."""

from collections.abc import Callable

from .protocol import (
    ActionCandidate,
    CandidateEvaluation,
    RolloutRequest,
    RolloutResult,
    SelectionResult,
)

RolloutFn = Callable[[RolloutRequest, ActionCandidate], RolloutResult]
JudgeFn = Callable[[str, ActionCandidate, RolloutResult], tuple[float, str]]
EventFn = Callable[[dict], None]


class AcWmSelector:
    def __init__(self, rollout: RolloutFn, judge) -> None:
        self.rollout, self.judge = rollout, judge

    def select(self, request: RolloutRequest, on_event: EventFn | None = None) -> SelectionResult:
        # Judge objects are reused across agent turns. Never let a failed/omitted
        # judge call inherit an earlier decision's response audit.
        previous_audit = getattr(self.judge, "last_audit", None)
        if isinstance(previous_audit, dict):
            previous_audit.clear()
        results = {}
        rollout_many = getattr(self.rollout, "run_many", None)
        if len(request.candidates) > 1 and callable(rollout_many):
            try:
                batched = rollout_many(request, request.candidates)
                for candidate in request.candidates:
                    result = batched.get(candidate.candidate_id)
                    if result is None or result.candidate_id != candidate.candidate_id:
                        raise ValueError(
                            f"batched rollout omitted or mismatched {candidate.candidate_id!r}"
                        )
                    results[candidate.candidate_id] = result
            except Exception as exc:
                for candidate in request.candidates:
                    results[candidate.candidate_id] = RolloutResult(
                        candidate.candidate_id,
                        "failed",
                        error=f"batched rollout error: {type(exc).__name__}: {exc}",
                    )
        else:
            for candidate in request.candidates:
                try:
                    result = self.rollout(request, candidate)
                    if result.candidate_id != candidate.candidate_id:
                        raise ValueError(f"rollout candidate id mismatch: {result.candidate_id}")
                except Exception as exc:
                    result = RolloutResult(
                        candidate.candidate_id,
                        "failed",
                        error=f"rollout error: {type(exc).__name__}: {exc}",
                    )
                results[candidate.candidate_id] = result
        for candidate in request.candidates:
            result = results[candidate.candidate_id]
            if on_event:
                on_event(
                    {
                        "type": "rollout.completed",
                        "candidate_id": candidate.candidate_id,
                        "status": result.status,
                        "video_path": result.video_path,
                        "metadata": dict(result.metadata),
                        "error": result.error,
                    }
                )

        valid = [
            c
            for c in request.candidates
            if results[c.candidate_id].status == "success" and results[c.candidate_id].video_path
        ]
        scores: dict[str, tuple[float, str]] = {}
        joint_judge = getattr(self.judge, "evaluate_candidates", None)
        if valid and callable(joint_judge):
            try:
                scores = joint_judge(
                    request, {c.candidate_id: results[c.candidate_id] for c in valid}
                )
                missing = {c.candidate_id for c in valid} - set(scores)
                if missing:
                    raise ValueError(f"joint judge omitted candidates: {sorted(missing)}")
                for candidate in valid:
                    score, rationale = scores[candidate.candidate_id]
                    if not 0.0 <= float(score) <= 1.0:
                        raise ValueError(
                            f"joint judge score out of range for {candidate.candidate_id}"
                        )
                    if on_event:
                        on_event(
                            {
                                "type": "candidate.judged",
                                "candidate_id": candidate.candidate_id,
                                "score": float(score),
                                "rationale": str(rationale),
                            }
                        )
            except Exception as exc:
                scores = {
                    c.candidate_id: (0.0, f"joint judge failed: {type(exc).__name__}: {exc}")
                    for c in valid
                }
        else:
            for candidate in valid:
                try:
                    score, rationale = self.judge(
                        request.task, candidate, results[candidate.candidate_id]
                    )
                    if not 0.0 <= float(score) <= 1.0:
                        raise ValueError("judge score outside [0,1]")
                    scores[candidate.candidate_id] = (float(score), str(rationale))
                except Exception as exc:
                    scores[candidate.candidate_id] = (
                        0.0,
                        f"judge error: {type(exc).__name__}: {exc}",
                    )
                if on_event:
                    score, rationale = scores[candidate.candidate_id]
                    on_event(
                        {
                            "type": "candidate.judged",
                            "candidate_id": candidate.candidate_id,
                            "score": score,
                            "rationale": rationale,
                        }
                    )

        evaluations = []
        for candidate in request.candidates:
            result = results[candidate.candidate_id]
            score, rationale = scores.get(
                candidate.candidate_id, (0.0, result.error or "rollout unavailable; not judged")
            )
            evaluations.append(
                CandidateEvaluation(candidate.candidate_id, float(score), str(rationale), result)
            )
        positive = [
            e
            for e in evaluations
            if e.rollout.status == "success" and e.rollout.video_path and e.score > 0
        ]
        if not positive:
            return SelectionResult(
                None,
                tuple(evaluations),
                "failed",
                "no successfully predicted candidate received a positive joint judge score",
            )
        best_score = max(e.score for e in positive)
        tied = [e for e in positive if abs(e.score - best_score) <= 1e-9]
        if len(tied) > 1:
            by_id = {candidate.candidate_id: candidate for candidate in request.candidates}
            priorities = {}
            for evaluation in tied:
                raw_priority = by_id[evaluation.candidate_id].metadata.get("planner_priority")
                if (
                    isinstance(raw_priority, bool)
                    or not isinstance(raw_priority, int)
                    or raw_priority < 1
                ):
                    priorities = {}
                    break
                priorities[evaluation.candidate_id] = raw_priority
            if priorities and len(set(priorities.values())) == len(priorities):
                chosen_id = min(priorities, key=priorities.get)
                tied_ids = [evaluation.candidate_id for evaluation in tied]
                audit = {
                    "policy": "planner_priority_positive_tie_break",
                    "candidate_ids": tied_ids,
                    "scores": {evaluation.candidate_id: evaluation.score for evaluation in tied},
                    "planner_priorities": priorities,
                    "selected_candidate_id": chosen_id,
                }
                return SelectionResult(
                    chosen_id,
                    tuple(evaluations),
                    "success",
                    "positive VLM top-score tie; selected the planner-preferred candidate by explicit priority",
                    "planner_priority_positive_tie_break",
                    audit,
                )
            tied_ids = sorted(e.candidate_id for e in tied)
            return SelectionResult(
                None,
                tuple(evaluations),
                "ambiguous",
                "joint judge assigned the same top score to candidates and no unique explicit planner_priority resolved the positive tie: "
                + ", ".join(tied_ids),
            )
        best = tied[0]
        return SelectionResult(best.candidate_id, tuple(evaluations), "success", best.rationale)

"""AC-WM planning mediator between the main agent and executable skills."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable

from Emerge.subagents.base import BaseSubagent
from Emerge.subagents.models import SubagentResult, SubagentTask

from .bridge import SelectedAction
from .protocol import RolloutRequest
from .selector import AcWmSelector
from .vlm_judge import _measure_action_contrast

CandidateProvider = Callable[[SubagentTask], Awaitable[RolloutRequest]]
SelectedActionDispatcher = Callable[[SubagentTask, SelectedAction], Awaitable[str]]

# Differences below this value over the dispatched prefix are only numerical
# noise for normalized LIBERO controls and do not define distinct actions.
_ACTION_EQUIVALENCE_EPSILON = 1e-5


def _deduplicate_action_candidates(candidates):
    """Collapse only proposals identical over the exact prefix dispatched now."""
    import numpy as np

    candidates = tuple(candidates)
    if len(candidates) < 2:
        return candidates, []
    representatives = []
    duplicates = []
    for candidate in candidates:
        actions = np.asarray(candidate.actions, dtype=np.float64)
        execute_steps = min(
            max(1, int(candidate.metadata.get("execute_steps", len(actions)))), len(actions)
        )
        duplicate = None
        for representative in representatives:
            rep_actions = np.asarray(representative.actions, dtype=np.float64)
            rep_execute_steps = min(
                max(1, int(representative.metadata.get("execute_steps", len(rep_actions)))),
                len(rep_actions),
            )
            if actions.shape[1] != rep_actions.shape[1] or execute_steps != rep_execute_steps:
                continue
            max_abs_difference = float(
                np.max(np.abs(actions[:execute_steps] - rep_actions[:rep_execute_steps]))
            )
            if max_abs_difference <= _ACTION_EQUIVALENCE_EPSILON:
                duplicate = representative
                duplicates.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "equivalent_to": representative.candidate_id,
                        "skill_name": candidate.skill_name,
                        "representative_skill_name": representative.skill_name,
                        "dispatched_action_steps": execute_steps,
                        "max_abs_action_difference": round(max_abs_difference, 10),
                        "equivalence_epsilon": _ACTION_EQUIVALENCE_EPSILON,
                    }
                )
                break
        if duplicate is None:
            representatives.append(candidate)
    return tuple(representatives), duplicates


class AcWmSubagent(BaseSubagent):
    """Generate multiple skill candidates, compare Cosmos futures jointly, dispatch winner."""

    def __init__(
        self,
        *,
        provider: Any,
        rollout,
        judge,
        candidate_provider: CandidateProvider | None = None,
        dispatch: SelectedActionDispatcher | None = None,
        timeout_s: float = 1800.0,
        dispatch_mode: str = "evaluated_prefix",
        on_no_positive: str = "reject",
    ):
        if dispatch_mode not in {"evaluated_prefix", "skill"}:
            raise ValueError(f"unsupported AC-WM dispatch_mode: {dispatch_mode!r}")
        if on_no_positive not in {"reject", "planner_priority"}:
            raise ValueError(f"unsupported AC-WM on_no_positive: {on_no_positive!r}")
        super().__init__(
            name="ac-wm",
            description=(
                "Internal planning mediator: gathers multiple action proposals from executable skills, "
                "predicts each from the same current observation with Cosmos3, jointly asks the VLM "
                "to compare plan/action/future evidence, then executes only the selected proposal."
            ),
            system_prompt="Coordinate planner context and multiple low-level skill proposals. Keep one shared observation, compare action-conditioned futures jointly, preserve evidence, and dispatch only the exact winner.",
            provider=provider,
            capabilities=(
                "multi-skill-candidate-generation",
                "shared-state-action-conditioned-prediction",
                "joint-vlm-ranking",
                "selected-action-dispatch",
                "decision-audit",
            ),
            input_modalities=("text", "image"),
            max_iterations=1,
        )
        self._selector = AcWmSelector(rollout, judge)
        self._candidate_provider = candidate_provider
        self._dispatch = dispatch
        self._timeout_s = timeout_s
        self._dispatch_mode = dispatch_mode
        self._on_no_positive = on_no_positive
        # asyncio.wait_for cannot stop a selection thread. Serialize selections so
        # an abandoned (timed-out) run cannot overlap the next decision or rewrite
        # the shared judge audit while that decision is being recorded.
        self._selection_lock = threading.Lock()

    @staticmethod
    def _dispatch_outcome(result: str) -> str:
        """Classify the controller result text returned for the selected-action dispatch."""
        lowered = result.strip().lower()
        if lowered.startswith(("error:", "failed:")):
            return "failed"
        if (
            lowered.startswith("interrupted:")
            or "execution cancelled" in lowered
            or "execution canceled" in lowered
            or "execution interrupted" in lowered
        ):
            return "interrupted"
        return "completed"

    @staticmethod
    def _json_default(value):
        if isinstance(value, Path):
            return str(value)
        return repr(value)

    @classmethod
    def _write_json(cls, path: Path, payload: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=cls._json_default) + "\n"
        )
        os.replace(tmp, path)

    @classmethod
    def _append_event(cls, path: Path, event: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp": time.time(), **event}
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=cls._json_default) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _request_record(request: RolloutRequest):
        return {
            "task": request.task,
            "planning_context": dict(request.planning_context),
            "observation_path": request.observation_path,
            "view_point": request.view_point,
            "domain_name": request.domain_name,
            "generation_errors": list(request.generation_errors),
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "skill_name": c.skill_name,
                    "action_rows": len(c.actions),
                    "action_width": len(c.actions[0]),
                    "execute_steps": c.metadata.get("execute_steps"),
                    "prediction_steps": c.metadata.get("prediction_steps", len(c.actions)),
                    "intent": c.metadata.get("intent", c.metadata.get("reasoning", "")),
                    "skill_parameters": c.metadata.get("skill_parameters", {}),
                    "actions": c.actions,
                    "metadata": dict(c.metadata),
                }
                for c in request.candidates
            ],
        }

    async def _run(self, task: SubagentTask) -> SubagentResult:
        raw_dir = task.input.get("output_dir")
        artifact_dir = Path(str(raw_dir)).expanduser().resolve() if raw_dir else None
        manifest = artifact_dir / "decision.json" if artifact_dir else None
        events = artifact_dir / "events.jsonl" if artifact_dir else None
        if manifest:
            self._write_json(
                manifest,
                {
                    "schema": "Emerge.ac_wm_decision.v1",
                    "stage": "started",
                    "task_id": task.task_id,
                    "instruction": task.input.get("instruction"),
                    "planning_context": task.input.get("planning_context", {}),
                    "dispatch": {"attempted": False},
                },
            )
        request = task.input.get("request")
        if not isinstance(request, RolloutRequest):
            if self._candidate_provider is None:
                error = "AC-WM has no executable skill candidate provider"
                if manifest:
                    self._write_json(
                        manifest,
                        {
                            "schema": "Emerge.ac_wm_decision.v1",
                            "stage": "failed",
                            "error": error,
                            "dispatch": {"attempted": False},
                        },
                    )
                return SubagentResult.failure(
                    task,
                    self.name,
                    error,
                    metadata={"artifact_dir": str(artifact_dir) if artifact_dir else None},
                )
            try:
                request = await self._candidate_provider(task)
            except Exception as exc:
                error = f"skill candidate generation failed: {type(exc).__name__}: {exc}"
                if events:
                    self._append_event(
                        events, {"type": "candidate_generation.failed", "error": error}
                    )
                if manifest:
                    self._write_json(
                        manifest,
                        {
                            "schema": "Emerge.ac_wm_decision.v1",
                            "stage": "candidate_generation_failed",
                            "error": error,
                            "instruction": task.input.get("instruction"),
                            "planning_context": task.input.get("planning_context", {}),
                            "dispatch": {"attempted": False},
                        },
                    )
                return SubagentResult.failure(
                    task,
                    self.name,
                    error,
                    metadata={"artifact_dir": str(artifact_dir) if artifact_dir else None},
                )
        if not isinstance(request, RolloutRequest):
            return SubagentResult.failure(
                task, self.name, "candidate provider must return RolloutRequest"
            )
        effective_candidates, duplicate_actions = _deduplicate_action_candidates(request.candidates)
        if len(request.candidates) > 1 and len(effective_candidates) < 2:
            decision_mode = "equivalent_candidate_rejection"
        elif len(effective_candidates) > 1:
            decision_mode = "joint_candidate_comparison"
        else:
            decision_mode = "single_candidate_screen"
        action_contrast = _measure_action_contrast(request.candidates)
        if events:
            self._append_event(
                events,
                {
                    "type": "candidates.ready",
                    "decision_mode": decision_mode,
                    "candidate_ids": [c.candidate_id for c in request.candidates],
                    "effective_candidate_ids": [c.candidate_id for c in effective_candidates],
                    "candidate_count": len(request.candidates),
                    "observation_path": request.observation_path,
                    "planning_context": dict(request.planning_context),
                    "generation_errors": list(request.generation_errors),
                    "action_contrast_diagnostic": action_contrast,
                    "equivalent_action_candidates": duplicate_actions,
                },
            )
        if manifest:
            self._write_json(
                manifest,
                {
                    "schema": "Emerge.ac_wm_decision.v1",
                    "stage": "candidates_ready",
                    "task_id": task.task_id,
                    "decision_mode": decision_mode,
                    "request": self._request_record(request),
                    "action_contrast_diagnostic": action_contrast,
                    "effective_candidate_ids": [c.candidate_id for c in effective_candidates],
                    "equivalent_action_candidates": duplicate_actions,
                    "dispatch": {"attempted": False},
                },
            )
        if len(request.candidates) > 1 and len(effective_candidates) < 2:
            reason = (
                "all proposed skills produce equivalent action prefixes over the dispatched control window; "
                "re-propose at least two candidates with materially different executable controls"
            )
            if events:
                self._append_event(
                    events,
                    {
                        "type": "candidates.equivalent_only",
                        "decision_mode": decision_mode,
                        "reason": reason,
                        "equivalent_action_candidates": duplicate_actions,
                    },
                )
            if manifest:
                self._write_json(
                    manifest,
                    {
                        "schema": "Emerge.ac_wm_decision.v1",
                        "stage": "no_distinct_action_candidates",
                        "task_id": task.task_id,
                        "decision_mode": decision_mode,
                        "request": self._request_record(request),
                        "action_contrast_diagnostic": action_contrast,
                        "effective_candidate_ids": [c.candidate_id for c in effective_candidates],
                        "equivalent_action_candidates": duplicate_actions,
                        "error": reason,
                        "dispatch": {"attempted": False},
                    },
                )
            return SubagentResult.failure(
                task,
                self.name,
                reason,
                metadata={
                    "evaluations": [],
                    "dispatched": False,
                    "decision_mode": decision_mode,
                    "selection_status": "no_distinct_action_candidates",
                    "equivalent_action_candidates": duplicate_actions,
                    "artifact_dir": str(artifact_dir) if artifact_dir else None,
                },
            )
        effective_request = (
            request
            if effective_candidates == request.candidates
            else replace(request, candidates=effective_candidates)
        )
        abandoned = threading.Event()

        def event_sink(event):
            # A timed-out selection may still be running; never let it append to
            # a decision log that has already been closed as selection_timeout.
            if events and not abandoned.is_set():
                self._append_event(events, event)

        def select_exclusive():
            with self._selection_lock:
                if abandoned.is_set():
                    raise TimeoutError("AC-WM selection abandoned before it started")
                selection = self._selector.select(effective_request, event_sink)
                audit = getattr(self._selector.judge, "last_audit", None)
                return selection, (
                    json.loads(json.dumps(audit, default=self._json_default))
                    if isinstance(audit, dict)
                    else None
                )

        def abandon_selection(stage: str):
            abandoned.set()
            if events:
                self._append_event(
                    events,
                    {
                        "type": "selection.timeout"
                        if stage == "selection_timeout"
                        else "selection.cancelled",
                        "decision_mode": decision_mode,
                        "note": "background rollout/judge work may still finish; its results are discarded",
                    },
                )
            if manifest:
                self._write_json(
                    manifest,
                    {
                        "schema": "Emerge.ac_wm_decision.v1",
                        "stage": stage,
                        "request": self._request_record(request),
                        "dispatch": {"attempted": False},
                    },
                )

        try:
            selection, judge_audit = await asyncio.wait_for(
                asyncio.to_thread(select_exclusive), timeout=task.timeout or self._timeout_s
            )
        except asyncio.TimeoutError:
            abandon_selection("selection_timeout")
            return SubagentResult.failure(
                task,
                self.name,
                "AC-WM rollout/joint judge timed out",
                metadata={"artifact_dir": str(artifact_dir) if artifact_dir else None},
            )
        except asyncio.CancelledError:
            # BaseSubagent.run enforces task.timeout by cancelling this coroutine.
            abandon_selection("selection_cancelled")
            raise
        except Exception as exc:
            error = f"AC-WM selection failed: {type(exc).__name__}: {exc}"
            if manifest:
                self._write_json(
                    manifest,
                    {
                        "schema": "Emerge.ac_wm_decision.v1",
                        "stage": "selection_failed",
                        "request": self._request_record(request),
                        "error": error,
                        "dispatch": {"attempted": False},
                    },
                )
            return SubagentResult.failure(
                task,
                self.name,
                error,
                metadata={"artifact_dir": str(artifact_dir) if artifact_dir else None},
            )
        evaluations = [
            {
                "candidate_id": e.candidate_id,
                "score": e.score,
                "rationale": e.rationale,
                "rollout_status": e.rollout.status,
                "predicted_video": e.rollout.video_path,
                "rollout_metadata": dict(e.rollout.metadata),
                "rollout_error": e.rollout.error,
            }
            for e in selection.evaluations
        ]
        common = {
            "schema": "Emerge.ac_wm_decision.v1",
            "task_id": task.task_id,
            "decision_mode": decision_mode,
            "request": self._request_record(request),
            "candidate_evaluations": evaluations,
            "action_contrast_diagnostic": action_contrast,
            "effective_candidate_ids": [c.candidate_id for c in effective_candidates],
            "equivalent_action_candidates": duplicate_actions,
            "judge_audit": judge_audit if isinstance(judge_audit, dict) else None,
            "selected_candidate_id": selection.selected_candidate_id,
            "selection_status": selection.status,
            "selection_reason": selection.reason,
            "selection_method": selection.selection_method,
            "tie_break": dict(selection.tie_break),
            "dispatch": {"attempted": False},
        }
        fallback = None
        if selection.selected_candidate_id is None:
            ambiguous = selection.status == "ambiguous"
            common["stage"] = "ambiguous_selection" if ambiguous else "no_positive_candidate"
            common["selection_status"] = selection.status
            if events:
                self._append_event(
                    events,
                    {
                        "type": "selection.ambiguous" if ambiguous else "selection.rejected",
                        "decision_mode": decision_mode,
                        "reason": selection.reason,
                        "candidate_evaluations": evaluations,
                    },
                )
            fallback = (
                self._fallback_candidate(request)
                if (
                    self._on_no_positive == "planner_priority"
                    and self._dispatch is not None
                    and not isinstance(task.input.get("request"), RolloutRequest)
                )
                else None
            )
            if fallback is None:
                if manifest:
                    self._write_json(manifest, common)
                return SubagentResult.failure(
                    task,
                    self.name,
                    selection.reason,
                    metadata={
                        "evaluations": evaluations,
                        "dispatched": False,
                        "selection_status": selection.status,
                        "decision_mode": decision_mode,
                        "artifact_dir": str(artifact_dir) if artifact_dir else None,
                    },
                )
        if self._dispatch is None and isinstance(task.input.get("request"), RolloutRequest):
            common["stage"] = "selected_not_dispatched"
            if manifest:
                self._write_json(manifest, common)
            return SubagentResult.success(
                task,
                self.name,
                selection.reason,
                output=selection,
                metadata={
                    "selected_candidate_id": selection.selected_candidate_id,
                    "decision_mode": decision_mode,
                    "evaluations": evaluations,
                    "dispatched": False,
                    "artifact_dir": str(artifact_dir) if artifact_dir else None,
                },
            )
        # A fallback is the planner's own first choice run as an ordinary skill call. It is
        # recorded separately and is never counted as an AC-WM selection.
        via = "fallback" if fallback is not None else "selection"
        chosen = (
            fallback
            if fallback is not None
            else next(
                c for c in request.candidates if c.candidate_id == selection.selected_candidate_id
            )
        )
        best = next(
            (e for e in selection.evaluations if e.candidate_id == chosen.candidate_id), None
        )
        mode = "skill" if via == "fallback" else self._dispatch_mode
        proposed_steps = max(
            1, min(int(chosen.metadata.get("execute_steps", 1)), len(chosen.actions))
        )
        evaluated_horizon = (
            best.rollout.metadata.get("action_horizon") if best is not None else None
        )
        dispatch_steps = (
            proposed_steps
            if evaluated_horizon is None
            else min(proposed_steps, int(evaluated_horizon))
        )
        if dispatch_steps < 1:
            raise RuntimeError("selected AC-WM candidate has no evaluated executable prefix")
        selected_metadata = dict(chosen.metadata)
        selected_metadata.update(
            {
                "execute_steps": dispatch_steps,
                "ac_wm_evaluated_action_horizon": evaluated_horizon,
                "ac_wm_proposed_action_rows": len(chosen.actions),
                "ac_wm_prediction_steps": chosen.metadata.get(
                    "prediction_steps", len(chosen.actions)
                ),
                "ac_wm_dispatch_mode": mode,
                "ac_wm_dispatch_via": via,
            }
        )
        selected = SelectedAction(
            chosen.candidate_id, chosen.skill_name, chosen.actions, selected_metadata
        )
        if mode == "skill":
            planned = {
                "skill_action_type": chosen.metadata.get("skill_action_type"),
                "parameters": chosen.metadata.get(
                    "planner_parameters", chosen.metadata.get("skill_parameters")
                ),
            }
            dispatch_record = {
                "mode": "skill",
                "via": via,
                "candidate_id": chosen.candidate_id,
                "skill_name": chosen.skill_name,
                "evaluated_action_horizon": evaluated_horizon,
                "skill_call": planned,
            }
        else:
            dispatch_record = {
                "mode": "evaluated_prefix",
                "via": via,
                "candidate_id": chosen.candidate_id,
                "skill_name": chosen.skill_name,
                "evaluated_action_horizon": evaluated_horizon,
                "proposed_action_rows": len(chosen.actions),
                "requested_prediction_steps": chosen.metadata.get(
                    "prediction_steps", len(chosen.actions)
                ),
                "executed_steps": dispatch_steps,
                "executed_action_rows": chosen.actions[:dispatch_steps],
            }
        stage_prefix = "fallback_" if via == "fallback" else ""
        event_prefix = "dispatch.fallback_" if via == "fallback" else "dispatch."
        common["stage"] = "fallback_selected" if via == "fallback" else "selected"
        common["dispatch"] = {"attempted": False, **dispatch_record}
        if via == "fallback":
            common["fallback"] = {
                "policy": "planner_priority",
                "candidate_id": chosen.candidate_id,
                "planner_priority": chosen.metadata.get("planner_priority"),
                "reason": selection.reason,
            }
        else:
            if events:
                self._append_event(
                    events,
                    {
                        "type": "selection.selected",
                        "decision_mode": decision_mode,
                        "candidate_id": chosen.candidate_id,
                        "skill_name": chosen.skill_name,
                        "score": best.score,
                        "rationale": best.rationale,
                        "selection_method": selection.selection_method,
                        "tie_break": dict(selection.tie_break),
                        "evaluated_action_horizon": evaluated_horizon,
                        "requested_prediction_steps": chosen.metadata.get(
                            "prediction_steps", len(chosen.actions)
                        ),
                        "proposed_action_rows": len(chosen.actions),
                        "executed_steps": dispatch_steps,
                        "dispatch_mode": mode,
                    },
                )
        if manifest:
            self._write_json(manifest, common)
        dispatch_result = None
        if self._dispatch is not None:
            common["dispatch"]["attempted"] = True
            if events:
                self._append_event(
                    events,
                    {
                        "type": event_prefix + "started",
                        "decision_mode": decision_mode,
                        "selection_method": selection.selection_method,
                        "tie_break": dict(selection.tie_break),
                        **dispatch_record,
                    },
                )
            # Persist intent before awaiting an external action; cancellation must
            # not leave the durable decision manifest claiming dispatch never began.
            if manifest:
                self._write_json(manifest, common)
            try:
                dispatch_result = await self._dispatch(task, selected)
            except asyncio.CancelledError:
                common["stage"] = stage_prefix + "dispatch_interrupted"
                common["dispatch"].update(
                    {
                        "success": None,
                        "status": "interrupted",
                        "error": "selection task cancelled while selected action dispatch was in flight",
                    }
                )
                if events:
                    self._append_event(
                        events,
                        {
                            "type": event_prefix + "interrupted",
                            "candidate_id": chosen.candidate_id,
                            "executed_steps": dispatch_steps,
                        },
                    )
                if manifest:
                    self._write_json(manifest, common)
                raise
            except Exception as exc:
                common["stage"] = stage_prefix + "dispatch_failed"
                common["dispatch"].update(
                    {"success": False, "error": f"{type(exc).__name__}: {exc}"}
                )
                if events:
                    self._append_event(
                        events,
                        {
                            "type": event_prefix + "failed",
                            "decision_mode": decision_mode,
                            "candidate_id": chosen.candidate_id,
                            "error": common["dispatch"]["error"],
                        },
                    )
                if manifest:
                    self._write_json(manifest, common)
                return SubagentResult.failure(
                    task,
                    self.name,
                    f"selected action dispatch failed: {type(exc).__name__}: {exc}",
                    metadata={
                        "evaluations": evaluations,
                        "selected_candidate_id": selected.candidate_id,
                        "dispatched": False,
                        "artifact_dir": str(artifact_dir) if artifact_dir else None,
                    },
                )
            outcome = self._dispatch_outcome(dispatch_result)
            if outcome == "interrupted":
                # The selected prefix may have been partially executed; never record it as a completed dispatch.
                common["stage"] = stage_prefix + "dispatch_interrupted"
                common["dispatch"].update(
                    {"success": None, "status": "interrupted", "result": dispatch_result}
                )
                if events:
                    self._append_event(
                        events,
                        {
                            "type": event_prefix + "interrupted",
                            "decision_mode": decision_mode,
                            "candidate_id": chosen.candidate_id,
                            "skill_name": chosen.skill_name,
                            "result": dispatch_result,
                        },
                    )
            else:
                failure = outcome == "failed"
                common["dispatch"].update(
                    {"success": not failure, "status": outcome, "result": dispatch_result}
                )
                common["stage"] = stage_prefix + ("dispatch_failed" if failure else "dispatched")
                if events:
                    self._append_event(
                        events,
                        {
                            "type": event_prefix + ("failed" if failure else "completed"),
                            "decision_mode": decision_mode,
                            "candidate_id": chosen.candidate_id,
                            "skill_name": chosen.skill_name,
                            "success": not failure,
                            "result": dispatch_result,
                        },
                    )
            if manifest:
                self._write_json(manifest, common)
            if outcome != "completed":
                return SubagentResult.failure(
                    task,
                    self.name,
                    dispatch_result,
                    metadata={
                        "evaluations": evaluations,
                        "selected_candidate_id": selected.candidate_id,
                        "dispatched": False,
                        "dispatch_status": outcome,
                        "dispatch_via": via,
                        "artifact_dir": str(artifact_dir) if artifact_dir else None,
                    },
                )
        output = {
            "selected_candidate_id": selected.candidate_id if via == "selection" else None,
            "dispatched_candidate_id": selected.candidate_id,
            "dispatch_via": via,
            "dispatch_mode": mode,
            "skill_name": selected.skill_name,
            "score": best.score if best is not None else None,
            "rationale": best.rationale if best is not None else selection.reason,
            "rollout_video": best.rollout.video_path if best is not None else None,
            "dispatch_result": dispatch_result,
            "decision_manifest": str(manifest) if manifest else None,
        }
        output["decision_mode"] = decision_mode
        output["selection_method"] = (
            "planner_priority_fallback" if via == "fallback" else selection.selection_method
        )
        output["tie_break"] = dict(selection.tie_break)
        summary = (
            (
                f"AC-WM found no candidate with a positive forecast score ({selection.reason}); "
                f"executed the planner's first choice {selected.candidate_id} ({selected.skill_name}) as an ordinary skill call. "
                "Re-read ROBOT_STATE.md before the next step."
            )
            if via == "fallback"
            else (
                f"AC-WM selected {selected.candidate_id} ({selected.skill_name}) at score {best.score:.3f}"
                + (
                    " and executed it"
                    + (" as an ordinary skill call" if mode == "skill" else " (evaluated prefix)")
                    if self._dispatch
                    else ""
                )
            )
        )
        return SubagentResult.success(
            task,
            self.name,
            summary,
            output=output,
            metadata={
                "evaluations": evaluations,
                "decision_mode": decision_mode,
                "selected_candidate_id": selected.candidate_id if via == "selection" else None,
                "dispatch_via": via,
                "dispatch_mode": mode,
                "dispatched": self._dispatch is not None,
                "artifact_dir": str(artifact_dir) if artifact_dir else None,
                "decision_manifest": str(manifest) if manifest else None,
                "event_log": str(events) if events else None,
            },
        )

    @staticmethod
    def _fallback_candidate(request):
        """Planner's first choice (lowest explicit planner_priority), if every candidate has one."""
        ranked = [
            c
            for c in request.candidates
            if isinstance(c.metadata.get("planner_priority"), int)
            and not isinstance(c.metadata.get("planner_priority"), bool)
        ]
        if not ranked or len(ranked) != len(request.candidates):
            return None
        return min(ranked, key=lambda c: c.metadata["planner_priority"])

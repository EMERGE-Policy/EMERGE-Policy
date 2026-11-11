"""Optional AC-WM candidate proposal and dispatch support for robot actions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4


class AcWmActionMixin:
    """Share the normal action queue while keeping ordinary actions direct."""

    def _ac_wm_policy_text(self) -> str:
        skill_mode = self.ac_wm_subagent._dispatch_mode == "skill"
        fallback = self.ac_wm_subagent._on_no_positive == "planner_priority"
        backend = os.environ.get("EMERGE_POLICY_BACKEND", "").strip().lower()
        candidates = (
            "This WAM-only run accepts geometric candidates; wam_execute remains a direct policy action. "
            if backend == "wam"
            else "Compare a VLA retry against a measured geometric re-anchoring move when both are plausible. "
            "Keep the original task clause in VLA candidate instructions; wording-only variants are not distinct actions. "
        )
        return (
            "AC-WM adds optional Cosmos3 prediction alongside the existing policy backends. "
            "Use ac_wm_select for recovery after a failed grasp or placement, at most once per subgoal. "
            "For first attempts and routine actions, use the active policy or geometry skill directly. "
            "Supply 2–8 materially different skill candidates, current plan_context, and unique planner_priority values (1 is preferred). "
            + candidates
            + "Proposals share one frozen observation and a VLM jointly evaluates their predicted futures. "
            + (
                "The selected skill runs with its original parameters. "
                if skill_mode
                else "Only the selected evaluated action prefix executes. "
            )
            + (
                "If no positive candidate is selected, the planner's first choice runs as an ordinary skill. "
                if fallback
                else "If no positive candidate is selected, no action executes. "
            )
            + "Re-read ROBOT_STATE.md after selection. "
        )

    async def _execute_action_through_ac_wm(
        self, action_type: str, parameters: dict[str, Any], reasoning: str
    ) -> str:
        from Emerge.subagents.content import TextContent
        from Emerge.subagents.models import SubagentTask

        specs = parameters.get("candidates")
        if not isinstance(specs, list) or not 2 <= len(specs) <= 8:
            return "Error: ac_wm_select requires 2..8 candidate skill specs"
        plan_context = parameters.get("plan_context") or {}
        if not isinstance(plan_context, dict):
            return "Error: plan_context must be an object"
        instruction = str(
            plan_context.get("current_subgoal") or parameters.get("instruction") or reasoning
        ).strip()
        if not instruction:
            return "Error: ac_wm_select requires plan_context.current_subgoal or instruction"
        candidate_specs = [dict(spec) for spec in specs]
        try:
            step = max(1, min(int(parameters.get("step", 64)), 300))
        except (TypeError, ValueError):
            return "Error: ac_wm_select step must be an integer"
        output_dir = self.workspace / "artifacts" / "ac-wm" / uuid4().hex
        task = SubagentTask(
            content=(TextContent(instruction),),
            input={
                "instruction": instruction,
                "step": step,
                "action_type": action_type,
                "candidate_specs": candidate_specs,
                "planning_context": plan_context,
                "parameters": dict(parameters),
                "reasoning": reasoning,
                "output_dir": str(output_dir),
            },
            timeout=2400.0,
        )
        result = await self.ac_wm_subagent.run(task)
        return json.dumps(
            {
                "status": result.status.value,
                "agent": "ac-wm",
                "summary": result.summary,
                "output": result.output,
                "error": result.error,
                "metadata": result.metadata,
            },
            ensure_ascii=False,
            default=str,
        )

    async def propose_action_candidates(self, task):
        """Collect alternative low-level skill proposals against one frozen observation."""
        from Emerge.ac_wm.protocol import ActionCandidate, RolloutRequest, is_valid_candidate_id

        action_type = str(task.input["action_type"])
        multi = action_type == "ac_wm_select"
        specs = task.input.get("candidate_specs") or []
        if not specs and not multi:
            # Backward-compatible single-skill mediation: promote the main-agent
            # action into one proposal. Explicit ac_wm_select still requires
            # multiple independently generated candidates.
            specs = [
                {
                    "candidate_id": f"{action_type}-single",
                    "skill": action_type,
                    "parameters": dict(task.input.get("parameters") or {}),
                    "reasoning": str(
                        task.input.get("reasoning") or task.input.get("instruction") or ""
                    ),
                }
            ]
        if not specs:
            raise RuntimeError("no candidate skill specs supplied")
        if multi:
            priorities = [spec.get("planner_priority") for spec in specs]
            if any(
                isinstance(priority, bool) or not isinstance(priority, int) or priority < 1
                for priority in priorities
            ):
                raise ValueError(
                    "ac_wm_select requires an explicit positive integer planner_priority on every candidate (1 is most preferred)"
                )
            if len(set(priorities)) != len(priorities):
                raise ValueError("ac_wm_select planner_priority values must be unique")
        output_dir = Path(task.input["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        capture = await self._dispatch_action(
            "capture_ac_wm_observation", {}, "AC-WM captures shared planning observation"
        )
        marker = "AC_WM_OBSERVATION:"
        if marker not in capture:
            raise RuntimeError(f"could not capture shared AC-WM observation: {capture}")
        shared = json.loads(capture.split(marker, 1)[1].strip())
        observation_path = str(shared["observation_path"])
        observation_manifest = Path(observation_path).with_suffix(".json")
        observation_info = (
            json.loads(observation_manifest.read_text()) if observation_manifest.is_file() else {}
        )
        candidates = []
        errors = []
        seen = set()
        allowed = {"vla_execute", "move_to_pose", "move_linear", "set_gripper", "follow_arc"}
        for index, spec in enumerate(specs):
            skill_params = dict(spec.get("parameters") or {})
            planner_parameters = dict(skill_params)
            skill = str(spec.get("skill") or self._infer_candidate_skill(skill_params)).strip()
            candidate_id = str(
                spec.get("candidate_id") or f"{skill or 'skill'}-{index + 1}"
            ).strip()
            intent = str(
                spec.get("reasoning", skill_params.get("instruction", task.input["instruction"]))
            ).strip()
            try:
                if (
                    skill == "vla_execute"
                    and os.environ.get("EMERGE_POLICY_BACKEND", "").lower() == "wam"
                ):
                    raise ValueError("VLA candidates are disabled by EMERGE_POLICY_BACKEND=wam")
                if skill not in allowed:
                    raise ValueError(f"unsupported candidate skill: {skill!r}")
                if not candidate_id or candidate_id in seen:
                    raise ValueError(f"candidate_id is empty or duplicated: {candidate_id!r}")
                if not is_valid_candidate_id(candidate_id):
                    raise ValueError(
                        f"candidate_id must use only letters, digits, '.', '_' or '-' (no '..'): {candidate_id!r}"
                    )
                seen.add(candidate_id)
                if skill == "vla_execute":
                    proposal_instruction = str(
                        skill_params.get(
                            "instruction", skill_params.get("prompt", task.input["instruction"])
                        )
                    ).strip()
                    horizon = max(
                        1,
                        min(
                            int(
                                spec.get(
                                    "prediction_steps", skill_params.get("prediction_steps", 52)
                                )
                            ),
                            64,
                        ),
                    )
                    raw = await self._dispatch_action(
                        "vla_propose",
                        {
                            "instruction": proposal_instruction,
                            "horizon": horizon,
                            "observation_path": observation_path,
                            "observation_revision": shared.get("observation_revision", 0),
                        },
                        f"AC-WM candidate {candidate_id}: VLA proposal only",
                    )
                    proposal_marker = "VLA_PROPOSAL:"
                    skill_name = str(spec.get("skill_name") or "vla")
                else:
                    prediction_horizon = max(
                        4,
                        min(
                            int(
                                spec.get(
                                    "prediction_steps", skill_params.get("prediction_steps", 52)
                                )
                            ),
                            64,
                        ),
                    )
                    skill_params["steps"] = prediction_horizon
                    raw = await self._dispatch_action(
                        "rule_propose",
                        {
                            "skill_action_type": skill,
                            "parameters": skill_params,
                            "observation_path": observation_path,
                            "observation_revision": shared.get("observation_revision", 0),
                        },
                        f"AC-WM candidate {candidate_id}: {skill} preview only",
                    )
                    proposal_marker = "RULE_PROPOSAL:"
                    skill_name = str(spec.get("skill_name") or f"rule:{skill}")
                if proposal_marker not in raw:
                    raise RuntimeError(raw)
                proposal = json.loads(raw.split(proposal_marker, 1)[1].strip())
                actions = tuple(tuple(float(v) for v in row) for row in proposal["actions"])
                execute_steps = max(
                    1,
                    min(
                        int(spec.get("execute_steps", proposal.get("execute_steps", len(actions)))),
                        len(actions),
                    ),
                )
                prediction_steps = max(
                    1,
                    min(
                        int(
                            spec.get(
                                "prediction_steps", proposal.get("prediction_steps", len(actions))
                            )
                        ),
                        len(actions),
                    ),
                )
                candidates.append(
                    ActionCandidate(
                        candidate_id,
                        skill_name,
                        actions,
                        {
                            "execute_steps": execute_steps,
                            "prediction_steps": prediction_steps,
                            **(
                                {"planner_priority": spec["planner_priority"]}
                                if isinstance(spec.get("planner_priority"), int)
                                and not isinstance(spec.get("planner_priority"), bool)
                                else {}
                            ),
                            "source": "live_skill_proposal",
                            "skill_action_type": skill,
                            "intent": intent,
                            "reasoning": str(spec.get("reasoning", "")),
                            "skill_parameters": skill_params,
                            "planner_parameters": planner_parameters,
                            "skill_inferred": not spec.get("skill"),
                            "observation_revision": proposal.get(
                                "observation_revision", shared.get("observation_revision")
                            ),
                            "action_control": {
                                key: proposal[key]
                                for key in (
                                    "position_scale",
                                    "rotation_scale",
                                    "position_gain",
                                    "rotation_gain",
                                )
                                if key in proposal
                            },
                            "observation_camera": (
                                observation_info.get("camera_names")
                                or [observation_info.get("camera_name", "agentview")]
                            )[0],
                            "view_point": observation_info.get("view_point", "third_person_view"),
                        },
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "candidate_id": candidate_id,
                        "skill": skill,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        tolerate_single = (
            getattr(self.ac_wm_subagent, "_on_no_positive", "reject") == "planner_priority"
            and len(candidates) == 1
        )
        if multi and len(candidates) < 2 and not tolerate_single:
            failure = {
                "stage": "candidate_generation_failed",
                "planning_context": task.input.get("planning_context", {}),
                "shared_observation": shared,
                "requested_candidate_count": len(specs),
                "generated_candidate_ids": [c.candidate_id for c in candidates],
                "errors": errors,
            }
            (output_dir / "proposal-summary.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2) + "\n"
            )
            raise RuntimeError(
                f"ac_wm_select needs at least two valid skill candidates; generated {len(candidates)}. Evidence: {output_dir / 'proposal-summary.json'}"
            )
        context = dict(task.input.get("planning_context") or {})
        context.setdefault("main_agent_reasoning", task.input.get("reasoning", ""))
        context["candidate_intents"] = [
            {
                "candidate_id": c.candidate_id,
                "skill": c.skill_name,
                "intent": c.metadata.get("intent"),
                "parameters": c.metadata.get("skill_parameters"),
                "prediction_steps": c.metadata.get("prediction_steps"),
                "execute_steps": c.metadata.get("execute_steps"),
            }
            for c in candidates
        ]
        task_text = (
            str(task.input["instruction"])
            + "\nCompare the supplied skill candidates for this planning horizon."
        )
        request = RolloutRequest(
            task=task_text,
            observation_path=observation_path,
            candidates=tuple(candidates),
            domain_name="libero",
            view_point=str(observation_info.get("view_point", "third_person_view")),
            output_dir=str(output_dir),
            planning_context=context,
            generation_errors=tuple(errors),
        )
        proposal_record = {
            "stage": "candidates_ready",
            "observation": shared,
            "camera": observation_info,
            "planning_context": context,
            "requested_candidate_count": len(specs),
            "generated_candidate_ids": [c.candidate_id for c in candidates],
            "generation_errors": errors,
            "candidate_actions": {
                c.candidate_id: {
                    "skill": c.skill_name,
                    "actions": c.actions,
                    "metadata": dict(c.metadata),
                }
                for c in candidates
            },
        }
        (output_dir / "proposal-summary.json").write_text(
            json.dumps(proposal_record, ensure_ascii=False, indent=2, default=str) + "\n"
        )
        return request

    @staticmethod
    def _infer_candidate_skill(parameters: dict[str, Any]) -> str:
        """Infer an omitted candidate skill only when its parameters are unambiguous."""
        keys = set(parameters)
        if keys & {"instruction", "prompt"} and not keys & {
            "position_m",
            "delta_m",
            "center",
            "command",
            "opening_m",
            "width_m",
        }:
            return "vla_execute"
        if keys & {"center", "axis", "radius_m"}:
            return "follow_arc"
        if keys & {"command", "opening_m", "width_m"} and not keys & {"position_m", "delta_m"}:
            return "set_gripper"
        if "delta_m" in keys:
            return "move_linear"
        if "position_m" in keys:
            # move_linear accepts a bare target position; a target with orientation is a pose move.
            return (
                "move_to_pose"
                if keys & {"orientation_euler", "orientation_quat"}
                else "move_linear"
            )
        return ""

    async def dispatch_selected_candidate(self, task, selected) -> str:
        """Execute the selected candidate: as an ordinary skill call or as its evaluated prefix."""
        if selected.metadata.get("ac_wm_dispatch_mode") == "skill":
            skill = str(selected.metadata.get("skill_action_type") or "")
            parameters = dict(
                selected.metadata.get("planner_parameters")
                or selected.metadata.get("skill_parameters")
                or {}
            )
            if skill == "vla_execute":
                parameters.setdefault("step", 40)
            via = selected.metadata.get("ac_wm_dispatch_via", "selection")
            return await self._dispatch_action(
                skill,
                parameters,
                f"AC-WM {via}: execute candidate {selected.candidate_id} as an ordinary {skill} call",
            )
        execute_steps = max(
            1, min(int(selected.metadata.get("execute_steps", 1)), len(selected.actions))
        )
        return await self._dispatch_action(
            "execute_action_chunk",
            {
                "actions": [list(row) for row in selected.actions[:execute_steps]],
                "candidate_id": selected.candidate_id,
                "skill_name": selected.skill_name,
            },
            f"AC-WM selected candidate {selected.candidate_id}; execute its evaluated action prefix",
        )


def extend_action_schema(schema):
    """Expose selection only when the optional mediator has been configured."""
    schema["properties"]["action_type"]["description"] += (
        "\n- 'ac_wm_select': Compare 2–8 skill candidates for the current recovery subgoal."
    )
    schema["properties"]["parameters"]["properties"] = {
        "plan_context": {
            "type": "object",
            "description": "Current subgoal, observed facts and completion criteria.",
        },
        "candidates": {
            "type": "array",
            "minItems": 2,
            "maxItems": 8,
            "description": "Materially different actions from the same state. Unique planner_priority values break positive-score ties; 1 is preferred.",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "skill": {
                        "type": "string",
                        "enum": [
                            "vla_execute",
                            "move_to_pose",
                            "move_linear",
                            "set_gripper",
                            "follow_arc",
                        ],
                    },
                    "parameters": {"type": "object"},
                    "planner_priority": {"type": "integer", "minimum": 1},
                    "prediction_steps": {"type": "integer", "minimum": 4, "maximum": 64},
                    "execute_steps": {"type": "integer", "minimum": 1},
                    "reasoning": {"type": "string"},
                },
                "required": ["skill", "parameters", "planner_priority"],
            },
        },
    }

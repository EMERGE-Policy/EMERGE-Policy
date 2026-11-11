import asyncio
import json

import pytest

from Emerge.ac_wm.protocol import ActionCandidate, RolloutRequest, RolloutResult
from Emerge.ac_wm.selector import AcWmSelector
from Emerge.ac_wm.subagent import AcWmSubagent
from Emerge.subagents.content import TextContent
from Emerge.subagents.models import SubagentTask


class Provider:
    def get_default_model(self):
        return "test-judge"


def candidate(name, direction, priority):
    return ActionCandidate(
        name,
        "rule:move_linear",
        ((direction, 0, 0, 0, 0, 0, -1),) * 8,
        {
            "execute_steps": 5,
            "planner_priority": priority,
            "skill_action_type": "move_linear",
            "planner_parameters": {"delta_m": [direction, 0, 0]},
        },
    )


def request(tmp_path, candidates=None):
    return RolloutRequest(
        "recover grasp",
        str(tmp_path / "observation.png"),
        candidates or (candidate("left", -0.1, 2), candidate("right", 0.2, 1)),
        "libero",
        output_dir=str(tmp_path),
    )


def rollout(_request, item):
    return RolloutResult(item.candidate_id, "success", "prediction.mp4", {"action_horizon": 4})


@pytest.mark.parametrize(
    "scores,expected,method",
    [
        ({"left": 0.7, "right": 0.2}, "left", "vlm_unique_top_score"),
        ({"left": 0.4, "right": 0.4}, "right", "planner_priority_positive_tie_break"),
        ({"left": 0.0, "right": 0.0}, None, None),
    ],
)
def test_joint_selection_and_positive_tie(tmp_path, scores, expected, method):
    class Judge:
        def evaluate_candidates(self, req, results):
            assert list(results) == ["left", "right"]
            return {name: (score, "visual evidence") for name, score in scores.items()}

    result = AcWmSelector(rollout, Judge()).select(request(tmp_path))
    assert result.selected_candidate_id == expected
    if expected:
        assert result.selection_method == method


def test_failed_rollout_is_not_selected(tmp_path):
    def predict(req, item):
        return (
            RolloutResult(item.candidate_id, "failed", error="no video")
            if item.candidate_id == "right"
            else rollout(req, item)
        )

    result = AcWmSelector(predict, lambda *_: (0.5, "progress")).select(request(tmp_path))
    assert result.selected_candidate_id == "left"
    assert result.evaluations[1].score == 0


@pytest.mark.parametrize("mode", ["evaluated_prefix", "skill"])
def test_selected_action_dispatch_and_manifest(tmp_path, mode):
    dispatched = []
    req = request(tmp_path)

    async def provide(_task):
        return req

    async def dispatch(_task, selected):
        dispatched.append(selected)
        return "Execution completed."

    agent = AcWmSubagent(
        provider=Provider(),
        rollout=rollout,
        judge=lambda _t, c, _r: (0.8 if c.candidate_id == "right" else 0.1, "progress"),
        candidate_provider=provide,
        dispatch=dispatch,
        dispatch_mode=mode,
    )
    result = asyncio.run(
        agent.run(
            SubagentTask(
                content=(TextContent("recover grasp"),), input={"output_dir": str(tmp_path)}
            )
        )
    )
    assert result.status.value == "success"
    assert len(dispatched) == 1
    selected = dispatched[0]
    assert selected.candidate_id == "right"
    assert selected.actions == req.candidates[1].actions
    assert selected.metadata["execute_steps"] == 4
    assert selected.metadata["ac_wm_dispatch_mode"] == mode
    record = json.loads((tmp_path / "decision.json").read_text())
    assert record["selected_candidate_id"] == "right"
    assert record["dispatch"]["attempted"] is True


@pytest.mark.parametrize("on_no_positive,expected_count", [("reject", 0), ("planner_priority", 1)])
def test_no_positive_policy_is_explicit(tmp_path, on_no_positive, expected_count):
    dispatched = []

    async def provide(_):
        return request(tmp_path)

    async def dispatch(_, selected):
        dispatched.append(selected)
        return "Execution completed."

    agent = AcWmSubagent(
        provider=Provider(),
        rollout=rollout,
        judge=lambda *_: (0, "no progress"),
        candidate_provider=provide,
        dispatch=dispatch,
        on_no_positive=on_no_positive,
    )
    result = asyncio.run(
        agent.run(
            SubagentTask(
                content=(TextContent("recover grasp"),), input={"output_dir": str(tmp_path)}
            )
        )
    )
    assert len(dispatched) == expected_count
    if dispatched:
        assert result.output["selected_candidate_id"] is None
        assert result.output["dispatch_via"] == "fallback"
        assert dispatched[0].candidate_id == "right"
        assert dispatched[0].metadata["ac_wm_dispatch_mode"] == "skill"
    else:
        assert result.status.value == "failed"


def test_identical_prefixes_do_not_create_a_false_choice(tmp_path):
    req = request(tmp_path, (candidate("a", 0.2, 1), candidate("b", 0.2, 2)))

    def unexpected(*_):
        raise AssertionError("equivalent proposals must be rejected before prediction")

    agent = AcWmSubagent(provider=Provider(), rollout=unexpected, judge=unexpected)
    result = asyncio.run(
        agent.run(
            SubagentTask(
                content=(TextContent("recover grasp"),),
                input={"request": req, "output_dir": str(tmp_path)},
            )
        )
    )
    assert result.status.value == "failed"
    assert result.metadata["selection_status"] == "no_distinct_action_candidates"

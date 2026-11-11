import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest

from Emerge.ac_wm.protocol import RolloutResult
from Emerge.ac_wm.runtime import build_components
from Emerge.ac_wm.subagent import AcWmSubagent
from Emerge.agent.tools.embodied import EmbodiedActionTool
from Emerge.config.schema import Config
from robot.mujoco_simulation.mujoco_actions import MujocoActionController
from robot.vla.mujoco_policy_executor import VLAExecutor


class Environment:
    def __init__(self, workspace):
        self.workspace = workspace
        self.steps = []
        self.latest_obs = {
            "agentview_image": np.zeros((16, 16, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.ones((16, 16, 3), dtype=np.uint8),
        }

    def get_eef_pose(self):
        return np.zeros(3), np.array([0, 0, 0, 1])

    def current_gripper_signal(self):
        return -1.0

    def step(self, action, **kwargs):
        self.steps.append(np.array(action))
        return {}, 0.0, False, {}

    def check_success(self):
        return False

    def refresh_cameras(self):
        pass


class Provider:
    def get_default_model(self):
        return "test"


def test_disabled_ac_wm_does_not_construct_optional_models(monkeypatch):
    monkeypatch.delenv("EMERGE_AC_WM", raising=False)
    assert build_components(Config()) == (None, None)


@pytest.mark.parametrize("key", ["acWm", "ac_wm"])
def test_runtime_wiring_accepts_documented_config_and_keeps_specialists(tmp_path, monkeypatch, key):
    from Emerge.runtime.service import build_agent

    monkeypatch.delenv("EMERGE_AC_WM", raising=False)
    monkeypatch.setattr("Emerge.providers.factory.create_provider", lambda _: Provider())
    config = Config.model_validate(
        {
            key: {"enabled": True, "judgeModel": "independent-judge", "dispatchMode": "skill"},
            "agents": {"defaults": {"workspace": str(tmp_path), "model": "main-planner"}},
            "providers": {"custom": {"apiKey": "test", "apiBase": "https://example.test/v1"}},
            "visual_monitor": {"enabled": False},
        }
    )
    agent = build_agent(config)
    assert agent.ac_wm_subagent._dispatch_mode == "skill"
    assert agent.ac_wm_subagent._selector.judge.config.model == "independent-judge"
    assert "object_location" in agent.subagent_registry.names
    assert "task_verification" in agent.subagent_registry.names
    assert agent.tools.get("execute_robot_action").ac_wm_subagent is agent.ac_wm_subagent
    assert (
        "ac_wm_select"
        in agent.tools.get("execute_robot_action").parameters["properties"]["action_type"][
            "description"
        ]
    )
    assert (
        "ac-wm"
        not in agent.tools.get("delegate_subagent").parameters["properties"]["agent_name"]["enum"]
    )


@pytest.mark.parametrize(
    "skill,params",
    [
        ("move_to_pose", {"position_m": [0.1, 0, 0], "orientation_quat": [0, 0, 0, 1]}),
        ("move_linear", {"delta_m": [0.1, 0, 0]}),
        ("set_gripper", {"command": "close"}),
        (
            "follow_arc",
            {"center": [0, 0.1, 0], "axis": [0, 0, 1], "radius_m": 0.1, "angle_deg": 30},
        ),
    ],
)
def test_geometry_preview_keeps_simulator_paused(tmp_path, skill, params):
    env = Environment(tmp_path)
    controller = MujocoActionController(env)
    result = controller.execute(
        "rule_propose",
        {"skill_action_type": skill, "parameters": {**params, "steps": 8, "settle_steps": 0}},
    )
    proposal = json.loads(result.split("RULE_PROPOSAL:", 1)[1])
    assert np.array(proposal["actions"]).shape == (8, 7)
    assert env.steps == []


def test_vla_proposal_and_exact_dispatch(tmp_path):
    env = Environment(tmp_path)
    rows = np.array([[0.1, 0, 0, 0, 0, 0, -1], [0.2, 0, 0, 0, 0, 0, -1]], dtype=np.float32)
    executor = VLAExecutor(env, client=SimpleNamespace(infer=lambda _: {"actions": rows}))
    executor._build_element = lambda _: {}
    proposed = executor.propose("grasp bowl", 2)
    assert env.steps == []
    controller = MujocoActionController(env, vla_executor=executor)
    result = controller.execute(
        "execute_action_chunk",
        {"actions": proposed.tolist(), "candidate_id": "vla", "skill_name": "vla"},
    )
    np.testing.assert_array_equal(env.steps, rows)
    assert "steps=2" in result


def test_full_candidate_flow_has_one_observation_and_one_dispatch(tmp_path, monkeypatch):
    monkeypatch.delenv("EMERGE_POLICY_BACKEND", raising=False)
    env = Environment(tmp_path)
    controller = MujocoActionController(env)
    tool = EmbodiedActionTool(tmp_path)
    calls = []

    async def dispatch(action_type, parameters, reasoning):
        calls.append(action_type)
        return controller.execute(action_type, parameters)

    tool._dispatch_action = dispatch
    seen = []

    def predict(req, candidate):
        assert env.steps == []
        seen.append(req.observation_path)
        return RolloutResult(candidate.candidate_id, "success", "future.mp4", {"action_horizon": 8})

    tool.ac_wm_subagent = AcWmSubagent(
        provider=Provider(),
        rollout=predict,
        judge=lambda _t, c, _r: (0.8 if c.candidate_id == "right" else 0.2, "progress"),
        candidate_provider=tool.propose_action_candidates,
        dispatch=tool.dispatch_selected_candidate,
    )
    result = json.loads(
        asyncio.run(
            tool.execute(
                "ac_wm_select",
                {
                    "plan_context": {"current_subgoal": "recover grasp"},
                    "candidates": [
                        {
                            "candidate_id": name,
                            "skill": "move_linear",
                            "parameters": {"delta_m": [direction, 0, 0]},
                            "prediction_steps": 8,
                            "execute_steps": 3,
                            "planner_priority": priority,
                        }
                        for name, direction, priority in [("left", -0.1, 2), ("right", 0.2, 1)]
                    ],
                },
                "compare recovery moves",
            )
        )
    )
    assert result["output"]["selected_candidate_id"] == "right"
    assert calls == [
        "capture_ac_wm_observation",
        "rule_propose",
        "rule_propose",
        "execute_action_chunk",
    ]
    assert len(set(seen)) == 1
    assert len(env.steps) == 3
    assert all(row[0] > 0 for row in env.steps)


@pytest.mark.parametrize(
    "backend,action,params",
    [
        ("vla", "vla_execute", {"instruction": "grasp bowl", "step": 5}),
        ("wam", "wam_execute", {"phase_instruction": "grasp bowl", "step": 5}),
    ],
)
def test_ordinary_policies_bypass_ac_wm_and_keep_queue_contract(
    tmp_path, monkeypatch, backend, action, params
):
    monkeypatch.setenv("EMERGE_POLICY_BACKEND", backend)
    monkeypatch.setenv("EMERGE_WAM_TASK_INSTRUCTION", "put bowl on plate")
    monkeypatch.setenv("EMERGE_WAM_CONDITIONING_MODE", "task")
    (tmp_path / "EMBODIED.md").write_text("robot")

    async def unexpected(*_):
        raise AssertionError("ordinary policy was routed into AC-WM")

    tool = EmbodiedActionTool(tmp_path, ac_wm_subagent=SimpleNamespace(run=unexpected))

    async def completed(*args, **kwargs):
        return "completed"

    tool._wait_for_action_result = completed
    assert asyncio.run(tool.execute(action, params, "act")) == "completed"
    queued = tool._load_action_document(tmp_path / "ACTION.md")["actions"][0]
    assert queued["action_type"] == action
    if backend == "wam":
        assert queued["parameters"]["task_instruction"] == "put bowl on plate"
        assert queued["parameters"]["conditioning_mode"] == "task"


def test_wam_driver_still_executes_and_rejects_disabled_vla_proposals(tmp_path):
    calls = []

    def execute(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(
            success=True,
            total_steps=1,
            reason="step_completed",
            last_gripper_command=-1,
            task_success=False,
            search_decisions=(),
        )

    controller = MujocoActionController(
        Environment(tmp_path),
        wam_executor=SimpleNamespace(execute=execute),
        enabled_policy_backends={"wam"},
    )
    assert (
        "completed"
        in controller.execute("wam_execute", {"task_instruction": "grasp bowl", "step": 1}).lower()
    )
    assert len(calls) == 1
    assert "disabled" in controller.execute("vla_propose", {"instruction": "grasp bowl"})


def test_ac_wm_stays_internal_to_delegate_tool():
    from Emerge.agent.tools.delegate import DelegateSubagentTool
    from Emerge.subagents.registry import SubagentRegistry

    registry = SubagentRegistry()
    registry.register(
        AcWmSubagent(provider=Provider(), rollout=lambda *_: None, judge=lambda *_: None)
    )
    delegate = DelegateSubagentTool(registry, hidden_agents=("ac-wm",))
    assert "ac-wm" not in delegate.parameters["properties"]["agent_name"]["enum"]
    assert asyncio.run(delegate.execute("ac-wm", "do something")).startswith("Error:")

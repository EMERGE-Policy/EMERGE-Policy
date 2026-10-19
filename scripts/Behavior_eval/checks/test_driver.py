"""Checks for the BEHAVIOR driver contract, readiness and state synchronization."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from scipy.spatial.transform import Rotation

from robot.behavior import contracts
from robot.behavior.r1pro_kinematics import R1ProKinematics
from robot.behavior.r1pro_motion import MotionPlan
from robot.drivers.behavior_r1pro_driver import BehaviorR1ProDriver

TASK_NAME = "turning_on_radio"
PROMPT = "Turn on the radio receiver that's on the table in the living room."
URDF = (
    Path(__file__).resolve().parents[3]
    / "third_party/behavior-1k/datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf"
)


def _make_driver(workspace: Path, **kwargs: Any) -> BehaviorR1ProDriver:
    kwargs.setdefault("required_instruction", PROMPT)
    kwargs.setdefault("expected_task", TASK_NAME)
    return BehaviorR1ProDriver(workspace=workspace, **kwargs)


class _FakeGateway:
    """Minimal stand-in for :class:`B1KProxyGateway`."""

    def __init__(self, **state: Any) -> None:
        self.state: dict[str, Any] = {
            "status": "ready",
            "active_connections": 1,
            "requests_forwarded": 0,
            "evaluator_connected": True,
            "upstream_connected": True,
            **state,
        }
        # A dispatched execution reaches a terminal state rather than hanging.
        self.execution: dict[str, Any] = {
            "execution_id": 1,
            "status": "step_completed",
            "completed_steps": 0,
            "detail": None,
        }
        self.observation_count = 0
        self.sync_result = True
        self.begin_error: Exception | None = None
        self.published = 0
        self.started = False
        self.stopped = False
        self.baseline_seen: int | None = None
        self.wait_hook: Any | None = None
        self.hold_action = [0.0] * 23
        self.local_action_seen: list[float] | None = None
        self.local_trajectory_seen: list[list[float]] | None = None
        self.robot_state = {
            "trunk": {"qpos": [1.025, -1.45, -0.47, 0.0]},
            "arms": {
                "left": {
                    "qpos": [0.0] * 7,
                    "eef_position": [0.0, 0.0, 0.0],
                    "eef_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
                "right": {
                    "qpos": [0.0] * 7,
                    "eef_position": [0.0, 0.0, 0.0],
                    "eef_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        }

    def start(self, **_kwargs: Any) -> None:
        self.started = True

    def stop(self, **_kwargs: Any) -> None:
        self.stopped = True

    def snapshot(self) -> dict[str, Any]:
        return {**self.state, "execution": self.execution}

    def begin_execution(
        self, instruction: str, requested_steps: int, *, baseline_evaluator_steps=None
    ) -> int:
        if self.begin_error is not None:
            raise self.begin_error
        self.baseline_seen = baseline_evaluator_steps
        self.execution = {
            "execution_id": 1,
            "instruction": instruction,
            "requested_steps": requested_steps,
            "completed_steps": self.execution.get("completed_steps", 0),
            "status": self.execution.get("status", "step_completed"),
            "detail": self.execution.get("detail"),
            "sync_observation_baseline": self.execution.get("sync_observation_baseline", 0),
            "baseline_evaluator_steps": baseline_evaluator_steps,
        }
        return 1

    def current_hold_action(self) -> list[float]:
        return list(self.hold_action)

    def begin_local_execution(
        self,
        label: str,
        requested_steps: int,
        action: list[float],
        *,
        baseline_evaluator_steps=None,
    ) -> int:
        self.local_action_seen = list(action)
        return self.begin_execution(
            label,
            requested_steps,
            baseline_evaluator_steps=baseline_evaluator_steps,
        )

    def current_robot_state(self) -> dict[str, Any]:
        return self.robot_state

    def begin_local_trajectory(
        self, label: str, actions: list[list[float]], *, baseline_evaluator_steps=None
    ) -> int:
        self.local_trajectory_seen = actions
        self.execution["completed_steps"] = len(actions)
        return self.begin_execution(
            label, len(actions), baseline_evaluator_steps=baseline_evaluator_steps
        )

    def wait_for_execution(self, execution_id: int, **_kwargs: Any) -> dict[str, Any]:
        if self.wait_hook is not None:
            self.wait_hook()
        return dict(self.execution)

    def wait_for_observation_after(
        self, baseline: int | None, *, timeout_s: float, cancel_check: Any | None = None
    ) -> bool:
        return self.sync_result

    def publish_observation(self) -> None:
        self.published += 1
        return None


def _status_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema_version": contracts.EVALUATION_STATUS_SCHEMA,
        "task": TASK_NAME,
        "instance_id": 301,
        "rollout_id": 0,
        "phase": "running",
        "steps": 10,
        "success": False,
        "done": False,
        "terminated": False,
        "truncated": False,
    }
    payload.update(overrides)
    return payload


class BehaviorDriverContractTest(unittest.TestCase):
    """Parameter validation must reject a request the gateway cannot honor."""

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.driver = _make_driver(self.workspace, listen_port=0)
        self.gateway = _FakeGateway()
        # Valid requests must reach a terminal execution state, not hang.
        self.gateway.execution["status"] = "step_completed"
        self.driver._gateway = self.gateway

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _run(self, params: Any) -> str:
        return self.driver.execute_action("vla_execute", params)

    def test_exact_instruction_is_required(self) -> None:
        result = self._run({"instruction": "turn on the radio", "step": 10})
        self.assertTrue(result.startswith("Failed:"))
        self.assertIn(PROMPT, result)

    def test_whitespace_differences_are_tolerated(self) -> None:
        result = self._run({"instruction": f"  {PROMPT}  ", "step": 10})
        self.assertFalse(result.startswith("Failed:"))

    def test_missing_instruction_and_step_are_rejected(self) -> None:
        self.assertIn("instruction", self._run({"step": 10}))
        self.assertIn("requires step", self._run({"instruction": PROMPT}))

    def test_non_object_parameters_are_rejected(self) -> None:
        self.assertTrue(self._run(["instruction"]).startswith("Failed:"))

    def test_boolean_step_is_rejected(self) -> None:
        result = self._run({"instruction": PROMPT, "step": True})
        self.assertTrue(result.startswith("Failed:"))
        self.assertIn("boolean", result)

    def test_fractional_and_negative_and_oversized_steps_are_rejected(self) -> None:
        self.assertTrue(self._run({"instruction": PROMPT, "step": 10.5}).startswith("Failed:"))
        self.assertTrue(self._run({"instruction": PROMPT, "step": -1}).startswith("Failed:"))
        self.assertTrue(self._run({"instruction": PROMPT, "step": 0}).startswith("Failed:"))
        over = self.driver.max_action_steps + 1
        result = self._run({"instruction": PROMPT, "step": over})
        self.assertTrue(result.startswith("Failed:"))
        self.assertIn(str(self.driver.max_action_steps), result)

    def test_integral_float_and_digit_string_stay_compatible(self) -> None:
        self.assertFalse(self._run({"instruction": PROMPT, "step": 5.0}).startswith("Failed:"))
        self.assertFalse(self._run({"instruction": PROMPT, "step": "5"}).startswith("Failed:"))

    def test_unknown_action_type_is_reported(self) -> None:
        self.assertIn("Unknown action type", self.driver.execute_action("follow_arc", {}))

    def test_navigation_only_blocks_champion_and_other_physical_actions(self) -> None:
        self.driver._navigation_only = True
        for action_type, parameters in (
            ("vla_execute", {"instruction": PROMPT, "step": 1}),
            ("hold_position", {"step": 1}),
            ("move_to_pose", {"side": "left", "position_m": [0.5, 0.0, 1.0]}),
        ):
            with self.subTest(action_type=action_type):
                result = self.driver.execute_action(action_type, parameters)
                self.assertIn("disabled in this navigation-only experiment", result)
        self.assertIsNone(self.gateway.local_action_seen)
        self.assertIn("Gateway status", self.driver.execute_action("gateway_status", {}))

    def test_navigation_only_requires_boolean_configuration(self) -> None:
        with self.assertRaises(contracts.ConfigValidationError):
            _make_driver(self.workspace, navigation_only="yes")

    def test_navigation_only_uses_existing_local_gateway_action_path(self) -> None:
        self.driver._navigation_only = True
        self.gateway.execution["completed_steps"] = 7
        self.gateway.publish_observation = lambda: self.workspace / "observation.json"
        websocket = MagicMock()

        def exercise_loop(**kwargs):
            self.assertEqual(kwargs["instruction"], "Approach the radio")
            frame = kwargs["frame_source"]()
            self.assertEqual(frame.observation_seq, 1)
            outcome = kwargs["execute_waypoint"]((0.0875, 0.0, 0.0), 7)
            self.assertEqual(outcome["completed_steps"], 7)
            return {"finish_reason": "model_stop_candidate", "nav_steps": 7}

        frame = SimpleNamespace(observation_seq=1, episode_id=1, jpeg=b"frame")
        with patch("integrations.lightnav_behavior.probe.load_head_frame", return_value=frame), \
             patch("integrations.lightnav_behavior.closed_loop.run_closed_loop", side_effect=exercise_loop), \
             patch("integrations.lightnav_sonic_emerge.bridge.lightnav_ws.LightNavWsClient", return_value=websocket):
            result = self.driver.execute_action(
                "navigate_vln", {"instruction": "Approach the radio", "max_steps": 7}
            )
        self.assertIn("model_stop_candidate", result)
        self.assertIsNotNone(self.gateway.local_action_seen)
        self.assertEqual(self.gateway.local_action_seen[3:], [0.0] * 20)
        self.assertEqual(self.driver.last_navigation_result["nav_steps"], 7)

    def test_navigation_calls_reuse_history_and_share_the_episode_budget(self) -> None:
        self.driver._navigation_only = True
        self.driver.max_action_steps = 14
        self.gateway.execution["completed_steps"] = 7
        self.gateway.publish_observation = lambda: self.workspace / "observation.json"
        client = MagicMock()
        seen = []

        def segment(**kwargs):
            seen.append((kwargs["client"], kwargs["instruction"], kwargs["call_id"]))
            kwargs["frame_source"]()
            kwargs["execute_waypoint"]((0.08, 0.0, 0.0), 7)
            return {"finish_reason": "model_stop_candidate", "nav_steps": 7, "model_frames": 1}

        frames = [SimpleNamespace(observation_seq=i, episode_id=1, jpeg=b"frame") for i in (1, 8)]
        with patch("integrations.lightnav_behavior.probe.load_head_frame", side_effect=frames), \
             patch("integrations.lightnav_behavior.closed_loop.run_closed_loop", side_effect=segment), \
             patch("integrations.lightnav_sonic_emerge.bridge.lightnav_ws.LightNavWsClient", return_value=client) as factory:
            for instruction in ("approach", "use another route"):
                result = self.driver.execute_action("navigate_vln", {"instruction": instruction, "max_steps": 7})
                self.assertFalse(result.startswith("Failed:"), result)
            self.assertIn("exhausted", self.driver.execute_action("navigate_vln", {"instruction": "again", "max_steps": 1}))
        self.assertEqual(factory.call_count, 1)
        client.connect.assert_called_once()
        self.assertEqual(seen, [(client, "approach", 1), (client, "use another route", 2)])
        self.assertFalse(self.driver._navigation_results[0]["history_continued"])
        self.assertTrue(self.driver._navigation_results[1]["history_continued"])
        self.assertEqual(self.driver.get_runtime_state()["robots"]["behavior_r1pro"]["remaining_action_steps"], 0)
        self.driver.close()
        client.close.assert_called_once()

    def test_complete_task_can_alternate_vln_and_fixed_prompt_vla(self) -> None:
        self.assertFalse(self.driver._navigation_only)
        self.gateway.execution["completed_steps"] = 7
        self.gateway.publish_observation = lambda: self.workspace / "observation.json"
        client = MagicMock()
        frame = SimpleNamespace(observation_seq=1, episode_id=1, jpeg=b"frame")

        def segment(**kwargs):
            kwargs["frame_source"]()
            outcome = kwargs["execute_waypoint"]((0.08, 0.0, 0.0), 7)
            return {"finish_reason": "step_budget", "nav_steps": outcome["completed_steps"]}

        with patch("integrations.lightnav_behavior.probe.load_head_frame", return_value=frame), \
             patch("integrations.lightnav_behavior.closed_loop.run_closed_loop", side_effect=segment), \
             patch("integrations.lightnav_sonic_emerge.bridge.lightnav_ws.LightNavWsClient", return_value=client):
            for instruction in ("Approach the object", "Reposition beside the table"):
                nav = self.driver.execute_action("navigate_vln", {"instruction": instruction, "max_steps": 7})
                self.assertIn("Navigation segment finished", nav)
                vla = self.driver.execute_action("vla_execute", {"instruction": PROMPT, "step": 7})
                self.assertIn("VLA execution finished", vla)
                self.assertEqual(self.gateway.execution["instruction"], PROMPT)
        self.assertEqual(len(self.driver._navigation_results), 2)
        self.assertTrue(self.driver._navigation_results[-1]["history_continued"])
        self.assertEqual(self.driver._gateway_action_steps, 28)
        rewritten = self.driver.execute_action("vla_execute", {"instruction": "press the button", "step": 7})
        self.assertTrue(rewritten.startswith("Failed:"), rewritten)
        self.driver.close()

    def test_navigation_resets_history_when_episode_changes(self) -> None:
        self.gateway.publish_observation = lambda: self.workspace / "observation.json"
        frames = [SimpleNamespace(observation_seq=1, episode_id=i, jpeg=b"frame") for i in (1, 2)]
        clients = [MagicMock(), MagicMock()]
        with patch("integrations.lightnav_behavior.probe.load_head_frame", side_effect=frames), \
             patch("integrations.lightnav_behavior.closed_loop.run_closed_loop",
                   side_effect=lambda **kw: {"finish_reason": "model_stop_candidate", "nav_steps": 0}), \
             patch("integrations.lightnav_sonic_emerge.bridge.lightnav_ws.LightNavWsClient", side_effect=clients):
            for _ in frames:
                self.driver.execute_action("navigate_vln", {"instruction": "approach", "max_steps": 7})
        clients[0].close.assert_called_once()
        self.assertFalse(self.driver._navigation_results[-1]["history_continued"])
        self.driver.close()

    @unittest.skipUnless(URDF.is_file(), "official R1Pro URDF required")
    def test_navigation_posture_navigation_preserves_history_and_all_frames(self) -> None:
        self.driver._navigation_only = True
        self.driver.max_action_steps = 40
        self.driver._kinematics = R1ProKinematics(URDF)
        state = self.gateway.robot_state

        def update_geometry():
            for side in ("left", "right"):
                state["arms"][side]["eef_position"] = self.driver._kinematics.forward(
                    side, state["trunk"]["qpos"], state["arms"][side]["qpos"]
                )[:3, 3].tolist()

        update_geometry()
        self.gateway.hold_action[3:7] = state["trunk"]["qpos"]
        self.gateway.observation_count = 1
        self.gateway.publish_observation = lambda: self.workspace / "observation.json"
        begin_local = self.gateway.begin_local_execution

        def begin(label, steps, action, **kwargs):
            self.gateway.execution["completed_steps"] = steps
            return begin_local(label, steps, action, **kwargs)

        self.gateway.begin_local_execution = begin

        def applied():
            self.gateway.observation_count += self.gateway.execution["completed_steps"]
            if self.gateway.execution["instruction"].startswith("adjust_posture"):
                self.gateway.hold_action = self.gateway.local_trajectory_seen[-1].copy()
                state["trunk"]["qpos"] = self.gateway.hold_action[3:7]
                update_geometry()

        self.gateway.wait_hook = applied
        client = MagicMock()
        client.predict.return_value = SimpleNamespace(
            first_waypoint=(0.08, 0.0, 0.0), stop=False, visible=False, latency_ms=1.0,
        )
        frame = lambda _manifest: SimpleNamespace(
            jpeg=b"frame", observation_seq=self.gateway.observation_count, episode_id=1,
        )
        with patch("integrations.lightnav_behavior.probe.load_head_frame", side_effect=frame), \
             patch("integrations.lightnav_sonic_emerge.bridge.lightnav_ws.LightNavWsClient", return_value=client) as factory:
            first = self.driver.execute_action("navigate_vln", {"instruction": "search", "max_steps": 7})
            pose = self.driver.execute_action("adjust_posture", {
                "joint_deltas_rad": {"torso_joint4": 0.1}, "max_steps": 25,
            })
            second = self.driver.execute_action("navigate_vln", {"instruction": "approach", "max_steps": 7})
        for result in (first, pose, second):
            self.assertFalse(result.startswith("Failed:"), result)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual([call.args[1] for call in client.predict.call_args_list], ["search", "approach"])
        self.assertTrue(self.driver.last_navigation_result["history_continued"])
        self.assertAlmostEqual(self.gateway.local_action_seen[6], 0.1)
        self.assertEqual(self.driver._gateway_action_steps, 34)
        self.assertEqual(self.driver._remaining_steps(), 6)
        frames = sorted((self.workspace / "lightnav_frames").glob("*.jpg"))
        self.assertEqual(len(frames), 5)
        records = [json.loads(line) for line in (self.workspace / "lightnav_trace.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 5)
        self.assertEqual(records[2]["event"], "posture_completed")
        self.assertEqual(records[-1]["call_id"], 2)
        self.driver.close()

    @unittest.skipUnless(URDF.is_file(), "official R1Pro URDF required")
    def test_posture_uses_measured_joint_state_and_reports_tracking_failure(self) -> None:
        self.driver._navigation_only = True
        self.driver._kinematics = R1ProKinematics(URDF)
        state = self.gateway.robot_state
        for side in ("left", "right"):
            pose = self.driver._kinematics.forward(side, state["trunk"]["qpos"], state["arms"][side]["qpos"])
            state["arms"][side]["eef_position"] = pose[:3, 3].tolist()
        self.gateway.hold_action[3:7] = state["trunk"]["qpos"]
        self.gateway.publish_observation = lambda: self.workspace / "observation.json"
        frame = SimpleNamespace(observation_seq=21, episode_id=1, jpeg=b"posture")
        with patch("integrations.lightnav_behavior.probe.load_head_frame", return_value=frame):
            # A completed action budget is not proof that the joints followed.
            result = self.driver.execute_action("adjust_posture", {
                "joint_deltas_rad": {"torso_joint4": 0.1}, "max_steps": 30,
            })
        self.assertTrue(result.startswith("Failed:"), result)
        self.assertIn("actual posture differs", result)
        self.assertTrue(all(action[:3] == [0.0, 0.0, 0.0] for action in self.gateway.local_trajectory_seen))
        record = self.driver._posture_results[-1]
        self.assertAlmostEqual(record["max_joint_tracking_error_rad"], 0.1)
        self.assertEqual(record["actual_body_posture"]["joint_positions_rad"]["torso_joint4"], 0.0)
        self.assertTrue(Path(record["image_path"]).is_file())
        self.assertEqual(self.driver._gateway_action_steps, len(self.gateway.local_trajectory_seen))

    @unittest.skipUnless(URDF.is_file(), "official R1Pro URDF required")
    def test_posture_success_publishes_actual_pose_and_consumes_shared_budget(self) -> None:
        self.driver._navigation_only = True
        self.driver._kinematics = R1ProKinematics(URDF)
        state = self.gateway.robot_state
        for side in ("left", "right"):
            state["arms"][side]["eef_position"] = self.driver._kinematics.forward(
                side, state["trunk"]["qpos"], state["arms"][side]["qpos"]
            )[:3, 3].tolist()
        self.gateway.hold_action[3:7] = state["trunk"]["qpos"]
        self.gateway.publish_observation = lambda: self.workspace / "observation.json"
        self.gateway.wait_hook = lambda: state["trunk"]["qpos"].__setitem__(3, 0.1)
        frame = SimpleNamespace(observation_seq=21, episode_id=1, jpeg=b"posture")
        with patch("integrations.lightnav_behavior.probe.load_head_frame", return_value=frame):
            result = self.driver.execute_action("adjust_posture", {
                "joint_deltas_rad": {"torso_joint4": 0.1}, "max_steps": 30,
            })
        self.assertTrue(result.startswith("Posture execution finished:"), result)
        self.assertTrue(self.driver._posture_results[-1]["pose_reached"])
        before = self.driver._gateway_action_steps
        self.driver.max_action_steps = before
        self.assertIn("exhausted", self.driver.execute_action("adjust_posture", {"joint_deltas_rad": {"torso_joint4": 0.1}}))

    def test_removed_control_mode_is_not_supported_or_advertised(self) -> None:
        result = self.driver.execute_action("set_control_mode", {"mode": "manipulation"})
        self.assertIn("Unknown action type", result)
        robot = self.driver.get_runtime_state()["robots"]["behavior_r1pro"]
        self.assertNotIn("control_mode", robot)
        self.assertNotIn("base_motion_enabled", robot)
        self.assertNotIn("control_mode", self.driver.execute_action("gateway_status", {}))

    def test_hold_position_uses_the_gateway_local_action(self) -> None:
        result = self.driver.execute_action("hold_position", {"step": 2})
        self.assertTrue(result.startswith("Hold execution finished:"))
        self.assertEqual(self.gateway.local_action_seen, [0.0] * 23)

    def test_hold_position_rejects_unknown_parameters(self) -> None:
        result = self.driver.execute_action("hold_position", {"step": 1, "pose": []})
        self.assertTrue(result.startswith("Failed:"))
        self.assertIn("unknown parameters", result)

    def test_primitive_requires_the_official_urdf_configuration(self) -> None:
        result = self.driver.execute_action(
            "move_to_pose", {"side": "left", "position_m": [0.0, 0.0, 1.0]}
        )
        self.assertTrue(result.startswith("Failed:"))
        self.assertIn("r1pro_urdf_path", result)

    @unittest.skipUnless(URDF.is_file(), f"official R1Pro URDF not found at {URDF}")
    def test_move_to_pose_dispatches_a_bounded_local_trajectory(self) -> None:
        driver = _make_driver(
            self.workspace,
            listen_port=0,
            r1pro_urdf_path=URDF,
        )
        gateway = _FakeGateway()
        kinematics = R1ProKinematics(URDF)
        trunk = gateway.robot_state["trunk"]["qpos"]
        for side in ("left", "right"):
            pose = kinematics.forward(side, trunk, [0.0] * 7)
            gateway.robot_state["arms"][side]["eef_position"] = pose[:3, 3].tolist()
            gateway.robot_state["arms"][side]["eef_orientation_xyzw"] = (
                Rotation.from_matrix(pose[:3, :3]).as_quat().tolist()
            )
        driver._gateway = gateway
        target_qpos = [-0.1, 0.2, 0.1, -0.3, 0.1, 0.05, -0.1]
        target = kinematics.forward("left", trunk, target_qpos)[:3, 3].tolist()

        result = driver.execute_action(
            "move_to_pose", {"side": "left", "position_m": target, "steps": 40, "settle_steps": 10}
        )

        self.assertTrue(result.startswith("Move-to-pose execution finished:"), result)
        self.assertIsNotNone(gateway.local_trajectory_seen)
        trajectory = gateway.local_trajectory_seen or []
        self.assertGreater(len(trajectory), 1)
        self.assertEqual(len(trajectory), 50)
        self.assertTrue(all(action == trajectory[-1] for action in trajectory[-10:]))
        self.assertTrue(all(len(action) == 23 for action in trajectory))

    @unittest.skipUnless(URDF.is_file(), f"official R1Pro URDF not found at {URDF}")
    def test_bbox_move_to_pose_derives_a_precontact_target(self) -> None:
        driver = _make_driver(
            self.workspace,
            listen_port=0,
            r1pro_urdf_path=URDF,
        )
        gateway = _FakeGateway()
        kinematics = R1ProKinematics(URDF)
        trunk = gateway.robot_state["trunk"]["qpos"]
        for side in ("left", "right"):
            pose = kinematics.forward(side, trunk, [0.0] * 7)
            gateway.robot_state["arms"][side]["eef_position"] = pose[:3, 3].tolist()
            gateway.robot_state["arms"][side]["eef_orientation_xyzw"] = (
                Rotation.from_matrix(pose[:3, :3]).as_quat().tolist()
            )
        driver._gateway = gateway
        left = gateway.robot_state["arms"]["left"]["eef_position"]
        bbox_min = [left[0] - 0.03, left[1] - 0.03, left[2] - 0.13]
        bbox_max = [left[0] + 0.03, left[1] + 0.03, left[2] - 0.07]

        result = driver.execute_action(
            "move_to_pose",
            {
                "side": "left",
                "bbox_world_min_m": bbox_min,
                "bbox_world_max_m": bbox_max,
                "standoff_m": 0.08,
                "steps": 40,
            },
        )

        self.assertTrue(result.startswith("Move-to-pose execution finished:"), result)
        self.assertTrue(gateway.local_trajectory_seen)
        self.assertEqual(len(gateway.local_trajectory_seen), 40)

    def test_bbox_move_requires_both_bounds(self) -> None:
        self.driver._kinematics = object()
        result = self.driver.execute_action(
            "move_to_pose", {"bbox_world_min_m": [0.0, 0.0, 0.0]}
        )
        self.assertTrue(result.startswith("Failed:"), result)
        self.assertIn("bbox_world_min_m and bbox_world_max_m", result)

    def test_direct_position_rejects_bbox_only_standoff(self) -> None:
        self.driver._kinematics = object()
        result = self.driver.execute_action(
            "move_to_pose",
            {"side": "left", "position_m": [0.4, 0.2, 0.8], "standoff_m": 0.1},
        )
        self.assertTrue(result.startswith("Failed:"), result)
        self.assertIn("valid only with bbox", result)

    def test_move_linear_forwards_relative_displacement_to_the_planner(self) -> None:
        self.driver._kinematics = object()
        planned = SimpleNamespace(
            description="relative lift",
            actions=((0.0,) * 23,),
        )
        with patch(
            "robot.drivers.behavior_r1pro_driver.plan_move_linear",
            return_value=planned,
        ) as planner:
            result = self.driver.execute_action(
                "move_linear",
                {"side": "right", "delta_m": [0.0, 0.0, 0.08], "steps": 30, "settle_steps": 20},
            )

        self.assertTrue(result.startswith("Linear execution finished:"), result)
        self.assertEqual(planner.call_args.kwargs["delta_m"], [0.0, 0.0, 0.08])
        self.assertIsNone(planner.call_args.kwargs["position_m"])
        self.assertEqual(planner.call_args.kwargs["steps"], 30)
        self.assertEqual(planner.call_args.kwargs["settle_steps"], 20)

    @unittest.skipUnless(URDF.is_file(), f"official R1Pro URDF not found at {URDF}")
    def test_attach_linear_timing_reaches_the_gateway_with_real_ik(self) -> None:
        driver = _make_driver(self.workspace, listen_port=0, r1pro_urdf_path=URDF)
        gateway = _FakeGateway()
        kinematics = R1ProKinematics(URDF)
        for side in ("left", "right"):
            pose = kinematics.forward(side, gateway.robot_state["trunk"]["qpos"], [0.0] * 7)
            gateway.robot_state["arms"][side]["eef_position"] = pose[:3, 3].tolist()
            gateway.robot_state["arms"][side]["eef_orientation_xyzw"] = Rotation.from_matrix(pose[:3, :3]).as_quat().tolist()
        driver._gateway = gateway
        result = driver.execute_action(
            "move_linear", {"side": "right", "delta_m": [0.01, 0.0, 0.0], "steps": 30, "settle_steps": 20},
        )
        self.assertTrue(result.startswith("Linear execution finished:"), result)
        trajectory = gateway.local_trajectory_seen
        self.assertIsNotNone(trajectory)
        self.assertEqual(len(trajectory), 50)
        self.assertTrue(all(action == trajectory[-1] for action in trajectory[-20:]))

    @unittest.skipUnless(URDF.is_file(), f"official R1Pro URDF not found at {URDF}")
    def test_invalid_motion_timing_cannot_dispatch_a_trajectory(self) -> None:
        driver = _make_driver(self.workspace, listen_port=0, r1pro_urdf_path=URDF)
        gateway = _FakeGateway()
        driver._gateway = gateway
        for action, params in (
            ("move_to_pose", {"side": "left", "position_m": [0.1, 0.2, 0.3]}),
            ("move_linear", {"side": "right", "delta_m": [0.01, 0.0, 0.0]}),
        ):
            for timing in ({"steps": True}, {"steps": 0}, {"settle_steps": -1}, {"steps": 230, "settle_steps": 20}, {"stepz": 40}):
                with self.subTest(action=action, timing=timing):
                    result = driver.execute_action(action, {**params, **timing})
                    self.assertTrue(result.startswith("Failed:"), result)
                    self.assertIsNone(gateway.local_trajectory_seen)

    def _kinematic_driver(self):
        driver = _make_driver(self.workspace, listen_port=0, r1pro_urdf_path=URDF)
        gateway = _FakeGateway()
        kinematics = R1ProKinematics(URDF)
        trunk = gateway.robot_state["trunk"]["qpos"]
        gateway.hold_action[3:7] = trunk
        for side in ("left", "right"):
            pose = kinematics.forward(side, trunk, [0.0] * 7)
            gateway.robot_state["arms"][side]["eef_position"] = pose[:3, 3].tolist()
            gateway.robot_state["arms"][side]["eef_orientation_xyzw"] = Rotation.from_matrix(pose[:3, :3]).as_quat().tolist()
        driver._gateway = gateway
        return driver, gateway, kinematics

    @unittest.skipUnless(URDF.is_file(), f"official R1Pro URDF not found at {URDF}")
    def test_coarse_direct_target_dispatches_and_reports_a_distinct_reachable_endpoint(self) -> None:
        driver, gateway, kinematics = self._kinematic_driver()
        requested = [1.5, 0.3, 0.6]
        result = driver.execute_action("move_to_pose", {"side": "left", "position_m": requested, "coarse": True})
        self.assertTrue(result.startswith("Move-to-pose execution finished:"), result)
        report = json.loads(result.split("Coarse approach plan: ", 1)[1])
        self.assertEqual(report["requested_position_m"], requested)
        self.assertEqual(report["side"], "left")
        self.assertEqual(report["frame"], "robot_base")
        self.assertGreater(report["remaining_distance_m"], 0.01)
        self.assertFalse(report["collision_checked"])
        self.assertNotIn("measured_position_m", report)
        endpoint = kinematics.forward("left", gateway.robot_state["trunk"]["qpos"], gateway.local_trajectory_seen[-1][7:14])[:3, 3]
        self.assertLess(sum((a - b) ** 2 for a, b in zip(endpoint, report["planned_reachable_position_m"])), 1e-12)

    @unittest.skipUnless(URDF.is_file(), f"official R1Pro URDF not found at {URDF}")
    def test_coarse_bbox_dispatches_with_explicit_left_hand(self) -> None:
        driver, gateway, _ = self._kinematic_driver()
        left = gateway.robot_state["arms"]["left"]["eef_position"]
        result = driver.execute_action("move_to_pose", {
            "side": "left", "coarse": True, "steps": 40,
            "bbox_world_min_m": [left[0] - 0.03, left[1] - 0.03, left[2] - 0.13],
            "bbox_world_max_m": [left[0] + 0.03, left[1] + 0.03, left[2] - 0.07],
            "standoff_m": 0.08,
        })
        self.assertTrue(result.startswith("Move-to-pose execution finished:"), result)
        report = json.loads(result.split("Coarse approach plan: ", 1)[1])
        self.assertEqual(report["side"], "left")
        self.assertEqual(len(gateway.local_trajectory_seen), 40)

    @unittest.skipUnless(URDF.is_file(), f"official R1Pro URDF not found at {URDF}")
    def test_coarse_rejects_invalid_flag_and_endpoint_inside_object_without_dispatch(self) -> None:
        driver, gateway, _ = self._kinematic_driver()
        invalid = driver.execute_action("move_to_pose", {"side": "left", "position_m": [0.4, 0.3, 0.6], "coarse": "true"})
        self.assertIn("coarse must be a boolean", invalid)
        self.assertIsNone(gateway.local_trajectory_seen)
        plan = MotionPlan(((0.0,) * 23,), 0.0, "coarse fixture", {
            "planned_reachable_position_m": [0.5, 0.0, 0.5],
        })
        with patch("robot.drivers.behavior_r1pro_driver.plan_move_to_pose", return_value=plan) as planner:
            result = driver.execute_action("move_to_pose", {
                "side": "left", "coarse": True,
                "bbox_world_min_m": [0.4, -0.1, 0.4],
                "bbox_world_max_m": [0.6, 0.1, 0.6],
            })
            self.assertTrue(planner.call_args.kwargs["coarse"])
        self.assertIn("inside the measured object box", result)
        self.assertIsNone(gateway.local_trajectory_seen)

    def test_move_linear_does_not_accept_approximate_motion(self) -> None:
        self.driver._kinematics = object()
        result = self.driver.execute_action("move_linear", {"side": "left", "delta_m": [0.01, 0, 0], "coarse": True})
        self.assertIn("unknown parameters: coarse", result)
        self.assertIsNone(self.gateway.local_trajectory_seen)

    def test_invalid_configuration_is_rejected_at_construction(self) -> None:
        with self.assertRaises(contracts.ConfigValidationError):
            _make_driver(self.workspace, max_action_steps=0)
        with self.assertRaises(contracts.ConfigValidationError):
            _make_driver(self.workspace, required_instruction="")
        with self.assertRaises(contracts.ConfigValidationError):
            _make_driver(self.workspace, execution_timeout_s=-1)


class BehaviorDriverReadinessTest(unittest.TestCase):
    """An action that cannot produce an episode step must fail before dispatch."""

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.driver = _make_driver(self.workspace, listen_port=0)
        self.gateway = _FakeGateway()
        self.driver._gateway = self.gateway

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_missing_evaluator_blocks_dispatch(self) -> None:
        self.gateway.state["evaluator_connected"] = False
        result = self.driver.execute_action("vla_execute", {"instruction": PROMPT, "step": 5})
        self.assertTrue(result.startswith("Failed:"))
        self.assertIn("evaluator", result)
        self.assertIsNone(self.gateway.baseline_seen)

    def test_missing_upstream_blocks_dispatch(self) -> None:
        self.gateway.state["upstream_connected"] = False
        result = self.driver.execute_action("vla_execute", {"instruction": PROMPT, "step": 5})
        self.assertIn("upstream", result)

    def test_finished_episode_blocks_dispatch(self) -> None:
        (self.workspace / "BEHAVIOR_STATUS.json").write_text(
            json.dumps(_status_payload(phase="completed", done=True, success=True)),
            encoding="utf-8",
        )
        result = self.driver.execute_action("vla_execute", {"instruction": PROMPT, "step": 5})
        self.assertTrue(result.startswith("Failed:"))
        self.assertIn("already finished", result)

    def test_unreadable_status_file_blocks_dispatch(self) -> None:
        (self.workspace / "BEHAVIOR_STATUS.json").write_text("{not json", encoding="utf-8")
        result = self.driver.execute_action("vla_execute", {"instruction": PROMPT, "step": 5})
        self.assertTrue(result.startswith("Failed:"))
        self.assertIn("unusable", result)

    def test_wrong_task_in_status_file_blocks_dispatch(self) -> None:
        (self.workspace / "BEHAVIOR_STATUS.json").write_text(
            json.dumps(_status_payload(task="storing_food")), encoding="utf-8"
        )
        result = self.driver.execute_action("vla_execute", {"instruction": PROMPT, "step": 5})
        self.assertIn("unusable", result)

    def test_instance_id_is_learned_then_enforced(self) -> None:
        (self.workspace / "BEHAVIOR_STATUS.json").write_text(
            json.dumps(_status_payload(instance_id=301, steps=10)), encoding="utf-8"
        )
        self.assertFalse(
            self.driver.execute_action("vla_execute", {"instruction": PROMPT, "step": 5}).startswith(
                "Failed:"
            )
        )
        (self.workspace / "BEHAVIOR_STATUS.json").write_text(
            json.dumps(_status_payload(instance_id=999, steps=10)), encoding="utf-8"
        )
        self.assertIn(
            "unusable",
            self.driver.execute_action("vla_execute", {"instruction": PROMPT, "step": 5}),
        )

    def test_begin_execution_failure_is_reported(self) -> None:
        self.gateway.begin_error = RuntimeError("Another VLA execution is already active")
        result = self.driver.execute_action("vla_execute", {"instruction": PROMPT, "step": 5})
        self.assertTrue(result.startswith("Failed:"))
        self.assertIn("already active", result)


class BehaviorDriverResultTest(unittest.TestCase):
    """Execution outcomes must map onto the queue's terminal-status prefixes."""

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.driver = _make_driver(
            self.workspace, listen_port=0, state_sync_timeout_s=0.05
        )
        self.gateway = _FakeGateway()
        self.driver._gateway = self.gateway
        (self.workspace / "BEHAVIOR_STATUS.json").write_text(
            json.dumps(_status_payload(steps=15)), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _params(self, step: int = 5) -> dict[str, Any]:
        return {"instruction": PROMPT, "step": step}

    def test_synced_completion_reports_applied_steps(self) -> None:
        self.gateway.execution = {
            "execution_id": 1,
            "status": "step_completed",
            "completed_steps": 5,
            "detail": "requested action budget exhausted",
            "sync_observation_baseline": 0,
            "baseline_evaluator_steps": 10,
        }
        result = self.driver.execute_action("vla_execute", self._params())
        self.assertTrue(result.startswith("VLA execution finished: step_completed"))
        self.assertIn("applied_steps=15", result)
        self.assertEqual(self.gateway.published, 1)
        self.assertTrue(self.driver.get_runtime_state()["robots"]["behavior_r1pro"]["state_synced"])

    def test_completion_without_simulator_confirmation_fails(self) -> None:
        self.gateway.sync_result = False
        self.gateway.execution = {
            "execution_id": 1,
            "status": "step_completed",
            "completed_steps": 5,
            "detail": "requested action budget exhausted",
            "sync_observation_baseline": 0,
            "baseline_evaluator_steps": 10,
        }
        result = self.driver.execute_action("vla_execute", self._params())
        self.assertTrue(result.startswith("Failed:"))
        self.assertIn("could not be confirmed", result)

    def test_terminal_evaluator_status_confirms_the_final_action(self) -> None:
        self.gateway.sync_result = False
        status_file = self.workspace / "BEHAVIOR_STATUS.json"
        # The episode ends while the action is in flight, which is why no
        # further observation frame can be expected.
        self.gateway.wait_hook = lambda: status_file.write_text(
            json.dumps(_status_payload(phase="completed", done=True, success=True, steps=15)),
            encoding="utf-8",
        )
        self.gateway.execution = {
            "execution_id": 1,
            "status": "episode_finished",
            "completed_steps": 5,
            "detail": "BEHAVIOR evaluator disconnected",
            "sync_observation_baseline": 0,
            "baseline_evaluator_steps": 10,
        }
        result = self.driver.execute_action("vla_execute", self._params())
        self.assertTrue(result.startswith("VLA execution finished: episode_finished"))
        self.assertIn("applied_steps=15", result)

    def test_interruption_uses_the_cancelled_prefix(self) -> None:
        self.gateway.execution = {
            "execution_id": 1,
            "status": "interrupted",
            "completed_steps": 2,
            "detail": "action interrupted",
        }
        result = self.driver.execute_action("vla_execute", self._params())
        self.assertTrue(result.startswith("Interrupted:"))

    def test_failure_states_use_the_failed_prefix(self) -> None:
        for status in ("error", "timeout", "no_progress"):
            with self.subTest(status=status):
                self.gateway.execution = {
                    "execution_id": 1,
                    "status": status,
                    "completed_steps": 1,
                    "detail": "something went wrong",
                }
                result = self.driver.execute_action("vla_execute", self._params())
                self.assertTrue(result.startswith("Failed:"))
                self.assertIn(status, result)

    def test_unexpected_status_is_a_failure(self) -> None:
        self.gateway.execution = {
            "execution_id": 1,
            "status": "banana",
            "completed_steps": 0,
            "detail": None,
        }
        self.assertTrue(self.driver.execute_action("vla_execute", self._params()).startswith("Failed:"))

    def test_runtime_state_exposes_the_control_contract(self) -> None:
        robot = self.driver.get_runtime_state()["robots"]["behavior_r1pro"]
        self.assertEqual(robot["mode"], "controlled_pi05_gateway")
        self.assertEqual(robot["gateway_status"], "ready")
        self.assertEqual(robot["status"], "idle")
        self.assertEqual(robot["max_action_steps"], self.driver.max_action_steps)
        self.assertIn("state_synced", robot)
        self.assertIn("applied_steps", robot)
        self.assertEqual(
            robot["runtime_progress"],
            {
                "source": "emerge_gateway_diagnostic",
                "gateway_action_steps": 0,
                "execution_count": 0,
                "official_steps_available": True,
            },
        )
        self.assertEqual(robot["evaluation"]["task"], TASK_NAME)

    def test_runtime_progress_accumulates_gateway_steps_without_claiming_official_steps(self) -> None:
        (self.workspace / "BEHAVIOR_STATUS.json").write_text(
            json.dumps(_status_payload(steps=None)), encoding="utf-8"
        )
        self.gateway.execution = {
            "execution_id": 1,
            "status": "step_completed",
            "completed_steps": 3,
            "detail": "requested action budget exhausted",
            "sync_observation_baseline": 0,
            "baseline_evaluator_steps": None,
        }

        result = self.driver.execute_action("vla_execute", self._params(step=3))
        progress = self.driver.get_runtime_state()["robots"]["behavior_r1pro"][
            "runtime_progress"
        ]

        self.assertTrue(result.startswith("VLA execution finished:"), result)
        self.assertEqual(progress["gateway_action_steps"], 3)
        self.assertEqual(progress["execution_count"], 1)
        self.assertFalse(progress["official_steps_available"])

    def test_runtime_state_reports_an_unusable_evaluation_without_crashing(self) -> None:
        (self.workspace / "BEHAVIOR_STATUS.json").write_text("{oops", encoding="utf-8")
        robot = self.driver.get_runtime_state()["robots"]["behavior_r1pro"]
        self.assertIn("unusable", robot["evaluation"])
        self.assertEqual(robot["status"], "idle")

    def test_gateway_status_action_is_a_diagnostic_only(self) -> None:
        result = self.driver.execute_action("gateway_status", {})
        self.assertIn("Gateway status", result)
        self.assertIn("ready", result)

    def test_close_stops_the_gateway(self) -> None:
        self.driver.close()
        self.assertTrue(self.gateway.stopped)


if __name__ == "__main__":
    unittest.main()

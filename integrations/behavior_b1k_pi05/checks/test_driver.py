"""Checks for the BEHAVIOR driver contract, readiness and state synchronization."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from integrations.behavior_b1k_pi05 import contracts
from integrations.behavior_b1k_pi05.driver import BehaviorB1KPi05Driver

PROMPT = contracts.OFFICIAL_PROMPT


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
        "task": contracts.TASK_NAME,
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
        self.driver = BehaviorB1KPi05Driver(workspace=self.workspace, listen_port=0)
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
        self.assertIn("Unknown action type", self.driver.execute_action("move_to_pose", {}))

    def test_invalid_configuration_is_rejected_at_construction(self) -> None:
        with self.assertRaises(contracts.ConfigValidationError):
            BehaviorB1KPi05Driver(workspace=self.workspace, max_action_steps=0)
        with self.assertRaises(contracts.ConfigValidationError):
            BehaviorB1KPi05Driver(workspace=self.workspace, required_instruction="")
        with self.assertRaises(contracts.ConfigValidationError):
            BehaviorB1KPi05Driver(workspace=self.workspace, execution_timeout_s=-1)


class BehaviorDriverReadinessTest(unittest.TestCase):
    """An action that cannot produce an episode step must fail before dispatch."""

    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.driver = BehaviorB1KPi05Driver(workspace=self.workspace, listen_port=0)
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
        self.driver = BehaviorB1KPi05Driver(
            workspace=self.workspace, listen_port=0, state_sync_timeout_s=0.05
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
        self.assertEqual(robot["evaluation"]["task"], contracts.TASK_NAME)

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

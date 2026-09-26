"""Checks for runner supervision, verdict logic and artifact acceptance.

These tests need neither a GPU nor Isaac Sim.  The process-group test is a real
regression guard: the previous ``_stop`` returned without signalling whenever
the process-group leader had already exited, which leaked the evaluator's
descendants.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from integrations.behavior_b1k_pi05 import contracts
from integrations.behavior_b1k_pi05.checks import verify_artifacts
from integrations.behavior_b1k_pi05.run_turning_on_radio_agent import (
    _actions_settled,
    _prepare_workspace,
    _resume_decision,
)
from integrations.behavior_b1k_pi05.runner_common import (
    assert_fresh_output_dir,
    assert_port_available,
    install_workspace_skill,
    read_json,
    spawn_process,
    stop_process_group,
    wait_for_condition,
)
from Emerge.utils.action_queue import append_action, update_action_document


def _pid_state(pid: int) -> str | None:
    """Return the /proc state letter, or None when the pid no longer exists."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    return stat.rsplit(") ", 1)[-1].split(" ", 1)[0]


def _alive(pid: int) -> bool:
    state = _pid_state(pid)
    return state is not None and state != "Z"


class ProcessSupervisionTest(unittest.TestCase):
    def test_process_group_is_stopped_even_after_its_leader_exited(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "child.log"
            with log_path.open("wb") as log:
                process = spawn_process(
                    [
                        sys.executable,
                        "-c",
                        "import subprocess; p = subprocess.Popen(['sleep', '30']);"
                        " print(p.pid, flush=True)",
                    ],
                    cwd=root,
                    log=log,
                )
            process.wait(timeout=15)
            self.assertIsNotNone(process.poll(), "the group leader should have exited")

            deadline = time.monotonic() + 10
            child_pid: int | None = None
            while time.monotonic() < deadline and child_pid is None:
                text = log_path.read_text(encoding="utf-8").strip()
                if text:
                    child_pid = int(text.splitlines()[0])
                else:
                    time.sleep(0.05)
            self.assertIsNotNone(child_pid, "the leader never reported its child pid")
            assert child_pid is not None
            self.assertTrue(_alive(child_pid), "the helper process should be running")

            stop_process_group(process, timeout_s=5)

            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and _alive(child_pid):
                time.sleep(0.05)
            self.assertFalse(_alive(child_pid), "the descendant process leaked")

    def test_wait_for_condition_fails_fast_when_a_supervised_process_dies(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (root / "log").open("wb") as log:
                process = spawn_process(
                    [sys.executable, "-c", "import sys; sys.exit(3)"], cwd=root, log=log
                )
            process.wait(timeout=15)
            with self.assertRaises(RuntimeError) as caught:
                wait_for_condition(
                    lambda: False,
                    timeout_s=10,
                    label="something that never happens",
                    processes=[process],
                )
            self.assertIn("exited early", str(caught.exception))

    def test_wait_for_condition_times_out_with_a_useful_message(self) -> None:
        with self.assertRaises(TimeoutError) as caught:
            wait_for_condition(lambda: False, timeout_s=0.2, label="the impossible")
        self.assertIn("the impossible", str(caught.exception))


class PortAndOutputDirectoryTest(unittest.TestCase):
    def test_a_listening_port_is_detected_before_spawning(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            port = int(sock.getsockname()[1])
            with self.assertRaises(RuntimeError) as caught:
                assert_port_available(port, label="OpenPI server")
            self.assertIn("already in use", str(caught.exception))

    def test_a_free_port_is_accepted(self) -> None:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
        assert_port_available(port)

    def test_reusing_an_episode_output_directory_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "episode"
            self.assertEqual(assert_fresh_output_dir(root), root)
            root.mkdir(parents=True)
            (root / "BEHAVIOR_STATUS.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                assert_fresh_output_dir(root)
            self.assertEqual(assert_fresh_output_dir(root, allow_existing=True), root)

    def test_read_json_is_safe_for_missing_or_broken_files(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(read_json(root / "missing.json"), {})
            (root / "broken.json").write_text("{oops", encoding="utf-8")
            self.assertEqual(read_json(root / "broken.json"), {})
            (root / "list.json").write_text("[1, 2]", encoding="utf-8")
            self.assertEqual(read_json(root / "list.json"), {})


class EpisodeVerdictTest(unittest.TestCase):
    """The old ``done=true`` + ``phase=error`` trap must stay closed."""

    def test_evaluator_error_is_never_success(self) -> None:
        status = {"task": "turning_on_radio", "phase": "error", "done": True, "success": False}
        self.assertEqual(contracts.classify_episode(status), "evaluator_error")
        self.assertEqual(
            contracts.agent_exit_code(verdict="evaluator_error", success=False, require_success=False),
            1,
        )

    def test_task_failure_is_a_clean_finish_by_default(self) -> None:
        status = {"phase": "completed", "done": True, "success": False}
        self.assertEqual(contracts.classify_episode(status), "task_failed")
        self.assertEqual(
            contracts.agent_exit_code(verdict="task_failed", success=False, require_success=False),
            0,
        )
        self.assertEqual(
            contracts.agent_exit_code(verdict="task_failed", success=False, require_success=True),
            1,
        )

    def test_success_is_zero_in_both_modes(self) -> None:
        status = {"phase": "completed", "done": True, "success": True}
        self.assertEqual(contracts.classify_episode(status), "success")
        self.assertTrue(contracts.episode_succeeded(status))
        for require in (False, True):
            self.assertEqual(
                contracts.agent_exit_code(verdict="success", success=True, require_success=require),
                0,
            )

    def test_unfinished_episode_is_incomplete(self) -> None:
        status = {"phase": "running", "done": False, "success": False}
        self.assertEqual(contracts.classify_episode(status), "incomplete")
        self.assertFalse(contracts.episode_terminated(status))

    def test_infrastructure_error_overrides_everything(self) -> None:
        status = {"phase": "completed", "done": True, "success": True}
        self.assertEqual(
            contracts.classify_episode(status, infrastructure_error="controller_exit_1"),
            "infrastructure_error",
        )
        self.assertEqual(
            contracts.agent_exit_code(
                verdict="infrastructure_error", success=False, require_success=False
            ),
            1,
        )

    def test_nonzero_evaluator_exit_without_a_final_status_is_infrastructure(self) -> None:
        status = {"phase": "running", "done": False}
        self.assertEqual(
            contracts.classify_episode(status, evaluator_returncode=1), "infrastructure_error"
        )

    def test_only_benign_finish_reasons_are_resumable(self) -> None:
        self.assertTrue(contracts.is_resumable_finish_reason("max_iterations"))
        self.assertTrue(contracts.is_resumable_finish_reason("length"))
        for reason in (
            "error",
            "timeout",
            "cancel_requested",
            "cancellation_unconfirmed",
            "unfinished_action",
            "cleanup_error",
            "runtime_error",
            "invalid_result",
            "",
            None,
        ):
            with self.subTest(reason=reason):
                self.assertFalse(contracts.is_resumable_finish_reason(reason))


class ResumeDecisionTest(unittest.TestCase):
    def _decide(self, reason: str, *, rounds: int = 0, maximum: int = 2, settled: bool = True) -> str:
        return _resume_decision(
            finish_reason=reason,
            feedback_rounds=rounds,
            max_rounds=maximum,
            actions_settled=settled,
        )

    def test_max_iterations_resumes_the_same_session(self) -> None:
        self.assertEqual(self._decide("max_iterations"), "resume")

    def test_unconfirmed_cancellation_never_resumes(self) -> None:
        self.assertEqual(self._decide("cancellation_unconfirmed"), "infrastructure")
        self.assertEqual(self._decide("unfinished_action"), "infrastructure")

    def test_provider_error_stops_without_being_infrastructure(self) -> None:
        self.assertEqual(self._decide("error"), "stop")

    def test_timeout_stops_because_the_budget_is_already_spent(self) -> None:
        self.assertEqual(self._decide("timeout"), "stop")

    def test_round_limit_is_reported_separately(self) -> None:
        self.assertEqual(self._decide("max_iterations", rounds=2, maximum=2), "exhausted")

    def test_an_unsettled_action_blocks_the_resume(self) -> None:
        self.assertEqual(self._decide("max_iterations", settled=False), "blocked")

    def test_actions_settled_reads_the_real_queue(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            self.assertTrue(_actions_settled(workspace))
            action_file = workspace / "ACTION.md"
            action_id = str(
                update_action_document(
                    action_file,
                    lambda current: append_action(
                        current, action_type="vla_execute", parameters={"instruction": "x", "step": 1}
                    ),
                )["actions"][-1]["id"]
            )
            self.assertFalse(_actions_settled(workspace))

            def finish(document):
                for item in document["actions"]:
                    if item["id"] == action_id:
                        item.update(status="completed", result="done")

            update_action_document(action_file, finish)
            self.assertTrue(_actions_settled(workspace))


class AgentWorkspacePreparationTest(unittest.TestCase):
    def test_preparation_installs_the_skill_and_does_not_duplicate_instructions(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            _prepare_workspace(workspace)
            _prepare_workspace(workspace)
            agents = (workspace / "AGENTS.md").read_text(encoding="utf-8")
            self.assertEqual(agents.count("BEHAVIOR-1K Evaluation Rules"), 1)
            self.assertTrue((workspace / "skills/vla/SKILL.md").is_file())
            self.assertTrue((workspace / "ACTION.md").is_file())
            self.assertIn("turning_on_radio", (workspace / "ROBOT_STATE.md").read_text())


class VerifyArtifactsTest(unittest.TestCase):
    """The acceptance checker is the only judge of a successful episode."""

    def _episode(
        self,
        root: Path,
        *,
        success: bool = True,
        phase: str = "completed",
        recorded_success: bool | None = None,
        break_image: bool = False,
        agent_success: bool | None = None,
    ) -> Path:
        import numpy as np

        from integrations.behavior_b1k_pi05.observation import ObservationPublisher

        workspace = root / "workspace"
        workspace.mkdir(parents=True)
        json_dir = root / "json"
        json_dir.mkdir()
        instance_id = 301
        result_path = json_dir / f"turning_on_radio_{instance_id}_0.json"
        result_path.write_text(
            json.dumps(
                {
                    "task": "turning_on_radio",
                    "instance_id": instance_id,
                    "rollout_id": 0,
                    "steps": 501,
                    "success": success if recorded_success is None else recorded_success,
                    "q_score": {"final": 1.0 if success else 0.0},
                }
            ),
            encoding="utf-8",
        )
        (workspace / "BEHAVIOR_STATUS.json").write_text(
            json.dumps(
                {
                    "schema_version": contracts.EVALUATION_STATUS_SCHEMA,
                    "task": "turning_on_radio",
                    "instance_id": instance_id,
                    "rollout_id": 0,
                    "phase": phase,
                    "steps": 501,
                    "success": success,
                    "done": True,
                    "terminated": False,
                    "truncated": False,
                    "termination_reason": "success" if success else "truncated",
                    "result_path": str(result_path),
                    "q_score": 1.0 if success else 0.0,
                }
            ),
            encoding="utf-8",
        )
        workspace.joinpath("ACTION.md").write_text("", encoding="utf-8")
        publisher = ObservationPublisher(workspace, publish_interval=1)
        import msgpack

        def pack(value):
            return {
                b"__ndarray__": True,
                b"data": value.tobytes(),
                b"dtype": value.dtype.str,
                b"shape": value.shape,
            }

        image = np.zeros((4, 5, 3), dtype=np.uint8)
        publisher.capture(
            msgpack.packb({"robot::robot:zed_link:Camera:0::rgb": image}, default=pack)
        )
        if break_image:
            manifest = json.loads(publisher.manifest_path.read_text(encoding="utf-8"))
            (workspace / manifest["views"][0]["image_path"]).unlink()

        agent_success = success if agent_success is None else agent_success
        (root / "agent_episode_result.json").write_text(
            json.dumps(
                {
                    "task": "turning_on_radio",
                    "success": agent_success,
                    "verdict": "success" if agent_success else "task_failed",
                    "infrastructure_error": None,
                }
            ),
            encoding="utf-8",
        )
        return root

    def _failures(self, checks) -> list[str]:
        return [message for ok, message in checks if not ok]

    def test_a_successful_episode_passes_every_check(self) -> None:
        with TemporaryDirectory() as temporary:
            root = self._episode(Path(temporary))
            checks = verify_artifacts.verify(root, require_success=True, require_agent_result=True)
            self.assertEqual(self._failures(checks), [])
            self.assertGreater(len(checks), 8)

    def test_unsuccessful_episode_fails_only_when_success_is_required(self) -> None:
        with TemporaryDirectory() as temporary:
            root = self._episode(Path(temporary), success=False)
            self.assertEqual(self._failures(verify_artifacts.verify(root)), [])
            failures = self._failures(verify_artifacts.verify(root, require_success=True))
            self.assertTrue(any("official success" in message for message in failures))

    def test_evaluator_error_phase_fails(self) -> None:
        with TemporaryDirectory() as temporary:
            root = self._episode(Path(temporary), success=False, phase="error")
            failures = self._failures(verify_artifacts.verify(root))
            self.assertTrue(any("evaluator did not fail" in message for message in failures))

    def test_disagreement_between_status_and_result_fails(self) -> None:
        with TemporaryDirectory() as temporary:
            root = self._episode(Path(temporary), success=True, recorded_success=False)
            failures = self._failures(verify_artifacts.verify(root))
            self.assertTrue(any("agrees with the live status" in message for message in failures))

    def test_missing_camera_image_fails(self) -> None:
        with TemporaryDirectory() as temporary:
            root = self._episode(Path(temporary), break_image=True)
            failures = self._failures(verify_artifacts.verify(root))
            self.assertTrue(any("camera image exists" in message for message in failures))

    def test_agent_result_disagreement_fails(self) -> None:
        with TemporaryDirectory() as temporary:
            root = self._episode(Path(temporary), success=True, agent_success=False)
            failures = self._failures(verify_artifacts.verify(root))
            self.assertTrue(any("agent episode result agrees" in message for message in failures))

    def test_missing_output_directory_fails_without_crashing(self) -> None:
        with TemporaryDirectory() as temporary:
            checks = verify_artifacts.verify(Path(temporary) / "nope", require_success=True)
            self.assertTrue(self._failures(checks))
            self.assertEqual(verify_artifacts.main(["--output-dir", str(Path(temporary) / "nope")]), 1)

    def test_main_returns_zero_for_a_verified_episode(self) -> None:
        with TemporaryDirectory() as temporary:
            root = self._episode(Path(temporary))
            self.assertEqual(
                verify_artifacts.main(["--output-dir", str(root), "--require-success"]), 0
            )


if __name__ == "__main__":
    unittest.main()

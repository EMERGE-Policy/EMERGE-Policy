"""Checks for runner supervision, verdict logic and artifact acceptance.

These tests need neither a GPU nor Isaac Sim.  The process-group test is a real
regression guard: the previous ``_stop`` returned without signalling whenever
the process-group leader had already exited, which leaked the evaluator's
descendants.
"""

from __future__ import annotations

import json
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from Emerge.utils.action_queue import append_action, update_action_document
from robot.behavior import contracts
from robot.behavior.observation import CAMERA_POSE_ORDER
from scripts.Behavior_eval.checks import verify_artifacts
from scripts.Behavior_eval.eval_behavior_agent import (
    DEFAULT_GATEWAY_PORT,
    DEFAULT_OPENPI_PORT,
    _actions_settled,
    _episode_driver_config,
    _evaluator_command,
    _gracefully_stop_evaluator,
    _prepare_workspace,
    _render_navigation_video,
    _navigation_summary,
    _read_official_result,
    _resume_decision,
    _validate_behavior_perception,
    _validate_external_openpi,
    _validate_lightnav,
    _write_completed_evaluation_status,
    _write_running_evaluation_status,
    NAVIGATION_PROFILE,
    parse_args,
)
from scripts.Behavior_eval.runner_common import (
    assert_fresh_output_dir,
    assert_port_available,
    read_json,
    service_health_ok,
    spawn_process,
    stop_process_group,
    wait_for_condition,
)

TEST_TASK = "turning_on_radio"
TEST_PROMPT = "Turn on the radio receiver that's on the table in the living room."


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


class RunnerCliTest(unittest.TestCase):
    def test_defaults_use_the_shorter_budget_and_headless_mode(self) -> None:
        with patch.object(sys, "argv", ["eval_behavior_agent.py"]):
            args = parse_args()
        self.assertIsNone(args.max_steps)
        self.assertIsNone(args.agent_max_iterations)
        self.assertTrue(args.headless)
        self.assertFalse(args.show_agent_events)
        self.assertFalse(args.preflight_only)
        self.assertEqual(args.robot_config.name, "behavior_r1pro_openpi.yaml")
        self.assertEqual(args.gateway_port, DEFAULT_GATEWAY_PORT)
        self.assertEqual(args.upstream_port, DEFAULT_OPENPI_PORT)
        self.assertEqual(args.gateway_port, 8000)
        self.assertEqual(args.upstream_port, 8011)
        self.assertEqual(
            args.behavior_root,
            Path(__file__).resolve().parents[3] / "third_party/behavior-1k",
        )
        self.assertIsNone(args.output_dir)
        self.assertIsNone(args.task_name)
        self.assertIsNone(args.task_instruction)
        self.assertFalse(args.navigation_only)
        self.assertIsNone(args.navigation_goal)

    def test_navigation_mode_is_explicit_and_can_open_simulator_window(self) -> None:
        with patch.object(sys, "argv", [
            "eval_behavior_agent.py", "--navigation-only",
            "--navigation-goal", "Approach the radio", "--max-steps", "900",
            "--no-headless",
        ]):
            args = parse_args()
        self.assertTrue(args.navigation_only)
        self.assertEqual(args.navigation_goal, "Approach the radio")
        self.assertEqual(args.max_steps, 900)
        self.assertFalse(args.headless)

    def test_native_config_name_and_live_options_are_accepted(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "eval_behavior_agent.py",
                "--config",
                "/tmp/provider.json",
                "--upstream-port",
                "9011",
                "--gateway-port",
                "9010",
                "--agent-max-iterations",
                "80",
                "--no-headless",
                "--show-agent-events",
            ],
        ):
            args = parse_args()
        self.assertEqual(args.agent_config, Path("/tmp/provider.json"))
        self.assertEqual(args.upstream_port, 9011)
        self.assertEqual(args.gateway_port, 9010)
        self.assertEqual(args.agent_max_iterations, 80)
        self.assertFalse(args.headless)
        self.assertTrue(args.show_agent_events)

    def test_preflight_only_is_an_explicit_mode(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["eval_behavior_agent.py", "--preflight-only"],
        ):
            args = parse_args()
        self.assertTrue(args.preflight_only)


class NavigationVideoShutdownTest(unittest.TestCase):
    def test_navigation_summary_counts_all_segments_not_only_the_last(self) -> None:
        calls = [{"nav_steps": 30, "model_frames": 4, "finish_reason": "step_budget"},
                 {"nav_steps": 15, "model_frames": 3, "finish_reason": "model_stop_candidate"}]
        result = _navigation_summary({"navigation_calls": calls, "last_navigation": calls[-1]})
        self.assertEqual(result["nav_steps"], 45)
        self.assertEqual(result["model_frames"], 7)
        self.assertEqual(result["call_count"], 2)
        self.assertNotIn("success", result)
        self.assertIsNone(_navigation_summary({}))

    def test_evaluator_receives_sigint_before_waiting(self) -> None:
        process = Mock()
        process.poll.return_value = None
        process.wait.return_value = 0
        self.assertTrue(_gracefully_stop_evaluator(process, timeout_s=3.0))
        self.assertEqual(process.mock_calls[:3], [
            unittest.mock.call.poll(),
            unittest.mock.call.send_signal(signal.SIGINT),
            unittest.mock.call.wait(timeout=3.0),
        ])

    def test_evaluator_timeout_allows_existing_force_cleanup(self) -> None:
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("evaluator", 0.01)
        self.assertFalse(_gracefully_stop_evaluator(process, timeout_s=0.01))

    def test_navigation_frames_produce_playable_mp4(self) -> None:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            self.skipTest("ffmpeg and ffprobe are required")
        from PIL import Image

        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            frames = output_dir / "lightnav_frames"
            frames.mkdir()
            for index in range(2):
                Image.new("RGB", (32, 32), (200, index * 100, 0)).save(
                    frames / f"frame_{index:05d}.jpg"
                )
            output = _render_navigation_video(frames, output_dir)
            self.assertEqual(output, output_dir / "navigation_head.mp4")
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1", str(output)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)
            self.assertIn("duration=", probe.stdout)


class PerceptionPreflightTest(unittest.TestCase):
    def test_behavior_requires_rgbd_geometry_mode(self) -> None:
        from Emerge.config.schema import Config

        config = Config()
        config.subagents.object_location.geometry_backend = "vggt"
        with self.assertRaisesRegex(ValueError, "calibrated RGB-D"):
            _validate_behavior_perception(config)

    def test_missing_sam3_fails_when_perception_is_required(self) -> None:
        from Emerge.config.schema import Config

        with patch(
            "scripts.Behavior_eval.eval_behavior_agent.service_health_ok",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "SAM3 is not healthy"):
                _validate_behavior_perception(Config())

    def test_complete_task_can_run_vla_without_optional_segmentation(self) -> None:
        from Emerge.config.schema import Config

        with patch("scripts.Behavior_eval.eval_behavior_agent.service_health_ok", return_value=False):
            self.assertEqual(
                _validate_behavior_perception(Config(), required=False), "ws://localhost:8002"
            )

    def test_lightnav_is_optional_for_complete_task_but_required_for_nav_debug(self) -> None:
        with patch("integrations.lightnav_sonic_emerge.bridge.lightnav_ws.LightNavWsClient",
                   side_effect=ConnectionRefusedError("test server is offline")):
            self.assertFalse(_validate_lightnav(required=False))
            with self.assertRaisesRegex(RuntimeError, "LightNav is unavailable"):
                _validate_lightnav(required=True)

    def test_lightnav_preflight_checks_protocol_without_predicting(self) -> None:
        with patch("integrations.lightnav_sonic_emerge.bridge.lightnav_ws.LightNavWsClient") as factory:
            self.assertTrue(_validate_lightnav(required=False))
            factory.return_value.__enter__.assert_called_once()
            factory.return_value.predict.assert_not_called()

    def test_healthy_sam3_passes(self) -> None:
        from Emerge.config.schema import Config

        config = Config()
        with patch(
            "scripts.Behavior_eval.eval_behavior_agent.service_health_ok",
            return_value=True,
        ):
            self.assertEqual(
                _validate_behavior_perception(config),
                "ws://localhost:8002",
            )

    def test_service_health_rejects_invalid_urls(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid service URL"):
            service_health_ok("not-a-url")

    def test_external_openpi_must_already_be_healthy(self) -> None:
        with patch(
            "scripts.Behavior_eval.eval_behavior_agent.service_health_ok",
            return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "OpenPI is not healthy"):
                _validate_external_openpi(8011)
        with patch(
            "scripts.Behavior_eval.eval_behavior_agent.service_health_ok",
            return_value=True,
        ):
            self.assertEqual(_validate_external_openpi(8011), "ws://127.0.0.1:8011")


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


class NativeDriverConfigurationTest(unittest.TestCase):
    def test_navigation_only_uses_an_episode_profile_and_action_guard(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = _episode_driver_config(
                {
                    "workspace": "unused", "upstream_url": "ws://127.0.0.1:1",
                    "required_instruction": TEST_PROMPT, "max_action_steps": 1,
                    "expected_task": TEST_TASK,
                },
                output_dir=root, workspace=root / "workspace",
                status_path=root / "workspace/BEHAVIOR_STATUS.json",
                upstream_port=8123, gateway_port=8124,
                max_action_steps=900, state_sync_timeout_s=45.0,
                task_name=TEST_TASK, task_instruction=TEST_PROMPT,
                navigation_only=True,
            )
            config = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(config["navigation_only"])
        self.assertEqual(config["profile_path"], str(NAVIGATION_PROFILE))
        self.assertEqual(config["max_action_steps"], 900)

    def test_episode_config_targets_the_native_driver_and_workspace(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / "workspace"
            status_path = workspace / "BEHAVIOR_STATUS.json"
            path = _episode_driver_config(
                {
                    "workspace": "unused",
                    "upstream_url": "ws://127.0.0.1:1",
                    "required_instruction": TEST_PROMPT,
                    "max_action_steps": 1,
                    "expected_task": TEST_TASK,
                },
                output_dir=root,
                workspace=workspace,
                status_path=status_path,
                upstream_port=8123,
                gateway_port=8124,
                max_action_steps=20_002,
                state_sync_timeout_s=45.0,
                task_name=TEST_TASK,
                task_instruction=TEST_PROMPT,
            )
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(config["workspace"], str(workspace))
            self.assertEqual(config["upstream_url"], "ws://127.0.0.1:8123")
            self.assertEqual(config["listen_port"], 8124)
            self.assertEqual(config["max_action_steps"], 20_002)
            self.assertEqual(config["status_path"], str(status_path))
            self.assertEqual(config["expected_task"], TEST_TASK)
            self.assertFalse(config["navigation_only"])
            self.assertIsNone(config["profile_path"])

    def test_behavior_driver_is_registered_without_a_plugin(self) -> None:
        from robot.drivers import DRIVER_REGISTRY

        self.assertEqual(
            DRIVER_REGISTRY["behavior_r1pro"],
            "robot.drivers.behavior_r1pro_driver.BehaviorR1ProDriver",
        )

    def test_checked_in_driver_ports_do_not_conflict_with_perception_services(self) -> None:
        from Emerge.config.schema import ObjectLocationSubagentConfig

        config_path = Path(__file__).resolve().parents[3] / "dev/behavior_agent_eval.json"
        driver_config = json.loads(config_path.read_text(encoding="utf-8"))
        perception = ObjectLocationSubagentConfig()

        self.assertEqual(driver_config["listen_port"], DEFAULT_GATEWAY_PORT)
        self.assertEqual(driver_config["upstream_url"], f"ws://127.0.0.1:{DEFAULT_OPENPI_PORT}")
        self.assertEqual(perception.vggt_url, "ws://localhost:8001")
        self.assertEqual(perception.sam3_url, "ws://localhost:8002")
        self.assertEqual(
            len({DEFAULT_GATEWAY_PORT, 8001, 8002, DEFAULT_OPENPI_PORT}),
            4,
        )

    def test_behavior_service_launcher_uses_the_shared_model_server_entrypoint(self) -> None:
        launcher = Path(__file__).resolve().parents[1] / "start_behavior_services.sh"
        result = subprocess.run(
            ["bash", "-n", str(launcher)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        text = launcher.read_text(encoding="utf-8")
        self.assertIn('OPENPI_SERVER_MODE="behavior_champion"', text)
        self.assertIn("third_party/behavior-1k-solution", text)
        self.assertIn("task_checkpoint_mapping.json", text)

    def test_checked_in_camera_intrinsics_match_official_rgbd_wrapper(self) -> None:
        config_path = Path(__file__).resolve().parents[3] / "dev/behavior_agent_eval.json"
        driver_config = contracts.validate_driver_config(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
        cameras = driver_config["camera_calibration"]["cameras"]

        self.assertEqual(cameras["head"]["resolution"], [720, 720])
        self.assertEqual(cameras["left_wrist"]["resolution"], [480, 480])
        self.assertAlmostEqual(cameras["head"]["intrinsics"][0][0], 306.0)
        self.assertAlmostEqual(cameras["left_wrist"]["intrinsics"][0][0], 388.6639)
        self.assertEqual(cameras["right_wrist"], cameras["left_wrist"])

    def test_evaluator_command_targets_stock_behavior_393(self) -> None:
        command = _evaluator_command(
            behavior_python=Path("/env/bin/python"),
            robot_config=Path("/checkout/OmniGibson/omnigibson/eval/r1pro.yaml"),
            gateway_port=8124,
            task_name=TEST_TASK,
            instance_index=3,
            max_steps=6000,
            output_dir=Path("/tmp/episode"),
            write_video=True,
            headless=False,
        )
        self.assertNotIn("--status-file", command)
        self.assertEqual(command[command.index("--num-envs") + 1], "1")
        self.assertEqual(command[command.index("--replay-action-chunk-size") + 1], "0")
        self.assertEqual(command[command.index("--instance-indices") + 1], "3")
        self.assertEqual(command[command.index("--max-steps") + 1], "6000")
        self.assertIn("/checkout/OmniGibson/omnigibson/eval/r1pro.yaml", command)
        self.assertIn("--write-video", command)
        self.assertIn("--no-headless", command)
        self.assertNotIn("--verification-observation-workspace", command)
        self.assertNotIn("--verification-publish-interval", command)
        self.assertEqual(
            command[command.index("--env-wrapper") + 1],
            "omnigibson.eval.wrappers.RGBDFullResWrapper",
        )

    def test_openpi_robot_config_exposes_the_checkpoint_observation_names(self) -> None:
        import yaml

        config_path = Path(__file__).resolve().parents[3] / "dev/behavior_r1pro_openpi.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["model"], "r1pro")
        self.assertEqual(config["name"], "robot")
        self.assertEqual(
            config["eval"]["camera_sensor_names"],
            {
                "left_wrist": "robot:left_realsense_link:Camera:0",
                "right_wrist": "robot:right_realsense_link:Camera:0",
                "head": "robot:zed_link:Camera:0",
            },
        )
        self.assertEqual(
            list(config["eval"]["camera_sensor_names"]),
            list(CAMERA_POSE_ORDER),
        )
        self.assertEqual(
            config["proprio_obs"],
            [
                "base_qvel",
                "arm_left_qpos",
                "arm_left_qvel",
                "eef_left_pos",
                "eef_left_quat",
                "gripper_left_qpos",
                "gripper_left_qvel",
                "arm_right_qpos",
                "arm_right_qvel",
                "eef_right_pos",
                "eef_right_quat",
                "gripper_right_qpos",
                "gripper_right_qvel",
                "trunk_qpos",
                "trunk_qvel",
            ],
        )


class OfficialResultBridgeTest(unittest.TestCase):
    def _write_result(self, root: Path, **overrides) -> Path:
        result = {
            "task": TEST_TASK,
            "instance_id": 301,
            "rollout_id": 0,
            "steps": 6001,
            "success": False,
            "q_score": {"final": 0.25},
        }
        result.update(overrides)
        path = root / "json/turning_on_radio_301_0.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(result), encoding="utf-8")
        return path

    def test_running_status_is_owned_by_emerge_and_not_final(self) -> None:
        with TemporaryDirectory() as temporary:
            status_path = Path(temporary) / "workspace/BEHAVIOR_STATUS.json"
            status = _write_running_evaluation_status(
                status_path, task_name=TEST_TASK
            )
            self.assertEqual(status["source"], "emerge_supervisor")
            self.assertEqual(status["phase"], "running")
            self.assertIsNone(status["steps"])
            self.assertEqual(status["steps_source"], "unavailable_until_official_result")
            self.assertFalse(status["done"])
            self.assertEqual(read_json(status_path), status)

    def test_official_result_becomes_the_final_episode_status(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_path = self._write_result(root, success=True)
            status_path = root / "workspace/BEHAVIOR_STATUS.json"
            status = _write_completed_evaluation_status(
                root, status_path, task_name=TEST_TASK
            )
            self.assertEqual(status["source"], "official_result")
            self.assertEqual(status["instance_id"], 301)
            self.assertEqual(status["steps"], 6001)
            self.assertTrue(status["success"])
            self.assertTrue(status["done"])
            self.assertEqual(status["q_score"], 0.25)
            self.assertEqual(status["result_path"], str(result_path.resolve()))
            self.assertEqual(read_json(status_path), status)

    def test_missing_or_ambiguous_result_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(RuntimeError):
                _read_official_result(root, task_name=TEST_TASK)
            self._write_result(root)
            second = root / "json/turning_on_radio_302_0.json"
            second.write_text(
                json.dumps(
                    {
                        "task": TEST_TASK,
                        "instance_id": 302,
                        "rollout_id": 0,
                        "steps": 1,
                        "success": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                _read_official_result(root, task_name=TEST_TASK)

    def test_mismatched_or_invalid_official_result_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_result(root, task="storing_food")
            with self.assertRaises(RuntimeError):
                _read_official_result(root, task_name=TEST_TASK)


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

class ResumeDecisionTest(unittest.TestCase):
    def _decide(
        self,
        reason: str,
        *,
        returncode: int = 0,
        status: dict | None = None,
        rounds: int = 0,
        maximum: int = 2,
        settled: bool = True,
    ) -> str:
        return _resume_decision(
            returncode=returncode,
            finish_reason=reason,
            status=status or {"success": False, "done": False},
            feedback_rounds=rounds,
            max_rounds=maximum,
            actions_settled=settled,
        )

    def test_clean_agent_stop_resumes_the_same_session(self) -> None:
        self.assertEqual(self._decide("stop"), "resume_feedback")

    def test_max_iterations_is_a_same_session_checkpoint(self) -> None:
        self.assertEqual(
            self._decide("max_iterations", returncode=1),
            "resume_checkpoint",
        )

    def test_context_limit_does_not_resume(self) -> None:
        self.assertEqual(self._decide("length", returncode=1), "stop")

    def test_unconfirmed_cancellation_never_resumes(self) -> None:
        self.assertEqual(self._decide("cancellation_unconfirmed"), "infrastructure")
        self.assertEqual(self._decide("unfinished_action"), "infrastructure")

    def test_provider_error_is_infrastructure(self) -> None:
        self.assertEqual(self._decide("error", returncode=1), "infrastructure")

    def test_timeout_stops_because_the_budget_is_already_spent(self) -> None:
        self.assertEqual(self._decide("timeout", returncode=124), "stop")

    def test_round_limit_is_reported_separately(self) -> None:
        self.assertEqual(self._decide("stop", rounds=2, maximum=2), "exhausted")
        self.assertEqual(
            self._decide("max_iterations", returncode=1, rounds=2, maximum=2),
            "resume_checkpoint",
        )

    def test_an_unsettled_action_blocks_the_resume(self) -> None:
        self.assertEqual(self._decide("stop", settled=False), "blocked")
        self.assertEqual(
            self._decide("max_iterations", returncode=1, settled=False),
            "blocked",
        )

    def test_finished_official_episode_does_not_resume(self) -> None:
        self.assertEqual(
            self._decide("stop", status={"success": True, "done": True}),
            "stop",
        )

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
    def test_preparation_uses_generic_skills_and_does_not_duplicate_instructions(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary) / "workspace"
            _prepare_workspace(
                workspace, task_name=TEST_TASK, instruction=TEST_PROMPT
            )
            _prepare_workspace(
                workspace, task_name=TEST_TASK, instruction=TEST_PROMPT
            )
            agents = (workspace / "AGENTS.md").read_text(encoding="utf-8")
            self.assertEqual(
                agents,
                (Path(__file__).resolve().parents[3] / "Emerge/templates/AGENTS.md").read_text(
                    encoding="utf-8"
                ),
            )
            self.assertFalse((workspace / "skills/vla/SKILL.md").exists())
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
        break_depth: bool = False,
        agent_success: bool | None = None,
    ) -> Path:
        import numpy as np

        from robot.behavior.observation import ObservationPublisher

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
        depth = np.ones((4, 5, 1), dtype=np.float32)
        publisher.capture(
            msgpack.packb(
                {
                    "robot::robot:zed_link:Camera:0::rgb": image,
                    "robot::robot:zed_link:Camera:0::depth_linear": depth,
                },
                default=pack,
            )
        )
        if break_image:
            manifest = json.loads(publisher.manifest_path.read_text(encoding="utf-8"))
            (workspace / manifest["views"][0]["image_path"]).unlink()
        if break_depth:
            manifest = json.loads(publisher.manifest_path.read_text(encoding="utf-8"))
            (workspace / manifest["views"][0]["depth_path"]).unlink()

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

    def test_missing_depth_array_fails_for_the_rgbd_manifest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = self._episode(Path(temporary), break_depth=True)
            failures = self._failures(verify_artifacts.verify(root))
            self.assertTrue(any("linear-depth array exists" in message for message in failures))

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

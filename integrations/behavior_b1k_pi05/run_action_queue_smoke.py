#!/usr/bin/env python3
"""Run turning_on_radio through the real EMERGE Controller and ACTION.md.

The evaluator owns the simulator, the Controller owns the gateway as an EMERGE
external driver, and this runner only places bounded ``vla_execute`` actions in
the real action queue.  It needs a GPU and Isaac Sim but no language model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parent
EMERGE_ROOT = PLUGIN_ROOT.parents[1]
WORKSPACE_ROOT = EMERGE_ROOT.parent
if str(EMERGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EMERGE_ROOT))

try:
    from .contracts import OFFICIAL_PROMPT, TASK_NAME
except ImportError:  # Direct script execution adds this directory to sys.path.
    sys.path.insert(0, str(PLUGIN_ROOT))
    from contracts import OFFICIAL_PROMPT, TASK_NAME  # type: ignore[no-redef]

try:
    from .runner_common import (
        assert_fresh_output_dir,
        assert_port_available,
        atomic_write_json,
        behavior_environment,
        default_output_dir,
        install_workspace_skill,
        prepare_workspace,
        read_json,
        require_executable,
        require_path,
        robot_state_section,
        spawn_process,
        stop_process_group,
        wait_for_condition,
        wait_for_health,
        wait_for_process,
    )
except ImportError:  # Direct script execution.
    from runner_common import (  # type: ignore[no-redef]
        assert_fresh_output_dir,
        assert_port_available,
        atomic_write_json,
        behavior_environment,
        default_output_dir,
        install_workspace_skill,
        prepare_workspace,
        read_json,
        require_executable,
        require_path,
        robot_state_section,
        spawn_process,
        stop_process_group,
        wait_for_condition,
        wait_for_health,
        wait_for_process,
    )

from Emerge.utils.action_queue import (  # noqa: E402
    append_action,
    normalize_action_document,
    parse_action_markdown,
    update_action_document,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpi-root", type=Path, default=WORKSPACE_ROOT / "openpi")
    parser.add_argument("--behavior-root", type=Path, default=WORKSPACE_ROOT / "BEHAVIOR-1K")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=WORKSPACE_ROOT / "checkpoints/pi05TurningOnRadio/pi05_turn_on_the_radio",
    )
    parser.add_argument(
        "--controller-python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter containing the EMERGE runtime dependencies.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir("turning_on_radio_action_queue"),
    )
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--action-steps",
        type=int,
        default=None,
        help="Total VLA action budget. Default is max_steps + 2.",
    )
    parser.add_argument(
        "--chunk-steps",
        type=int,
        default=None,
        help=(
            "Split the action budget into several queued actions of this size. "
            "Each chunk is verified against the official evaluator status before "
            "the next one is queued, so a pause between chunks can be observed."
        ),
    )
    parser.add_argument(
        "--chunk-pause",
        type=float,
        default=2.0,
        help="Pause between chunks while asserting the simulator is not stepping.",
    )
    parser.add_argument("--upstream-port", type=int, default=8001)
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--episode-timeout", type=float, default=1200.0)
    parser.add_argument(
        "--action-timeout",
        type=float,
        default=None,
        help="Per-action execution limit. Default is derived from the action budget.",
    )
    parser.add_argument("--state-sync-timeout", type=float, default=120.0)
    parser.add_argument(
        "--write-video",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Workspace and driver configuration
# --------------------------------------------------------------------------- #


def _register_plugin(controller_python: Path, plugin_home: Path) -> dict[str, str]:
    """Register this integration into an output-local plugin registry."""
    environment = dict(os.environ)
    environment["Emerge_PLUGIN_HOME"] = str(plugin_home)
    command = [
        str(controller_python),
        "-c",
        (
            "import sys; from robot.plugins import register_plugin; "
            "spec=register_plugin(sys.argv[1]); print(spec.driver_name)"
        ),
        str(PLUGIN_ROOT),
    ]
    subprocess.run(command, cwd=EMERGE_ROOT, env=environment, check=True)
    return environment


def _execution_timeout(action_steps: int) -> float:
    """Generous per-action backstop, dominated by the episode timeout anyway."""
    return max(600.0, min(int(action_steps), 800) * 20.0)


def _write_driver_config(
    path: Path,
    *,
    workspace: Path,
    upstream_port: int,
    gateway_port: int,
    max_action_steps: int,
    status_path: Path,
    execution_timeout_s: float | None = None,
    state_sync_timeout_s: float = 120.0,
) -> dict[str, Any]:
    payload = {
        "workspace": str(workspace),
        "upstream_url": f"ws://127.0.0.1:{upstream_port}",
        "listen_host": "127.0.0.1",
        "listen_port": gateway_port,
        "upstream_open_timeout_s": 10.0,
        "required_instruction": OFFICIAL_PROMPT,
        "max_action_steps": max_action_steps,
        "status_path": str(status_path),
        "observation_publish_interval": 5,
        "execution_timeout_s": (
            execution_timeout_s if execution_timeout_s is not None else _execution_timeout(max_action_steps)
        ),
        "progress_timeout_s": 300.0,
        "state_sync_timeout_s": state_sync_timeout_s,
        "expected_task": TASK_NAME,
        "expected_instance_id": None,
    }
    atomic_write_json(path, payload)
    return payload


# --------------------------------------------------------------------------- #
# Action queue helpers
# --------------------------------------------------------------------------- #


def _enqueue_vla_action(action_file: Path, *, steps: int) -> str:
    document = update_action_document(
        action_file,
        lambda current: append_action(
            current,
            action_type="vla_execute",
            parameters={"instruction": OFFICIAL_PROMPT, "step": steps},
        ),
    )
    return str(document["actions"][-1]["id"])


def _wait_for_action_terminal(
    action_file: Path,
    action_id: str,
    *,
    controller: subprocess.Popen[bytes],
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        if controller.poll() is not None:
            raise RuntimeError(f"Controller exited early with code {controller.returncode}")
        try:
            payload = parse_action_markdown(action_file.read_text(encoding="utf-8"))
            document = normalize_action_document(payload) if payload else None
        except FileNotFoundError:
            document = None
        if document is not None:
            for action in document["actions"]:
                if action["id"] == action_id:
                    last = dict(action)
                    if action["status"] in {"completed", "failed", "cancelled"}:
                        return last
        time.sleep(0.1)
    raise TimeoutError(
        f"action {action_id} did not reach a terminal state within {timeout_s:.0f}s "
        f"(last={last})"
    )


def _evaluator_steps(workspace: Path) -> int | None:
    value = read_json(workspace / "BEHAVIOR_STATUS.json").get("steps")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _assert_simulator_paused(
    workspace: Path,
    *,
    controller: subprocess.Popen[bytes],
    evaluator: subprocess.Popen[bytes],
    pause_s: float,
) -> None:
    """Between chunks no action is queued, so the evaluator must not advance."""
    before = _evaluator_steps(workspace)
    time.sleep(max(0.0, pause_s))
    for process, label in ((controller, "Controller"), (evaluator, "Evaluator")):
        if process.poll() is not None:
            raise RuntimeError(f"{label} exited with code {process.returncode} during the chunk pause")
    after = _evaluator_steps(workspace)
    if before is not None and after is not None and after != before:
        raise RuntimeError(
            f"the evaluator advanced from {before} to {after} steps while no action was queued"
        )


# --------------------------------------------------------------------------- #
# Readiness
# --------------------------------------------------------------------------- #


def _wait_for_evaluator_connection(
    workspace: Path,
    *,
    controller: subprocess.Popen[bytes],
    evaluator: subprocess.Popen[bytes],
    timeout_s: float,
) -> None:
    wait_for_condition(
        lambda: bool(robot_state_section(workspace).get("evaluator_connected")),
        timeout_s=timeout_s,
        label="the BEHAVIOR evaluator to connect to the controlled gateway",
        processes=[controller, evaluator],
    )


def _wait_for_ready(
    workspace: Path,
    *,
    controller: subprocess.Popen[bytes],
    evaluator: subprocess.Popen[bytes],
    timeout_s: float,
) -> None:
    """Require the upstream, the first status frame and the first observation."""
    _wait_for_evaluator_connection(
        workspace, controller=controller, evaluator=evaluator, timeout_s=timeout_s
    )
    wait_for_condition(
        lambda: bool(robot_state_section(workspace).get("upstream_connected")),
        timeout_s=timeout_s,
        label="the OpenPI upstream to connect",
        processes=[controller, evaluator],
    )
    wait_for_condition(
        lambda: bool(read_json(workspace / "BEHAVIOR_STATUS.json")),
        timeout_s=timeout_s,
        label="the first official evaluator status frame",
        processes=[controller, evaluator],
    )
    wait_for_condition(
        lambda: (workspace / "artifacts/observations/observation.json").is_file(),
        timeout_s=timeout_s,
        label="the first published camera observation",
        processes=[controller, evaluator],
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    action_steps = args.action_steps if args.action_steps is not None else args.max_steps + 2
    if action_steps < 1:
        raise ValueError("--action-steps must be positive")
    if args.chunk_steps is not None and args.chunk_steps < 1:
        raise ValueError("--chunk-steps must be positive")
    if args.chunk_steps is None and action_steps < args.max_steps + 1:
        raise ValueError(
            "--action-steps must be at least --max-steps + 1 for a single-action run"
        )

    openpi_root = require_path(args.openpi_root, "OpenPI root")
    behavior_root = require_path(args.behavior_root, "BEHAVIOR root")
    checkpoint = require_path(args.checkpoint, "π0.5 checkpoint")
    openpi_python = require_executable(openpi_root / ".venv/bin/python", "OpenPI Python")
    controller_python = require_executable(args.controller_python, "EMERGE Controller Python")
    behavior_python = require_executable(
        WORKSPACE_ROOT / "envs/behavior/bin/python", "BEHAVIOR Python"
    )
    robot_config = require_path(
        behavior_root / "OmniGibson/omnigibson/eval/r1pro_openpi.yaml",
        "R1Pro evaluation config",
    )

    output_dir = assert_fresh_output_dir(args.output_dir)
    logs_dir = output_dir / "logs"
    workspace = output_dir / "workspace"
    status_path = workspace / "BEHAVIOR_STATUS.json"
    plugin_home = output_dir / ".emerge_plugins"
    logs_dir.mkdir(parents=True, exist_ok=True)
    assert_port_available(args.upstream_port, label="OpenPI server")
    assert_port_available(args.gateway_port, label="EMERGE gateway")

    prepare_workspace(workspace, task_name=TASK_NAME, instruction=OFFICIAL_PROMPT)
    install_workspace_skill(workspace)
    controller_environment = _register_plugin(controller_python, plugin_home)
    driver_config = output_dir / "driver_config.json"
    _write_driver_config(
        driver_config,
        workspace=workspace,
        upstream_port=args.upstream_port,
        gateway_port=args.gateway_port,
        max_action_steps=action_steps,
        status_path=status_path,
        execution_timeout_s=args.action_timeout,
        state_sync_timeout_s=args.state_sync_timeout,
    )

    processes: list = []
    executed_steps = 0
    verdict = "incomplete"
    evaluator_code: int | None = None
    try:
        with (
            (logs_dir / "openpi.log").open("wb") as openpi_log,
            (logs_dir / "controller.log").open("wb") as controller_log,
            (logs_dir / "evaluator.log").open("wb") as evaluator_log,
        ):
            openpi_command = [
                str(openpi_python),
                "scripts/b1k/serve_b1k.py",
                "--robot",
                "b1k/R1Pro",
                "--task",
                "b1k/turning_on_radio",
                "--repo-id",
                "turning_on_radio",
                "--policy.config",
                "pi05_b1k",
                "--policy.dir",
                str(checkpoint),
                "--port",
                str(args.upstream_port),
            ]
            print(f"Starting OpenPI; log: {logs_dir / 'openpi.log'}", flush=True)
            openpi_process = spawn_process(openpi_command, cwd=openpi_root, log=openpi_log)
            processes.append(openpi_process)
            wait_for_health(
                args.upstream_port, openpi_process, args.startup_timeout, label="OpenPI server"
            )

            controller_command = [
                str(controller_python),
                "-m",
                "robot.controller",
                "--driver",
                "behavior_b1k_pi05",
                "--workspace",
                str(workspace),
                "--driver-config",
                str(driver_config),
                "--interval",
                "0.05",
            ]
            print(f"Starting EMERGE Controller; log: {logs_dir / 'controller.log'}", flush=True)
            controller_process = spawn_process(
                controller_command,
                cwd=EMERGE_ROOT,
                log=controller_log,
                env=controller_environment,
            )
            processes.append(controller_process)
            wait_for_health(args.gateway_port, controller_process, 30.0, label="EMERGE Controller")

            evaluator_command = [
                str(behavior_python),
                "-m",
                "omnigibson.eval.eval",
                "--task-name",
                "turning_on_radio",
                "--robot-config",
                str(robot_config),
                "--mode",
                "public_test",
                "--host",
                "127.0.0.1",
                "--port",
                str(args.gateway_port),
                "--instance-indices",
                str(args.instance_index),
                "--max-steps",
                str(args.max_steps),
                "--output-dir",
                str(output_dir),
                "--status-file",
                str(status_path),
                "--write-video" if args.write_video else "--no-write-video",
            ]
            print(f"Starting BEHAVIOR evaluator; log: {logs_dir / 'evaluator.log'}", flush=True)
            evaluator_process = spawn_process(
                evaluator_command,
                cwd=behavior_root / "OmniGibson",
                log=evaluator_log,
                env=behavior_environment(),
            )
            processes.append(evaluator_process)

            _wait_for_ready(
                workspace,
                controller=controller_process,
                evaluator=evaluator_process,
                timeout_s=args.startup_timeout,
            )

            deadline = time.monotonic() + args.episode_timeout
            while True:
                status = read_json(status_path)
                if status.get("done"):
                    break
                remaining = action_steps - executed_steps
                if remaining <= 0:
                    break
                chunk = remaining if args.chunk_steps is None else min(args.chunk_steps, remaining)
                if chunk < 1:
                    break
                action_id = _enqueue_vla_action(workspace / "ACTION.md", steps=chunk)
                print(f"Enqueued vla_execute action {action_id} for {chunk} step(s)", flush=True)
                remaining_timeout = max(30.0, deadline - time.monotonic())
                action = _wait_for_action_terminal(
                    workspace / "ACTION.md",
                    action_id,
                    controller=controller_process,
                    timeout_s=remaining_timeout,
                )
                if action["status"] != "completed":
                    raise RuntimeError(
                        f"EMERGE action ended as {action['status']}: {action.get('result')}"
                    )
                executed_steps += chunk
                print(
                    f"ACTION.md result: {action.get('result')} "
                    f"(evaluator steps={_evaluator_steps(workspace)})",
                    flush=True,
                )
                if args.chunk_steps is not None:
                    _assert_simulator_paused(
                        workspace,
                        controller=controller_process,
                        evaluator=evaluator_process,
                        pause_s=args.chunk_pause,
                    )
                if read_json(status_path).get("done"):
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"episode did not finish within {args.episode_timeout:.0f}s"
                    )

            evaluator_code = wait_for_process(
                evaluator_process, args.episode_timeout, "BEHAVIOR evaluator"
            )
            if evaluator_code != 0:
                raise RuntimeError(
                    f"BEHAVIOR evaluator exited with code {evaluator_code}; "
                    f"see {logs_dir / 'evaluator.log'}"
                )

            final = read_json(status_path)
            if final.get("phase") == "error":
                verdict = "evaluator_error"
            elif not final.get("done"):
                verdict = "incomplete"
            elif final.get("success"):
                verdict = "success"
            else:
                verdict = "task_failed"
    finally:
        for process in reversed(processes):
            stop_process_group(process)

    result_files = sorted((output_dir / "json").glob("turning_on_radio_*.json"))
    summary = {
        "schema_version": "Emerge.behavior_smoke_result.v1",
        "task": TASK_NAME,
        "instance_index": args.instance_index,
        "requested_action_steps": action_steps,
        "executed_action_steps": executed_steps,
        "chunk_steps": args.chunk_steps,
        "verdict": verdict,
        "evaluator_returncode": evaluator_code,
        "evaluator_status": read_json(status_path),
        "result_files": [str(path) for path in result_files],
        "workspace": str(workspace),
    }
    atomic_write_json(output_dir / "action_queue_result.json", summary)

    if not result_files:
        raise FileNotFoundError(
            f"BEHAVIOR result JSON was not written under {output_dir / 'json'}"
        )
    print(
        f"Action-queue smoke finished: verdict={verdict} "
        f"steps={executed_steps}; result: {result_files[-1]}",
        flush=True,
    )
    return 0 if verdict in {"success", "task_failed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

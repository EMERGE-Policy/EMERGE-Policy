#!/usr/bin/env python3
"""Run one BEHAVIOR turning_on_radio episode through the EMERGE gateway.

This is the policy-transparency smoke path: the official BEHAVIOR evaluator
talks to a standalone gateway that relays frames to the official OpenPI π0.5
server.  EMERGE's Agent and Controller are not involved, so this run proves the
transport, not the framework integration.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    from .runner_common import (
        EMERGE_ROOT,
        PLUGIN_ROOT,
        WORKSPACE_ROOT,
        assert_fresh_output_dir,
        assert_port_available,
        behavior_environment,
        default_output_dir,
        require_executable,
        require_path,
        spawn_process,
        stop_process_group,
        wait_for_health,
        wait_for_process,
    )
except ImportError:  # Direct script execution adds this directory to sys.path.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from runner_common import (  # type: ignore[no-redef]
        EMERGE_ROOT,
        PLUGIN_ROOT,
        WORKSPACE_ROOT,
        assert_fresh_output_dir,
        assert_port_available,
        behavior_environment,
        default_output_dir,
        require_executable,
        require_path,
        spawn_process,
        stop_process_group,
        wait_for_health,
        wait_for_process,
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
        "--output-dir",
        type=Path,
        default=default_output_dir("turning_on_radio_smoke"),
    )
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--upstream-port", type=int, default=8001)
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument(
        "--episode-timeout",
        type=float,
        default=1800.0,
        help="Wall-clock limit for the evaluator process. Default: 1800 s.",
    )
    parser.add_argument(
        "--write-video",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")

    openpi_root = require_path(args.openpi_root, "OpenPI root")
    behavior_root = require_path(args.behavior_root, "BEHAVIOR root")
    checkpoint = require_path(args.checkpoint, "π0.5 checkpoint")
    openpi_python = require_executable(openpi_root / ".venv/bin/python", "OpenPI Python")
    behavior_python = require_executable(
        WORKSPACE_ROOT / "envs/behavior/bin/python", "BEHAVIOR Python"
    )
    robot_config = require_path(
        behavior_root / "OmniGibson/omnigibson/eval/r1pro_openpi.yaml",
        "R1Pro evaluation config",
    )

    output_dir = assert_fresh_output_dir(args.output_dir)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    openpi_log_path = logs_dir / "openpi.log"
    gateway_log_path = logs_dir / "gateway.log"
    evaluator_log_path = logs_dir / "evaluator.log"
    assert_port_available(args.upstream_port, label="OpenPI server")
    assert_port_available(args.gateway_port, label="EMERGE gateway")

    processes: list = []
    started = time.monotonic()
    try:
        with (
            open(openpi_log_path, "wb") as openpi_log,
            open(gateway_log_path, "wb") as gateway_log,
            open(evaluator_log_path, "wb") as evaluator_log,
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
            print(f"Starting OpenPI; log: {openpi_log_path}", flush=True)
            openpi_process = spawn_process(openpi_command, cwd=openpi_root, log=openpi_log)
            processes.append(openpi_process)
            wait_for_health(
                args.upstream_port,
                openpi_process,
                args.startup_timeout,
                label="OpenPI server",
            )

            gateway_command = [
                str(openpi_python),
                str(PLUGIN_ROOT / "gateway.py"),
                "--upstream-url",
                f"ws://127.0.0.1:{args.upstream_port}",
                "--listen-host",
                "127.0.0.1",
                "--listen-port",
                str(args.gateway_port),
            ]
            print(f"Starting EMERGE gateway; log: {gateway_log_path}", flush=True)
            gateway_process = spawn_process(gateway_command, cwd=EMERGE_ROOT, log=gateway_log)
            processes.append(gateway_process)
            wait_for_health(args.gateway_port, gateway_process, 30.0, label="EMERGE gateway")

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
            ]
            evaluator_command.append("--write-video" if args.write_video else "--no-write-video")
            print(f"Starting BEHAVIOR evaluator; log: {evaluator_log_path}", flush=True)
            evaluator_process = spawn_process(
                evaluator_command,
                cwd=behavior_root / "OmniGibson",
                log=evaluator_log,
                env=behavior_environment(),
            )
            processes.append(evaluator_process)
            return_code = wait_for_process(
                evaluator_process, args.episode_timeout, "BEHAVIOR evaluator"
            )
            if return_code != 0:
                raise RuntimeError(
                    f"BEHAVIOR evaluator exited with code {return_code}; see {evaluator_log_path}"
                )
    finally:
        for process in reversed(processes):
            stop_process_group(process)

    results = sorted((output_dir / "json").glob("turning_on_radio_*.json"))
    if not results:
        raise FileNotFoundError(
            f"BEHAVIOR result JSON was not written under {output_dir / 'json'}"
        )
    print(
        f"Evaluation completed in {time.monotonic() - started:.1f}s; "
        f"result: {results[-1]}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

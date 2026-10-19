#!/usr/bin/env python3
"""Fixed LightNav navigation -> champion VLA handoff ablation for one episode.

This deliberately is not an autonomous EMERGE Agent run: the runner fixes the
handoff sequence so navigation and policy failures can be attributed separately.
The stock BEHAVIOR evaluator remains the sole source of task success.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EMERGE_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = EMERGE_ROOT.parent
if str(EMERGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EMERGE_ROOT))

from integrations.lightnav_behavior.arrival import verify_radio_handoff  # noqa: E402
from integrations.lightnav_sonic_emerge.bridge.lightnav_ws import LightNavWsClient  # noqa: E402
from robot.drivers.behavior_r1pro_driver import BehaviorR1ProDriver  # noqa: E402
from scripts.Behavior_eval.eval_behavior_agent import (  # noqa: E402
    _evaluator_command,
    _load_driver_config,
    _read_official_result,
    _write_completed_evaluation_status,
    _write_evaluator_error_status,
    _write_running_evaluation_status,
)
from scripts.Behavior_eval.runner_common import (  # noqa: E402
    assert_port_available,
    atomic_write_json,
    behavior_environment,
    read_json,
    require_executable,
    require_path,
    service_health_ok,
    spawn_process,
    stop_process_group,
    wait_for_condition,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="turning_on_radio", choices=("turning_on_radio",))
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--upstream-port", type=int, default=8011)
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--lightnav-server", default="ws://127.0.0.1:8051")
    parser.add_argument("--sam3-server", default="ws://127.0.0.1:8002")
    parser.add_argument("--driver-config", type=Path, default=EMERGE_ROOT / "dev/behavior_agent_eval.json")
    parser.add_argument("--robot-config", type=Path, default=EMERGE_ROOT / "dev/behavior_r1pro_openpi.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--nav-max-steps", type=int, default=900)
    parser.add_argument("--nav-timeout", type=float, default=1800)
    parser.add_argument("--max-normalized-speed", type=float, default=0.6)
    parser.add_argument("--handoff-distance-m", type=float, default=1.0)
    parser.add_argument("--vla-action-budget", type=int, default=4000)
    parser.add_argument("--vla-max-calls", type=int, default=2)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--episode-timeout", type=float, default=7200)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def preflight(args: argparse.Namespace) -> tuple[Path, Path, Path, dict[str, Any]]:
    if not 0 <= args.instance_index <= 9:
        raise ValueError("public_test instance-index must be 0-9")
    if args.gateway_port == args.upstream_port:
        raise ValueError("gateway-port and upstream-port must differ")
    if min(args.nav_max_steps, args.vla_action_budget, args.vla_max_calls) < 1:
        raise ValueError("navigation and VLA budgets must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max-steps must be positive")
    if not 0 < args.max_normalized_speed <= 1 or not 0 < args.handoff_distance_m:
        raise ValueError("speed and handoff distance must be positive")
    if min(args.nav_timeout, args.startup_timeout, args.episode_timeout) <= 0:
        raise ValueError("timeouts must be positive")
    behavior_root = require_path(EMERGE_ROOT / "third_party/behavior-1k", "BEHAVIOR source")
    behavior_python = require_executable(WORKSPACE_ROOT / "envs/behavior/bin/python", "BEHAVIOR Python")
    robot_config = require_path(args.robot_config, "R1Pro robot config")
    driver_config = _load_driver_config(args.driver_config)
    if driver_config.get("expected_task") != args.task_name:
        raise ValueError("driver config is not for the requested task")
    if not service_health_ok(f"http://127.0.0.1:{args.upstream_port}"):
        raise RuntimeError("champion service is not healthy")
    if not service_health_ok(args.sam3_server):
        raise RuntimeError("SAM3 service is not healthy")
    with LightNavWsClient(args.lightnav_server, client_id="emerge-behavior-active-preflight", request_timeout_s=15):
        pass
    assert_port_available(args.gateway_port, label="controlled BEHAVIOR gateway")
    return behavior_root, behavior_python, robot_config, driver_config


def _render_navigation_video(frame_dir: Path, output: Path) -> str | None:
    """Render a reviewable navigation clip even if no official video closes."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not (frame_dir / "frame_00000.jpg").is_file():
        return None
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", "4",
         "-i", str(frame_dir / "frame_%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         str(output)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    return str(output) if result.returncode == 0 and output.is_file() else None


def run(args: argparse.Namespace) -> int:
    behavior_root, behavior_python, robot_config, base_config = preflight(args)
    if args.preflight_only:
        print("Preflight OK: champion, LightNav, SAM3 and controlled gateway port are ready.", flush=True)
        return 0
    instance_id = 301 + args.instance_index
    output = (args.output_dir or WORKSPACE_ROOT / "outputs/Emerge" / args.task_name /
              f"lightnav_handoff_{instance_id}").expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing experiment: {output}")
    output.mkdir(parents=True, exist_ok=True)
    workspace = output / "workspace"
    workspace.mkdir()
    logs = output / "logs"
    logs.mkdir()
    status_path = workspace / "BEHAVIOR_STATUS.json"
    summary: dict[str, Any] = {
        "experiment": "fixed_lightnav_to_champion_handoff",
        "task": args.task_name,
        "instance_id": instance_id,
        "navigation": None,
        "arrival": None,
        "vla_steps": 0,
        "official_result": None,
        "official_success": None,
        "official_steps": None,
        "error": None,
    }
    baseline_path = (WORKSPACE_ROOT / "outputs/champion_baseline" / args.task_name /
                     "json" / f"{args.task_name}_{instance_id}_0.json")
    baseline = read_json(baseline_path)
    if baseline.get("task") == args.task_name and baseline.get("instance_id") == instance_id:
        summary["baseline"] = {
            "result_path": str(baseline_path), "success": baseline.get("success"),
            "steps": baseline.get("steps"), "q_score": baseline.get("q_score"),
        }
    evaluator = None
    driver = None
    began = time.monotonic()
    try:
        config = dict(base_config)
        config.update({
            "workspace": str(workspace),
            "upstream_url": f"ws://127.0.0.1:{args.upstream_port}",
            "listen_port": args.gateway_port,
            "status_path": str(status_path),
            "max_action_steps": max(args.nav_max_steps, args.vla_action_budget),
            "execution_timeout_s": max(args.episode_timeout, args.nav_timeout + 120),
            "expected_instance_id": instance_id,
            # Navigation forces one coherent RGB-D snapshot per 7/8 action steps;
            # avoid serializing three full-resolution views on every action.
            "observation_publish_interval": 30,
        })
        atomic_write_json(output / "driver_config.json", config)
        driver = BehaviorR1ProDriver(**config)
        driver.load_scene({})
        with (logs / "evaluator.log").open("wb") as evaluator_log:
            evaluator = spawn_process(_evaluator_command(
                behavior_python=behavior_python, robot_config=robot_config,
                gateway_port=args.gateway_port, task_name=args.task_name,
                instance_index=args.instance_index, max_steps=args.max_steps,
                output_dir=output, write_video=args.write_video, headless=args.headless,
            ), cwd=behavior_root / "OmniGibson", log=evaluator_log,
                env=behavior_environment())
            print(f"Evaluator started: {output}", flush=True)
            wait_for_condition(
                lambda: all(driver.get_runtime_state()["robots"][driver.ROBOT_ID].get(key)
                            for key in ("evaluator_connected", "upstream_connected")),
                timeout_s=args.startup_timeout, label="BEHAVIOR evaluator and champion connections",
                processes=[evaluator],
            )
            _write_running_evaluation_status(status_path, task_name=args.task_name)
            nav_result_text = driver.execute_action("navigate_vln", {
                "instruction": "Go to the radio receiver on the table in the living room and stop nearby.",
                "max_steps": args.nav_max_steps,
                "max_wall_time_s": args.nav_timeout,
                "max_normalized_speed": args.max_normalized_speed,
                "server_url": args.lightnav_server,
                "trace_path": str(workspace / "lightnav_trace.jsonl"),
            })
            summary["navigation"] = driver.last_navigation_result
            print(nav_result_text, flush=True)
            if (summary["navigation"] or {}).get("finish_reason") != "model_stop_candidate":
                summary["error"] = "navigation did not produce a stop candidate; champion was not given control"
                return 2
            arrival = asyncio.run(verify_radio_handoff(
                workspace, sam3_url=args.sam3_server,
                max_horizontal_distance_m=args.handoff_distance_m,
                overlay_path=output / "arrival_radio_overlay.png",
            ))
            summary["arrival"] = arrival
            print("Arrival check: " + json.dumps(arrival, ensure_ascii=False), flush=True)
            if not arrival["verified"]:
                summary["error"] = "radio was not independently verified in the handoff region"
                return 2
            print("Radio proximity verified; handing control to champion VLA.", flush=True)
            nav_steps = int(summary["navigation"]["nav_steps"])
            for call_index in range(args.vla_max_calls):
                if time.monotonic() - began >= args.episode_timeout:
                    raise TimeoutError("episode wall-time budget exceeded")
                if evaluator.poll() is not None:
                    break
                result = driver.execute_action("vla_execute", {
                    "instruction": config["required_instruction"],
                    "step": args.vla_action_budget,
                })
                print(f"Champion call {call_index + 1}: {result}", flush=True)
                progress = driver.get_runtime_state()["robots"][driver.ROBOT_ID]["runtime_progress"]
                summary["vla_steps"] = max(0, progress["gateway_action_steps"] - nav_steps)
                if result.startswith("Failed:") and evaluator.poll() is None:
                    state = driver.get_runtime_state()["robots"][driver.ROBOT_ID]
                    if not state.get("evaluator_connected"):
                        # The final action can close the websocket before the
                        # evaluator writes its authoritative result and video.
                        try:
                            evaluator.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            pass
                        if evaluator.poll() is not None:
                            break
                    summary["error"] = result
                    return 2
            if evaluator.poll() is None:
                try:
                    evaluator.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    summary["error"] = "VLA call budget exhausted before official episode ended"
                    return 2
            if evaluator.returncode != 0:
                summary["error"] = f"official evaluator exited with code {evaluator.returncode}"
                _write_evaluator_error_status(status_path, summary["error"], task_name=args.task_name)
                return 2
            result_path, official = _read_official_result(output, task_name=args.task_name)
            _write_completed_evaluation_status(output, status_path, task_name=args.task_name)
            summary.update({"official_result": str(result_path), "official_success": official["success"],
                            "official_steps": official["steps"], "q_score": official.get("q_score")})
            summary["unattributed_official_steps"] = (
                int(official["steps"]) - nav_steps - int(summary["vla_steps"])
            )
            summary["navigation_step_fraction"] = (
                nav_steps / int(official["steps"]) if int(official["steps"]) else None
            )
            if isinstance((summary.get("baseline") or {}).get("steps"), int):
                summary["comparison"] = {
                    "hybrid_minus_baseline_steps": int(official["steps"]) - int(summary["baseline"]["steps"]),
                    "baseline_success": summary["baseline"]["success"],
                    "hybrid_success": bool(official["success"]),
                }
            return 0 if official["success"] or not args.require_success else 1
    except KeyboardInterrupt:
        summary["error"] = "interrupted_by_user"
        print("Interrupted; stopping the controlled evaluator.", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        print(f"Experiment failed: {summary['error']}", file=sys.stderr, flush=True)
        if status_path.is_file():
            _write_evaluator_error_status(status_path, summary["error"], task_name=args.task_name)
        return 2
    finally:
        stop_process_group(evaluator)
        if driver is not None:
            driver.close()
        try:
            summary["navigation_video"] = _render_navigation_video(
                workspace / "lightnav_frames", output / "navigation.mp4"
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            summary["navigation_video_error"] = f"{type(exc).__name__}: {exc}"
        summary["evaluator_exit_code"] = evaluator.poll() if evaluator is not None else None
        summary["elapsed_wall_time_s"] = round(time.monotonic() - began, 3)
        atomic_write_json(output / "lightnav_handoff_summary.json", summary)
        print("Experiment summary: " + str(output / "lightnav_handoff_summary.json"), flush=True)


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))

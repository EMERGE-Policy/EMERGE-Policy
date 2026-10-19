#!/usr/bin/env python3
"""Run one BEHAVIOR-1K episode with the real EMERGE Agent.

The actions are chosen by the LLM Agent through ``execute_robot_action``. The
runner owns the episode workspace and supervises OpenPI, the native EMERGE
Controller, the BEHAVIOR evaluator, and the Agent process.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
EMERGE_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = EMERGE_ROOT.parent
if str(EMERGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EMERGE_ROOT))

from Emerge.runtime.configuration import load_runtime_config  # noqa: E402
from Emerge.runtime.protocol import RunRequest, RunResult  # noqa: E402
from Emerge.utils.action_queue import (  # noqa: E402
    normalize_action_document,
    parse_action_markdown,
)
from Emerge.utils.helpers import sync_workspace_templates  # noqa: E402
from robot.behavior.contracts import (  # noqa: E402
    EVALUATION_STATUS_SCHEMA,
    agent_exit_code,
    classify_episode,
    episode_terminated,
    validate_driver_config,
)
from scripts.Behavior_eval.runner_common import (  # noqa: E402
    assert_fresh_output_dir,
    assert_port_available,
    atomic_write_json,
    behavior_environment,
    prepare_workspace,
    read_json,
    require_executable,
    require_path,
    robot_state_section,
    service_health_ok,
    spawn_process,
    stop_process_group,
    wait_for_condition,
    wait_for_health,
)

DEFAULT_DRIVER_CONFIG = EMERGE_ROOT / "dev/behavior_agent_eval.json"
DEFAULT_ROBOT_CONFIG = EMERGE_ROOT / "dev/behavior_r1pro_openpi.yaml"
DEFAULT_GATEWAY_PORT = 8000
DEFAULT_OPENPI_PORT = 8011
BEHAVIOR_RGBD_WRAPPER = "omnigibson.eval.wrappers.RGBDFullResWrapper"
NAVIGATION_PROFILE = EMERGE_ROOT / "robot/profiles/behavior_r1pro_navigation.md"
NAVIGATION_SKILL = EMERGE_ROOT / "Emerge/skills/lightnav/SKILL.md"

#: Agent finish reasons that indicate the framework, not the task, is broken.
INFRASTRUCTURE_FINISH_REASONS = frozenset(
    {
        "cancellation_unconfirmed",
        "error",
        "unfinished_action",
        "cleanup_error",
        "runtime_error",
        "invalid_request",
        "invalid_result",
    }
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior-root", type=Path, default=EMERGE_ROOT / "third_party/behavior-1k")
    parser.add_argument(
        "--agent-python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter containing EMERGE's Agent and Controller dependencies.",
    )
    parser.add_argument(
        "--config",
        "--agent-config",
        dest="agent_config",
        type=Path,
        help="EMERGE runtime/provider config (same meaning as `Emerge agent --config`).",
    )
    parser.add_argument("--agent-model")
    parser.add_argument(
        "--driver-config",
        type=Path,
        default=DEFAULT_DRIVER_CONFIG,
        help="Base configuration for the native behavior_r1pro driver.",
    )
    parser.add_argument(
        "--task-name",
        help="Override expected_task from the driver config.",
    )
    parser.add_argument(
        "--task-instruction",
        help="Override required_instruction from the driver config.",
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=DEFAULT_ROBOT_CONFIG,
        help=(
            "R1Pro evaluator config whose runtime robot name matches the public OpenPI "
            "checkpoint observation keys."
        ),
    )
    parser.add_argument(
        "--agent-max-iterations",
        type=int,
        default=None,
        help=(
            "Override EMERGE's configured maximum tool iterations for each Agent run. "
            "By default the native EMERGE configuration is used."
        ),
    )
    parser.add_argument(
        "--max-agent-feedback-rounds",
        type=int,
        default=2,
        help=(
            "How many times a cleanly completed Agent response may receive official "
            "evaluator feedback while the episode is still running."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Episode artifact directory. Default: "
            f"{WORKSPACE_ROOT / 'outputs/Emerge'}/<task>."
        ),
    )
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument(
        "--navigation-only", action="store_true",
        help="Use the normal EMERGE loop for VLN navigation and bounded body-posture observation actions.",
    )
    parser.add_argument(
        "--navigation-goal",
        help="Navigation objective shown to the Agent; required with --navigation-only.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Optional BEHAVIOR simulator-step override. When omitted, the official "
            "evaluator uses the task's configured limit."
        ),
    )
    parser.add_argument(
        "--upstream-port",
        type=int,
        default=DEFAULT_OPENPI_PORT,
        help=(
            "OpenPI policy server port. The default is kept separate from "
            "VGGT (8001) and SAM3 (8002)."
        ),
    )
    parser.add_argument("--gateway-port", type=int, default=DEFAULT_GATEWAY_PORT)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument(
        "--episode-timeout",
        type=float,
        default=1800.0,
        help=(
            "Wall-clock seconds available to the complete episode after the Agent starts, "
            "including reasoning and robot actions. Independent of --max-steps."
        ),
    )
    parser.add_argument("--poll-interval", type=float, default=0.2)
    parser.add_argument(
        "--state-sync-timeout",
        type=float,
        default=120.0,
        help="How long the driver waits for the simulator to confirm applied actions.",
    )
    parser.add_argument(
        "--require-success",
        action="store_true",
        help=(
            "Exit non-zero unless the official evaluator reports success. "
            "Without this flag a cleanly terminated but unsuccessful episode exits 0."
        ),
    )
    parser.add_argument(
        "--write-video",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Isaac Sim without a window. Use --no-headless to open the simulator UI.",
    )
    parser.add_argument(
        "--show-agent-events",
        action="store_true",
        help="Print Agent messages, tool calls, and action state changes while the run is active.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Validate environments, provider, external model services, ports, "
            "and configs, then exit without creating an episode or calling the LLM."
        ),
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Workspace
# --------------------------------------------------------------------------- #


def _prepare_workspace(workspace: Path, *, task_name: str, instruction: str) -> Path:
    """Sync generic EMERGE templates and reset the episode queue."""
    workspace.mkdir(parents=True, exist_ok=True)
    sync_workspace_templates(workspace, silent=True)
    prepare_workspace(workspace, task_name=task_name, instruction=instruction)
    return workspace


def _load_driver_config(path: Path) -> dict[str, Any]:
    """Load the checked-in driver defaults before applying episode values."""
    resolved = require_path(path, "BEHAVIOR driver config")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid BEHAVIOR driver config: {resolved}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"BEHAVIOR driver config must be a JSON object: {resolved}")
    return payload


def _execution_timeout(action_steps: int) -> float:
    """Return the existing generous per-action timeout for a policy rollout."""
    return max(600.0, min(int(action_steps), 800) * 20.0)


def _episode_driver_config(
    base_config: dict[str, Any],
    *,
    output_dir: Path,
    workspace: Path,
    status_path: Path,
    upstream_port: int,
    gateway_port: int,
    max_action_steps: int,
    state_sync_timeout_s: float,
    task_name: str,
    task_instruction: str,
    r1pro_urdf_path: Path | None = None,
    navigation_only: bool = False,
) -> Path:
    """Write one episode-local config, following the native LIBERO runner."""
    config = copy.deepcopy(base_config)
    config.update(
        {
            "workspace": str(workspace),
            "upstream_url": f"ws://127.0.0.1:{upstream_port}",
            "listen_host": "127.0.0.1",
            "listen_port": gateway_port,
            "required_instruction": task_instruction,
            "max_action_steps": max_action_steps,
            "status_path": str(status_path),
            "execution_timeout_s": _execution_timeout(max_action_steps),
            "state_sync_timeout_s": state_sync_timeout_s,
            "expected_task": task_name,
            "expected_instance_id": None,
            "r1pro_urdf_path": str(r1pro_urdf_path) if r1pro_urdf_path else None,
            "navigation_only": navigation_only,
        }
    )
    if navigation_only:
        config["profile_path"] = str(NAVIGATION_PROFILE)
    normalized = validate_driver_config(config)
    path = output_dir / "driver_config.json"
    atomic_write_json(path, normalized)
    return path


def _write_running_evaluation_status(
    status_path: Path, *, task_name: str
) -> dict[str, Any]:
    """Publish EMERGE's live view of an official evaluator that is running.

    BEHAVIOR 3.9.3 writes its authoritative metrics only when the rollout
    finishes.  The gateway connection is the live readiness signal; this file
    gives the Agent a stable episode contract while the evaluator is blocked at
    the controlled gateway waiting for an action budget.
    """
    status = {
        "schema_version": EVALUATION_STATUS_SCHEMA,
        "source": "emerge_supervisor",
        "task": task_name,
        "rollout_id": 0,
        "phase": "running",
        "steps": None,
        "steps_source": "unavailable_until_official_result",
        "success": False,
        "done": False,
    }
    atomic_write_json(status_path, status)
    return status


def _read_official_result(
    output_dir: Path, *, task_name: str
) -> tuple[Path, dict[str, Any]]:
    """Read the single result emitted by the official BEHAVIOR evaluator."""
    result_paths = sorted((output_dir / "json").glob(f"{task_name}_*_0.json"))
    if len(result_paths) != 1:
        raise RuntimeError(
            "expected exactly one official BEHAVIOR result for the single-instance run, "
            f"found {len(result_paths)} under {output_dir / 'json'}"
        )
    result_path = result_paths[0].resolve()
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"official BEHAVIOR result is unreadable: {result_path}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"official BEHAVIOR result must be a JSON object: {result_path}")
    required = {"task", "instance_id", "rollout_id", "steps", "success"}
    missing = sorted(required.difference(result))
    if missing:
        raise RuntimeError(
            f"official BEHAVIOR result is missing {', '.join(missing)}: {result_path}"
        )
    if str(result["task"]) != task_name or int(result["rollout_id"]) != 0:
        raise RuntimeError(
            f"official BEHAVIOR result does not match this run: {result_path}"
        )
    if not isinstance(result["success"], bool):
        raise RuntimeError(f"official BEHAVIOR success must be a boolean: {result_path}")
    return result_path, result


def _write_completed_evaluation_status(
    output_dir: Path,
    status_path: Path,
    *,
    task_name: str,
) -> dict[str, Any]:
    """Translate the official final result into the Agent's episode contract."""
    result_path, result = _read_official_result(output_dir, task_name=task_name)
    q_score = result.get("q_score")
    if isinstance(q_score, dict):
        q_score = q_score.get("final")
    status = {
        "schema_version": EVALUATION_STATUS_SCHEMA,
        "source": "official_result",
        "task": task_name,
        "instance_id": int(result["instance_id"]),
        "rollout_id": int(result["rollout_id"]),
        "phase": "completed",
        "steps": int(result["steps"]),
        "success": result["success"],
        "done": True,
        "termination_reason": "official_evaluator_completed",
        "result_path": str(result_path),
        "q_score": q_score,
    }
    atomic_write_json(status_path, status)
    return status


def _write_evaluator_error_status(
    status_path: Path, detail: str, *, task_name: str
) -> dict[str, Any]:
    status = {
        "schema_version": EVALUATION_STATUS_SCHEMA,
        "source": "emerge_supervisor",
        "task": task_name,
        "rollout_id": 0,
        "phase": "error",
        "success": False,
        "done": True,
        "termination_reason": "evaluator_error",
        "error": str(detail),
    }
    atomic_write_json(status_path, status)
    return status


def _evaluator_command(
    *,
    behavior_python: Path,
    robot_config: Path,
    gateway_port: int,
    task_name: str,
    instance_index: int,
    max_steps: int | None,
    output_dir: Path,
    write_video: bool,
    headless: bool,
) -> list[str]:
    """Build the stock BEHAVIOR 3.9.3 single-environment command."""
    command = [
        str(behavior_python),
        "-m",
        "omnigibson.eval.eval",
        "--task-name",
        task_name,
        "--robot-config",
        str(robot_config),
        "--mode",
        "public_test",
        "--host",
        "127.0.0.1",
        "--port",
        str(gateway_port),
        "--instance-indices",
        str(instance_index),
        "--num-envs",
        "1",
        "--replay-action-chunk-size",
        "0",
        "--env-wrapper",
        BEHAVIOR_RGBD_WRAPPER,
        "--output-dir",
        str(output_dir),
        "--write-video" if write_video else "--no-write-video",
        "--headless" if headless else "--no-headless",
    ]
    if max_steps is not None:
        insertion = command.index("--env-wrapper")
        command[insertion:insertion] = ["--max-steps", str(max_steps)]
    return command


def _agent_message(*, task_instruction: str, navigation_goal: str | None = None) -> str:
    if navigation_goal is not None:
        return (
            "Your BEHAVIOR navigation mission is:\n\n"
            f"{navigation_goal}\n\n"
            "Act through the normal EMERGE robot action tool. Follow EMBODIED.md "
            "and the LightNav navigation skill."
        )
    return (
        "Complete this BEHAVIOR-1K simulated robot task:\n\n"
        f"{task_instruction}\n\n"
        "Act autonomously through execute_robot_action. Follow the active "
        "EMBODIED.md profile for robot-specific actions and ROBOT_STATE.md for "
        "the official evaluator status."
    )


def _feedback_message(status: dict[str, Any], *, task_instruction: str) -> str:
    return (
        "Your previous turn ended before the episode finished. The official BEHAVIOR "
        "evaluator is still running; continue the same episode.\n\n"
        f"Task: {task_instruction}\n"
        f"Current evaluator status: {json.dumps(status, ensure_ascii=False)}\n"
        "Read the latest ROBOT_STATE.md before deciding what to do next. "
        "Do not restart the rollout."
    )


# --------------------------------------------------------------------------- #
# Agent process
# --------------------------------------------------------------------------- #


def _agent_runtime_config(args: argparse.Namespace):
    request = {
        "message": "provider preflight",
        "config": (
            str(args.agent_config.expanduser().resolve())
            if args.agent_config is not None
            else None
        ),
        "model": args.agent_model,
        "visual_monitor_enabled": False if args.navigation_only else None,
    }
    return load_runtime_config(RunRequest.model_validate(request))


def _validate_agent_provider(args: argparse.Namespace, config=None) -> str:
    """Fail before loading Isaac Sim when the Agent has no usable provider.

    This preflight runs in the runner's interpreter, so it is only conclusive
    when that interpreter is the same one used for the Agent.
    """
    from Emerge.providers.factory import create_provider

    if Path(sys.executable).resolve() != Path(args.agent_python).resolve():
        print(
            f"WARNING: provider preflight runs in {sys.executable} but the Agent "
            f"will run in {args.agent_python}; a provider missing from that "
            "interpreter will only fail later.",
            file=sys.stderr,
            flush=True,
        )
    config = config or _agent_runtime_config(args)
    create_provider(config, model=args.agent_model)
    selected_model = str(args.agent_model or config.agents.defaults.model)
    if selected_model.startswith("openai-codex/"):
        import tomllib

        codex_root = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        model_cache = codex_root / "models_cache.json"
        try:
            cache = json.loads(model_cache.read_text(encoding="utf-8"))
            available = {
                str(item["slug"])
                for item in cache.get("models", [])
                if isinstance(item, dict) and item.get("slug")
            }
        except (FileNotFoundError, json.JSONDecodeError, OSError, AttributeError):
            available = set()
        try:
            codex_config = tomllib.loads((codex_root / "config.toml").read_text(encoding="utf-8"))
            configured_model = codex_config.get("model")
            if configured_model:
                available.add(str(configured_model))
        except (FileNotFoundError, OSError, tomllib.TOMLDecodeError):
            pass
        model_slug = selected_model.split("/", 1)[1]
        if available and model_slug not in available:
            raise ValueError(
                f"Codex OAuth model {model_slug!r} is not available for this account. "
                f"Cached available models: {', '.join(sorted(available))}"
            )
    return selected_model


def _validate_behavior_perception(config, *, required: bool = True) -> str:
    """Ensure this RGB-D benchmark cannot silently fall back to the wrong geometry."""
    perception = config.subagents.object_location
    if perception.geometry_backend == "vggt":
        raise ValueError(
            "BEHAVIOR publishes calibrated RGB-D; set "
            "subagents.objectLocation.geometryBackend to 'auto' or 'rgbd'"
        )
    sam3_url = str(perception.sam3_url)
    if service_health_ok(sam3_url):
        return sam3_url
    message = (
        f"SAM3 is not healthy at {sam3_url}. Start the local SAM3 service using "
        "the command in scripts/Behavior_eval/README.md"
    )
    if required:
        raise RuntimeError(message)
    print(f"Optional object-location service unavailable: {message}", flush=True)
    return sam3_url


def _validate_lightnav(*, required: bool) -> bool:
    """Navigation is an optional skill, not a prerequisite for champion VLA."""
    from integrations.lightnav_sonic_emerge.bridge.lightnav_ws import LightNavWsClient
    from websockets.exceptions import WebSocketException

    try:
        with LightNavWsClient(
            "ws://127.0.0.1:8051", client_id="emerge-behavior-navigation-preflight",
            request_timeout_s=15.0,
        ):
            pass
    except (OSError, TimeoutError, RuntimeError, WebSocketException) as exc:
        message = f"LightNav is unavailable at ws://127.0.0.1:8051: {exc}"
        if required:
            raise RuntimeError(message) from exc
        print(f"Optional navigation skill unavailable; VLA remains available. {message}", flush=True)
        return False
    print("LightNav service ready on port 8051", flush=True)
    return True


def _validate_external_openpi(port: int) -> str:
    url = f"ws://127.0.0.1:{int(port)}"
    if not service_health_ok(url):
        raise RuntimeError(
            f"OpenPI is not healthy at {url}. Start the BEHAVIOR external model "
            "services using scripts/Behavior_eval/start_behavior_services.sh"
        )
    return url


def _start_agent(
    *,
    args: argparse.Namespace,
    message: str,
    session_id: str,
    workspace: Path,
    environment: dict[str, str],
    log,
) -> subprocess.Popen[bytes]:
    run_id = uuid.uuid4().hex
    run_dir = workspace.parent / "agent_runs" / run_id
    request_path = workspace.parent / "agent_requests" / f"{run_id}.json"
    request: dict[str, Any] = {
        "schema_version": "Emerge.run_request.v1",
        "run_id": run_id,
        "message": message,
        "session_id": session_id,
        "workspace": str(workspace),
        "restrict_to_workspace": True,
        "stream": False,
    }
    if args.navigation_only:
        request["visual_monitor_enabled"] = False
    if args.agent_max_iterations is not None:
        request["max_iterations"] = args.agent_max_iterations
    if args.agent_config is not None:
        request["config"] = str(args.agent_config.expanduser().resolve())
    if args.agent_model:
        request["model"] = args.agent_model
    atomic_write_json(request_path, request)
    process = spawn_process(
        [
            str(args.agent_python),
            "-m",
            "Emerge.cli.headless",
            "--request",
            str(request_path),
            "--output-dir",
            str(run_dir),
        ],
        cwd=EMERGE_ROOT,
        log=log,
        env=environment,
    )
    process.emerge_result_path = run_dir / "result.json"
    process.emerge_run_id = run_id
    process.emerge_session_id = session_id
    return process


def _read_agent_result(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    result = RunResult.model_validate(read_json(process.emerge_result_path))
    if result.run_id != process.emerge_run_id or result.session_id != process.emerge_session_id:
        raise ValueError("Agent result does not match its request")
    return result.model_dump(mode="json")


def _drain_agent_events(process: subprocess.Popen[bytes] | None) -> None:
    """Print newly persisted, user-visible Agent events in a compact form."""
    if process is None:
        return
    events_path = Path(process.emerge_result_path).parent / "events.jsonl"
    offset = int(getattr(process, "emerge_events_offset", 0))
    try:
        with events_path.open("r", encoding="utf-8") as stream:
            stream.seek(offset)
            lines = stream.readlines()
            process.emerge_events_offset = stream.tell()
    except (FileNotFoundError, OSError):
        return

    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = str(event.get("type") or "")
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if event_type == "llm.started":
            print(
                f"[Agent] thinking: iteration={data.get('iteration')} "
                f"model={data.get('model')}",
                flush=True,
            )
        elif event_type == "assistant.message":
            text = str(data.get("text") or "").strip()
            if text:
                print(f"[Agent] {text}", flush=True)
        elif event_type == "tool.started":
            arguments = json.dumps(data.get("arguments") or {}, ensure_ascii=False)
            if len(arguments) > 600:
                arguments = arguments[:597] + "..."
            print(f"[Agent] tool: {data.get('tool')} {arguments}", flush=True)
        elif event_type in {"tool.completed", "tool.failed", "tool.cancelled"}:
            print(
                f"[Agent] {event_type}: {data.get('tool')} "
                f"duration_ms={data.get('duration_ms')}",
                flush=True,
            )
        elif event_type == "action.updated":
            print(
                f"[Agent] action: {data.get('action_type')} "
                f"status={data.get('status')} id={data.get('action_id') or data.get('id')}",
                flush=True,
            )


def _record_agent_result(process: subprocess.Popen[bytes], results: list[dict[str, Any]]) -> None:
    if any(item.get("run_id") == process.emerge_run_id for item in results):
        return
    try:
        results.append(_read_agent_result(process))
    except Exception as exc:
        results.append(
            {
                "run_id": process.emerge_run_id,
                "run_status": "failed",
                "finish_reason": "invalid_result",
                "error": str(exc),
            }
        )


def _settle_agent(
    agent: subprocess.Popen[bytes] | None,
    results: list[dict[str, Any]],
    *,
    timeout_s: float = 30.0,
) -> None:
    if agent is None:
        return
    if agent.poll() is None:
        try:
            agent.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return
    _record_agent_result(agent, results)


def _gracefully_stop_evaluator(process: subprocess.Popen[bytes] | None, *, timeout_s: float = 90.0) -> bool:
    """Let the official evaluator flush and close its PyAV MP4 writer.

    BEHAVIOR handles SIGINT by calling BatchedEvaluator.__exit__, whereas the
    generic process-group cleanup sends SIGTERM and can leave an MP4 without
    its final moov atom. The generic cleanup still runs afterward as a backstop.
    """
    if process is None:
        return False
    if process.poll() is not None:
        return True
    try:
        process.send_signal(signal.SIGINT)
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False
    except ProcessLookupError:
        return process.poll() is not None
    return True


def _render_navigation_video(frame_dir: Path, output_dir: Path) -> Path | None:
    """Save a playable head-camera clip even when the official rollout stops early."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not (frame_dir / "frame_00000.jpg").is_file():
        return None
    output = output_dir / "navigation_head.mp4"
    partial = output_dir / "navigation_head.partial.mp4"
    result = subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-framerate", "4",
            "-i", str(frame_dir / "frame_%05d.jpg"), "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(partial),
        ],
        capture_output=True, text=True, timeout=120, check=False,
    )
    if result.returncode != 0 or not partial.is_file():
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"could not render navigation video: {result.stderr.strip()}")
    partial.replace(output)
    return output


def _navigation_summary(progress: dict[str, Any]) -> dict[str, Any] | None:
    """Aggregate all segments without converting execution completion to arrival."""
    calls = progress.get("navigation_calls") or []
    if not calls:
        last = progress.get("last_navigation")
        calls = [last] if isinstance(last, dict) else []
    if not calls:
        return None
    return {
        **calls[-1],
        "nav_steps": sum(int(call.get("nav_steps") or 0) for call in calls),
        "model_frames": sum(int(call.get("model_frames") or 0) for call in calls),
        "call_count": len(calls),
    }


def _actions_settled(workspace: Path) -> bool:
    """True when no action is pending or running in the real queue."""
    action_file = workspace / "ACTION.md"
    try:
        content = action_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return True
    if not content:
        return True
    payload = parse_action_markdown(content)
    if payload is None:
        return False
    document = normalize_action_document(payload)
    if document is None:
        return False
    return all(
        str(action.get("status")) in {"completed", "failed", "cancelled"}
        for action in document.get("actions", [])
    )


def _resume_decision(
    *,
    returncode: int,
    finish_reason: str,
    status: dict[str, Any],
    feedback_rounds: int,
    max_rounds: int,
    actions_settled: bool,
) -> str:
    """Decide what to do after the Agent process exits.

    Returns ``infrastructure``, ``stop``, ``exhausted``, ``blocked``,
    ``resume_checkpoint`` or ``resume_feedback``. ``max_iterations`` is an
    internal Agent-segment boundary for long BEHAVIOR episodes, while a clean
    response uses the bounded evaluator-feedback path. Either continuation is
    safe only when no action is still in flight.
    """
    if finish_reason in INFRASTRUCTURE_FINISH_REASONS:
        return "infrastructure"
    if status.get("success") or status.get("done"):
        return "stop"
    if finish_reason == "max_iterations":
        return "resume_checkpoint" if actions_settled else "blocked"
    if returncode != 0:
        return "stop"
    if feedback_rounds >= max_rounds:
        return "exhausted"
    if not actions_settled:
        return "blocked"
    return "resume_feedback"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    args = parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.agent_max_iterations is not None and args.agent_max_iterations < 1:
        raise ValueError("--agent-max-iterations must be positive")
    if args.max_agent_feedback_rounds < 0:
        raise ValueError("--max-agent-feedback-rounds must not be negative")
    if args.upstream_port == args.gateway_port:
        raise ValueError("--upstream-port and --gateway-port must be different")

    behavior_root = require_path(args.behavior_root, "BEHAVIOR root")
    behavior_python = require_executable(
        WORKSPACE_ROOT / "envs/behavior/bin/python", "BEHAVIOR Python"
    )
    agent_python = require_executable(args.agent_python, "EMERGE Agent Python")
    args.agent_python = agent_python
    base_driver_config = _load_driver_config(args.driver_config)
    task_name = str(args.task_name or base_driver_config.get("expected_task") or "").strip()
    task_instruction = str(
        args.task_instruction or base_driver_config.get("required_instruction") or ""
    ).strip()
    if not task_name:
        raise ValueError(
            "task name is required: set expected_task in --driver-config or pass --task-name"
        )
    if not task_instruction:
        raise ValueError(
            "task instruction is required: set required_instruction in --driver-config "
            "or pass --task-instruction"
        )
    navigation_goal = str(args.navigation_goal or "").strip()
    if args.navigation_only:
        if not navigation_goal:
            raise ValueError("--navigation-goal is required with --navigation-only")
        if args.max_steps is None:
            raise ValueError("--max-steps is required with --navigation-only")
        if args.require_success:
            raise ValueError("--require-success is not meaningful for navigation-only runs")
        require_path(NAVIGATION_PROFILE, "navigation-only R1Pro profile")
    elif navigation_goal:
        raise ValueError("--navigation-goal requires --navigation-only")
    runtime_config = _agent_runtime_config(args)
    agent_model = _validate_agent_provider(args, runtime_config)
    print(f"EMERGE Agent provider ready for model: {agent_model}", flush=True)
    require_path(NAVIGATION_SKILL, "built-in LightNav skill")
    lightnav_ready = _validate_lightnav(required=args.navigation_only)
    # Neither navigation nor segmentation is mandatory for a complete task:
    # the Agent chooses skills, while champion VLA requires only its own server.
    sam3_url = _validate_behavior_perception(runtime_config, required=False)
    openpi_url = _validate_external_openpi(args.upstream_port)
    print(f"OpenPI service ready: {openpi_url}", flush=True)
    robot_config = require_path(args.robot_config, "R1Pro OpenPI evaluation config")

    assert_port_available(args.gateway_port, label="EMERGE gateway")
    if args.preflight_only:
        print(
            "BEHAVIOR preflight passed: "
            f"task={task_name}, OpenPI={openpi_url}, "
            f"LightNav={'ready' if lightnav_ready else 'unavailable (optional)'}, SAM3={sam3_url}, "
            f"BEHAVIOR={behavior_root}, "
            f"robot_config={robot_config}, gateway_port={args.gateway_port}.",
            flush=True,
        )
        return 0

    default_output = WORKSPACE_ROOT / "outputs/Emerge" / task_name
    if args.navigation_only:
        default_output /= f"navigation_only_{301 + args.instance_index}"
    requested_output = args.output_dir or default_output
    output_dir = assert_fresh_output_dir(requested_output)
    logs_dir = output_dir / "logs"
    workspace = output_dir / "workspace"
    status_path = workspace / "BEHAVIOR_STATUS.json"
    logs_dir.mkdir(parents=True, exist_ok=True)
    _prepare_workspace(
        workspace,
        task_name=task_name,
        instruction=navigation_goal if args.navigation_only else task_instruction,
    )
    runtime_environment = dict(os.environ)
    runtime_environment["NO_COLOR"] = "1"
    runtime_environment["TERM"] = "dumb"
    max_action_steps = (
        args.max_steps if args.navigation_only else args.max_steps + 2
    ) if args.max_steps is not None else int(base_driver_config.get("max_action_steps", 4000))
    driver_config = _episode_driver_config(
        base_driver_config,
        output_dir=output_dir,
        workspace=workspace,
        status_path=status_path,
        upstream_port=args.upstream_port,
        gateway_port=args.gateway_port,
        max_action_steps=max_action_steps,
        state_sync_timeout_s=args.state_sync_timeout,
        task_name=task_name,
        task_instruction=task_instruction,
        navigation_only=args.navigation_only,
        r1pro_urdf_path=behavior_root
        / "datasets/omnigibson-robot-assets/models/r1pro/urdf/r1pro.urdf",
    )

    processes: list = []
    agent_results: list[dict[str, Any]] = []
    feedback_rounds = 0
    max_iteration_restarts = 0
    termination_reason = "unknown"
    infrastructure_error: str | None = None
    evaluator_code: int | None = None
    failure: Exception | None = None
    video_shutdown_graceful: bool | None = None
    navigation_video: Path | None = None
    navigation_video_error: str | None = None
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.monotonic()
    agent: subprocess.Popen[bytes] | None = None
    evaluator: subprocess.Popen[bytes] | None = None

    try:
        with (
            (logs_dir / "controller.log").open("wb") as controller_log,
            (logs_dir / "evaluator.log").open("wb") as evaluator_log,
            (logs_dir / "agent.log").open("ab") as agent_log,
        ):
            controller = spawn_process(
                [
                    str(agent_python),
                    "-m",
                    "robot.controller",
                    "--driver",
                    "behavior_r1pro",
                    "--workspace",
                    str(workspace),
                    "--driver-config",
                    str(driver_config),
                    "--interval",
                    "0.05",
                ],
                cwd=EMERGE_ROOT,
                log=controller_log,
                env=runtime_environment,
            )
            processes.append(controller)
            print(f"Starting EMERGE Controller; log: {logs_dir / 'controller.log'}", flush=True)
            wait_for_health(args.gateway_port, controller, 30.0, label="EMERGE Controller")

            evaluator_command = _evaluator_command(
                behavior_python=behavior_python,
                robot_config=robot_config,
                gateway_port=args.gateway_port,
                task_name=task_name,
                instance_index=args.instance_index,
                # Two evaluator steps of headroom let the final camera frame
                # reach the controller before the official rollout terminates.
                max_steps=args.max_steps + 2 if args.navigation_only else args.max_steps,
                output_dir=output_dir,
                write_video=args.write_video,
                headless=args.headless,
            )
            evaluator = spawn_process(
                evaluator_command,
                cwd=behavior_root / "OmniGibson",
                log=evaluator_log,
                env=behavior_environment(),
            )
            processes.append(evaluator)
            print(f"Starting BEHAVIOR evaluator; log: {logs_dir / 'evaluator.log'}", flush=True)
            wait_for_condition(
                lambda: bool(robot_state_section(workspace).get("evaluator_connected")),
                timeout_s=args.startup_timeout,
                label="the BEHAVIOR evaluator to connect to the controlled gateway",
                processes=[controller, evaluator],
            )
            _write_running_evaluation_status(status_path, task_name=task_name)

            session_id = f"cli:behavior-{task_name.replace('_', '-')}-{uuid.uuid4().hex}"
            agent = _start_agent(
                args=args,
                message=_agent_message(
                    task_instruction=task_instruction,
                    navigation_goal=navigation_goal if args.navigation_only else None,
                ),
                session_id=session_id,
                workspace=workspace,
                environment=runtime_environment,
                log=agent_log,
            )
            processes.append(agent)
            print(f"Starting EMERGE Agent; log: {logs_dir / 'agent.log'}", flush=True)

            deadline = time.monotonic() + args.episode_timeout
            while True:
                if args.show_agent_events:
                    _drain_agent_events(agent)
                status = read_json(status_path)
                if evaluator.poll() is not None:
                    evaluator_code = evaluator.returncode
                    if evaluator_code == 0:
                        try:
                            status = _write_completed_evaluation_status(
                                output_dir,
                                status_path,
                                task_name=task_name,
                            )
                            termination_reason = str(
                                status.get("termination_reason") or "evaluator_done"
                            )
                        except Exception as exc:
                            infrastructure_error = f"official_result_error: {exc}"
                            termination_reason = "official_result_error"
                            _write_evaluator_error_status(
                                status_path,
                                infrastructure_error,
                                task_name=task_name,
                            )
                    else:
                        termination_reason = f"evaluator_exit_{evaluator_code}"
                        infrastructure_error = termination_reason
                        _write_evaluator_error_status(
                            status_path,
                            infrastructure_error,
                            task_name=task_name,
                        )
                    _settle_agent(agent, agent_results, timeout_s=5.0)
                    break
                if controller.poll() is not None:
                    infrastructure_error = f"controller_exit_{controller.returncode}"
                    termination_reason = infrastructure_error
                    _settle_agent(agent, agent_results, timeout_s=5.0)
                    break
                if agent.poll() is not None:
                    if args.show_agent_events:
                        _drain_agent_events(agent)
                    _record_agent_result(agent, agent_results)
                    finish_reason = str(agent_results[-1].get("finish_reason") or "agent_exit")
                    termination_reason = f"agent_{finish_reason}"
                    decision = _resume_decision(
                        returncode=int(agent.returncode),
                        finish_reason=finish_reason,
                        status=status,
                        feedback_rounds=feedback_rounds,
                        max_rounds=0 if args.navigation_only else args.max_agent_feedback_rounds,
                        actions_settled=_actions_settled(workspace),
                    )
                    if decision == "infrastructure":
                        infrastructure_error = termination_reason
                        break
                    if decision == "stop":
                        break
                    if decision == "exhausted":
                        termination_reason = f"agent_feedback_exhausted_{finish_reason}"
                        break
                    if decision == "blocked":
                        infrastructure_error = "agent_resume_blocked_by_active_action"
                        termination_reason = infrastructure_error
                        break
                    if decision == "resume_checkpoint":
                        max_iteration_restarts += 1
                        continuation_label = (
                            f"max-iteration checkpoint {max_iteration_restarts}"
                        )
                    else:
                        feedback_rounds += 1
                        continuation_label = f"feedback round {feedback_rounds}"
                    agent_log.write(
                        f"\n=== {continuation_label}: resuming same session after "
                        f"{finish_reason} ===\n".encode()
                    )
                    agent_log.flush()
                    agent = _start_agent(
                        args=args,
                        message=_feedback_message(
                            read_json(status_path),
                            task_instruction=task_instruction,
                        ),
                        session_id=session_id,
                        workspace=workspace,
                        environment=runtime_environment,
                        log=agent_log,
                    )
                    processes.append(agent)
                if time.monotonic() >= deadline:
                    termination_reason = "episode_timeout"
                    infrastructure_error = infrastructure_error or "episode_timeout"
                    _settle_agent(agent, agent_results, timeout_s=5.0)
                    break
                time.sleep(args.poll_interval)
    except Exception as exc:  # still write an episode result on every failure path
        failure = exc
        infrastructure_error = infrastructure_error or f"{type(exc).__name__}: {exc}"
        termination_reason = "runner_exception"
        _settle_agent(agent, agent_results, timeout_s=5.0)
    finally:
        if args.show_agent_events:
            _drain_agent_events(agent)
        if (
            args.navigation_only and not args.headless and sys.stdin.isatty()
            and evaluator is not None and evaluator.poll() is None
        ):
            print("Navigation has ended. Inspect the simulator window; press Enter to close it.", flush=True)
            try:
                input()
            except (EOFError, KeyboardInterrupt):
                pass
        if evaluator is not None and evaluator.poll() is None:
            print("Closing the official evaluator and finalizing its video...", flush=True)
            video_shutdown_graceful = _gracefully_stop_evaluator(evaluator)
            if not video_shutdown_graceful:
                print("WARNING: evaluator did not stop gracefully; the official MP4 may be incomplete.",
                      file=sys.stderr, flush=True)
        for process in reversed(processes):
            stop_process_group(process)
        if args.navigation_only and args.write_video:
            try:
                navigation_video = _render_navigation_video(
                    workspace / "lightnav_frames", output_dir
                )
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
                navigation_video_error = f"{type(exc).__name__}: {exc}"

    status = read_json(status_path)
    robot_state = robot_state_section(workspace)
    progress = robot_state.get("runtime_progress") or {}
    navigation_result = _navigation_summary(progress)
    if args.navigation_only:
        finish_reason = (navigation_result or {}).get("finish_reason")
        finished = finish_reason in {"model_stop_candidate", "step_budget"}
        verdict = "navigation_run_finished" if finished and not infrastructure_error else "navigation_run_incomplete"
        success = False  # Navigation is not official BEHAVIOR task success.
        exit_code = 0 if verdict == "navigation_run_finished" else 2
    else:
        verdict = classify_episode(
            status,
            evaluator_returncode=evaluator_code,
            infrastructure_error=infrastructure_error,
        )
        success = verdict == "success"
        exit_code = agent_exit_code(
            verdict=verdict,
            success=success,
            require_success=args.require_success,
        )
    result = {
        "schema_version": "Emerge.behavior_episode_result.v1",
        "task": task_name,
        "instance_index": args.instance_index,
        "instruction": task_instruction,
        "success": success,
        "episode_terminated": episode_terminated(status),
        "verdict": verdict,
        "infrastructure_error": infrastructure_error,
        "exit_code": exit_code,
        "require_success": bool(args.require_success),
        "termination_reason": termination_reason,
        "evaluator_returncode": evaluator_code,
        "evaluator_status": status,
        "agent_feedback_rounds": feedback_rounds,
        "agent_max_iteration_restarts": max_iteration_restarts,
        "agent_results": agent_results,
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.monotonic() - start_time, 3),
        "workspace": str(workspace),
        "navigation": navigation_result,
        "navigation_calls": progress.get("navigation_calls") or [],
        "posture_calls": progress.get("posture_calls") or [],
        "total_robot_action_steps": int(progress.get("gateway_action_steps") or 0),
        "evaluator_resets_forwarded": int(robot_state.get("evaluator_resets_forwarded") or 0),
        "upstream_cache_invalidations_sent": int(robot_state.get("upstream_cache_invalidations_sent") or 0),
        "video_shutdown_graceful": video_shutdown_graceful,
    }
    if args.navigation_only:
        result["navigation_goal"] = navigation_goal
        result["posture_steps"] = sum(
            int(call.get("completed_steps") or 0) for call in result["posture_calls"]
        )
        result["official_task_success_measured"] = False
        result["navigation_arrival_verified"] = False
        result["navigation_video"] = str(navigation_video) if navigation_video else None
        if navigation_video_error is not None:
            result["navigation_video_error"] = navigation_video_error
    if failure is not None:
        result["error"] = f"{type(failure).__name__}: {failure}"
    atomic_write_json(output_dir / "agent_episode_result.json", result)
    print(
        f"Episode finished: verdict={verdict} success={success} "
        f"reason={termination_reason}; result: "
        f"{output_dir / 'agent_episode_result.json'}",
        flush=True,
    )
    if failure is not None:
        print(f"Runner error: {failure}", file=sys.stderr, flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

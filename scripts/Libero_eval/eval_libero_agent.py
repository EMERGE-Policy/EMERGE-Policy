#!/usr/bin/env python3
"""Run LIBERO episodes with Emerge as the decision-making entrypoint."""

from __future__ import annotations

import argparse
import atexit
import concurrent.futures as futures
import copy
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Emerge.utils.action_queue import (  # noqa: E402
    normalize_action_document,
    parse_action_markdown,
)
from Emerge.utils.helpers import sync_workspace_templates  # noqa: E402
from external_model_server.model_service.discovery import ServiceDiscovery  # noqa: E402
from external_model_server.schemas import OPENPI  # noqa: E402
from robot.mujoco_simulation.scene_io import (  # noqa: E402
    default_robot_state_doc,
    load_robot_state_doc,
    save_robot_state_doc,
)
from robot.wam.protocol import WAM_SERVICE  # noqa: E402

DEFAULT_DRIVER_CONFIG = REPO_ROOT / "dev/libero_agent_eval.json"
DEFAULT_LIBERO_SOURCE = REPO_ROOT / "third_party/openpi/third_party/libero"
DEFAULT_VLA_PROFILE = REPO_ROOT / "robot/profiles/libero_mujoco.md"
DEFAULT_WAM_PROFILE = REPO_ROOT / "robot/profiles/libero_wam_mujoco.md"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MAX_STEPS = {
    "libero_spatial": 620,
    "libero_object": 680,
    "libero_goal": 700,
    "libero_10": 920,
}
DEFAULT_MAX_AGENT_FEEDBACK_ROUNDS = 2
DEFAULT_NUM_STEPS_WAIT = 50
_ACTIVE_PROCESSES: set[subprocess.Popen] = set()
_ACTIVE_PROCESSES_LOCK = threading.RLock()
_SHUTDOWN_SIGNAL: int | None = None
_SHUTDOWN_HANDLERS_INSTALLED = False
EVALUATION_INSTRUCTIONS = """

## LIBERO Evaluation Rules

You are the decision-making entrypoint for a LIBERO robot episode.

- Complete the user's manipulation instruction with `execute_robot_action`.
- Do not inspect processes, files, or directories outside this workspace.
- The official completion signal is `robots.libero_mujoco.success` in `ROBOT_STATE.md`.
- Treat that signal and any evaluator continuation feedback as authoritative.
- Do not claim the task is complete or stop while the official completion signal is false.
- After each action, inspect the latest environment state before deciding whether to continue.
- Treat an action result as an execution result, then decide whether another action is needed.
- Stop only after official success, an environment/step-limit termination, or when no safe useful action remains.
""".strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_environment(policy_backend: str | None = None) -> dict[str, str]:
    """Build the environment inherited by evaluation subprocesses."""
    runtime_env = dict(os.environ)
    runtime_env.setdefault("MUJOCO_GL", "egl")
    runtime_env["NO_COLOR"] = "1"
    runtime_env["TERM"] = "dumb"
    if policy_backend is not None:
        runtime_env["EMERGE_POLICY_BACKEND"] = policy_backend
    return runtime_env


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _policy_profile_path(
    base_config: dict[str, Any],
    *,
    policy_backend: str,
    override: Path | None,
) -> Path:
    """Select exactly one profile for a reproducible formal evaluation."""
    if override is not None:
        selected = _resolve_path(override)
    elif policy_backend == "wam":
        selected = DEFAULT_WAM_PROFILE.resolve()
    else:
        configured = base_config.get("profile_path")
        selected = _resolve_path(configured) if configured else DEFAULT_VLA_PROFILE.resolve()
    if not selected.is_file():
        raise FileNotFoundError(f"policy profile does not exist: {selected}")
    return selected


def _write_evaluation_plan(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    specs: list[dict[str, Any]],
    suite_names: list[str],
    task_ids_text: str,
    trials_per_task: int,
    driver_config_path: Path,
    profile_path: Path,
) -> None:
    """Persist backend/profile identity and reject incompatible resumes."""
    stable_plan = {
        "suites": suite_names,
        "task_ids": task_ids_text,
        "trials_per_task": trials_per_task,
        "start_trial": args.start_trial,
        "seed": args.seed,
        "episode_keys": [spec["key"] for spec in specs],
        "policy_backend": args.policy_backend,
        "wam_conditioning_mode": (
            args.wam_conditioning_mode if args.policy_backend == "wam" else None
        ),
        "profile_path": _display_path(profile_path),
        "profile_sha256": _sha256_file(profile_path),
        "driver_config_path": _display_path(driver_config_path),
        "driver_config_sha256": _sha256_file(driver_config_path),
    }
    plan_path = output_dir / "evaluation_plan.json"
    if args.resume and plan_path.exists():
        existing = _load_json(plan_path)
        previous = {key: existing.get(key) for key in stable_plan}
        if previous != stable_plan:
            raise ValueError(
                "resume arguments, policy backend, or profile do not match "
                "evaluation_plan.json; use the original settings or a new output directory"
            )
    _atomic_write_json(
        plan_path,
        {
            "schema_version": "Emerge.libero_evaluation_plan.v1",
            "updated_at": _utc_now(),
            **stable_plan,
        },
    )


def parse_task_ids(value: str, *, task_count: int) -> list[int]:
    """Parse ``all``, comma-separated ids, and inclusive ranges."""
    text = value.strip().lower()
    if text == "all":
        return list(range(task_count))
    selected: set[int] = set()
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"descending task range is not allowed: {token!r}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(token))
    invalid = sorted(item for item in selected if item < 0 or item >= task_count)
    if invalid:
        raise ValueError(
            f"task ids out of range for a {task_count}-task suite: {invalid}"
        )
    if not selected:
        raise ValueError("at least one task id is required")
    return sorted(selected)


def _resolve_path(value: str | Path, *, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _prepare_libero_config(output_dir: Path, source_path: Path) -> Path:
    benchmark_root = source_path / "libero/libero"
    config_dir = output_dir / ".libero"
    config_dir.mkdir(parents=True, exist_ok=True)
    datasets_path = source_path / "libero/datasets"
    if not datasets_path.exists():
        datasets_path = config_dir / "datasets"
        datasets_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "benchmark_root": benchmark_root,
        "bddl_files": benchmark_root / "bddl_files",
        "init_states": benchmark_root / "init_files",
        "datasets": datasets_path,
        "assets": benchmark_root / "assets",
    }
    required_names = ("benchmark_root", "bddl_files", "init_states", "assets")
    missing = [f"{name}: {paths[name]}" for name in required_names if not paths[name].exists()]
    if missing:
        raise FileNotFoundError("LIBERO source tree is incomplete:\n" + "\n".join(missing))
    config_text = "".join(f"{key}: {value}\n" for key, value in paths.items())
    (config_dir / "config.yaml").write_text(config_text, encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    source_text = str(source_path)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    return config_dir


def _load_benchmark_api():
    from libero.libero import benchmark, get_libero_path

    return benchmark, get_libero_path


def _suite_names(raw: list[str], *, full: bool) -> list[str]:
    if full or "all" in raw:
        return list(SUITES)
    unknown = sorted(set(raw) - set(SUITES))
    if unknown:
        raise ValueError(f"unknown suite(s): {', '.join(unknown)}")
    return list(dict.fromkeys(raw))


def _build_episode_specs(
    *,
    suite_names: Iterable[str],
    task_ids_text: str,
    trials_per_task: int,
    start_trial: int,
    seed: int,
) -> list[dict[str, Any]]:
    benchmark, get_libero_path = _load_benchmark_api()
    benchmark_dict = benchmark.get_benchmark_dict()
    task_contexts: list[dict[str, Any]] = []
    end_trial = start_trial + trials_per_task
    for suite_name in suite_names:
        suite = benchmark_dict[suite_name]()
        task_ids = parse_task_ids(task_ids_text, task_count=suite.n_tasks)
        for task_id in task_ids:
            task = suite.get_task(task_id)
            initial_states = np.asarray(suite.get_task_init_states(task_id))
            if end_trial > len(initial_states):
                raise ValueError(
                    f"{suite_name} task {task_id} only has {len(initial_states)} "
                    f"initial states, requested trials [{start_trial}, {end_trial})"
                )
            bddl_path = (
                Path(get_libero_path("bddl_files"))
                / task.problem_folder
                / task.bddl_file
            ).resolve()
            task_contexts.append(
                {
                    "suite": suite_name,
                    "task_id": task_id,
                    "instruction": str(task.language),
                    "bddl_file": str(bddl_path),
                    "initial_states": initial_states,
                }
            )

    # Round-robin by trial so every selected task receives one attempt before
    # any task receives its next attempt. This provides balanced coverage when
    # a long evaluation is interrupted or inspected before all trials finish.
    specs: list[dict[str, Any]] = []
    for trial in range(start_trial, end_trial):
        for context in task_contexts:
            suite_name = context["suite"]
            task_id = context["task_id"]
            initial_states = context["initial_states"]
            specs.append(
                {
                    "key": f"{suite_name}:task-{task_id:02d}:trial-{trial:02d}:seed-{seed}",
                    "suite": suite_name,
                    "task_id": task_id,
                    "trial": trial,
                    "seed": seed,
                    "instruction": context["instruction"],
                    "bddl_file": context["bddl_file"],
                    "initial_state": initial_states[trial].astype(float).tolist(),
                }
            )
    return specs


def _prepare_workspace(workspace: Path, spec: dict[str, Any]) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    sync_workspace_templates(workspace)
    agents_path = workspace / "AGENTS.md"
    existing = agents_path.read_text(encoding="utf-8") if agents_path.exists() else ""
    agents_path.write_text(
        existing.rstrip() + "\n\n" + EVALUATION_INSTRUCTIONS + "\n",
        encoding="utf-8",
    )
    robot_state = default_robot_state_doc()
    robot_state["task"] = {
        "benchmark": "LIBERO",
        "suite": spec["suite"],
        "instruction": spec["instruction"],
    }
    if spec.get("dimension"):
        robot_state["task"]["dimension"] = spec["dimension"]
    save_robot_state_doc(workspace / "ROBOT_STATE.md", robot_state)
    (workspace / "ACTION.md").write_text("", encoding="utf-8")


def _episode_driver_config(
    base_config: dict[str, Any],
    spec: dict[str, Any],
    *,
    episode_dir: Path,
    max_action_steps: int,
    num_steps_wait: int,
    wam_conditioning_mode: str,
    policy_backend: str,
    profile_path: Path,
    record_video: bool,
    stream_manifest_path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    config = copy.deepcopy(base_config)
    config["workspace"] = str((episode_dir / "workspace").resolve())
    config["profile_path"] = str(profile_path)
    config.setdefault("libero", {})["bddl_file_name"] = spec["bddl_file"]
    vla_config = config.setdefault("vla", {})
    vla_config["stop_on_success"] = True
    if policy_backend == "wam":
        wam_config = config.setdefault("wam", {})
        wam_config["stop_on_success"] = True
        wam_config["task_instruction"] = spec["instruction"]
        wam_config["conditioning_mode"] = wam_conditioning_mode
        wam_config["lock_conditioning_mode"] = True
    config["evaluation"] = {
        **dict(config.get("evaluation") or {}),
        "policy_backend": policy_backend,
        "initial_state": spec["initial_state"],
        "seed": spec["seed"],
        "num_steps_wait": num_steps_wait,
        "max_action_steps": max_action_steps,
        "status_path": str((episode_dir / "status.json").resolve()),
        "metadata": {
            "suite": spec["suite"],
            "task_id": spec["task_id"],
            "trial": spec["trial"],
            "seed": spec["seed"],
            "policy_backend": policy_backend,
            "wam_conditioning_mode": (
                wam_conditioning_mode if policy_backend == "wam" else None
            ),
            "profile_path": _display_path(profile_path),
            "profile_sha256": _sha256_file(profile_path),
        },
    }
    if spec.get("dimension"):
        config["evaluation"]["metadata"]["dimension"] = spec["dimension"]
    cameras = config["cameras"]
    for name, camera in cameras.items():
        if "observation" in camera:
            camera["observation"]["directory"] = str(
                (episode_dir / "workspace/artifacts/cameras" / name).resolve()
            )
        if "recording" in camera:
            camera["recording"]["enabled"] = False
    if record_video:
        if "agentview" not in cameras:
            raise ValueError("--record-video requires an agentview camera")
        cameras["agentview"]["recording"] = {
            "enabled": True,
            "path": str((episode_dir / "rollout.mp4").resolve()),
            "fps": 20.0,
            "codec": "mp4v",
        }
    if stream_manifest_path is not None:
        if "agentview" not in cameras:
            raise ValueError("--stream requires an agentview camera")
        has_reference = any(
            bool(dict(camera.get("observation") or {}).get("reference", False))
            for camera in cameras.values()
        )
        cameras["agentview"]["observation"] = {
            "enabled": True,
            "reference": not has_reference,
            "directory": str(
                (episode_dir / "workspace/artifacts/cameras/agentview").resolve()
            ),
        }

    path = episode_dir / "driver_config.json"
    _atomic_write_json(path, config)
    return config, path


def _server_is_ready(expectation, timeout_s: float = 2.0) -> bool:
    return ServiceDiscovery().is_ready(expectation, timeout=timeout_s)


def _read_status(path: Path) -> dict[str, Any]:
    status = _load_json(path)
    if status.get("schema_version") != "Emerge.libero_evaluation_status.v1":
        return {}
    return status


def _wait_for_watchdog(
    process: subprocess.Popen,
    status_path: Path,
    robot_state_path: Path,
    *,
    timeout_s: float,
    poll_interval_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = _read_status(status_path)
        robot_state = load_robot_state_doc(robot_state_path)
        robots = robot_state.get("robots")
        agent_state_ready = isinstance(robots, dict) and isinstance(
            robots.get("libero_mujoco"), dict
        )
        if status.get("ready") and agent_state_ready:
            return status
        exit_code = process.poll()
        if exit_code is not None:
            raise RuntimeError(
                f"Controller exited before the robot state was ready (code {exit_code})"
            )
        time.sleep(poll_interval_s)
    raise TimeoutError(f"Controller did not initialize the robot state within {timeout_s:.1f}s")


def _terminate_process(process: subprocess.Popen | None, *, grace_s: float = 15.0) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGINT)
        else:
            process.send_signal(signal.SIGINT)
        process.wait(timeout=grace_s)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)


def _track_process(process: subprocess.Popen) -> None:
    """Track an evaluator child and stop it if shutdown has already started."""
    with _ACTIVE_PROCESSES_LOCK:
        _ACTIVE_PROCESSES.add(process)
        shutting_down = _SHUTDOWN_SIGNAL is not None
    if shutting_down:
        _terminate_process(process)


def _untrack_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    with _ACTIVE_PROCESSES_LOCK:
        _ACTIVE_PROCESSES.discard(process)


def _terminate_tracked_process(process: subprocess.Popen | None) -> None:
    try:
        _terminate_process(process)
    finally:
        _untrack_process(process)


def _terminate_active_processes() -> None:
    """Best-effort cleanup for all agent/watchdog children owned by this run."""
    with _ACTIVE_PROCESSES_LOCK:
        processes = tuple(_ACTIVE_PROCESSES)
    for process in processes:
        _terminate_tracked_process(process)


def _evaluation_signal_handler(signum: int, _frame: Any) -> None:
    global _SHUTDOWN_SIGNAL
    with _ACTIVE_PROCESSES_LOCK:
        if _SHUTDOWN_SIGNAL is None:
            _SHUTDOWN_SIGNAL = signum
    _terminate_active_processes()
    if signum == signal.SIGINT:
        raise KeyboardInterrupt
    raise SystemExit(128 + signum)


def _install_evaluation_shutdown_handlers() -> None:
    """Install cleanup only in processes importing the evaluation lifecycle."""
    global _SHUTDOWN_HANDLERS_INSTALLED
    if (
        _SHUTDOWN_HANDLERS_INSTALLED
        or threading.current_thread() is not threading.main_thread()
    ):
        return
    for signal_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        signum = getattr(signal, signal_name, None)
        if signum is not None:
            signal.signal(signum, _evaluation_signal_handler)
    atexit.register(_terminate_active_processes)
    _SHUTDOWN_HANDLERS_INSTALLED = True


def _transcode_to_h264(path: Path) -> None:
    """Re-encode an mp4 in place to H.264 so browsers/VSCode can play it.

    OpenCV writes mp4v (MPEG-4 Part 2), which Chromium-based players (including
    the VSCode preview) cannot decode. The system ffmpeg carries libx264, so we
    transcode via a temp file and atomically replace. Any failure leaves the
    original untouched — this is a convenience step, never fatal to the run.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None or not path.exists():
        return
    temporary = path.with_name(f".{path.stem}.h264.mp4")
    result = subprocess.run(
        [
            ffmpeg, "-y", "-loglevel", "error", "-i", str(path),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(temporary),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and temporary.exists():
        os.replace(temporary, path)
    else:
        temporary.unlink(missing_ok=True)
        print(
            f"Warning: H.264 transcode failed for {path.name}; kept original. "
            f"{result.stderr.strip()[:200]}",
            file=sys.stderr,
        )


def _action_metrics(action_path: Path) -> dict[str, Any]:
    try:
        payload = parse_action_markdown(action_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        payload = None
    document = normalize_action_document(payload) if payload else None
    actions = document.get("actions", []) if document else []
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    for action in actions:
        action_type = str(action.get("action_type", "unknown"))
        status = str(action.get("status", "unknown"))
        by_type[action_type] = by_type.get(action_type, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "action_count": len(actions),
        "actions_by_type": by_type,
        "actions_by_status": by_status,
    }


def _agent_prompt(instruction: str) -> str:
    return (
        "Complete this LIBERO simulated robot task:\n\n"
        f"{instruction}\n\n"
        "Act autonomously through execute_robot_action. "
        "Do not finish until robots.libero_mujoco.success in ROBOT_STATE.md is true."
    )


def _agent_feedback_prompt(
    instruction: str,
    *,
    status: dict[str, Any],
    max_action_steps: int,
    policy_backend: str,
) -> str:
    action_steps = int(status.get("action_steps", 0))
    remaining_steps = max(0, max_action_steps - action_steps)
    return (
        "The official LIBERO evaluator reports success=false after your previous response. "
        "The task is not complete; continue the same episode and correct the execution.\n\n"
        f"Task: {instruction}\n"
        f"Action budget: {action_steps}/{max_action_steps} used, "
        f"{remaining_steps} remaining.\n\n"
        "Inspect the latest ROBOT_STATE.md and ACTION.md before acting. "
        "Do not merely repeat the previous completion summary. "
        f"Continue with the selected policy backend ({policy_backend}) when "
        "model-backed control is appropriate. "
        "Use execute_robot_action to make further progress, and do not finish until "
        "robots.libero_mujoco.success in ROBOT_STATE.md is true."
    )


def _should_continue_after_agent_exit(
    *,
    returncode: int,
    status: dict[str, Any],
    feedback_rounds: int,
    max_feedback_rounds: int,
) -> bool:
    return (
        returncode == 0
        and not status.get("success")
        and not status.get("done")
        and feedback_rounds < max_feedback_rounds
    )


def _read_agent_result(process) -> dict:
    """Validate the protocol independently of simulator task success."""
    from Emerge.runtime.protocol import RunResult

    result = RunResult.model_validate(_load_json(process.emerge_result_path))
    if (result.run_id != process.emerge_run_id
            or result.session_id != process.emerge_session_id
            or result.exit_code != process.returncode):
        raise ValueError("Agent result does not match its execution")
    return result.model_dump()


def _start_agent(
    *,
    args: argparse.Namespace,
    message: str,
    session_id: str,
    workspace: Path,
    runtime_env: dict[str, str],
    agent_log: Any,
) -> subprocess.Popen:
    # Each continuation is a separate run, sharing only this episode's session.
    import uuid
    run_id = uuid.uuid4().hex
    run_dir = workspace.parent / "agent_runs" / run_id
    request_path = workspace.parent / "agent_requests" / f"{run_id}.json"
    _atomic_write_json(request_path, {
        "schema_version": "Emerge.run_request.v1",
        "run_id": run_id,
        "message": message,
        "session_id": session_id,
        "workspace": str(workspace),
        "config": str(args.agent_config) if args.agent_config else None,
        "restrict_to_workspace": True,
        "stream": False,
    })
    command = [
        args.agent_python, "-m", "Emerge.cli.headless",
        "--request", str(request_path), "--output-dir", str(run_dir),
    ]
    process = subprocess.Popen(
        command,
        cwd=REPO_ROOT,
        env=runtime_env,
        stdout=agent_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    process.emerge_result_path = run_dir / "result.json"
    process.emerge_run_id = run_id
    process.emerge_session_id = session_id
    _track_process(process)
    return process


def _stream_manifest_path(episode_dir: Path) -> Path:
    """Where the environment publishes its current calibrated observation."""
    return (
        episode_dir / "workspace/artifacts/observations/observation.json"
    ).resolve()


def _worker_slot(workers: int) -> int:
    """Map the current worker thread to a stable grid slot in [0, workers).

    The ThreadPoolExecutor names threads ``libero-episode_<n>``; the serial
    (workers==1) path runs on the main thread and always uses slot 0.
    """
    name = threading.current_thread().name
    _, _, suffix = name.rpartition("_")
    return int(suffix) % workers if suffix.isdigit() else 0


def _episode_root_path(output_dir: Path, spec: dict[str, Any]) -> Path:
    """Return the episode root, optionally nested under a dimension slug."""
    episode_output_dir = output_dir
    if spec.get("dimension_slug"):
        episode_output_dir = episode_output_dir / str(spec["dimension_slug"])
    return (
        episode_output_dir
        / spec["suite"]
        / f"task_{spec['task_id']:02d}"
        / f"trial_{spec['trial']:02d}_seed_{spec['seed']}"
    )


def _run_episode(
    spec: dict[str, Any],
    *,
    args: argparse.Namespace,
    base_driver_config: dict[str, Any],
    output_dir: Path,
    board: Any | None = None,
) -> dict[str, Any]:
    episode_root = _episode_root_path(output_dir, spec)
    attempt_dir = episode_root / datetime.now().strftime("attempt_%Y%m%d_%H%M%S_%f")
    workspace = attempt_dir / "workspace"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    _prepare_workspace(workspace, spec)

    max_action_steps = args.max_steps or MAX_STEPS[spec["suite"]]
    stream_path = _stream_manifest_path(attempt_dir) if board is not None else None
    profile_path = _policy_profile_path(
        base_driver_config,
        policy_backend=args.policy_backend,
        override=args.profile_path,
    )
    _, driver_config_path = _episode_driver_config(
        base_driver_config,
        spec,
        episode_dir=attempt_dir,
        max_action_steps=max_action_steps,
        num_steps_wait=args.num_steps_wait,
        wam_conditioning_mode=args.wam_conditioning_mode,
        policy_backend=args.policy_backend,
        profile_path=profile_path,
        record_video=args.record_video,
        stream_manifest_path=stream_path,
    )
    slot = _worker_slot(args.workers)
    if board is not None and stream_path is not None:
        board.register(
            slot,
            stream_path,
            "agentview",
            f"{spec['suite']} t{spec['task_id']:02d}",
        )
    status_path = attempt_dir / "status.json"
    watchdog_log_path = attempt_dir / "watchdog.log"
    agent_log_path = attempt_dir / "agent.log"
    started_at = _utc_now()
    start_time = time.monotonic()
    watchdog: subprocess.Popen | None = None
    agent: subprocess.Popen | None = None
    success = False
    termination_reason = "unknown"
    status: dict[str, Any] = {}
    agent_feedback_rounds = 0
    watchdog_log = watchdog_log_path.open("w", encoding="utf-8")
    agent_log = agent_log_path.open("w", encoding="utf-8")

    runtime_env = _runtime_environment(args.policy_backend)
    if args.policy_backend == "wam":
        runtime_env["EMERGE_WAM_CONDITIONING_MODE"] = args.wam_conditioning_mode
        runtime_env["EMERGE_WAM_TASK_INSTRUCTION"] = spec["instruction"]
    try:
        watchdog_command = [
            args.watchdog_python,
            "-m",
            "robot.controller",
            "--driver",
            "libero_mujoco",
            "--workspace",
            str(workspace),
            "--driver-config",
            str(driver_config_path),
            "--interval",
            str(args.watchdog_poll_interval_s),
        ]
        watchdog = subprocess.Popen(
            watchdog_command,
            cwd=REPO_ROOT,
            env=runtime_env,
            stdout=watchdog_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        _track_process(watchdog)
        _wait_for_watchdog(
            watchdog,
            status_path,
            workspace / "ROBOT_STATE.md",
            timeout_s=args.watchdog_ready_timeout_s,
            poll_interval_s=args.poll_interval_s,
        )

        session_id = (
            f"cli:libero-eval-{spec['suite']}-{spec['task_id']}-{spec['trial']}-{spec['seed']}"
        )
        agent = _start_agent(
            args=args,
            message=_agent_prompt(spec["instruction"]),
            session_id=session_id,
            workspace=workspace,
            runtime_env=runtime_env,
            agent_log=agent_log,
        )

        def environment_terminal_reason(status: dict[str, Any]) -> str | None:
            """Return the terminal reason if the environment reports one, else None."""
            if status.get("success"):
                return "success"
            if status.get("done"):
                return status.get("termination_reason") or "environment_done"
            return None

        deadline = time.monotonic() + args.episode_timeout_s
        while True:
            status = _read_status(status_path)
            reason = environment_terminal_reason(status)
            if reason is not None:
                success = reason == "success"
                termination_reason = reason
                break
            if watchdog.poll() is not None:
                termination_reason = f"watchdog_exit_{watchdog.returncode}"
                break
            if agent.poll() is not None:
                returncode = int(agent.returncode)
                try:
                    run_result = _read_agent_result(agent)
                except ValueError:
                    run_result = {"finish_reason": "invalid_run_result"}
                    returncode = 1
                _untrack_process(agent)
                agent_log.flush()
                status = _read_status(status_path)
                reason = environment_terminal_reason(status)
                if reason is not None:
                    success = reason == "success"
                    termination_reason = reason
                    break
                if time.monotonic() >= deadline:
                    termination_reason = "episode_timeout"
                    break
                if not _should_continue_after_agent_exit(
                    returncode=returncode,
                    status=status,
                    feedback_rounds=agent_feedback_rounds,
                    max_feedback_rounds=args.max_agent_feedback_rounds,
                ):
                    reason = run_result.get("finish_reason", "missing_run_result")
                    if reason in {"runtime_error", "invalid_request", "invalid_run_result", "cleanup_error", "error"}:
                        termination_reason = f"infrastructure_error:agent:{reason}"
                    else:
                        termination_reason = f"agent_exit_{returncode}:{reason}"
                    break

                agent_feedback_rounds += 1
                agent_log.write(
                    f"\n=== LIBERO evaluator feedback round {agent_feedback_rounds}: "
                    "success=false; continuing same session ===\n"
                )
                agent_log.flush()
                agent = _start_agent(
                    args=args,
                    message=_agent_feedback_prompt(
                        spec["instruction"],
                        status=status,
                        max_action_steps=max_action_steps,
                        policy_backend=args.policy_backend,
                    ),
                    session_id=session_id,
                    workspace=workspace,
                    runtime_env=runtime_env,
                    agent_log=agent_log,
                )
                continue
            if time.monotonic() >= deadline:
                termination_reason = "episode_timeout"
                break
            time.sleep(args.poll_interval_s)
    except Exception as exc:
        success = False
        status = _read_status(status_path)
        termination_reason = f"infrastructure_error:{type(exc).__name__}:{exc}"
    finally:
        _terminate_tracked_process(agent)
        _terminate_tracked_process(watchdog)
        agent_log.close()
        watchdog_log.close()
        if board is not None:
            board.update_status(slot, "success" if success else "failed")

    rollout_path = attempt_dir / "rollout.mp4"
    if args.record_video and args.web_video:
        _transcode_to_h264(rollout_path)

    result = {
        "schema_version": "Emerge.libero_episode_result.v1",
        "episode_key": spec["key"],
        "suite": spec["suite"],
        "task_id": spec["task_id"],
        "trial": spec["trial"],
        "seed": spec["seed"],
        "instruction": spec["instruction"],
        "policy_backend": args.policy_backend,
        "wam_conditioning_mode": (
            args.wam_conditioning_mode if args.policy_backend == "wam" else None
        ),
        "profile_path": _display_path(profile_path),
        "profile_sha256": _sha256_file(profile_path),
        "success": success,
        "termination_reason": termination_reason,
        "action_steps": int(status.get("action_steps", 0)),
        "environment_steps": int(status.get("environment_steps", 0)),
        "max_action_steps": max_action_steps,
        "agent_feedback_rounds": agent_feedback_rounds,
        "agent_runs_dir": str(attempt_dir / "agent_runs"),
        "duration_s": round(time.monotonic() - start_time, 3),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "attempt_dir": str(attempt_dir),
        **_action_metrics(workspace / "ACTION.md"),
    }
    if spec.get("dimension"):
        result["dimension"] = spec["dimension"]
        result["dimension_slug"] = spec["dimension_slug"]
    _atomic_write_json(attempt_dir / "result.json", result)
    return result


def _read_results(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and item.get("episode_key"):
            results[str(item["episode_key"])] = item
    return results


def _write_summary(output_dir: Path, results: dict[str, dict[str, Any]]) -> None:
    suites: dict[str, dict[str, Any]] = {}
    for result in results.values():
        item = suites.setdefault(result["suite"], {"episodes": 0, "successes": 0})
        item["episodes"] += 1
        item["successes"] += int(bool(result.get("success")))
    for item in suites.values():
        item["success_rate"] = item["successes"] / item["episodes"] if item["episodes"] else 0.0
    total_episodes = sum(item["episodes"] for item in suites.values())
    total_successes = sum(item["successes"] for item in suites.values())
    summary = {
        "schema_version": "Emerge.libero_evaluation_summary.v1",
        "updated_at": _utc_now(),
        "episodes": total_episodes,
        "successes": total_successes,
        "success_rate": (total_successes / total_episodes if total_episodes else 0.0),
        "suites": suites,
    }
    _atomic_write_json(output_dir / "summary.json", summary)


def _is_infrastructure_error(result: dict[str, Any]) -> bool:
    return str(result.get("termination_reason", "")).startswith("infrastructure_error:")


def _iter_episode_results(
    indexed_specs: Iterable[tuple[int, dict[str, Any]]],
    *,
    total_specs: int,
    args: argparse.Namespace,
    base_driver_config: dict[str, Any],
    output_dir: Path,
    board: Any | None = None,
) -> Iterable[tuple[int, dict[str, Any], dict[str, Any]]]:
    """Run episodes with bounded concurrency and yield completed results.

    Result persistence intentionally remains outside this function so only the
    main thread writes results.jsonl and summary.json.
    """

    spec_iterator = iter(indexed_specs)

    def announce(index: int, spec: dict[str, Any]) -> None:
        print(
            f"[{index}/{total_specs}] run {spec['key']}: {spec['instruction']}",
            flush=True,
        )

    def run_one(index: int, spec: dict[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any]]:
        result = _run_episode(
            spec,
            args=args,
            base_driver_config=base_driver_config,
            output_dir=output_dir,
            board=board,
        )
        return index, spec, result

    if args.workers == 1:
        for index, spec in spec_iterator:
            announce(index, spec)
            item = run_one(index, spec)
            yield item
            if _is_infrastructure_error(item[2]) and not args.continue_on_error:
                return
        return

    active: dict[futures.Future[tuple[int, dict[str, Any], dict[str, Any]]], None] = {}
    stop_scheduling = False
    with futures.ThreadPoolExecutor(
        max_workers=args.workers,
        thread_name_prefix="libero-episode",
    ) as executor:

        def submit_next() -> bool:
            try:
                index, spec = next(spec_iterator)
            except StopIteration:
                return False
            announce(index, spec)
            active[executor.submit(run_one, index, spec)] = None
            return True

        while len(active) < args.workers and submit_next():
            pass

        while active:
            done, _ = futures.wait(
                active,
                return_when=futures.FIRST_COMPLETED,
            )
            for future in done:
                active.pop(future)
                item = future.result()
                if _is_infrastructure_error(item[2]) and not args.continue_on_error:
                    stop_scheduling = True
                yield item

            if not stop_scheduling:
                while len(active) < args.workers and submit_next():
                    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate LIBERO with Emerge as the agent entrypoint."
    )
    parser.add_argument(
        "--suite",
        action="append",
        default=[],
        help="Suite name; repeat for multiple suites, or use 'all'.",
    )
    parser.add_argument(
        "--task-ids",
        default="0",
        help="Task ids: 'all', comma-separated ids, or ranges such as 0,2-4.",
    )
    parser.add_argument("--trials-per-task", type=int, default=1)
    parser.add_argument("--start-trial", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of episodes to run concurrently. The policy server batches "
            "compatible worker requests; start with 3-4 workers."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run all four suites, all tasks, and 50 trials per task.",
    )
    parser.add_argument(
        "--driver-config",
        type=Path,
        default=DEFAULT_DRIVER_CONFIG,
    )
    parser.add_argument("--agent-config", type=Path, default=None)
    parser.add_argument("--agent-python", default=sys.executable)
    parser.add_argument("--watchdog-python", default=sys.executable)
    parser.add_argument(
        "--policy-backend",
        choices=("vla", "wam"),
        default="wam",
        help="Policy and profile used for this formal evaluation (default: wam).",
    )
    parser.add_argument(
        "--wam-conditioning-mode",
        choices=("task", "phase", "task_with_phase"),
        default="task",
        help="Text encoded by WAM: full task, current phase, or both.",
    )
    parser.add_argument(
        "--profile-path",
        type=Path,
        default=None,
        help="Override the profile selected for the active policy backend.",
    )
    parser.add_argument("--skip-policy-server-check", action="store_true")
    parser.add_argument("--skip-wam-server-check", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--num-steps-wait",
        type=int,
        default=DEFAULT_NUM_STEPS_WAIT,
        help=(
            "Dummy simulation steps after applying the official initial state "
            f"(default: {DEFAULT_NUM_STEPS_WAIT})."
        ),
    )
    parser.add_argument("--episode-timeout-s", type=float, default=900.0)
    parser.add_argument(
        "--max-agent-feedback-rounds",
        type=int,
        default=DEFAULT_MAX_AGENT_FEEDBACK_ROUNDS,
        help=(
            "Maximum number of same-session continuation turns after the agent exits "
            "normally while official LIBERO success is false."
        ),
    )
    parser.add_argument("--watchdog-ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--watchdog-poll-interval-s", type=float, default=0.1)
    parser.add_argument("--poll-interval-s", type=float, default=0.2)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument(
        "--web-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Transcode recorded mp4 to H.264 so it plays in browsers/VSCode.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Serve a live MJPEG grid of running episodes for browser viewing.",
    )
    parser.add_argument("--stream-host", default="127.0.0.1")
    parser.add_argument("--stream-port", type=int, default=8008)
    parser.add_argument("--stream-fps", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.trials_per_task <= 0:
        parser.error("--trials-per-task must be positive")
    if args.start_trial < 0:
        parser.error("--start-trial cannot be negative")
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if args.num_steps_wait < 0:
        parser.error("--num-steps-wait cannot be negative")
    if args.max_agent_feedback_rounds < 0:
        parser.error("--max-agent-feedback-rounds cannot be negative")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        _resolve_path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "artifacts/libero_agent_eval" / timestamp
    )
    results_path = output_dir / "results.jsonl"
    if results_path.exists() and not args.resume:
        parser.error(
            f"{results_path} already exists; choose another output directory or use --resume"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    driver_config_path = _resolve_path(args.driver_config)
    base_driver_config = _load_json(driver_config_path)
    if not base_driver_config:
        parser.error(f"driver config is missing or invalid: {driver_config_path}")
    libero_config = base_driver_config.get("libero")
    if not isinstance(libero_config, dict):
        parser.error("driver config must contain a libero object")
    source_path = _resolve_path(libero_config.get("libero_source_path", DEFAULT_LIBERO_SOURCE))
    _prepare_libero_config(output_dir, source_path)

    raw_suites = args.suite or ["libero_spatial"]
    try:
        suite_names = _suite_names(raw_suites, full=args.full)
        task_ids_text = "all" if args.full else args.task_ids
        trials_per_task = 50 if args.full else args.trials_per_task
        specs = _build_episode_specs(
            suite_names=suite_names,
            task_ids_text=task_ids_text,
            trials_per_task=trials_per_task,
            start_trial=args.start_trial,
            seed=args.seed,
        )
        profile_path = _policy_profile_path(
            base_driver_config,
            policy_backend=args.policy_backend,
            override=args.profile_path,
        )
        _write_evaluation_plan(
            output_dir,
            args=args,
            specs=specs,
            suite_names=suite_names,
            task_ids_text=task_ids_text,
            trials_per_task=trials_per_task,
            driver_config_path=driver_config_path,
            profile_path=profile_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"Prepared {len(specs)} episode(s): suites={suite_names}, "
        f"tasks={task_ids_text}, trials_per_task={trials_per_task}, "
        f"workers={args.workers}, policy={args.policy_backend}"
    )
    print(
        f"Profile: {_display_path(profile_path)} "
        f"(sha256={_sha256_file(profile_path)[:12]})"
    )
    print(f"Output: {output_dir}")
    if args.dry_run:
        for spec in specs:
            print(f"{spec['key']} | {spec['instruction']} | {spec['bddl_file']}")
        return 0

    if (
        args.policy_backend == "vla"
        and not args.skip_policy_server_check
        and not _server_is_ready(OPENPI)
    ):
        parser.error(
            "VLA policy server is not reachable through discovery; "
            "start scripts/model_server/start_external_model_servers.sh --services openpi first or pass "
            "--skip-policy-server-check"
        )
    if (
        args.policy_backend == "wam"
        and not args.skip_wam_server_check
        and not _server_is_ready(WAM_SERVICE)
    ):
        parser.error(
            "WAM policy server is not reachable through discovery; "
            "start external_model_server/cosmos_policy_server.py first or pass "
            "--skip-wam-server-check"
        )

    completed = _read_results(results_path)
    pending_specs: list[tuple[int, dict[str, Any]]] = []
    for index, spec in enumerate(specs, start=1):
        if args.resume and spec["key"] in completed:
            print(f"[{index}/{len(specs)}] skip completed {spec['key']}")
            continue
        pending_specs.append((index, spec))

    infrastructure_failed = False
    for _, spec, result in _iter_episode_results(
        pending_specs,
        total_specs=len(specs),
        args=args,
        base_driver_config=base_driver_config,
        output_dir=output_dir,
    ):
        with results_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        completed[spec["key"]] = result
        _write_summary(output_dir, completed)
        print(
            f"  success={result['success']} reason={result['termination_reason']} "
            f"steps={result['action_steps']} duration={result['duration_s']:.1f}s"
        )
        infrastructure_failed = infrastructure_failed or _is_infrastructure_error(result)

    if infrastructure_failed and not args.continue_on_error:
        print(
            "Stopped scheduling new episodes after infrastructure error.",
            file=sys.stderr,
        )
        return 2
    return 0


_install_evaluation_shutdown_handlers()


if __name__ == "__main__":
    raise SystemExit(main())

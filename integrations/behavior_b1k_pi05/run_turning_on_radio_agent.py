#!/usr/bin/env python3
"""Run one BEHAVIOR turning_on_radio episode with the real EMERGE Agent.

Unlike the action-queue smoke runner, the actions here are chosen by the LLM
Agent through ``execute_robot_action``.  The runner only supervises the three
processes, installs the BEHAVIOR ``vla`` skill, and decides when to resume a
session that stopped early without the episode being over.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parent
EMERGE_ROOT = PLUGIN_ROOT.parents[1]
WORKSPACE_ROOT = EMERGE_ROOT.parent
if str(EMERGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EMERGE_ROOT))

try:
    from .contracts import (
        OFFICIAL_PROMPT,
        TASK_NAME,
        agent_exit_code,
        classify_episode,
        episode_terminated,
        is_resumable_finish_reason,
    )
except ImportError:  # Direct script execution adds this directory to sys.path.
    sys.path.insert(0, str(PLUGIN_ROOT))
    from contracts import (  # type: ignore[no-redef]
        OFFICIAL_PROMPT,
        TASK_NAME,
        agent_exit_code,
        classify_episode,
        episode_terminated,
        is_resumable_finish_reason,
    )

try:
    from .run_action_queue_smoke import (
        _register_plugin,
        _write_driver_config,
    )
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
    )
except ImportError:  # Direct script execution.
    from run_action_queue_smoke import (  # type: ignore[no-redef]
        _register_plugin,
        _write_driver_config,
    )
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
    )

from Emerge.runtime.configuration import load_runtime_config  # noqa: E402
from Emerge.runtime.protocol import RunRequest, RunResult  # noqa: E402
from Emerge.utils.action_queue import (  # noqa: E402
    normalize_action_document,
    parse_action_markdown,
)
from Emerge.utils.helpers import sync_workspace_templates  # noqa: E402

#: Agent finish reasons that indicate the framework, not the task, is broken.
INFRASTRUCTURE_FINISH_REASONS = frozenset(
    {
        "cancellation_unconfirmed",
        "unfinished_action",
        "cleanup_error",
        "runtime_error",
        "invalid_request",
        "invalid_result",
    }
)

EVALUATION_INSTRUCTIONS = f"""
## BEHAVIOR-1K Evaluation Rules

You control one R1Pro episode for `turning_on_radio` through the
`behavior_b1k_pi05` driver.

- Use `execute_robot_action` with action type `vla_execute`.
- The instruction must be exactly, character for character: `{OFFICIAL_PROMPT}`
- `step` is an action budget and must not exceed
  `robots.behavior_r1pro.max_action_steps` in `ROBOT_STATE.md`.
- Use bounded chunks and inspect `ROBOT_STATE.md` after every action.
- The authoritative completion signal is
  `robots.behavior_r1pro.evaluation.success` together with
  `robots.behavior_r1pro.evaluation.phase == "completed"`.
- A `step_completed` action result only means the requested action budget was
  consumed. It does not mean the task succeeded.
- `robots.behavior_r1pro.state_synced` tells you whether the simulator has been
  confirmed to apply your last action. Do not judge progress while it is false.
- `robots.behavior_r1pro.evaluation.phase == "error"` is an evaluator failure,
  not a task failure.
- Current camera views are listed in
  `artifacts/observations/observation.json`. They are uncalibrated; use them
  only for qualitative judgement.
- Stop dispatching once `evaluation.done` is true and report the official
  result.
""".strip()


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
        "--agent-python",
        type=Path,
        default=Path(sys.executable),
        help="Python interpreter containing EMERGE's Agent and Controller dependencies.",
    )
    parser.add_argument("--agent-config", type=Path)
    parser.add_argument("--agent-model")
    parser.add_argument("--agent-max-iterations", type=int, default=20)
    parser.add_argument(
        "--max-agent-feedback-rounds",
        type=int,
        default=2,
        help=(
            "How many times a session may be resumed after the Agent stopped early "
            "for a benign reason (max_iterations / context length)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir("turning_on_radio_agent"),
    )
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20_000,
        help="Maximum BEHAVIOR simulator steps for the episode. Default: 20000.",
    )
    parser.add_argument("--upstream-port", type=int, default=8001)
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--startup-timeout", type=float, default=600.0)
    parser.add_argument("--episode-timeout", type=float, default=1800.0)
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
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Workspace
# --------------------------------------------------------------------------- #


def _prepare_workspace(workspace: Path) -> Path:
    """Sync templates, install the BEHAVIOR skill, and reset the queue."""
    workspace.mkdir(parents=True, exist_ok=True)
    sync_workspace_templates(workspace, silent=True)
    agents_path = workspace / "AGENTS.md"
    existing = agents_path.read_text(encoding="utf-8") if agents_path.exists() else ""
    if "BEHAVIOR-1K Evaluation Rules" not in existing:
        agents_path.write_text(
            existing.rstrip() + "\n\n" + EVALUATION_INSTRUCTIONS + "\n",
            encoding="utf-8",
        )
    prepare_workspace(workspace, task_name=TASK_NAME, instruction=OFFICIAL_PROMPT)
    install_workspace_skill(workspace)
    return workspace


def _agent_message(max_action_steps: int) -> str:
    return (
        "Complete this BEHAVIOR-1K simulated robot task:\n\n"
        f"{OFFICIAL_PROMPT}\n\n"
        "Act autonomously through execute_robot_action. Use the exact instruction above "
        f"and request at most {max_action_steps} steps per action. Start with a useful "
        "bounded chunk, then inspect the official evaluation state and camera snapshots. "
        "Do not finish while official success is false unless the evaluator has terminated."
    )


def _feedback_message(status: dict[str, Any], max_action_steps: int) -> str:
    return (
        "Your previous turn ended before the episode finished. The official BEHAVIOR "
        "evaluator is still running, so continue the same episode and use "
        "execute_robot_action again.\n\n"
        f"Task: {OFFICIAL_PROMPT}\n"
        f"Current evaluator status: {json.dumps(status, ensure_ascii=False)}\n"
        f"Maximum steps per action: {max_action_steps}.\n\n"
        "Inspect ROBOT_STATE.md, ACTION.md, and the latest camera manifest before acting."
    )


# --------------------------------------------------------------------------- #
# Agent process
# --------------------------------------------------------------------------- #


def _validate_agent_provider(args: argparse.Namespace) -> str:
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
    request = {
        "message": "provider preflight",
        "config": (
            str(args.agent_config.expanduser().resolve()) if args.agent_config is not None else None
        ),
        "model": args.agent_model,
    }
    config = load_runtime_config(RunRequest.model_validate(request))
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
        "max_iterations": args.agent_max_iterations,
    }
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
    finish_reason: str,
    feedback_rounds: int,
    max_rounds: int,
    actions_settled: bool,
) -> str:
    """Decide what to do after the Agent process exits.

    Returns ``infrastructure``, ``stop``, ``exhausted``, ``blocked`` or
    ``resume``.  Resuming is only safe once the previous run's cancellation was
    confirmed and no action is still in flight.
    """
    if finish_reason in INFRASTRUCTURE_FINISH_REASONS:
        return "infrastructure"
    if not is_resumable_finish_reason(finish_reason):
        return "stop"
    if feedback_rounds >= max_rounds:
        return "exhausted"
    if not actions_settled:
        return "blocked"
    return "resume"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.agent_max_iterations < 1:
        raise ValueError("--agent-max-iterations must be positive")
    if args.max_agent_feedback_rounds < 0:
        raise ValueError("--max-agent-feedback-rounds must not be negative")

    openpi_root = require_path(args.openpi_root, "OpenPI root")
    behavior_root = require_path(args.behavior_root, "BEHAVIOR root")
    checkpoint = require_path(args.checkpoint, "π0.5 checkpoint")
    openpi_python = require_executable(openpi_root / ".venv/bin/python", "OpenPI Python")
    behavior_python = require_executable(
        WORKSPACE_ROOT / "envs/behavior/bin/python", "BEHAVIOR Python"
    )
    agent_python = require_executable(args.agent_python, "EMERGE Agent Python")
    args.agent_python = agent_python
    agent_model = _validate_agent_provider(args)
    print(f"EMERGE Agent provider ready for model: {agent_model}", flush=True)
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

    _prepare_workspace(workspace)
    runtime_environment = _register_plugin(agent_python, plugin_home)
    runtime_environment["NO_COLOR"] = "1"
    runtime_environment["TERM"] = "dumb"
    driver_config = output_dir / "driver_config.json"
    max_action_steps = args.max_steps + 2
    _write_driver_config(
        driver_config,
        workspace=workspace,
        upstream_port=args.upstream_port,
        gateway_port=args.gateway_port,
        max_action_steps=max_action_steps,
        status_path=status_path,
        state_sync_timeout_s=args.state_sync_timeout,
    )

    processes: list = []
    agent_results: list[dict[str, Any]] = []
    feedback_rounds = 0
    termination_reason = "unknown"
    infrastructure_error: str | None = None
    evaluator_code: int | None = None
    failure: Exception | None = None
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.monotonic()
    agent: subprocess.Popen[bytes] | None = None

    try:
        with (
            (logs_dir / "openpi.log").open("wb") as openpi_log,
            (logs_dir / "controller.log").open("wb") as controller_log,
            (logs_dir / "evaluator.log").open("wb") as evaluator_log,
            (logs_dir / "agent.log").open("ab") as agent_log,
        ):
            openpi = spawn_process(
                [
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
                ],
                cwd=openpi_root,
                log=openpi_log,
            )
            processes.append(openpi)
            print(f"Starting OpenPI; log: {logs_dir / 'openpi.log'}", flush=True)
            wait_for_health(
                args.upstream_port, openpi, args.startup_timeout, label="OpenPI server"
            )

            controller = spawn_process(
                [
                    str(agent_python),
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
                ],
                cwd=EMERGE_ROOT,
                log=controller_log,
                env=runtime_environment,
            )
            processes.append(controller)
            print(f"Starting EMERGE Controller; log: {logs_dir / 'controller.log'}", flush=True)
            wait_for_health(args.gateway_port, controller, 30.0, label="EMERGE Controller")

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
            wait_for_condition(
                lambda: bool(read_json(status_path)),
                timeout_s=args.startup_timeout,
                label="the first official evaluator status frame",
                processes=[controller, evaluator],
            )

            session_id = f"cli:behavior-turning-on-radio-{uuid.uuid4().hex}"
            agent = _start_agent(
                args=args,
                message=_agent_message(max_action_steps),
                session_id=session_id,
                workspace=workspace,
                environment=runtime_environment,
                log=agent_log,
            )
            processes.append(agent)
            print(f"Starting EMERGE Agent; log: {logs_dir / 'agent.log'}", flush=True)

            deadline = time.monotonic() + args.episode_timeout
            while True:
                status = read_json(status_path)
                if status.get("done"):
                    try:
                        evaluator.wait(timeout=120.0)
                    except subprocess.TimeoutExpired:
                        infrastructure_error = "evaluator_finalize_timeout"
                        termination_reason = "evaluator_finalize_timeout"
                        break
                    evaluator_code = evaluator.returncode
                    status = read_json(status_path)
                    termination_reason = str(status.get("termination_reason") or "evaluator_done")
                    if evaluator_code not in (0, None):
                        infrastructure_error = f"evaluator_exit_{evaluator_code}"
                    _settle_agent(agent, agent_results)
                    break
                if evaluator.poll() is not None:
                    evaluator_code = evaluator.returncode
                    termination_reason = f"evaluator_exit_{evaluator_code}"
                    if evaluator_code != 0:
                        infrastructure_error = termination_reason
                    _settle_agent(agent, agent_results, timeout_s=5.0)
                    break
                if controller.poll() is not None:
                    infrastructure_error = f"controller_exit_{controller.returncode}"
                    termination_reason = infrastructure_error
                    _settle_agent(agent, agent_results, timeout_s=5.0)
                    break
                if agent.poll() is not None:
                    _record_agent_result(agent, agent_results)
                    finish_reason = str(agent_results[-1].get("finish_reason") or "agent_exit")
                    termination_reason = f"agent_{finish_reason}"
                    decision = _resume_decision(
                        finish_reason=finish_reason,
                        feedback_rounds=feedback_rounds,
                        max_rounds=args.max_agent_feedback_rounds,
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
                    feedback_rounds += 1
                    agent_log.write(
                        f"\n=== feedback round {feedback_rounds}: resuming after "
                        f"{finish_reason} ===\n".encode()
                    )
                    agent_log.flush()
                    agent = _start_agent(
                        args=args,
                        message=_feedback_message(read_json(status_path), max_action_steps),
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
        for process in reversed(processes):
            stop_process_group(process)

    status = read_json(status_path)
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
        "task": TASK_NAME,
        "instance_index": args.instance_index,
        "instruction": OFFICIAL_PROMPT,
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
        "agent_results": agent_results,
        "workspace_skill": str(workspace / "skills/vla/SKILL.md"),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": round(time.monotonic() - start_time, 3),
        "workspace": str(workspace),
    }
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

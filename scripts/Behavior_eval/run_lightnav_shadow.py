#!/usr/bin/env python3
"""Observe LightNav predictions beside an unchanged champion BEHAVIOR rollout.

The gateway is passive. The Agent may choose when to preview a navigation
subgoal, but neither this runner nor its tools can change simulator actions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shutil
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
EMERGE_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = EMERGE_ROOT.parent
if str(EMERGE_ROOT) not in sys.path:
    sys.path.insert(0, str(EMERGE_ROOT))

from Emerge.runtime.protocol import RunRequest  # noqa: E402
from Emerge.runtime.service import AgentRuntime, build_agent  # noqa: E402
from integrations.lightnav_behavior.preview import LightNavPreviewTool  # noqa: E402
from integrations.lightnav_sonic_emerge.bridge.lightnav_ws import LightNavWsClient  # noqa: E402
from scripts.Behavior_eval.eval_behavior_agent import _evaluator_command  # noqa: E402
from scripts.Behavior_eval.shadow_common import (  # noqa: E402
    MANIFEST,
    STATUS,
    WatchObservationTool,
    gateway_command,
    prepare_observer_workspace,
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
    wait_for_health,
)


SKILL_SOURCE = EMERGE_ROOT / "integrations/lightnav_behavior/workspace_skill/SKILL.md"
PREVIEW_TOOLS = frozenset({
    "read_file", "list_dir", "update_plan", "watch_observation", "preview_navigation",
})


def install_preview_skill(workspace: Path) -> Path:
    """Expose this experiment's guidance only inside its own Agent workspace."""
    destination = workspace / "skills/lightnav-behavior-preview/SKILL.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SKILL_SOURCE, destination)
    return destination


def build_preview_observer(config, *, server_url: str, tools: list[LightNavPreviewTool]):
    """Retain the native Agent loop, but expose only read-only navigation tools."""
    config.tools.restrict_to_workspace = True
    config.visual_monitor.enabled = False
    agent = build_agent(config)
    agent.subagent_registry.unregister("object_location")
    agent.subagent_registry.unregister("task_verification")
    agent.tools.register(WatchObservationTool(config.workspace_path))
    preview = LightNavPreviewTool(config.workspace_path, server_url=server_url)
    agent.tools.register(preview)
    for name in tuple(agent.tools.tool_names):
        if name not in PREVIEW_TOOLS:
            agent.tools.unregister(name)
    if set(agent.tools.tool_names) != PREVIEW_TOOLS:
        raise RuntimeError(f"unexpected preview observer tools: {agent.tools.tool_names}")
    if agent.subagent_registry.names:
        raise RuntimeError(f"unexpected preview observer specialists: {agent.subagent_registry.names}")
    tools.append(preview)
    return agent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", default="turning_on_radio")
    parser.add_argument("--instance-index", type=int, default=0)
    parser.add_argument("--agent-config", type=Path)
    parser.add_argument("--agent-model", default="gpt-6.1-sol")
    parser.add_argument("--agent-max-iterations", type=int, default=80)
    parser.add_argument("--agent-timeout", type=float, default=7200)
    parser.add_argument("--episode-timeout", type=float, default=7200)
    parser.add_argument("--agent-finish-grace", type=float, default=180)
    parser.add_argument("--first-observation-timeout", type=float, default=900)
    parser.add_argument("--upstream-port", type=int, default=8011)
    parser.add_argument("--gateway-port", type=int, default=8000)
    parser.add_argument("--lightnav-server", default="ws://127.0.0.1:8051")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-video", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def preflight(args):
    if not 0 <= args.instance_index <= 9:
        raise ValueError("public_test instance-index must be 0-9")
    if not args.task_name.replace("_", "").isalnum():
        raise ValueError("task-name must contain only letters, digits, and underscores")
    if args.agent_max_iterations < 1 or min(args.agent_timeout, args.episode_timeout,
                                             args.agent_finish_grace, args.first_observation_timeout) <= 0:
        raise ValueError("iteration and timeout limits must be positive")
    behavior_root = require_path(EMERGE_ROOT / "third_party/behavior-1k", "BEHAVIOR source")
    behavior_python = require_executable(WORKSPACE_ROOT / "envs/behavior/bin/python", "BEHAVIOR Python")
    robot_config = require_path(EMERGE_ROOT / "dev/behavior_r1pro_openpi.yaml", "robot config")
    camera_config = require_path(EMERGE_ROOT / "dev/behavior_agent_eval.json", "camera config")
    if args.agent_config:
        require_path(args.agent_config, "EMERGE config")
    if not service_health_ok(f"http://127.0.0.1:{args.upstream_port}"):
        raise RuntimeError(f"champion policy is not healthy on port {args.upstream_port}")
    assert_port_available(args.gateway_port, label="passive gateway")
    if not SKILL_SOURCE.is_file():
        raise FileNotFoundError(f"LightNav workspace skill is missing: {SKILL_SOURCE}")
    with LightNavWsClient(
        args.lightnav_server,
        client_id="emerge-behavior-shadow-preflight",
        request_timeout_s=10.0,
    ):
        pass
    return behavior_root, behavior_python, robot_config, camera_config


async def run(args):
    behavior_root, behavior_python, robot_config, camera_config = await asyncio.to_thread(preflight, args)
    instance_id = 301 + args.instance_index
    output = (args.output_dir or WORKSPACE_ROOT / "outputs/Emerge" / args.task_name /
              f"lightnav_shadow_{instance_id}").expanduser().resolve()
    if args.preflight_only:
        print("Preflight OK: champion, LightNav, passive gateway port, and paths are ready.")
        return 0
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output is not empty; refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir()
    workspace = output / "workspace"
    camera_data = json.loads(camera_config.read_text(encoding="utf-8"))
    instruction = camera_data["required_instruction"]
    if camera_data.get("expected_task") != args.task_name:
        raise ValueError("driver config instruction belongs to another task")
    prepare_observer_workspace(workspace, task_name=args.task_name, instruction=instruction)
    install_preview_skill(workspace)

    gateway = evaluator = None
    cancel = asyncio.Event()
    agent_task = None
    agent_events = []
    preview_tools: list[LightNavPreviewTool] = []
    try:
        with (logs / "gateway.log").open("wb") as gateway_log, (logs / "evaluator.log").open("wb") as eval_log:
            env = dict(os.environ, PYTHONPATH=str(EMERGE_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
            gateway = spawn_process(gateway_command(
                upstream_port=args.upstream_port, gateway_port=args.gateway_port,
                workspace=workspace, camera_config=camera_config,
            ), cwd=EMERGE_ROOT, log=gateway_log, env=env)
            await asyncio.to_thread(wait_for_health, args.gateway_port, gateway, 60, label="passive gateway")
            evaluator = spawn_process(_evaluator_command(
                behavior_python=behavior_python, robot_config=robot_config,
                gateway_port=args.gateway_port, task_name=args.task_name,
                instance_index=args.instance_index, max_steps=None, output_dir=output,
                write_video=args.write_video, headless=args.headless,
            ), cwd=behavior_root, log=eval_log, env=behavior_environment())
            print(f"Official evaluator started; passive LightNav output: {output}", flush=True)
            deadline = time.monotonic() + args.first_observation_timeout
            while not (workspace / MANIFEST).is_file():
                if evaluator.poll() is not None or time.monotonic() >= deadline:
                    break
                await asyncio.sleep(0.25)
            if (workspace / MANIFEST).is_file():
                request = RunRequest(
                    message=(
                        f"Observe the ongoing episode: {instruction} "
                        "The champion policy alone controls the robot. Independently decide "
                        "which available read-only observations or skills are useful "
                        "and when to use them. Monitor until the episode ends, then summarize "
                        "what you actually observed and what the official result reports."
                    ),
                    config=str(args.agent_config.resolve()) if args.agent_config else None,
                    workspace=str(workspace), model=args.agent_model,
                    restrict_to_workspace=True, max_iterations=args.agent_max_iterations,
                    timeout_s=args.agent_timeout,
                )

                def on_event(event):
                    if event.type in {"tool.started", "tool.completed", "tool.failed", "run.finished"}:
                        agent_events.append(event.model_dump(mode="json"))
                        if event.type == "tool.started":
                            print(f"Agent tool: {event.data.get('tool')}", flush=True)

                def observer_factory(config):
                    return build_preview_observer(
                        config, server_url=args.lightnav_server, tools=preview_tools,
                    )

                agent_task = asyncio.create_task(AgentRuntime(observer_factory).run(
                    request, output_dir=output / "agent_run", on_event=on_event, cancel=cancel,
                ))
                print("Read-only EMERGE Agent started; no execute_robot_action tool is available.", flush=True)
            else:
                print("No camera manifest arrived; official evaluator continues without observer.", flush=True)
            deadline = time.monotonic() + args.episode_timeout
            while evaluator.poll() is None and time.monotonic() < deadline:
                await asyncio.sleep(0.5)
            if evaluator.poll() is None:
                raise TimeoutError("official evaluator exceeded episode-timeout")
            official_path = output / "json" / f"{args.task_name}_{instance_id}_0.json"
            official = read_json(official_path)
            phase = "completed" if evaluator.returncode == 0 and official else "error"
            atomic_write_json(workspace / STATUS, {
                "phase": phase, "task": args.task_name,
                "official_result": str(official_path) if official else None,
                "success": official.get("success") if official else None,
                "evaluator_exit_code": evaluator.returncode,
            })
            agent_result = None
            if agent_task:
                try:
                    agent_result = await asyncio.wait_for(
                        asyncio.shield(agent_task), timeout=args.agent_finish_grace,
                    )
                except asyncio.TimeoutError:
                    cancel.set()
                    agent_result = await asyncio.wait_for(agent_task, timeout=30)
            preview_calls = [event for event in agent_events if event["type"] == "tool.started"
                             and event["data"].get("tool") == "preview_navigation"]
            preview_ids = {event["data"]["tool_call_id"] for event in preview_calls}
            preview_outcomes = [event for event in agent_events
                                if event["type"] in {"tool.completed", "tool.failed"}
                                and event["data"].get("tool_call_id") in preview_ids]
            prediction_log = workspace / "artifacts/lightnav_preview/predictions.jsonl"
            prediction_count = (
                sum(1 for line in prediction_log.read_text(encoding="utf-8").splitlines() if line)
                if prediction_log.is_file() else 0
            )
            summary = {
                "task": args.task_name, "instance_id": instance_id,
                "official_result": str(official_path) if official else None,
                "official_success": official.get("success") if official else None,
                "official_steps": official.get("steps") if official else None,
                "evaluator_exit_code": evaluator.returncode,
                "agent_run_status": agent_result.run_status if agent_result else "not_started",
                "agent_finish_reason": agent_result.finish_reason if agent_result else None,
                "preview_calls": len(preview_calls),
                "prediction_count": prediction_count,
                "preview_call_events": preview_calls,
                "preview_outcome_events": preview_outcomes,
                "prediction_log": str(prediction_log) if prediction_count else None,
                "gateway_mode": "passive_uncontrolled",
            }
            atomic_write_json(output / "lightnav_shadow_summary.json", summary)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
            return 0 if phase == "completed" else 1
    finally:
        cancel.set()
        if agent_task and not agent_task.done():
            agent_task.cancel()
        stop_process_group(evaluator)
        stop_process_group(gateway)
        if agent_task and not agent_task.done():
            await asyncio.gather(agent_task, return_exceptions=True)
        for preview_tool in preview_tools:
            preview_tool.close()


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(run(parse_args())))
    except (KeyboardInterrupt, Exception) as exc:
        print(f"LightNav shadow failed: {exc}", file=sys.stderr)
        raise SystemExit(130 if isinstance(exc, KeyboardInterrupt) else 1) from exc

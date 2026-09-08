#!/usr/bin/env python3
"""
robot/controller.py

Controller — polls ACTION.md for commands, dispatches them to the
active driver, and writes updated state back to ROBOT_STATE.md.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from robot.drivers.base_driver import BaseDriver
from robot.mujoco_simulation.scene_io import (
    load_robot_state_doc,
    load_scene_from_md,
    merge_robot_state_doc,
    save_robot_state_doc,
)
from Emerge.utils.action_queue import (
    first_pending_action,
    infer_terminal_status,
    normalize_action_document,
    parse_action_markdown,
    update_action_document,
    action_timestamp,
)


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[Controller {ts}] {msg}", flush=True)


def _load_scene(path: Path) -> dict[str, dict]:
    return load_scene_from_md(path)


def load_driver_config(path: Path | None) -> dict[str, object]:
    """Load a driver config JSON object for transparent kwargs passthrough."""
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"driver-config file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"failed to parse driver-config JSON: {path}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"driver-config must be a JSON object: {path}")
    return data


def _publish_runtime_state(driver: BaseDriver, path: Path) -> None:
    existing = load_robot_state_doc(path)
    runtime_state = driver.get_runtime_state()
    updated = merge_robot_state_doc(
        existing,
        objects=runtime_state.get("objects"),
        robots=runtime_state.get("robots"),
        scene_graph=runtime_state.get("scene_graph"),
        map_data=runtime_state.get("map"),
        tf_data=runtime_state.get("tf"),
        updated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat() + "Z",
    )
    save_robot_state_doc(path, updated)


def _install_profile(driver, workspace: Path) -> None:
    """Copy the driver's EMBODIED.md profile into the workspace."""
    src = driver.get_profile_path()
    dst = workspace / "EMBODIED.md"
    if src.exists():
        workspace.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        _log(f"Profile installed: {src.name} -> {dst}")
    else:
        _log(f"WARNING: profile not found at {src}")


def _driver_kwargs_for_workspace(
    driver_name: str,
    workspace: Path,
    driver_kwargs: dict[str, object] | None,
) -> dict[str, object]:
    """Supply the agent workspace to drivers that store local artifacts."""
    resolved = dict(driver_kwargs or {})
    if driver_name == "libero_mujoco":
        resolved.setdefault("workspace", str(workspace))
    return resolved


def watch_loop(
    workspace: Path,
    driver_name: str = "simulation",
    gui: bool = False,
    poll_interval: float = 1.0,
    *,
    driver_kwargs: dict[str, object] | None = None,
    robot_state_file: Path | None = None,
) -> None:
    """Load a driver, install its profile, then poll ACTION.md forever."""
    from robot.drivers import load_driver

    robot_state_file = robot_state_file or (workspace / "ROBOT_STATE.md")

    _log(f"Workspace : {workspace}")
    _log(f"Driver    : {driver_name}")
    _log(f"GUI       : {gui}")
    _log(f"State File: {robot_state_file}")
    if driver_kwargs:
        _log(f"DriverCfg : {json.dumps(driver_kwargs, ensure_ascii=False, sort_keys=True)}")

    resolved_driver_kwargs = _driver_kwargs_for_workspace(
        driver_name,
        workspace,
        driver_kwargs,
    )
    driver = load_driver(driver_name, gui=gui, **resolved_driver_kwargs)

    with driver:
        _install_profile(driver, workspace)
        scene = _load_scene(robot_state_file)
        driver.load_scene(scene)
        _publish_runtime_state(driver, robot_state_file)
        _log(f"Scene loaded ({len(scene)} object(s))")
        _log("Watching ACTION.md ... Ctrl+C to stop.\n")

        action_file = workspace / "ACTION.md"
        try:
            while True:
                _poll_once(driver, action_file, robot_state_file)
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            _log("Shutdown.")


def _poll_once(driver, action_file: Path, robot_state_file: Path) -> None:
    """Single poll: publish runtime state, then execute pending ACTION.md."""
    _publish_runtime_state(driver, robot_state_file)

    if not action_file.exists():
        return
    content = action_file.read_text(encoding="utf-8").strip()
    if not content:
        return

    payload = parse_action_markdown(content)
    if payload is None:
        _log("ACTION.md has content but no valid JSON - skipping.")
        return
    document = normalize_action_document(payload)
    if document is None:
        _log("ACTION.md contains unreadable action data - skipping.")
        return
    pending = first_pending_action(document)
    if pending is None:
        _log("ACTION.md has no pending actions - skipping.")
        return
    _, action = pending
    action_id = str(action["id"])
    claimed = False
    def claim(latest):
        nonlocal claimed
        for item in latest["actions"]:
            if item["id"] != action_id or item["status"] != "pending":
                continue
            if item.get("cancel_requested"):
                item.update(status="cancelled", result="Cancelled before execution.",
                            finished_at=action_timestamp(), cancel_acknowledged_at=action_timestamp())
            else:
                item.update(status="running", started_at=action_timestamp())
                claimed = True
    update_action_document(action_file, claim)
    if not claimed:
        return
    started = time.monotonic()

    action_type = action.get("action_type", "unknown")
    params = action.get("parameters", {})
    _log(f"Action: {action_type!r}  params={params}")

    time.sleep(0.3)

    action_id = str(action["id"])

    def cancel_check() -> str | None:
        latest = parse_action_markdown(action_file.read_text(encoding="utf-8"))
        if latest is None:
            return None
        latest_document = normalize_action_document(latest)
        if latest_document is None:
            return None
        for queued_action in latest_document["actions"]:
            if str(queued_action["id"]) == action_id and queued_action.get("cancel_requested"):
                return str(queued_action.get("cancel_reason") or "action interrupted")
        return None

    try:
        result = driver.execute_action(action_type, params, cancel_check=cancel_check)
    except Exception as exc:
        result = f"Error: {type(exc).__name__}: {exc}"
    _log(f"Result: {result}")

    _publish_runtime_state(driver, robot_state_file)
    _log("ROBOT_STATE.md updated.")

    def finish(latest):
        for item in latest["actions"]:
            if item["id"] == action_id:
                item.update(status=infer_terminal_status(result), result=result,
                            finished_at=action_timestamp(),
                            duration_ms=round((time.monotonic() - started) * 1000))
                if item.get("cancel_requested"):
                    item["cancel_acknowledged_at"] = action_timestamp()
    update_action_document(action_file, finish)
    _log("ACTION.md updated.\n")


def main() -> None:
    from robot.drivers import list_drivers

    parser = argparse.ArgumentParser(
        description="Controller - Emerge-Policy robot control layer",
    )
    parser.add_argument(
        "--driver",
        default="libero_mujoco",
        help=f"Driver name (available: {', '.join(list_drivers())})",
    )
    parser.add_argument(
        "--workspace",
        default=None,
        help="Workspace directory",
    )
    parser.add_argument("--gui", action="store_true", help="Open 3-D viewer")
    parser.add_argument(
        "--interval", type=float, default=1.0, help="Poll interval (seconds)",
    )
    parser.add_argument(
        "--driver-config",
        default=None,
        help="Path to a JSON object file that will be passed through to the selected driver as keyword args.",
    )
    args = parser.parse_args()

    workspace = Path(args.workspace).expanduser().resolve() if args.workspace else None
    driver_config_path = Path(args.driver_config).expanduser().resolve() if args.driver_config else None
    try:
        driver_kwargs = load_driver_config(driver_config_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    robot_workspace = workspace or (Path.home() / ".Emerge" / "workspace")
    robot_state_file = robot_workspace / "ROBOT_STATE.md"

    if not robot_workspace.exists():
        print(f"Error: workspace not found: {robot_workspace}", file=sys.stderr)
        print("Run 'emerge onboard' first.", file=sys.stderr)
        sys.exit(1)

    watch_loop(
        robot_workspace,
        driver_name=args.driver,
        gui=args.gui,
        poll_interval=args.interval,
        driver_kwargs=driver_kwargs,
        robot_state_file=robot_state_file,
    )


if __name__ == "__main__":
    main()

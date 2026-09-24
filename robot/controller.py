#!/usr/bin/env python3
"""
robot/controller.py

Controller — polls ACTION.md for commands, dispatches them to the
active driver, and writes updated state back to ROBOT_STATE.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from Emerge.runtime.controller_control import (
    control_lock,
    expired,
    read_json,
    request_file,
    result_file,
    write_result,
)
from Emerge.runtime.storage import WorkspaceLease, atomic_json
from Emerge.utils.action_queue import (
    action_timestamp,
    first_pending_action,
    infer_terminal_status,
    normalize_action_document,
    parse_action_markdown,
    update_action_document,
)
from robot.drivers.base_driver import BaseDriver
from robot.mujoco_simulation.scene_io import (
    load_robot_state_doc,
    merge_robot_state_doc,
    save_robot_state_doc,
)


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[Controller {ts}] {msg}", flush=True)


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
        robots=runtime_state.get("robots"),
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
        raise FileNotFoundError(f"driver profile not found: {src}")


def _driver_kwargs_for_workspace(
    driver_name: str,
    workspace: Path,
    driver_kwargs: dict[str, object] | None,
) -> dict[str, object]:
    resolved = dict(driver_kwargs or {})
    if driver_name == "libero_mujoco":
        resolved["workspace"] = str(workspace)
    return resolved


class Controller:
    """Serial driver lifecycle: actions and environment changes share one thread."""

    CLEANUP_TIMEOUT_S = 120.0

    def __init__(self, workspace, driver_name, gui=False, *, driver_config_path=None,
                 driver_kwargs=None, robot_state_file=None):
        self.workspace = workspace
        self.driver_name = driver_name
        self.gui = gui
        self.driver_config_path = driver_config_path
        self.driver_kwargs = driver_kwargs
        self.robot_state_file = robot_state_file or workspace / "ROBOT_STATE.md"
        self.driver = None
        self.ready = False
        self.active_request = None
        self.cleanup_deadline = 0.0
        self.scene_catalog = {"root": "", "current": None, "entries": []}
        self.publish_scene()

    def publish_scene(self):
        atomic_json(self.workspace / ".controller/scene.json", {
            **self.scene_catalog,
            "pid": os.getpid(),
            "driver": self.driver_name,
            "ready": self.ready,
            "current": self.scene_catalog["current"] if self.ready else None,
        })

    def load(self, *, operation="load", scene_id=None):
        from robot.drivers import load_driver

        configured = (
            load_driver_config(self.driver_config_path)
            if self.driver_config_path is not None else self.driver_kwargs
        )
        resolved = _driver_kwargs_for_workspace(self.driver_name, self.workspace, configured)
        self.ready = False
        self.publish_scene()
        try:
            self.driver = load_driver(self.driver_name, gui=self.gui, **resolved)
            self.scene_catalog = self.driver.get_scene_catalog()
            self.publish_scene()
            _install_profile(self.driver, self.workspace)
            if operation == "reset":
                self.driver.reset_environment()
            elif operation == "switch_scene":
                self.driver.switch_scene(scene_id)
            else:
                self.driver.load_environment()
            _publish_runtime_state(self.driver, self.robot_state_file)
            self.scene_catalog = self.driver.get_scene_catalog()
        except Exception:
            try:
                self.close()
            except Exception as exc:
                _log(f"Driver cleanup failed: {exc}")
            raise
        self.ready = True
        self.publish_scene()
        _log("Environment loaded")

    def close(self):
        self.ready = False
        self.publish_scene()
        if self.driver is not None:
            self.driver.close()
            self.driver = None

    def poll(self):
        if self.active_request is None:
            with control_lock(self.workspace):
                for path in sorted((self.workspace / ".controller/requests").glob("*.json")):
                    request = read_json(path)
                    request_id = path.stem
                    # Any persisted result means this ID has already been claimed.
                    if read_json(result_file(self.workspace, request_id)) is not None:
                        continue
                    if request["phase"] != "requested":
                        continue
                    if expired(request):
                        write_result(self.workspace, request_id, "expired")
                        continue
                    if request["operation"] == "switch_scene":
                        if request.get("scene_id") not in {entry["id"] for entry in self.scene_catalog["entries"]}:
                            write_result(self.workspace, request_id, "failed", {
                                "code": "invalid_scene",
                                "message": "Select a scene from the driver's catalog.",
                            })
                            continue
                    elif request["operation"] != "reset":
                        write_result(self.workspace, request_id, "failed", {
                            "code": "unknown_operation", "message": "Unknown controller operation.",
                        })
                        continue
                    write_result(self.workspace, request_id, "stopping")
                    self.active_request = request_id
                    break
            if self.active_request is not None:
                try:
                    self.close()
                except Exception as exc:
                    write_result(self.workspace, self.active_request, "failed",
                                 {"code": "close_failed", "message": str(exc)})
                    self.active_request = None
                    return
                write_result(self.workspace, self.active_request, "can_clean")
                self.cleanup_deadline = time.monotonic() + self.CLEANUP_TIMEOUT_S

        if self.active_request is not None:
            request_id = self.active_request
            request = read_json(request_file(self.workspace, request_id))
            phase = request["phase"]
            if phase == "cleanup_failed":
                write_result(self.workspace, request_id, "failed", request["error"])
                self.active_request = None
            elif phase == "cleanup_completed":
                write_result(self.workspace, request_id, "loading")
                try:
                    if request["operation"] == "switch_scene":
                        self.load(operation="switch_scene", scene_id=request["scene_id"])
                    elif request["operation"] == "reset":
                        self.load(operation="reset")
                except Exception as exc:
                    write_result(self.workspace, request_id, "failed",
                                 {"code": "load_failed", "message": str(exc)})
                else:
                    write_result(self.workspace, request_id, "succeeded")
                self.active_request = None
            elif time.monotonic() >= self.cleanup_deadline:
                write_result(self.workspace, request_id, "failed",
                             {"code": "cleanup_timeout", "message": "Emerge did not confirm workspace cleanup"})
                self.active_request = None
            return

        if self.ready:
            _poll_once(self.driver, self.workspace / "ACTION.md", self.robot_state_file)


def watch_loop(
    workspace: Path,
    driver_name: str = "simulation",
    gui: bool = False,
    poll_interval: float = 1.0,
    *,
    driver_kwargs: dict[str, object] | None = None,
    driver_config_path: Path | None = None,
    robot_state_file: Path | None = None,
) -> None:
    """Load a driver, install its profile, then poll ACTION.md forever."""
    robot_state_file = robot_state_file or (workspace / "ROBOT_STATE.md")

    _log(f"Workspace : {workspace}")
    _log(f"Driver    : {driver_name}")
    _log(f"GUI       : {gui}")
    _log(f"State File: {robot_state_file}")
    if driver_kwargs:
        _log(f"DriverCfg : {json.dumps(driver_kwargs, ensure_ascii=False, sort_keys=True)}")

    # A workspace has one controller, independent of the AgentRuntime lease.
    with WorkspaceLease(workspace / ".controller"):
        interrupted = False
        with control_lock(workspace):
            for path in (workspace / ".controller/results").glob("*.json"):
                result = read_json(path)
                if result["status"] in {"stopping", "can_clean", "loading"}:
                    interrupted = True
                    write_result(workspace, path.stem, "failed",
                                  {"code": "controller_restarted", "message": "Controller exited before confirming environment change"})
        controller = Controller(
            workspace, driver_name, gui, driver_config_path=driver_config_path,
            driver_kwargs=driver_kwargs, robot_state_file=robot_state_file,
        )
        try:
            if not interrupted:
                controller.load()
            else:
                _log("Previous environment change interrupted; awaiting a new request.")
            _log("Watching ACTION.md and environment requests ... Ctrl+C to stop.\n")
            while True:
                controller.poll()
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            _log("Shutdown.")
        finally:
            controller.close()


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
        driver_config_path=driver_config_path,
        robot_state_file=robot_state_file,
    )


if __name__ == "__main__":
    main()

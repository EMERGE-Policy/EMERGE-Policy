"""Process, port, and workspace helpers for native BEHAVIOR evaluation."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, BinaryIO, Callable
from urllib import request as urllib_request
from urllib.parse import urlsplit, urlunsplit

SCRIPT_DIR = Path(__file__).resolve().parent
EMERGE_ROOT = SCRIPT_DIR.parents[1]
WORKSPACE_ROOT = EMERGE_ROOT.parent

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def require_path(path: Path | str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def require_executable(path: Path | str, label: str) -> Path:
    """Validate an interpreter without resolving its virtualenv symlink."""
    expanded = Path(path).expanduser().absolute()
    if not expanded.exists():
        raise FileNotFoundError(f"{label} not found: {expanded}")
    if not os.access(expanded, os.X_OK):
        raise PermissionError(f"{label} is not executable: {expanded}")
    return expanded


def assert_fresh_output_dir(output_dir: Path, *, allow_existing: bool = False) -> Path:
    """Refuse to write an episode into a directory that already holds results."""
    resolved = Path(output_dir).expanduser().resolve()
    if resolved.exists() and not allow_existing:
        offenders = [
            name
            for name in ("BEHAVIOR_STATUS.json", "agent_episode_result.json", "json")
            if (resolved / name).exists()
        ]
        if offenders:
            raise FileExistsError(
                f"output directory already holds episode artifacts "
                f"({', '.join(offenders)}): {resolved}. Use a new --output-dir."
            )
    return resolved


# --------------------------------------------------------------------------- #
# Ports and processes
# --------------------------------------------------------------------------- #


def assert_port_available(port: int, *, host: str = "127.0.0.1", label: str = "service") -> None:
    """Fail before spawning when something is already listening on ``port``.

    An HTTP health check cannot distinguish this run's process from a leftover
    one, so ownership is asserted by trying to bind instead.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, int(port)))
        except OSError as exc:
            raise RuntimeError(
                f"port {port} is already in use, so the {label} cannot be started: {exc}"
            ) from exc


def behavior_environment() -> dict[str, str]:
    """Let the pip-installed Isaac Sim package configure its own paths."""
    environment = dict(os.environ)
    for name in ("ISAAC_PATH", "ISAACSIM_PATH", "EXP_PATH", "CARB_APP_PATH"):
        environment.pop(name, None)
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    return environment


def spawn_process(
    command: list[str],
    *,
    cwd: Path,
    log: BinaryIO,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )


def _group_exists(pid: int) -> bool:
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _signal_group(pid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        os.killpg(pid, sig)


def stop_process_group(
    process: subprocess.Popen[bytes] | None,
    *,
    timeout_s: float = 15.0,
    kill_grace_s: float = 5.0,
) -> None:
    """Terminate a whole process group, even after its leader has exited.

    ``start_new_session=True`` makes the leader's pid the group id, so the group
    can still be signalled after the leader was reaped.  Returning early on
    ``poll() is not None`` is exactly the leak this function removes.
    """
    if process is None:
        return
    pid = process.pid
    _signal_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                process.wait(timeout=0.2)
        if not _group_exists(pid):
            return
        time.sleep(0.1)
    _signal_group(pid, signal.SIGKILL)
    kill_deadline = time.monotonic() + kill_grace_s
    while time.monotonic() < kill_deadline:
        if not _group_exists(pid):
            break
        time.sleep(0.1)
    with contextlib.suppress(Exception):
        process.wait(timeout=kill_grace_s)


def wait_for_health(
    port: int,
    process: subprocess.Popen[bytes],
    timeout_s: float,
    *,
    host: str = "127.0.0.1",
    label: str = "service",
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"{label} on port {port} exited early with code {process.returncode}"
            )
        if health_ok(port, host=host):
            return
        time.sleep(1)
    raise TimeoutError(f"{label} on port {port} was not ready after {timeout_s:.0f}s")


def health_ok(port: int, *, host: str = "127.0.0.1") -> bool:
    opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
    try:
        with opener.open(f"http://{host}:{port}/healthz", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def service_health_ok(url: str, *, timeout_s: float = 2.0) -> bool:
    """Probe the ``/healthz`` endpoint corresponding to an HTTP/WS service URL."""
    parts = urlsplit(str(url))
    if parts.scheme not in {"ws", "wss", "http", "https"} or not parts.netloc:
        raise ValueError(f"invalid service URL: {url!r}")
    health_url = urlunsplit(
        (
            "https" if parts.scheme in {"wss", "https"} else "http",
            parts.netloc,
            "/healthz",
            "",
            "",
        )
    )
    opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
    try:
        with opener.open(health_url, timeout=float(timeout_s)) as response:
            return response.status == 200
    except Exception:
        return False


def wait_for_process(
    process: subprocess.Popen[bytes],
    timeout_s: float,
    label: str,
) -> int:
    try:
        return process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"{label} did not finish after {timeout_s:.0f}s") from exc


# --------------------------------------------------------------------------- #
# Workspace
# --------------------------------------------------------------------------- #


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def wait_for_condition(
    predicate: Callable[[], bool],
    *,
    timeout_s: float,
    label: str,
    poll_interval_s: float = 0.2,
    processes: list[subprocess.Popen[bytes] | None] | None = None,
) -> None:
    """Poll ``predicate`` until it holds, failing fast when a process dies."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for process in processes or []:
            if process is not None and process.poll() is not None:
                raise RuntimeError(
                    f"a supervised process exited early with code {process.returncode} "
                    f"while waiting for {label}"
                )
        if predicate():
            return
        time.sleep(poll_interval_s)
    raise TimeoutError(f"timed out after {timeout_s:.0f}s while waiting for {label}")


def prepare_workspace(
    workspace: Path,
    *,
    task_name: str,
    instruction: str,
) -> Path:
    """Create an episode workspace with an empty queue and a neutral state doc."""
    from robot.mujoco_simulation.scene_io import (  # local import: keeps stdlib-only importers light
        default_robot_state_doc,
        save_robot_state_doc,
    )

    workspace = Path(workspace).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    state = default_robot_state_doc()
    state["task"] = {
        "benchmark": "BEHAVIOR-1K",
        "name": task_name,
        "instruction": instruction,
    }
    save_robot_state_doc(workspace / "ROBOT_STATE.md", state)
    (workspace / "ACTION.md").write_text("", encoding="utf-8")
    return workspace


def robot_state_section(workspace: Path, robot_id: str = "behavior_r1pro") -> dict[str, Any]:
    from robot.mujoco_simulation.scene_io import load_robot_state_doc

    state = load_robot_state_doc(Path(workspace) / "ROBOT_STATE.md")
    robots = state.get("robots") if isinstance(state, dict) else None
    robot = robots.get(robot_id) if isinstance(robots, dict) else None
    return dict(robot) if isinstance(robot, dict) else {}

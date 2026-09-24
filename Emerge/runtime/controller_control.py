"""File based environment control channel shared by Emerge and the controller."""

from __future__ import annotations

import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from Emerge.runtime.storage import atomic_json

SCHEMA_VERSION = "Emerge.controller.v1"
REQUEST_PHASES = {"requested", "cleanup_completed", "cleanup_failed"}
RESULT_STATUSES = {"stopping", "can_clean", "loading", "succeeded", "failed", "expired"}


@contextmanager
def control_lock(workspace: Path):
    """Serialize request claiming and expiry; never hold this while loading."""
    directory = workspace / ".controller"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".control.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def new_reset_request(expires_in_s: float = 15.0) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": uuid4().hex,
        "operation": "reset",
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=expires_in_s)).isoformat(),
        "phase": "requested",
    }


def new_scene_request(scene_id: str, expires_in_s: float = 15.0) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": uuid4().hex,
        "operation": "switch_scene",
        "scene_id": scene_id,
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=expires_in_s)).isoformat(),
        "phase": "requested",
    }


def request_file(workspace: Path, request_id: str) -> Path:
    return workspace / ".controller" / "requests" / f"{request_id}.json"


def result_file(workspace: Path, request_id: str) -> Path:
    return workspace / ".controller" / "results" / f"{request_id}.json"


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    return value


def write_request(workspace: Path, request: dict[str, Any]) -> None:
    atomic_json(request_file(workspace, request["request_id"]), request)


def update_request(
    workspace: Path,
    request_id: str,
    phase: str,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    if phase not in REQUEST_PHASES:
        raise ValueError(f"unknown controller request phase: {phase}")
    request = read_json(request_file(workspace, request_id))
    if request is None:
        raise FileNotFoundError(f"controller request not found: {request_id}")
    request["phase"] = phase
    request["updated_at"] = datetime.now(timezone.utc).isoformat()
    if error is not None:
        request["error"] = error
    write_request(workspace, request)
    return request


def write_result(
    workspace: Path,
    request_id: str,
    status: str,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    if status not in RESULT_STATUSES:
        raise ValueError(f"unknown controller result status: {status}")
    request = read_json(request_file(workspace, request_id))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "operation": request["operation"] if request else "reset",
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if error is not None:
        result["error"] = error
    atomic_json(result_file(workspace, request_id), result)
    return result


def expired(request: dict[str, Any]) -> bool:
    expires_at = datetime.fromisoformat(request["expires_at"])
    return datetime.now(timezone.utc) >= expires_at

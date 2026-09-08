from __future__ import annotations

import json
import uuid
import os
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Any


_FENCE_OPEN = "```json"
_FENCE_CLOSE = "```"
ACTION_QUEUE_SCHEMA_VERSION = "Emerge.action_queue.v1"


def parse_action_markdown(content: str) -> dict[str, Any] | None:
    content = content.strip()
    if not content:
        return None
    try:
        _, json_block = content.split(_FENCE_OPEN, 1)
        json_block, _ = json_block.split(_FENCE_CLOSE, 1)
        payload = json.loads(json_block)
    except (ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def normalize_action_document(payload: dict[str, Any]) -> dict[str, Any] | None:
    actions = payload.get("actions")
    if isinstance(actions, list):
        normalized_actions: list[dict[str, Any]] = []
        for item in actions:
            normalized = normalize_action_item(item)
            if normalized is None:
                return None
            normalized_actions.append(normalized)
        return {
            "schema_version": str(payload.get("schema_version") or ACTION_QUEUE_SCHEMA_VERSION),
            "actions": normalized_actions,
        }

    return None


def normalize_action_item(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    action_type = str(payload.get("action_type") or "").strip()
    if not action_type:
        return None
    parameters = payload.get("parameters")
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        return None
    status = str(payload.get("status") or "pending").strip().lower() or "pending"
    item = {
        "id": str(payload.get("id") or _next_action_id()),
        "action_type": action_type,
        "parameters": parameters,
        "status": status,
    }
    for key in ("started_at", "finished_at", "duration_ms", "cancel_requested_at",
                "cancel_acknowledged_at", "run_id"):
        if key in payload:
            item[key] = payload[key]
    if "result" in payload:
        item["result"] = payload["result"]
    if payload.get("cancel_requested"):
        item["cancel_requested"] = True
        item["cancel_reason"] = str(payload.get("cancel_reason") or "action interrupted")
    return item


def empty_action_document() -> dict[str, Any]:
    return {
        "schema_version": ACTION_QUEUE_SCHEMA_VERSION,
        "actions": [],
    }


def first_pending_action(document: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    for index, item in enumerate(document.get("actions", [])):
        if str(item.get("status") or "pending").strip().lower() == "pending":
            return index, item
    return None


def first_active_action(document: dict[str, Any]) -> tuple[int, dict[str, Any]] | None:
    return next(((i, item) for i, item in enumerate(document.get("actions", []))
                 if item.get("status", "pending") in {"pending", "running"}), None)


def action_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_action_document(path: Path, mutate) -> dict[str, Any]:
    """Serialize read/modify/write across Agent and Controller; atomically publish."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(".ACTION.lock").open("a+b") as lock:
        if os.name == "nt":
            import msvcrt
            lock.write(b"0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists() and path.read_text(encoding="utf-8").strip():
            payload = parse_action_markdown(path.read_text(encoding="utf-8"))
            document = normalize_action_document(payload) if payload else None
            if document is None:
                raise ValueError("ACTION.md contains invalid action data")
        else:
            document = empty_action_document()
        replacement = mutate(document)
        if replacement is not None:
            document = replacement
        fd, name = tempfile.mkstemp(prefix=".ACTION.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as output:
                output.write(dump_action_document(document))
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)
        return document


def pending_action_type(document: dict[str, Any]) -> str | None:
    pending = first_active_action(document)
    if pending is None:
        return None
    _, item = pending
    return str(item.get("action_type") or "unknown")


def append_action(document: dict[str, Any], *, action_type: str, parameters: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_action_document(document) or empty_action_document()
    normalized.setdefault("actions", []).append(
        {
            "id": _next_action_id(),
            "action_type": action_type,
            "parameters": parameters,
            "status": "pending",
        }
    )
    return normalized


def dump_action_document(document: dict[str, Any]) -> str:
    normalized = normalize_action_document(document) or empty_action_document()
    return (
        _FENCE_OPEN + "\n"
        + json.dumps(normalized, indent=2, ensure_ascii=False) + "\n"
        + _FENCE_CLOSE + "\n"
    )


def infer_terminal_status(result: str) -> str:
    lowered = result.strip().lower()
    if (
        lowered.startswith("error:")
        or lowered.startswith("failed:")
        or " failed" in lowered
        or lowered.startswith("unknown action")
    ):
        return "failed"
    if (
        "cancelled" in lowered
        or "canceled" in lowered
        or "interrupted" in lowered
        or lowered.endswith("stopped.")
    ):
        return "cancelled"
    return "completed"


def _next_action_id() -> str:
    return uuid.uuid4().hex[:12]

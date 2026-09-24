"""Structured ``ROBOT_STATE.md`` persistence helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_FENCE_OPEN = "```json"
_FENCE_CLOSE = "```"
_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


def _load_json_block(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8")
    match = _BLOCK_RE.search(content)
    if not match:
        return {}
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def load_robot_state_doc(path: Path) -> dict[str, Any]:
    """Return the full robot-state document from ``ROBOT_STATE.md``."""
    return _load_json_block(path)


def default_robot_state_doc() -> dict[str, Any]:
    """Return the minimal structured robot-state document."""
    return {
        "schema_version": "Emerge.robot_state.v1",
        "robots": {},
    }


def save_robot_state_doc(path: Path, robot_state: dict[str, Any]) -> None:
    """Write a full robot-state document to ``ROBOT_STATE.md``."""
    state_json = json.dumps(robot_state, indent=2, ensure_ascii=False)
    content = (
        "# Robot State\n\n"
        "Auto-updated by Controller with the current runtime state.\n\n"
        "Agent usage:\n"
        "- The robot state initially included in the agent context is only a snapshot.\n"
        "- After every `execute_robot_action` call, the agent MUST use `read_file` to re-read "
        "`ROBOT_STATE.md` before checking task progress or choosing the next action.\n"
        "- Do not rely on the earlier copy already present in the conversation context, because "
        "Controller may have updated the file after the action.\n\n"
        f"{_FENCE_OPEN}\n{state_json}\n{_FENCE_CLOSE}\n"
    )
    path.write_text(content, encoding="utf-8")


def merge_robot_state_doc(
    existing: dict[str, Any] | None,
    *,
    robots: dict[str, Any] | None = None,
    map_data: dict[str, Any] | None = None,
    tf_data: dict[str, Any] | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Merge robot-state partitions while preserving unrelated sections."""
    base = default_robot_state_doc()
    if isinstance(existing, dict):
        base.update(existing)
    base.pop("updated_at", None)

    if robots is not None:
        merged_robots = dict(base.get("robots", {}))
        merged_robots.update(robots)
        base["robots"] = merged_robots
    if map_data is not None:
        base["map"] = map_data
    if tf_data is not None:
        base["tf"] = tf_data
    if updated_at is not None:
        base["updated_at"] = updated_at

    return base

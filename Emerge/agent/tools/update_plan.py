"""Validated PLAN.md updates for embodied task decomposition."""

from __future__ import annotations

import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from Emerge.base import Tool


_VALID_STATUSES = {"pending", "active", "done"}
_UPDATE_KINDS = {"new_mission", "progress", "replan"}
_RETRY_LIMIT = 3
_BRANCH_DEPTH_LIMIT = 2


class UpdatePlanTool(Tool):
    """Validate and persist a complete task-decomposition snapshot."""

    def __init__(
        self,
        workspace: Path,
        clock: Callable[[], datetime] | None = None,
    ):
        self.workspace = workspace
        self.plan_file = workspace / "PLAN.md"
        self._clock = clock or (lambda: datetime.now().astimezone())

    @property
    def name(self) -> str:
        return "update_plan"

    @property
    def description(self) -> str:
        return (
            "Validate and replace PLAN.md with a complete plan snapshot. "
            "This tool persists decisions; it does not decide transitions or execute the plan. "
            "Use update_kind='new_mission' only to start a different mission "
            "(or restart a completed one), 'progress' for normal pointer/status/retry/branch "
            "changes without rewriting the main line, and 'replan' when intentionally "
            "rewriting the main line. "
            "Retries may never decrease for retained subgoals. Always pass the full desired state."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "update_kind": {
                    "type": "string",
                    "enum": sorted(_UPDATE_KINDS),
                    "description": "Why this snapshot is being written.",
                },
                "mission": {
                    "type": "string",
                    "minLength": 1,
                    "description": "One-line overall task objective.",
                },
                "main_line": {
                    "type": "array",
                    "description": "Complete ordered list of coarse main-line subgoals.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "minimum": 1},
                            "subgoal": {"type": "string", "minLength": 1},
                            "done_criterion": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Observable completion condition.",
                            },
                            "status": {
                                "type": "string",
                                "enum": sorted(_VALID_STATUSES),
                            },
                            "retries": {"type": "integer", "minimum": 0},
                        },
                        "required": [
                            "id",
                            "subgoal",
                            "done_criterion",
                            "status",
                            "retries",
                        ],
                    },
                },
                "pointer": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Current main-line subgoal id. Use len(main_line)+1 after all "
                        "subgoals are done."
                    ),
                },
                "branch_stack": {
                    "type": "array",
                    "description": (
                        "Complete recovery stack in bottom-to-top order; the final item "
                        "is the active top."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "subgoal": {"type": "string", "minLength": 1},
                            "done_criterion": {
                                "type": "string",
                                "minLength": 1,
                            },
                        },
                        "required": ["id", "subgoal", "done_criterion"],
                    },
                },
                "change_summary": {
                    "type": "string",
                    "minLength": 1,
                    "description": "One-line explanation recorded with the timestamp.",
                },
            },
            "required": [
                "update_kind",
                "mission",
                "main_line",
                "pointer",
                "branch_stack",
                "change_summary",
            ],
        }

    async def execute(
        self,
        update_kind: str,
        mission: str,
        main_line: list[dict[str, Any]],
        pointer: int,
        branch_stack: list[dict[str, Any]],
        change_summary: str,
        **kwargs: Any,
    ) -> str:
        proposed = {
            "mission": mission.strip(),
            "main_line": main_line,
            "pointer": pointer,
            "branch_stack": branch_stack,
        }

        errors = self._validate_snapshot(proposed, update_kind, change_summary)
        if errors:
            return "Error: PLAN.md was not changed:\n- " + "\n- ".join(errors)

        previous, parse_error = self._load_previous()
        transition_errors = self._validate_transition(
            previous=previous,
            parse_error=parse_error,
            proposed=proposed,
            update_kind=update_kind,
        )
        if transition_errors:
            return "Error: PLAN.md was not changed:\n- " + "\n- ".join(transition_errors)

        timestamp = self._clock().isoformat(timespec="seconds")
        content = self._render(
            proposed,
            timestamp=timestamp,
            update_kind=update_kind,
            change_summary=change_summary.strip(),
        )
        self._atomic_write(content)

        warnings = self._guardrail_warnings(proposed)
        logger.info(
            "Updated {} ({}) at {}: {}",
            self.plan_file,
            update_kind,
            timestamp,
            change_summary.strip(),
        )

        result = f"Successfully updated PLAN.md at {timestamp} ({update_kind})."
        if warnings:
            result += (
                "\n\nGUARDRAIL TRIGGERED — stop blind retries. Replan or report that "
                "intervention is needed:\n- "
                + "\n- ".join(warnings)
            )
        return result

    def _validate_snapshot(
        self,
        snapshot: dict[str, Any],
        update_kind: str,
        change_summary: str,
    ) -> list[str]:
        errors: list[str] = []
        mission = snapshot["mission"]
        main_line = snapshot["main_line"]
        pointer = snapshot["pointer"]
        branch_stack = snapshot["branch_stack"]

        if update_kind not in _UPDATE_KINDS:
            errors.append(f"update_kind must be one of {sorted(_UPDATE_KINDS)}")
        errors.extend(self._validate_one_line(mission, "mission"))
        errors.extend(self._validate_one_line(change_summary.strip(), "change_summary"))

        if not main_line:
            errors.append("main_line must contain at least one subgoal")
        elif len(main_line) > 50:
            errors.append("main_line cannot contain more than 50 subgoals")

        seen_subgoal_ids: set[int] = set()
        for index, item in enumerate(main_line, start=1):
            label = f"main_line[{index - 1}]"
            if not isinstance(item, dict):
                errors.append(f"{label} must be an object")
                continue

            subgoal_id = item.get("id")
            if subgoal_id != index:
                errors.append(
                    f"{label}.id must be {index}; main-line ids must be consecutive from 1"
                )
            if isinstance(subgoal_id, int):
                if subgoal_id in seen_subgoal_ids:
                    errors.append(f"duplicate main-line id {subgoal_id}")
                seen_subgoal_ids.add(subgoal_id)

            errors.extend(
                self._validate_one_line(item.get("subgoal"), f"{label}.subgoal")
            )
            errors.extend(
                self._validate_one_line(
                    item.get("done_criterion"),
                    f"{label}.done_criterion",
                )
            )

            status = item.get("status")
            if status not in _VALID_STATUSES:
                errors.append(f"{label}.status must be one of {sorted(_VALID_STATUSES)}")
            retries = item.get("retries")
            if (
                not isinstance(retries, int)
                or isinstance(retries, bool)
                or retries < 0
            ):
                errors.append(f"{label}.retries must be a non-negative integer")
            if status == "done" and not str(item.get("done_criterion", "")).strip():
                errors.append(f"{label} cannot be marked done without a Done Criterion")

        if not isinstance(pointer, int) or isinstance(pointer, bool) or pointer < 1:
            errors.append("pointer must be a positive integer")
        elif main_line:
            max_pointer = len(main_line) + 1
            if pointer > max_pointer:
                errors.append(f"pointer must be between 1 and {max_pointer}")
            elif pointer == max_pointer:
                if any(item.get("status") != "done" for item in main_line):
                    errors.append(
                        "pointer may move past the main line only when every subgoal is done"
                    )
                if branch_stack:
                    errors.append(
                        "pointer may move past the main line only when the branch stack is empty"
                    )

        seen_branch_ids: set[str] = set()
        for index, item in enumerate(branch_stack):
            label = f"branch_stack[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{label} must be an object")
                continue
            branch_id = str(item.get("id", "")).strip()
            errors.extend(self._validate_one_line(branch_id, f"{label}.id"))
            errors.extend(
                self._validate_one_line(item.get("subgoal"), f"{label}.subgoal")
            )
            errors.extend(
                self._validate_one_line(
                    item.get("done_criterion"),
                    f"{label}.done_criterion",
                )
            )
            if branch_id in seen_branch_ids:
                errors.append(f"duplicate branch id {branch_id!r}")
            seen_branch_ids.add(branch_id)

        return errors

    @staticmethod
    def _validate_one_line(value: Any, label: str) -> list[str]:
        if not isinstance(value, str) or not value.strip():
            return [f"{label} must be a non-empty string"]
        if "\n" in value or "\r" in value:
            return [f"{label} must be one line"]
        if "|" in value:
            return [f"{label} cannot contain '|' because PLAN.md uses Markdown tables"]
        return []

    def _validate_transition(
        self,
        previous: dict[str, Any] | None,
        parse_error: str | None,
        proposed: dict[str, Any],
        update_kind: str,
    ) -> list[str]:
        if previous is None:
            if update_kind == "new_mission":
                return []
            detail = f" ({parse_error})" if parse_error else ""
            return [f"no readable existing plan{detail}; use update_kind='new_mission'"]

        old_mission = previous["mission"].strip()
        new_mission = proposed["mission"].strip()
        old_is_placeholder = old_mission.lower() in {"", "none"}
        old_complete = bool(previous["main_line"]) and all(
            item["status"] == "done" for item in previous["main_line"]
        )

        if update_kind == "new_mission":
            if not old_is_placeholder and old_mission == new_mission and not old_complete:
                return [
                    "new_mission cannot reset an incomplete mission with the same name; "
                    "use progress or replan"
                ]
            return []

        if old_mission != new_mission:
            return [
                f"{update_kind} cannot change Mission; use new_mission to start a different task"
            ]

        if update_kind == "progress":
            errors = self._validate_static_main_line(previous, proposed)
            errors.extend(self._validate_retry_monotonic(previous, proposed))
            return errors

        if update_kind == "replan":
            return self._validate_retry_monotonic(previous, proposed, retained_only=True)

        return []

    @staticmethod
    def _validate_static_main_line(
        previous: dict[str, Any],
        proposed: dict[str, Any],
    ) -> list[str]:
        old_items = previous["main_line"]
        new_items = proposed["main_line"]
        if len(old_items) != len(new_items):
            return ["progress cannot change main_line length; use replan"]

        errors: list[str] = []
        for old, new in zip(old_items, new_items):
            for field in ("id", "subgoal", "done_criterion"):
                if old[field] != new[field]:
                    errors.append(
                        f"progress cannot change main-line {field} for subgoal #{old['id']}; "
                        "use replan"
                    )
            if old["status"] == "done" and new["status"] != "done":
                errors.append(
                    f"progress cannot move completed subgoal #{old['id']} back to "
                    f"{new['status']!r}; use replan if the main line is no longer valid"
                )
        return errors

    @staticmethod
    def _validate_retry_monotonic(
        previous: dict[str, Any],
        proposed: dict[str, Any],
        retained_only: bool = False,
    ) -> list[str]:
        if retained_only:
            old_by_subgoal: dict[str, int] = {}
            for item in previous["main_line"]:
                key = item["subgoal"].strip().casefold()
                old_by_subgoal[key] = max(old_by_subgoal.get(key, 0), item["retries"])
            pairs = (
                (old_by_subgoal.get(item["subgoal"].strip().casefold()), item)
                for item in proposed["main_line"]
            )
        else:
            pairs = (
                (old["retries"], new)
                for old, new in zip(previous["main_line"], proposed["main_line"])
            )

        errors: list[str] = []
        for old_retries, new in pairs:
            if old_retries is not None and new["retries"] < old_retries:
                errors.append(
                    f"Retries for retained subgoal {new['subgoal']!r} cannot decrease "
                    f"from {old_retries} to {new['retries']}"
                )
        return errors

    def _load_previous(self) -> tuple[dict[str, Any] | None, str | None]:
        if not self.plan_file.exists():
            return None, None
        try:
            content = self.plan_file.read_text(encoding="utf-8")
        except OSError as exc:
            return None, f"could not read PLAN.md: {exc}"
        try:
            return self._parse(content), None
        except ValueError as exc:
            return None, str(exc)

    @classmethod
    def _parse(cls, content: str) -> dict[str, Any]:
        mission_match = re.search(
            r"^## Mission\s*$\n(.*?)(?=^## |\Z)",
            content,
            flags=re.MULTILINE | re.DOTALL,
        )
        if not mission_match:
            raise ValueError("missing Mission section")
        mission_lines = [
            line.strip() for line in mission_match.group(1).splitlines() if line.strip()
        ]
        if len(mission_lines) != 1:
            raise ValueError("Mission must contain exactly one non-empty line")

        main_match = re.search(
            r"^## Main Line\s*$\n(.*?)(?=^## |\Z)",
            content,
            flags=re.MULTILINE | re.DOTALL,
        )
        if not main_match:
            raise ValueError("missing Main Line section")
        main_line: list[dict[str, Any]] = []
        for raw_line in main_match.group(1).splitlines():
            line = raw_line.strip()
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) != 5 or cells[0] == "#" or set(cells[0]) <= {"-", ":"}:
                continue
            try:
                item = {
                    "id": int(cells[0]),
                    "subgoal": cells[1],
                    "done_criterion": cells[2],
                    "status": cells[3],
                    "retries": int(cells[4]),
                }
            except ValueError as exc:
                raise ValueError(f"invalid Main Line row: {line}") from exc
            if item["status"] not in _VALID_STATUSES:
                raise ValueError(f"invalid Main Line status: {item['status']!r}")
            main_line.append(item)
        if not main_line:
            raise ValueError("Main Line contains no readable subgoals")

        pointer_match = re.search(r"^current:\s*(\d+)\s*$", content, flags=re.MULTILINE)
        if not pointer_match:
            raise ValueError("missing numeric Pointer")

        branch_match = re.search(
            r"^## Branch Stack\s*$\n(.*?)(?=^## |\Z)",
            content,
            flags=re.MULTILINE | re.DOTALL,
        )
        if not branch_match:
            raise ValueError("missing Branch Stack section")
        branch_stack: list[dict[str, str]] = []
        for raw_line in branch_match.group(1).splitlines():
            line = raw_line.strip()
            if not line.startswith("- "):
                continue
            payload = line[2:].strip()
            if payload == "(empty)":
                continue
            cells = [cell.strip() for cell in payload.split("|")]
            if len(cells) != 3:
                raise ValueError(f"invalid Branch Stack row: {line}")
            branch_stack.append(
                {
                    "id": cells[0],
                    "subgoal": cells[1],
                    "done_criterion": cells[2],
                }
            )

        return {
            "mission": mission_lines[0],
            "main_line": main_line,
            "pointer": int(pointer_match.group(1)),
            "branch_stack": branch_stack,
        }

    @staticmethod
    def _render(
        snapshot: dict[str, Any],
        timestamp: str,
        update_kind: str,
        change_summary: str,
    ) -> str:
        lines = [
            "# Plan",
            "",
            (
                "Task decomposition state. Static main line of subgoals, plus a LIFO "
                "branch stack for recovery. Maintained through `update_plan`."
            ),
            "",
            "## Mission",
            "",
            snapshot["mission"],
            "",
            "## Main Line",
            "",
            "| # | Subgoal | Done Criterion | Status | Retries |",
            "|---|---------|----------------|--------|---------|",
        ]
        for item in snapshot["main_line"]:
            lines.append(
                f"| {item['id']} | {item['subgoal'].strip()} | "
                f"{item['done_criterion'].strip()} | {item['status']} | "
                f"{item['retries']} |"
            )

        lines.extend(["", "## Pointer", "", f"current: {snapshot['pointer']}", ""])
        lines.extend(["## Branch Stack", ""])
        if snapshot["branch_stack"]:
            for item in snapshot["branch_stack"]:
                lines.append(
                    f"- {item['id'].strip()} | {item['subgoal'].strip()} | "
                    f"{item['done_criterion'].strip()}"
                )
        else:
            lines.append("- (empty)")

        lines.extend(
            [
                "",
                "## Last Update",
                "",
                f"timestamp: {timestamp}",
                f"kind: {update_kind}",
                f"summary: {change_summary}",
                "",
            ]
        )
        return "\n".join(lines)

    def _atomic_write(self, content: str) -> None:
        self.plan_file.parent.mkdir(parents=True, exist_ok=True)
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.plan_file.parent,
                prefix=".PLAN.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_file.write(content)
                temp_name = temp_file.name
            Path(temp_name).replace(self.plan_file)
        finally:
            if temp_name:
                temp_path = Path(temp_name)
                if temp_path.exists():
                    temp_path.unlink()

    @staticmethod
    def _guardrail_warnings(snapshot: dict[str, Any]) -> list[str]:
        warnings = [
            (
                f"subgoal #{item['id']} {item['subgoal']!r} has "
                f"Retries={item['retries']} (limit {_RETRY_LIMIT})"
            )
            for item in snapshot["main_line"]
            if item["retries"] >= _RETRY_LIMIT
        ]
        depth = len(snapshot["branch_stack"])
        if depth > _BRANCH_DEPTH_LIMIT:
            warnings.append(
                f"Branch Stack depth is {depth} (limit {_BRANCH_DEPTH_LIMIT})"
            )
        return warnings

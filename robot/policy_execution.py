"""Shared result contract for model-backed robot action executors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class PolicyExecutionResult:
    """Result of one bounded policy execution request.

    ``success`` means that the bounded execution completed normally.
    ``task_success`` is reserved for the environment's authoritative success
    signal and is never inferred from exhausting a step budget.
    """

    success: bool
    total_steps: int
    reason: str
    error_message: str | None = None
    last_gripper_command: float | None = None
    task_success: bool = False
    search_decisions: tuple[dict[str, Any], ...] = ()

"""Structured visual task-verification result."""

from __future__ import annotations

import json
from typing import Any

from Emerge.base import Tool


_STATE_VALUES = {
    "satisfied": True,
    "not_satisfied": False,
    "uncertain": None,
}


class SubmitVerificationTool(Tool):
    """Store one evidence-backed verification of the current scene."""

    def __init__(self) -> None:
        self.last_result: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return "submit_task_verification"

    @property
    def description(self) -> str:
        return (
            "Submit the final visual verification for every condition required "
            "by the task. Use only visible evidence from the current views."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "predicates": {
                    "type": "array",
                    "description": (
                        "All visually observable conditions required for success"
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": (
                                    "Concise snake_case condition such as "
                                    "apple_inside_basket or apple_released"
                                ),
                                "minLength": 1,
                            },
                            "state": {
                                "type": "string",
                                "enum": list(_STATE_VALUES),
                                "description": (
                                    "satisfied for direct visible support, "
                                    "not_satisfied for direct contradiction, or "
                                    "uncertain when the views cannot decide"
                                ),
                            },
                            "evidence": {
                                "type": "string",
                                "description": "Brief visible evidence for the state",
                                "minLength": 1,
                            },
                        },
                        "required": ["name", "state", "evidence"],
                    },
                    "minItems": 1,
                },
                "scene_context": {
                    "type": "string",
                    "description": (
                        "One brief sentence about relevant nearby objects or "
                        "visible collateral effects"
                    ),
                    "minLength": 1,
                },
            },
            "required": ["predicates", "scene_context"],
        }

    def reset(self) -> None:
        self.last_result = None

    async def execute(
        self,
        predicates: list[dict[str, str]],
        scene_context: str,
    ) -> str:
        if not predicates:
            raise ValueError("At least one verification predicate is required")

        states = [predicate["state"] for predicate in predicates]
        if "not_satisfied" in states:
            outcome = "not_achieved"
        elif "uncertain" in states:
            outcome = "uncertain"
        else:
            outcome = "achieved"

        self.last_result = {
            "outcome": outcome,
            "predicates": [
                {
                    "name": predicate["name"],
                    "value": _STATE_VALUES[predicate["state"]],
                    "evidence": predicate["evidence"],
                }
                for predicate in predicates
            ],
            "scene_context": scene_context,
        }
        return json.dumps(self.last_result, ensure_ascii=False)

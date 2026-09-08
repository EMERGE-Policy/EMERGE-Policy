"""Final localization tool exposed to the Object Location Subagent."""

from __future__ import annotations

import json
from typing import Any

from Emerge.base import Tool
from Emerge.subagents.object_location.tools.location.engine import (
    LocationEngine,
)


class LocateCandidatesTool(Tool):
    """Turn agent-confirmed candidate masks into measured world geometry."""

    def __init__(self, engine: LocationEngine) -> None:
        self._engine = engine
        self.last_result: dict[str, Any] | None = None
        self.last_error: str | None = None

    @property
    def name(self) -> str:
        return "locate_candidates"

    @property
    def description(self) -> str:
        return (
            "Compute world geometry from candidate masks already returned by "
            "segment_candidates. Call this only after visually reviewing the "
            "full-view overlays and confirming which candidate is the requested "
            "semantic object. For every camera view, select the # instance whose "
            "mask covers that same physical object; ranks can change by view. "
            "This tool reuses cached "
            "VGGT geometry and SAM3 masks; it does not run either model again."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selections": {
                    "type": "array",
                    "description": (
                        "One visually confirmed candidate for each requested "
                        "object that requires a precise location."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "object_key": {
                                "type": "string",
                                "description": (
                                    "Requested semantic object name in snake_case"
                                ),
                                "minLength": 1,
                            },
                            "candidate_id": {
                                "type": "string",
                                "description": (
                                    "Candidate ID confirmed from the segmentation "
                                    "overlays"
                                ),
                                "minLength": 1,
                            },
                            "verified_instances": {
                                "type": "array",
                                "description": (
                                    "Camera views and per-view instance ranks whose "
                                    "reviewed masks cover this exact physical object. "
                                    "Do not assume #1 or #2 has the same identity "
                                    "across different views."
                                ),
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "view_name": {
                                            "type": "string",
                                            "minLength": 1,
                                        },
                                        "instance_index": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": 2,
                                        },
                                    },
                                    "required": ["view_name", "instance_index"],
                                },
                                "minItems": 2,
                            },
                        },
                        "required": [
                            "object_key",
                            "candidate_id",
                            "verified_instances",
                        ],
                    },
                    "minItems": 1,
                }
            },
            "required": ["selections"],
        }

    def reset(self) -> None:
        self.last_result = None
        self.last_error = None

    async def execute(
        self,
        selections: list[dict[str, Any]],
    ) -> str:
        self.last_error = None
        try:
            self.last_result = self._engine.locate_candidates(selections)
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            raise
        return json.dumps(self.last_result, ensure_ascii=False)

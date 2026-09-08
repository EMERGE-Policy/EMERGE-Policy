"""Multi-camera observation tool for visual task verification."""

from __future__ import annotations

import json
from typing import Any

from Emerge.base import Tool
from Emerge.subagents import ImageContent, SubagentToolResult, TextContent
from Emerge.subagents.task_verification.tools.observation import (
    ObservationStore,
)


class ObserveSceneTool(Tool):
    def __init__(self, store: ObservationStore) -> None:
        self._store = store

    @property
    def name(self) -> str:
        return "observe_scene"

    @property
    def description(self) -> str:
        return (
            "Load every current camera view for visual task verification. "
            "Call this before judging any requested object state or relation."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self) -> SubagentToolResult:
        observation = self._store.load()
        content = []
        for view in observation.views:
            content.extend(
                (
                    TextContent(f"Camera view: {view.name}"),
                    ImageContent(view.image_data_url(), detail="high"),
                )
            )
        summary = {
            "reference_view": observation.reference_view,
            "views": [view.name for view in observation.views],
        }
        return SubagentToolResult(
            text=json.dumps(summary, ensure_ascii=False),
            content=tuple(content),
        )

"""Object Location Subagent implementation."""

from __future__ import annotations

from typing import Any

from Emerge.subagents.base import BaseSubagent
from Emerge.subagents.context import SubagentRunContext
from Emerge.subagents.models import SubagentResult, SubagentTask
from Emerge.subagents.object_location.tools.location import (
    LocateCandidatesTool,
    LocationEngine,
)
from Emerge.subagents.object_location.tools.observation import ObservationStore
from Emerge.subagents.object_location.tools.segment_candidates import (
    SegmentCandidatesTool,
)


class ObjectLocationSubagent(BaseSubagent):
    """Understand camera views and return measured object locations."""

    def __init__(
        self,
        *,
        observation_store: ObservationStore,
        location_engine: LocationEngine,
        candidate_tool: SegmentCandidatesTool,
        location_tool: LocateCandidatesTool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._observation_store = observation_store
        self._location_engine = location_engine
        self._candidate_tool = candidate_tool
        self._location_tool = location_tool

    @property
    def last_diagnostics(self) -> dict[str, Any] | None:
        """Full localization details for the standalone debugging entrypoint."""
        return self._location_tool.last_result

    async def _run(self, task: SubagentTask) -> SubagentResult:
        self._observation_store.reset()
        self._location_engine.reset()
        self._candidate_tool.reset()
        self._location_tool.reset()
        return await super()._run(task)

    def _tool_work_complete(self) -> bool:
        """Finish immediately once measured geometry is available."""
        return self._location_tool.last_result is not None

    def build_result(
        self,
        *,
        task: SubagentTask,
        content: str | None,
        context: SubagentRunContext,
        metadata: dict[str, Any],
    ) -> SubagentResult:
        tool_error = (
            self._candidate_tool.last_error or self._location_tool.last_error
        )
        if tool_error:
            return SubagentResult.failure(
                task,
                self.name,
                tool_error,
                metadata=metadata,
            )
        scene_context = (
            content.strip()
            if content
            else "No qualitative scene context was reported."
        )
        objects = _compact_objects(self._location_tool.last_result)
        return SubagentResult.success(
            task,
            self.name,
            _compact_summary(objects),
            output={
                "objects": objects,
                "scene_context": scene_context,
            },
            metadata=metadata,
        )


def _compact_objects(
    diagnostics: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if diagnostics is None:
        return []

    compact = []
    for result in diagnostics["objects"]:
        item = {
            "name": result["name"],
            "found": bool(result["found"]),
        }
        if result["found"]:
            item.update(
                {
                    "frame": result["frame"],
                    "position_m": result["position_m"],
                    "size_m": result["size_m"],
                    "rpy_rad": result["rpy_rad"],
                }
            )
        else:
            item["failure_reason"] = result["failure_reason"]
        compact.append(item)
    return compact


def _compact_summary(objects: list[dict[str, Any]]) -> str:
    located = [item["name"] for item in objects if item["found"]]
    unlocated = [item["name"] for item in objects if not item["found"]]
    parts = []
    if located:
        parts.append(f"Localized: {', '.join(located)}.")
    if unlocated:
        parts.append(f"Unlocated: {', '.join(unlocated)}.")
    return " ".join(parts) or "No object was localized."

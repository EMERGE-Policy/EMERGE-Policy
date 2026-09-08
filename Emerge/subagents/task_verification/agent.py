"""Task Verification Subagent implementation."""

from __future__ import annotations

from typing import Any

from Emerge.subagents.base import BaseSubagent
from Emerge.subagents.context import SubagentRunContext
from Emerge.subagents.models import SubagentResult, SubagentTask
from Emerge.subagents.task_verification.tools import (
    ObservationStore,
    SubmitVerificationTool,
)


class TaskVerificationSubagent(BaseSubagent):
    """Verify visible task outcomes from current multi-camera images."""

    def __init__(
        self,
        *,
        observation_store: ObservationStore,
        submission_tool: SubmitVerificationTool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._observation_store = observation_store
        self._submission_tool = submission_tool

    async def _run(self, task: SubagentTask) -> SubagentResult:
        self._observation_store.reset()
        self._submission_tool.reset()
        return await super()._run(task)

    def _tool_work_complete(self) -> bool:
        return self._submission_tool.last_result is not None

    def build_result(
        self,
        *,
        task: SubagentTask,
        content: str | None,
        context: SubagentRunContext,
        metadata: dict[str, Any],
    ) -> SubagentResult:
        verification = self._submission_tool.last_result
        if verification is None:
            return SubagentResult.failure(
                task,
                self.name,
                "No visual task verification was submitted.",
                metadata=metadata,
            )

        return SubagentResult.success(
            task,
            self.name,
            f"Verification outcome: {verification['outcome']}.",
            output=verification,
            metadata=metadata,
        )

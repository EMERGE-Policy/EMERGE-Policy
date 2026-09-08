"""Assemble the complete Task Verification Subagent instance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Emerge.base import ToolRegistry
from Emerge.providers.base import LLMProvider
from Emerge.subagents.skills import SkillRegistry
from Emerge.subagents.task_verification.agent import (
    TaskVerificationSubagent,
)
from Emerge.subagents.task_verification.context import (
    TASK_VERIFICATION_SYSTEM_PROMPT,
    TaskVerificationContextBuilder,
)
from Emerge.subagents.task_verification.tools import (
    ObservationStore,
    ObserveSceneTool,
    SubmitVerificationTool,
)


_TASK_VERIFICATION_DIR = Path(__file__).resolve().parent


def build_task_verification_subagent(
    provider: LLMProvider,
    workspace: str | Path,
    *,
    model: str | None = None,
    config: dict[str, Any] | None = None,
) -> TaskVerificationSubagent:
    """Build one self-contained Task Verification Subagent."""
    settings = dict(config or {})
    observation_store = ObservationStore(workspace)
    observe_tool = ObserveSceneTool(observation_store)
    submission_tool = SubmitVerificationTool()

    tools = ToolRegistry()
    tools.register(observe_tool)
    tools.register(submission_tool)

    skills = SkillRegistry()
    skills.register_directory(_TASK_VERIFICATION_DIR / "skills")

    return TaskVerificationSubagent(
        name="task_verification",
        description=(
            "Inspect current multi-camera images and verify whether visible "
            "object states and task relations were achieved."
        ),
        system_prompt=TASK_VERIFICATION_SYSTEM_PROMPT,
        provider=provider,
        tools=tools,
        skills=skills,
        context_builder=TaskVerificationContextBuilder(skills),
        capabilities=(
            "visual_task_verification",
            "object_state_verification",
            "qualitative_scene_assessment",
        ),
        input_modalities=("text", "image"),
        model=model,
        max_iterations=int(settings.get("max_iterations", 2)),
        observation_store=observation_store,
        submission_tool=submission_tool,
    )

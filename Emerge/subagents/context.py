"""Isolated context construction for sub-agent runs."""

import json
from dataclasses import dataclass
from typing import Any

from Emerge.subagents.content import TextContent, to_message_content
from Emerge.subagents.models import SubagentTask, SubagentToolResult
from Emerge.subagents.skills import SkillRegistry
from Emerge.utils.helpers import build_assistant_message


@dataclass(slots=True)
class SubagentRunContext:
    """Messages and task state belonging to one sub-agent run."""

    task: SubagentTask
    messages: list[dict[str, Any]]

    def add_assistant_message(
        self,
        content: str | None,
        *,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> None:
        """Append one provider-safe assistant message."""
        self.messages.append(
            build_assistant_message(
                content,
                tool_calls=tool_calls,
                reasoning_content=reasoning_content,
                thinking_blocks=thinking_blocks,
            )
        )

    def add_tool_result(self, call_id: str, tool_name: str, result: str) -> None:
        """Append one tool result to this run only."""
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": call_id,
                "name": tool_name,
                "content": result,
            }
        )

    def add_tool_observations(
        self,
        observations: list[tuple[str, SubagentToolResult]],
    ) -> None:
        """Append multimodal tool observations after all tool responses."""
        content = [TextContent("# Tool Observations").to_message_part()]
        for tool_name, result in observations:
            label = TextContent(f"## Observation from `{tool_name}`")
            content.append(label.to_message_part())
            content.extend(to_message_content(result.content))
        self.messages.append({"role": "user", "content": content})


class SubagentContextBuilder:
    """Create a fresh, memory-free context for every delegated task."""

    def __init__(self, agent_name: str, system_prompt: str, skills: SkillRegistry):
        self.agent_name = agent_name
        self.system_prompt = system_prompt.strip()
        self.skills = skills

    def build(self, task: SubagentTask) -> SubagentRunContext:
        """Build an isolated context for ``task``."""
        return SubagentRunContext(
            task=task,
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": self._build_task_content(task)},
            ],
        )

    def _build_system_prompt(self) -> str:
        parts = [f"# {self.agent_name}\n\n{self.system_prompt}"]
        if skill_context := self.skills.render():
            parts.append(f"# Skills\n\n{skill_context}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _build_task_content(task: SubagentTask) -> list[dict[str, Any]]:
        parts = [TextContent("# Task").to_message_part()]
        parts.extend(to_message_content(task.content))
        if task.input:
            text = TextContent(f"# Input\n\n{_format_data(task.input)}")
            parts.append(text.to_message_part())
        if task.context:
            text = TextContent(f"# Shared Context\n\n{_format_data(task.context)}")
            parts.append(text.to_message_part())
        if task.resource_scope:
            resources = "\n".join(f"- {item}" for item in task.resource_scope)
            text = TextContent(f"# Resource Scope\n\n{resources}")
            parts.append(text.to_message_part())
        return parts


def _format_data(value: dict[str, Any]) -> str:
    """Format structured task data for an LLM message."""
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)

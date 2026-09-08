"""Common execution loop for self-contained sub-agents."""

import asyncio
import time
from typing import Any

from loguru import logger

from Emerge.base import ToolRegistry
from Emerge.providers.base import LLMProvider
from Emerge.subagents.context import SubagentContextBuilder, SubagentRunContext
from Emerge.subagents.models import (
    InputModality,
    SubagentDescriptor,
    SubagentResult,
    SubagentStatus,
    SubagentTask,
    SubagentToolResult,
)
from Emerge.subagents.skills import SkillRegistry


class BaseSubagent:
    """Base implementation inherited by concrete sub-agent types.

    A concrete sub-agent owns its tool registry, skill registry, and context
    builder. Run state stays local to :meth:`run`, so calls do not share message
    history or memory.
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        system_prompt: str,
        provider: LLMProvider,
        tools: ToolRegistry | None = None,
        skills: SkillRegistry | None = None,
        context_builder: SubagentContextBuilder | None = None,
        capabilities: tuple[str, ...] = (),
        input_modalities: tuple[InputModality, ...] = ("text", "image"),
        model: str | None = None,
        max_iterations: int = 20,
    ) -> None:
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least one")

        self.descriptor = SubagentDescriptor(
            name=name,
            description=description,
            capabilities=capabilities,
            input_modalities=input_modalities,
        )
        self.provider = provider
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.tools = tools if tools is not None else ToolRegistry()
        self.skills = skills if skills is not None else SkillRegistry()
        self.context_builder = context_builder or SubagentContextBuilder(
            agent_name=name,
            system_prompt=system_prompt,
            skills=self.skills,
        )

    @property
    def name(self) -> str:
        return self.descriptor.name

    @property
    def description(self) -> str:
        return self.descriptor.description

    async def run(self, task: SubagentTask) -> SubagentResult:
        """Execute one isolated task, enforcing its optional timeout."""
        if task.timeout is None:
            return await self._run(task)

        try:
            return await asyncio.wait_for(self._run(task), timeout=task.timeout)
        except asyncio.TimeoutError:
            return SubagentResult.failure(
                task,
                self.name,
                f"Sub-agent task timed out after {task.timeout:g} seconds",
                status=SubagentStatus.TIMED_OUT,
            )

    async def _run(self, task: SubagentTask) -> SubagentResult:
        context = self.context_builder.build(task)
        tools_used: list[str] = []

        for iteration in range(1, self.max_iterations + 1):
            llm_started = time.perf_counter()
            response = await self.provider.chat_with_retry(
                messages=context.messages,
                tools=self.tools.get_definitions(),
                model=self.model,
            )
            logger.info(
                "Sub-agent LLM timing | agent={} iteration={} model={} elapsed={:.3f}s",
                self.name,
                iteration,
                self.model,
                time.perf_counter() - llm_started,
            )

            if response.finish_reason == "error":
                return SubagentResult.failure(
                    task,
                    self.name,
                    response.content or "The model returned an error",
                    metadata={"iterations": iteration, "tools_used": tools_used},
                )

            if response.has_tool_calls:
                context.add_assistant_message(
                    response.content,
                    tool_calls=[call.to_openai_tool_call() for call in response.tool_calls],
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                observations: list[tuple[str, SubagentToolResult]] = []
                for call in response.tool_calls:
                    tools_used.append(call.name)
                    raw_result = await self.tools.execute(call.name, call.arguments)
                    result = _normalize_tool_result(raw_result)
                    context.add_tool_result(call.id, call.name, result.text)
                    if result.content:
                        observations.append((call.name, result))
                if observations:
                    context.add_tool_observations(observations)
                if self._tool_work_complete():
                    return self.build_result(
                        task=task,
                        content=response.content,
                        context=context,
                        metadata={
                            "iterations": iteration,
                            "tools_used": tools_used,
                            "usage": response.usage,
                        },
                    )
                continue

            context.add_assistant_message(
                response.content,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            )
            return self.build_result(
                task=task,
                content=response.content,
                context=context,
                metadata={
                    "iterations": iteration,
                    "tools_used": tools_used,
                    "usage": response.usage,
                },
            )

        return SubagentResult.failure(
            task,
            self.name,
            f"Maximum tool iterations reached ({self.max_iterations})",
            metadata={"iterations": self.max_iterations, "tools_used": tools_used},
        )

    def _tool_work_complete(self) -> bool:
        return False

    def build_result(
        self,
        *,
        task: SubagentTask,
        content: str | None,
        context: SubagentRunContext,
        metadata: dict[str, Any],
    ) -> SubagentResult:
        """Convert the final model response into a result.

        Concrete sub-agents may override this method when their ``output`` has a
        domain-specific schema.
        """
        summary = content.strip() if content else "Task completed"
        return SubagentResult.success(
            task,
            self.name,
            summary,
            output=content,
            metadata=metadata,
        )


def _normalize_tool_result(result: Any) -> SubagentToolResult:
    """Normalize text and multimodal tool outputs for the agent loop."""
    if isinstance(result, SubagentToolResult):
        return result
    if isinstance(result, str):
        return SubagentToolResult(text=result)
    raise TypeError(
        "Sub-agent tools must return str or SubagentToolResult, "
        f"got {type(result).__name__}"
    )

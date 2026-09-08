"""Main-agent tool for invoking registered specialist sub-agents."""

from __future__ import annotations

import json
from typing import Any

from loguru import logger

from Emerge.base import Tool
from Emerge.subagents import SubagentRegistry, SubagentTask, TextContent


class DelegateSubagentTool(Tool):
    def __init__(self, registry: SubagentRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "delegate_subagent"

    @property
    def description(self) -> str:
        agents = "; ".join(
            f"{item.name}: {item.description}" for item in self._registry.list()
        )
        return (
            "Delegate a focused task to a registered specialist and wait for "
            f"its structured result. Available specialists: {agents}"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent_name": {
                    "type": "string",
                    "enum": list(self._registry.names),
                    "description": "Registered specialist name",
                },
                "task": {
                    "type": "string",
                    "description": "Self-contained task for the specialist",
                    "minLength": 1,
                },
            },
            "required": ["agent_name", "task"],
        }

    async def execute(self, agent_name: str, task: str) -> str:
        result = await self._registry.run(
            agent_name,
            SubagentTask(content=(TextContent(task),)),
        )
        payload = {
            "status": result.status.value,
            "agent_name": result.agent_name,
            "summary": result.summary,
            "output": result.output,
            "error": result.error,
        }
        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )
        logger.info(
            "Sub-agent result | agent={} status={} payload={}",
            result.agent_name,
            result.status.value,
            serialized,
        )
        return serialized

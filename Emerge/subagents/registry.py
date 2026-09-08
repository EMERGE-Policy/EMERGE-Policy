"""Registry for complete sub-agent instances."""

from Emerge.subagents.base import BaseSubagent
from Emerge.subagents.models import (
    SubagentDescriptor,
    SubagentResult,
    SubagentTask,
)


class SubagentRegistry:
    """Store and invoke sub-agents without knowing their internal capabilities."""

    def __init__(self) -> None:
        self._agents: dict[str, BaseSubagent] = {}

    def register(self, agent: BaseSubagent) -> None:
        """Register one fully assembled sub-agent instance."""
        if agent.name in self._agents:
            raise ValueError(f"Sub-agent '{agent.name}' is already registered")
        self._agents[agent.name] = agent

    def unregister(self, name: str) -> BaseSubagent | None:
        """Remove and return a registered sub-agent, if present."""
        return self._agents.pop(name, None)

    def get(self, name: str) -> BaseSubagent:
        """Return a registered sub-agent by name."""
        if name not in self._agents:
            available = ", ".join(self.names) or "none"
            raise KeyError(f"Sub-agent '{name}' is not registered. Available: {available}")
        return self._agents[name]

    def list(self) -> tuple[SubagentDescriptor, ...]:
        """Return public metadata for all registered sub-agents."""
        return tuple(agent.descriptor for agent in self._agents.values())

    async def run(self, name: str, task: SubagentTask) -> SubagentResult:
        """Delegate ``task`` to a registered sub-agent."""
        return await self.get(name).run(task)

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in registration order."""
        return tuple(self._agents)

    def __len__(self) -> int:
        return len(self._agents)

    def __contains__(self, name: str) -> bool:
        return name in self._agents

"""Data contracts shared by the main agent and all sub-agents."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypeAlias
from uuid import uuid4

from Emerge.subagents.content import ContentPart, ImageContent, TextContent


InputModality: TypeAlias = Literal["text", "image"]


class SubagentStatus(str, Enum):
    """Outcome of one sub-agent run."""

    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


@dataclass(frozen=True, slots=True)
class SubagentDescriptor:
    """Stable metadata exposed when a sub-agent is registered."""

    name: str
    description: str
    capabilities: tuple[str, ...] = ()
    input_modalities: tuple[InputModality, ...] = ("text", "image")

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Sub-agent name cannot be empty")
        if not self.description.strip():
            raise ValueError("Sub-agent description cannot be empty")


@dataclass(frozen=True, slots=True)
class SubagentTask:
    """A self-contained task delegated to one sub-agent."""

    content: tuple[ContentPart, ...]
    task_id: str = field(default_factory=lambda: uuid4().hex)
    input: dict[str, Any] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    resource_scope: tuple[str, ...] = ()
    timeout: float | None = None

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError("Sub-agent task content cannot be empty")
        if not all(isinstance(part, (TextContent, ImageContent)) for part in self.content):
            raise TypeError("Sub-agent task content contains an unsupported content block")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Sub-agent task timeout must be greater than zero")


@dataclass(frozen=True, slots=True)
class SubagentToolResult:
    """Textual tool result with optional multimodal observations."""

    text: str
    content: tuple[ContentPart, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Sub-agent tool result text cannot be empty")
        if not all(isinstance(part, (TextContent, ImageContent)) for part in self.content):
            raise TypeError("Sub-agent tool result contains an unsupported content block")


@dataclass(frozen=True, slots=True)
class SubagentResult:
    """Standard envelope with an instance-defined output payload."""

    task_id: str
    agent_name: str
    status: SubagentStatus
    summary: str
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        task: SubagentTask,
        agent_name: str,
        summary: str,
        *,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SubagentResult":
        """Build a successful result for ``task``."""
        return cls(
            task_id=task.task_id,
            agent_name=agent_name,
            status=SubagentStatus.SUCCESS,
            summary=summary,
            output=output,
            metadata=metadata or {},
        )

    @classmethod
    def failure(
        cls,
        task: SubagentTask,
        agent_name: str,
        error: str,
        *,
        status: SubagentStatus = SubagentStatus.FAILED,
        metadata: dict[str, Any] | None = None,
    ) -> "SubagentResult":
        """Build a non-success result for ``task``."""
        return cls(
            task_id=task.task_id,
            agent_name=agent_name,
            status=status,
            summary=error,
            error=error,
            metadata=metadata or {},
        )

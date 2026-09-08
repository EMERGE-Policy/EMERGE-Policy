"""Versioned contracts shared by interactive and headless clients."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunRequest(Contract):
    schema_version: Literal["Emerge.run_request.v1"] = "Emerge.run_request.v1"
    run_id: str = Field(default_factory=lambda: uuid4().hex, pattern=r"^[a-zA-Z0-9_-]{1,100}$")
    session_id: str = Field(default_factory=lambda: "cli:" + uuid4().hex, min_length=1)
    message: str = Field(min_length=1)
    config: str | None = None
    workspace: str | None = None
    model: str | None = None
    restrict_to_workspace: bool | None = None
    max_iterations: int | None = Field(default=None, ge=1)
    timeout_s: float | None = Field(default=None, gt=0)
    cancel_timeout_s: float = Field(default=10, gt=0, le=60)
    stream: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("message", "session_id")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class RunEvent(Contract):
    schema_version: Literal["Emerge.run_event.v1"] = "Emerge.run_event.v1"
    run_id: str
    session_id: str
    seq: int = Field(ge=1)
    timestamp: str = Field(default_factory=utc_now)
    type: str
    data: dict[str, Any] = Field(default_factory=dict)


class RunError(Contract):
    code: str
    message: str


class RunResult(Contract):
    schema_version: Literal["Emerge.run_result.v1"] = "Emerge.run_result.v1"
    run_id: str
    session_id: str
    run_status: Literal["completed", "failed", "cancelled", "timed_out"]
    finish_reason: str
    assistant_text: str = ""
    model: str | None = None
    iterations: int = 0
    usage: dict[str, int] = Field(default_factory=dict)
    usage_scope: Literal["main_agent"] = "main_agent"
    started_at: str
    finished_at: str = Field(default_factory=utc_now)
    duration_ms: int = 0
    error: RunError | None = None
    # This is execution acknowledgement, never benchmark success.
    cancellation: dict[str, Any] | None = None

    @property
    def exit_code(self) -> int:
        return {"completed": 0, "failed": 1, "cancelled": 130, "timed_out": 124}[self.run_status]

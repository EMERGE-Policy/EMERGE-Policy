"""UI-independent runtime contracts. Heavy agent dependencies are loaded on demand."""

from Emerge.runtime.protocol import RunEvent, RunRequest, RunResult

__all__ = ["RunEvent", "RunRequest", "RunResult"]

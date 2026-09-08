"""Tool for reporting user-visible progress during the current turn."""

from typing import Any, Awaitable, Callable

from Emerge.base import Tool


ProgressCallback = Callable[..., Awaitable[None]]


class MessageTool(Tool):
    """Report a concise milestone to the user in the current conversation."""

    def __init__(
        self,
        progress_callback: ProgressCallback | None = None,
    ):
        self._progress_callback = progress_callback

    def set_progress_callback(self, callback: ProgressCallback | None) -> None:
        """Bind this tool to the active turn's progress output."""
        self._progress_callback = callback

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return (
            "Report a concise, user-visible milestone during a long-running task. "
            "Use only for meaningful stage changes, waits, recovery, or important new results. "
            "Do not reveal private reasoning or use this instead of the final answer."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "A short factual progress update for the current user",
                    "minLength": 1,
                    "maxLength": 600,
                }
            },
            "required": ["content"],
            "additionalProperties": False,
        }

    async def execute(self, content: str) -> str:
        content = content.strip()
        if not content:
            return "Error: Progress update cannot be empty"
        if not self._progress_callback:
            return "Error: Progress reporting is not configured for this turn"
        try:
            await self._progress_callback(content, tool_hint=False)
            return "Progress reported to the current user"
        except Exception as e:
            return f"Error reporting progress: {str(e)}"

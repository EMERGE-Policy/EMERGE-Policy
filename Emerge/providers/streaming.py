"""Request-scoped streaming bridge; no UI or Runtime imports."""

from __future__ import annotations

import inspect
import json
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Awaitable, Callable

from Emerge.providers.base import LLMResponse, ToolCallRequest


@dataclass
class TextSink:
    callback: Callable[[str], Awaitable[None]]
    delivered: bool = False

    async def send(self, text: str):
        if text:
            self.delivered = True
            await self.callback(text)


text_sink: ContextVar[TextSink | None] = ContextVar("emerge_text_sink", default=None)


async def collect_stream(stream) -> LLMResponse:
    """Assemble complete tool calls before execution; deltas are display-only."""
    text, reasoning, calls, usage = [], [], {}, {}
    finish = None
    sink = text_sink.get()
    try:
        async for chunk in stream:
            raw = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            if raw.get("usage"):
                usage = {k: int(v) for k, v in raw["usage"].items()
                         if k in {"prompt_tokens", "completion_tokens", "total_tokens"} and v is not None}
            for choice in raw.get("choices", []):
                delta = choice.get("delta") or {}
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                if delta.get("content"):
                    text.append(delta["content"])
                    if sink:
                        await sink.send(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                for call in delta.get("tool_calls") or []:
                    key = (choice.get("index", 0), call.get("index", 0))
                    assembled = calls.setdefault(key, {"id": "", "name": "", "arguments": ""})
                    if call.get("id"):
                        assembled["id"] = call["id"]
                    function = call.get("function") or {}
                    if function.get("name"):
                        assembled["name"] += function["name"]
                    assembled["arguments"] += function.get("arguments") or ""
                    for source in (call, function):
                        if source.get("provider_specific_fields"):
                            target = "function_provider_specific_fields" if source is function else "provider_specific_fields"
                            assembled[target] = source["provider_specific_fields"]
        if finish is None:
            raise RuntimeError("Model stream ended without a finish marker")
        tool_calls = []
        for call in calls.values():
            if not call["id"] or not call["name"]:
                raise ValueError("Incomplete streamed tool call")
            call["arguments"] = json.loads(call["arguments"] or "{}")
            if not isinstance(call["arguments"], dict):
                raise ValueError("Tool arguments must be an object")
            tool_calls.append(ToolCallRequest(**call))
        return LLMResponse(content="".join(text) or None, tool_calls=tool_calls,
                           finish_reason=finish, usage=usage,
                           reasoning_content="".join(reasoning) or None)
    finally:
        close = getattr(stream, "aclose", None) or getattr(stream, "close", None)
        if close:
            result = close()
            if inspect.isawaitable(result):
                await result

"""Shared base infrastructure used by both the main agent and sub-agents.

`Tool` and `ToolRegistry` are the common foundation for all tools. They live
here so the main agent and the sub-agent framework can both depend on a single,
explicit location instead of reaching into `agent/tools/`.
"""

from Emerge.base.registry import ToolRegistry
from Emerge.base.tool import Tool

__all__ = ["Tool", "ToolRegistry"]

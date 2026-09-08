"""Object Location Subagent public assembly entrypoint."""

from Emerge.subagents.object_location.agent import ObjectLocationSubagent
from Emerge.subagents.object_location.register import (
    build_object_location_subagent,
)

__all__ = ["ObjectLocationSubagent", "build_object_location_subagent"]

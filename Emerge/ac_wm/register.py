"""Assembly helper for registering AC-WM in Emerge's SubagentRegistry."""

from typing import Any

from Emerge.subagents.registry import SubagentRegistry

from .subagent import AcWmSubagent


def register_ac_wm(
    registry: SubagentRegistry,
    *,
    provider: Any,
    rollout,
    judge,
    candidate_provider=None,
    dispatch=None,
    dispatch_mode: str = "evaluated_prefix",
    on_no_positive: str = "reject",
) -> AcWmSubagent:
    agent = AcWmSubagent(
        provider=provider,
        rollout=rollout,
        judge=judge,
        candidate_provider=candidate_provider,
        dispatch=dispatch,
        dispatch_mode=dispatch_mode,
        on_no_positive=on_no_positive,
    )
    registry.register(agent)
    return agent

"""LightNav-0 to NVIDIA SONIC navigation adapter.

This package is deliberately independent from both upstream control stacks.
It consumes LightNav's first local SE(2) waypoint and produces the standard
world-frame SONIC kinematic-planner command.
"""

from .adapter import (
    SONIC_IDLE_MODE,
    SONIC_SLOW_WALK_MODE,
    AdapterLimits,
    LightNav2SonicAdapter,
    SonicPlannerCommand,
)

__all__ = [
    "SONIC_IDLE_MODE",
    "SONIC_SLOW_WALK_MODE",
    "AdapterLimits",
    "LightNav2SonicAdapter",
    "SonicPlannerCommand",
]

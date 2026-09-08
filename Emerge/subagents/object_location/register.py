"""Assemble the complete Object Location Subagent instance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from Emerge.base import ToolRegistry
from Emerge.providers.base import LLMProvider
from Emerge.subagents.skills import SkillRegistry
from Emerge.subagents.object_location.agent import ObjectLocationSubagent
from Emerge.subagents.object_location.context import (
    OBJECT_LOCATION_SYSTEM_PROMPT,
    ObjectLocationContextBuilder,
)
from Emerge.subagents.object_location.tools import (
    LocateCandidatesTool,
    LocationEngine,
    ObservationStore,
    ObserveSceneTool,
    SegmentCandidatesTool,
)


_OBJECT_LOCATION_DIR = Path(__file__).resolve().parent


def build_object_location_subagent(
    provider: LLMProvider,
    workspace: str | Path,
    *,
    model: str | None = None,
    config: dict[str, Any] | None = None,
) -> ObjectLocationSubagent:
    """Build one self-contained Object Location Subagent."""
    settings = dict(config or {})
    store = ObservationStore(workspace)
    engine = LocationEngine(
        store,
        vggt_url=str(settings.get("vggt_url", "ws://localhost:8001")),
        sam3_url=str(settings.get("sam3_url", "ws://localhost:8002")),
        timeout=float(settings.get("timeout", 120.0)),
        point_conf_threshold=float(
            settings.get("point_conf_threshold", 0.3)
        ),
        min_points=int(settings.get("min_points", 80)),
        bbox_padding_pixels=int(settings.get("bbox_padding_pixels", 4)),
        view_center_tolerance_m=float(
            settings.get("view_center_tolerance_m", 0.08)
        ),
        ray_consensus_tolerance_m=float(
            settings.get("ray_consensus_tolerance_m", 0.02)
        ),
        min_consistent_views=int(settings.get("min_consistent_views", 2)),
    )
    observe_tool = ObserveSceneTool(store)
    candidate_tool = SegmentCandidatesTool(engine)
    location_tool = LocateCandidatesTool(engine)

    tools = ToolRegistry()
    tools.register(observe_tool)
    tools.register(candidate_tool)
    tools.register(location_tool)

    skills = SkillRegistry()
    skills.register_directory(_OBJECT_LOCATION_DIR / "skills")

    return ObjectLocationSubagent(
        name="object_location",
        description=(
            "Locate discrete movable objects and independently segmentable "
            "destinations, obstacles, and task-relevant visible LIBERO fixtures "
            "in the world frame from calibrated camera views."
        ),
        system_prompt=OBJECT_LOCATION_SYSTEM_PROMPT,
        provider=provider,
        tools=tools,
        skills=skills,
        context_builder=ObjectLocationContextBuilder(skills),
        capabilities=(
            "object_localization",
            "candidate_segmentation_verification",
            "sam3_prompt_generation",
            "qualitative_spatial_summary",
        ),
        input_modalities=("text", "image"),
        model=model,
        max_iterations=int(settings.get("max_iterations", 8)),
        observation_store=store,
        location_engine=engine,
        candidate_tool=candidate_tool,
        location_tool=location_tool,
    )

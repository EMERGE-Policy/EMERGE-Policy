"""Candidate segmentation and visual verification tool."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from Emerge.base import Tool
from Emerge.subagents import ImageContent, SubagentToolResult, TextContent
from Emerge.subagents.object_location.tools.location.engine import (
    CandidateSegmentation,
    LocationEngine,
)


_CANDIDATE_COLORS = (
    (255, 70, 70),
    (70, 220, 110),
    (70, 140, 255),
    (255, 190, 55),
    (185, 90, 255),
    (30, 210, 210),
)


class SegmentCandidatesTool(Tool):
    """Segment plausible physical candidates and return labeled overlays."""

    def __init__(self, engine: LocationEngine) -> None:
        self._engine = engine
        self.last_error: str | None = None

    @property
    def name(self) -> str:
        return "segment_candidates"

    @property
    def description(self) -> str:
        return (
            "Segment every visual candidate description for the requested "
            "object and return labeled masks over all full camera views. First "
            "call observe_scene and inspect the images. Give each plausible "
            "candidate a concise prompt grounded in visible appearance. When a "
            "requested spatial relation distinguishes same-kind instances, include "
            "that relation and its visible reference object in the prompt. One "
            "prompt can return up to two instances labeled #1 and #2 in each view. "
            "Do not choose the final semantic target before reviewing appearance "
            "and relation evidence in these overlays. All candidates are sent to "
            "SAM3 together, while VGGT geometry is cached for localization."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "description": (
                        "Separate visible appearance hypotheses grouped by the "
                        "semantic target they could represent. When visually "
                        "identical instances differ by a requested relation, use "
                        "relation-aware candidate prompts tied to the visible "
                        "reference object."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {
                                "type": "string",
                                "description": (
                                    "Unique snake_case ID for one visual appearance"
                                ),
                                "minLength": 1,
                            },
                            "target_key": {
                                "type": "string",
                                "description": (
                                    "Requested semantic target in snake_case"
                                ),
                                "minLength": 1,
                            },
                            "prompt": {
                                "type": "string",
                                "description": (
                                    "English SAM3 prompt describing only this "
                                    "candidate using its visible physical category "
                                    "and discriminative color, shape, material, cap, "
                                    "label, or pattern. If a visible spatial relation "
                                    "from the request identifies the intended "
                                    "instance, include that relation and reference "
                                    "object. Exclude coordinates, unsupported "
                                    "semantic identities, and irrelevant scene "
                                    "details."
                                ),
                                "minLength": 1,
                            },
                        },
                        "required": [
                            "candidate_id",
                            "target_key",
                            "prompt",
                        ],
                    },
                    "minItems": 1,
                }
            },
            "required": ["candidates"],
        }

    async def execute(
        self,
        candidates: list[dict[str, str]],
    ) -> SubagentToolResult:
        self.last_error = None
        try:
            segmentation = await self._engine.segment_candidates(candidates)
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            raise
        summary = _segmentation_summary(segmentation)
        content = []
        for view_name, image_data_url in _render_overlays(segmentation):
            content.extend(
                (
                    TextContent(f"Candidate mask overlay: {view_name}"),
                    ImageContent(image_data_url, detail="high"),
                )
            )
        return SubagentToolResult(
            text=json.dumps(summary, ensure_ascii=False),
            content=tuple(content),
        )

    def reset(self) -> None:
        self.last_error = None


def _segmentation_summary(
    segmentation: CandidateSegmentation,
) -> dict[str, Any]:
    views = []
    for view in segmentation.masks["views"]:
        detections = [
            {
                "candidate_id": target["object_key"],
                "instance_index": int(target.get("instance_index", 1)),
                "instance_label": _instance_label(target),
                "found": bool(target["found"]),
                "confidence": float(target["confidence"]),
                "bbox_2d": target["bbox_2d"],
                "area_pixels": int(target["area_pixels"]),
            }
            for target in view["targets"]
        ]
        views.append(
            {
                "view_name": view["view_name"],
                "detections": detections,
            }
        )
    return {
        "candidates": list(segmentation.candidates),
        "views": views,
        "next_step": (
            "Review every full-view overlay. For each semantic target, choose "
            "the candidate and the per-view # instance whose mask covers that "
            "same physical object. Instance ranks may change between views. Then "
            "call locate_candidates with candidate_id and verified_instances."
        ),
    }


def _render_overlays(
    segmentation: CandidateSegmentation,
) -> list[tuple[str, str]]:
    labels = list(
        dict.fromkeys(
            _instance_label(target)
            for view in segmentation.masks["views"]
            for target in view["targets"]
        )
    )
    colors = {
        label: _CANDIDATE_COLORS[index % len(_CANDIDATE_COLORS)]
        for index, label in enumerate(labels)
    }
    geometry_views = {
        view["name"]: view for view in segmentation.geometry["views"]
    }
    rendered = []
    for mask_view in segmentation.masks["views"]:
        view_name = str(mask_view["view_name"])
        rgb = np.asarray(geometry_views[view_name]["rgb"], dtype=np.uint8).copy()
        for target in mask_view["targets"]:
            if not target["found"] or target["mask"] is None:
                continue
            color = colors[_instance_label(target)]
            mask = np.asarray(target["mask"], dtype=bool)
            rgb[mask] = (
                0.55 * rgb[mask] + 0.45 * np.asarray(color)
            ).astype(np.uint8)

        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        for target in mask_view["targets"]:
            if not target["found"] or target["bbox_2d"] is None:
                continue
            candidate_id = _instance_label(target)
            color = colors[candidate_id]
            x1, y1, x2, y2 = (int(value) for value in target["bbox_2d"])
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            text_box = draw.textbbox((x1, y1), candidate_id)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            label_top = max(0, y1 - text_height - 6)
            draw.rectangle(
                (x1, label_top, x1 + text_width + 6, y1),
                fill=color,
            )
            draw.text((x1 + 3, label_top + 2), candidate_id, fill=(0, 0, 0))

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        rendered.append((view_name, f"data:image/png;base64,{encoded}"))
    return rendered


def _instance_label(target: dict[str, Any]) -> str:
    return f"{target['object_key']}#{int(target.get('instance_index', 1))}"

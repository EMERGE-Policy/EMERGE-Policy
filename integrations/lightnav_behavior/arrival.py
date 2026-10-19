"""Independent RGB-D handoff check for the radio navigation ablation."""

from __future__ import annotations

import asyncio
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from Emerge.subagents.object_location.tools.location.client import ModelServerClient
from Emerge.subagents.object_location.tools.location.rgbd_geometry import estimate_rgbd_object_locations
from Emerge.subagents.object_location.tools.observation import ObservationStore


async def verify_radio_handoff(
    workspace: Path,
    *,
    sam3_url: str,
    max_horizontal_distance_m: float,
    overlay_path: Path,
) -> dict[str, Any]:
    """Require an independently segmented radio within a preset base distance.

    This is a navigation handoff gate, not proof that the arm can reach or that
    the official task is complete. It uses only the published robot observation.
    """
    if not math.isfinite(max_horizontal_distance_m) or max_horizontal_distance_m <= 0:
        raise ValueError("max_horizontal_distance_m must be positive and finite")
    observation = ObservationStore(workspace).load()
    head = next((view for view in observation.views if view.name == "head"), None)
    if head is None or head.depth_m is None:
        return {"verified": False, "reason": "head_rgbd_unavailable"}
    payload = await ModelServerClient(sam3_url, timeout=120.0).infer({
        "views": [{"name": "head", "rgb": head.rgb}],
        "targets": [{"object_key": "radio", "prompt": "radio receiver"}],
        "max_instances": 1,
    })
    mask_view = payload["views"][0]
    target = next((entry for entry in mask_view.get("targets", []) if entry.get("object_key") == "radio"), None)
    if target is None or not target.get("found") or target.get("mask") is None:
        return {"verified": False, "reason": "radio_not_segmented"}
    mask = np.asarray(target["mask"], dtype=bool)
    if mask.shape != head.rgb.shape[:2]:
        return {"verified": False, "reason": "mask_shape_mismatch"}
    overlay = head.rgb.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 255, 255])).astype(np.uint8)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(overlay_path)
    located = estimate_rgbd_object_locations(
        observation, payload,
        [{"object_key": "radio", "prompt": "radio receiver"}],
        min_points=30, min_consistent_views=1,
    )[0]
    if not located.get("found"):
        return {"verified": False, "reason": located.get("failure_reason", "radio_depth_unavailable"),
                "overlay_path": str(overlay_path)}
    x, y, _z = (float(value) for value in located["position_m"])
    horizontal_distance = math.hypot(x, y)
    verified = x > 0 and horizontal_distance <= max_horizontal_distance_m
    return {
        "verified": verified,
        "reason": "near_radio" if verified else "radio_outside_handoff_region",
        "radio_position_robot_base_m": located["position_m"],
        "horizontal_distance_m": horizontal_distance,
        "max_horizontal_distance_m": max_horizontal_distance_m,
        "sam3_confidence": located["sam3_confidence"],
        "overlay_path": str(overlay_path),
        "note": "RGB-D proximity gate; arm reach and task success remain unverified",
    }

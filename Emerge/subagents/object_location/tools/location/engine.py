"""VGGT + SAM3 orchestration for candidate-based object localization."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import time
from typing import Any

from loguru import logger

from Emerge.subagents.object_location.tools.location.client import (
    ModelServerClient,
)
from Emerge.subagents.object_location.tools.location.pose import (
    estimate_object_locations,
)
from Emerge.subagents.object_location.tools.observation import (
    ObservationStore,
)


@dataclass(slots=True)
class CandidateSegmentation:
    """SAM3 masks and VGGT geometry cached for one candidate round."""

    candidates: tuple[dict[str, str], ...]
    geometry: dict[str, Any]
    masks: dict[str, Any]


class LocationEngine:
    def __init__(
        self,
        store: ObservationStore,
        *,
        vggt_url: str,
        sam3_url: str,
        timeout: float,
        point_conf_threshold: float,
        min_points: int,
        bbox_padding_pixels: int,
        view_center_tolerance_m: float,
        ray_consensus_tolerance_m: float,
        min_consistent_views: int,
    ) -> None:
        self._store = store
        self._vggt = ModelServerClient(vggt_url, timeout=timeout)
        self._sam3 = ModelServerClient(sam3_url, timeout=timeout)
        self._point_conf_threshold = point_conf_threshold
        self._min_points = min_points
        self._bbox_padding_pixels = bbox_padding_pixels
        self._view_center_tolerance_m = view_center_tolerance_m
        self._ray_consensus_tolerance_m = ray_consensus_tolerance_m
        self._min_consistent_views = min_consistent_views
        self._geometry: dict[str, Any] | None = None
        self._segmentation: CandidateSegmentation | None = None

    def reset(self) -> None:
        """Clear all model outputs cached for the previous sub-agent run."""
        self._geometry = None
        self._segmentation = None

    async def segment_candidates(
        self,
        candidates: list[dict[str, str]],
    ) -> CandidateSegmentation:
        """Segment all visual candidates and cache their masks for review."""
        candidate_ids = [candidate["candidate_id"] for candidate in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("candidate_id must be unique within one candidate round")

        observation = self._store.current()
        targets = [
            {
                "object_key": candidate["candidate_id"],
                "prompt": candidate["prompt"],
            }
            for candidate in candidates
        ]
        logger.info(
            "SAM3 candidate prompts | {}",
            [
                {
                    "candidate_id": candidate["candidate_id"],
                    "prompt": candidate["prompt"],
                }
                for candidate in candidates
            ],
        )
        geometry, masks = await asyncio.gather(
            self._get_geometry(),
            self._infer_sam3(
                {
                    "views": [
                        {"name": view.name, "rgb": view.rgb}
                        for view in observation.views
                    ],
                    "targets": targets,
                    "max_instances": 2,
                }
            ),
        )
        self._segmentation = CandidateSegmentation(
            candidates=tuple(dict(candidate) for candidate in candidates),
            geometry=geometry,
            masks=masks,
        )
        return self._segmentation

    def locate_candidates(
        self,
        selections: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Locate semantic targets from candidate masks confirmed by the agent."""
        if self._segmentation is None:
            raise RuntimeError(
                "No candidate masks are available. Call segment_candidates first."
            )

        candidates_by_id = {
            candidate["candidate_id"]: candidate
            for candidate in self._segmentation.candidates
        }
        available_views = {
            str(view["view_name"])
            for view in self._segmentation.masks["views"]
        }
        targets = []
        for selection in selections:
            candidate = candidates_by_id[selection["candidate_id"]]
            if candidate["target_key"] != selection["object_key"]:
                raise ValueError(
                    f"Candidate {selection['candidate_id']!r} belongs to "
                    f"target {candidate['target_key']!r}, not "
                    f"{selection['object_key']!r}"
                )
            verified_instances = _verified_instances(selection)
            verified_views = list(verified_instances)
            if len(verified_views) < self._min_consistent_views:
                raise ValueError(
                    f"Selection {selection['object_key']!r} requires at least "
                    f"{self._min_consistent_views} distinct verified views"
                )
            unknown_views = set(verified_views) - available_views
            if unknown_views:
                raise ValueError(
                    f"Unknown verified views: {sorted(unknown_views)}"
                )
            targets.append(
                {
                    "object_key": selection["object_key"],
                    "prompt": candidate["prompt"],
                }
            )

        selected_masks = _select_candidate_masks(
            self._segmentation.masks,
            selections,
        )
        objects = estimate_object_locations(
            self._segmentation.geometry,
            selected_masks,
            targets,
            point_conf_threshold=self._point_conf_threshold,
            min_points=self._min_points,
            bbox_padding_pixels=self._bbox_padding_pixels,
            view_center_tolerance_m=self._view_center_tolerance_m,
            ray_consensus_tolerance_m=self._ray_consensus_tolerance_m,
            min_consistent_views=self._min_consistent_views,
        )
        for result, selection in zip(objects, selections, strict=True):
            result["candidate_id"] = selection["candidate_id"]
            result["semantic_verified"] = True
            verified_instances = _verified_instances(selection)
            result["verified_views"] = list(verified_instances)
            result["verified_instances"] = [
                {"view_name": view_name, "instance_index": instance_index}
                for view_name, instance_index in verified_instances.items()
            ]

        geometry = self._segmentation.geometry
        return {
            "reference_view": geometry["reference_view"],
            "objects": objects,
            "geometry_alignment": geometry["alignment"],
        }

    async def _get_geometry(self) -> dict[str, Any]:
        if self._geometry is not None:
            return self._geometry

        observation = self._store.current()
        started = time.perf_counter()
        self._geometry = await self._vggt.infer(
            {
                "reference_view": observation.reference_view,
                "views": [
                    {
                        "name": view.name,
                        "rgb": view.rgb,
                        "intrinsics": view.intrinsics,
                        "T_world_camera": view.T_world_camera,
                    }
                    for view in observation.views
                ],
            }
        )
        logger.info(
            "VGGT timing | views={} elapsed={:.3f}s",
            len(observation.views),
            time.perf_counter() - started,
        )
        return self._geometry

    async def _infer_sam3(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        result = await self._sam3.infer(request)
        logger.info(
            "SAM3 timing | views={} targets={} elapsed={:.3f}s",
            len(request["views"]),
            len(request["targets"]),
            time.perf_counter() - started,
        )
        return result


def _select_candidate_masks(
    masks: dict[str, Any],
    selections: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rename selected candidate masks to their requested semantic keys."""
    views = []
    for view in masks["views"]:
        targets = []
        for selection in selections:
            verified_instances = _verified_instances(selection)
            instance_index = verified_instances.get(view["view_name"])
            candidate_masks = [
                target
                for target in view["targets"]
                if target["object_key"] == selection["candidate_id"]
            ]
            if instance_index is None:
                candidate_mask = candidate_masks[0]
            else:
                candidate_mask = next(
                    target
                    for target in candidate_masks
                    if int(target.get("instance_index", 1)) == instance_index
                )
            verified = instance_index is not None
            targets.append(
                {
                    **candidate_mask,
                    "object_key": selection["object_key"],
                    "found": bool(candidate_mask["found"] and verified),
                    "confidence": (
                        candidate_mask["confidence"] if verified else 0.0
                    ),
                    "mask": candidate_mask["mask"] if verified else None,
                    "bbox_2d": (
                        candidate_mask["bbox_2d"] if verified else None
                    ),
                    "area_pixels": (
                        candidate_mask["area_pixels"] if verified else 0
                    ),
                    "instance_index": 1,
                }
            )
        views.append({**view, "targets": targets})
    return {**masks, "views": views}


def _verified_instances(selection: dict[str, Any]) -> dict[str, int]:
    """Map each verified camera view to its selected per-view mask rank."""
    entries = selection.get("verified_instances")
    if entries is not None:
        return {
            str(entry["view_name"]): int(entry["instance_index"])
            for entry in entries
        }
    return {str(view_name): 1 for view_name in selection["verified_views"]}

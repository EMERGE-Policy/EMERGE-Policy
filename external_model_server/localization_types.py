"""Shared data structures for the standalone VGGT/SAM3 localization stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class VGGTAlignment:
    """Metric calibration diagnostics for VGGT depth reconstruction."""

    method: str
    depth_scale: float
    baseline_pair_count: int
    baseline_scale_median: float
    baseline_scale_relative_mad: float
    rms_camera_center_error_m: float
    max_camera_center_error_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "depth_scale": float(self.depth_scale),
            "baseline_pair_count": int(self.baseline_pair_count),
            "baseline_scale_median": float(self.baseline_scale_median),
            "baseline_scale_relative_mad": float(
                self.baseline_scale_relative_mad
            ),
            "rms_camera_center_error_m": float(
                self.rms_camera_center_error_m
            ),
            "max_camera_center_error_m": float(
                self.max_camera_center_error_m
            ),
        }


@dataclass(slots=True)
class VGGTAlignedCamera:
    """Predicted camera diagnostics plus authoritative observed calibration."""

    name: str
    intrinsics_pred: np.ndarray
    intrinsics_observed: np.ndarray
    extrinsics_pred: np.ndarray
    T_world_camera_observed: np.ndarray


@dataclass(slots=True)
class VGGTViewPrediction:
    """Per-view metric depth and points in the authoritative world frame."""

    name: str
    rgb: np.ndarray
    point_map_world: np.ndarray
    depth_m: np.ndarray
    depth_conf: np.ndarray | None = None
    point_conf: np.ndarray | None = None


@dataclass(slots=True)
class VGGTResult:
    """Normalized VGGT inference result for localization consumers."""

    reference_view: str
    point_map_world: np.ndarray
    depth_m: np.ndarray
    depth_conf: np.ndarray | None
    point_conf: np.ndarray | None
    cameras: list[VGGTAlignedCamera]
    alignment: VGGTAlignment
    views: list[VGGTViewPrediction] = field(default_factory=list)
    _views_by_name_cache: dict[str, VGGTViewPrediction] | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def get_view(self, name: str) -> VGGTViewPrediction:
        view = self._views_by_name().get(name)
        if view is None:
            raise KeyError(f"Unknown VGGT view: {name}")
        return view

    def _views_by_name(self) -> dict[str, VGGTViewPrediction]:
        if self._views_by_name_cache is None:
            self._views_by_name_cache = {view.name: view for view in self.views}
        return self._views_by_name_cache


@dataclass(slots=True)
class TargetMask:
    """Single target mask aligned to one input view."""

    object_key: str
    prompt: str
    found: bool
    confidence: float
    mask: np.ndarray | None
    bbox_2d: list[int] | None
    area_pixels: int
    instance_index: int = 1


@dataclass(slots=True)
class TargetMaskResult:
    """SAM3 segmentation results for a single input view."""

    view_name: str
    image_shape: tuple[int, int]
    targets: list[TargetMask] = field(default_factory=list)


@dataclass(slots=True)
class ObjectPoseEstimate:
    """Minimal object pose estimate in the world frame."""

    object_key: str
    frame: str
    center: np.ndarray
    rotation_matrix: np.ndarray
    rpy: np.ndarray
    extent: np.ndarray
    bbox_3d_corners: np.ndarray
    num_points: int
    num_inliers: int
    pose_inlier_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict for serialisation / downstream consumers."""
        return {
            "frame": self.frame,
            "center": np.asarray(self.center, dtype=np.float64).tolist(),
            "rpy": np.asarray(self.rpy, dtype=np.float64).tolist(),
            "size": np.asarray(self.extent, dtype=np.float64).tolist(),
            "num_points": int(self.num_points),
            "num_inliers": int(self.num_inliers),
            "pose_inlier_ratio": float(self.pose_inlier_ratio),
        }


@dataclass(slots=True)
class VGGTViewInput:
    """Calibrated RGB view required by metric VGGT reconstruction."""

    name: str
    rgb: np.ndarray
    intrinsics: np.ndarray
    T_world_camera: np.ndarray

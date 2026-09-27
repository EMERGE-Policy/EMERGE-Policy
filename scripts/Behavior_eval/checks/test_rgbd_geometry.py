"""Deterministic checks for calibrated RGB-D object geometry."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from Emerge.subagents.object_location.tools.location.rgbd_geometry import (
    backproject_masked_depth,
    estimate_rgbd_object_locations,
)
from Emerge.subagents.object_location.tools.observation import (
    CameraObservation,
    CameraView,
)


class RGBDGeometryTest(unittest.TestCase):
    def test_backprojection_and_rigid_transform_are_metric(self) -> None:
        depth = np.full((3, 3), 2.0, dtype=np.float32)
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        intrinsics = np.array(
            [[2.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        transform = np.eye(4)
        transform[:3, 3] = [1.0, 2.0, 3.0]

        points = backproject_masked_depth(depth, mask, intrinsics, transform)

        np.testing.assert_allclose(points, [[1.0, 2.0, 5.0]], atol=1e-8)

    def test_two_verified_views_produce_a_visible_surface_box(self) -> None:
        depth = np.ones((4, 4), dtype=np.float32)
        intrinsics = np.eye(3, dtype=np.float64)
        views = tuple(
            CameraView(
                name=name,
                image_path=Path(f"{name}.png"),
                rgb=np.zeros((4, 4, 3), dtype=np.uint8),
                depth_path=Path(f"{name}.npy"),
                depth_m=depth,
                intrinsics=intrinsics,
                T_world_camera=np.eye(4),
            )
            for name in ("head", "left_wrist")
        )
        observation = CameraObservation("head", "robot_base", views)
        mask = np.zeros((4, 4), dtype=bool)
        mask[1:3, 1:3] = True
        masks = {
            "views": [
                {
                    "view_name": view.name,
                    "targets": [
                        {
                            "object_key": "radio",
                            "prompt": "radio receiver",
                            "found": True,
                            "confidence": 0.9,
                            "mask": mask,
                        }
                    ],
                }
                for view in views
            ]
        }

        result = estimate_rgbd_object_locations(
            observation,
            masks,
            [{"object_key": "radio", "prompt": "radio receiver"}],
            min_points=4,
            min_consistent_views=2,
            outlier_quantile=0.0,
        )[0]

        self.assertTrue(result["found"])
        self.assertEqual(result["frame"], "robot_base")
        self.assertEqual(result["position_type"], "visible_surface_bbox_center")
        np.testing.assert_allclose(result["position_m"], [1.5, 1.5, 1.0])
        np.testing.assert_allclose(result["size_m"], [1.0, 1.0, 0.0])
        self.assertEqual(result["visible_views"], ["head", "left_wrist"])

    def test_one_view_cannot_satisfy_two_view_consensus(self) -> None:
        view = CameraView(
            name="head",
            image_path=Path("head.png"),
            rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            depth_path=Path("head.npy"),
            depth_m=np.ones((2, 2), dtype=np.float32),
            intrinsics=np.eye(3),
            T_world_camera=np.eye(4),
        )
        observation = CameraObservation("head", "robot_base", (view,))
        masks = {
            "views": [
                {
                    "view_name": "head",
                    "targets": [
                        {
                            "object_key": "radio",
                            "found": True,
                            "confidence": 1.0,
                            "mask": np.ones((2, 2), dtype=bool),
                        }
                    ],
                }
            ]
        }

        result = estimate_rgbd_object_locations(
            observation,
            masks,
            [{"object_key": "radio", "prompt": "radio"}],
            min_points=1,
            min_consistent_views=2,
        )[0]

        self.assertFalse(result["found"])
        self.assertEqual(result["failure_reason"], "insufficient_consistent_views")


if __name__ == "__main__":
    unittest.main()

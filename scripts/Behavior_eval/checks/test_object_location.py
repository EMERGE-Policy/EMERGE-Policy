"""Compatibility checks between BEHAVIOR observations and object_location."""

from __future__ import annotations

import asyncio
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import msgpack
import numpy as np

from Emerge.subagents.object_location.agent import _compact_objects
from Emerge.subagents.object_location.tools.location.engine import (
    CandidateSegmentation,
    LocationEngine,
)
from Emerge.subagents.object_location.tools.observation import ObservationStore
from Emerge.subagents.object_location.tools.segment_candidates import (
    SegmentCandidatesTool,
)
from robot.behavior.observation import ObservationPublisher


def _pack_array(value):
    return {
        b"__ndarray__": True,
        b"data": value.tobytes(),
        b"dtype": value.dtype.str,
        b"shape": value.shape,
    }


def _calibration() -> dict:
    return {
        "source": "test calibration",
        "coordinate_frame": "robot_base",
        "cameras": {
            name: {
                "resolution": [224, 224],
                "intrinsics": [[100.0, 0.0, 112.0], [0.0, 100.0, 112.0], [0.0, 0.0, 1.0]],
            }
            for name in ("left_wrist", "right_wrist", "head")
        },
    }


def _publish_behavior_observation(workspace: Path) -> None:
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    depth = np.full((224, 224, 1), 1.25, dtype=np.float32)
    poses = np.array(
        [
            -0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0,
        ],
        dtype=np.float32,
    )[None, :]
    message = msgpack.packb(
        {
            "robot::robot:zed_link:Camera:0::rgb": image,
            "robot::robot:left_realsense_link:Camera:0::rgb": image,
            "robot::robot:right_realsense_link:Camera:0::rgb": image,
            "robot::robot:zed_link:Camera:0::depth_linear": depth,
            "robot::robot:left_realsense_link:Camera:0::depth_linear": depth,
            "robot::robot:right_realsense_link:Camera:0::depth_linear": depth,
            "robot::cam_rel_poses": poses,
            "robot::proprio": np.zeros((1, 61), dtype=np.float32),
        },
        default=_pack_array,
    )
    publisher = ObservationPublisher(
        workspace,
        publish_interval=1,
        camera_calibration=_calibration(),
    )
    publisher.capture(message)


class _RecordingClient:
    def __init__(self, response: dict) -> None:
        self.response = response
        self.requests: list[dict] = []

    async def infer(self, request: dict) -> dict:
        self.requests.append(request)
        return self.response


class BehaviorObjectLocationCompatibilityTest(unittest.TestCase):
    def test_agent_result_keeps_geometry_needed_for_precontact(self) -> None:
        compact = _compact_objects(
            {
                "objects": [
                    {
                        "name": "radio power button",
                        "found": True,
                        "frame": "robot_base",
                        "position_m": [0.5, -0.1, 0.9],
                        "size_m": [0.04, 0.03, 0.02],
                        "rpy_rad": [0.0, 0.0, 0.0],
                        "position_type": "visible_surface_bbox_center",
                        "bbox_world_min_m": [0.48, -0.115, 0.89],
                        "bbox_world_max_m": [0.52, -0.085, 0.91],
                    }
                ]
            }
        )

        self.assertEqual(
            compact[0]["bbox_world_min_m"], [0.48, -0.115, 0.89]
        )
        self.assertEqual(
            compact[0]["bbox_world_max_m"], [0.52, -0.085, 0.91]
        )
        self.assertEqual(
            compact[0]["position_type"], "visible_surface_bbox_center"
        )

    def test_object_location_loads_all_calibrated_behavior_views(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            _publish_behavior_observation(workspace)

            observation = ObservationStore(workspace).load()

            self.assertEqual(observation.reference_view, "head")
            self.assertEqual(observation.coordinate_frame, "robot_base")
            self.assertEqual(
                {view.name for view in observation.views},
                {"left_wrist", "right_wrist", "head"},
            )
            self.assertTrue(all(view.intrinsics.shape == (3, 3) for view in observation.views))
            self.assertTrue(all(view.T_world_camera.shape == (4, 4) for view in observation.views))
            self.assertTrue(observation.has_rgbd)
            self.assertTrue(all(view.depth_m is not None for view in observation.views))
            self.assertTrue(all(view.depth_m.shape == (224, 224) for view in observation.views))

    def test_rgbd_manifest_requires_every_depth_file(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            _publish_behavior_observation(workspace)
            manifest_path = workspace / "artifacts/observations/observation.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["views"][0].pop("depth_path")
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required depth_path"):
                ObservationStore(workspace).load()

    def test_rgb_and_depth_shapes_must_match(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            _publish_behavior_observation(workspace)
            manifest_path = workspace / "artifacts/observations/observation.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            depth_path = workspace / manifest["views"][0]["depth_path"]
            np.save(depth_path, np.ones((12, 13), dtype=np.float32), allow_pickle=False)

            with self.assertRaisesRegex(ValueError, "RGB/depth shapes differ"):
                ObservationStore(workspace).load()

    def test_depth_requires_metre_units_and_valid_samples(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            _publish_behavior_observation(workspace)
            manifest_path = workspace / "artifacts/observations/observation.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["views"][0]["depth_unit"] = "millimeter"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "depth_unit must be 'meter'"):
                ObservationStore(workspace).load()

            manifest["views"][0]["depth_unit"] = "meter"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            depth_path = workspace / manifest["views"][0]["depth_path"]
            np.save(
                depth_path,
                np.full((224, 224), np.nan, dtype=np.float32),
                allow_pickle=False,
            )
            with self.assertRaisesRegex(ValueError, "no finite positive samples"):
                ObservationStore(workspace).load()

    def test_legacy_rgb_only_manifest_remains_loadable_for_vggt(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            _publish_behavior_observation(workspace)
            manifest_path = workspace / "artifacts/observations/observation.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["rgbd"] = False
            for view in manifest["views"]:
                view.pop("depth_path", None)
                view.pop("depth_unit", None)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            observation = ObservationStore(workspace).load()
            self.assertFalse(observation.has_rgbd)

    def test_rgbd_observation_sends_views_to_sam3_without_calling_vggt(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            _publish_behavior_observation(workspace)
            store = ObservationStore(workspace)
            store.load()
            engine = LocationEngine(
                store,
                vggt_url="ws://unused:8001",
                sam3_url="ws://unused:8002",
                timeout=1.0,
                point_conf_threshold=0.3,
                min_points=1,
                bbox_padding_pixels=0,
                view_center_tolerance_m=0.08,
                ray_consensus_tolerance_m=0.02,
                min_consistent_views=2,
            )
            vggt = _RecordingClient({"reference_view": "head", "alignment": {}})
            sam3 = _RecordingClient({"views": []})
            engine._vggt = vggt
            engine._sam3 = sam3

            asyncio.run(
                engine.segment_candidates(
                    [{"candidate_id": "radio_1", "target_key": "radio", "prompt": "black radio"}]
                )
            )

            self.assertEqual(len(vggt.requests), 0)
            self.assertEqual(len(sam3.requests), 1)
            self.assertEqual(len(sam3.requests[0]["views"]), 3)
            self.assertEqual(engine._active_geometry_backend, "rgbd")

    def test_rgbd_engine_localizes_verified_masks_without_vggt(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            _publish_behavior_observation(workspace)
            store = ObservationStore(workspace)
            store.load()
            engine = LocationEngine(
                store,
                vggt_url="ws://unused:8001",
                sam3_url="ws://unused:8002",
                timeout=1.0,
                point_conf_threshold=0.3,
                min_points=10,
                bbox_padding_pixels=0,
                view_center_tolerance_m=0.08,
                ray_consensus_tolerance_m=0.02,
                min_consistent_views=2,
            )
            mask = np.ones((224, 224), dtype=bool)
            sam3 = _RecordingClient(
                {
                    "views": [
                        {
                            "view_name": name,
                            "image_shape": [224, 224],
                            "targets": [
                                {
                                    "object_key": "radio_1",
                                    "prompt": "black radio",
                                    "found": True,
                                    "confidence": 0.95,
                                    "instance_index": 1,
                                    "mask": mask,
                                    "bbox_2d": [0, 0, 223, 223],
                                    "area_pixels": int(mask.sum()),
                                }
                            ],
                        }
                        for name in ("head", "left_wrist", "right_wrist")
                    ]
                }
            )
            vggt = _RecordingClient({})
            engine._sam3 = sam3
            engine._vggt = vggt
            asyncio.run(
                engine.segment_candidates(
                    [{"candidate_id": "radio_1", "target_key": "radio", "prompt": "black radio"}]
                )
            )

            result = engine.locate_candidates(
                [
                    {
                        "object_key": "radio",
                        "candidate_id": "radio_1",
                        "verified_instances": [
                            {"view_name": "head", "instance_index": 1},
                            {"view_name": "left_wrist", "instance_index": 1},
                        ],
                    }
                ]
            )

            self.assertEqual(vggt.requests, [])
            self.assertEqual(result["geometry_alignment"]["method"], "calibrated_rgbd")
            self.assertTrue(result["objects"][0]["found"])
            self.assertEqual(result["objects"][0]["frame"], "robot_base")
            self.assertEqual(
                result["objects"][0]["position_type"],
                "visible_surface_bbox_center",
            )

    def test_rgbd_engine_localizes_one_verified_head_view(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            _publish_behavior_observation(workspace)
            store = ObservationStore(workspace)
            store.load()
            engine = LocationEngine(
                store,
                vggt_url="ws://unused:8001",
                sam3_url="ws://unused:8002",
                timeout=1.0,
                point_conf_threshold=0.3,
                min_points=10,
                bbox_padding_pixels=0,
                view_center_tolerance_m=0.08,
                ray_consensus_tolerance_m=0.02,
                min_consistent_views=2,
            )
            mask = np.zeros((224, 224), dtype=bool)
            mask[100:110, 100:110] = True
            engine._sam3 = _RecordingClient(
                {
                    "views": [
                        {
                            "view_name": name,
                            "image_shape": [224, 224],
                            "targets": [
                                {
                                    "object_key": "radio_1",
                                    "prompt": "red radio",
                                    "found": name == "head",
                                    "confidence": 0.9 if name == "head" else 0.0,
                                    "instance_index": 1,
                                    "mask": mask if name == "head" else None,
                                    "bbox_2d": [100, 100, 109, 109]
                                    if name == "head"
                                    else None,
                                    "area_pixels": int(mask.sum())
                                    if name == "head"
                                    else 0,
                                }
                            ],
                        }
                        for name in ("head", "left_wrist", "right_wrist")
                    ]
                }
            )

            asyncio.run(
                engine.segment_candidates(
                    [
                        {
                            "candidate_id": "radio_1",
                            "target_key": "radio",
                            "prompt": "red radio",
                        }
                    ]
                )
            )
            result = engine.locate_candidates(
                [
                    {
                        "object_key": "radio",
                        "candidate_id": "radio_1",
                        "verified_instances": [
                            {"view_name": "head", "instance_index": 1}
                        ],
                    }
                ]
            )["objects"][0]

            self.assertTrue(result["found"])
            self.assertEqual(result["frame"], "robot_base")
            self.assertEqual(result["verified_views"], ["head"])
            self.assertEqual(
                result["geometry_evidence"],
                "single_view_calibrated_rgbd",
            )

            engine._active_geometry_backend = "vggt"
            with self.assertRaisesRegex(
                ValueError,
                "requires at least 2 distinct verified views",
            ):
                engine.locate_candidates(
                    [
                        {
                            "object_key": "radio",
                            "candidate_id": "radio_1",
                            "verified_instances": [
                                {"view_name": "head", "instance_index": 1}
                            ],
                        }
                    ]
                )

    def test_rgbd_candidate_tool_renders_overlays_from_observation_rgb(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            _publish_behavior_observation(workspace)
            store = ObservationStore(workspace)
            store.load()
            engine = LocationEngine(
                store,
                vggt_url="ws://unused:8001",
                sam3_url="ws://unused:8002",
                timeout=1.0,
                point_conf_threshold=0.3,
                min_points=10,
                bbox_padding_pixels=0,
                view_center_tolerance_m=0.08,
                ray_consensus_tolerance_m=0.02,
                min_consistent_views=2,
            )
            mask = np.zeros((224, 224), dtype=bool)
            mask[100:110, 100:110] = True
            engine._sam3 = _RecordingClient(
                {
                    "views": [
                        {
                            "view_name": name,
                            "image_shape": [224, 224],
                            "targets": [
                                {
                                    "object_key": "radio_1",
                                    "prompt": "red radio",
                                    "found": name == "head",
                                    "confidence": 0.9 if name == "head" else 0.0,
                                    "instance_index": 1,
                                    "mask": mask if name == "head" else None,
                                    "bbox_2d": [100, 100, 109, 109]
                                    if name == "head"
                                    else None,
                                    "area_pixels": int(mask.sum())
                                    if name == "head"
                                    else 0,
                                }
                            ],
                        }
                        for name in ("head", "left_wrist", "right_wrist")
                    ]
                }
            )
            tool = SegmentCandidatesTool(engine)

            result = asyncio.run(
                tool.execute(
                    [
                        {
                            "candidate_id": "radio_1",
                            "target_key": "radio",
                            "prompt": "red radio",
                        }
                    ]
                )
            )

            self.assertIsNone(tool.last_error)
            self.assertEqual(
                [item.text for item in result.content if hasattr(item, "text")],
                [
                    "Candidate mask overlay: head",
                    "Candidate mask overlay: left_wrist",
                    "Candidate mask overlay: right_wrist",
                ],
            )

    def test_location_result_uses_the_manifest_coordinate_frame(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            _publish_behavior_observation(workspace)
            store = ObservationStore(workspace)
            store.load()
            engine = LocationEngine(
                store,
                vggt_url="ws://unused:8001",
                sam3_url="ws://unused:8002",
                timeout=1.0,
                point_conf_threshold=0.3,
                min_points=1,
                bbox_padding_pixels=0,
                view_center_tolerance_m=0.08,
                ray_consensus_tolerance_m=0.02,
                min_consistent_views=2,
            )
            masks = {
                "views": [
                    {
                        "view_name": name,
                        "targets": [
                            {
                                "object_key": "radio_1",
                                "found": True,
                                "confidence": 1.0,
                                "instance_index": 1,
                                "mask": np.ones((4, 4), dtype=bool),
                                "bbox_2d": [0, 0, 3, 3],
                                "area_pixels": 16,
                            }
                        ],
                    }
                    for name in ("head", "left_wrist")
                ]
            }
            engine._segmentation = CandidateSegmentation(
                candidates=(
                    {
                        "candidate_id": "radio_1",
                        "target_key": "radio",
                        "prompt": "black radio",
                    },
                ),
                geometry={"reference_view": "head", "alignment": {}},
                masks=masks,
            )
            selection = {
                "object_key": "radio",
                "candidate_id": "radio_1",
                "verified_instances": [
                    {"view_name": "head", "instance_index": 1},
                    {"view_name": "left_wrist", "instance_index": 1},
                ],
            }
            estimated = [{"name": "radio", "found": True, "frame": "world"}]

            with patch(
                "Emerge.subagents.object_location.tools.location.engine.estimate_object_locations",
                return_value=estimated,
            ):
                result = engine.locate_candidates([selection])

            self.assertEqual(result["coordinate_frame"], "robot_base")
            self.assertEqual(result["objects"][0]["frame"], "robot_base")

    def test_uncalibrated_manifest_fails_before_model_inference(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            _publish_behavior_observation(workspace)
            manifest_path = workspace / "artifacts/observations/observation.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["views"][0]["T_world_camera"] = None
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "finite 4x4"):
                ObservationStore(workspace).load()


if __name__ == "__main__":
    unittest.main()

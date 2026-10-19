"""Checks for the camera side-channel published alongside the wire frames."""

from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import msgpack
import numpy as np
from PIL import Image

from robot.behavior.observation import ObservationPublisher


def _pack_array(value):
    return {
        b"__ndarray__": True,
        b"data": value.tobytes(),
        b"dtype": value.dtype.str,
        b"shape": value.shape,
    }


def _camera_message(seed: int = 0) -> bytes:
    image = np.full((4, 5, 3), seed, dtype=np.uint8)
    depth = np.full((1, 4, 5, 1), 0.5 + seed, dtype=np.float32)
    camera_poses = np.array(
        [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,  # left wrist: identity at x=1
            0.0,
            2.0,
            0.0,
            0.0,
            0.0,
            np.sqrt(0.5),
            np.sqrt(0.5),  # right wrist: +90 degrees around z
            0.0,
            0.0,
            3.0,
            0.0,
            0.0,
            0.0,
            1.0,  # head: identity at z=3
        ],
        dtype=np.float32,
    )[None, :]
    return msgpack.packb(
        {
            "robot::robot:zed_link:Camera:0::rgb": image,
            "robot::robot:left_realsense_link:Camera:0::rgb": image,
            "robot::robot:right_realsense_link:Camera:0::rgb": image,
            "robot::robot:zed_link:Camera:0::depth_linear": depth,
            "robot::robot:left_realsense_link:Camera:0::depth_linear": depth,
            "robot::robot:right_realsense_link:Camera:0::depth_linear": depth,
            "robot::cam_rel_poses": camera_poses,
            "robot::proprio": np.arange(61, dtype=np.float32)[None, :],
        },
        default=_pack_array,
    )


class ObservationPublisherTest(unittest.TestCase):
    """Manifest contents must be self-describing and frame-consistent."""

    def _publisher(self, workspace: Path, **kwargs) -> ObservationPublisher:
        return ObservationPublisher(workspace, publish_interval=1, **kwargs)

    @staticmethod
    def _calibration() -> dict:
        return {
            "source": "unit test",
            "coordinate_frame": "robot_base",
            "cameras": {
                name: {
                    "resolution": [10, 8],
                    "intrinsics": [[8.0, 0.0, 5.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]],
                }
                for name in ("left_wrist", "right_wrist", "head")
            },
        }

    def test_manifest_records_sequence_episode_and_timestamp(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            publisher = self._publisher(workspace)
            publisher.note_reset()
            publisher.capture(_camera_message(seed=1))
            manifest = json.loads(publisher.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["observation_seq"], 1)
            self.assertEqual(manifest["episode_id"], 1)
            self.assertTrue(manifest["captured_at"].endswith("Z"))
            self.assertEqual(manifest["revision"], 1)

    def test_immediate_publish_waits_for_in_progress_capture(self) -> None:
        """The navigation thread must not race the first automatic publish."""
        with TemporaryDirectory() as temporary:
            publisher = self._publisher(Path(temporary))
            entered = threading.Event()
            release = threading.Event()
            forced_finished = threading.Event()
            original_publish = publisher._publish_latest_locked

            def delayed_publish():
                entered.set()
                if not release.wait(timeout=2):
                    raise TimeoutError("test did not release the first publish")
                return original_publish()

            def force_publish():
                publisher.publish_latest()
                forced_finished.set()

            with patch.object(publisher, "_publish_latest_locked", side_effect=delayed_publish):
                capture_thread = threading.Thread(target=publisher.capture, args=(_camera_message(),))
                force_thread = threading.Thread(target=force_publish)
                capture_thread.start()
                try:
                    self.assertTrue(entered.wait(timeout=2))
                    force_thread.start()
                    self.assertFalse(forced_finished.wait(timeout=0.05))
                finally:
                    release.set()
                    capture_thread.join(timeout=2)
                    if force_thread.ident is not None:
                        force_thread.join(timeout=2)
            self.assertFalse(capture_thread.is_alive())
            self.assertFalse(force_thread.is_alive())
            self.assertTrue(forced_finished.is_set())
            self.assertIsNone(publisher.last_error)
            self.assertEqual(publisher.observation_count, 1)
            self.assertEqual(publisher.revision, 2)

    def test_image_paths_are_revision_specific_and_pruned(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            publisher = self._publisher(workspace, keep_revisions=2)
            for seed in range(3):
                publisher.capture(_camera_message(seed=seed))
            images = sorted((workspace / "artifacts/cameras/head").glob("head.*.png"))
            depths = sorted((workspace / "artifacts/depth/head").glob("head.*.npy"))
            self.assertEqual(len(images), 2)
            self.assertEqual(len(depths), 2)
            manifest = json.loads(publisher.manifest_path.read_text(encoding="utf-8"))
            self.assertTrue((workspace / manifest["views"][0]["image_path"]).is_file())
            self.assertIn(
                f"head.{manifest['revision']:06d}.png",
                manifest["views"][0]["image_path"],
            )

    def test_manifest_and_images_come_from_the_same_frame(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            publisher = self._publisher(workspace)
            publisher.capture(_camera_message(seed=11))
            manifest = json.loads(publisher.manifest_path.read_text(encoding="utf-8"))
            revision = manifest["revision"]
            for view in manifest["views"]:
                self.assertIn(f".{revision:06d}.png", view["image_path"])
                with Image.open(workspace / view["image_path"]) as saved:
                    self.assertEqual(saved.size, (5, 4))

    def test_linear_depth_is_saved_losslessly_with_the_rgb_revision(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            publisher = self._publisher(workspace)
            publisher.capture(_camera_message(seed=2))
            manifest = json.loads(publisher.manifest_path.read_text(encoding="utf-8"))

            self.assertTrue(manifest["rgbd"])
            revision = manifest["revision"]
            for view in manifest["views"]:
                self.assertTrue(view["depth_available"])
                self.assertEqual(view["depth_unit"], "meter")
                self.assertIn(f".{revision:06d}.npy", view["depth_path"])
                depth = np.load(workspace / view["depth_path"], allow_pickle=False)
                self.assertEqual(depth.dtype, np.float32)
                self.assertEqual(depth.shape, (4, 5))
                np.testing.assert_array_equal(depth, np.full((4, 5), 2.5, dtype=np.float32))

    def test_depth_with_a_different_shape_is_not_published(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            publisher = self._publisher(workspace)
            payload = msgpack.unpackb(
                _camera_message(), raw=False, object_hook=lambda item: np.ndarray(
                    buffer=item[b"data"],
                    dtype=np.dtype(item[b"dtype"]),
                    shape=item[b"shape"],
                ) if item.get(b"__ndarray__") else item,
            )
            payload["robot::robot:zed_link:Camera:0::depth_linear"] = np.ones(
                (3, 3), dtype=np.float32
            )
            publisher.capture(msgpack.packb(payload, default=_pack_array))
            manifest = json.loads(publisher.manifest_path.read_text(encoding="utf-8"))
            views = {view["name"]: view for view in manifest["views"]}

            self.assertFalse(manifest["rgbd"])
            self.assertFalse(views["head"]["depth_available"])
            self.assertNotIn("depth_path", views["head"])
            self.assertIn("head:depth_shape", publisher.snapshot()["last_warning"])

    def test_camera_poses_are_published_in_the_robot_base_frame(self) -> None:
        with TemporaryDirectory() as temporary:
            publisher = self._publisher(Path(temporary))
            publisher.capture(_camera_message())
            manifest = json.loads(publisher.manifest_path.read_text(encoding="utf-8"))
            views = {view["name"]: view for view in manifest["views"]}

            self.assertEqual(manifest["coordinate_frame"], "robot_base")
            self.assertTrue(manifest["extrinsics_calibrated"])
            self.assertFalse(manifest["intrinsics_calibrated"])
            self.assertFalse(manifest["calibrated"])
            self.assertEqual(
                np.asarray(views["left_wrist"]["T_world_camera"])[:3, 3].tolist(),
                [1.0, 0.0, 0.0],
            )
            self.assertEqual(
                np.asarray(views["head"]["T_world_camera"])[:3, 3].tolist(),
                [0.0, 0.0, 3.0],
            )
            np.testing.assert_allclose(
                np.asarray(views["right_wrist"]["T_world_camera"])[:3, :3],
                [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
                atol=1e-6,
            )
            np.testing.assert_allclose(
                np.asarray(views["head"]["T_world_camera"])[:3, :3],
                np.diag([1.0, -1.0, -1.0]),
                atol=1e-6,
            )
            self.assertTrue(all(view["extrinsics_calibrated"] for view in views.values()))

    def test_configured_intrinsics_are_scaled_to_the_published_image(self) -> None:
        with TemporaryDirectory() as temporary:
            publisher = self._publisher(
                Path(temporary), camera_calibration=self._calibration()
            )
            publisher.capture(_camera_message())
            manifest = json.loads(publisher.manifest_path.read_text(encoding="utf-8"))

            self.assertTrue(manifest["intrinsics_calibrated"])
            self.assertTrue(manifest["calibrated"])
            self.assertEqual(manifest["calibration_source"], "unit test")
            for view in manifest["views"]:
                self.assertTrue(view["intrinsics_calibrated"])
                self.assertTrue(view["calibrated"])
                np.testing.assert_allclose(
                    view["intrinsics"],
                    [[4.0, 0.0, 2.5], [0.0, 4.0, 2.0], [0.0, 0.0, 1.0]],
                )

    def test_missing_calibration_is_explicit_instead_of_estimated(self) -> None:
        with TemporaryDirectory() as temporary:
            publisher = self._publisher(Path(temporary))
            publisher.capture(_camera_message())
            manifest = json.loads(publisher.manifest_path.read_text(encoding="utf-8"))

            self.assertFalse(manifest["intrinsics_calibrated"])
            self.assertTrue(all(view["intrinsics"] is None for view in manifest["views"]))

    def test_r1pro_proprio_is_preserved_and_split_into_named_fields(self) -> None:
        with TemporaryDirectory() as temporary:
            publisher = self._publisher(Path(temporary))
            publisher.capture(_camera_message())
            manifest = json.loads(publisher.manifest_path.read_text(encoding="utf-8"))
            state = manifest["robot_state"]

            self.assertEqual(state["source_observation_key"], "robot::proprio")
            self.assertEqual(state["coordinate_frame"], "robot_base")
            self.assertEqual(state["proprio_dim"], 61)
            self.assertEqual(state["proprio"], [float(i) for i in range(61)])
            self.assertEqual(state["base_qvel"], [0.0, 1.0, 2.0])
            self.assertEqual(state["arms"]["left"]["qpos"], [float(i) for i in range(3, 10)])
            self.assertEqual(
                state["arms"]["left"]["eef_orientation_xyzw"],
                [20.0, 21.0, 22.0, 23.0],
            )
            self.assertEqual(state["arms"]["right"]["qpos"], [float(i) for i in range(28, 35)])
            self.assertEqual(state["trunk"]["qpos"], [53.0, 54.0, 55.0, 56.0])

    def test_hold_action_uses_current_absolute_joints_and_zero_base_velocity(self) -> None:
        with TemporaryDirectory() as temporary:
            publisher = self._publisher(Path(temporary))
            publisher.capture(_camera_message())
            action = publisher.current_hold_action()

            self.assertEqual(len(action), 23)
            self.assertEqual(action[0:3], [0.0, 0.0, 0.0])
            self.assertEqual(action[3:7], [53.0, 54.0, 55.0, 56.0])
            self.assertEqual(action[7:14], [float(i) for i in range(3, 10)])
            self.assertEqual(action[15:22], [float(i) for i in range(28, 35)])
            self.assertEqual(action[14], 1.0)
            self.assertEqual(action[22], 1.0)

    def test_missing_geometry_does_not_publish_fake_identity_transforms(self) -> None:
        with TemporaryDirectory() as temporary:
            publisher = self._publisher(Path(temporary))
            image = np.zeros((4, 5, 3), dtype=np.uint8)
            publisher.capture(
                msgpack.packb(
                    {key: image for key in (
                        "robot::robot:zed_link:Camera:0::rgb",
                        "robot::robot:left_realsense_link:Camera:0::rgb",
                        "robot::robot:right_realsense_link:Camera:0::rgb",
                    )},
                    default=_pack_array,
                )
            )
            manifest = json.loads(publisher.manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(manifest["extrinsics_calibrated"])
            self.assertIsNone(manifest["robot_state"])
            self.assertTrue(all("T_world_camera" not in view for view in manifest["views"]))

    def test_missing_camera_is_reported_without_losing_the_frame(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            publisher = self._publisher(workspace)
            image = np.zeros((4, 5, 3), dtype=np.uint8)
            publisher.capture(
                msgpack.packb({"robot::robot:zed_link:Camera:0::rgb": image}, default=_pack_array)
            )
            manifest = json.loads(publisher.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual([view["name"] for view in manifest["views"]], ["head"])
            self.assertEqual(manifest["reference_view"], "head")
            self.assertIn("left_wrist:missing", publisher.snapshot()["last_warning"])

    def test_undecodable_frame_is_reported_and_not_published(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            publisher = self._publisher(workspace)
            publisher.capture(
                msgpack.packb(
                    {"robot::robot:zed_link:Camera:0::rgb": np.zeros((2, 2), dtype=np.uint8)},
                    default=_pack_array,
                )
            )
            self.assertFalse(publisher.manifest_path.exists())
            self.assertIsNotNone(publisher.last_error)
            self.assertIn("no decodable camera view", publisher.last_error)

    def test_reset_clears_the_previous_episode_frames(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            publisher = self._publisher(workspace)
            publisher.capture(_camera_message(seed=1))
            self.assertEqual(len(publisher.snapshot()["views"]), 3)
            publisher.note_reset()
            self.assertEqual(publisher.snapshot()["episode_id"], 1)
            self.assertEqual(publisher.snapshot()["views"], [])
            self.assertEqual(publisher.snapshot()["depth_views"], [])
            self.assertIsNone(publisher.snapshot()["robot_state"])

    def test_invalid_publish_interval_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                ObservationPublisher(Path(temporary), publish_interval=0)


if __name__ == "__main__":
    unittest.main()

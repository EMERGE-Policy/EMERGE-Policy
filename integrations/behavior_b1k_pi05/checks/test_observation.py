"""Checks for the camera side-channel published alongside the wire frames."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import msgpack
import numpy as np
from PIL import Image

from integrations.behavior_b1k_pi05.observation import ObservationPublisher


def _pack_array(value):
    return {
        b"__ndarray__": True,
        b"data": value.tobytes(),
        b"dtype": value.dtype.str,
        b"shape": value.shape,
    }


def _camera_message(seed: int = 0) -> bytes:
    image = np.full((4, 5, 3), seed, dtype=np.uint8)
    return msgpack.packb(
        {
            "robot::robot:zed_link:Camera:0::rgb": image,
            "robot::robot:left_realsense_link:Camera:0::rgb": image,
            "robot::robot:right_realsense_link:Camera:0::rgb": image,
        },
        default=_pack_array,
    )


class ObservationPublisherTest(unittest.TestCase):
    """Manifest contents must be self-describing and frame-consistent."""

    def _publisher(self, workspace: Path, **kwargs) -> ObservationPublisher:
        return ObservationPublisher(workspace, publish_interval=1, **kwargs)

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

    def test_image_paths_are_revision_specific_and_pruned(self) -> None:
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            publisher = self._publisher(workspace, keep_revisions=2)
            for seed in range(3):
                publisher.capture(_camera_message(seed=seed))
            images = sorted((workspace / "artifacts/cameras/head").glob("head.*.png"))
            self.assertEqual(len(images), 2)
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

    def test_invalid_publish_interval_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                ObservationPublisher(Path(temporary), publish_interval=0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from integrations.lightnav_behavior.probe import load_head_frame


class HeadFrameTest(unittest.TestCase):
    def test_loads_exact_head_view_and_encodes_jpeg(self) -> None:
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            image_path = workspace / "artifacts/cameras/head/head.000001.png"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (8, 6), (10, 20, 30)).save(image_path)
            manifest_path = workspace / "artifacts/observations/observation.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps({
                "revision": 1,
                "observation_seq": 7,
                "episode_id": 3,
                "views": [
                    {"name": "left_wrist", "image_path": "missing.png"},
                    {"name": "head", "image_path": "artifacts/cameras/head/head.000001.png", "width": 8, "height": 6},
                ],
            }))
            frame = load_head_frame(manifest_path)
            self.assertEqual((frame.width, frame.height), (8, 6))
            self.assertEqual((frame.revision, frame.observation_seq, frame.episode_id), (1, 7, 3))
            self.assertTrue(frame.jpeg.startswith(b"\xff\xd8"))

    def test_rejects_head_image_outside_workspace(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            outside = root / "outside.png"
            Image.new("RGB", (2, 2)).save(outside)
            manifest_path = workspace / "artifacts/observations/observation.json"
            manifest_path.parent.mkdir(parents=True)
            manifest_path.write_text(json.dumps({
                "views": [{"name": "head", "image_path": "../outside.png", "width": 2, "height": 2}],
            }))
            with self.assertRaisesRegex(ValueError, "outside"):
                load_head_frame(manifest_path)


if __name__ == "__main__":
    unittest.main()

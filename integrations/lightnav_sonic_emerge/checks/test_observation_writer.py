from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from integrations.lightnav_sonic_emerge.observation_writer import EgoObservationWriter


def _jpeg_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 6), (12, 34, 56)).save(output, format="JPEG")
    return output.getvalue()


class EgoObservationWriterTest(unittest.TestCase):
    def test_writes_png_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = EgoObservationWriter(directory)
            manifest_path = writer.write_jpeg(_jpeg_bytes())

            self.assertEqual(manifest_path, Path(directory) / "artifacts/observations/observation.json")
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["revision"], 1)
            self.assertEqual(payload["reference_view"], "ego_view")
            self.assertEqual(payload["views"][0]["width"], 8)
            self.assertEqual(payload["views"][0]["height"], 6)
            image_path = Path(directory) / payload["views"][0]["image_path"]
            with Image.open(image_path) as image:
                self.assertEqual(image.format, "PNG")
                self.assertEqual(image.size, (8, 6))

    def test_disabled_writer_has_no_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = EgoObservationWriter(directory, enabled=False)
            self.assertIsNone(writer.write_jpeg(_jpeg_bytes()))
            self.assertFalse((Path(directory) / "artifacts").exists())

    def test_rejects_empty_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            writer = EgoObservationWriter(directory)
            with self.assertRaises(ValueError):
                writer.write_jpeg(b"")

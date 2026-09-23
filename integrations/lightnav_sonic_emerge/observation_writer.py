"""Atomically expose the latest SONIC ego image to EMERGE visual tools."""

from __future__ import annotations

from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from threading import Lock

from PIL import Image


class EgoObservationWriter:
    """Write one reference RGB view and its observation manifest atomically."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        view_name: str = "ego_view",
        enabled: bool = True,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.view_name = view_name
        self.enabled = bool(enabled)
        self.directory = self.workspace / "artifacts/observations"
        self.image_path = self.directory / f"{view_name}.png"
        self.manifest_path = self.directory / "observation.json"
        self._revision = 0
        self._lock = Lock()

    @property
    def revision(self) -> int:
        return self._revision

    def write_jpeg(self, jpeg: bytes) -> Path | None:
        if not self.enabled:
            return None
        if not isinstance(jpeg, bytes) or not jpeg:
            raise ValueError("observation JPEG must be non-empty bytes")

        with Image.open(BytesIO(jpeg)) as source:
            rgb = source.convert("RGB")
            width, height = rgb.size
            self.directory.mkdir(parents=True, exist_ok=True)
            with self._lock:
                image_fd, image_tmp_name = tempfile.mkstemp(
                    prefix=f".{self.view_name}.", suffix=".tmp.png", dir=self.directory
                )
                os.close(image_fd)
                image_tmp = Path(image_tmp_name)
                try:
                    rgb.save(image_tmp, format="PNG")
                    os.replace(image_tmp, self.image_path)
                finally:
                    if image_tmp.exists():
                        image_tmp.unlink()

                self._revision += 1
                manifest = {
                    "revision": self._revision,
                    "reference_view": self.view_name,
                    "views": [
                        {
                            "name": self.view_name,
                            "image_path": self.image_path.relative_to(self.workspace).as_posix(),
                            "width": width,
                            "height": height,
                        }
                    ],
                }
                manifest_fd, manifest_tmp_name = tempfile.mkstemp(
                    prefix=".observation.", suffix=".tmp.json", dir=self.directory
                )
                manifest_tmp = Path(manifest_tmp_name)
                try:
                    with os.fdopen(manifest_fd, "w", encoding="utf-8") as output:
                        json.dump(manifest, output, ensure_ascii=False, indent=2)
                        output.write("\n")
                    os.replace(manifest_tmp, self.manifest_path)
                finally:
                    if manifest_tmp.exists():
                        manifest_tmp.unlink()
        return self.manifest_path

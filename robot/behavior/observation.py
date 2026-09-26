"""Publish BEHAVIOR camera observations in EMERGE's workspace format.

The wire bytes are never modified.  This module only decodes a side copy so the
Agent can look at what the policy is looking at.

Two properties matter for correctness:

* the manifest and the PNG files it references must come from the same frame,
  so images are written to revision-specific paths before the manifest is
  published atomically, and
* the Agent must be able to tell whether its last action has already been
  applied, which is why every frame carries a monotonic sequence number.
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CAMERA_KEYS = {
    "head": "robot::robot:zed_link:Camera:0::rgb",
    "left_wrist": "robot::robot:left_realsense_link:Camera:0::rgb",
    "right_wrist": "robot::robot:right_realsense_link:Camera:0::rgb",
}

MANIFEST_SCHEMA = "Emerge.behavior_camera_observation.v1"

#: How many published revisions of each camera to keep on disk.
DEFAULT_KEEP_REVISIONS = 4


def _unpack_array(item: dict[Any, Any]):
    import numpy as np

    marker = item.get(b"__ndarray__", item.get("__ndarray__"))
    if marker:
        data = item.get(b"data", item.get("data"))
        dtype = item.get(b"dtype", item.get("dtype"))
        shape = item.get(b"shape", item.get("shape"))
        return np.ndarray(buffer=data, dtype=np.dtype(dtype), shape=shape)
    generic = item.get(b"__npgeneric__", item.get("__npgeneric__"))
    if generic:
        data = item.get(b"data", item.get("data"))
        dtype = item.get(b"dtype", item.get("dtype"))
        return np.dtype(dtype).type(data)
    return item


def _as_rgb_uint8(value: Any):
    import numpy as np

    image = np.asarray(value)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"expected an HWC camera image, got shape {image.shape}")
    if image.shape[-1] not in (3, 4) and image.shape[0] in (3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] < 3:
        raise ValueError(f"expected at least three image channels, got shape {image.shape}")
    image = image[..., :3]
    if image.dtype.kind == "f":
        finite_max = float(np.nanmax(image)) if image.size else 0.0
        if finite_max <= 1.0:
            image = image * 255.0
    return np.ascontiguousarray(np.clip(image, 0, 255).astype(np.uint8))


def _approximate_intrinsics(width: int, height: int, horizontal_fov_degrees: float = 40.0):
    focal = width / (2.0 * math.tan(math.radians(horizontal_fov_degrees) / 2.0))
    return [
        [focal, 0.0, (width - 1) / 2.0],
        [0.0, focal, (height - 1) / 2.0],
        [0.0, 0.0, 1.0],
    ]


class ObservationPublisher:
    """Decode observations only for diagnostics while preserving wire bytes."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        publish_interval: int = 5,
        keep_revisions: int = DEFAULT_KEEP_REVISIONS,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.publish_interval = int(publish_interval)
        if self.publish_interval < 1:
            raise ValueError("publish_interval must be positive")
        self.keep_revisions = max(1, int(keep_revisions))
        self.manifest_path = self.workspace / "artifacts/observations/observation.json"
        self._observation_count = 0
        self._revision = 0
        self._episode_id = 0
        self._latest: dict[str, Any] = {}
        self._captured_at: str | None = None
        self._latest_captured_at: str | None = None
        self._latest_observation_count = 0
        self.last_error: str | None = None
        self.last_warning: str | None = None

    # -- read-only views ---------------------------------------------------- #

    @property
    def observation_count(self) -> int:
        """Monotonic count of every decoded observation frame."""
        return self._observation_count

    @property
    def revision(self) -> int:
        """Monotonic count of published manifests."""
        return self._revision

    @property
    def episode_id(self) -> int:
        return self._episode_id

    def note_reset(self) -> None:
        """Start a new episode; frames from the previous one must not leak."""
        self._episode_id += 1
        self._latest = {}
        self._latest_captured_at = None
        self._latest_observation_count = 0
        self.last_error = None
        self.last_warning = None

    # -- ingest ------------------------------------------------------------- #

    def capture(self, message: Any) -> None:
        if not isinstance(message, (bytes, bytearray, memoryview)):
            return
        try:
            import msgpack

            payload = msgpack.unpackb(message, raw=False, object_hook=_unpack_array)
            if not isinstance(payload, dict) or payload.get("reset") is True:
                return
            images: dict[str, Any] = {}
            failures: list[str] = []
            for name, key in CAMERA_KEYS.items():
                if key not in payload:
                    failures.append(f"{name}:missing")
                    continue
                try:
                    images[name] = _as_rgb_uint8(payload[key]).copy()
                except Exception as exc:  # a single bad view must not lose the frame
                    failures.append(f"{name}:{type(exc).__name__}")
            if not images:
                if failures:
                    self.last_error = "no decodable camera view (" + ", ".join(failures) + ")"
                return
            self._observation_count += 1
            self._captured_at = _utc_now()
            self._latest = images
            self._latest_captured_at = self._captured_at
            self._latest_observation_count = self._observation_count
            self.last_warning = ", ".join(failures) if failures else None
            if self._observation_count == 1 or self._observation_count % self.publish_interval == 0:
                self.publish_latest()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"

    # -- publish ------------------------------------------------------------ #

    def publish_latest(self) -> Path | None:
        if not self._latest:
            return None
        revision = self._revision + 1
        try:
            from PIL import Image

            views = []
            for name in CAMERA_KEYS:
                image = self._latest.get(name)
                if image is None:
                    continue
                height, width = image.shape[:2]
                image_dir = self.workspace / "artifacts/cameras" / name
                image_dir.mkdir(parents=True, exist_ok=True)
                image_path = image_dir / f"{name}.{revision:06d}.png"
                temporary_image = image_path.with_name(
                    f".{image_path.stem}.{os.getpid()}.tmp.png"
                )
                Image.fromarray(image, mode="RGB").save(temporary_image, format="PNG")
                os.replace(temporary_image, image_path)
                views.append(
                    {
                        "name": name,
                        "image_path": image_path.relative_to(self.workspace).as_posix(),
                        "width": width,
                        "height": height,
                        "intrinsics": _approximate_intrinsics(width, height),
                        "T_world_camera": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        "calibrated": False,
                        "source_observation_key": CAMERA_KEYS[name],
                    }
                )
            if not views:
                return None
            manifest = {
                "schema_version": MANIFEST_SCHEMA,
                "revision": revision,
                "observation_seq": self._latest_observation_count,
                "episode_id": self._episode_id,
                "captured_at": self._latest_captured_at or _utc_now(),
                "reference_view": "head" if "head" in self._latest else views[0]["name"],
                "calibrated": False,
                "views": views,
            }
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_manifest = self.manifest_path.with_name(
                f".{self.manifest_path.stem}.{os.getpid()}.tmp.json"
            )
            temporary_manifest.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_manifest, self.manifest_path)
            self._revision = revision
            self.last_error = None
            for name in CAMERA_KEYS:
                _prune_revisions(
                    self.workspace / "artifacts/cameras" / name,
                    name,
                    keep=self.keep_revisions,
                )
            return self.manifest_path
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return None

    def snapshot(self) -> dict[str, Any]:
        return {
            "observation_count": self._observation_count,
            "observation_seq": self._observation_count,
            "revision": self._revision,
            "episode_id": self._episode_id,
            "captured_at": self._latest_captured_at,
            "manifest_path": str(self.manifest_path),
            "views": list(self._latest),
            "last_error": self.last_error,
            "last_warning": self.last_warning,
        }


def _prune_revisions(directory: Path, name: str, *, keep: int) -> None:
    """Keep only the newest ``keep`` published revisions of one camera."""
    try:
        revisions = sorted(directory.glob(f"{name}.*.png"))
    except OSError:
        return
    for stale in revisions[:-keep]:
        try:
            stale.unlink()
        except OSError:
            pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

"""Read the current calibrated camera observation from the workspace."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class CameraView:
    name: str
    image_path: Path
    rgb: np.ndarray
    intrinsics: np.ndarray
    T_world_camera: np.ndarray

    def image_data_url(self) -> str:
        encoded = base64.b64encode(self.image_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"


@dataclass(frozen=True, slots=True)
class CameraObservation:
    reference_view: str
    views: tuple[CameraView, ...]


class ObservationStore:
    """Keep one camera snapshot for the duration of a sub-agent run."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.manifest_path = (
            self.workspace / "artifacts/observations/observation.json"
        )
        self._current: CameraObservation | None = None

    def reset(self) -> None:
        self._current = None

    def load(self) -> CameraObservation:
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        views = []
        for item in payload["views"]:
            image_path = self.workspace / item["image_path"]
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            views.append(
                CameraView(
                    name=str(item["name"]),
                    image_path=image_path,
                    rgb=np.ascontiguousarray(rgb),
                    intrinsics=np.asarray(item["intrinsics"], dtype=np.float64),
                    T_world_camera=np.asarray(
                        item["T_world_camera"],
                        dtype=np.float64,
                    ),
                )
            )
        self._current = CameraObservation(
            reference_view=str(payload["reference_view"]),
            views=tuple(views),
        )
        return self._current

    def current(self) -> CameraObservation:
        if self._current is None:
            raise RuntimeError(
                "No scene observation is loaded. Call observe_scene before "
                "segment_candidates."
            )
        return self._current

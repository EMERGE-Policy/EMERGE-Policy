"""Read current camera images for one verification run."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class CameraView:
    name: str
    image_path: Path

    def image_data_url(self) -> str:
        encoded = base64.b64encode(self.image_path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"


@dataclass(frozen=True, slots=True)
class CameraObservation:
    reference_view: str
    views: tuple[CameraView, ...]


class ObservationStore:
    """Keep one current camera snapshot for a verification run."""

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
        views = tuple(
            CameraView(
                name=str(item["name"]),
                image_path=self.workspace / item["image_path"],
            )
            for item in payload["views"]
        )
        self._current = CameraObservation(
            reference_view=str(payload["reference_view"]),
            views=views,
        )
        return self._current

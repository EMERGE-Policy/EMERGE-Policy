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
        self.default_manifest_path = (
            self.workspace / "artifacts/observations/observation.json"
        )
        self.verification_manifest_path = (
            self.workspace / "artifacts/verification_observations/observation.json"
        )
        self._current: CameraObservation | None = None

    @property
    def manifest_path(self) -> Path:
        """Prefer observer views when the active benchmark publishes them."""

        if self.verification_manifest_path.is_file():
            return self.verification_manifest_path
        return self.default_manifest_path

    def reset(self) -> None:
        self._current = None

    def load(self) -> CameraObservation:
        manifest_path = self.manifest_path
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("verification observation manifest must be a JSON object")
        views = tuple(
            CameraView(
                name=str(item["name"]),
                image_path=self.workspace / item["image_path"],
            )
            for item in payload["views"]
        )
        if not views:
            raise ValueError("verification observation requires at least one camera view")
        names = [view.name for view in views]
        if len(names) != len(set(names)):
            raise ValueError("verification camera view names must be unique")
        reference_view = str(payload["reference_view"])
        if reference_view not in names:
            raise ValueError(
                f"verification reference_view {reference_view!r} is not present in {manifest_path}"
            )
        if payload.get("schema_version") == "Emerge.behavior_verification_observation.v1":
            expected = ["room_overview", "table_overhead", "table_side", "head"]
            if names != expected:
                raise ValueError(
                    f"BEHAVIOR verification observation must contain {expected}, got {names}"
                )
        self._current = CameraObservation(
            reference_view=reference_view,
            views=views,
        )
        return self._current

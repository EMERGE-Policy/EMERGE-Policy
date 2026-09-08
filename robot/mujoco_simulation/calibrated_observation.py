"""Save the current calibrated camera observation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from robot.mujoco_simulation.mujoco_camera import MujocoCamera


class CalibratedObservationWriter:
    """Overwrite camera images and their shared observation JSON."""

    def __init__(
        self,
        workspace: str | Path,
        cameras: Iterable[MujocoCamera],
    ) -> None:
        self._workspace = Path(workspace).expanduser().resolve()
        self._cameras = tuple(
            camera
            for camera in cameras
            if camera.get_observation_output().enabled
        )
        if not self._cameras:
            raise ValueError("calibrated observation requires at least one camera")
        reference_views = [
            camera.get_name()
            for camera in self._cameras
            if camera.get_observation_output().reference
        ]
        if len(reference_views) != 1:
            raise ValueError(
                "saved camera observations require exactly one reference camera"
            )
        self._reference_view = reference_views[0]
        self._revision = 0
        self.manifest_path = (
            self._workspace / "artifacts/observations/observation.json"
        )

    def write(self) -> Path:
        import cv2

        self._revision += 1
        views: list[dict[str, object]] = []
        for camera in self._cameras:
            output = camera.get_observation_output()
            image_path = output.directory / f"{camera.get_name()}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image = cv2.cvtColor(camera.get_rgb(), cv2.COLOR_RGB2BGR)
            temporary_image = image_path.with_name(
                f".{image_path.stem}.{os.getpid()}.tmp.png"
            )
            if not cv2.imwrite(str(temporary_image), image):
                raise OSError(f"failed to write camera image: {image_path}")
            os.replace(temporary_image, image_path)
            views.append(
                {
                    "name": camera.get_name(),
                    "image_path": image_path.relative_to(
                        self._workspace
                    ).as_posix(),
                    "width": camera.get_width(),
                    "height": camera.get_height(),
                    "intrinsics": camera.get_intrinsics().astype(float).tolist(),
                    "T_world_camera": (
                        camera.get_world_camera_transform().astype(float).tolist()
                    ),
                }
            )

        manifest = {
            "revision": self._revision,
            "reference_view": self._reference_view,
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
        return self.manifest_path

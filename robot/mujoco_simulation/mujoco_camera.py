"""MuJoCo camera capture, calibration, and MP4 recording."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True, frozen=True)
class MujocoCameraInfo:
    id: int
    fovy_deg: float


@dataclass(slots=True, frozen=True)
class MujocoCameraObservationOutput:
    enabled: bool
    directory: Path
    reference: bool


class MujocoCamera:
    """Read and interpret one camera already present in the MuJoCo model."""

    def __init__(
        self,
        environment: Any,
        *,
        name: str,
        workspace: str | Path | None = None,
    ) -> None:
        self._environment = environment
        self.name = name
        self._workspace = Path(workspace or Path.cwd()).expanduser().resolve()
        self._info = self._read_camera_info()
        resolver = getattr(self._environment, "get_camera_runtime_settings", None)
        if not callable(resolver):
            raise RuntimeError("environment does not provide camera runtime settings")
        self._runtime_settings_resolver = resolver
        settings = dict(self._runtime_settings_resolver(self.name))
        self._render_width = int(settings["width"])
        self._render_height = int(settings["height"])
        self._depth_enabled = bool(settings.get("depth", False))
        self._flip_vertical = bool(settings.get("flip_vertical", True))

        observation_config = dict(settings.get("observation") or {})
        observation_directory = self._resolve_output_path(
            observation_config.get(
                "directory",
                f"artifacts/cameras/{self.name}",
            )
        )
        if not observation_directory.is_relative_to(self._workspace):
            raise ValueError("camera observation directory must be inside workspace")
        self._observation_output = MujocoCameraObservationOutput(
            enabled=bool(observation_config.get("enabled", False)),
            directory=observation_directory,
            reference=bool(observation_config.get("reference", False)),
        )

        recording_config = dict(settings.get("recording") or {})
        self._recording_enabled = bool(recording_config.get("enabled", False))
        self._recording_path = self._resolve_output_path(
            recording_config.get("path", f"artifacts/cameras/{name}.mp4")
        )
        self._recording_fps = float(recording_config.get("fps", 20.0))
        self._recording_codec = str(recording_config.get("codec", "mp4v"))
        self._video_writer: Any | None = None

        self._rgb = np.empty(
            (self._render_height, self._render_width, 3), dtype=np.uint8
        )
        self._depth: np.ndarray | None = None
        self._intrinsics_matrix = np.eye(3, dtype=np.float64)
        self._world_camera_transform = np.eye(4, dtype=np.float64)
        self.refresh()

    def get_rgb(self) -> np.ndarray:
        return self._rgb

    def get_intrinsics(self) -> np.ndarray:
        return self._intrinsics_matrix

    def get_name(self) -> str:
        return self.name

    def get_observation_output(self) -> MujocoCameraObservationOutput:
        return self._observation_output

    def get_world_camera_transform(self) -> np.ndarray:
        return self._world_camera_transform

    def get_width(self) -> int:
        return int(self._rgb.shape[1])

    def get_height(self) -> int:
        return int(self._rgb.shape[0])

    def is_recording(self) -> bool:
        return self._recording_enabled

    def refresh(self) -> None:
        settings = dict(self._runtime_settings_resolver(self.name))
        self._render_width = int(settings["width"])
        self._render_height = int(settings["height"])
        self._depth_enabled = bool(settings.get("depth", False))
        self._flip_vertical = bool(settings.get("flip_vertical", True))
        rendered = self._environment.sim.render(
            camera_name=self.name,
            width=self._render_width,
            height=self._render_height,
            depth=self._depth_enabled,
        )
        if self._depth_enabled:
            rgb, depth = rendered
        else:
            rgb, depth = rendered, None

        self._rgb = np.asarray(rgb, dtype=np.uint8)
        self._depth = None if depth is None else np.asarray(depth)
        if self._flip_vertical:
            self._rgb = self._rgb[::-1]
            if self._depth is not None:
                self._depth = self._depth[::-1]
        self._rgb = np.ascontiguousarray(self._rgb)
        self._info = self._read_camera_info()
        self._intrinsics_matrix = self._intrinsics()
        self._world_camera_transform = self._world_camera_pose()

        if self._recording_enabled:
            if self._video_writer is None:
                self._open_video_writer()
            import cv2

            self._video_writer.write(cv2.cvtColor(self._rgb, cv2.COLOR_RGB2BGR))

    def stop_recording(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
        self._video_writer = None
        self._recording_enabled = False

    def close(self) -> None:
        self.stop_recording()

    def _open_video_writer(self) -> None:
        import cv2

        self._recording_path.parent.mkdir(parents=True, exist_ok=True)
        self._video_writer = cv2.VideoWriter(
            str(self._recording_path),
            cv2.VideoWriter_fourcc(*self._recording_codec),
            self._recording_fps,
            (self.get_width(), self.get_height()),
        )
        if not self._video_writer.isOpened():
            raise OSError(f"failed to open MP4 writer: {self._recording_path}")

    def _resolve_output_path(self, path: str | Path) -> Path:
        """Resolve relative camera artifacts inside the agent workspace."""
        output = Path(path).expanduser()
        if output.is_absolute():
            return output
        return (self._workspace / output).resolve()

    def _intrinsics(self) -> np.ndarray:
        fovy = self._read_camera_info().fovy_deg
        height = float(self.get_height())
        width = float(self.get_width())
        focal = 0.5 * height / np.tan(np.deg2rad(fovy) / 2.0)
        return np.array(
            [
                [focal, 0.0, width / 2.0],
                [0.0, focal, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def _world_camera_pose(self) -> np.ndarray:
        sim = self._environment.sim
        camera_id = self._read_camera_info().id
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.asarray(sim.data.cam_xmat[camera_id]).reshape(3, 3)
        transform[:3, 3] = np.asarray(sim.data.cam_xpos[camera_id], dtype=np.float64)
        return transform @ np.diag([1.0, -1.0, -1.0, 1.0])

    def _read_camera_info(self) -> MujocoCameraInfo:
        sim = self._environment.sim
        camera_id = int(sim.model.camera_name2id(self.name))
        return MujocoCameraInfo(
            id=camera_id,
            fovy_deg=float(sim.model.cam_fovy[camera_id]),
        )

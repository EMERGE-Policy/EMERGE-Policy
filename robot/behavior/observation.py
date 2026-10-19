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
import os
import copy
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CAMERA_KEYS = {
    "head": "robot::robot:zed_link:Camera:0::rgb",
    "left_wrist": "robot::robot:left_realsense_link:Camera:0::rgb",
    "right_wrist": "robot::robot:right_realsense_link:Camera:0::rgb",
}

DEPTH_KEYS = {
    name: key.removesuffix("::rgb") + "::depth_linear"
    for name, key in CAMERA_KEYS.items()
}

# BatchedEvaluator concatenates camera poses in the order declared by
# eval.camera_sensor_names in dev/behavior_r1pro_openpi.yaml.
CAMERA_POSE_ORDER = ("left_wrist", "right_wrist", "head")
CAMERA_POSE_KEY = "robot::cam_rel_poses"
PROPRIO_KEYS = ("robot::proprio", "robot::robot:proprio")
R1PRO_PROPRIO_DIM = 61

MANIFEST_SCHEMA = "Emerge.behavior_camera_observation.v3"

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


def _as_depth_float32(value: Any):
    """Normalize one batched or channelized linear-depth image to HxW metres."""
    import numpy as np

    depth = np.asarray(value)
    while depth.ndim > 2 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"expected an HW depth image, got shape {depth.shape}")
    if depth.dtype.kind not in "fiu":
        raise ValueError(f"expected numeric linear depth, got dtype {depth.dtype}")
    return np.ascontiguousarray(depth.astype(np.float32, copy=False))


def _scaled_intrinsics(camera: dict[str, Any], width: int, height: int) -> list[list[float]]:
    """Scale a pinhole calibration when the published image was resized."""
    reference_width, reference_height = camera["resolution"]
    scale_x = float(width) / float(reference_width)
    scale_y = float(height) / float(reference_height)
    matrix = camera["intrinsics"]
    return [
        [float(matrix[0][0]) * scale_x, 0.0, float(matrix[0][2]) * scale_x],
        [0.0, float(matrix[1][1]) * scale_y, float(matrix[1][2]) * scale_y],
        [0.0, 0.0, 1.0],
    ]


def _as_vector(value: Any, *, name: str):
    import numpy as np

    vector = np.asarray(value, dtype=np.float64)
    while vector.ndim > 1 and vector.shape[0] == 1:
        vector = vector[0]
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a vector, got shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector


def _pose_matrix(position, quaternion):
    import numpy as np

    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if position.shape != (3,) or quaternion.shape != (4,) or norm < 1e-12:
        raise ValueError("camera pose must contain xyz and a non-zero xyzw quaternion")
    x, y, z, w = quaternion / norm
    rotation = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    # OmniGibson reports the camera prim pose, while pinhole backprojection
    # uses the optical convention. Match the official depth_to_pcd helper by
    # applying its 180-degree rotation about camera X.
    transform[:3, :3] = rotation @ np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = position
    return transform


def _camera_transforms(payload: dict[str, Any]) -> dict[str, list[list[float]]]:
    if CAMERA_POSE_KEY not in payload:
        return {}
    poses = _as_vector(payload[CAMERA_POSE_KEY], name=CAMERA_POSE_KEY)
    expected = 7 * len(CAMERA_POSE_ORDER)
    if poses.size != expected:
        raise ValueError(f"{CAMERA_POSE_KEY} must have {expected} values, got {poses.size}")
    return {
        name: _pose_matrix(pose[:3], pose[3:]).tolist()
        for name, pose in zip(CAMERA_POSE_ORDER, poses.reshape(-1, 7), strict=True)
    }


def _r1pro_robot_state(payload: dict[str, Any]) -> dict[str, Any] | None:
    source_key = next((key for key in PROPRIO_KEYS if key in payload), None)
    if source_key is None:
        return None
    state = _as_vector(payload[source_key], name=source_key)
    if state.size != R1PRO_PROPRIO_DIM:
        raise ValueError(
            f"{source_key} must have {R1PRO_PROPRIO_DIM} values, got {state.size}"
        )

    def values(start: int, stop: int) -> list[float]:
        return state[start:stop].astype(float).tolist()

    return {
        "source_observation_key": source_key,
        "coordinate_frame": "robot_base",
        "proprio_dim": R1PRO_PROPRIO_DIM,
        "proprio": state.astype(float).tolist(),
        "base_qvel": values(0, 3),
        "trunk": {"qpos": values(53, 57), "qvel": values(57, 61)},
        "arms": {
            "left": {
                "qpos": values(3, 10),
                "qvel": values(10, 17),
                "eef_position": values(17, 20),
                "eef_orientation_xyzw": values(20, 24),
                "gripper_qpos": values(24, 26),
                "gripper_qvel": values(26, 28),
            },
            "right": {
                "qpos": values(28, 35),
                "qvel": values(35, 42),
                "eef_position": values(42, 45),
                "eef_orientation_xyzw": values(45, 49),
                "gripper_qpos": values(49, 51),
                "gripper_qvel": values(51, 53),
            },
        },
    }


class ObservationPublisher:
    """Decode observations only for diagnostics while preserving wire bytes."""

    def __init__(
        self,
        workspace: str | Path,
        *,
        publish_interval: int = 5,
        keep_revisions: int = DEFAULT_KEEP_REVISIONS,
        camera_calibration: dict[str, Any] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.publish_interval = int(publish_interval)
        if self.publish_interval < 1:
            raise ValueError("publish_interval must be positive")
        self.keep_revisions = max(1, int(keep_revisions))
        self.camera_calibration = camera_calibration
        self.manifest_path = self.workspace / "artifacts/observations/observation.json"
        # Capture runs in the gateway thread while navigation may request an
        # immediate publish from another thread.  They must never write the
        # same revision/temporary paths concurrently.
        self._lock = threading.RLock()
        self._observation_count = 0
        self._revision = 0
        self._episode_id = 0
        self._latest: dict[str, Any] = {}
        self._latest_depths: dict[str, Any] = {}
        self._latest_camera_transforms: dict[str, list[list[float]]] = {}
        self._latest_robot_state: dict[str, Any] | None = None
        self._captured_at: str | None = None
        self._latest_captured_at: str | None = None
        self._latest_observation_count = 0
        self.last_error: str | None = None
        self.last_warning: str | None = None

    # -- read-only views ---------------------------------------------------- #

    @property
    def observation_count(self) -> int:
        """Monotonic count of every decoded observation frame."""
        with self._lock:
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
        with self._lock:
            self._note_reset_locked()

    def _note_reset_locked(self) -> None:
        self._episode_id += 1
        self._latest = {}
        self._latest_depths = {}
        self._latest_camera_transforms = {}
        self._latest_robot_state = None
        self._latest_captured_at = None
        self._latest_observation_count = 0
        self.last_error = None
        self.last_warning = None

    # -- ingest ------------------------------------------------------------- #

    def capture(self, message: Any) -> None:
        with self._lock:
            self._capture_locked(message)

    def _capture_locked(self, message: Any) -> None:
        if not isinstance(message, (bytes, bytearray, memoryview)):
            return
        try:
            import msgpack

            payload = msgpack.unpackb(message, raw=False, object_hook=_unpack_array)
            if not isinstance(payload, dict) or payload.get("reset") is True:
                return
            images: dict[str, Any] = {}
            depths: dict[str, Any] = {}
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
            for name, key in DEPTH_KEYS.items():
                if key not in payload:
                    failures.append(f"{name}:depth_missing")
                    continue
                try:
                    depth = _as_depth_float32(payload[key]).copy()
                    image = images.get(name)
                    if image is None:
                        failures.append(f"{name}:depth_without_rgb")
                        continue
                    if depth.shape != image.shape[:2]:
                        failures.append(
                            f"{name}:depth_shape_{depth.shape}_rgb_shape_{image.shape[:2]}"
                        )
                        continue
                    depths[name] = depth
                except Exception as exc:
                    failures.append(f"{name}:depth_{type(exc).__name__}")
            try:
                camera_transforms = _camera_transforms(payload)
            except ValueError as exc:
                camera_transforms = {}
                failures.append(f"camera_poses:{exc}")
            try:
                robot_state = _r1pro_robot_state(payload)
            except ValueError as exc:
                robot_state = None
                failures.append(f"proprio:{exc}")
            self._observation_count += 1
            self._captured_at = _utc_now()
            self._latest = images
            self._latest_depths = depths
            self._latest_camera_transforms = camera_transforms
            self._latest_robot_state = robot_state
            self._latest_captured_at = self._captured_at
            self._latest_observation_count = self._observation_count
            self.last_warning = ", ".join(failures) if failures else None
            if self._observation_count == 1 or self._observation_count % self.publish_interval == 0:
                self.publish_latest()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"

    # -- publish ------------------------------------------------------------ #

    def publish_latest(self) -> Path | None:
        with self._lock:
            return self._publish_latest_locked()

    def _publish_latest_locked(self) -> Path | None:
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
                depth = self._latest_depths.get(name)
                depth_path = None
                if depth is not None:
                    depth_dir = self.workspace / "artifacts/depth" / name
                    depth_dir.mkdir(parents=True, exist_ok=True)
                    depth_path = depth_dir / f"{name}.{revision:06d}.npy"
                    temporary_depth = depth_path.with_name(
                        f".{depth_path.stem}.{os.getpid()}.tmp.npy"
                    )
                    import numpy as np

                    with temporary_depth.open("wb") as stream:
                        np.save(stream, depth, allow_pickle=False)
                    os.replace(temporary_depth, depth_path)
                configured_camera = (
                    self.camera_calibration.get("cameras", {}).get(name)
                    if self.camera_calibration
                    else None
                )
                view = {
                    "name": name,
                    "image_path": image_path.relative_to(self.workspace).as_posix(),
                    "width": width,
                    "height": height,
                    "intrinsics": (
                        _scaled_intrinsics(configured_camera, width, height)
                        if configured_camera
                        else None
                    ),
                    "intrinsics_calibrated": configured_camera is not None,
                    "extrinsics_calibrated": name in self._latest_camera_transforms,
                    "calibrated": configured_camera is not None
                    and name in self._latest_camera_transforms,
                    "source_observation_key": CAMERA_KEYS[name],
                    "depth_available": depth_path is not None,
                }
                if depth_path is not None:
                    view.update(
                        {
                            "depth_path": depth_path.relative_to(self.workspace).as_posix(),
                            "depth_unit": "meter",
                            "source_depth_observation_key": DEPTH_KEYS[name],
                        }
                    )
                if name in self._latest_camera_transforms:
                    # Existing localization services call this field
                    # T_world_camera. For BEHAVIOR its declared world frame is
                    # the current robot base, as recorded by coordinate_frame.
                    view["T_world_camera"] = self._latest_camera_transforms[name]
                views.append(view)
            if not views:
                return None
            manifest = {
                "schema_version": MANIFEST_SCHEMA,
                "revision": revision,
                "observation_seq": self._latest_observation_count,
                "episode_id": self._episode_id,
                "captured_at": self._latest_captured_at or _utc_now(),
                "reference_view": "head" if "head" in self._latest else views[0]["name"],
                "coordinate_frame": "robot_base",
                "camera_pose_source": CAMERA_POSE_KEY,
                "extrinsics_calibrated": bool(views)
                and all(view["extrinsics_calibrated"] for view in views),
                "intrinsics_calibrated": bool(views)
                and all(view["intrinsics_calibrated"] for view in views),
                "calibration_source": (
                    self.camera_calibration.get("source") if self.camera_calibration else None
                ),
                "calibrated": bool(views) and all(view["calibrated"] for view in views),
                "rgbd": bool(views) and all(view["depth_available"] for view in views),
                "views": views,
                "robot_state": self._latest_robot_state,
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
                    suffix="png",
                )
                _prune_revisions(
                    self.workspace / "artifacts/depth" / name,
                    name,
                    keep=self.keep_revisions,
                    suffix="npy",
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
            "depth_views": list(self._latest_depths),
            "rgbd": bool(self._latest)
            and all(name in self._latest_depths for name in self._latest),
            "coordinate_frame": "robot_base",
            "extrinsics_calibrated": bool(self._latest)
            and all(name in self._latest_camera_transforms for name in self._latest),
            "intrinsics_calibrated": bool(self.camera_calibration)
            and all(
                name in self.camera_calibration.get("cameras", {})
                for name in self._latest
            ),
            "robot_state": self._latest_robot_state,
            "last_error": self.last_error,
            "last_warning": self.last_warning,
        }

    def current_hold_action(self) -> list[float]:
        """Build one legal 23-D R1Pro action that holds the latest joint pose."""
        state = self._latest_robot_state
        if state is None:
            raise RuntimeError("no R1Pro proprioception has been observed yet")

        def gripper_command(values: list[float]) -> float:
            # The official smooth gripper controller maps [-1, 1] onto a
            # per-finger [0.0, 0.05] position. Proprio stores both fingers.
            width = float(sum(values))
            return max(-1.0, min(1.0, 2.0 * width / 0.1 - 1.0))

        left = state["arms"]["left"]
        right = state["arms"]["right"]
        action = (
            [0.0, 0.0, 0.0]
            + list(state["trunk"]["qpos"])
            + list(left["qpos"])
            + [gripper_command(left["gripper_qpos"])]
            + list(right["qpos"])
            + [gripper_command(right["gripper_qpos"])]
        )
        if len(action) != 23:
            raise RuntimeError(f"internal R1Pro hold action has {len(action)} values, expected 23")
        return [float(value) for value in action]

    def current_robot_state(self) -> dict[str, Any]:
        state = self._latest_robot_state
        if state is None:
            raise RuntimeError("no R1Pro proprioception has been observed yet")
        return copy.deepcopy(state)


def _prune_revisions(
    directory: Path,
    name: str,
    *,
    keep: int,
    suffix: str,
) -> None:
    """Keep only the newest ``keep`` published revisions of one camera."""
    try:
        revisions = sorted(directory.glob(f"{name}.*.{suffix}"))
    except OSError:
        return
    for stale in revisions[:-keep]:
        try:
            stale.unlink()
        except OSError:
            pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

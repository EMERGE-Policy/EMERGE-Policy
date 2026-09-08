"""Lightweight OpenPI executor that steps LIBERO action7 directly."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

from loguru import logger
import numpy as np


@dataclass(slots=True)
class VLAResult:
    success: bool
    total_steps: int
    reason: str
    error_message: str | None = None
    last_gripper_command: float | None = None


class VLAExecutor:
    """Run observe-infer-step chunks without PyBullet IK or joint targets."""

    def __init__(
        self,
        environment: Any,
        *,
        config: dict[str, Any] | None = None,
        client: Any | None = None,
    ) -> None:
        self._environment = environment
        self.config = dict(config or {})
        self._client = client
        self._owns_client = client is None
        self.base_image_key = str(
            self.config.get("base_image_key", "agentview_image")
        )
        self.wrist_image_key = str(
            self.config.get("wrist_image_key", "robot0_eye_in_hand_image")
        )
        self.resize_size = int(self.config.get("resize_size", 224))
        self.replan_steps = max(1, int(self.config.get("replan_steps", 5)))
        self.stop_on_success = bool(self.config.get("stop_on_success", True))

    def execute(
        self,
        instruction: str,
        step: int,
        *,
        cancel_check: Callable[[], str | None] | None = None,
    ) -> VLAResult:
        """Run the VLA policy for `step` action steps, then return control.

        Returns as done either when the requested step budget is used up or when
        the task goal is reached earlier. Both are normal completions.
        """
        client = self._get_client()
        health_check = getattr(client, "health_check", None)
        if callable(health_check) and not health_check():
            return VLAResult(False, 0, "server_unavailable", "policy server health check failed")
        action_count = 0
        last_gripper: float | None = None
        inference_count = 0
        inference_wall_seconds = 0.0
        server_infer_seconds = 0.0
        server_queue_seconds = 0.0
        execution_started = time.perf_counter()
        try:
            while True:
                cancel_reason = cancel_check() if cancel_check else None
                if cancel_reason:
                    return VLAResult(
                        False,
                        action_count,
                        "interrupted",
                        cancel_reason,
                        last_gripper,
                    )
                inference_started = time.perf_counter()
                result = client.infer(self._build_element(instruction))
                inference_wall_seconds += time.perf_counter() - inference_started
                inference_count += 1
                server_timing = result.get("server_timing")
                if isinstance(server_timing, dict):
                    server_infer_seconds += (
                        float(server_timing.get("infer_ms", 0.0)) / 1000.0
                    )
                    server_queue_seconds += (
                        float(server_timing.get("queue_ms", 0.0)) / 1000.0
                    )
                chunk = np.asarray(result.get("actions"), dtype=np.float32)
                if chunk.ndim != 2 or chunk.shape[1] != 7:
                    raise ValueError(
                        f"openpi returned action shape {chunk.shape}, expected (chunk, 7)"
                    )
                execute_count = min(len(chunk), self.replan_steps)
                for action in chunk[:execute_count]:
                    cancel_reason = cancel_check() if cancel_check else None
                    if cancel_reason:
                        return VLAResult(
                            False,
                            action_count,
                            "interrupted",
                            cancel_reason,
                            last_gripper,
                        )
                    _, _, done, _ = self._environment.step(
                        action,
                        publish_observation=True,
                    )
                    action_count += 1
                    # Remember the gripper command VLA last issued so a
                    # following rule action can keep holding instead of
                    # re-deriving a slack signal from gripper qpos.
                    last_gripper = float(np.clip(action[6], -1.0, 1.0))
                    if self.stop_on_success and self._environment.check_success():
                        return VLAResult(True, action_count, "goal_reached", None, last_gripper)
                    if done:
                        return VLAResult(False, action_count, "environment_done", None, last_gripper)
                    if action_count >= step:
                        return VLAResult(True, action_count, "step_completed", None, last_gripper)
        except Exception as exc:
            return VLAResult(
                False,
                action_count,
                "error",
                f"{type(exc).__name__}: {exc}",
                last_gripper,
            )
        finally:
            if action_count:
                self._environment.refresh_cameras()
            logger.info(
                "VLA timing | total={:.3f}s actions={} openpi_calls={} "
                "openpi_wall={:.3f}s server_infer={:.3f}s server_queue={:.3f}s",
                time.perf_counter() - execution_started,
                action_count,
                inference_count,
                inference_wall_seconds,
                server_infer_seconds,
                server_queue_seconds,
            )

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
        self._client = None

    def _build_element(
        self,
        instruction: str,
    ) -> dict[str, Any]:
        obs = self._environment.latest_obs
        missing = [
            key
            for key in (
                self.base_image_key,
                self.wrist_image_key,
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            )
            if key not in obs
        ]
        if missing:
            raise KeyError(f"LIBERO observation is missing: {', '.join(missing)}")
        base_image = self._prepare_image(obs[self.base_image_key])
        wrist_image = self._prepare_image(obs[self.wrist_image_key])
        state = np.concatenate(
            (
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
                self._quat2axisangle(obs["robot0_eef_quat"]),
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
            )
        ).astype(np.float32)
        return {
            "observation/image": base_image,
            "observation/wrist_image": wrist_image,
            "observation/state": state,
            "prompt": str(instruction),
        }

    @staticmethod
    def _quat2axisangle(quat: Any) -> np.ndarray:
        """Match robosuite / openpi LIBERO state encoding exactly.

        Copied from openpi examples/libero/main.py so the policy receives the
        same axis-angle convention it was trained on (no w>=0 hemisphere flip).
        """
        quat = np.asarray(quat, dtype=np.float64).reshape(-1).copy()
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0
        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            return np.zeros(3, dtype=np.float32)
        return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)

    def _prepare_image(self, image: Any) -> np.ndarray:
        from openpi_client import image_tools

        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"LIBERO image must be HxWx3, got {rgb.shape}")
        # IMPORTANT: rotate 180 degrees to match training preprocessing, exactly
        # as openpi examples/libero/main.py does on the raw obs image.
        rgb = np.ascontiguousarray(rgb[::-1, ::-1])
        resized = image_tools.resize_with_pad(rgb, self.resize_size, self.resize_size)
        return np.ascontiguousarray(image_tools.convert_to_uint8(resized))

    def _get_client(self) -> Any:
        if self._client is None:
            from robot.vla.openpi_bridge import Pi05Client

            url = str(self.config.get("server_url", "ws://localhost:8000"))
            timeout = float(self.config.get("timeout", 120.0))
            self._client = Pi05Client(url, timeout=timeout)
        return self._client

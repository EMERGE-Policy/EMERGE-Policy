"""Non-executing LIBERO skill previews and selected-prefix dispatch for AC-WM."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from robot.mujoco_simulation.pose_utils import PoseUtils


class AcWmControllerMixin:
    """Preview controls from one simulator state without changing that state."""

    def _vla_propose(self, params: dict[str, Any]) -> str:
        """Generate, but do not execute, a VLA candidate for AC-WM."""
        if self._vla is None:
            return "Failed: LIBERO VLA executor is not initialized"
        instruction = str(params.get("instruction", "")).strip()
        if not instruction:
            return "Failed: vla_propose requires instruction"
        horizon = max(1, min(int(params.get("horizon", 32)), 64))
        # Capture the conditioning frame before policy inference. The simulator
        # is paused during inference, so these controls and pixels now refer to
        # exactly the same environment state.
        observation_path, revision = self._resolve_ac_wm_observation(params)
        actions = self._vla.propose(instruction, horizon)
        payload = {
            "actions": actions.astype(float).tolist(),
            "execute_steps": min(self._vla.replan_steps, len(actions)),
            "prediction_steps": len(actions),
            "observation_path": str(observation_path),
            "observation_revision": revision,
            "position_scale": self.position_scale,
            "rotation_scale": self.rotation_scale,
            "position_gain": self.position_gain,
            "rotation_gain": self.rotation_gain,
        }
        import json

        return "VLA_PROPOSAL:" + json.dumps(payload, separators=(",", ":"))

    def _rule_propose(self, params: dict[str, Any]) -> str:
        """Build a non-executing action7 preview for a rule-based skill."""
        action_type = str(params.get("skill_action_type", "")).strip()
        skill_params = dict(params.get("parameters") or {})
        if action_type not in {"move_to_pose", "move_linear", "set_gripper", "follow_arc"}:
            raise ValueError(f"unsupported rule skill for AC-WM: {action_type!r}")
        observation_path, revision = self._resolve_ac_wm_observation(params)
        actions = self._preview_rule_actions(action_type, skill_params)
        payload = {
            "actions": actions.astype(float).tolist(),
            "execute_steps": min(
                int(skill_params.get("execute_steps", self.config.get("ac_wm_replan_steps", 8))),
                len(actions),
            ),
            "prediction_steps": len(actions),
            "observation_path": str(observation_path),
            "observation_revision": revision,
            "position_scale": self.position_scale,
            "rotation_scale": self.rotation_scale,
            "position_gain": self.position_gain,
            "rotation_gain": self.rotation_gain,
        }
        import json

        return "RULE_PROPOSAL:" + json.dumps(payload, separators=(",", ":"))

    def _preview_rule_actions(self, action_type: str, params: dict[str, Any]) -> np.ndarray:
        """Translate the current geometric skill intent into a bounded OSC action chunk."""
        max_horizon = 64
        gripper = self._active_gripper_signal()
        start_position, start_orientation = self._environment.get_eef_pose()

        if action_type == "set_gripper":
            if "opening_m" in params:
                opening = float(params["opening_m"])
            elif "width_m" in params:
                opening = float(params["width_m"])
            elif str(params.get("command", "")).strip().lower() == "open":
                opening = self.gripper_max_opening
            elif str(params.get("command", "")).strip().lower() == "close":
                opening = 0.0
            else:
                raise ValueError("set_gripper requires opening_m, width_m, or command")
            if not 0.0 <= opening <= self.gripper_max_opening:
                raise ValueError(
                    f"gripper opening must be between 0 and {self.gripper_max_opening:.4f} m"
                )
            signal = self._opening_to_signal(opening)
            count = max(
                1, min(int(params.get("steps", self.config.get("gripper_steps", 30))), max_horizon)
            )
            return np.tile(np.asarray([0.0] * 6 + [signal], dtype=np.float32), (count, 1))

        if action_type == "move_to_pose":
            if "position_m" not in params:
                raise ValueError("move_to_pose requires position_m")
            target_position = PoseUtils.vector(params["position_m"], name="position_m")
            target_orientation = PoseUtils.resolve_orientation(params)
            if target_orientation is None:
                raise ValueError("move_to_pose requires orientation_quat or orientation_euler")
            count = max(1, min(int(params.get("steps", self.max_steps)), max_horizon))
            position_delta = (target_position - start_position) / count
            rotation_delta = (
                PoseUtils.quaternion_error(start_orientation, target_orientation) / count
            )
            row = self._pose_action(position_delta, rotation_delta, gripper)
            return np.tile(row, (count, 1))

        if action_type == "move_linear":
            if "position_m" in params:
                target_position = PoseUtils.vector(params["position_m"], name="position_m")
            elif "delta_m" in params:
                target_position = start_position + PoseUtils.vector(
                    params["delta_m"], name="delta_m"
                )
            else:
                raise ValueError("move_linear requires position_m or delta_m")
            target_orientation = PoseUtils.resolve_orientation(params)
            if target_orientation is None:
                target_orientation = start_orientation
            requested_steps = max(
                1, int(params.get("steps", self.config.get("move_linear_steps", 80)))
            )
            count = max(1, min(requested_steps, max_horizon))
            position_delta = (target_position - start_position) / count
            rotation_delta = (
                PoseUtils.quaternion_error(start_orientation, target_orientation) / count
            )
            row = self._pose_action(position_delta, rotation_delta, gripper)
            rows = [row.copy() for _ in range(count)]
            settle = max(0, min(int(params.get("settle_steps", 40)), max_horizon - count))
            rows.extend(np.asarray([0.0] * 6 + [gripper], dtype=np.float32) for _ in range(settle))
            return np.asarray(rows, dtype=np.float32)

        center = PoseUtils.vector(params.get("center"), name="center")
        axis = PoseUtils.vector(params.get("axis"), name="axis")
        radius = float(params.get("radius_m", 0.0))
        if radius <= 0.0:
            raise ValueError("follow_arc.radius_m must be positive")
        angle = self._resolve_angle(params)
        mode = str(params.get("orientation_mode", "fixed")).strip().lower()
        if mode not in {"fixed", "co_rotate"}:
            raise ValueError("orientation_mode must be fixed or co_rotate")
        requested_steps = max(1, int(params.get("steps", self.config.get("follow_arc_steps", 100))))
        count = max(1, min(requested_steps, max_horizon))
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-10:
            raise ValueError("follow_arc.axis must be non-zero")
        axis_unit = axis / axis_norm
        start_vector = start_position - center
        start_norm = float(np.linalg.norm(start_vector))
        if start_norm <= 1e-10:
            raise ValueError("end-effector cannot start at the arc center")
        start_vector = start_vector / start_norm * radius
        requested_orientation = PoseUtils.resolve_orientation(params)
        if requested_orientation is not None:
            start_orientation = requested_orientation
        previous_position = start_position.copy()
        previous_orientation = start_orientation.copy()
        rows = []
        for index in range(count):
            step_angle = angle * float(index + 1) / count
            target_position = center + PoseUtils.rotate_vector(start_vector, axis_unit, step_angle)
            if mode == "co_rotate":
                delta_q = PoseUtils.axis_angle_to_quaternion(axis_unit * step_angle)
                target_orientation = PoseUtils.quaternion_multiply(delta_q, start_orientation)
            else:
                target_orientation = start_orientation
            rows.append(
                self._pose_action(
                    target_position - previous_position,
                    PoseUtils.quaternion_error(previous_orientation, target_orientation),
                    gripper,
                )
            )
            previous_position = target_position
            previous_orientation = target_orientation
        return np.asarray(rows, dtype=np.float32)

    def _capture_ac_wm_observation(self, _params: dict[str, Any]) -> str:
        import json

        path, revision = self._write_ac_wm_observation()
        return "AC_WM_OBSERVATION:" + json.dumps(
            {"observation_path": str(path), "observation_revision": revision},
            separators=(",", ":"),
        )

    def _resolve_ac_wm_observation(self, params: dict[str, Any]):
        path = str(params.get("observation_path", "")).strip()
        if not path:
            return self._write_ac_wm_observation()
        revision = int(params.get("observation_revision", 0))
        return Path(path), revision

    def _write_ac_wm_observation(self):
        """Save the exact current VLA camera observation as a single RGB image."""
        import json
        import time

        import cv2
        import numpy as np

        self._environment.refresh_cameras()
        camera_key = getattr(self._vla, "base_image_key", "agentview_image")
        wrist_key = getattr(self._vla, "wrist_image_key", "robot0_eye_in_hand_image")
        observation = getattr(self._environment, "latest_obs", {})
        if camera_key not in observation:
            raise RuntimeError(f"AC-WM primary observation is missing VLA camera key: {camera_key}")
        if wrist_key not in observation:
            raise RuntimeError(f"AC-WM wrist observation is missing VLA camera key: {wrist_key}")
        primary = np.asarray(observation[camera_key], dtype=np.uint8)
        wrist = np.asarray(observation[wrist_key], dtype=np.uint8)
        for key, image in ((camera_key, primary), (wrist_key, wrist)):
            if image.ndim != 3 or image.shape[-1] != 3:
                raise RuntimeError(f"AC-WM camera {key} must be an RGB image, got {image.shape}")
        if primary.shape[:2] != wrist.shape[:2]:
            wrist = cv2.resize(
                wrist, (primary.shape[1], primary.shape[0]), interpolation=cv2.INTER_AREA
            )
        # Raw LIBERO sim frames are rotated 180 degrees relative to the training
        # data; rotate each view upright as the VLA preprocessing and the official
        # Cosmos LIBERO client do, then build concat_view (third-person left, wrist right).
        primary, wrist = primary[::-1, ::-1], wrist[::-1, ::-1]
        frame = np.ascontiguousarray(np.concatenate([primary, wrist], axis=1))
        height, width = frame.shape[:2]
        output = (
            self._environment.workspace
            / "artifacts"
            / "ac-wm"
            / f"observation-{time.time_ns()}.png"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(output), bgr):
            raise RuntimeError(f"could not write AC-WM observation image: {output}")
        manifest_path = (
            self._environment.workspace / "artifacts" / "observations" / "observation.json"
        )
        revision = (
            int(json.loads(manifest_path.read_text()).get("revision", 0))
            if manifest_path.is_file()
            else 0
        )
        manifest = {
            "revision": revision,
            "camera_keys": [camera_key, wrist_key],
            "camera_names": [camera_key.removesuffix("_image"), wrist_key.removesuffix("_image")],
            "view_point": "concat_view",
            "layout": "third_person_left_wrist_right",
            "image_transform": "rotate_180_per_view",
            "frame_size": [width, height],
            "rgb": True,
        }
        output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
        return output, revision

    def _execute_action_chunk(self, params: dict[str, Any]) -> str:
        """Execute selected VLA rows verbatim through the active LIBERO environment."""
        import numpy as np

        raw = np.asarray(params.get("actions"), dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 7 or not 1 <= len(raw) <= 64:
            raise ValueError("execute_action_chunk requires 1..64 action rows of width 7")
        if not np.all(np.isfinite(raw)):
            raise ValueError("execute_action_chunk actions must be finite")
        executed = 0
        reason = "chunk_completed"
        try:
            for action in raw:
                self._check_interrupted()
                _, _, done, _ = self._environment.step(action, publish_observation=True)
                executed += 1
                self._gripper_command = float(np.clip(action[6], -1.0, 1.0))
                if self._environment.check_success():
                    reason = "goal_reached"
                    break
                if done:
                    reason = "environment_done"
                    break
        finally:
            if executed:
                self._environment.refresh_cameras()
        return (
            f"Selected candidate {params.get('candidate_id', 'unknown')} from skill "
            f"{params.get('skill_name', 'unknown')} executed its exact proposed prefix: "
            f"steps={executed}, reason={reason}."
        )

"""Emerge BaseDriver backed by LIBERO MuJoCo / robosuite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robot.drivers.base_driver import BaseDriver, CancelCheck
from robot.mujoco_simulation.mujoco_actions import MujocoActionController
from robot.mujoco_simulation.mujoco_env import MujocoEnvManager
from robot.mujoco_simulation.scene_io import SceneConfigParser
from robot.vla.mujoco_policy_executor import VLAExecutor
from robot.wam.mujoco_policy_executor import CosmosWAMExecutor


class LiberoMujocoDriver(BaseDriver):
    """Expose rule, pi0.5 VLA, and Cosmos WAM actions in LIBERO."""

    def __init__(
        self,
        gui: bool = False,
        workspace: str | Path | None = None,
        libero: dict[str, Any] | None = None,
        cameras: dict[str, dict[str, Any]] | None = None,
        motion: dict[str, Any] | None = None,
        vla: dict[str, Any] | None = None,
        wam: dict[str, Any] | None = None,
        evaluation: dict[str, Any] | None = None,
        profile_path: str | Path | None = None,
        **_kwargs: Any,
    ) -> None:
        if gui:
            raise ValueError(
                "libero_mujoco is an EGL offscreen driver; gui must be false"
        )
        self._evaluation_config = dict(evaluation or {})
        self._policy_backend = str(
            self._evaluation_config.get("policy_backend", "both")
        ).strip().lower()
        if self._policy_backend not in {"vla", "wam", "both"}:
            raise ValueError(
                "evaluation.policy_backend must be one of: vla, wam, both"
            )
        enabled_policy_backends = (
            {"vla", "wam"}
            if self._policy_backend == "both"
            else {self._policy_backend}
        )
        self._workspace = Path(workspace or Path.cwd()).expanduser().resolve()
        if profile_path is None:
            self._profile_path = (
                Path(__file__).resolve().parents[1] / "profiles/libero_mujoco.md"
            )
        else:
            configured_profile = Path(profile_path).expanduser()
            if not configured_profile.is_absolute():
                configured_profile = (
                    Path(__file__).resolve().parents[2] / configured_profile
                )
            self._profile_path = configured_profile.resolve()
        if self._evaluation_config:
            from robot.mujoco_simulation.mujuco_env_eval import MujocoEvalEnvManager

            self._environment = MujocoEvalEnvManager(
                libero_config=libero,
                camera_config=cameras,
                evaluation_config=self._evaluation_config,
                workspace=self._workspace,
            )
        else:
            self._environment = MujocoEnvManager(
                libero_config=libero,
                camera_config=cameras,
                workspace=self._workspace,
            )
        self._scene_parser = SceneConfigParser()
        self._vla_config = dict(vla or {})
        self._vla = (
            VLAExecutor(self._environment, config=self._vla_config)
            if "vla" in enabled_policy_backends
            else None
        )
        self._wam_config = dict(wam or {})
        self._wam = (
            CosmosWAMExecutor(self._environment, config=self._wam_config)
            if "wam" in enabled_policy_backends
            else None
        )
        self._actions = MujocoActionController(
            self._environment,
            motion,
            vla_executor=self._vla,
            wam_executor=self._wam,
            enabled_policy_backends=enabled_policy_backends,
        )

    def get_profile_path(self) -> Path:
        return self._profile_path

    def load_scene(self, scene: dict[str, dict]) -> None:

        normalized = self._scene_parser.parse(scene)
        self._environment.create(normalized)

    def execute_action(
        self,
        action_type: str,
        params: dict,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        if not self._environment.is_connected():
            return "Failed: LIBERO scene has not been loaded"
        return self._actions.execute(action_type, params, cancel_check=cancel_check)

    def get_runtime_state(self) -> dict[str, Any]:
        robot_state = self._environment.get_robot_state()
        return {
            "robots": {"libero_mujoco": robot_state},
        }

    def close(self) -> None:
        if self._vla is not None:
            self._vla.close()
        if self._wam is not None:
            self._wam.close()
        self._environment.close()

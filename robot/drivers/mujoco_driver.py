"""Emerge BaseDriver backed by LIBERO MuJoCo / robosuite."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from robot.drivers.base_driver import BaseDriver, CancelCheck, SceneCatalog
from robot.mujoco_simulation.mujoco_actions import MujocoActionController
from robot.mujoco_simulation.mujoco_env import MujocoEnvManager
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
        self._enabled_policy_backends = (
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
        self._libero_config = dict(libero or {})
        if not self._libero_config.get("bddl_root") and not self._libero_config.get("bddl_file_name"):
            raise ValueError("Configure libero.bddl_root and libero.bddl_file_name.")
        self._camera_config = cameras
        self._motion_config = motion
        self._vla_config = dict(vla or {})
        self._wam_config = dict(wam or {})
        initial_path = Path(self._libero_config.get("bddl_file_name") or "").expanduser()
        self._bddl_root = (
            Path(self._libero_config["bddl_root"]).expanduser().resolve()
            if self._libero_config.get("bddl_root") else initial_path.resolve().parent
        )
        self._environment = None
        self._vla = None
        self._wam = None
        self._actions = None

    def get_profile_path(self) -> Path:
        return self._profile_path

    def load_environment(self) -> None:
        if self._evaluation_config:
            from robot.mujoco_simulation.mujuco_env_eval import MujocoEvalEnvManager

            self._environment = MujocoEvalEnvManager(
                libero_config=self._libero_config,
                camera_config=self._camera_config,
                evaluation_config=self._evaluation_config,
                workspace=self._workspace,
            )
        else:
            self._environment = MujocoEnvManager(
                libero_config=self._libero_config,
                camera_config=self._camera_config,
                workspace=self._workspace,
            )
        self._vla = (
            VLAExecutor(self._environment, config=self._vla_config)
            if "vla" in self._enabled_policy_backends
            else None
        )
        self._wam = (
            CosmosWAMExecutor(self._environment, config=self._wam_config)
            if "wam" in self._enabled_policy_backends
            else None
        )
        self._actions = MujocoActionController(
            self._environment,
            self._motion_config,
            vla_executor=self._vla,
            wam_executor=self._wam,
            enabled_policy_backends=self._enabled_policy_backends,
        )
        self._environment.create()

    def reset_environment(self) -> None:
        self.close()
        self.load_environment()

    def get_scene_catalog(self) -> SceneCatalog:
        entries = [
            {"id": path.relative_to(self._bddl_root).as_posix(),
             "label": path.relative_to(self._bddl_root).as_posix()}
            for path in sorted(self._bddl_root.rglob("*.bddl"))
            if path.is_file() and path.resolve().is_relative_to(self._bddl_root)
        ]
        current = None
        if self._environment is not None and self._environment.is_connected():
            current = self._environment._bddl_file.relative_to(self._bddl_root).as_posix()
        return {"root": self._bddl_root.name, "current": current, "entries": entries}

    def switch_scene(self, scene_id: str) -> None:
        target = (self._bddl_root / scene_id).resolve()
        if (not target.is_relative_to(self._bddl_root)
                or target.suffix != ".bddl" or not target.is_file()):
            raise ValueError("Select a BDDL file under the configured BDDL root.")
        self.close()
        self._libero_config.update(
            bddl_root=str(self._bddl_root),
            bddl_file_name=target.relative_to(self._bddl_root).as_posix(),
        )
        self.load_environment()

    def execute_action(
        self,
        action_type: str,
        params: dict,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        if self._environment is None or not self._environment.is_connected():
            return "Failed: LIBERO environment has not been loaded"
        return self._actions.execute(action_type, params, cancel_check=cancel_check)

    def get_runtime_state(self) -> dict[str, Any]:
        robot_state = self._environment.get_robot_state()
        return {
            "robots": {"libero_mujoco": robot_state},
        }

    def close(self) -> None:
        try:
            if self._vla is not None:
                self._vla.close()
        finally:
            try:
                if self._wam is not None:
                    self._wam.close()
            finally:
                if self._environment is not None:
                    self._environment.close()
                self._vla = self._wam = self._environment = self._actions = None

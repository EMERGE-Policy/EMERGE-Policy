"""LIBERO OffScreenRenderEnv lifecycle and direct action stepping."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import re
import sys
import types
from typing import Any

import numpy as np

from Emerge.utils.geometry import quaternion_xyzw_to_rpy
from robot.mujoco_simulation.calibrated_observation import (
    CalibratedObservationWriter,
)
from robot.mujoco_simulation.camera_injector import CameraInjector
from robot.mujoco_simulation.mujoco_camera import MujocoCamera
from robot.mujoco_simulation.pose_utils import PoseUtils
from robot.mujoco_simulation.scene_io import NormalizedSceneConfig
from robot.mujoco_simulation.urdf_loader import UrdfLoader
from robot.mujoco_simulation.vggt_camera_rig import CameraPose, VggtCameraRig


class RobosuiteCompatibility:
    """Install narrowly-scoped aliases required by LIBERO on robosuite 1.5+."""

    _requires_libero_model_patch = False

    @classmethod
    def install(cls) -> None:
        robosuite = importlib.import_module("robosuite")
        legacy_module = "robosuite.environments.manipulation.single_arm_env"
        try:
            importlib.import_module(legacy_module)
            return
        except ModuleNotFoundError as exc:
            if exc.name != legacy_module:
                raise
        cls._requires_libero_model_patch = True
        module = importlib.import_module(
            "robosuite.environments.manipulation.manipulation_env"
        )

        class SingleArmCompatibility(module.ManipulationEnv):
            def __init__(self, *args: Any, mount_types: Any = "default", **kwargs: Any) -> None:
                kwargs.setdefault("base_types", mount_types)
                super().__init__(*args, **kwargs)

        compatibility = types.ModuleType(legacy_module)
        compatibility.SingleArmEnv = SingleArmCompatibility
        sys.modules[legacy_module] = compatibility

        robot_module_name = "robosuite.robots.single_arm"
        if robot_module_name not in sys.modules:
            fixed_base_module = importlib.import_module("robosuite.robots.fixed_base_robot")
            robot_compatibility = types.ModuleType(robot_module_name)
            robot_compatibility.SingleArm = fixed_base_module.FixedBaseRobot
            sys.modules[robot_module_name] = robot_compatibility

        if not hasattr(robosuite, "load_controller_config"):
            robosuite.load_controller_config = cls._load_controller_config

        # robosuite 1.4's MujocoXMLObject had a private ``_get_geoms`` helper that
        # 1.5 dropped. LIBERO-Plus's custom object loader (envs/objects/custom_objects.py)
        # still calls it when building articulated fixtures (e.g. WoodenCabinet), so
        # restore the 1.4 implementation on the base class. Pure XML-tree walk; no
        # behavioral change for callers that never used it (standard LIBERO).
        objects_module = importlib.import_module("robosuite.models.objects.objects")
        xml_object = objects_module.MujocoXMLObject
        if not hasattr(xml_object, "_get_geoms"):
            xml_object._get_geoms = cls._get_geoms

    @staticmethod
    def _get_geoms(self: Any, root: Any, _parent: Any = None) -> list:
        """robosuite 1.4 MujocoXMLObject._get_geoms: collect (parent, geom) pairs.

        Recursively walks the XML tree rooted at ``root`` and returns a list of
        ``(parent_element, geom_element)`` tuples for every ``geom`` descendant.
        ``_parent`` is internal to the recursion and must not be passed by callers.
        """
        geom_pairs = []
        if _parent is not None and root.tag == "geom":
            geom_pairs.append((_parent, root))
        for child in root:
            geom_pairs.extend(self._get_geoms(child, _parent=root))
        return geom_pairs

    @classmethod
    def patch_libero_robot_models(cls) -> None:
        if not cls._requires_libero_model_patch:
            return
        robot_module = importlib.import_module("libero.libero.envs.robots")
        for class_name in ("MountedPanda", "OnTheGroundPanda"):
            robot_class = getattr(robot_module, class_name)
            if not hasattr(robot_class, "arms"):
                robot_class.arms = ["right"]
            if "default_base" not in robot_class.__dict__:
                robot_class.default_base = (
                    property(cls._null_mount)
                    if class_name == "OnTheGroundPanda"
                    else robot_class.default_mount
                )
            robot_class.default_gripper = property(cls._panda_gripper)
            robot_class.default_controller_config = property(cls._panda_controller)

    @staticmethod
    def _null_mount(_instance: Any) -> str:
        return "NullMount"

    @staticmethod
    def _panda_gripper(_instance: Any) -> dict[str, str]:
        return {"right": "PandaGripper"}

    @staticmethod
    def _panda_controller(_instance: Any) -> dict[str, str]:
        return {"right": "default_panda"}

    @staticmethod
    def _load_controller_config(
        custom_fpath: str | None = None,
        default_controller: str | None = None,
    ) -> dict[str, Any]:
        part_module = importlib.import_module(
            "robosuite.controllers.parts.controller_factory"
        )
        composite_module = importlib.import_module(
            "robosuite.controllers.composite.composite_controller_factory"
        )
        part_config = part_module.load_part_controller_config(
            custom_fpath=custom_fpath,
            default_controller=default_controller,
        )
        return composite_module.refactor_composite_controller_config(
            part_config,
            robot_type="Panda",
            arms=["right"],
        )


class MujocoEnvManager:
    """Own one LIBERO environment and its MuJoCo cameras."""

    def __init__(
        self,
        *,
        libero_config: dict[str, Any] | None = None,
        camera_config: dict[str, dict[str, Any]] | None = None,
        workspace: str | Path | None = None,
    ) -> None:
        self.config = dict(libero_config or {})
        self.workspace = Path(workspace or Path.cwd()).expanduser().resolve()

        path_text = str(self.config.get("bddl_file_name", "")).strip()
        self._bddl_file = Path(path_text).expanduser().resolve() if path_text else None
        bddl_source = self._bddl_source_file(self._bddl_file)
        if bddl_source is None or not bddl_source.is_file():
            detail = path_text or "<empty>"
            if bddl_source is not None and bddl_source != self._bddl_file:
                detail += f" (LIBERO-Plus base file: {bddl_source})"
            raise FileNotFoundError(f"LIBERO BDDL file not found: {detail}")

        source = camera_config
        if not isinstance(source, dict) or not source:
            raise ValueError("cameras must be a non-empty mapping")

        self._camera_configs: dict[str, dict[str, Any]] = {}
        for name, raw in source.items():
            camera_name = str(name).strip()
            if not camera_name:
                raise ValueError("camera name cannot be empty")
            if not isinstance(raw, dict):
                raise ValueError(f"camera {camera_name!r} must be an object")
            mode = str(raw.get("mode", "existing")).strip() or "existing"
            if mode not in {"existing", "fixed", "auto"}:
                raise ValueError(
                    f"camera {camera_name!r} mode must be 'existing', 'fixed', or 'auto'"
                )
            camera_settings = {
                **raw,
                "name": camera_name,
                "mode": mode,
                "width": int(raw["width"]),
                "height": int(raw["height"]),
            }

            if mode == "fixed":
                position = raw.get("pos", raw.get("position", raw.get("position_m")))
                quaternion = raw.get("quat", raw.get("orientation_quat"))
                if position is None or quaternion is None:
                    raise ValueError(
                        f"fixed camera {camera_name!r} requires pos/quat (or position/orientation_quat)"
                    )
                camera_settings["pos"] = (
                    PoseUtils.vector(position, name=f"{camera_name}.pos", length=3)
                    .astype(float)
                    .tolist()
                )
                camera_settings["quat"] = (
                    PoseUtils.normalize_quaternion(quaternion).astype(float).tolist()
                )
                if "fovy" in raw:
                    camera_settings["fovy"] = float(raw["fovy"])
            elif mode == "auto":
                camera_settings["pos"] = [0.0, 0.0, 2.0]
                camera_settings["quat"] = [1.0, 0.0, 0.0, 0.0]
                camera_settings["fovy"] = float(raw.get("fovy", 62.0))
            self._camera_configs[camera_name] = camera_settings

        self._environment: Any | None = None
        self._latest_obs: dict[str, Any] = {}
        self._latest_reward = 0.0
        self._latest_done = False
        self._latest_info: dict[str, Any] = {}
        self._cameras: dict[str, MujocoCamera] = {}
        self._observation_writer: CalibratedObservationWriter | None = None
        self._scene: NormalizedSceneConfig | None = None
        self._merged_xml: str | None = None
        self._auto_camera_poses: dict[str, CameraPose] = {}
        self._connected = False

    @staticmethod
    def _bddl_source_file(bddl_file: Path | None) -> Path | None:
        """Return the physical BDDL backing a possibly virtual Plus task name.

        LIBERO-Plus encodes camera pose, robot initial state, and sensor noise in
        a virtual filename such as ``task_view_36_0_100_0_0_initstate_0.bddl``.
        Its ``ControlEnv`` parses that suffix and then opens ``task.bddl``. Keep
        passing the virtual path to ControlEnv, but validate the physical base
        file here so the controller does not reject a supported Plus task first.
        """
        if bddl_file is None:
            return None
        match = re.fullmatch(
            r"(?P<base>.+)_view_[+-]?\d+(?:_[+-]?\d+){4}"
            r"_initstate_[+-]?\d+(?:_noise_\d+)?\.bddl",
            str(bddl_file),
        )
        if match is None:
            return bddl_file
        return Path(f"{match.group('base')}.bddl")

    @property
    def sim(self) -> Any:
        if self._environment is None:
            raise RuntimeError("LIBERO environment has not been created")
        return self._environment.sim

    @property
    def latest_obs(self) -> dict[str, Any]:
        return self._latest_obs

    @property
    def cameras(self) -> dict[str, MujocoCamera]:
        return dict(self._cameras)

    def create(self, scene: NormalizedSceneConfig) -> None:
        self.close()
        self._scene = scene
        os.environ.setdefault("MUJOCO_GL", str(self.config.get("mujoco_gl", "egl")))
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/phyagentos-matplotlib")
        if bool(self.config.get("disable_numba_jit", True)):
            os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
        offscreen_env = self._import_offscreen_environment()
        bddl_file = self._bddl_file
        existing_camera_configs = [
            item for item in self._camera_configs.values() if item["mode"] == "existing"
        ]
        camera_names = [item["name"] for item in existing_camera_configs]
        heights = [int(item["height"]) for item in existing_camera_configs]
        widths = [int(item["width"]) for item in existing_camera_configs]
        depths = [bool(item.get("depth", False)) for item in existing_camera_configs]
        env_kwargs = dict(self.config.get("env_kwargs") or {})
        env_kwargs.update(
            {
                "bddl_file_name": str(bddl_file),
                "robots": self.config.get("robots", ["Panda"]),
                "controller": "OSC_POSE",
                "control_freq": int(self.config.get("control_freq", 20)),
                "camera_names": camera_names,
                "camera_heights": heights,
                "camera_widths": widths,
                "camera_depths": depths,
            }
        )
        self._environment = offscreen_env(**env_kwargs)
        self._latest_obs = dict(self._environment.reset())
        self._install_runtime_model(scene)
        self._create_cameras(list(self._camera_configs))
        self._connected = True

    def reset(self) -> dict[str, Any]:
        if self._environment is None:
            raise RuntimeError("LIBERO environment has not been created")
        self._latest_obs = dict(self._environment.reset())
        if self._merged_xml is not None:
            self._reload_runtime_model_preserving_state(self._merged_xml)
        self._latest_reward = 0.0
        self._latest_done = False
        self._latest_info = {}
        self.refresh_cameras()
        return self._latest_obs

    def step(
        self,
        action7: Any,
        *,
        publish_observation: bool = True,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self._environment is None:
            raise RuntimeError("LIBERO environment has not been created")
        action = np.asarray(action7, dtype=np.float32).reshape(-1)
        if action.size != 7 or not np.all(np.isfinite(action)):
            raise ValueError("LIBERO action must contain exactly 7 finite values")
        action = np.clip(action, -1.0, 1.0)
        obs, reward, done, info = self._environment.step(action)
        self._latest_obs = dict(obs)
        self._latest_reward = float(reward)
        self._latest_done = bool(done)
        self._latest_info = dict(info or {})
        if publish_observation:
            self.refresh_cameras()
        else:
            self.refresh_recording_cameras()
        return self._latest_obs, self._latest_reward, self._latest_done, self._latest_info

    def refresh_cameras(self) -> None:
        for camera in self._cameras.values():
            camera.refresh()
        if self._observation_writer is not None:
            self._observation_writer.write()

    def refresh_recording_cameras(self) -> None:
        """Record rollout frames without rendering perception-only cameras."""
        for camera in self._cameras.values():
            if camera.is_recording():
                camera.refresh()

    def get_eef_pose(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            position = self._latest_obs["robot0_eef_pos"]
            quaternion = self._latest_obs["robot0_eef_quat"]
        except KeyError as exc:
            raise RuntimeError(f"LIBERO observation is missing {exc.args[0]}") from exc
        return (
            PoseUtils.vector(position, name="robot0_eef_pos"),
            PoseUtils.normalize_quaternion(quaternion),
        )

    def get_gripper_qpos(self) -> np.ndarray:
        value = self._latest_obs.get("robot0_gripper_qpos")
        if value is None:
            raise RuntimeError("LIBERO observation is missing robot0_gripper_qpos")
        return PoseUtils.vector(value, name="robot0_gripper_qpos", length=2)

    def current_gripper_signal(self) -> float:
        opening = float(np.sum(np.abs(self.get_gripper_qpos())))
        max_opening = float(self.config.get("gripper_max_opening_m", 0.08))
        if max_opening <= 0.0:
            return -1.0
        return float(np.clip(1.0 - 2.0 * opening / max_opening, -1.0, 1.0))

    def check_success(self) -> bool:
        if self._environment is None:
            return False
        return bool(self._environment.check_success())

    def get_robot_state(self) -> dict[str, Any]:
        position, quaternion = self.get_eef_pose()
        return {
            "type": "franka_panda",
            "eef_pose": {
                "position": PoseUtils.vector_dict(position),
                "orientation_euler": PoseUtils.rpy_dict(
                    quaternion_xyzw_to_rpy(quaternion)
                ),
            },
            "gripper_qpos": self.get_gripper_qpos().astype(float).tolist(),
            "control": {"controller": "OSC_POSE", "frequency_hz": int(self.config.get("control_freq", 20))},
            "done": self._latest_done,
            "success": self.check_success(),
        }

    def is_connected(self) -> bool:
        return self._connected and self._environment is not None

    def close(self) -> None:
        for camera in self._cameras.values():
            camera.close()
        self._cameras.clear()
        self._observation_writer = None
        if self._environment is not None:
            self._environment.close()
        self._environment = None
        self._merged_xml = None
        self._latest_obs = {}
        self._latest_reward = 0.0
        self._latest_done = False
        self._latest_info = {}
        self._auto_camera_poses.clear()
        self._connected = False

    def _import_offscreen_environment(self) -> Any:
        RobosuiteCompatibility.install()
        try:
            module = importlib.import_module("libero.libero.envs")
        except ImportError:
            source_path = self.config.get("libero_source_path")
            if source_path is None:
                source_path = Path(__file__).resolve().parents[2] / "third_party/openpi/third_party/libero"
            source = str(Path(source_path).expanduser().resolve())
            if source not in sys.path:
                sys.path.insert(0, source)
            try:
                module = importlib.import_module("libero.libero.envs")
            except ImportError as exc:
                raise RuntimeError(
                    "LIBERO could not be imported; install it or set libero_source_path"
                ) from exc
        RobosuiteCompatibility.patch_libero_robot_models()
        return module.OffScreenRenderEnv

    def get_camera_runtime_settings(self, name: str) -> dict[str, Any]:
        settings = self._camera_configs.get(str(name).strip())
        if settings is None:
            raise KeyError(f"camera is not registered: {name!r}")
        return dict(settings)

    def _create_cameras(self, names: list[str]) -> None:
        for name in names:
            camera = MujocoCamera(
                self,
                name=name,
                workspace=self.workspace,
            )
            self._cameras[camera.name] = camera
        saved_cameras = [
            self._cameras[name]
            for name in names
            if bool(
                dict(self._camera_configs[name].get("observation") or {}).get(
                    "enabled",
                    False,
                )
            )
        ]
        if saved_cameras:
            self._observation_writer = CalibratedObservationWriter(
                self.workspace,
                saved_cameras,
            )
            self._observation_writer.write()

    def _install_runtime_model(self, scene: NormalizedSceneConfig) -> None:
        if self._environment is None:
            raise RuntimeError("LIBERO environment has not been created")
        raw_env = self._environment.env
        base_xml = raw_env.model.get_xml()
        runtime_xml = base_xml
        changed = False

        if scene.external_assets:
            runtime_xml = UrdfLoader().merge_assets(runtime_xml, scene.external_assets)
            changed = True

        injected_cameras = [
            item
            for item in self._camera_configs.values()
            if item["mode"] in {"fixed", "auto"}
        ]
        if injected_cameras:
            runtime_xml = CameraInjector().merge_cameras(runtime_xml, injected_cameras)
            changed = True

        self._merged_xml = runtime_xml if changed else None
        if not changed:
            return
        self._reload_runtime_model_preserving_state(runtime_xml)

    def _reload_runtime_model_preserving_state(
        self,
        runtime_xml: str,
    ) -> None:
        if self._environment is None:
            raise RuntimeError("LIBERO environment has not been created")
        raw_env = self._environment.env
        fixture_poses = {}
        for fixture in raw_env.fixtures_dict.values():
            body_name = fixture.root_body
            body_id = int(self.sim.model.body_name2id(body_name))
            fixture_poses[body_name] = (
                np.asarray(self.sim.model.body_pos[body_id]).copy(),
                np.asarray(self.sim.model.body_quat[body_id]).copy(),
            )

        # Movable objects are placed through qpos, while LIBERO places fixtures
        # by editing model.body_pos/body_quat. Both must survive the XML reload.
        preserved_state = np.asarray(
            self._environment.sim.get_state().flatten(),
            dtype=np.float64,
        ).copy()
        self._environment.reset_from_xml_string(runtime_xml)
        reloaded_state = np.asarray(
            self._environment.sim.get_state().flatten(),
            dtype=np.float64,
        )
        if reloaded_state.shape != preserved_state.shape:
            raise RuntimeError(
                "runtime MuJoCo XML changed the simulation state shape from "
                f"{preserved_state.shape} to {reloaded_state.shape}; "
                "cannot safely restore the initialized LIBERO object poses"
            )
        self._environment.sim.set_state_from_flattened(preserved_state)
        for body_name, (position, quaternion) in fixture_poses.items():
            body_id = int(self.sim.model.body_name2id(body_name))
            self.sim.model.body_pos[body_id] = position
            self.sim.model.body_quat[body_id] = quaternion
        self._environment.sim.forward()
        self._apply_auto_camera_rig()
        self._latest_obs = dict(
            self._environment.env._get_observations(force_update=True)
        )

    def _apply_auto_camera_rig(self) -> None:
        auto_settings = {
            name: settings
            for name, settings in self._camera_configs.items()
            if settings["mode"] == "auto"
        }
        if not auto_settings:
            return
        if not self._auto_camera_poses:
            self._auto_camera_poses = VggtCameraRig().plan(
                self._environment.env,
                auto_settings,
            )

        for name, pose in self._auto_camera_poses.items():
            camera_id = int(self.sim.model.camera_name2id(name))
            self.sim.model.cam_pos[camera_id] = pose.position_m
            self.sim.model.cam_quat[camera_id] = pose.quaternion_wxyz
            self._camera_configs[name]["pos"] = list(pose.position_m)
            self._camera_configs[name]["quat"] = list(pose.quaternion_wxyz)
        self.sim.forward()

"""Evaluation-only LIBERO lifecycle built on the regular MuJoCo environment."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from robot.mujoco_simulation.mujoco_env import MujocoEnvManager


class MujocoEvalEnvManager(MujocoEnvManager):
    """Add benchmark reset, limits, and private status to ``MujocoEnvManager``."""

    def __init__(
        self,
        *,
        libero_config: dict[str, Any] | None = None,
        camera_config: dict[str, dict[str, Any]] | None = None,
        evaluation_config: dict[str, Any] | None = None,
        workspace: str | Path | None = None,
    ) -> None:
        super().__init__(
            libero_config=libero_config,
            camera_config=camera_config,
            workspace=workspace,
        )
        # The evaluator is the sole authority on the step budget (see
        # ``_max_action_steps`` / ``_step_limit_result``). robosuite otherwise
        # self-terminates at its internal ``horizon`` (default 1000) and then
        # raises "executing action in terminated episode" on every later step.
        # When ``max_action_steps`` exceeds that horizon the episode dies mid-run
        # and every remaining action fails. Disable robosuite's own termination
        # and keep its horizon above any budget so the harness owns termination.
        env_kwargs = dict(self.config.get("env_kwargs") or {})
        env_kwargs.setdefault("ignore_done", True)
        env_kwargs.setdefault("horizon", 1_000_000)
        self.config["env_kwargs"] = env_kwargs
        self.evaluation_config = dict(evaluation_config or {})
        initial_state = self.evaluation_config.get("initial_state")
        self._initial_state = (
            np.asarray(initial_state, dtype=np.float64).reshape(-1)
            if initial_state is not None
            else None
        )
        if self._initial_state is not None and (
            self._initial_state.size == 0 or not np.all(np.isfinite(self._initial_state))
        ):
            raise ValueError("evaluation.initial_state must contain finite values")
        self._num_steps_wait = max(
            0, int(self.evaluation_config.get("num_steps_wait", 0))
        )
        seed = self.evaluation_config.get("seed")
        self._evaluation_seed = int(seed) if seed is not None else None
        max_action_steps = self.evaluation_config.get("max_action_steps")
        self._max_action_steps = (
            max(1, int(max_action_steps)) if max_action_steps is not None else None
        )
        status_path = str(self.evaluation_config.get("status_path", "")).strip()
        self._evaluation_status_path = (
            Path(status_path).expanduser().resolve() if status_path else None
        )
        metadata = self.evaluation_config.get("metadata")
        self._evaluation_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        self._reset_evaluation_counters()

    def create(self, scene) -> None:
        super().create(scene)
        self._reset_from_official_state()

    def reset(self) -> dict[str, Any]:
        if self._environment is None:
            raise RuntimeError("LIBERO environment has not been created")
        return self._reset_from_official_state()

    def step(
        self,
        action7: Any,
        *,
        publish_observation: bool = True,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if (
            self._max_action_steps is not None
            and self._action_steps >= self._max_action_steps
        ):
            return self._step_limit_result()

        obs, reward, done, info = super().step(
            action7,
            publish_observation=publish_observation,
        )
        self._environment_steps += 1
        self._action_steps += 1
        if self.check_success():
            self._episode_success = True
            self._termination_reason = "success"
        elif done:
            self._termination_reason = "environment_done"
        elif (
            self._max_action_steps is not None
            and self._action_steps >= self._max_action_steps
        ):
            self._latest_done = True
            self._termination_reason = "step_limit"
            self._latest_info["evaluation_step_limit"] = self._max_action_steps
            done = True
            info = dict(self._latest_info)
        self._write_evaluation_status(ready=True)
        return obs, reward, done, info

    def close(self) -> None:
        super().close()
        self._reset_evaluation_counters()

    def _reset_from_official_state(self) -> dict[str, Any]:
        if self._environment is None:
            raise RuntimeError("LIBERO environment has not been created")
        if self._evaluation_seed is not None:
            self._seed_environment(self._evaluation_seed)
        if self._merged_xml is not None:
            self._reload_runtime_model_preserving_state(self._merged_xml)
            self._latest_obs = dict(
                self._environment.env._get_observations(force_update=True)
            )
        else:
            self._latest_obs = dict(self._environment.reset())
        self._reset_evaluation_counters()
        if self._initial_state is not None:
            self._latest_obs = dict(
                self._environment.set_init_state(self._initial_state.copy())
            )
        self._apply_auto_camera_rig()
        self.refresh_cameras()
        self._wait_for_stabilization()
        self._write_evaluation_status(ready=True)
        return self._latest_obs

    def _seed_environment(self, seed: int) -> None:
        if self._environment is None:
            return
        raw_environment = getattr(self._environment, "env", self._environment)
        seed_method = getattr(raw_environment, "seed", None)
        if callable(seed_method):
            seed_method(seed)
            return
        # robosuite 1.5 stores ``seed`` as data on the instance, shadowing
        # LIBERO's legacy seed method. LIBERO 0.1 only seeded NumPy globally;
        # also refresh robosuite's Generator when the compatibility path is used.
        np.random.seed(seed)
        if hasattr(raw_environment, "rng"):
            raw_environment.rng = np.random.default_rng(seed)
        raw_environment.seed = seed

    def _reset_evaluation_counters(self) -> None:
        self._latest_reward = 0.0
        self._latest_done = False
        self._latest_info = {}
        self._environment_steps = 0
        self._settling_steps = 0
        self._action_steps = 0
        self._episode_success = False
        self._termination_reason: str | None = None

    def _wait_for_stabilization(self) -> None:
        if self._environment is None or self._num_steps_wait <= 0:
            return
        dummy_action = np.asarray([0.0] * 6 + [-1.0], dtype=np.float32)
        for _ in range(self._num_steps_wait):
            obs, reward, done, info = self._environment.step(dummy_action)
            self._latest_obs = dict(obs)
            self._latest_reward = float(reward)
            self._latest_done = bool(done)
            self._latest_info = dict(info or {})
            self._environment_steps += 1
            self._settling_steps += 1
            self.refresh_cameras()
            if self.check_success():
                self._episode_success = True
                self._termination_reason = "success_during_stabilization"
                break
            if self._latest_done:
                self._termination_reason = "environment_done_during_stabilization"
                break

    def _step_limit_result(
        self,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        self._latest_done = True
        self._termination_reason = "step_limit"
        self._latest_info = {
            **self._latest_info,
            "evaluation_step_limit": self._max_action_steps,
        }
        self._write_evaluation_status(ready=True)
        return (
            self._latest_obs,
            self._latest_reward,
            self._latest_done,
            self._latest_info,
        )

    def _write_evaluation_status(self, *, ready: bool) -> None:
        path = self._evaluation_status_path
        if path is None:
            return
        if self.check_success():
            self._episode_success = True
        payload = {
            "schema_version": "Emerge.libero_evaluation_status.v1",
            "ready": bool(ready),
            "success": self._episode_success,
            "done": bool(self._latest_done),
            "reward": float(self._latest_reward),
            "environment_steps": self._environment_steps,
            "settling_steps": self._settling_steps,
            "action_steps": self._action_steps,
            "max_action_steps": self._max_action_steps,
            "termination_reason": self._termination_reason,
            "metadata": self._evaluation_metadata,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

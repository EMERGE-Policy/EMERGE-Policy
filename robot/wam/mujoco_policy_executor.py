"""Cosmos Policy WAM executor for the LIBERO MuJoCo driver."""

from __future__ import annotations

from typing import Any, Callable

from loguru import logger
import numpy as np

from robot.policy_execution import PolicyExecutionResult

from .planner import CosmosWAMPlanner

CancelCheck = Callable[[], str | None]
WAMResult = PolicyExecutionResult


class CosmosWAMExecutor:
    """Observe, query the isolated WAM server, and execute action7 chunks."""

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
        self.chunk_size = max(1, int(self.config.get("chunk_size", 16)))
        self.replan_steps = max(
            1,
            int(
                self.config.get(
                    "replan_steps",
                    self.config.get("num_open_loop_steps", self.chunk_size),
                )
            ),
        )
        if self.replan_steps > self.chunk_size:
            raise ValueError("wam.replan_steps cannot exceed wam.chunk_size")
        self.stop_on_success = bool(self.config.get("stop_on_success", True))
        self.seed = int(self.config.get("seed", 1))
        self.conditioning_mode = str(self.config.get("conditioning_mode", "task"))
        if self.conditioning_mode not in {"task", "phase", "task_with_phase"}:
            raise ValueError(
                "wam.conditioning_mode must be task, phase, or task_with_phase"
            )
        self.task_instruction = str(self.config.get("task_instruction", "")).strip()
        self.lock_conditioning_mode = bool(
            self.config.get("lock_conditioning_mode", True)
        )
        self.timeout = max(0.1, float(self.config.get("timeout", 180.0)))
        self.base_image_key = str(
            self.config.get("base_image_key", "agentview_image")
        )
        self.wrist_image_key = str(
            self.config.get("wrist_image_key", "robot0_eye_in_hand_image")
        )
        self._planner = CosmosWAMPlanner(dict(self.config.get("search") or {}))

    def execute(
        self,
        instruction: str | None,
        step: int,
        *,
        task_instruction: str | None = None,
        phase_instruction: str | None = None,
        conditioning_mode: str | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> WAMResult:
        configured_task = self.task_instruction
        task_instruction = str(
            configured_task or task_instruction or instruction or ""
        ).strip()
        phase_instruction = str(phase_instruction or "").strip() or None
        mode = self.conditioning_mode
        if conditioning_mode is not None and not self.lock_conditioning_mode:
            mode = str(conditioning_mode)
        if mode == "task":
            phase_instruction = None
        if not task_instruction:
            return WAMResult(
                False,
                0,
                "invalid_instruction",
                "task_instruction must be non-empty",
            )
        if mode in {"phase", "task_with_phase"} and not phase_instruction:
            return WAMResult(
                False,
                0,
                "invalid_instruction",
                f"phase_instruction is required for conditioning_mode={mode!r}",
            )
        try:
            step = int(step)
        except (TypeError, ValueError):
            return WAMResult(False, 0, "invalid_step", "step must be a positive integer")
        if step <= 0:
            return WAMResult(False, 0, "invalid_step", "step must be a positive integer")

        cancel_reason = cancel_check() if cancel_check else None
        if cancel_reason:
            return WAMResult(False, 0, "interrupted", cancel_reason)

        client = self._get_client()
        health_check = getattr(client, "health_check", None)
        if callable(health_check) and not health_check():
            return WAMResult(
                False,
                0,
                "server_unavailable",
                "Cosmos WAM server health check failed",
            )

        action_count = 0
        last_gripper: float | None = None
        task_success = False
        decisions: list[dict[str, Any]] = []
        try:
            while action_count < step:
                cancel_reason = cancel_check() if cancel_check else None
                if cancel_reason:
                    return WAMResult(
                        False,
                        action_count,
                        "interrupted",
                        cancel_reason,
                        last_gripper,
                        task_success,
                        tuple(decisions),
                    )
                plan = self._planner.plan(
                    client,
                    self._build_observation(),
                    task_instruction,
                    phase_instruction=phase_instruction,
                    conditioning_mode=mode,
                    seed=self.seed + action_count,
                )
                if plan.search_decision is not None:
                    decisions.append(plan.search_decision)
                chunk = np.asarray(plan.actions, dtype=np.float32)
                if chunk.ndim != 2 or chunk.shape[1] != 7 or len(chunk) == 0:
                    raise ValueError(
                        f"Cosmos WAM returned action shape {chunk.shape}, expected (N, 7)"
                    )
                if len(chunk) != self.chunk_size:
                    raise ValueError(
                        "Cosmos WAM returned action chunk length "
                        f"{len(chunk)}, expected {self.chunk_size}"
                    )
                execute_count = min(
                    len(chunk), self.replan_steps, step - action_count
                )
                for action in chunk[:execute_count]:
                    cancel_reason = cancel_check() if cancel_check else None
                    if cancel_reason:
                        return WAMResult(
                            False,
                            action_count,
                            "interrupted",
                            cancel_reason,
                            last_gripper,
                            task_success,
                            tuple(decisions),
                        )
                    _, _, done, _ = self._environment.step(
                        action, publish_observation=True
                    )
                    action_count += 1
                    last_gripper = float(np.clip(action[6], -1.0, 1.0))
                    task_success = bool(self._environment.check_success())
                    if self.stop_on_success and task_success:
                        return WAMResult(
                            True,
                            action_count,
                            "goal_reached",
                            None,
                            last_gripper,
                            True,
                            tuple(decisions),
                        )
                    if done:
                        return WAMResult(
                            False,
                            action_count,
                            "environment_done",
                            None,
                            last_gripper,
                            task_success,
                            tuple(decisions),
                        )
                    if action_count >= step:
                        return WAMResult(
                            True,
                            action_count,
                            "step_completed",
                            None,
                            last_gripper,
                            task_success,
                            tuple(decisions),
                        )
        except Exception as exc:
            return WAMResult(
                False,
                action_count,
                "error",
                f"{type(exc).__name__}: {exc}",
                last_gripper,
                task_success,
                tuple(decisions),
            )
        finally:
            if action_count:
                self._environment.refresh_cameras()
            logger.info(
                "WAM execution finished | actions={} replans={} success={}",
                action_count,
                len(decisions),
                task_success,
            )
        return WAMResult(
            True,
            action_count,
            "step_completed",
            None,
            last_gripper,
            bool(self._environment.check_success()),
            tuple(decisions),
        )

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
        self._client = None

    def _build_observation(self) -> dict[str, Any]:
        obs = self._environment.latest_obs
        required = (
            self.base_image_key,
            self.wrist_image_key,
            "robot0_gripper_qpos",
            "robot0_eef_pos",
            "robot0_eef_quat",
        )
        missing = [key for key in required if key not in obs]
        if missing:
            raise KeyError(f"LIBERO observation is missing: {', '.join(missing)}")
        proprio = np.concatenate(
            (
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
                np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1),
            )
        ).astype(np.float32)
        if proprio.shape != (9,) or not np.isfinite(proprio).all():
            raise ValueError(
                f"Cosmos proprio must be finite with shape (9,), got {proprio.shape}"
            )
        return {
            "primary_image": np.asarray(obs[self.base_image_key]),
            "wrist_image": np.asarray(obs[self.wrist_image_key]),
            "proprio": proprio,
        }

    def _get_client(self) -> Any:
        if self._client is None:
            from .client import CosmosWAMClient

            endpoint = str(self.config.get("server_url", "ws://127.0.0.1:8003"))
            self._client = CosmosWAMClient(endpoint, timeout=self.timeout)
        return self._client

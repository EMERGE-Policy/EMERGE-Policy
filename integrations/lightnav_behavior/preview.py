"""Read-only, Agent-invoked LightNav preview over live BEHAVIOR head frames.

This module has no reference to the controlled gateway or the robot action
queue. A preview call maintains one LightNav session across fresh observations;
the independent champion policy remains the only source of simulator actions.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

from Emerge.base import Tool

from integrations.lightnav_behavior.probe import load_head_frame
from integrations.lightnav_sonic_emerge.bridge.lightnav_ws import LightNavWsClient


MANIFEST = Path("artifacts/observations/observation.json")
STATUS = Path("SHADOW_STATUS.json")
MAX_FRAMES_PER_CALL = 12


class LightNavPreviewTool(Tool):
    """Let the Agent choose a navigation goal and preview bounded predictions."""

    def __init__(
        self,
        workspace: Path,
        *,
        server_url: str = "ws://127.0.0.1:8051",
        client_factory: Callable[..., Any] | None = None,
        min_observation_steps: int = 5,
    ) -> None:
        if min_observation_steps < 1:
            raise ValueError("min_observation_steps must be positive")
        self.workspace = Path(workspace).expanduser().resolve()
        self.server_url = server_url
        self.client_factory = client_factory or (
            lambda url, *, client_id: LightNavWsClient(
                url, client_id=client_id, request_timeout_s=10.0,
            )
        )
        self.min_observation_steps = min_observation_steps
        self._client: Any | None = None
        self._episode_id: int | None = None
        self._last_observation_seq: int | None = None
        self._lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "preview_navigation"

    @property
    def description(self) -> str:
        return (
            "Read-only LightNav prediction for a local navigation goal. "
            "Samples a bounded series of fresh BEHAVIOR head-camera frames while "
            "the independent champion policy moves the robot. Returns robot-local "
            "waypoints and stop suggestions; never moves or stops the robot."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "instruction": {
                    "type": "string",
                    "description": "One current navigation subgoal, not the whole manipulation task.",
                },
                "max_frames": {
                    "type": "integer", "minimum": 1, "maximum": MAX_FRAMES_PER_CALL,
                    "description": "Maximum new head frames to preview in this call.",
                },
                "timeout_s": {
                    "type": "number", "minimum": 1, "maximum": 120,
                    "description": "Wall-clock limit for waiting for new observations.",
                },
            },
            "required": ["instruction"],
        }

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self._episode_id = None
        self._last_observation_seq = None

    def _read_json(self, path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _client_for_episode(self, episode_id: int) -> Any:
        if self._episode_id != episode_id:
            self.close()
            self._episode_id = episode_id
        if self._client is None:
            self._client = self.client_factory(
                self.server_url, client_id=f"emerge-behavior-shadow-episode-{episode_id}"
            )
        return self._client

    def _record(self, record: dict[str, Any], jpeg: bytes) -> None:
        image_path = self.workspace / record["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(jpeg)
        log_path = self.workspace / "artifacts/lightnav_preview/predictions.jsonl"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, ensure_ascii=False) + "\n")

    async def execute(
        self,
        instruction: str,
        max_frames: int = 4,
        timeout_s: float = 60,
    ) -> str:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be non-empty")
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or not 1 <= max_frames <= MAX_FRAMES_PER_CALL:
            raise ValueError(f"max_frames must be an integer in [1, {MAX_FRAMES_PER_CALL}]")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(timeout_s) or not 1 <= timeout_s <= 120:
            raise ValueError("timeout_s must be a finite number in [1, 120]")

        async with self._lock:
            deadline = time.monotonic() + timeout_s
            predictions: list[dict[str, Any]] = []
            finish_reason = "frame_budget"
            while len(predictions) < max_frames:
                status = self._read_json(self.workspace / STATUS)
                if status.get("phase") in {"completed", "error"}:
                    finish_reason = "episode_finished"
                    break
                manifest_path = self.workspace / MANIFEST
                manifest = self._read_json(manifest_path)
                seq = manifest.get("observation_seq")
                episode_id = manifest.get("episode_id")
                fresh = (
                    isinstance(seq, int) and not isinstance(seq, bool)
                    and isinstance(episode_id, int) and not isinstance(episode_id, bool)
                    and (
                        self._episode_id != episode_id
                        or self._last_observation_seq is None
                        or seq - self._last_observation_seq >= self.min_observation_steps
                    )
                )
                if fresh:
                    try:
                        frame = load_head_frame(manifest_path)
                    except FileNotFoundError:
                        # The publisher can prune an old revision while we read it.
                        frame = None
                    if frame is not None and frame.observation_seq == seq and frame.episode_id == episode_id:
                        client = self._client_for_episode(episode_id)
                        try:
                            # The local HF service currently responds in ~0.1 s. Avoid
                            # an executor thread outliving this read-only Agent run.
                            reply = client.predict(frame.jpeg, instruction.strip())
                        except Exception:
                            self.close()
                            raise
                        record = {
                            "episode_id": episode_id,
                            "observation_seq": seq,
                            "revision": frame.revision,
                            "instruction": instruction.strip(),
                            "image_path": f"artifacts/lightnav_preview/frames/episode_{episode_id}_seq_{seq}.jpg",
                            "first_waypoint_robot_local": list(reply.first_waypoint),
                            "stop": reply.stop,
                            "visible": getattr(reply, "visible", None),
                            "latency_ms": reply.latency_ms,
                        }
                        self._record(record, frame.jpeg)
                        self._last_observation_seq = seq
                        predictions.append(record)
                        if reply.stop:
                            finish_reason = "model_suggested_stop"
                            break
                        continue
                if time.monotonic() >= deadline:
                    finish_reason = "observation_timeout"
                    break
                await asyncio.sleep(0.2)
            return json.dumps({
                "mode": "read_only",
                "finish_reason": finish_reason,
                "sample_count": len(predictions),
                "predictions": predictions,
                "note": "LightNav predictions were not sent to the robot or evaluator.",
            }, ensure_ascii=False)

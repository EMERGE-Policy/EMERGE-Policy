"""Robot-agnostic, frame-by-frame LightNav navigation experiment loop."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

def run_closed_loop(
    *,
    instruction: str,
    max_steps: int,
    frame_source: Callable[[], Any],
    execute_waypoint: Callable[[tuple[float, float, float], int], dict[str, Any]],
    steps_per_frame: Callable[[int], int],
    client: Any,
    trace_path: Path,
    max_wall_time_s: float,
    append_trace: bool = False,
    call_id: int | None = None,
) -> dict[str, Any]:
    """Keep one LightNav history session; execute only the first predicted pose.

    ``frame_source`` and ``execute_waypoint`` are supplied by an embodiment.
    One failed or stale action terminates this ablation; it never silently
    falls back to another navigation policy.
    """
    if not instruction.strip() or max_steps < 1 or max_wall_time_s <= 0:
        raise ValueError("instruction, max_steps and max_wall_time_s must be valid")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = trace_path.parent / "lightnav_frames"
    frame_dir.mkdir(exist_ok=True)
    saved_indices = [int(path.stem.split("_")[-1])
                     for path in frame_dir.glob("frame_*.jpg")
                     if path.stem.split("_")[-1].isdigit()] if append_trace else []
    frame_offset = max(saved_indices, default=-1) + 1
    deadline = time.monotonic() + max_wall_time_s
    steps = 0
    frames = 0
    episode_id: int | None = None
    previous_seq: int | None = None
    finish_reason = "step_budget"
    last_frame: dict[str, int | None] | None = None
    error: str | None = None
    post_budget_observation_confirmed = False
    with trace_path.open("a" if append_trace else "w", encoding="utf-8") as trace:
        while steps < max_steps:
            if time.monotonic() >= deadline:
                finish_reason = "wall_time_budget"
                break
            try:
                frame = frame_source()
            except Exception as exc:
                finish_reason = "observation_error"
                error = f"{type(exc).__name__}: {exc}"
                break
            if frame.observation_seq is None or frame.episode_id is None:
                finish_reason = "missing_frame_identity"
                break
            if episode_id is None:
                episode_id = frame.episode_id
            if frame.episode_id != episode_id:
                finish_reason = "episode_changed"
                break
            if previous_seq is not None and frame.observation_seq <= previous_seq:
                finish_reason = "stale_observation"
                break
            previous_seq = frame.observation_seq
            last_frame = {"episode_id": frame.episode_id, "observation_seq": frame.observation_seq}
            image_path = frame_dir / f"frame_{frame_offset + frames:05d}.jpg"
            image_path.write_bytes(frame.jpeg)
            try:
                reply = client.predict(frame.jpeg, instruction)
            except Exception as exc:
                finish_reason = "inference_error"
                error = f"{type(exc).__name__}: {exc}"
                break
            record: dict[str, Any] = {
                **last_frame,
                "frame_index": frame_offset + frames,
                "call_id": call_id,
                "instruction": instruction,
                "image_path": str(image_path),
                "first_waypoint_robot_local": list(reply.first_waypoint),
                "stop": reply.stop,
                "visible": reply.visible,
                "latency_ms": reply.latency_ms,
                "nav_steps_before": steps,
            }
            frames += 1
            if reply.stop:
                finish_reason = "model_stop_candidate"
                trace.write(json.dumps(record, ensure_ascii=False) + "\n")
                trace.flush()
                break
            requested = min(steps_per_frame(frames - 1), max_steps - steps)
            try:
                outcome = execute_waypoint(reply.first_waypoint, requested)
            except Exception as exc:
                finish_reason = "execution_error"
                error = f"{type(exc).__name__}: {exc}"
                record["error"] = error
                trace.write(json.dumps(record, ensure_ascii=False) + "\n")
                trace.flush()
                break
            completed = outcome.get("completed_steps")
            if isinstance(completed, bool) or not isinstance(completed, int) or not 0 <= completed <= requested:
                raise RuntimeError("gateway returned an invalid navigation step count")
            steps += completed
            record.update({"requested_steps": requested, "completed_steps": completed,
                           "gateway_status": outcome.get("status"), "nav_steps_after": steps,
                           "action": outcome.get("action")})
            trace.write(json.dumps(record, ensure_ascii=False) + "\n")
            trace.flush()
            if outcome.get("status") != "step_completed" or completed != requested:
                finish_reason = f"gateway_{outcome.get('status', 'invalid')}"
                break
        if finish_reason == "step_budget" and steps > 0:
            # The last local action is not confirmed by a newer camera frame
            # until we explicitly wait for one; retain that final frame for
            # review even though no more model prediction is allowed.
            try:
                final_frame = frame_source()
                if final_frame.episode_id != episode_id or final_frame.observation_seq <= (previous_seq or 0):
                    raise RuntimeError("final observation did not confirm the last action in the same episode")
                final_path = frame_dir / f"frame_{frame_offset + frames:05d}.jpg"
                final_path.write_bytes(final_frame.jpeg)
                post_budget_observation_confirmed = True
                last_frame = {"episode_id": final_frame.episode_id, "observation_seq": final_frame.observation_seq}
                trace.write(json.dumps({
                    "event": "final_observation", "image_path": str(final_path),
                    "observation_seq": final_frame.observation_seq,
                    "episode_id": final_frame.episode_id,
                    "frame_index": frame_offset + frames,
                    "call_id": call_id,
                    "instruction": instruction,
                    "nav_steps": steps,
                }, ensure_ascii=False) + "\n")
                trace.flush()
            except Exception as exc:
                error = f"final_observation: {type(exc).__name__}: {exc}"
                finish_reason = "observation_error"
    return {
        "finish_reason": finish_reason,
        "nav_steps": steps,
        "model_frames": frames,
        "last_frame": last_frame,
        "trace_path": str(trace_path),
        "frame_dir": str(frame_dir),
        "error": error,
        "post_budget_observation_confirmed": post_budget_observation_confirmed,
        "call_id": call_id,
        "instruction": instruction,
    }

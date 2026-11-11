"""OpenAI-compatible VLM judge for AC-WM rollouts."""

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _sample_forecast_steps(
    horizon: int, max_intervals: int = 8, required_steps: tuple[int, ...] = ()
) -> list[int]:
    """Sample rollout endpoints, spaced stills, and required decision boundaries."""
    horizon = max(0, int(horizon))
    if horizon == 0:
        return [0]
    intervals = min(max(1, int(max_intervals)), horizon)
    steps = {round(i * horizon / intervals) for i in range(intervals + 1)}
    steps.update(max(0, min(int(step), horizon)) for step in required_steps)
    return sorted(steps)


def _measure_action_contrast(candidates):
    """Record how much each pair of real proposals differs over its shared prefix."""
    import numpy as np

    candidates = list(candidates)
    pairs = []
    for i, left in enumerate(candidates):
        left_actions = np.asarray(left.actions, dtype=np.float64)
        for right in candidates[i + 1 :]:
            right_actions = np.asarray(right.actions, dtype=np.float64)
            common = min(len(left_actions), len(right_actions))
            entry = {
                "candidate_ids": [left.candidate_id, right.candidate_id],
                "shared_action_steps": common,
                "action_widths": [int(left_actions.shape[1]), int(right_actions.shape[1])],
            }
            if left_actions.shape[1] != right_actions.shape[1]:
                entry["error"] = "action widths differ; raw control contrast is not comparable"
            elif common:
                delta = np.abs(left_actions[:common] - right_actions[:common])
                entry.update(
                    {
                        "mean_abs_action_difference": round(float(delta.mean()), 8),
                        "per_dimension_mean_abs_difference": [
                            round(float(v), 8) for v in delta.mean(axis=0)
                        ],
                        "identical_shared_prefix": bool(
                            np.allclose(
                                left_actions[:common], right_actions[:common], atol=1e-8, rtol=0
                            )
                        ),
                    }
                )
            else:
                entry["error"] = "no shared action rows"
            pairs.append(entry)
    return pairs


def _measure_forecast_contrast(frame_sequences):
    """Record aligned pixel contrast between candidate forecasts as an audit signal."""
    import cv2
    import numpy as np

    candidate_ids = list(frame_sequences)
    pairs = []
    for i, left_id in enumerate(candidate_ids):
        left = {int(item["action_step"]): item["image_path"] for item in frame_sequences[left_id]}
        for right_id in candidate_ids[i + 1 :]:
            right = {
                int(item["action_step"]): item["image_path"] for item in frame_sequences[right_id]
            }
            shared = sorted(set(left) & set(right))
            measurements = []
            errors = []
            for step in shared:
                a = cv2.imread(left[step], cv2.IMREAD_COLOR)
                b = cv2.imread(right[step], cv2.IMREAD_COLOR)
                if a is None or b is None:
                    errors.append(f"could not decode aligned keyframe at action step {step}")
                    continue
                if a.shape != b.shape:
                    errors.append(
                        f"keyframe shape mismatch at action step {step}: {a.shape} vs {b.shape}"
                    )
                    continue
                measurements.append(
                    {
                        "action_step": step,
                        "mean_abs_pixel_difference": round(
                            float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))), 6
                        ),
                    }
                )
            pairs.append(
                {
                    "candidate_ids": [left_id, right_id],
                    "shared_action_steps": shared,
                    "per_step": measurements,
                    "mean_abs_pixel_difference": round(
                        float(np.mean([m["mean_abs_pixel_difference"] for m in measurements])), 6
                    )
                    if measurements
                    else None,
                    "normalized_mean_abs_pixel_difference": round(
                        float(np.mean([m["mean_abs_pixel_difference"] for m in measurements]))
                        / 255.0,
                        8,
                    )
                    if measurements
                    else None,
                    "errors": errors,
                }
            )
    return pairs


@dataclass(frozen=True, slots=True)
class VlmJudgeConfig:
    base_url: str
    model: str
    api_key_env: str = "EMERGE_VLM_API_KEY"
    timeout_s: float = 120.0
    api_key: str = ""


class OpenAICompatibleVlmJudge:
    def __init__(self, config: VlmJudgeConfig):
        self.config = config
        self.last_audit = {}

    def evaluate_candidates(self, request, rollouts):
        """Jointly compare every successful predicted candidate in one VLM call."""
        key = (
            self.config.api_key_env
            and os.environ.get(self.config.api_key_env)
            or self.config.api_key
        )
        if not key:
            raise RuntimeError(f"missing {self.config.api_key_env}")
        import cv2

        by_id = {candidate.candidate_id: candidate for candidate in request.candidates}
        content = []
        frame_sources = []

        def append_frame(path, label, frame_index=0, *, action_step=None, persist_dir=None):
            source_path = Path(path)
            if source_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                frame = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                ok = frame is not None
            else:
                capture = cv2.VideoCapture(str(source_path))
                if not capture.isOpened():
                    raise RuntimeError(f"cannot open AC-WM image/video: {path}")
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                capture.release()
            if not ok:
                raise RuntimeError(f"cannot decode frame {frame_index} from {path}")
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                raise RuntimeError(f"cannot encode frame {frame_index} from {path}")
            image_bytes = encoded.tobytes()
            image_path = None
            if persist_dir is not None:
                persist_dir = Path(persist_dir)
                persist_dir.mkdir(parents=True, exist_ok=True)
                image_path = persist_dir / f"step_{int(action_step):04d}.jpg"
                image_path.write_bytes(image_bytes)
            content.append({"type": "text", "text": label})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
                    },
                }
            )
            source = {
                "label": label,
                "path": str(image_path or path),
                "source_media_path": str(path),
                "frame_index": int(frame_index),
            }
            if action_step is not None:
                source["action_step"] = int(action_step)
            frame_sources.append(source)

        append_frame(
            request.observation_path,
            "Shared current observation (same state and camera for all candidates):",
        )
        candidate_context = []
        frame_sequences = {}
        for candidate_id, rollout in rollouts.items():
            candidate = by_id[candidate_id]
            capture = cv2.VideoCapture(str(rollout.video_path))
            if not capture.isOpened():
                raise RuntimeError(f"cannot open predicted video: {rollout.video_path}")
            count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
            capture.release()
            horizon = max(0, int(rollout.metadata.get("action_horizon", count - 1)))
            requested_execute_steps = max(
                1,
                min(
                    int(candidate.metadata.get("execute_steps", len(candidate.actions))),
                    len(candidate.actions),
                ),
            )
            evaluated_steps = min(requested_execute_steps, horizon)
            if count < evaluated_steps + 1:
                # Clamping would silently label an earlier frame as the dispatch endpoint.
                raise RuntimeError(
                    f"predicted video for {candidate_id!r} has {count} frames; "
                    f"needs at least {evaluated_steps + 1} to show the dispatched prefix endpoint"
                )
            sampled_steps = _sample_forecast_steps(horizon, required_steps=(evaluated_steps,))
            selected_steps = []
            for step in sampled_steps:
                frame_index = min(step, count - 1)
                if frame_index not in [item["frame_index"] for item in selected_steps]:
                    selected_steps.append({"action_step": step, "frame_index": frame_index})
            frame_dir = Path(rollout.video_path).parent / "vlm_frames"
            sequence = []
            safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in candidate_id)
            for item in selected_steps:
                step = item["action_step"]
                frame_index = item["frame_index"]
                boundary_label = (
                    "; end of currently dispatched action prefix" if step == evaluated_steps else ""
                )
                append_frame(
                    rollout.video_path,
                    f"Candidate {candidate_id} predicted observation at action step {step}{boundary_label}; "
                    f"ordered forecast frame {len(sequence) + 1}/{len(selected_steps)}:",
                    frame_index,
                    action_step=step,
                    persist_dir=frame_dir / safe_id,
                )
                sequence.append(
                    {
                        "action_step": step,
                        "video_frame_index": frame_index,
                        "image_path": str(frame_dir / safe_id / f"step_{step:04d}.jpg"),
                    }
                )
            frame_sequences[candidate_id] = sequence
            rollout_horizon = horizon
            candidate_context.append(
                {
                    "candidate_id": candidate_id,
                    "skill": candidate.skill_name,
                    "intent": candidate.metadata.get(
                        "intent", candidate.metadata.get("reasoning", "")
                    ),
                    "skill_parameters": candidate.metadata.get("skill_parameters", {}),
                    "proposal_rows": len(candidate.actions),
                    "requested_execute_steps": requested_execute_steps,
                    "execute_steps": evaluated_steps,
                    "executed_prefix_endpoint_step": evaluated_steps,
                    "action7_prefix": candidate.actions[: min(evaluated_steps, 8)],
                    "rollout_horizon": rollout_horizon,
                    "rollout_frame_count": count,
                    "predicted_frame_sequence": sequence,
                    "vlm_sampling_policy": "up_to_10_evenly_spaced_ordered_stills_including_rollout_and_dispatch_endpoints",
                    "rollout_metadata": dict(rollout.metadata),
                }
            )
        forecast_contrast = _measure_forecast_contrast(frame_sequences)
        action_contrast = _measure_action_contrast(
            [by_id[candidate_id] for candidate_id in rollouts]
        )
        plan = {
            "task": request.task,
            "planning_context": dict(request.planning_context),
            "camera_viewpoint": request.view_point,
            "domain": request.domain_name,
            "candidate_generation_errors": list(request.generation_errors),
            "forecast_contrast_diagnostic": forecast_contrast,
            "action_contrast_diagnostic": action_contrast,
            "candidates": candidate_context,
        }
        prompt = (
            "You are the comparative action-selection judge inside an embodied agent. "
            "The main planner's task and current subgoal, candidate skill intents/actions, the shared current observation, "
            "and each candidate's ORDERED Cosmos action-conditioned future-image sequence are provided. "
            "The ordered frames span each candidate's own evaluated prediction horizon; these horizons may differ because each skill "
            "proposes a different number of real action rows. Do not treat a longer horizon as better by itself. Candidate context separately "
            "states execute_steps: only that initial action prefix is dispatched now; frames after its endpoint depict hypothetical continued "
            "execution, not actions guaranteed to run before replanning. Prioritize visible progress by each candidate's executed-prefix endpoint "
            "and use later frames only as lower-confidence directional context. Compare local progress, risk, and control duration relative to "
            "the CURRENT subgoal and planning horizon; "
            "do not demand completion of the full task. "
            "Check temporal consistency across the ordered frames and whether gripper/object relations change in the commanded direction. "
            "Do not credit a predicted effect after execute_steps as an effect already achieved by the currently dispatched action. "
            "Cosmos can distort objects or backgrounds: do not interpret global scene warping, object morphing, or camera drift as task progress; "
            "when later frames drift, rely more on consistent early-to-middle motion and reduce confidence in the rollout. "
            "Treat this as the agent's next small control window, not an entire episode: do not require task completion, large displacement, or a visible full-task outcome. "
            "Inspect adjacent ordered frames for directionally consistent changes in gripper/object/target geometry. A small but coherent trend toward the active subgoal is useful local progress and should receive a positive score (typically at least 0.25), even when the target remains far away. "
            "Pixel-contrast diagnostics compare corresponding forecast frames; action-contrast diagnostics compare the candidates' actual shared action rows. "
            "Use both as evidence-quality checks, never as a score or preference by themselves. If action prefixes or predicted outcomes are nearly identical, "
            "do not use candidate names or intent descriptions to invent a ranking; report equal scores or low confidence unless the visual evidence distinguishes them. "
            "Do not infer progress from action intent alone: if the forecast is essentially static, directionally inconsistent, unsafe, or too distorted to trust, score it 0 or lower confidence. Penalize wrong direction and collision risk. "
            "First explain the comparative evidence, then assign each candidate an independent utility score from 0 to 1; use 0 only for unsafe, clearly wrong, untrustworthy, or no coherent useful trend. "
            'Return JSON only, exactly shaped as {"candidates":[{"candidate_id":string,"score":number,"rationale":string},...]}. '
            "Include every candidate_id exactly once. Context follows:\n"
            + json.dumps(plan, ensure_ascii=False)
        )
        body = {
            "model": self.config.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, *content]}],
        }
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        self.last_audit = {
            "stage": "request_ready",
            "model": self.config.model,
            "endpoint": endpoint,
            "candidate_ids": list(rollouts),
            "frame_sources": frame_sources,
            "frame_sequences": frame_sequences,
            "frame_sampling_policy": "up_to_10_evenly_spaced_ordered_stills_including_rollout_and_dispatch_endpoints",
            "forecast_contrast_diagnostic": forecast_contrast,
            "action_contrast_diagnostic": action_contrast,
            "prompt": prompt,
            "temperature": 0,
            "image_payloads_included": len(frame_sources),
        }
        req = Request(
            endpoint,
            data=json.dumps(body).encode(),
            headers={
                "Authorization": "Bearer " + key,
                "Content-Type": "application/json",
                "User-Agent": "OpenAI-Python/1.0",
            },
        )
        retryable_status = {429, 500, 502, 503, 504}
        retry_errors = []
        for attempt in range(1, 4):
            try:
                with urlopen(req, timeout=self.config.timeout_s) as response:
                    data = json.loads(response.read())
                self.last_audit.update(
                    {"request_attempts": attempt, "transient_request_errors": retry_errors}
                )
                break
            except HTTPError as exc:
                retry_errors.append({"attempt": attempt, "status": exc.code})
                if exc.code not in retryable_status or attempt == 3:
                    self.last_audit.update(
                        {
                            "stage": "request_failed",
                            "request_attempts": attempt,
                            "transient_request_errors": retry_errors,
                            "error": f"HTTPError: HTTP {exc.code}",
                        }
                    )
                    raise
                time.sleep(float(attempt))
            except Exception as exc:
                self.last_audit.update(
                    {
                        "stage": "request_failed",
                        "request_attempts": attempt,
                        "transient_request_errors": retry_errors,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raise
        text = data["choices"][0]["message"]["content"]
        if not isinstance(text, str):
            text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        self.last_audit.update(
            {
                "stage": "response_received",
                "response_text": text.strip(),
                "response_id": data.get("id"),
                "response_model": data.get("model"),
                "usage": data.get("usage"),
            }
        )
        parsed = json.loads(text.strip())
        entries = parsed.get("candidates")
        if not isinstance(entries, list):
            raise ValueError("joint judge response must contain a candidates array")
        results = {}
        for entry in entries:
            candidate_id = str(entry.get("candidate_id", ""))
            if candidate_id in results or candidate_id not in rollouts:
                raise ValueError(
                    f"unexpected or duplicate candidate id from judge: {candidate_id!r}"
                )
            score = float(entry["score"])
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"VLM score outside [0,1] for {candidate_id}")
            results[candidate_id] = (score, str(entry.get("rationale", "")))
        if set(results) != set(rollouts):
            raise ValueError("joint judge did not score every successfully predicted candidate")
        self.last_audit.update(
            {
                "stage": "parsed",
                "scores": {k: {"score": v[0], "rationale": v[1]} for k, v in results.items()},
            }
        )
        return results

    @classmethod
    def from_config(cls, config):
        """Build the judge from the same loaded Emerge Config as the agent run (``ac_wm`` section)."""
        settings = config.ac_wm
        provider = getattr(config.providers, settings.judge_provider, None)
        if provider is None:
            raise RuntimeError(
                f"ac_wm.judgeProvider {settings.judge_provider!r} is not a known provider"
            )
        if not provider.api_key or not provider.api_base:
            raise RuntimeError(
                f"provider {settings.judge_provider!r} needs api_key and api_base for the AC-WM judge"
            )
        return cls(
            VlmJudgeConfig(
                base_url=provider.api_base,
                model=settings.judge_model,
                api_key_env="",
                api_key=provider.api_key,
            )
        )

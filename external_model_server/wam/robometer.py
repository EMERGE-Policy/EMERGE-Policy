"""Lightweight client for the official Robometer HTTP eval server.

Wire format mirrors make_progress_sample/build_multipart_payload/
post_evaluate_batch_npy in robometer scripts/example_inference.py at
352d160389daa964788de1ec933d1925f3a6de4f. Importing that example also imports
matplotlib; keep only its small multipart transport here, not model processing.
The official server owns checkpoint loading, preprocessing and head decoding.
"""

from __future__ import annotations

import io
import json
import logging
import math
import time
from typing import Any, Sequence
from urllib.parse import urlsplit

import httpx
import numpy as np

from robot.wam.protocol import ProtocolError

DEFAULT_MODEL_PATH = "jesbu1/robometer-4b-fft-libero"
SCORE_SEMANTICS = "final_frame_task_progress"
logger = logging.getLogger(__name__)


def rgb_image(value: Any, field: str) -> np.ndarray:
    array = np.asarray(value)
    if (array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3
            or min(array.shape[:2]) < 2):
        raise ProtocolError(
            f"{field} must be RGB uint8 (H,W,3), got {array.dtype} {array.shape}",
            code="invalid_model_output",
        )
    return np.ascontiguousarray(array)


class RobometerClient:
    """Rank candidate futures by final-frame progress, never by success/Q value.

    Each sample is exactly [current, predicted future] from ONE camera. No
    history, temporal interpolation, resampling or image flipping is done here.
    The selected checkpoint must support multi-image per-frame progress tokens.
    """

    def __init__(self, url: str, *, model_path: str = DEFAULT_MODEL_PATH,
                 timeout: float = 120.0, view: str = "primary"):
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username:
            raise ValueError("robometer URL must be an http(s) base URL without credentials")
        if parts.query or parts.fragment:
            raise ValueError("robometer URL cannot contain a query or fragment")
        if not model_path.strip() or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("robometer model_path and a positive finite timeout are required")
        if view not in {"primary", "wrist"}:
            raise ValueError("robometer view must be primary or wrist")
        self.url = url.rstrip("/")
        self.model_path, self.timeout, self.view = model_path, timeout, view
        self._http = httpx.Client(trust_env=False, timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def _model_info(self) -> dict[str, Any]:
        response = self._http.get(self.url + "/model_info")
        response.raise_for_status()
        info = response.json()
        if not isinstance(info, dict) or info.get("model_path") != self.model_path:
            raise ValueError("Robometer /model_info model_path does not match configured model_path")
        config = info.get("experiment_config", {}).get("model", {})
        for key in ("train_progress_head", "use_multi_image", "use_per_frame_progress_token"):
            if config.get(key) is not True:
                raise ValueError(f"Robometer checkpoint requires model.{key}=true")
        data_config = info.get("experiment_config", {}).get("data", {})
        # The official collator reads data flags, while RBM reads model flags.
        for key in ("use_multi_image", "use_per_frame_progress_token"):
            if data_config.get(key) is not True:
                raise ValueError(f"Robometer checkpoint requires data.{key}=true")
        if data_config.get("shuffle_progress_frames") is True:
            raise ValueError("Robometer checkpoint must not shuffle progress frames")
        return info

    def check_ready(self) -> dict[str, Any]:
        """Check the official server and checkpoint contract without inference."""
        try:
            return self._model_info()
        except httpx.TimeoutException as exc:
            raise ProtocolError(f"Robometer readiness timed out at {self.url}",
                                code="robometer_timeout") from exc
        except httpx.HTTPError as exc:
            raise ProtocolError(f"Robometer HTTP failure at {self.url}: {exc}",
                                code="robometer_unavailable") from exc
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            raise ProtocolError(f"Invalid Robometer /model_info at {self.url}: {exc}",
                                code="robometer_invalid_output") from exc

    def score(self, current: np.ndarray, futures: Sequence[np.ndarray],
              task_instruction: str, indices: Sequence[int]) -> tuple[list[float], dict[str, Any]]:
        current = rgb_image(current, "current frame")
        if not isinstance(task_instruction, str) or not task_instruction.strip():
            raise ProtocolError("Robometer requires original task_instruction")
        if not futures or list(indices) != list(range(len(futures))):
            raise ProtocolError("Robometer candidate indices must be contiguous and match futures")
        files, data = {}, {"use_frame_steps": "false"}
        for position, (index, future) in enumerate(zip(indices, futures)):
            future = rgb_image(future, f"candidate {index} future")
            if current.shape != future.shape:
                raise ProtocolError("Current and future frames must share the Cosmos-processed image shape",
                                    code="invalid_model_output")
            frames = np.stack((current, future))
            buffer = io.BytesIO()
            np.save(buffer, frames, allow_pickle=False)
            key = f"sample_{position}_trajectory_frames"
            files[key] = (key + ".npy", buffer.getvalue(), "application/octet-stream")
            data[f"sample_{position}"] = json.dumps({
                "sample_type": "progress",
                "trajectory": {
                    "frames": {"__numpy_file__": key}, "frames_shape": list(frames.shape),
                    "task": task_instruction, "id": f"candidate_{index}",
                    "metadata": {"subsequence_length": 2}, "video_embeddings": None,
                },
            })
        started = time.perf_counter()
        try:
            info = self.check_ready()
            response = self._http.post(self.url + "/evaluate_batch_npy", data=data, files=files)
            response.raise_for_status()
            payload = response.json()
            traces = payload.get("outputs_progress", {}).get("progress_pred")
            if not isinstance(traces, list) or len(traces) != len(futures):
                raise ValueError("Robometer progress output count does not match candidates")
            scores = []
            for index, trace in zip(indices, traces):
                # Official multi-image mode (use_frame_steps=false) preserves our
                # two image tokens. A different length would make [-1] ambiguous.
                if not isinstance(trace, list) or len(trace) != 2:
                    raise ValueError(f"Candidate {index} requires exactly two per-frame progress outputs")
                if any(isinstance(x, bool) or not isinstance(x, (int, float))
                       or not math.isfinite(x) for x in trace):
                    raise ValueError(f"Candidate {index} progress contains invalid/non-finite values")
                scores.append(float(trace[-1]))
        except httpx.TimeoutException as exc:
            raise ProtocolError("Robometer scoring timed out; no candidate was selected",
                                code="robometer_timeout") from exc
        except httpx.HTTPError as exc:
            raise ProtocolError(f"Robometer HTTP failure: {exc}", code="robometer_unavailable") from exc
        except (ValueError, TypeError, AttributeError, OverflowError) as exc:
            raise ProtocolError(f"Invalid Robometer response: {exc}",
                                code="robometer_invalid_output") from exc
        metadata = {
            "backend": "robometer", "model_id": info["model_path"],
            "score_semantics": SCORE_SEMANTICS, "view": self.view,
            "input_frame_count": 2, "use_frame_steps": False,
            "inference_dtype": info["experiment_config"]["model"].get("torch_dtype", "unknown"),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        logger.info("Robometer scores | model=%s indices=%s scores=%s latency_ms=%.1f",
                    info["model_path"], list(indices), scores, metadata["latency_ms"])
        return scores, metadata

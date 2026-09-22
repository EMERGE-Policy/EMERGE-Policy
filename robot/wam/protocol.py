"""Dependency-light wire protocol shared by the WAM client and server."""

from __future__ import annotations

import base64
import binascii
import io
import math
from typing import Any, Mapping

from external_model_server.model_service.contracts import ServiceExpectation

REQUEST_SCHEMA_V1 = "Emerge.cosmos_policy_wam.request.v1"
REQUEST_SCHEMA_V2 = "Emerge.cosmos_policy_wam.request.v2"
REQUEST_SCHEMA = REQUEST_SCHEMA_V2
RESPONSE_SCHEMA = "Emerge.cosmos_policy_wam.response.v1"
WAM_SERVICE = ServiceExpectation("cosmos_policy_wam", REQUEST_SCHEMA, RESPONSE_SCHEMA)
ERROR_SCHEMA = "Emerge.cosmos_policy_wam.error.v1"
ACTION_DIM = 7
MAX_SEARCH_CANDIDATES = 64
SEARCH_SCORE_MODES = frozenset({"none", "joint_value", "q_value"})
DEFAULT_SEARCH_CANDIDATES = 4
DEFAULT_SEARCH_SCORE_MODE = "joint_value"
CONDITIONING_MODES = frozenset({"task", "phase", "task_with_phase"})


class ProtocolError(ValueError):
    """A malformed request or model response."""

    def __init__(self, message: str, *, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.code = code


def resolve_conditioning_text(
    task_instruction: str,
    phase_instruction: str | None,
    conditioning_mode: str,
) -> str:
    """Resolve the exact text encoded by Cosmos while retaining both intents."""
    task = str(task_instruction).strip()
    phase = str(phase_instruction or "").strip()
    if not task:
        raise ProtocolError("task_instruction must be a non-empty string")
    if conditioning_mode not in CONDITIONING_MODES:
        raise ProtocolError(
            f"conditioning_mode must be one of {sorted(CONDITIONING_MODES)}"
        )
    if conditioning_mode in {"phase", "task_with_phase"} and not phase:
        raise ProtocolError(
            f"phase_instruction is required for conditioning_mode={conditioning_mode!r}"
        )
    if conditioning_mode == "task":
        return task
    if conditioning_mode == "phase":
        return phase
    return f"Task: {task}\nCurrent phase: {phase}"


def _as_uint8_rgb(image: Any):
    import numpy as np

    array = np.asarray(image)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ProtocolError("image must be uint8 with shape (H, W, 3)")
    if min(array.shape[:2]) < 2:
        raise ProtocolError("image height and width must be at least 2")
    return np.ascontiguousarray(array)


def encode_image(image: Any) -> dict[str, Any]:
    """Encode one RGB image as a self-describing PNG payload."""
    from PIL import Image

    array = _as_uint8_rgb(image)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return {
        "encoding": "png_base64",
        "shape": list(array.shape),
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def decode_image(payload: Any, *, field: str):
    """Decode and validate an RGB image payload or protocol ndarray."""
    import numpy as np
    from PIL import Image

    if not isinstance(payload, Mapping):
        try:
            return _as_uint8_rgb(payload)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"{field} must be an RGB uint8 image") from exc
    if payload.get("encoding") != "png_base64":
        raise ProtocolError(f"{field}.encoding must be png_base64")
    encoded = payload.get("data")
    if not isinstance(encoded, str) or not encoded:
        raise ProtocolError(f"{field}.data must be a non-empty base64 string")
    try:
        raw = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (binascii.Error, OSError, ValueError) as exc:
        raise ProtocolError(f"{field} is not a valid PNG image") from exc
    expected_shape = payload.get("shape")
    if expected_shape is not None and list(array.shape) != list(expected_shape):
        raise ProtocolError(f"{field}.shape does not match decoded image")
    return np.ascontiguousarray(array)


def _validate_finite_vector(value: Any, *, field: str, size: int):
    import numpy as np

    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,):
        raise ProtocolError(f"{field} must have shape ({size},)")
    if not np.isfinite(array).all():
        raise ProtocolError(f"{field} must contain only finite values")
    return array


def validate_candidate_request(payload: Any) -> dict[str, Any]:
    """Validate the requested best-of-N candidate configuration."""
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise ProtocolError("candidate_request must be an object")
    count = payload.get("num_candidates", DEFAULT_SEARCH_CANDIDATES)
    if isinstance(count, bool) or not isinstance(count, int):
        raise ProtocolError("candidate_request.num_candidates must be an integer")
    if not 1 <= count <= MAX_SEARCH_CANDIDATES:
        raise ProtocolError(
            f"candidate_request.num_candidates must be in [1, {MAX_SEARCH_CANDIDATES}]"
        )
    default_mode = DEFAULT_SEARCH_SCORE_MODE if count > 1 else "none"
    score_mode = payload.get("score_mode", default_mode)
    if not isinstance(score_mode, str) or score_mode not in SEARCH_SCORE_MODES:
        raise ProtocolError("candidate_request.score_mode is not supported")
    if count > 1 and score_mode == "none":
        raise ProtocolError(
            "score_mode none is invalid when num_candidates is greater than one"
        )
    return {"num_candidates": count, "score_mode": score_mode}


def validate_actions(actions: Any):
    """Validate and normalize a model action chunk to contiguous float32."""
    import numpy as np

    array = np.asarray(actions, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != ACTION_DIM or array.shape[0] < 1:
        raise ProtocolError("actions must have shape (N, 7)", code="invalid_model_output")
    if not np.isfinite(array).all():
        raise ProtocolError(
            "actions must contain only finite values", code="invalid_model_output"
        )
    return np.ascontiguousarray(array)


def validate_candidate_response(payload: Any) -> dict[str, Any]:
    """Validate all returned candidates without selecting one."""
    if not isinstance(payload, Mapping):
        raise ProtocolError("response body must be an object", code="invalid_model_output")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ProtocolError(
            "candidates must be a non-empty list", code="invalid_model_output"
        )
    request = validate_candidate_request(
        {
            "num_candidates": len(candidates),
            "score_mode": payload.get("score_mode", "none"),
        }
    )
    normalized = []
    for expected_index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            raise ProtocolError("candidate must be an object", code="invalid_model_output")
        if candidate.get("index") != expected_index:
            raise ProtocolError(
                "candidate indices must be contiguous from zero",
                code="invalid_model_output",
            )
        seed = candidate.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ProtocolError(
                "candidate.seed must be an integer", code="invalid_model_output"
            )
        score = candidate.get("score")
        if request["score_mode"] == "none":
            if score is not None:
                raise ProtocolError(
                    "unscored candidates must use score=null",
                    code="invalid_model_output",
                )
            normalized_score = None
        else:
            try:
                normalized_score = float(score)
            except (TypeError, ValueError) as exc:
                raise ProtocolError(
                    "candidate.score must be numeric", code="invalid_model_output"
                ) from exc
            if not math.isfinite(normalized_score):
                raise ProtocolError(
                    "candidate.score must be finite", code="invalid_model_output"
                )
        normalized.append(
            {
                "index": expected_index,
                "seed": seed,
                "actions": validate_actions(candidate.get("actions")),
                "score": normalized_score,
            }
        )
    declared_chunk_size = payload.get("chunk_size")
    if declared_chunk_size is not None:
        if (
            isinstance(declared_chunk_size, bool)
            or not isinstance(declared_chunk_size, int)
            or declared_chunk_size < 1
        ):
            raise ProtocolError(
                "chunk_size must be a positive integer", code="invalid_model_output"
            )
        if any(
            len(candidate["actions"]) != declared_chunk_size for candidate in normalized
        ):
            raise ProtocolError(
                "chunk_size does not match candidate actions",
                code="invalid_model_output",
            )
    chunk_sizes = {len(candidate["actions"]) for candidate in normalized}
    if len(chunk_sizes) != 1:
        raise ProtocolError(
            "all candidates must use the same action chunk size",
            code="invalid_model_output",
        )
    return {"score_mode": request["score_mode"], "candidates": normalized}


def validate_request(payload: Any) -> dict[str, Any]:
    """Validate a request and return decoded, normalized values."""
    if not isinstance(payload, Mapping):
        raise ProtocolError("request body must be a JSON object")
    schema = payload.get("schema")
    if schema not in {REQUEST_SCHEMA_V1, REQUEST_SCHEMA_V2}:
        raise ProtocolError(
            f"schema must be {REQUEST_SCHEMA_V1!r} or {REQUEST_SCHEMA_V2!r}"
        )
    if schema == REQUEST_SCHEMA_V1:
        task_instruction = payload.get("instruction")
        phase_instruction = None
        conditioning_mode = "task"
    else:
        task_instruction = payload.get("task_instruction")
        phase_instruction = payload.get("phase_instruction")
        conditioning_mode = payload.get("conditioning_mode", "task")
    if not isinstance(task_instruction, str) or not task_instruction.strip():
        raise ProtocolError("task_instruction must be a non-empty string")
    if phase_instruction is not None and not isinstance(phase_instruction, str):
        raise ProtocolError("phase_instruction must be a string when provided")
    conditioning_text = resolve_conditioning_text(
        task_instruction, phase_instruction, conditioning_mode
    )
    if conditioning_mode == "task":
        phase_instruction = None
    observation = payload.get("observation")
    if not isinstance(observation, Mapping):
        raise ProtocolError("observation must be an object")
    primary = decode_image(
        observation.get("primary_image"), field="observation.primary_image"
    )
    wrist = decode_image(
        observation.get("wrist_image"), field="observation.wrist_image"
    )
    proprio = _validate_finite_vector(
        observation.get("proprio"), field="observation.proprio", size=9
    )
    seed = payload.get("seed", 1)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ProtocolError("seed must be an integer")
    return {
        "instruction": conditioning_text,
        "task_instruction": task_instruction.strip(),
        "phase_instruction": str(phase_instruction or "").strip() or None,
        "conditioning_mode": conditioning_mode,
        "conditioning_text": conditioning_text,
        "primary_image": primary,
        "wrist_image": wrist,
        "proprio": proprio,
        "seed": seed,
        "candidate_request": validate_candidate_request(payload.get("candidate_request")),
    }


def make_response(actions: Any, *, model: str, latency_ms: float) -> dict[str, Any]:
    array = validate_actions(actions)
    return {
        "schema": RESPONSE_SCHEMA,
        "actions": array.tolist(),
        "chunk_size": int(array.shape[0]),
        "action_dim": ACTION_DIM,
        "model": model,
        "latency_ms": round(ensure_finite_latency(latency_ms), 3),
    }


def make_candidate_response(
    candidates: Any,
    *,
    score_mode: str,
    model: str,
    latency_ms: float,
) -> dict[str, Any]:
    """Serialize every candidate; selection stays on the Emerge side."""
    normalized = validate_candidate_response(
        {"candidates": candidates, "score_mode": score_mode}
    )
    chunk_size = len(normalized["candidates"][0]["actions"])
    return {
        "schema": RESPONSE_SCHEMA,
        "candidates": [
            {
                "index": candidate["index"],
                "seed": candidate["seed"],
                "actions": candidate["actions"].tolist(),
                "score": candidate["score"],
            }
            for candidate in normalized["candidates"]
        ],
        "candidate_count": len(normalized["candidates"]),
        "chunk_size": chunk_size,
        "score_mode": normalized["score_mode"],
        "action_dim": ACTION_DIM,
        "model": model,
        "latency_ms": round(ensure_finite_latency(latency_ms), 3),
    }


def make_error(code: str, message: str, *, retryable: bool = False) -> dict[str, Any]:
    return {
        "schema": ERROR_SCHEMA,
        "error": {
            "code": str(code),
            "message": str(message),
            "retryable": bool(retryable),
        },
    }


def ensure_finite_latency(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) else 0.0

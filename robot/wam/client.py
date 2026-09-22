"""Emerge-side client for the shared Cosmos Policy WAM server protocol."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import httpx
import numpy as np

from external_model_server.model_service.client import ModelClient
from external_model_server.model_service.contracts import ServiceError

from .protocol import (
    DEFAULT_SEARCH_CANDIDATES,
    DEFAULT_SEARCH_SCORE_MODE,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    WAM_SERVICE,
    validate_actions,
    validate_candidate_request,
    validate_candidate_response,
)

logger = logging.getLogger(__name__)


class CosmosWAMClient:
    """Synchronous client using the repository-wide websocket/msgpack service."""

    def __init__(
        self,
        *,
        timeout: float = 120.0,
        model_id: str | None = None,
        discovery=None,
    ) -> None:
        self._client = ModelClient(replace(WAM_SERVICE, model_id=model_id),
                                   timeout=timeout, discovery=discovery)

    def health_check(self) -> bool:
        try:
            return self._client.health()["status"] == "ready"
        except (httpx.HTTPError, ServiceError, ValueError, OSError) as exc:
            logger.warning("Cosmos WAM health check failed: %s", exc)
            return False

    def infer(
        self,
        primary_image: Any,
        wrist_image: Any,
        proprio: Any,
        instruction: str | None = None,
        *,
        task_instruction: str | None = None,
        phase_instruction: str | None = None,
        conditioning_mode: str = "task",
        seed: int = 1,
        num_candidates: int = DEFAULT_SEARCH_CANDIDATES,
        score_mode: str = DEFAULT_SEARCH_SCORE_MODE,
    ) -> dict[str, Any]:
        candidate_request = validate_candidate_request(
            {"num_candidates": num_candidates, "score_mode": score_mode}
        )
        task_instruction = str(task_instruction or instruction or "").strip()
        conditioning_mode = str(conditioning_mode)
        phase_instruction = (
            str(phase_instruction).strip() if phase_instruction is not None else None
        )
        if conditioning_mode == "task":
            phase_instruction = None
        request_payload = {
            "schema": REQUEST_SCHEMA,
            "task_instruction": task_instruction,
            "conditioning_mode": conditioning_mode,
            "seed": int(seed),
            "candidate_request": candidate_request,
            "observation": {
                "primary_image": np.asarray(primary_image, dtype=np.uint8),
                "wrist_image": np.asarray(wrist_image, dtype=np.uint8),
                "proprio": np.asarray(proprio, dtype=np.float32),
            },
        }
        if phase_instruction is not None:
            request_payload["phase_instruction"] = phase_instruction
        payload = self._client.infer(request_payload)
        if payload.get("schema") != RESPONSE_SCHEMA:
            raise RuntimeError(
                f"Cosmos WAM response schema mismatch: {payload.get('schema')!r}"
            )
        if "candidates" in payload:
            normalized = validate_candidate_response(payload)
            if len(normalized["candidates"]) != candidate_request["num_candidates"]:
                raise RuntimeError("Cosmos WAM returned an unexpected candidate count")
            if normalized["score_mode"] != candidate_request["score_mode"]:
                raise RuntimeError("Cosmos WAM returned an unexpected score mode")
            payload["candidates"] = normalized["candidates"]
            payload["score_mode"] = normalized["score_mode"]
            if len(normalized["candidates"]) == 1:
                payload["actions"] = normalized["candidates"][0]["actions"]
        else:
            if candidate_request != {"num_candidates": 1, "score_mode": "none"}:
                raise RuntimeError("Cosmos WAM response omitted requested candidates")
            actions = validate_actions(payload.get("actions"))
            payload["actions"] = actions
            payload["score_mode"] = "none"
            payload["candidates"] = [
                {
                    "index": 0,
                    "seed": int(seed),
                    "actions": actions,
                    "score": None,
                }
            ]
        return payload

    def close(self) -> None:
        """Release the public model service connection."""
        self._client.close()

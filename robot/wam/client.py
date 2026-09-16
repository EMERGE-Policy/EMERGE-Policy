"""Emerge-side client for the shared Cosmos Policy WAM server protocol."""

from __future__ import annotations

import logging
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import numpy as np

from external_model_server.protocol import pack_message, unpack_message

from .protocol import (
    DEFAULT_SEARCH_CANDIDATES,
    DEFAULT_SEARCH_SCORE_MODE,
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    validate_actions,
    validate_candidate_request,
    validate_candidate_response,
)

logger = logging.getLogger(__name__)
DEFAULT_ENDPOINT = "ws://127.0.0.1:8003"


def _websocket_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme in {"ws", "wss"}:
        return endpoint
    if parsed.scheme == "http":
        return urllib.parse.urlunparse(parsed._replace(scheme="ws"))
    if parsed.scheme == "https":
        return urllib.parse.urlunparse(parsed._replace(scheme="wss"))
    raise ValueError("WAM endpoint must use ws://, wss://, http://, or https://")


def _http_endpoint(endpoint: str) -> str:
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.scheme in {"http", "https"}:
        return endpoint
    if parsed.scheme == "ws":
        return urllib.parse.urlunparse(parsed._replace(scheme="http"))
    if parsed.scheme == "wss":
        return urllib.parse.urlunparse(parsed._replace(scheme="https"))
    raise ValueError("WAM endpoint must use ws://, wss://, http://, or https://")


class CosmosWAMClient:
    """Synchronous client using the repository-wide websocket/msgpack service."""

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        *,
        timeout: float = 120.0,
    ) -> None:
        endpoint = str(endpoint).strip().rstrip("/")
        if not endpoint:
            raise ValueError("WAM endpoint must be non-empty")
        self.endpoint = endpoint
        self.websocket_endpoint = _websocket_endpoint(endpoint)
        self.http_endpoint = _http_endpoint(endpoint)
        self.timeout = max(0.1, float(timeout))
        self._websocket: Any | None = None

    def health_check(self) -> bool:
        try:
            request = urllib.request.Request(f"{self.http_endpoint}/healthz")
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(request, timeout=min(self.timeout, 3.0)) as response:
                return response.status == 200
        except (OSError, urllib.error.URLError, ValueError) as exc:
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
        packed_request = pack_message(request_payload)
        response: Any = None
        for attempt in range(2):
            try:
                websocket = self._get_websocket()
                websocket.send(packed_request, opcode=2)
                response = unpack_message(websocket.recv())
                break
            except Exception as exc:
                self._close_websocket()
                if attempt == 1:
                    raise RuntimeError(
                        f"Cosmos WAM server unavailable at {self.websocket_endpoint}"
                    ) from exc
                logger.info("Reconnecting to Cosmos WAM after a stale connection")

        if not isinstance(response, dict):
            raise RuntimeError("Cosmos WAM response must be a mapping")
        if not response.get("ok", False):
            error = response.get("error") or {}
            raise RuntimeError(
                f"Cosmos WAM inference failed "
                f"[{error.get('type', 'unknown')}]: "
                f"{error.get('message', 'unknown error')}"
            )
        payload = response.get("result")
        if not isinstance(payload, dict):
            raise RuntimeError("Cosmos WAM result must be a mapping")
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
        """Close the persistent websocket connection, if it was opened."""
        self._close_websocket()

    def _get_websocket(self) -> Any:
        if self._websocket is not None:
            return self._websocket
        import websocket

        self._websocket = websocket.create_connection(
            self.websocket_endpoint,
            timeout=self.timeout,
            http_proxy_host=None,
            http_proxy_port=None,
            http_no_proxy=["localhost", "127.0.0.1", "::1"],
        )
        metadata = unpack_message(self._websocket.recv())
        if not isinstance(metadata, dict) or metadata.get("type") != "metadata":
            self._close_websocket()
            raise RuntimeError("Cosmos WAM server did not return metadata")
        return self._websocket

    def _close_websocket(self) -> None:
        if self._websocket is None:
            return
        try:
            self._websocket.close()
        finally:
            self._websocket = None

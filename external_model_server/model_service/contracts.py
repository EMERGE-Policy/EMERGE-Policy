"""Dependency-free contracts at the model service network boundary."""

import math
import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

PROTOCOL_VERSION = 1
HEALTH_SCHEMA = "emerge.model-health.v1"
STATES = {"starting", "ready", "draining", "failed", "stopped"}
MAX_MESSAGE_BYTES = 256 * 1024 * 1024


class ServiceError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable

    def to_dict(self) -> dict:
        return {"code": self.code, "message": str(self), "retryable": self.retryable}


class ModelFailureError(RuntimeError):
    """An adapter explicitly reports that its model can no longer serve."""


@dataclass(frozen=True)
class ServiceDescriptor:
    service: str
    model_id: str
    input_schema: str
    output_schema: str
    capabilities: tuple[str, ...] = ("infer",)

    def __post_init__(self):
        if not isinstance(self.service, str) or not re.fullmatch(r"[a-z][a-z0-9_\-]*", self.service):
            raise ValueError("service must be a lowercase machine identifier")
        if any(not isinstance(v, str) or not v for v in (
            self.model_id, self.input_schema, self.output_schema,
        )):
            raise ValueError("model_id and input/output schema must be non-empty strings")
        if not isinstance(self.capabilities, (list, tuple)) or any(
            not isinstance(v, str) for v in self.capabilities
        ) or "infer" not in self.capabilities:
            raise ValueError("capabilities must include infer")

    def to_dict(self) -> dict:
        return asdict(self)


def validate_description(payload: Any, *, health: bool = False) -> dict:
    """Validate either a health document or an initial websocket handshake."""
    try:
        if not isinstance(payload, dict):
            raise ValueError("expected an object")
        if health:
            if payload.get("schema") != HEALTH_SCHEMA:
                raise ValueError("unsupported health schema")
        elif payload.get("type") != "metadata":
            raise ValueError("expected model service metadata")
        if type(payload.get("protocol_version")) is not int or payload["protocol_version"] != PROTOCOL_VERSION:
            raise ValueError("unsupported model protocol version")
        ServiceDescriptor(**{k: payload[k] for k in (
            "service", "model_id", "input_schema", "output_schema", "capabilities",
        )})
        if not isinstance(payload["instance_id"], str) or not payload["instance_id"]:
            raise ValueError("missing instance_id")
        if not isinstance(payload["status"], str) or payload["status"] not in STATES:
            raise ValueError("invalid model service state")
    except (KeyError, TypeError, ValueError) as exc:
        raise ServiceError("INVALID_RESPONSE", str(exc)) from exc
    return payload


@dataclass(frozen=True)
class ServiceExpectation:
    service: str
    input_schema: str
    output_schema: str
    model_id: str | None = None

    def check(self, description: dict, *, ready: bool = True) -> None:
        for field, expected in asdict(self).items():
            if expected is not None and description[field] != expected:
                code = "SERVICE_MISMATCH" if field in {"service", "model_id"} else "SCHEMA_MISMATCH"
                raise ServiceError(code, f"Expected {field}={expected!r}, got {description[field]!r}")
        if ready and description["status"] != "ready":
            raise ServiceError("NOT_READY", f"{self.service} is {description['status']}", retryable=True)


def validate_request(message: Any) -> dict:
    if not isinstance(message, dict):
        raise ServiceError("INVALID_REQUEST", "Request must be an object")
    if not isinstance(message.get("request_id"), str) or not 0 < len(message["request_id"]) <= 128:
        raise ServiceError("INVALID_REQUEST", "request_id must be a non-empty string of at most 128 characters")
    if type(message.get("protocol_version")) is not int or message["protocol_version"] != PROTOCOL_VERSION:
        raise ServiceError("SCHEMA_MISMATCH", "Unsupported model protocol version")
    if message.get("operation") != "infer" or not isinstance(message.get("payload"), dict):
        raise ServiceError("INVALID_REQUEST", "Expected infer operation and object payload")
    timeout = message.get("timeout_s")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ServiceError("INVALID_REQUEST", "timeout_s must be finite and positive")
    return message


def response_result(response: Any, request_id: str) -> dict:
    if not isinstance(response, dict) or response.get("request_id") != request_id:
        raise ServiceError("INVALID_RESPONSE", "Response request_id mismatch")
    if response.get("ok") is True and isinstance(response.get("result"), dict):
        return response["result"]
    error = response.get("error")
    if response.get("ok") is False and isinstance(error, dict) and all(
        isinstance(error.get(k), str) for k in ("code", "message")
    ) and type(error.get("retryable")) is bool:
        raise ServiceError(error["code"], error["message"], retryable=error["retryable"])
    raise ServiceError("INVALID_RESPONSE", "Malformed model response")


def websocket_url(endpoint: str) -> str:
    parts = urlsplit(endpoint)
    if parts.scheme not in {"http", "https", "ws", "wss"} or not parts.hostname or parts.username:
        raise ValueError("Model endpoint must be an http(s) or ws(s) URL without credentials")
    port = parts.port  # validate the port at the configuration boundary
    host = parts.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    scheme = "wss" if parts.scheme in {"https", "wss"} else "ws"
    authority = host if port is None else f"{host}:{port}"
    return urlunsplit((scheme, authority, parts.path.rstrip("/"), parts.query, ""))


def health_url(endpoint: str) -> str:
    parts = urlsplit(websocket_url(endpoint))
    return urlunsplit(("https" if parts.scheme == "wss" else "http", parts.netloc, "/healthz", "", ""))

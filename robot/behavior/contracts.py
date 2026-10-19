"""Contracts for the first native BEHAVIOR-1K evaluation path.

Everything in this module is standard library only.  It is imported by the
controller driver, the evaluation runner, and the offline checks, so it must
never pull in numpy, PIL, msgpack, or websockets.

The module owns three kinds of truth that used to be duplicated across the
evaluation path:

* the shape rules for ``vla_execute`` parameters and for the driver
  configuration object, and
* the classification of an episode's outcome.
"""

from __future__ import annotations

import math
import re
from typing import Any

BENCHMARK = "BEHAVIOR-1K"

EVALUATION_STATUS_SCHEMA = "behavior.eval_status.v1"
EVALUATION_PHASES = frozenset({"running", "completed", "error"})

#: Action result prefixes understood by ``Emerge.utils.action_queue.infer_terminal_status``.
FAILED_PREFIX = "Failed:"
INTERRUPTED_PREFIX = "Interrupted:"


class ContractError(ValueError):
    """Base class for contract violations reported to the Agent."""


class ActionValidationError(ContractError):
    """Raised when a ``vla_execute`` request cannot be honored."""


class EvaluationValidationError(ContractError):
    """Raised when the official evaluator status is not usable."""


class ConfigValidationError(ContractError):
    """Raised when the driver configuration object is not usable."""


# --------------------------------------------------------------------------- #
# Action contract
# --------------------------------------------------------------------------- #


def normalize_instruction(text: str) -> str:
    """Collapse whitespace so cosmetic differences never reject a valid prompt."""
    return " ".join(str(text).split())


def validate_instruction(value: Any, *, required_instruction: str) -> str:
    """Return the normalized instruction or raise :class:`ActionValidationError`.

    This initial task runner deliberately exposes only the official episode
    instruction to the policy. Rejecting a rewrite locally keeps the Agent from
    believing it changed the prompt.
    """
    if value is None:
        raise ActionValidationError("vla_execute requires instruction or prompt")
    if not isinstance(value, str):
        raise ActionValidationError("vla_execute instruction must be a string")
    instruction = normalize_instruction(value)
    if not instruction:
        raise ActionValidationError("vla_execute requires a non-empty instruction")
    expected = normalize_instruction(required_instruction)
    if instruction != expected:
        raise ActionValidationError(
            "this evaluation entrypoint requires the exact instruction: " f"{expected}"
        )
    return instruction


def validate_step(value: Any, *, max_steps: int) -> int:
    """Return ``step`` as a plain ``int`` or raise :class:`ActionValidationError`.

    ``bool`` and fractional numbers are rejected because ``int(True) == 1`` and
    ``int(100.7) == 100`` would otherwise silently turn a malformed request into
    a valid budget.  Integral floats (``100.0``) and digit strings (``"100"``)
    stay accepted for JSON-encoder compatibility.
    """
    if isinstance(value, bool):
        raise ActionValidationError("vla_execute step must be an integer, not a boolean")
    if isinstance(value, int):
        step = value
    elif isinstance(value, float):
        if not math.isfinite(value) or value != int(value):
            raise ActionValidationError("vla_execute step must be a whole number")
        step = int(value)
    elif isinstance(value, str):
        text = value.strip()
        if not re.fullmatch(r"[+-]?\d+", text):
            raise ActionValidationError("vla_execute step must be an integer")
        step = int(text)
    else:
        raise ActionValidationError(
            f"vla_execute step must be an integer, got {type(value).__name__}"
        )
    if step < 1:
        raise ActionValidationError("vla_execute step must be a positive integer")
    if step > max_steps:
        raise ActionValidationError(f"vla_execute step cannot exceed {max_steps}")
    return step


def validate_action(
    params: Any,
    *,
    required_instruction: str,
    max_steps: int,
) -> tuple[str, int]:
    """Validate one ``vla_execute`` parameter object.

    Returns the normalized ``(instruction, step)`` pair.
    """
    if not isinstance(params, dict):
        raise ActionValidationError("vla_execute parameters must be an object")
    instruction = validate_instruction(
        params.get("instruction", params.get("prompt")),
        required_instruction=required_instruction,
    )
    if "step" not in params:
        raise ActionValidationError("vla_execute requires step")
    step = validate_step(params["step"], max_steps=max_steps)
    return instruction, step


# --------------------------------------------------------------------------- #
# Driver configuration contract
# --------------------------------------------------------------------------- #

_REQUIRED_CONFIG_FIELDS = (
    "workspace",
    "upstream_url",
    "required_instruction",
    "max_action_steps",
    "expected_task",
)


def _require_mapping(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ConfigValidationError("driver config must be a JSON object")
    return payload


def _positive_int(payload: dict[str, Any], key: str, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigValidationError(f"{key} must be a positive integer")
    return int(value)


def _positive_float(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool):
        raise ConfigValidationError(f"{key} must be a positive number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"{key} must be a positive number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ConfigValidationError(f"{key} must be a positive number")
    return number


def _camera_calibration(value: Any) -> dict[str, Any] | None:
    """Validate the optional camera calibration without importing NumPy."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigValidationError("camera_calibration must be an object or null")
    cameras = value.get("cameras")
    if not isinstance(cameras, dict):
        raise ConfigValidationError("camera_calibration.cameras must be an object")
    required = {"left_wrist", "right_wrist", "head"}
    missing = sorted(required - set(cameras))
    if missing:
        raise ConfigValidationError(
            "camera_calibration.cameras is missing: " + ", ".join(missing)
        )
    normalized_cameras: dict[str, Any] = {}
    for name in sorted(required):
        camera = cameras[name]
        if not isinstance(camera, dict):
            raise ConfigValidationError(f"camera_calibration.cameras.{name} must be an object")
        resolution = camera.get("resolution")
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in resolution)
        ):
            raise ConfigValidationError(
                f"camera_calibration.cameras.{name}.resolution must be [width, height]"
            )
        matrix = camera.get("intrinsics")
        if not isinstance(matrix, list) or len(matrix) != 3 or any(
            not isinstance(row, list) or len(row) != 3 for row in matrix
        ):
            raise ConfigValidationError(
                f"camera_calibration.cameras.{name}.intrinsics must be a 3x3 matrix"
            )
        try:
            numeric = [[float(item) for item in row] for row in matrix]
        except (TypeError, ValueError) as exc:
            raise ConfigValidationError(
                f"camera_calibration.cameras.{name}.intrinsics must contain numbers"
            ) from exc
        if not all(math.isfinite(item) for row in numeric for item in row):
            raise ConfigValidationError(
                f"camera_calibration.cameras.{name}.intrinsics must contain finite numbers"
            )
        if numeric[0][0] <= 0.0 or numeric[1][1] <= 0.0 or numeric[2] != [0.0, 0.0, 1.0]:
            raise ConfigValidationError(
                f"camera_calibration.cameras.{name}.intrinsics is not a valid pinhole matrix"
            )
        normalized_cameras[name] = {
            "resolution": [int(resolution[0]), int(resolution[1])],
            "intrinsics": numeric,
        }
    return {
        "source": str(value.get("source") or "configured calibration"),
        "coordinate_frame": str(value.get("coordinate_frame") or "robot_base"),
        "cameras": normalized_cameras,
    }


def validate_driver_config(payload: Any) -> dict[str, Any]:
    """Validate and normalize the ``driver_config.json`` object.

    ``robot.controller.load_driver_config`` only guarantees that the file holds
    a JSON object, so field-level truth has to live here.
    """
    raw = _require_mapping(payload)
    missing = [key for key in _REQUIRED_CONFIG_FIELDS if raw.get(key) in (None, "")]
    if missing:
        raise ConfigValidationError(
            "driver config is missing required fields: " + ", ".join(missing)
        )

    required_instruction = normalize_instruction(raw["required_instruction"])
    if not required_instruction:
        raise ConfigValidationError("required_instruction must not be blank")
    expected_task = str(raw["expected_task"]).strip()
    if not expected_task:
        raise ConfigValidationError("expected_task must not be blank")

    expected_instance_id = raw.get("expected_instance_id")
    if expected_instance_id is not None:
        if isinstance(expected_instance_id, bool) or not isinstance(expected_instance_id, int):
            raise ConfigValidationError("expected_instance_id must be an integer or null")

    navigation_only = raw.get("navigation_only", False)
    if not isinstance(navigation_only, bool):
        raise ConfigValidationError("navigation_only must be a boolean")

    listen_host = str(raw.get("listen_host", "127.0.0.1")).strip() or "127.0.0.1"
    return {
        "workspace": str(raw["workspace"]),
        "upstream_url": str(raw["upstream_url"]),
        "listen_host": listen_host,
        "listen_port": int(raw.get("listen_port", 8000)),
        "upstream_open_timeout_s": _positive_float(raw, "upstream_open_timeout_s", 10.0),
        "required_instruction": required_instruction,
        "max_action_steps": _positive_int(raw, "max_action_steps", 4000),
        "status_path": raw.get("status_path"),
        "profile_path": raw.get("profile_path"),
        "observation_publish_interval": _positive_int(
            raw, "observation_publish_interval", 5
        ),
        "camera_calibration": _camera_calibration(raw.get("camera_calibration")),
        "r1pro_urdf_path": raw.get("r1pro_urdf_path"),
        "primitive_max_steps": _positive_int(raw, "primitive_max_steps", 240),
        "execution_timeout_s": _positive_float(raw, "execution_timeout_s", 1800.0),
        "progress_timeout_s": _positive_float(raw, "progress_timeout_s", 300.0),
        "state_sync_timeout_s": _positive_float(raw, "state_sync_timeout_s", 120.0),
        "expected_task": expected_task,
        "expected_instance_id": expected_instance_id,
        "navigation_only": navigation_only,
    }


# --------------------------------------------------------------------------- #
# Official evaluator status contract
# --------------------------------------------------------------------------- #


def validate_evaluation(
    payload: Any,
    *,
    expected_task: str | None = None,
    expected_instance_id: int | None = None,
) -> dict[str, Any]:
    """Validate one ``BEHAVIOR_STATUS.json`` object.

    Only hard inconsistencies raise.  Unknown extra fields are preserved so a
    newer evaluator can add information without breaking the driver.
    """
    if not isinstance(payload, dict):
        raise EvaluationValidationError("BEHAVIOR_STATUS.json must hold a JSON object")

    task = payload.get("task")
    if expected_task and task not in (None, "") and str(task) != str(expected_task):
        raise EvaluationValidationError(
            f"evaluator is running task {task!r}, expected {expected_task!r}"
        )

    instance_id = payload.get("instance_id")
    if (
        expected_instance_id is not None
        and instance_id is not None
        and int(instance_id) != int(expected_instance_id)
    ):
        raise EvaluationValidationError(
            f"evaluator is running instance {instance_id}, "
            f"expected {expected_instance_id}"
        )

    phase = str(payload.get("phase") or "").strip().lower()
    if phase and phase not in EVALUATION_PHASES:
        raise EvaluationValidationError(f"unknown evaluator phase {phase!r}")

    for key in ("success", "done", "terminated", "truncated"):
        value = payload.get(key)
        if value is not None and not isinstance(value, bool):
            raise EvaluationValidationError(f"evaluator field {key} must be a boolean")
    steps = payload.get("steps")
    if steps is not None and (
        isinstance(steps, bool) or not isinstance(steps, int) or steps < 0
    ):
        raise EvaluationValidationError(
            "evaluator field steps must be a non-negative integer or null"
        )
    return payload


def episode_terminated(status: dict[str, Any] | None) -> bool:
    """True once the official evaluator has published a final state."""
    if not isinstance(status, dict):
        return False
    return bool(status.get("done"))


def episode_succeeded(status: dict[str, Any] | None) -> bool:
    """True only for an official success, never for a lifecycle event."""
    if not episode_terminated(status):
        return False
    if str((status or {}).get("phase") or "").strip().lower() == "error":
        return False
    return bool((status or {}).get("success"))


def classify_episode(
    status: dict[str, Any] | None,
    *,
    evaluator_returncode: int | None = None,
    infrastructure_error: str | None = None,
) -> str:
    """Reduce an episode to one verdict.

    Returns one of ``success``, ``task_failed``, ``evaluator_error``,
    ``infrastructure_error`` or ``incomplete``.
    """
    if infrastructure_error:
        return "infrastructure_error"
    if not isinstance(status, dict) or not status:
        return "incomplete"
    phase = str(status.get("phase") or "").strip().lower()
    if phase == "error":
        return "evaluator_error"
    if not episode_terminated(status):
        if evaluator_returncode not in (None, 0):
            return "infrastructure_error"
        return "incomplete"
    if episode_succeeded(status):
        return "success"
    return "task_failed"


def agent_exit_code(
    *,
    verdict: str,
    success: bool,
    require_success: bool,
) -> int:
    """Map an episode verdict to the runner's process exit code.

    Infrastructure and evaluator failures are always non-zero, including the
    old ``done=true`` + ``phase=error`` trap.
    """
    if verdict in {"infrastructure_error", "evaluator_error", "incomplete"}:
        return 1
    if success:
        return 0
    return 1 if require_success else 0

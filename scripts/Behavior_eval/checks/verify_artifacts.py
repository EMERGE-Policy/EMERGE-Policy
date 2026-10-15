#!/usr/bin/env python3
"""Verify that one BEHAVIOR episode output directory satisfies acceptance.

This module only reads artifacts, so it runs without a GPU, Isaac Sim or a
language model.  It exists because the runners' exit codes alone are not
evidence: an episode can end cleanly and still not have turned the radio on.

Usage::

    python -m scripts.Behavior_eval.checks.verify_artifacts \
        --output-dir <episode output dir> --require-success
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from robot.behavior import contracts
except ImportError:  # Direct script execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import contracts  # type: ignore[no-redef]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def verify(
    output_dir: Path,
    *,
    require_success: bool = False,
    require_agent_result: bool = False,
) -> list[tuple[bool, str]]:
    """Return ``(ok, message)`` for every acceptance check."""
    output_dir = Path(output_dir).expanduser().resolve()
    checks: list[tuple[bool, str]] = []

    def check(ok: bool, message: str) -> bool:
        checks.append((bool(ok), message))
        return bool(ok)

    status_path = output_dir / "workspace/BEHAVIOR_STATUS.json"
    status = _read_json(status_path)
    if not check(status is not None, f"official status readable: {status_path}"):
        return checks

    assert status is not None
    phase = str(status.get("phase") or "").strip().lower()
    check(
        phase in contracts.EVALUATION_PHASES,
        f"official phase is a known value (phase={phase!r})",
    )
    check(phase != "error", f"the evaluator did not fail (phase={phase!r})")
    check(bool(status.get("done")), "the episode reached a final state (done=true)")
    task_name = str(status.get("task") or "").strip()
    check(bool(task_name), f"the evaluator recorded a task name (task={task_name!r})")

    success = bool(status.get("success"))
    if require_success:
        check(success, f"official success is true (success={success})")

    result_path_raw = status.get("result_path")
    if check(bool(result_path_raw), "the official result path is recorded"):
        result_path = Path(str(result_path_raw)).expanduser().resolve()
        inside = output_dir == result_path.parent or output_dir in result_path.parents
        check(inside, f"the official result lives in this output directory ({result_path})")
        official = _read_json(result_path)
        if check(official is not None, f"official result JSON readable: {result_path}"):
            assert official is not None
            check(
                bool(official.get("success")) == success,
                "the official result agrees with the live status on success",
            )
            for key in ("task", "instance_id", "rollout_id"):
                check(
                    str(official.get(key)) == str(status.get(key)),
                    f"the official result agrees with the live status on {key}",
                )

    action_path = output_dir / "workspace/ACTION.md"
    if action_path.is_file():
        from Emerge.utils.action_queue import (  # noqa: PLC0415
            normalize_action_document,
            parse_action_markdown,
        )

        payload = parse_action_markdown(action_path.read_text(encoding="utf-8").strip() or "{}")
        document = normalize_action_document(payload) if payload else None
        if document is not None:
            unfinished = [
                action["id"]
                for action in document.get("actions", [])
                if str(action.get("status")) in {"pending", "running"}
            ]
            check(not unfinished, f"no action was left pending or running ({unfinished})")

    manifest_path = output_dir / "workspace/artifacts/observations/observation.json"
    manifest = _read_json(manifest_path)
    if check(manifest is not None, f"camera manifest readable: {manifest_path}"):
        assert manifest is not None
        views = manifest.get("views") or []
        check(bool(views), "the camera manifest lists at least one view")
        workspace = output_dir / "workspace"
        missing = [
            str(view.get("image_path"))
            for view in views
            if not (workspace / str(view.get("image_path"))).is_file()
        ]
        check(not missing, f"every referenced camera image exists ({missing})")
        revision = manifest.get("revision")
        mismatched = [
            view.get("image_path")
            for view in views
            if revision is not None
            and f".{int(revision):06d}.png" not in str(view.get("image_path"))
        ]
        check(not mismatched, "every referenced image belongs to the published revision")
        if manifest.get("schema_version") == "Emerge.behavior_camera_observation.v3":
            check(bool(manifest.get("rgbd")), "the current observation contains aligned RGB-D")
            missing_depth = [
                str(view.get("depth_path"))
                for view in views
                if not (workspace / str(view.get("depth_path"))).is_file()
            ]
            check(not missing_depth, f"every referenced linear-depth array exists ({missing_depth})")
            mismatched_depth = [
                view.get("depth_path")
                for view in views
                if revision is not None
                and f".{int(revision):06d}.npy" not in str(view.get("depth_path"))
            ]
            check(
                not mismatched_depth,
                "every referenced depth array belongs to the published revision",
            )

    agent_path = output_dir / "agent_episode_result.json"
    agent = _read_json(agent_path)
    if agent is None:
        check(not require_agent_result, f"agent episode result present: {agent_path}")
    else:
        check(
            bool(agent.get("success")) == success,
            "the agent episode result agrees with the official status on success",
        )
        check(
            str(agent.get("task") or "") == task_name,
            "the agent episode result is for the same task",
        )
        check(
            not agent.get("infrastructure_error"),
            f"no infrastructure error was recorded ({agent.get('infrastructure_error')!r})",
        )

    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--require-success",
        action="store_true",
        help="Fail unless the official evaluator reported success.",
    )
    parser.add_argument(
        "--require-agent-result",
        action="store_true",
        help="Fail when the episode has no agent_episode_result.json.",
    )
    args = parser.parse_args(argv)

    checks = verify(
        args.output_dir,
        require_success=args.require_success,
        require_agent_result=args.require_agent_result,
    )
    failures = 0
    for ok, message in checks:
        mark = "ok  " if ok else "FAIL"
        print(f"[{mark}] {message}")
        if not ok:
            failures += 1
    if not checks:
        print("[FAIL] no acceptance check could be evaluated")
        return 1
    print("PASS" if failures == 0 else f"FAIL ({failures} check(s) failed)")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

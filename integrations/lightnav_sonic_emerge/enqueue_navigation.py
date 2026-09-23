"""Queue one ``navigate_vln`` action without invoking the EMERGE LLM agent."""

from __future__ import annotations

import argparse
from pathlib import Path

from Emerge.utils.action_queue import (
    append_action,
    pending_action_type,
    update_action_document,
)


def enqueue(
    workspace: str | Path,
    *,
    instruction: str,
    max_steps: int,
    control_hz: float,
) -> str:
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"EMERGE workspace not found: {root}")
    if not instruction.strip():
        raise ValueError("instruction must be non-empty")
    if max_steps < 1:
        raise ValueError("max_steps must be at least one")
    if control_hz <= 0:
        raise ValueError("control_hz must be positive")

    def append(document):
        active = pending_action_type(document)
        if active:
            raise RuntimeError(
                f"action {active!r} is still pending/running; wait for it to finish"
            )
        return append_action(
            document,
            action_type="navigate_vln",
            parameters={
                "instruction": instruction.strip(),
                "max_steps": max_steps,
                "control_hz": control_hz,
            },
        )

    result = update_action_document(root / "ACTION.md", append)
    return str(result["actions"][-1]["id"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="~/.Emerge/workspace")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--control-hz", type=float, default=4.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    action_id = enqueue(
        args.workspace,
        instruction=args.instruction,
        max_steps=args.max_steps,
        control_hz=args.control_hz,
    )
    print(f"queued navigate_vln action: {action_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

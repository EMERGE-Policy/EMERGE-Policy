"""Standalone terminal entrypoint for the Task Verification Subagent."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from Emerge.config.loader import get_config_path, load_config
from Emerge.providers.factory import create_provider
from Emerge.subagents import SubagentResult, SubagentTask, TextContent
from Emerge.subagents.task_verification.agent import (
    TaskVerificationSubagent,
)
from Emerge.subagents.task_verification.register import (
    build_task_verification_subagent,
)


_EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Talk directly to the standalone Task Verification Subagent"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=get_config_path(),
        help="Emerge config file",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Workspace containing artifacts/observations/observation.json",
    )
    parser.add_argument(
        "--model",
        help="Multimodal model override for this Task Verification Subagent",
    )
    parser.add_argument(
        "--task",
        help="Run one verification and exit instead of opening a prompt",
    )
    return parser


async def run(args: argparse.Namespace) -> None:
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    verification_config = config.subagents.task_verification
    model = (
        args.model
        or verification_config.model
        or config.agents.defaults.model
    )
    workspace = (
        args.workspace.expanduser().resolve()
        if args.workspace
        else config.workspace_path.expanduser().resolve()
    )
    provider = create_provider(config, model=model)
    agent = build_task_verification_subagent(
        provider=provider,
        workspace=workspace,
        model=model,
        config=verification_config.model_dump(),
    )

    print(
        f"Task Verification Subagent ready | model={model} | "
        f"workspace={workspace}"
    )
    if args.task:
        _print_result(await _run_task(agent, args.task))
        return

    print("Each request uses a fresh context. Type 'exit' to quit.")
    while True:
        try:
            text = (await asyncio.to_thread(input, "task_verification> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text.lower() in _EXIT_COMMANDS:
            return
        _print_result(await _run_task(agent, text))


async def _run_task(
    agent: TaskVerificationSubagent,
    text: str,
) -> SubagentResult:
    return await agent.run(SubagentTask(content=(TextContent(text),)))


def _print_result(result: SubagentResult) -> None:
    if result.error:
        print(f"Error: {result.error}")
        return
    print(f"\n{result.summary}")
    print(json.dumps(result.output, ensure_ascii=False, indent=2))
    print()


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

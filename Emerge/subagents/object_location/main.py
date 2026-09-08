"""Standalone terminal entrypoint for the Object Location Subagent."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Sequence

from Emerge.config.loader import get_config_path, load_config
from Emerge.providers.factory import create_provider
from Emerge.subagents import SubagentResult, SubagentTask, TextContent
from Emerge.subagents.object_location.agent import ObjectLocationSubagent
from Emerge.subagents.object_location.register import (
    build_object_location_subagent,
)


_EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Talk directly to the standalone Object Location Subagent"
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
        help="Multimodal model override for this Object Location Subagent",
    )
    parser.add_argument(
        "--task",
        help="Run one task and exit instead of opening an interactive prompt",
    )
    return parser


async def run(args: argparse.Namespace) -> None:
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    object_location_config = config.subagents.object_location
    model = (
        args.model
        or object_location_config.model
        or config.agents.defaults.model
    )
    workspace = (
        args.workspace.expanduser().resolve()
        if args.workspace
        else config.workspace_path.expanduser().resolve()
    )
    provider = create_provider(config, model=model)
    agent = build_object_location_subagent(
        provider=provider,
        workspace=workspace,
        model=model,
        config=object_location_config.model_dump(),
    )

    print(
        f"Object Location Subagent ready | model={model} | "
        f"workspace={workspace}"
    )
    if args.task:
        result = await _run_task(agent, args.task)
        _print_result(result, diagnostics=agent.last_diagnostics)
        return

    print("Each request uses a fresh context. Type 'exit' to quit.")
    while True:
        try:
            text = (await asyncio.to_thread(input, "object_location> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not text:
            continue
        if text.lower() in _EXIT_COMMANDS:
            return
        result = await _run_task(agent, text)
        _print_result(result, diagnostics=agent.last_diagnostics)


async def _run_task(
    agent: ObjectLocationSubagent,
    text: str,
) -> SubagentResult:
    task = SubagentTask(content=(TextContent(text),))
    return await agent.run(task)


def _print_result(
    result: SubagentResult,
    *,
    diagnostics: dict | None,
) -> None:
    if result.error:
        print(f"Error: {result.error}")
        return
    print(f"\n{result.summary}")
    if diagnostics is not None:
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print()


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

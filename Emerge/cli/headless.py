"""Machine-only single-run entrypoint; imports no TUI or Rich.

stdout: exactly one RunResult JSON object.
stderr: diagnostics. Artifacts: request.json, events.jsonl, result.json, runtime.log.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import signal
import sys
from pathlib import Path

from Emerge.runtime.protocol import RunError, RunRequest, RunResult, utc_now


def parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(description=__doc__)
    app.add_argument("message", nargs="?")
    app.add_argument("--request", type=Path)
    app.add_argument("--config", "-c")
    app.add_argument("--workspace", "-w")
    app.add_argument("--session", "-s", dest="session_id")
    app.add_argument("--model")
    app.add_argument("--output-dir", type=Path)
    app.add_argument("--max-iterations", type=int)
    app.add_argument("--timeout-s", type=float)
    app.add_argument("--cancel-timeout-s", type=float)
    app.add_argument("--restrict-to-workspace", action="store_true", default=None)
    app.add_argument("--stream", action="store_true", default=None)
    return app


async def execute(request, output_dir, runtime=None):
    from Emerge.runtime.service import AgentRuntime
    cancel = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            previous = signal.getsignal(sig)
            loop.add_signal_handler(sig, cancel.set)
            installed[sig] = previous
        except (NotImplementedError, RuntimeError):
            pass
    try:
        return await (runtime or AgentRuntime()).run(request, output_dir=output_dir, cancel=cancel)
    finally:
        for sig, previous in installed.items():
            loop.remove_signal_handler(sig)
            signal.signal(sig, previous)


@contextlib.contextmanager
def diagnostics_to_stderr():
    """Keep native libraries/subprocesses as well as Python prints off stdout."""
    sys.stdout.flush()
    saved = os.dup(1)
    try:
        os.dup2(2, 1)
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    request = None
    started = utc_now()
    try:
        if args.request and args.message:
            raise ValueError("Use either a request file or a message")
        payload = json.loads(args.request.read_text(encoding="utf-8")) if args.request else {
            "message": args.message or "",
        }
        if not isinstance(payload, dict):
            raise ValueError("Request must be a JSON object")
        for key in ("config", "workspace", "session_id", "model", "max_iterations",
                    "timeout_s", "cancel_timeout_s", "restrict_to_workspace", "stream"):
            value = getattr(args, key)
            if value is not None:
                payload[key] = value
        request = RunRequest.model_validate(payload)
        # All runtime/import diagnostics are diverted from the JSON protocol.
        with diagnostics_to_stderr():
            from Emerge.runtime.configuration import load_runtime_config
            if args.output_dir:
                directory = args.output_dir.expanduser().resolve()
            else:
                config = load_runtime_config(request)
                directory = config.workspace_path / "runs" / request.run_id
            result = asyncio.run(execute(request, directory))
    except Exception as exc:
        result = RunResult(run_id=request.run_id if request else "invalid",
                           session_id=request.session_id if request else "invalid",
                           run_status="failed", finish_reason="runtime_error" if request else "invalid_request",
                           started_at=started, error=RunError(code=type(exc).__name__, message=str(exc)))
    print(result.model_dump_json())
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())

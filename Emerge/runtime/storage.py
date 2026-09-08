"""Run artifacts and workspace ownership; no presentation dependencies."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from Emerge.runtime.protocol import RunEvent, RunRequest, RunResult

_SECRET_KEYS = re.compile(r"api.?key|authorization|password|secret|access.?token|refresh.?token", re.I)


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: ("[redacted]" if _SECRET_KEYS.search(k) else redact(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(content)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class RunArtifacts:
    """Exclusive run directory: reruns must use a new run_id/output directory."""

    def __init__(self, directory: Path, request: RunRequest):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        atomic_json(directory / "request.json", redact(request.model_dump(mode="json")))
        self.stream = (directory / "events.jsonl").open("x", encoding="utf-8")

    def event(self, event: RunEvent) -> None:
        self.stream.write(json.dumps(redact(event.model_dump(mode="json")), ensure_ascii=False) + "\n")
        self.stream.flush()

    def finish(self, result: RunResult) -> None:
        atomic_json(self.directory / "result.json", redact(result.model_dump(mode="json")))

    def close(self) -> None:
        self.stream.close()


class WorkspaceLease:
    """One writer per robot workspace, including across independent CLI processes.

    An advisory OS lock is released automatically on process exit. The file itself
    is intentionally retained so concurrent processes always lock the same inode.
    """

    def __init__(self, workspace: Path):
        self.path = workspace / ".runtime.lock"
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                self.stream.write(b"0")
                self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            self.stream.close()
            self.stream = None
            raise RuntimeError("Workspace is already running another agent") from exc
        return self

    def __exit__(self, *args):
        if self.stream:
            self.stream.close()
            self.stream = None

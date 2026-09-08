"""Task workspace lifecycle, independent from presentation clients."""

import shutil
from pathlib import Path

from Emerge.runtime.storage import atomic_text
from Emerge.utils.action_queue import empty_action_document, update_action_document


RESET_TEXT_FILES = (
    Path("EMBODIED.md"),
    Path("ROBOT_STATE.md"),
    Path("PLAN.md"),
    Path("memory/MEMORY.md"),
)


def reset_workspace_context(workspace: Path) -> None:
    """Clear task state while keeping workspace structure and run history."""
    workspace.mkdir(parents=True, exist_ok=True)
    for relative_path in RESET_TEXT_FILES:
        atomic_text(workspace / relative_path, "")
    update_action_document(
        workspace / "ACTION.md",
        lambda _: empty_action_document(),
    )
    artifacts = workspace / "artifacts"
    if artifacts.exists():
        shutil.rmtree(artifacts)
    artifacts.mkdir()

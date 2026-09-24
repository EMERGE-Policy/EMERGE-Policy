"""Base interface implemented by every controller driver."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, TypedDict

CancelCheck = Callable[[], str | None]


class SceneEntry(TypedDict):
    id: str
    label: str  # Display path, e.g. "Kitchen/Task 1"; '/' creates browser folders.


class SceneCatalog(TypedDict):
    root: str  # Display name; does not have to be a filesystem path.
    current: str | None
    entries: list[SceneEntry]


class BaseDriver(ABC):
    """Contract shared by hardware and simulation drivers."""

    @abstractmethod
    def get_profile_path(self) -> Path:
        """Return the filesystem path to this driver's embodied profile."""

    @abstractmethod
    def load_environment(self) -> None:
        """Initialize the environment from the driver's configuration."""

    @abstractmethod
    def reset_environment(self) -> None:
        """Load a fresh initial state, including when called on a new driver.

        The controller closes the previous driver before clearing the workspace,
        then calls this method on a new instance built from the current config.
        Publish fresh observations before returning; raise on failure.
        """

    @abstractmethod
    def get_scene_catalog(self) -> SceneCatalog:
        """Return selectable scenes, even before loading or after closing.

        IDs must be unique and stable across driver instances. Labels are relative
        display paths with '/' separating folders. An empty catalog explicitly
        means no scene selection is available; current is None when not loaded.
        """

    @abstractmethod
    def switch_scene(self, scene_id: str) -> None:
        """Validate an ID from get_scene_catalog and load its initial environment.

        Must work on a new driver, without calling load_environment first.
        Reject unknown/unsupported scenes before changing the environment.
        Return only once state and observations are ready, or raise on failure.
        """

    @abstractmethod
    def execute_action(
        self,
        action_type: str,
        params: dict,
        *,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        """Execute one action and return its human-readable result."""

    @abstractmethod
    def get_runtime_state(self) -> dict[str, Any]:
        """Return runtime state exposed to the agent workspace."""

    @abstractmethod
    def close(self) -> None:
        """Stop all writes and release resources, including after a partial load.

        Must be safe to call before loading and more than once. The controller
        clears workspace artifacts only after this method returns successfully.
        """

    def __enter__(self) -> "BaseDriver":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

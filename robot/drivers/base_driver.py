"""Base interface implemented by every controller driver."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable


CancelCheck = Callable[[], str | None]


class BaseDriver(ABC):
    """Contract shared by hardware and simulation drivers."""

    @abstractmethod
    def get_profile_path(self) -> Path:
        """Return the filesystem path to this driver's embodied profile."""

    @abstractmethod
    def load_scene(self, scene: dict[str, dict]) -> None:
        """Initialize the driver from the workspace scene."""

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

    def close(self) -> None:
        """Release resources owned by the driver."""

    def __enter__(self) -> "BaseDriver":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

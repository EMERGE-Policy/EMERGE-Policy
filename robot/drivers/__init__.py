from __future__ import annotations

import importlib
from typing import Any

from robot.drivers.base_driver import BaseDriver
from robot.plugins import activate_external_driver, list_external_drivers, resolve_external_driver

# ── Registry ────────────────────────────────────────────────────────────────
# Format:  "short_name": "module_path.ClassName"

DRIVER_REGISTRY: dict[str, str] = {
    "libero_mujoco": "robot.drivers.mujoco_driver.LiberoMujocoDriver",
}


def load_driver(name: str, **kwargs: Any) -> BaseDriver:

    dotted = DRIVER_REGISTRY.get(name)
    if dotted is None:
        spec = resolve_external_driver(name)
        if spec is None:
            available = ", ".join(list_drivers())
            raise KeyError(
                f"Unknown driver {name!r}. Available drivers: {available}"
            )
        activate_external_driver(spec)
        dotted = spec.dotted_path
    module_path, class_name = dotted.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    if not (isinstance(cls, type) and issubclass(cls, BaseDriver)):
        raise TypeError(f"{dotted} is not a BaseDriver subclass")

    return cls(**kwargs)


def list_drivers() -> list[str]:
    """Return sorted list of registered driver names."""
    return sorted(set(DRIVER_REGISTRY) | set(list_external_drivers()))

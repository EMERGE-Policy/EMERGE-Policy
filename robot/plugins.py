"""Helpers for installing and resolving external controller driver plugins."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any
import tomllib


_PLUGIN_HOME_ENV = "Emerge_PLUGIN_HOME"
_DEFAULT_PLUGIN_HOME = Path.home() / ".Emerge" / "plugins"
_REGISTRY_FILENAME = "registry.json"
_MANIFEST_FILENAME = "Emerge_plugin.toml"


@dataclass(frozen=True)
class ExternalDriverSpec:
    """Resolved metadata for one external controller driver."""

    driver_name: str
    plugin_name: str
    repo_path: Path
    module_path: str
    class_name: str
    profile_path: Path
    sys_paths: tuple[Path, ...]
    version: str = ""
    source_url: str = ""
    ref: str = ""

    @property
    def dotted_path(self) -> str:
        return f"{self.module_path}.{self.class_name}"


def get_plugin_home() -> Path:
    raw = os.environ.get(_PLUGIN_HOME_ENV, "").strip()
    return Path(raw).expanduser().resolve() if raw else _DEFAULT_PLUGIN_HOME


def get_registry_path() -> Path:
    return get_plugin_home() / _REGISTRY_FILENAME


def get_plugin_repos_root() -> Path:
    return get_plugin_home() / "repos"


def get_plugin_repo_path(driver_name: str) -> Path:
    return get_plugin_repos_root() / driver_name


def load_plugin_registry() -> dict[str, Any]:
    path = get_registry_path()
    if not path.exists():
        return {"drivers": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid plugin registry format: {path}")
    drivers = data.get("drivers")
    if not isinstance(drivers, dict):
        data["drivers"] = {}
    return data


def save_plugin_registry(data: dict[str, Any]) -> Path:
    path = get_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def read_plugin_manifest(repo_path: Path | str) -> dict[str, Any]:
    root = Path(repo_path).expanduser().resolve()
    manifest_path = root / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"Plugin manifest not found: {manifest_path}")
    with manifest_path.open("rb") as fh:
        data = tomllib.load(fh)
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid plugin manifest: {manifest_path}")
    return data


def build_external_driver_spec(
    manifest: dict[str, Any],
    repo_path: Path | str,
    *,
    source_url: str = "",
    ref: str = "",
) -> ExternalDriverSpec:
    root = Path(repo_path).expanduser().resolve()
    plugin_meta = manifest.get("plugin") or {}
    driver_meta = manifest.get("driver") or {}
    python_meta = manifest.get("python") or {}

    driver_name = str(driver_meta.get("name", "")).strip()
    module_path = str(driver_meta.get("module", "")).strip()
    class_name = str(driver_meta.get("class", "")).strip()
    profile_rel = str(driver_meta.get("profile_path", "")).strip()
    plugin_name = str(plugin_meta.get("name", driver_name)).strip() or driver_name

    missing = [
        label
        for label, value in (
            ("driver.name", driver_name),
            ("driver.module", module_path),
            ("driver.class", class_name),
            ("driver.profile_path", profile_rel),
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"Plugin manifest is missing required fields: {', '.join(missing)}"
        )

    sys_paths_raw = python_meta.get("sys_paths", ["."])
    if not isinstance(sys_paths_raw, list) or not sys_paths_raw:
        sys_paths_raw = ["."]
    sys_paths = tuple((root / str(item)).resolve() for item in sys_paths_raw)

    return ExternalDriverSpec(
        driver_name=driver_name,
        plugin_name=plugin_name,
        repo_path=root,
        module_path=module_path,
        class_name=class_name,
        profile_path=(root / profile_rel).resolve(),
        sys_paths=sys_paths,
        version=str(plugin_meta.get("version", "")).strip(),
        source_url=source_url.strip(),
        ref=ref.strip(),
    )


def register_plugin(
    repo_path: Path | str,
    *,
    source_url: str = "",
    ref: str = "",
) -> ExternalDriverSpec:
    root = Path(repo_path).expanduser().resolve()
    manifest = read_plugin_manifest(root)
    spec = build_external_driver_spec(
        manifest,
        root,
        source_url=source_url,
        ref=ref,
    )

    registry = load_plugin_registry()
    registry.setdefault("drivers", {})
    registry["drivers"][spec.driver_name] = {
        "plugin_name": spec.plugin_name,
        "repo_path": str(spec.repo_path),
        "module_path": spec.module_path,
        "class_name": spec.class_name,
        "profile_path": str(spec.profile_path.relative_to(spec.repo_path)),
        "sys_paths": [str(path.relative_to(spec.repo_path)) for path in spec.sys_paths],
        "version": spec.version,
        "source_url": spec.source_url,
        "ref": spec.ref,
        "installed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }
    save_plugin_registry(registry)
    return spec


def unregister_plugin(driver_name: str) -> bool:
    registry = load_plugin_registry()
    drivers = registry.get("drivers", {})
    if driver_name not in drivers:
        return False
    del drivers[driver_name]
    save_plugin_registry(registry)
    return True


def list_external_drivers() -> list[str]:
    try:
        registry = load_plugin_registry()
    except Exception:
        return []
    drivers = registry.get("drivers", {})
    if not isinstance(drivers, dict):
        return []
    return sorted(str(name) for name in drivers)


def resolve_external_driver(driver_name: str) -> ExternalDriverSpec | None:
    registry = load_plugin_registry()
    raw = (registry.get("drivers") or {}).get(driver_name)
    if not isinstance(raw, dict):
        return None
    repo_path = Path(str(raw.get("repo_path", ""))).expanduser().resolve()
    profile_rel = str(raw.get("profile_path", "")).strip()
    sys_paths_raw = raw.get("sys_paths", ["."])
    if not isinstance(sys_paths_raw, list) or not sys_paths_raw:
        sys_paths_raw = ["."]
    return ExternalDriverSpec(
        driver_name=driver_name,
        plugin_name=str(raw.get("plugin_name", driver_name)).strip() or driver_name,
        repo_path=repo_path,
        module_path=str(raw.get("module_path", "")).strip(),
        class_name=str(raw.get("class_name", "")).strip(),
        profile_path=(repo_path / profile_rel).resolve(),
        sys_paths=tuple((repo_path / str(item)).resolve() for item in sys_paths_raw),
        version=str(raw.get("version", "")).strip(),
        source_url=str(raw.get("source_url", "")).strip(),
        ref=str(raw.get("ref", "")).strip(),
    )


def activate_external_driver(spec: ExternalDriverSpec) -> None:
    for path in reversed(spec.sys_paths or (spec.repo_path,)):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

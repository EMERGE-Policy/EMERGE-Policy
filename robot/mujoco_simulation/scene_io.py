"""Scene and ``ROBOT_STATE.md`` helpers for the LIBERO MuJoCo driver."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

_FENCE_OPEN = "```json"
_FENCE_CLOSE = "```"
_BLOCK_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


def _pose_utils():
    from robot.mujoco_simulation.pose_utils import PoseUtils

    return PoseUtils


def _load_json_block(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    content = path.read_text(encoding="utf-8")
    match = _BLOCK_RE.search(content)
    if not match:
        return {}
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def load_robot_state_doc(path: Path) -> dict[str, Any]:
    """Return the full robot-state document from ``ROBOT_STATE.md``."""
    return _load_json_block(path)


def default_robot_state_doc() -> dict[str, Any]:
    """Return the minimal structured robot-state document."""
    return {
        "schema_version": "Emerge.robot_state.v1",
        "scene_graph": {"nodes": [], "edges": []},
        "robots": {},
        "objects": {},
    }


def save_robot_state_doc(path: Path, robot_state: dict[str, Any]) -> None:
    """Write a full robot-state document to ``ROBOT_STATE.md``."""
    state_json = json.dumps(robot_state, indent=2, ensure_ascii=False)
    content = (
        "# Robot State\n\n"
        "Auto-updated by Controller after each action execution.\n"
        "Edit the JSON block below to set up or reset the test scene.\n\n"
        "Agent usage:\n"
        "- The robot state initially included in the agent context is only a snapshot.\n"
        "- After every `execute_robot_action` call, the agent MUST use `read_file` to re-read "
        "`ROBOT_STATE.md` before checking task progress or choosing the next action.\n"
        "- Do not rely on the earlier copy already present in the conversation context, because "
        "Controller may have updated the file after the action.\n\n"
        f"{_FENCE_OPEN}\n{state_json}\n{_FENCE_CLOSE}\n"
    )
    path.write_text(content, encoding="utf-8")


def merge_robot_state_doc(
    existing: dict[str, Any] | None,
    *,
    objects: dict[str, dict] | None = None,
    robots: dict[str, Any] | None = None,
    scene_graph: dict[str, Any] | None = None,
    map_data: dict[str, Any] | None = None,
    tf_data: dict[str, Any] | None = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Merge robot-state partitions while preserving unrelated sections."""
    base = default_robot_state_doc()
    if isinstance(existing, dict):
        base.update(existing)
    base.pop("updated_at", None)

    if objects is not None:
        merged_objects = dict(base.get("objects", {}))
        for name, payload in objects.items():
            if not isinstance(name, str) or not isinstance(payload, dict):
                continue
            merged_objects[name] = {
                **dict(merged_objects.get(name, {})),
                **payload,
            }
        base["objects"] = merged_objects
    if robots is not None:
        merged_robots = dict(base.get("robots", {}))
        merged_robots.update(robots)
        base["robots"] = merged_robots
    if scene_graph is not None:
        base["scene_graph"] = scene_graph
    if map_data is not None:
        base["map"] = map_data
    if tf_data is not None:
        base["tf"] = tf_data
    if updated_at is not None:
        base["updated_at"] = updated_at

    return base


def load_scene_from_md(path: Path) -> dict[str, dict]:
    """Extract only ``ROBOT_STATE.objects`` for driver ``load_scene(...)``."""
    document = load_robot_state_doc(path)
    objects = document.get("objects")
    return dict(objects) if isinstance(objects, dict) else {}


@dataclass(slots=True)
class ExternalAssetConfig:
    name: str
    path: Path
    kind: str
    position: list[float]
    orientation_quat: list[float]
    dynamic: bool


@dataclass(slots=True)
class NormalizedSceneConfig:
    external_assets: list[ExternalAssetConfig]


class SceneConfigParser:
    """Normalize structured and flat scene dictionaries for LIBERO runtime assets."""

    def parse(self, scene: dict[str, Any] | None) -> NormalizedSceneConfig:
        source = dict(scene or {})
        objects = source.get("objects")
        if not isinstance(objects, dict):
            reserved = {
                "robot",
                "urdf_assets",
                "mujoco_assets",
            }
            objects = {key: value for key, value in source.items() if key not in reserved}

        assets = self._parse_assets_from_objects(objects)
        assets.extend(self._parse_asset_group(source.get("urdf_assets"), kind="urdf"))
        assets.extend(self._parse_asset_group(source.get("mujoco_assets"), kind="mjcf"))
        return NormalizedSceneConfig(external_assets=assets)

    def _parse_asset_group(self, value: Any, *, kind: str) -> list[ExternalAssetConfig]:
        if value is None:
            return []
        if isinstance(value, dict):
            entries = []
            for name, payload in value.items():
                item = dict(payload) if isinstance(payload, dict) else {"path": payload}
                item.setdefault("name", name)
                entries.append(item)
        elif isinstance(value, list):
            entries = value
        else:
            raise ValueError(f"{kind}_assets must be a mapping or list")

        pose_utils = _pose_utils()
        assets: list[ExternalAssetConfig] = []
        for index, raw in enumerate(entries):
            if not isinstance(raw, dict):
                raise ValueError(f"{kind}_assets[{index}] must be an object")
            name = str(raw.get("name", f"{kind}_{index}")).strip()
            path_text = str(raw.get("path", raw.get("file", ""))).strip()
            if not name or not path_text:
                raise ValueError(f"{kind}_assets[{index}] requires name and path")
            orientation = pose_utils.resolve_orientation(raw)
            if orientation is None:
                orientation = [0.0, 0.0, 0.0, 1.0]
            assets.append(
                ExternalAssetConfig(
                    name=name,
                    path=Path(path_text).expanduser(),
                    kind=kind,
                    position=pose_utils.vector(
                        raw.get("position", raw.get("position_m", [0.0, 0.0, 0.0])),
                        name=f"{name}.position",
                    ).astype(float).tolist(),
                    orientation_quat=pose_utils.normalize_quaternion(orientation)
                    .astype(float)
                    .tolist(),
                    dynamic=bool(raw.get("dynamic", False)),
                )
            )
        return assets

    def _parse_assets_from_objects(
        self, objects: dict[str, Any]
    ) -> list[ExternalAssetConfig]:
        assets: list[ExternalAssetConfig] = []
        for name, raw in objects.items():
            if not isinstance(name, str) or not isinstance(raw, dict):
                continue
            kind = ""
            path: Any = None
            for candidate, keys in (
                ("urdf", ("urdf", "urdf_path")),
                ("mjcf", ("mjcf", "mjcf_path", "xml_path")),
            ):
                for key in keys:
                    if raw.get(key):
                        kind = candidate
                        path = raw[key]
                        break
                if path is not None:
                    break
            if path is None:
                continue
            asset_raw = {**raw, "name": name, "path": path}
            assets.extend(self._parse_asset_group([asset_raw], kind=kind))
        return assets

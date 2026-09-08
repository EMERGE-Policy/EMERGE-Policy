"""Scene graph query tool backed by ROBOT_STATE.md."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from Emerge.base import Tool
from robot.mujoco_simulation.scene_io import load_robot_state_doc


class SceneGraphQueryTool(Tool):
    """Read-only queries against the structured scene graph."""

    def __init__(self, workspace: Path):
        self.workspace = workspace

    @property
    def name(self) -> str:
        return "query_scene_graph"

    @property
    def description(self) -> str:
        return "Query the structured scene graph and robot navigation state."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["find_by_class", "find_by_id", "nearest_to_robot", "list_zones"],
                },
                "target_class": {"type": "string"},
                "target_id": {"type": "string"},
                "robot_id": {"type": "string"},
            },
            "required": ["query_type"],
        }

    async def execute(
        self,
        query_type: str,
        target_class: str | None = None,
        target_id: str | None = None,
        robot_id: str | None = None,
    ) -> str:
        robot_state = load_robot_state_doc(self.workspace / "ROBOT_STATE.md")
        scene_graph = robot_state.get("scene_graph", {})
        nodes = scene_graph.get("nodes", [])
        objects = robot_state.get("objects", {})

        if query_type == "find_by_class":
            matches = [node for node in nodes if node.get("class") == target_class]
            return json.dumps({"matches": matches}, ensure_ascii=False, indent=2)

        if query_type == "find_by_id":
            for node in nodes:
                if node.get("id") == target_id:
                    return json.dumps({"match": node}, ensure_ascii=False, indent=2)
            return json.dumps({"match": None}, ensure_ascii=False, indent=2)

        if query_type == "list_zones":
            map_data = robot_state.get("map", {})
            zones = map_data.get("zones", [])
            return json.dumps({"zones": zones}, ensure_ascii=False, indent=2)

        if query_type == "nearest_to_robot":
            if not robot_id:
                return "Error: robot_id is required for nearest_to_robot."
            pose = (((robot_state.get("robots") or {}).get(robot_id) or {}).get("robot_pose") or {})
            if "x" not in pose or "y" not in pose:
                return f"Error: robot pose unavailable for '{robot_id}'."

            annotated = []
            for node in nodes:
                center = self._resolve_node_center(node, objects)
                if "x" not in center or "y" not in center:
                    continue
                dist = math.hypot(center["x"] - pose["x"], center["y"] - pose["y"])
                enriched = dict(node)
                enriched["distance_to_robot"] = round(dist, 4)
                annotated.append(enriched)
            annotated.sort(key=lambda item: item["distance_to_robot"])
            return json.dumps({"matches": annotated}, ensure_ascii=False, indent=2)

        return f"Error: Unsupported query_type '{query_type}'."

    @staticmethod
    def _resolve_node_center(
        node: dict[str, Any],
        objects: dict[str, Any],
    ) -> dict[str, Any]:
        object_key = node.get("object_key")
        if isinstance(object_key, str):
            obj = objects.get(object_key)
            if isinstance(obj, dict):
                pose = obj.get("pose")
                if isinstance(pose, dict):
                    position = pose.get("position")
                    if isinstance(position, dict):
                        return position
                position = obj.get("position")
                if isinstance(position, dict):
                    return position
        center = node.get("center")
        return center if isinstance(center, dict) else {}

"""Read-only adapters for workspace documents and service health."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

_JSON_BLOCK = re.compile(r"```json\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def plan_snapshot(workspace: Path) -> dict:
    path = workspace / "PLAN.md"
    if not path.exists():
        return {"mission": "", "main_line": [], "branch_stack": [], "pointer": 1}
    try:
        from Emerge.agent.tools.update_plan import UpdatePlanTool
        return UpdatePlanTool._parse(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError) as exc:
        return {"mission": "", "main_line": [], "error": str(exc)}


def _document(path: Path, empty_data: dict | None = None) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.strip() and empty_data is not None:
        return empty_data
    match = _JSON_BLOCK.search(text)
    payload = json.loads(match.group(1) if match else text)
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload


def _snapshot_file(path: Path, empty_data: dict | None = None) -> dict:
    try:
        modified = path.stat().st_mtime
        data = _document(path, empty_data)
        return {"data": data, "age_s": max(0, time.time() - modified), "error": None}
    except FileNotFoundError:
        return {"data": {}, "age_s": None, "error": None}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"data": {}, "age_s": None, "error": str(exc)}


def _observation_summary(workspace: Path, snapshot: dict) -> dict:
    if snapshot["error"]:
        return {"status": "invalid"}
    data = snapshot["data"]
    if snapshot["age_s"] is None:
        return {"status": "missing"}
    views = data.get("views")
    reference = data.get("reference_view")
    if not isinstance(views, list) or not views or not isinstance(reference, str):
        return {"status": "invalid"}
    names = {str(view.get("name")) for view in views if isinstance(view, dict)}
    missing_images = []
    for view in views:
        if not isinstance(view, dict) or not isinstance(view.get("image_path"), str):
            missing_images.append("<invalid view>")
            continue
        if not (workspace / view["image_path"]).is_file():
            missing_images.append(view["image_path"])
    status = (
        "invalid" if reference not in names
        else "incomplete" if missing_images
        else "ready"
    )
    return {
        "status": status,
        "revision": data.get("revision"),
        "reference_view": reference,
        "view_count": len(views),
        "missing_images": missing_images,
    }


def workspace_snapshot(workspace: Path) -> dict:
    snapshot = {"plan": plan_snapshot(workspace)}
    for name, relative in (
        ("robot", "ROBOT_STATE.md"),
        ("actions", "ACTION.md"),
        ("observation", "artifacts/observations/observation.json"),
    ):
        empty_data = {"actions": []} if name == "actions" else None
        snapshot[name] = _snapshot_file(workspace / relative, empty_data)
    snapshot["observation"]["summary"] = _observation_summary(
        workspace, snapshot["observation"],
    )
    return snapshot


async def service_health(config) -> list[dict]:
    """Discover self-described services on the configured local port range."""
    from external_model_server.model_service.discovery import ServiceDiscovery

    result = await ServiceDiscovery(config.model_services.discovery_config()).scan()
    rows = []
    for item in result.services:
        rows.append({
            "name": item.description["service"], "url": item.endpoint,
            "status": item.description["status"],
            "instance_id": item.description["instance_id"],
            "model_id": item.description["model_id"], "latency_ms": item.latency_ms,
            "detail": item.description.get("detail", ""),
        })
    if result.incomplete:
        rows.append({"name": "Scan", "status": "incomplete", "error": "Discovery deadline exceeded"})
    if not rows:
        rows.append({"name": "Services", "status": "not_found"})
    return rows

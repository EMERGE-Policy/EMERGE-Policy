"""Merge the bundled LightNav ProcTHOR room into a derived SONIC G1 scene.

Only a new output MJCF and one G1 include sidecar are written.  The original
SONIC scene and LightNav's ``mujoco_demo`` assets remain read-only.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

from .room_scene import _materialize_direct_includes


# ``mujoco_demo/vln_mujoco/model.py`` spawns its TurtleBot here.  Anchoring the
# environment at the negative of this point puts that same navigable location
# at SONIC G1's canonical world origin.
_DEMO_SPAWN_XY = (6.5, 13.8)


def _freeze_environment_joints(root: ET.Element) -> None:
    """Turn all ProcTHOR articulation bodies into static room geometry."""

    for parent in root.iter():
        for child in tuple(parent):
            if child.tag in {"joint", "freejoint"}:
                parent.remove(child)


def _absolute_environment_file_paths(root: ET.Element, scene: Path) -> None:
    """Resolve bundled ProcTHOR meshes and textures without copying assets."""

    assets_root = scene.parents[2].resolve()
    for element in root.iter():
        raw_path = element.get("file")
        if not raw_path:
            continue
        source = (scene.parent / raw_path).resolve()
        try:
            source.relative_to(assets_root)
        except ValueError as exc:
            raise ValueError(f"ProcTHOR asset escapes bundled asset root: {raw_path}") from exc
        if not source.is_file():
            raise FileNotFoundError(f"bundled ProcTHOR asset is missing: {source}")
        element.set("file", str(source))


def _append_environment_defaults(g1_root: ET.Element, room_root: ET.Element) -> None:
    room_default = room_root.find("default")
    if room_default is None:
        return
    g1_default = g1_root.find("default")
    if g1_default is None:
        g1_default = ET.Element("default")
        g1_root.insert(0, g1_default)
    for child in room_default:
        g1_default.append(deepcopy(child))


def _append_environment_assets(g1_root: ET.Element, room_root: ET.Element) -> None:
    room_asset = room_root.find("asset")
    if room_asset is None:
        raise ValueError("ProcTHOR scene has no <asset> block")
    g1_asset = g1_root.find("asset")
    if g1_asset is None:
        g1_asset = ET.Element("asset")
        g1_root.insert(0, g1_asset)
    existing_names = {child.get("name") for child in g1_asset if child.get("name")}
    duplicates = [
        child.get("name")
        for child in room_asset
        if child.get("name") in existing_names
    ]
    if duplicates:
        preview = ", ".join(name for name in duplicates[:5] if name is not None)
        raise ValueError(f"SONIC and ProcTHOR asset names collide: {preview}")
    for child in room_asset:
        g1_asset.append(deepcopy(child))


def _append_environment_worldbody(g1_root: ET.Element, room_root: ET.Element) -> None:
    room_worldbody = room_root.find("worldbody")
    g1_worldbody = g1_root.find("worldbody")
    if room_worldbody is None or g1_worldbody is None:
        raise ValueError("both SONIC and ProcTHOR scenes must contain <worldbody>")

    # The SONIC root scene supplies a checkerboard floor with the generic name
    # ``floor``.  Remove it in the derivative so the ProcTHOR structural floor
    # is the sole support surface.
    for child in tuple(g1_worldbody):
        if child.tag == "geom" and child.get("name") == "floor":
            g1_worldbody.remove(child)

    anchor = ET.SubElement(
        g1_worldbody,
        "body",
        {
            "name": "lightnav_procthor_anchor",
            "pos": f"{-_DEMO_SPAWN_XY[0]} {-_DEMO_SPAWN_XY[1]} 0",
        },
    )
    for child in room_worldbody:
        anchor.append(deepcopy(child))


def _append_environment_contacts(g1_root: ET.Element, room_root: ET.Element) -> None:
    room_contact = room_root.find("contact")
    if room_contact is None:
        return
    g1_contact = g1_root.find("contact")
    if g1_contact is None:
        g1_contact = ET.SubElement(g1_root, "contact")
    for child in room_contact:
        g1_contact.append(deepcopy(child))


def _copy_contact_capacity(g1_root: ET.Element, room_root: ET.Element) -> None:
    """Retain ProcTHOR's conservative contact capacity without changing G1 config."""

    if g1_root.find("size") is not None:
        return
    room_size = room_root.find("size")
    if room_size is not None:
        g1_root.insert(0, deepcopy(room_size))


def compose_procthor_g1_scene(
    g1_base_scene: Path,
    procthor_scene: Path,
    output_scene: Path,
) -> Path:
    """Write a static ProcTHOR ``val_2`` room around SONIC's unmodified G1."""

    g1_base_scene = g1_base_scene.resolve()
    procthor_scene = procthor_scene.resolve()
    output_scene = output_scene.resolve()
    if not g1_base_scene.is_file():
        raise FileNotFoundError(f"SONIC base scene does not exist: {g1_base_scene}")
    if not procthor_scene.is_file():
        raise FileNotFoundError(f"ProcTHOR scene does not exist: {procthor_scene}")

    g1_tree = ET.parse(g1_base_scene)
    g1_root = g1_tree.getroot()
    room_tree = ET.parse(procthor_scene)
    room_root = room_tree.getroot()
    if g1_root.tag != "mujoco" or room_root.tag != "mujoco":
        raise ValueError("both input files must have a <mujoco> root")

    output_scene.parent.mkdir(parents=True, exist_ok=True)
    _materialize_direct_includes(g1_root, g1_base_scene, output_scene.parent)
    _freeze_environment_joints(room_root)
    _absolute_environment_file_paths(room_root, procthor_scene)
    _append_environment_defaults(g1_root, room_root)
    _append_environment_assets(g1_root, room_root)
    _append_environment_worldbody(g1_root, room_root)
    _append_environment_contacts(g1_root, room_root)
    _copy_contact_capacity(g1_root, room_root)

    g1_root.set("model", f"{g1_root.get('model', 'g1')} lightnav_procthor")
    ET.indent(g1_tree, space="  ")
    g1_tree.write(output_scene, encoding="utf-8", xml_declaration=True)
    return output_scene

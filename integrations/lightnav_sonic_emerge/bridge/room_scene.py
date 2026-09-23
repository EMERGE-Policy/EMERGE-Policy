"""Compose the small furnished navigation room without HumanClawAgent.

The result is a derivative MJCF.  The canonical SONIC G1 scene and every
upstream asset remain read-only; only the requested output path is written.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET


def _add_geom(
    worldbody: ET.Element,
    *,
    name: str,
    geom_type: str,
    pos: str,
    size: str,
    material: str,
) -> None:
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": name,
            "type": geom_type,
            "pos": pos,
            "size": size,
            "material": material,
            "contype": "1",
            "conaffinity": "1",
            "friction": "0.9 0.01 0.001",
        },
    )


def _add_materials(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(0, asset)
    colors = {
        "lightnav_wall": "0.82 0.81 0.76 1",
        "lightnav_wood": "0.38 0.20 0.08 1",
        "lightnav_fabric": "0.12 0.29 0.50 1",
        "lightnav_target": "0.85 0.06 0.04 1",
        "lightnav_cabinet": "0.52 0.52 0.49 1",
    }
    existing = {material.get("name") for material in asset.findall("material")}
    for name, rgba in colors.items():
        if name not in existing:
            ET.SubElement(asset, "material", {"name": name, "rgba": rgba})


def _materialize_direct_includes(root: ET.Element, base_scene: Path, output_dir: Path) -> None:
    """Write sidecars that preserve the upstream model's relative mesh paths."""

    for include in root.findall("include"):
        raw_path = include.get("file")
        if raw_path is None:
            continue
        source = Path(raw_path)
        if not source.is_absolute():
            source = (base_scene.parent / source).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"included SONIC model does not exist: {source}")
        include_tree = ET.parse(source)
        compiler = include_tree.getroot().find("compiler")
        if compiler is not None:
            meshdir = compiler.get("meshdir")
            if meshdir is not None and not Path(meshdir).is_absolute():
                compiler.set("meshdir", str((source.parent / meshdir).resolve()))
        sidecar = output_dir / f"_lightnav_{source.name}"
        ET.indent(include_tree, space="  ")
        include_tree.write(sidecar, encoding="utf-8", xml_declaration=True)
        include.set("file", sidecar.name)


def add_lightnav_ego_camera(output_dir: Path) -> None:
    """Add a level, forward-looking RGB camera to the derived G1 sidecar.

    SONIC's built-in ``head_camera`` is retained unchanged for upstream use,
    but its pose is angled down and sideways for its original use case.  This
    additional camera is created only in the bridge-owned include sidecar and
    is selected by ``room_sim`` as LightNav's RGB source.
    """

    sidecar = output_dir / "_lightnav_g1_29dof_with_hand.xml"
    if not sidecar.is_file():
        raise FileNotFoundError(f"derived G1 include sidecar is missing: {sidecar}")
    tree = ET.parse(sidecar)
    root = tree.getroot()
    torso = root.find(".//body[@name='torso_link']")
    if torso is None:
        raise ValueError("derived G1 include has no torso_link for LightNav camera")
    for camera in tuple(torso.findall("camera")):
        if camera.get("name") == "lightnav_ego_camera":
            torso.remove(camera)
    # MuJoCo's camera looks down its local -Z axis.  The explicit xy axes
    # below make it look along robot +X, with world +Z up, matching the
    # forward convention used by the adapter.
    ET.SubElement(
        torso,
        "camera",
        {
            "name": "lightnav_ego_camera",
            "pos": "0.10 0 0.45",
            "xyaxes": "0 -1 0 0 0 1",
            "fovy": "80",
        },
    )
    ET.indent(tree, space="  ")
    tree.write(sidecar, encoding="utf-8", xml_declaration=True)


def compose_lightnav_room_scene(base_scene: Path, output_scene: Path) -> Path:
    """Create a 6 m × 6 m room with furniture and a north-wall door opening."""

    base_scene = base_scene.resolve()
    output_scene = output_scene.resolve()
    if not base_scene.is_file():
        raise FileNotFoundError(f"SONIC base scene does not exist: {base_scene}")

    tree = ET.parse(base_scene)
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError(f"expected an MJCF <mujoco> root, got <{root.tag}>")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("SONIC base scene has no <worldbody>")
    if root.find(".//geom[@name='lightnav_room_south']") is not None:
        raise ValueError("base scene already contains a LightNav-Sonic room")

    output_scene.parent.mkdir(parents=True, exist_ok=True)
    _materialize_direct_includes(root, base_scene, output_scene.parent)
    _add_materials(root)

    # A 1.2 m doorway lies in the north wall (+y), centred on x=0.  Furniture
    # keeps the canonical G1 spawn at (0, 0) clear while providing visible
    # landmark geometry for visual-navigation experiments.
    _add_geom(worldbody, name="lightnav_room_south", geom_type="box", pos="0 -3 1.5", size="3 0.08 1.5", material="lightnav_wall")
    _add_geom(worldbody, name="lightnav_room_west", geom_type="box", pos="-3 0 1.5", size="0.08 3 1.5", material="lightnav_wall")
    _add_geom(worldbody, name="lightnav_room_east", geom_type="box", pos="3 0 1.5", size="0.08 3 1.5", material="lightnav_wall")
    _add_geom(worldbody, name="lightnav_room_north_left", geom_type="box", pos="-1.8 3 1.5", size="1.2 0.08 1.5", material="lightnav_wall")
    _add_geom(worldbody, name="lightnav_room_north_right", geom_type="box", pos="1.8 3 1.5", size="1.2 0.08 1.5", material="lightnav_wall")

    _add_geom(worldbody, name="lightnav_table_top", geom_type="box", pos="0.65 1.45 0.75", size="0.75 0.48 0.06", material="lightnav_wood")
    _add_geom(worldbody, name="lightnav_table_base", geom_type="box", pos="0.65 1.45 0.36", size="0.12 0.12 0.36", material="lightnav_wood")
    _add_geom(worldbody, name="lightnav_sofa_base", geom_type="box", pos="1.85 -1.15 0.36", size="0.65 0.42 0.36", material="lightnav_fabric")
    _add_geom(worldbody, name="lightnav_sofa_back", geom_type="box", pos="2.35 -1.15 0.82", size="0.10 0.42 0.46", material="lightnav_fabric")
    _add_geom(worldbody, name="lightnav_chair", geom_type="box", pos="-1.35 0.65 0.42", size="0.27 0.27 0.42", material="lightnav_wood")
    _add_geom(worldbody, name="lightnav_cabinet", geom_type="box", pos="-1.85 -1.55 0.65", size="0.42 0.28 0.65", material="lightnav_cabinet")
    _add_geom(worldbody, name="lightnav_red_target", geom_type="sphere", pos="-1.85 -1.55 1.36", size="0.12", material="lightnav_target")

    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "lightnav_overview",
            "pos": "0 0 8",
            "xyaxes": "1 0 0 0 1 0",
        },
    )
    root.set("model", f"{root.get('model', 'g1')} lightnav_sonic_room")
    ET.indent(tree, space="  ")
    tree.write(output_scene, encoding="utf-8", xml_declaration=True)
    return output_scene

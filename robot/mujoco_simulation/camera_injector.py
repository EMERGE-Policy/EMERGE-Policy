"""Inject world-frame MuJoCo cameras into a compiled LIBERO XML model."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any


class CameraInjector:
    """Apply configured world-frame cameras onto an existing MuJoCo XML string."""

    def merge_cameras(
        self,
        base_xml: str,
        cameras: list[dict[str, Any]],
    ) -> str:
        if not cameras:
            return base_xml
        try:
            root = ET.fromstring(base_xml)
        except ET.ParseError as exc:
            raise ValueError(f"failed to parse MuJoCo XML for camera injection: {exc}") from exc
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError("MuJoCo XML is missing <worldbody>; cannot inject cameras")

        for camera in cameras:
            self._upsert_camera(root, worldbody, camera)
        return ET.tostring(root, encoding="utf8").decode("utf8")

    def _upsert_camera(
        self,
        root: ET.Element,
        worldbody: ET.Element,
        camera: dict[str, Any],
    ) -> None:
        name = str(camera["name"]).strip()
        target = None
        for node in root.findall(".//camera"):
            if str(node.get("name", "")).strip() == name:
                target = node
                break
        if target is None:
            target = ET.SubElement(worldbody, "camera")
        target.set("name", name)
        target.set("mode", "fixed")
        target.set("pos", self._format_vector(camera["pos"]))
        target.set("quat", self._format_vector(camera["quat"]))
        if "fovy" in camera:
            target.set("fovy", self._format_scalar(camera["fovy"]))

    def _format_vector(self, value: Any) -> str:
        return " ".join(self._format_scalar(item) for item in value)

    def _format_scalar(self, value: Any) -> str:
        return format(float(value), ".15g")

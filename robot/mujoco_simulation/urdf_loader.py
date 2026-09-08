"""MuJoCo MjSpec based URDF and MJCF attachment support."""

from __future__ import annotations

from typing import Any

import numpy as np

from robot.mujoco_simulation.scene_io import ExternalAssetConfig


class UrdfLoader:
    """Attach external robot-description assets to a compiled LIBERO XML model."""

    def merge_assets(
        self,
        base_xml: str,
        assets: list[ExternalAssetConfig],
    ) -> str:
        if not assets:
            return base_xml
        mujoco = self._import_mujoco()
        try:
            parent_spec = mujoco.MjSpec.from_string(base_xml)
        except Exception as exc:
            raise ValueError(f"failed to parse LIBERO model XML: {exc}") from exc

        for asset in assets:
            self._attach_asset(parent_spec, asset, mujoco)
        try:
            parent_spec.compile()
            return parent_spec.to_xml()
        except Exception as exc:
            raise ValueError(f"merged MuJoCo model failed to compile: {exc}") from exc

    def _attach_asset(self, parent_spec: Any, asset: ExternalAssetConfig, mujoco: Any) -> None:
        path = asset.path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"external {asset.kind} asset not found: {path}")
        if asset.kind == "urdf" and path.suffix.lower() != ".urdf":
            raise ValueError(f"URDF asset must end in .urdf: {path}")
        try:
            child_spec = mujoco.MjSpec.from_file(str(path))
        except Exception as exc:
            raise ValueError(f"failed to parse {asset.kind} asset {path}: {exc}") from exc

        body_names = [
            str(body.name)
            for body in child_spec.bodies
            if str(body.name).strip() and body.parent is not None
        ]
        if not body_names:
            raise ValueError(f"external asset has no attachable body: {path}")
        root_body = self._find_root_body(child_spec, body_names[0])
        if asset.dynamic and not list(root_body.joints):
            root_body.add_freejoint()

        prefix = f"{asset.name}_"
        try:
            frame = parent_spec.worldbody.add_frame()
            frame.name = f"{asset.name}_mount"
            frame.pos = np.asarray(asset.position, dtype=np.float64)
            xyzw = np.asarray(asset.orientation_quat, dtype=np.float64)
            frame.quat = np.asarray([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])
            parent_spec.attach(child_spec, prefix=prefix, frame=frame)
        except Exception as exc:
            raise ValueError(f"failed to attach external asset {path}: {exc}") from exc

    def _find_root_body(self, spec: Any, fallback_name: str) -> Any:
        for body in spec.worldbody.bodies:
            return body
        for body in spec.bodies:
            if str(body.name) == fallback_name:
                return body
        raise ValueError("external asset root body could not be resolved")

    def _import_mujoco(self) -> Any:
        try:
            import mujoco
        except ImportError as exc:
            raise RuntimeError("MuJoCo is required to load URDF or MJCF assets") from exc
        return mujoco

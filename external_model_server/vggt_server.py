"""Persistent VGGT inference server."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Hashable, Sequence

import numpy as np

from external_model_server.model_service.contracts import ServiceDescriptor
from external_model_server.model_service.runtime import (
    ModelServerRuntime,
    add_runtime_arguments,
    runtime_arguments,
)
from external_model_server.schemas import VGGT

logger = logging.getLogger(__name__)
_PATCH_SIZE = 14


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return parsed


class VGGTInferenceService:
    """Own one VGGT backend and expose calibrated multi-view inference."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        backend: Any | None = None,
    ) -> None:
        self.backend = backend
        self.config = dict(config)

    @property
    def descriptor(self) -> ServiceDescriptor:
        return ServiceDescriptor(VGGT.service, self.config.get("model_id") or Path(self.config["model_path"]).name,
                                 VGGT.input_schema, VGGT.output_schema, ("infer", "batch"))

    def load(self) -> None:
        if self.backend is None:
            from external_model_server.vggt_backend import VGGTBackend
            self.backend = VGGTBackend(self.config)
        self.backend._load_model(str(self.config.get("device", "cuda")).strip().lower())

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.infer_batch([request])[0]

    def batch_key(self, request: dict[str, Any]) -> Hashable:
        views, _reference_view = self._parse_request(request)
        return (
            len(views),
            tuple(int(value) for value in np.asarray(views[0].rgb).shape),
        )

    def infer_batch(
        self,
        requests: Sequence[dict[str, Any]],
    ) -> Sequence[dict[str, Any]]:
        parsed = [self._parse_request(request) for request in requests]
        results = self.backend.infer_batch(parsed)
        if len(results) != len(parsed):
            raise RuntimeError("VGGT backend returned a mismatched number of results")
        return [_serialize_vggt_result(result) for result in results]

    @staticmethod
    def _parse_request(
        request: dict[str, Any],
    ) -> tuple[list[Any], str]:
        views_payload = request.get("views")
        if not isinstance(views_payload, list) or not views_payload:
            raise ValueError("VGGT request requires a non-empty 'views' list")

        from external_model_server.localization_types import VGGTViewInput

        views: list[VGGTViewInput] = []
        image_shapes: set[tuple[int, ...]] = set()
        for item in views_payload:
            if not isinstance(item, dict):
                raise TypeError("Each VGGT view must be a mapping")
            name = str(item["name"])
            rgb = np.asarray(item["rgb"], dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(
                    f"view {name!r} RGB must have shape HxWx3, got {rgb.shape}"
                )
            height, width = (int(rgb.shape[0]), int(rgb.shape[1]))
            if height % _PATCH_SIZE != 0 or width % _PATCH_SIZE != 0:
                raise ValueError(
                    "VGGT requires every RGB view height and width to be divisible "
                    f"by patch size {_PATCH_SIZE}; view {name!r} has "
                    f"height={height}, width={width}."
                )
            image_shapes.add(tuple(int(value) for value in rgb.shape))
            views.append(
                VGGTViewInput(
                    name=name,
                    rgb=rgb,
                    intrinsics=np.asarray(item["intrinsics"], dtype=np.float64),
                    T_world_camera=np.asarray(
                        item["T_world_camera"],
                        dtype=np.float64,
                    ),
                )
            )

        if len(image_shapes) != 1:
            raise ValueError(
                f"all VGGT RGB views must share one shape, got {sorted(image_shapes)}"
            )

        reference_view = str(request.get("reference_view", "")).strip()
        return views, reference_view


def _serialize_vggt_result(result: Any) -> dict[str, Any]:
    return {
        "reference_view": result.reference_view,
        "alignment": result.alignment.to_dict(),
        "views": [
            {
                "name": view.name,
                "rgb": view.rgb,
                "point_map_world": view.point_map_world,
                "depth_m": view.depth_m,
                "depth_conf": view.depth_conf,
                "point_conf": view.point_conf,
            }
            for view in result.views
        ],
        "cameras": [
            {
                "name": camera.name,
                "intrinsics_pred": camera.intrinsics_pred,
                "intrinsics_observed": camera.intrinsics_observed,
                "extrinsics_pred": camera.extrinsics_pred,
                "T_world_camera_observed": camera.T_world_camera_observed,
            }
            for camera in result.cameras
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VGGT websocket inference server")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--max-baseline-scale-relative-mad", type=float, default=0.25)
    parser.add_argument("--max-camera-center-rms-error-m", type=float, default=0.1)
    parser.add_argument("--max-batch-size", type=_positive_int, default=1)
    parser.add_argument("--batch-wait-ms", type=_non_negative_float, default=0)
    add_runtime_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    args = build_arg_parser().parse_args(argv)
    config = {
        "model_path": args.model_path,
        "model_id": args.model_id,
        "device": args.device,
        "precision": args.precision,
        "max_baseline_scale_relative_mad": args.max_baseline_scale_relative_mad,
        "max_camera_center_rms_error_m": args.max_camera_center_rms_error_m,
    }
    service = VGGTInferenceService(config)
    ModelServerRuntime(
        service,
        host=args.host,
        port=args.port,
        max_batch_size=args.max_batch_size,
        batch_wait_ms=args.batch_wait_ms,
        **runtime_arguments(args),
    ).serve_forever()


if __name__ == "__main__":
    main()

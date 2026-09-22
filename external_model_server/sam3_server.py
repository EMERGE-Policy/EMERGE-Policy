"""Persistent SAM3 image-segmentation server."""

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
from external_model_server.schemas import SAM3


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


class SAM3InferenceService:
    """Own one SAM3 backend and segment prompted targets in multiple views."""

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
        return ServiceDescriptor(SAM3.service, self.config.get("model_id") or Path(self.config["model_path"]).name,
                                 SAM3.input_schema, SAM3.output_schema, ("infer", "batch"))

    def load(self) -> None:
        if self.backend is None:
            from external_model_server.sam3_backend import SAM3Backend
            self.backend = SAM3Backend(self.config)
        self.backend._load_processor(
            device=str(self.config.get("device", "cuda")).strip().lower(),
            confidence_threshold=float(self.config.get("confidence_threshold", 0.5)),
            resolution=int(self.config.get("resolution", 1008)),
        )

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.infer_batch([request])[0]

    def batch_key(self, request: dict[str, Any]) -> Hashable:
        views, object_keys, _prompts, max_instances = self._parse_request(request)
        return (
            tuple(tuple(int(value) for value in rgb.shape) for rgb, _name in views),
            len(object_keys),
            max_instances,
        )

    def infer_batch(
        self,
        requests: Sequence[dict[str, Any]],
    ) -> Sequence[dict[str, Any]]:
        parsed = [self._parse_request(request) for request in requests]
        batched_views = [
            (rgb, view_name, object_keys, prompts, max_instances)
            for views, object_keys, prompts, max_instances in parsed
            for rgb, view_name in views
        ]
        results = self.backend.segment_views_batched(batched_views)
        if len(results) != len(batched_views):
            raise RuntimeError("SAM3 backend returned a mismatched number of view results")

        response: list[dict[str, Any]] = []
        result_offset = 0
        for views, _object_keys, _prompts, _max_instances in parsed:
            view_count = len(views)
            response.append(
                {
                    "views": [
                        _serialize_sam3_result(result)
                        for result in results[result_offset : result_offset + view_count]
                    ]
                }
            )
            result_offset += view_count
        return response

    @staticmethod
    def _parse_request(
        request: dict[str, Any],
    ) -> tuple[list[tuple[np.ndarray, str]], list[str], list[str], int]:
        views = request.get("views")
        targets = request.get("targets")
        if not isinstance(views, list) or not views:
            raise ValueError("SAM3 request requires a non-empty 'views' list")
        if not isinstance(targets, list) or not targets:
            raise ValueError("SAM3 request requires a non-empty 'targets' list")

        object_keys: list[str] = []
        prompts: list[str] = []
        max_instances = int(request.get("max_instances", 1))
        for target in targets:
            if not isinstance(target, dict):
                raise TypeError("Each SAM3 target must be a mapping")
            object_keys.append(str(target["object_key"]))
            prompts.append(str(target["prompt"]))

        parsed_views: list[tuple[np.ndarray, str]] = []
        for view in views:
            if not isinstance(view, dict):
                raise TypeError("Each SAM3 view must be a mapping")
            parsed_views.append(
                (
                    np.asarray(view["rgb"], dtype=np.uint8),
                    str(view["name"]),
                )
            )
        return parsed_views, object_keys, prompts, max_instances


def _serialize_sam3_result(result: Any) -> dict[str, Any]:
    return {
        "view_name": result.view_name,
        "image_shape": list(result.image_shape),
        "targets": [
            {
                "object_key": target.object_key,
                "prompt": target.prompt,
                "found": target.found,
                "confidence": target.confidence,
                "mask": target.mask,
                "bbox_2d": target.bbox_2d,
                "area_pixels": target.area_pixels,
                "instance_index": target.instance_index,
            }
            for target in result.targets
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAM3 websocket inference server")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--warmup-prompt", default="object")
    parser.add_argument("--no-warmup", action="store_true")
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
        "confidence_threshold": args.confidence_threshold,
        "resolution": args.resolution,
        "warmup": not args.no_warmup,
        "warmup_prompt": args.warmup_prompt,
    }
    service = SAM3InferenceService(config)
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

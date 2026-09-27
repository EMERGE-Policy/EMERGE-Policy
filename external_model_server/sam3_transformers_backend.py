"""Persistent SAM3 backend for a local Transformers checkpoint directory."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from external_model_server.localization_types import TargetMask, TargetMaskResult


class TransformersSAM3Backend:
    """Load one local SAM3 model once and reuse it for all prompted images."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None
        self._device: Any | None = None

    def preload(self) -> None:
        self._load()

    def segment_view(
        self,
        rgb: np.ndarray,
        view_name: str,
        target_objects: list[str],
        *,
        prompts: list[str],
        max_instances: int = 1,
    ) -> TargetMaskResult:
        return self.segment_views_batched(
            [(rgb, view_name, target_objects, prompts, max_instances)]
        )[0]

    def segment_views_batched(
        self,
        views: Sequence[tuple[np.ndarray, str, list[str], list[str], int]],
    ) -> list[TargetMaskResult]:
        if not views:
            raise ValueError("SAM3 requires at least one view")
        self._load()
        assert self._processor is not None
        assert self._model is not None
        assert self._torch is not None

        results: list[TargetMaskResult] = []
        for rgb, view_name, object_keys, prompts, max_instances in views:
            image_array = np.asarray(rgb, dtype=np.uint8)
            if image_array.ndim != 3 or image_array.shape[2] != 3:
                raise ValueError(
                    f"Transformers SAM3 expects HxWx3 RGB, got {image_array.shape!r}"
                )
            if not object_keys or len(object_keys) != len(prompts):
                raise ValueError("SAM3 requires one prompt per non-empty target list")
            targets: list[TargetMask] = []
            image = Image.fromarray(image_array, mode="RGB")
            for object_key, prompt in zip(object_keys, prompts, strict=True):
                inputs = self._processor(
                    images=image,
                    text=prompt,
                    return_tensors="pt",
                ).to(self._device)
                with self._torch.inference_mode():
                    outputs = self._model(**inputs)
                processed = self._processor.post_process_instance_segmentation(
                    outputs,
                    threshold=float(self.config.get("confidence_threshold", 0.5)),
                    mask_threshold=float(self.config.get("mask_threshold", 0.5)),
                    target_sizes=[image_array.shape[:2]],
                )
                targets.extend(
                    _targets_from_result(
                        object_key=object_key,
                        prompt=prompt,
                        result=processed[0] if processed else {},
                        max_instances=max_instances,
                    )
                )
            results.append(
                TargetMaskResult(
                    view_name=view_name,
                    image_shape=image_array.shape[:2],
                    targets=targets,
                )
            )
        return results

    def _load(self) -> None:
        if self._model is not None:
            return
        model_path = Path(str(self.config.get("model_path", ""))).expanduser().resolve()
        if not (model_path / "config.json").is_file():
            raise FileNotFoundError(
                "Transformers SAM3 model_path must be a local checkpoint directory "
                f"containing config.json: {model_path}"
            )
        try:
            import torch
            from transformers import Sam3Model, Sam3Processor
        except ImportError as exc:
            raise RuntimeError(
                "Transformers SAM3 requires torch and a transformers build with Sam3Model"
            ) from exc
        requested = str(self.config.get("device", "cuda")).strip().lower()
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Transformers SAM3 requested CUDA but CUDA is unavailable")
        device = torch.device(requested)
        processor = Sam3Processor.from_pretrained(model_path, local_files_only=True)
        model = Sam3Model.from_pretrained(model_path, local_files_only=True).to(device)
        model.eval()
        self._torch = torch
        self._device = device
        self._processor = processor
        self._model = model


def _targets_from_result(
    *,
    object_key: str,
    prompt: str,
    result: dict[str, Any],
    max_instances: int,
) -> list[TargetMask]:
    if max_instances < 1:
        raise ValueError("max_instances must be positive")
    scores = result.get("scores")
    masks = result.get("masks")
    boxes = result.get("boxes")
    if scores is None or masks is None or len(scores) == 0:
        return [_missing_target(object_key, prompt)]

    scores_np = _to_numpy(scores).reshape(-1)
    masks_np = _to_numpy(masks)
    boxes_np = _to_numpy(boxes) if boxes is not None else None
    ranked = np.argsort(scores_np)[::-1][:max_instances]
    targets = []
    for rank, source_index in enumerate(ranked, start=1):
        mask = np.asarray(masks_np[source_index]).squeeze() > 0.5
        confidence = float(scores_np[source_index])
        area = int(mask.sum())
        if area <= 0:
            targets.append(
                TargetMask(
                    object_key,
                    prompt,
                    False,
                    confidence,
                    None,
                    None,
                    0,
                    rank,
                )
            )
            continue
        if boxes_np is not None and len(boxes_np) > source_index:
            bbox = [
                int(round(float(value)))
                for value in np.asarray(boxes_np[source_index]).reshape(-1)[:4]
            ]
        else:
            rows, cols = np.where(mask)
            bbox = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
        targets.append(
            TargetMask(
                object_key,
                prompt,
                True,
                confidence,
                mask,
                bbox,
                area,
                rank,
            )
        )
    return targets or [_missing_target(object_key, prompt)]


def _missing_target(object_key: str, prompt: str) -> TargetMask:
    return TargetMask(object_key, prompt, False, 0.0, None, None, 0, 1)


def _to_numpy(value: Any) -> np.ndarray:
    candidate = value.detach() if hasattr(value, "detach") else value
    candidate = candidate.cpu() if hasattr(candidate, "cpu") else candidate
    return np.asarray(candidate)

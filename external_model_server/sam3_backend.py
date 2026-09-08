"""Thin adapter around the official SAM3 image model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from sam3.model.utils.misc import copy_data_to_device
from sam3.model_builder import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint, FindQueryLoaded, Image as SAMImage, InferenceMetadata
from sam3.train.transforms.basic_for_api import ComposeAPI, NormalizeAPI, RandomResizeAPI, ToTensorAPI

from external_model_server.localization_types import TargetMask, TargetMaskResult

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_safetensors_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    """Load a direct SAM3 image-model state dict with strict key validation."""
    try:
        from safetensors.torch import load_model
    except ImportError as exc:
        raise RuntimeError(
            "Loading a SAM3 .safetensors checkpoint requires the 'safetensors' package"
        ) from exc

    try:
        load_model(
            model,
            checkpoint_path,
            strict=True,
            device="cpu",
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"SAM3 safetensors checkpoint does not match the image model: {checkpoint_path}"
        ) from exc


def _build_sam3_model(*, checkpoint_path: Path, device: str) -> torch.nn.Module:
    """Build SAM3 from either an official PyTorch checkpoint or a direct safetensors state dict."""
    if checkpoint_path.suffix.lower() != ".safetensors":
        return build_sam3_image_model(
            device=device,
            checkpoint_path=str(checkpoint_path),
            load_from_HF=False,
        )

    # The official builder expects a PyTorch checkpoint containing detector.*
    # keys. Fine-tuned safetensors checkpoints contain the image model's direct
    # state dict, so load them before moving the model to CUDA.
    model = build_sam3_image_model(
        device="cpu",
        checkpoint_path=None,
        load_from_HF=False,
    )
    _load_safetensors_checkpoint(model, checkpoint_path)
    return model.to(device=device)


class SAM3Backend:
    """Load SAM3, run image segmentation, and normalize outputs."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self._model: Any | None = None
        self._device: str | None = None
        self._resolution: int | None = None
        self._warmed_up = False
        self._batched_model: Any | None = None
        self._transform: Any | None = None
        self._postprocessor: Any | None = None

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
        """Segment multiple prompted views in one SAM3 collated model call."""
        if not views:
            raise ValueError("SAM3Backend.segment_views_batched requires at least one view")
        for rgb, _view_name, target_objects, prompts, _max_instances in views:
            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise ValueError(
                    f"SAM3Backend expects RGB image with shape HxWx3, got {rgb.shape!r}"
                )
            if not target_objects:
                raise ValueError(
                    "SAM3Backend.segment_views_batched requires non-empty target_objects"
                )
            if len(prompts) != len(target_objects):
                raise ValueError(
                    "SAM3Backend.segment_views_batched requires one prompt per target object"
                )

        device = str(self.config.get("device", "cuda")).strip().lower()
        if device != "cuda":
            raise RuntimeError(f"SAM3Backend currently requires device='cuda', got {device!r}")
        if not torch.cuda.is_available():
            raise RuntimeError("SAM3Backend requires CUDA but torch.cuda.is_available() is False")

        confidence_threshold = float(self.config.get("confidence_threshold", 0.5))

        resolution = int(self.config.get("resolution", 1008))

        self._load_processor(
            device=device,
            confidence_threshold=confidence_threshold,
            resolution=resolution,
        )
        batch_inputs = [
            (
                object_keys,
                prompts,
                Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB"),
                max_instances,
            )
            for rgb, _view_name, object_keys, prompts, max_instances in views
        ]
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                targets_per_view = self._segment_many_targets_batched(batch_inputs)

        return [
            TargetMaskResult(
                view_name=view_name,
                image_shape=tuple(int(value) for value in rgb.shape[:2]),
                targets=targets,
            )
            for (rgb, view_name, _object_keys, _prompts, _max_instances), targets in zip(
                views,
                targets_per_view,
                strict=True,
            )
        ]

    def save_result(
        self,
        result: TargetMaskResult,
        rgb: np.ndarray,
        output_dir: str | Path,
    ) -> dict[str, Any]:
        output_dir = Path(output_dir).expanduser()
        if not output_dir.is_absolute():
            output_dir = (_REPO_ROOT / output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        rgb = np.asarray(rgb, dtype=np.uint8)
        targets = []
        for target in result.targets:
            entry = {
                "object_key": target.object_key,
                "prompt": target.prompt,
                "found": target.found,
                "confidence": target.confidence,
                "area_pixels": target.area_pixels,
                "bbox_2d": target.bbox_2d,
            }
            if not target.found or target.mask is None:
                targets.append(entry)
                continue

            mask = np.asarray(target.mask, dtype=bool)
            mask_path = output_dir / f"{target.object_key}_mask.png"
            overlay_path = output_dir / f"{target.object_key}_mask_overlay.png"

            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)

            overlay = rgb.copy()
            overlay[mask] = (
                0.4 * overlay[mask] + 0.6 * np.array([255, 0, 0], dtype=np.float32)
            ).astype(np.uint8)
            Image.fromarray(overlay, mode="RGB").save(overlay_path)

            entry.update(
                {
                    "mask_path": mask_path.name,
                    "overlay_path": overlay_path.name,
                }
            )
            targets.append(entry)

        summary_path = output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "view_name": result.view_name,
                    "image_shape": list(result.image_shape),
                    "targets": targets,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "output_dir": str(output_dir),
            "summary_path": str(summary_path),
            "targets": targets,
        }

    def _segment_targets_batched(
        self,
        *,
        object_keys: list[str],
        prompts: list[str],
        raw_image: Image.Image,
        max_instances: int = 1,
    ) -> list[TargetMask]:
        return self._segment_many_targets_batched(
            [(object_keys, prompts, raw_image, max_instances)]
        )[0]

    def _segment_many_targets_batched(
        self,
        inputs: Sequence[tuple[list[str], list[str], Image.Image, int]],
    ) -> list[list[TargetMask]]:
        model = self._batched_model
        transform = self._transform
        postprocessor = self._postprocessor
        device = self._device
        if model is None or transform is None or postprocessor is None or device is None:
            raise RuntimeError("SAM3 batched inference components are not initialized")

        datapoints: list[Datapoint] = []
        query_offsets: list[int] = []
        query_offset = 0
        for object_keys, prompts, raw_image, _max_instances in inputs:
            datapoint = Datapoint(find_queries=[], images=[])
            width, height = raw_image.size
            datapoint.images = [
                SAMImage(data=raw_image, objects=[], size=[height, width])
            ]
            query_offsets.append(query_offset)
            for prompt_idx, prompt in enumerate(prompts):
                query_id = query_offset + prompt_idx
                datapoint.find_queries.append(
                    FindQueryLoaded(
                        query_text=prompt,
                        image_id=0,
                        object_ids_output=[],
                        is_exhaustive=True,
                        query_processing_order=0,
                        inference_metadata=InferenceMetadata(
                            coco_image_id=query_id,
                            original_image_id=query_id,
                            original_category_id=1,
                            original_size=[height, width],
                            object_id=0,
                            frame_index=0,
                        ),
                    )
                )
            query_offset += len(object_keys)
            datapoints.append(transform(datapoint))

        batch = collate_fn_api(datapoints, dict_key="sam3_batch")["sam3_batch"]
        batch = copy_data_to_device(batch, torch.device(device), non_blocking=True)
        outputs = model(batch)
        processed = postprocessor.process_results(outputs, batch.find_metadatas)
        return [
            self._build_target_masks_from_processed(
                object_keys=object_keys,
                prompts=prompts,
                processed={
                    prompt_idx: processed[query_offset + prompt_idx]
                    for prompt_idx in range(len(object_keys))
                    if query_offset + prompt_idx in processed
                },
                max_instances=max_instances,
            )
            for (object_keys, prompts, _raw_image, max_instances), query_offset in zip(
                inputs,
                query_offsets,
                strict=True,
            )
        ]

    def _build_target_masks_from_processed(
        self,
        *,
        object_keys: list[str],
        prompts: list[str],
        processed: dict[int, dict[str, Any]],
        max_instances: int = 1,
    ) -> list[TargetMask]:
        # Keep the best few detections from the existing model output. This does
        # not run another forward pass; it only preserves masks that were
        # previously discarded by argmax.
        collected: list[list[tuple[float, torch.Tensor, torch.Tensor]]] = []
        for prompt_idx in range(len(object_keys)):
            result = processed.get(prompt_idx)
            if result is None:
                collected.append([])
                continue

            scores = result.get("scores")
            boxes = result.get("boxes")
            masks = result.get("masks")
            if scores is None or boxes is None or masks is None or len(scores) == 0:
                collected.append([])
                continue

            count = min(max_instances, len(scores))
            indices = torch.topk(scores, k=count, largest=True, sorted=True).indices
            instances = []
            for index in indices.tolist():
                mask = masks[index]
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask[0]
                instances.append(
                    (float(scores[index].item()), mask, boxes[index])
                )
            collected.append(instances)

        # --- Phase 2: batch transfer GPU -> CPU ---
        flat_instances = [item for instances in collected for item in instances]
        gpu_masks = [item[1] for item in flat_instances]
        gpu_boxes = [item[2] for item in flat_instances]

        if gpu_masks:
            # Stack and transfer in one operation to reduce sync overhead
            masks_cpu = torch.stack(gpu_masks).detach().cpu().numpy()
            boxes_cpu = torch.stack(gpu_boxes).detach().cpu().numpy()
        else:
            masks_cpu = np.empty((0,), dtype=bool)
            boxes_cpu = np.empty((0, 4), dtype=np.float32)

        # --- Phase 3: assemble TargetMask results ---
        targets: list[TargetMask] = []
        transfer_idx = 0
        for prompt_idx, (object_key, instances) in enumerate(
            zip(object_keys, collected, strict=True)
        ):
            if not instances:
                targets.append(
                    TargetMask(
                        object_key=object_key,
                        prompt=prompts[prompt_idx],
                        found=False,
                        confidence=0.0,
                        mask=None,
                        bbox_2d=None,
                        area_pixels=0,
                        instance_index=1,
                    )
                )
                continue

            for instance_index, item in enumerate(instances, start=1):
                score = item[0]
                mask_np = masks_cpu[transfer_idx].astype(bool, copy=False)
                box_np = boxes_cpu[transfer_idx]
                transfer_idx += 1

                area_pixels = int(mask_np.sum())
                if area_pixels <= 0:
                    targets.append(
                        TargetMask(
                            object_key=object_key,
                            prompt=prompts[prompt_idx],
                            found=False,
                            confidence=score,
                            mask=None,
                            bbox_2d=None,
                            area_pixels=0,
                            instance_index=instance_index,
                        )
                    )
                    continue

                bbox_2d = [int(round(float(v))) for v in box_np.tolist()]
                targets.append(
                    TargetMask(
                        object_key=object_key,
                        prompt=prompts[prompt_idx],
                        found=True,
                        confidence=score,
                        mask=mask_np,
                        bbox_2d=bbox_2d,
                        area_pixels=area_pixels,
                        instance_index=instance_index,
                    )
                )
        return targets

    def _load_processor(
        self,
        *,
        device: str,
        confidence_threshold: float,
        resolution: int,
    ) -> None:
        """Ensure model/transform/postprocessor are loaded. Lazy-loads on first call."""
        if (
            self._model is not None
            and self._device == device
            and self._resolution == resolution
        ):
            return

        model_path = Path(str(self.config["model_path"])).expanduser()
        if not model_path.is_absolute():
            model_path = (_REPO_ROOT / model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {model_path}")

        model = _build_sam3_model(checkpoint_path=model_path, device=device)
        self._model = model
        self._device = device
        self._resolution = resolution

        self._warmed_up = False
        self._batched_model = model
        self._transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=resolution,
                    max_size=resolution,
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self._postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=confidence_threshold,
            to_cpu=False,
        )
        if bool(self.config.get("warmup", True)):
            self._warmup_processor(resolution=resolution)

    def _warmup_processor(self, *, resolution: int) -> None:
        if self._warmed_up:
            return

        prompt = str(self.config.get("warmup_prompt", "object")).strip() or "object"
        warmup_image = Image.fromarray(
            np.zeros((resolution, resolution, 3), dtype=np.uint8),
            mode="RGB",
        )
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                self._segment_targets_batched(
                    object_keys=[prompt],
                    prompts=[prompt],
                    raw_image=warmup_image,
                )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._warmed_up = True

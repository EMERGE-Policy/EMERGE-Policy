"""Native multi-observation LIBERO inference for the pinned Cosmos model.

The upstream get_action helper tiles one observation. This adapter instead
builds one row per environment, using the same image transforms, latent frame
layout, normalization, and noise generator. Imports stay out of the client env.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def prepare_libero_batch(cfg, model, stats, utils, observations, embeddings):
    import torch

    if not observations or len(observations) != len(embeddings):
        raise ValueError("Expected one embedding per observation")
    if not (
        cfg.suite == "libero" and cfg.use_proprio and cfg.use_wrist_image
        and cfg.use_third_person_image and cfg.num_wrist_images == 1
        and cfg.num_third_person_images == 1
    ):
        raise ValueError("Native batching requires the LIBERO two-camera layout")
    device = model.tensor_kwargs["device"]
    videos, proprios = [], []
    repeat = utils.COSMOS_TEMPORAL_COMPRESSION_FACTOR
    for obs in observations:
        wrist, primary = utils.prepare_images_for_model(
            [obs["wrist_image"], obs["primary_image"]], cfg
        )
        blank = np.zeros_like(primary)
        # Latents: blank, proprio, wrist, primary, action, future proprio,
        # future wrist, future primary, value. The VAE compresses 1 + 4*T frames.
        frames = [blank, wrist, primary, blank, blank, wrist, primary, blank]
        video = np.concatenate(
            [blank[None]] + [np.repeat(frame[None], repeat, axis=0) for frame in frames]
        )
        videos.append(video.transpose(3, 0, 1, 2))
        proprio = obs["proprio"]
        if cfg.normalize_proprio:
            proprio = utils.rescale_proprio(proprio, stats)
        proprios.append(proprio)
    size = len(observations)
    image_size = utils.COSMOS_IMAGE_SIZE
    batch = {
        "dataset_name": "video_data",
        "video": torch.as_tensor(np.stack(videos), dtype=torch.uint8, device=device),
        "proprio": torch.as_tensor(np.stack(proprios), dtype=torch.bfloat16, device=device),
        "t5_text_embeddings": torch.as_tensor(
            np.concatenate(embeddings, axis=0), dtype=torch.bfloat16, device=device
        ),
        "fps": torch.full((size,), 16, dtype=torch.bfloat16, device=device),
        "padding_mask": torch.zeros(
            (size, 1, image_size, image_size), dtype=torch.bfloat16, device=device
        ),
        "num_conditional_frames": model.config.min_num_conditional_frames,
    }
    indices = {
        "current_proprio": 1, "current_wrist_image": 2, "current_image": 3,
        "action": 4, "future_proprio": 5, "future_wrist_image": 6,
        "future_image": 7, "value": 8,
        "current_wrist_image2": -1, "current_image2": -1,
        "future_wrist_image2": -1, "future_image2": -1,
    }
    for name, index in indices.items():
        batch[name + "_latent_idx"] = torch.full(
            (size,), index, dtype=torch.int64, device=device
        )
    return batch


def generate_candidates_batch(
    cfg, model, stats, utils, observations, embeddings,
    *, seeds: Sequence[int], index: int, score_mode: str, num_steps: int,
) -> list[dict[str, Any]]:
    import torch
    from cosmos_policy._src.imaginaire.utils.misc import arch_invariant_rand

    if score_mode not in {"none", "joint_value"} or cfg.use_variance_scale:
        raise ValueError("This configuration must use single-request inference")
    if len(seeds) != len(observations):
        raise ValueError("Expected one seed per observation")
    with torch.inference_mode():
        batch = prepare_libero_batch(cfg, model, stats, utils, observations, embeddings)
        _, _, frames, height, width = batch["video"].shape
        shape = (
            1, model.config.state_ch, model.tokenizer.get_latent_num_frames(frames),
            height // model.tokenizer.spatial_compression_factor,
            width // model.tokenizer.spatial_compression_factor,
        )
        # Each row uses exactly its own seed, independent of batch membership
        # and ordering. One shared RNG stream would change worker semantics.
        noise = torch.cat([
            arch_invariant_rand(shape, torch.float32, model.tensor_kwargs["device"], seed)
            for seed in seeds
        ]) * model.sde.sigma_max
        latent = model.generate_samples_from_batch(
            batch, n_sample=len(seeds), num_steps=num_steps, seed=seeds[0],
            x_sigma_max=noise, is_negative_prompt=False, use_variance_scale=False,
        )
        actions = utils.extract_action_chunk_from_latent_sequence(
            latent, action_shape=(cfg.chunk_size, utils.ACTION_DIM),
            action_indices=batch["action_latent_idx"],
        ).float().cpu().numpy()
        if cfg.unnormalize_actions:
            actions = utils.unnormalize_actions(actions, stats)
        scores = [None] * len(seeds)
        if score_mode == "joint_value":
            values = utils.extract_value_from_latent_sequence(latent, batch["value_latent_idx"])
            scores = ((values + 1) / 2).clamp(0, 1).float().cpu().tolist()
        # Future images are not part of the WAM wire response; no VAE decode
        # is needed to extract the actions and joint value from the same latent.
        return [
            {"index": index, "seed": seed, "actions": action, "score": score}
            for seed, action, score in zip(seeds, actions, scores)
        ]

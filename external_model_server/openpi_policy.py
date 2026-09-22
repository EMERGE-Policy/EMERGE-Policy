"""OpenPI policy loading and native batch inference helpers.

This module lives outside ``third_party/openpi`` so the upstream checkout stays
untouched. The shared model runtime owns the public network endpoint; this
module only adapts OpenPI's policy API to the runtime's batch adapter contract.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)


class OpenPIBatchPolicy:
    """Expose batched inference for an upstream ``openpi.policies.Policy``.

    OpenPI's public ``Policy.infer`` transforms one observation, inserts a
    singleton batch dimension, calls ``sample_actions``, and removes that
    dimension. This adapter performs the same operations for multiple
    observations while preserving per-observation input/output transforms.
    """

    _REQUIRED_ATTRIBUTES = (
        "_input_transform",
        "_output_transform",
        "_sample_actions",
        "_sample_kwargs",
        "_is_pytorch_model",
    )

    def __init__(self, policy: Any) -> None:
        missing = [
            name for name in self._REQUIRED_ATTRIBUTES if not hasattr(policy, name)
        ]
        if missing:
            names = ", ".join(missing)
            raise TypeError(
                "OpenPI batching requires an openpi policy.Policy; "
                f"missing internal attribute(s): {names}"
            )
        if not policy._is_pytorch_model and not hasattr(policy, "_rng"):
            raise TypeError("JAX OpenPI policy is missing its RNG state")
        self._policy = policy

    def infer_batch(
        self,
        observations: Sequence[dict[str, Any]],
        *,
        pad_to: int | None = None,
    ) -> list[dict[str, Any]]:
        if not observations:
            return []

        request_count = len(observations)
        if pad_to is not None and pad_to < request_count:
            raise ValueError(
                f"pad_to ({pad_to}) cannot be smaller than batch ({request_count})"
            )
        effective_batch_size = pad_to or request_count

        import jax
        import numpy as np
        from openpi.models import model as openpi_model

        transformed_inputs = []
        for observation in observations:
            # Match upstream Policy.infer: transforms may mutate their input.
            inputs = jax.tree.map(lambda value: value, observation)
            transformed_inputs.append(self._policy._input_transform(inputs))

        if effective_batch_size > request_count:
            transformed_inputs.extend(
                transformed_inputs[-1]
                for _ in range(effective_batch_size - request_count)
            )

        if self._policy._is_pytorch_model:
            import torch

            device = self._policy._pytorch_device

            def stack_torch(*values):
                array = np.ascontiguousarray(
                    np.stack([np.asarray(value) for value in values], axis=0)
                )
                return torch.from_numpy(array).to(device)

            batched_inputs = jax.tree.map(stack_torch, *transformed_inputs)
            sample_rng_or_device = device
        else:
            import jax.numpy as jnp

            batched_inputs = jax.tree.map(
                lambda *values: jnp.stack(
                    [jnp.asarray(value) for value in values], axis=0
                ),
                *transformed_inputs,
            )
            self._policy._rng, sample_rng_or_device = jax.random.split(
                self._policy._rng
            )

        observation = openpi_model.Observation.from_dict(batched_inputs)
        sample_kwargs = dict(self._policy._sample_kwargs)
        started = time.monotonic()
        actions = self._policy._sample_actions(
            sample_rng_or_device,
            observation,
            **sample_kwargs,
        )
        infer_ms = (time.monotonic() - started) * 1000

        batched_outputs = {
            "state": batched_inputs["state"],
            "actions": actions,
        }
        if self._policy._is_pytorch_model:
            batched_outputs = jax.tree.map(
                lambda value: np.asarray(value.detach().cpu()),
                batched_outputs,
            )
        else:
            # np.asarray synchronizes JAX device work before timing/result return.
            batched_outputs = jax.tree.map(np.asarray, batched_outputs)

        results: list[dict[str, Any]] = []
        for index in range(request_count):
            item = jax.tree.map(
                lambda value: value[index, ...],
                batched_outputs,
            )
            output = dict(self._policy._output_transform(item))
            output["policy_timing"] = {
                "infer_ms": infer_ms,
                "batch_size": request_count,
                "padded_batch_size": effective_batch_size,
            }
            results.append(output)
        return results


def create_policy(*, config_name: str, checkpoint_dir: str) -> Any:
    """Load an OpenPI policy without importing heavyweight modules at startup."""
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(config_name)
    logger.info(
        "Loaded openpi TrainConfig(name=%s): %s",
        config_name,
        train_config.model,
    )
    return policy_config.create_trained_policy(train_config, checkpoint_dir)

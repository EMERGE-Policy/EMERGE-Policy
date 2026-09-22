"""OpenPI model adapter for the shared model service runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Hashable, Sequence

import numpy as np

from external_model_server.model_service.contracts import ServiceDescriptor
from external_model_server.schemas import OPENPI


class OpenPIAdapter:
    """Load the official OpenPI policy directly inside the shared runtime."""

    def __init__(
        self,
        *,
        config_name: str,
        checkpoint_dir: str,
        model_id: str | None = None,
        max_batch_size: int = 1,
        pad_to_max: bool = False,
    ) -> None:
        self.config_name = config_name
        self.checkpoint_dir = checkpoint_dir
        self.model_id = model_id or config_name or Path(checkpoint_dir).name
        self.max_batch_size = max_batch_size
        self.pad_to_max = pad_to_max
        self._batch_policy = None

    @property
    def descriptor(self) -> ServiceDescriptor:
        return ServiceDescriptor(
            OPENPI.service,
            self.model_id,
            OPENPI.input_schema,
            OPENPI.output_schema,
            ("infer", "batch"),
        )

    def load(self) -> None:
        # Heavy OpenPI/JAX/Torch imports happen only on the runtime executor.
        from external_model_server.openpi_policy import (
            OpenPIBatchPolicy,
            create_policy,
        )

        policy = create_policy(
            config_name=self.config_name,
            checkpoint_dir=self.checkpoint_dir,
        )
        self._batch_policy = OpenPIBatchPolicy(policy)

    def close(self) -> None:
        self._batch_policy = None

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.infer_batch([request])[0]

    def batch_key(self, request: dict[str, Any]) -> Hashable:
        """Keep requests with incompatible tensor shapes out of one batch."""
        observation = request
        signature = []
        for key in sorted(observation):
            value = observation[key]
            if isinstance(value, np.ndarray):
                signature.append((key, value.shape, value.dtype.str))
            else:
                array = np.asarray(value)
                signature.append((key, array.shape, array.dtype.str))
        return tuple(signature)

    def infer_batch(
        self,
        requests: Sequence[dict[str, Any]],
    ) -> Sequence[dict[str, Any]]:
        if self._batch_policy is None:
            raise RuntimeError("OpenPI policy is not loaded")
        return self._batch_policy.infer_batch(
            requests,
            pad_to=self.max_batch_size if self.pad_to_max else None,
        )

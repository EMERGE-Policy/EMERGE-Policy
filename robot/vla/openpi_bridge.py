"""OpenPI business client for the shared model service."""

import logging
from dataclasses import replace

import httpx

from external_model_server.model_service.client import ModelClient
from external_model_server.model_service.contracts import ServiceError
from external_model_server.schemas import OPENPI

logger = logging.getLogger(__name__)


class Pi05Client:
    def __init__(self, *, timeout=120, model_id=None, discovery=None):
        self._client = ModelClient(replace(OPENPI, model_id=model_id),
                                   timeout=timeout, discovery=discovery)

    def health_check(self) -> bool:
        try:
            return self._client.health()["status"] == "ready"
        except (httpx.HTTPError, ServiceError, ValueError, OSError) as exc:
            logger.warning("OpenPI health check failed: %s", exc)
            return False

    def infer(self, element):
        import numpy as np
        result = self._client.infer(element)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError("OpenPI actions must be a finite (N, 7) array")
        result["actions"] = actions
        return result

    def close(self):
        self._client.close()

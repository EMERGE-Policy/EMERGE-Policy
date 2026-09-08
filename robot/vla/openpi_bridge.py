"""Lazy client bridge to the external OpenPI websocket server.

Importing this module does not import ``openpi`` or ``openpi_client``. The
client dependency is loaded only when the first inference connection is made,
so the lightweight controller modules remain importable without the OpenPI runtime.
"""

from __future__ import annotations

import logging
import os
from typing import Any
import urllib.parse
import urllib.request


logger = logging.getLogger(__name__)

_DEFAULT_SERVER_URL = "ws://localhost:8000"
_DEFAULT_SERVER_PORT = 8000


def _parse_server_endpoint(server_url: str) -> tuple[str, int, str]:
    """Resolve a server URL into ``(host, port, http_base)``."""
    raw = str(server_url or _DEFAULT_SERVER_URL).strip()
    if not raw:
        raw = _DEFAULT_SERVER_URL
    if "://" not in raw:
        raw = f"ws://{raw}"

    parsed = urllib.parse.urlparse(raw)
    host = parsed.hostname or "localhost"
    port = int(parsed.port or _DEFAULT_SERVER_PORT)
    http_scheme = "https" if parsed.scheme in {"https", "wss"} else "http"
    return host, port, f"{http_scheme}://{host}:{port}"


class Pi05Client:
    """Lazy wrapper around ``openpi_client.WebsocketClientPolicy``."""

    def __init__(
        self,
        server_url: str = _DEFAULT_SERVER_URL,
        *,
        timeout: float = 120.0,
    ) -> None:
        self._host, self._port, self._http_base = _parse_server_endpoint(
            server_url
        )
        self._timeout = float(timeout)
        self._client: Any | None = None

    @property
    def host(self) -> str:
        return self._host

    @property
    def port(self) -> int:
        return self._port

    def health_check(self) -> bool:
        """Return whether the OpenPI server responds on ``/healthz``."""
        timeout_s = min(max(self._timeout, 0.1), 2.0)
        url = f"{self._http_base}/healthz"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(
                urllib.request.Request(url),
                timeout=timeout_s,
            ) as response:
                return response.status == 200
        except Exception as error:
            logger.warning(
                "OpenPI health check failed for %s: %s: %s",
                url,
                type(error).__name__,
                error,
            )
            return False

    def infer(self, element: dict[str, Any]) -> dict[str, Any]:
        """Run one policy inference and normalize its action array."""
        if self._client is None and not self.health_check():
            raise RuntimeError(
                f"OpenPI policy server is unavailable at {self._http_base}"
            )

        result = self._ensure_client().infer(element)
        if "actions" in result:
            import numpy as np

            if not isinstance(result["actions"], np.ndarray):
                result["actions"] = np.asarray(
                    result["actions"],
                    dtype=np.float32,
                )
        return result

    def close(self) -> None:
        """Release the lazy websocket client reference."""
        self._client = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client

        for env_name in ("NO_PROXY", "no_proxy"):
            current = os.environ.get(env_name, "")
            hosts = {item.strip() for item in current.split(",") if item.strip()}
            hosts.update({"localhost", "127.0.0.1", "::1"})
            os.environ[env_name] = ",".join(sorted(hosts))

        from openpi_client import websocket_client_policy

        self._client = websocket_client_policy.WebsocketClientPolicy(
            self._host,
            self._port,
        )
        return self._client

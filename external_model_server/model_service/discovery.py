"""Self-describing model-service discovery."""

from __future__ import annotations

import asyncio
import ipaddress
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx

from .contracts import (
    ServiceError,
    ServiceExpectation,
    health_url,
    validate_description,
    websocket_url,
)
from .deadlines import timeout as deadline_after


@dataclass(frozen=True)
class DiscoveryConfig:
    host: str = "127.0.0.1"
    start_port: int = 8000
    end_port: int = 8099
    concurrency: int = 16
    connect_timeout: float = 0.5
    endpoint_timeout: float = 2
    scan_timeout: float = 15

    def __post_init__(self):
        if self.host != "localhost" and not ipaddress.ip_address(self.host).is_loopback:
            raise ValueError(
                "Port discovery is limited to loopback"
            )
        if not 1 <= self.start_port <= self.end_port <= 65535 or self.concurrency < 1:
            raise ValueError("Invalid discovery port range or concurrency")
        if any(not math.isfinite(value) or value <= 0 for value in (
            self.connect_timeout, self.endpoint_timeout, self.scan_timeout,
        )):
            raise ValueError("Discovery timeouts must be finite and positive")


@dataclass(frozen=True)
class DiscoveredService:
    endpoint: str
    description: dict
    latency_ms: int


@dataclass
class DiscoveryResult:
    services: list[DiscoveredService] = field(default_factory=list)
    incomplete: bool = False


def _parse_health(response: httpx.Response) -> dict:
    if response.status_code not in (200, 503):
        raise ServiceError("INVALID_RESPONSE", f"Health returned HTTP {response.status_code}")
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip()
    if content_type != "application/json":
        raise ServiceError("INVALID_RESPONSE", "Health must return application/json")
    description = validate_description(response.json(), health=True)
    if (response.status_code == 200) != (description["status"] == "ready"):
        raise ServiceError("INVALID_RESPONSE", "Health status and HTTP code disagree")
    return description


class ServiceDiscovery:
    """Probe and resolve self-describing model services for one caller."""

    def __init__(self, config: DiscoveryConfig | None = None):
        self.config = config or DiscoveryConfig()

    def probe(
        self,
        endpoint: str,
        *,
        expectation: ServiceExpectation,
        timeout: float | None = None,
    ) -> dict:
        endpoint = websocket_url(endpoint)
        with httpx.Client(timeout=timeout or self.config.endpoint_timeout, trust_env=False) as client:
            description = _parse_health(client.get(health_url(endpoint)))
        expectation.check(description, ready=False)
        return description

    async def probe_async(
        self,
        endpoint: str,
        *,
        expectation: ServiceExpectation,
        timeout: float | None = None,
    ) -> dict:
        endpoint = websocket_url(endpoint)
        limit = timeout or self.config.endpoint_timeout
        async with httpx.AsyncClient(timeout=limit, trust_env=False) as client:
            async with deadline_after(limit):
                description = _parse_health(await client.get(health_url(endpoint)))
        expectation.check(description, ready=False)
        return description

    async def scan(self) -> DiscoveryResult:
        """Scan the configured local port range."""
        result = DiscoveryResult()
        host = f"[{self.config.host}]" if ":" in self.config.host else self.config.host
        candidates = [
            websocket_url(f"ws://{host}:{port}")
            for port in range(self.config.start_port, self.config.end_port + 1)
        ]
        order = {endpoint: index for index, endpoint in enumerate(candidates)}
        semaphore = asyncio.Semaphore(self.config.concurrency)
        client_timeout = httpx.Timeout(
            self.config.endpoint_timeout,
            connect=self.config.connect_timeout,
        )

        async with httpx.AsyncClient(timeout=client_timeout, trust_env=False) as client:
            async def probe_candidate(endpoint: str):
                async with semaphore:
                    started = time.monotonic()
                    try:
                        async with deadline_after(self.config.endpoint_timeout):
                            description = _parse_health(
                                await client.get(health_url(endpoint))
                            )
                    except (httpx.HTTPError, ServiceError, ValueError, TimeoutError):
                        return
                    result.services.append(DiscoveredService(
                        endpoint,
                        description,
                        round((time.monotonic() - started) * 1000),
                    ))

            try:
                async with deadline_after(self.config.scan_timeout):
                    await asyncio.gather(*(probe_candidate(endpoint) for endpoint in candidates))
            except TimeoutError:
                result.incomplete = True

        result.services.sort(key=lambda item: order[item.endpoint])
        unique = {}
        for item in result.services:
            unique.setdefault(item.description["instance_id"], item)
        result.services = list(unique.values())
        return result

    def resolve(self, expectation: ServiceExpectation) -> str:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(lambda: asyncio.run(self.scan())).result()
        return self._select(result, expectation)

    async def resolve_async(self, expectation: ServiceExpectation) -> str:
        return self._select(await self.scan(), expectation)

    def is_ready(
        self,
        expectation: ServiceExpectation,
        *,
        timeout: float = 2,
    ) -> bool:
        try:
            resolved = self.resolve(expectation)
            return self.probe(
                resolved,
                expectation=expectation,
                timeout=timeout,
            )["status"] == "ready"
        except (httpx.HTTPError, ServiceError, ValueError, OSError):
            return False

    @staticmethod
    def _select(result: DiscoveryResult, expectation: ServiceExpectation) -> str:
        matches = []
        for item in result.services:
            try:
                expectation.check(item.description, ready=False)
            except ServiceError:
                continue
            matches.append(item.endpoint)
        if result.incomplete:
            raise ServiceError(
                "DISCOVERY_INCOMPLETE",
                "Discovery deadline exceeded; narrow the local port range",
            )
        if len(matches) != 1:
            raise ServiceError(
                "DISCOVERY_FAILED",
                f"Expected one {expectation.service} endpoint, found {matches}",
            )
        return matches[0]

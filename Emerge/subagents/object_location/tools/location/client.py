"""Client for the shared external model server protocol."""

from __future__ import annotations

import asyncio
from typing import Any

from external_model_server.protocol import pack_message, unpack_message


class ModelServerClient:
    def __init__(self, url: str, *, timeout: float) -> None:
        self.url = url
        self.timeout = timeout

    async def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        from websockets.asyncio.client import connect

        async with asyncio.timeout(self.timeout):
            async with connect(
                self.url,
                compression=None,
                max_size=None,
                proxy=None,
            ) as websocket:
                await websocket.recv()  # server metadata
                await websocket.send(pack_message(request))
                response = unpack_message(await websocket.recv())

        if not response["ok"]:
            error = response["error"]
            raise RuntimeError(
                f"{error['type']} from {self.url}: {error['message']}"
            )
        return dict(response["result"])

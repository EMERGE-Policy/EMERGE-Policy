"""Common sync and async clients. Sent requests are never replayed automatically."""

import asyncio
import contextlib
import math
import socket
import threading
import time
from uuid import uuid4

from websockets.asyncio.client import connect as async_connect
from websockets.sync.client import connect

from .codec import pack_message, unpack_message
from .contracts import (
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    response_result,
    validate_description,
)
from .deadlines import timeout as deadline_after
from .discovery import ServiceDiscovery


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Inference deadline exceeded; a sent request may still be running")
    return remaining


def _request(payload, expectation, request_id, timeout):
    return pack_message({"request_id": request_id, "protocol_version": PROTOCOL_VERSION,
                         "operation": "infer", "input_schema": expectation.input_schema,
                         "timeout_s": timeout, "payload": payload})


class ModelClient:
    def __init__(self, expectation, *, timeout=120, connect_timeout=2, discovery=None):
        if any(not math.isfinite(v) or v <= 0 for v in (timeout, connect_timeout)):
            raise ValueError("Client timeouts must be finite and positive")
        self.expectation = expectation
        self.timeout, self.connect_timeout = timeout, connect_timeout
        self.discovery = discovery or ServiceDiscovery()
        self.description = None
        self._connection = None
        self._resolved_endpoint = None
        self._lock = threading.Lock()

    def health(self):
        return self.discovery.probe(
            self._resolve(), timeout=self.connect_timeout, expectation=self.expectation,
        )

    def _resolve(self):
        if self._resolved_endpoint is None:
            self._resolved_endpoint = self.discovery.resolve(self.expectation)
        return self._resolved_endpoint

    def infer(self, payload, *, timeout=None):
        budget = self.timeout if timeout is None else timeout
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("Inference timeout must be finite and positive")
        deadline = time.monotonic() + budget
        if not self._lock.acquire(timeout=budget):
            raise TimeoutError("Deadline exceeded waiting for the client connection")
        timer = None
        try:
            if self._connection is None:
                endpoint = self._resolve()
                self._connection = connect(endpoint, compression=None, proxy=None,
                                           open_timeout=min(self.connect_timeout, _remaining(deadline)),
                                           close_timeout=0.5, max_size=MAX_MESSAGE_BYTES)
                self.description = validate_description(unpack_message(
                    self._connection.recv(timeout=min(self.connect_timeout, _remaining(deadline)))))
                self.expectation.check(self.description)
            connection = self._connection
            # recv has a deadline; this timer additionally interrupts a blocked send.
            def abort():
                with contextlib.suppress(OSError):
                    connection.socket.shutdown(socket.SHUT_RDWR)
            timer = threading.Timer(_remaining(deadline), abort)
            timer.daemon = True
            timer.start()
            request_id = uuid4().hex
            connection.send(_request(payload, self.expectation, request_id, _remaining(deadline)))
            response = unpack_message(connection.recv(timeout=_remaining(deadline)))
            return response_result(response, request_id)
        except BaseException:
            self._close()
            raise
        finally:
            if timer:
                timer.cancel()
            self._lock.release()

    def _close(self):
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.close()

    def close(self):
        with self._lock:
            self._close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class AsyncModelClient:
    def __init__(self, expectation, *, timeout=120, connect_timeout=2, discovery=None):
        if any(not math.isfinite(v) or v <= 0 for v in (timeout, connect_timeout)):
            raise ValueError("Client timeouts must be finite and positive")
        self.expectation = expectation
        self.timeout, self.connect_timeout = timeout, connect_timeout
        self.discovery = discovery or ServiceDiscovery()
        self.description = None
        self._connection = None
        self._resolved_endpoint = None
        self._lock = asyncio.Lock()

    async def health(self):
        return await self.discovery.probe_async(
            await self._resolve(),
            timeout=self.connect_timeout,
            expectation=self.expectation,
        )

    async def _resolve(self):
        if self._resolved_endpoint is None:
            self._resolved_endpoint = await self.discovery.resolve_async(self.expectation)
        return self._resolved_endpoint

    async def infer(self, payload, *, timeout=None):
        budget = self.timeout if timeout is None else timeout
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("Inference timeout must be finite and positive")
        deadline = time.monotonic() + budget
        async with deadline_after(budget):
            async with self._lock:
                try:
                    if self._connection is None:
                        endpoint = await self._resolve()
                        self._connection = await async_connect(endpoint, compression=None, proxy=None,
                                                               open_timeout=min(self.connect_timeout, _remaining(deadline)),
                                                               close_timeout=0.5, max_size=MAX_MESSAGE_BYTES)
                        async with deadline_after(min(self.connect_timeout, _remaining(deadline))):
                            self.description = validate_description(unpack_message(await self._connection.recv()))
                        self.expectation.check(self.description)
                    request_id = uuid4().hex
                    await self._connection.send(_request(payload, self.expectation, request_id, _remaining(deadline)))
                    return response_result(unpack_message(await self._connection.recv()), request_id)
                except BaseException:
                    await self._close()
                    raise

    async def _close(self):
        connection, self._connection = self._connection, None
        if connection is not None:
            await connection.close()

    async def close(self):
        async with self._lock:
            await self._close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

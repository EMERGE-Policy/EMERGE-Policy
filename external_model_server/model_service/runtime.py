"""One public runtime for local models and adapters to existing services."""

import argparse
import asyncio
import http
import json
import logging
import math
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from .adapter import BatchModelAdapter
from .batching import Scheduler
from .codec import pack_message, unpack_message
from .contracts import (
    HEALTH_SCHEMA,
    MAX_MESSAGE_BYTES,
    PROTOCOL_VERSION,
    ModelFailureError,
    ServiceError,
    validate_request,
)

logger = logging.getLogger(__name__)


class ModelServerRuntime:
    def __init__(self, adapter, *, host="127.0.0.1", port=8000,
                 max_batch_size=1, batch_wait_ms=0, queue_capacity=64,
                 max_message_bytes=MAX_MESSAGE_BYTES, shutdown_timeout=30,
                 remote=False, concurrency=1, health_interval=2):
        if min(max_batch_size, queue_capacity, max_message_bytes, concurrency) < 1:
            raise ValueError("Batch size, capacity, message limit and concurrency must be positive")
        if not math.isfinite(batch_wait_ms) or batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be finite and non-negative")
        if any(not math.isfinite(v) or v <= 0 for v in (shutdown_timeout, health_interval)):
            raise ValueError("Shutdown and health intervals must be finite and positive")
        if not remote and concurrency != 1:
            raise ValueError("Local model execution is serialized")
        self.adapter = adapter
        self.descriptor = adapter.descriptor
        self.host, self.port = host, port
        self.remote = remote
        self.batched = not remote and isinstance(adapter, BatchModelAdapter)
        if max_batch_size > 1 and not self.batched:
            raise ValueError("This adapter does not support runtime batching")
        self.max_batch_size, self.batch_wait_ms = max_batch_size, batch_wait_ms
        self.queue_capacity, self.concurrency = queue_capacity, concurrency
        self.max_message_bytes = max_message_bytes
        self.shutdown_timeout, self.health_interval = shutdown_timeout, health_interval
        self.instance_id = uuid4().hex
        self.status, self.detail = "starting", ""
        self._stop = asyncio.Event()
        self.bound = asyncio.Event()
        self.scheduler = None
        self._executor = None

    def description(self):
        return {**self.descriptor.to_dict(), "instance_id": self.instance_id,
                "protocol_version": PROTOCOL_VERSION, "status": self.status, "detail": self.detail}

    def _health(self, connection, request):
        if request.path != "/healthz":
            return None
        status = http.HTTPStatus.OK if self.status == "ready" else http.HTTPStatus.SERVICE_UNAVAILABLE
        response = connection.respond(status, json.dumps({"schema": HEALTH_SCHEMA, **self.description()}) + "\n")
        del response.headers["Content-Type"]
        response.headers["Content-Type"] = "application/json"
        return response

    async def _local(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    async def _initialize(self):
        try:
            if self.remote:
                await self.adapter.load()
            else:
                await self._local(self.adapter.load)
        except Exception:
            self.status, self.detail = "failed", "Model initialization failed; see server logs"
            logger.exception("service=%s instance=%s load failed", self.descriptor.service, self.instance_id)
            return
        if self._stop.is_set():
            return
        self.status, self.detail = "ready", ""
        message = "✓ service=%s instance=%s READY ws://%s:%s"
        if sys.stderr.isatty() and "NO_COLOR" not in os.environ:
            message = "\033[1;32m" + message + "\033[0m"
        logger.info(message, self.descriptor.service, self.instance_id, self.host, self.port)
        if self.remote:
            while not self._stop.is_set():
                await asyncio.sleep(self.health_interval)
                try:
                    await self.adapter.check()
                except (OSError, TimeoutError, ConnectionClosed, ServiceError) as exc:
                    self.status, self.detail = "starting", str(exc)
                else:
                    self.status, self.detail = "ready", ""

    async def _execute(self, requests):
        try:
            if self.remote:
                return [await self.adapter.infer(requests[0])]
            if self.batched:
                return await self._local(self.adapter.infer_batch, requests)
            return [await self._local(self.adapter.infer, requests[0])]
        except ModelFailureError:
            self.status, self.detail = "failed", "Model unavailable; see server logs"
            self.scheduler.reject_pending(ServiceError("NOT_READY", self.detail))
            raise
        except ServiceError as exc:
            if self.remote and exc.code == "NOT_READY":
                self.status, self.detail = "starting", str(exc)
            raise

    async def _request(self, raw):
        request_id = None
        started = time.monotonic()
        try:
            try:
                decoded = unpack_message(raw)
                if isinstance(decoded, dict):
                    candidate = decoded.get("request_id")
                    if isinstance(candidate, str) and 0 < len(candidate) <= 128:
                        request_id = candidate
                message = validate_request(decoded)
            except (ValueError, TypeError, KeyError) as exc:
                raise ServiceError("INVALID_REQUEST", str(exc)) from exc
            request_id = message["request_id"]
            if self.status != "ready":
                raise ServiceError("NOT_READY", f"Service is {self.status}", retryable=True)
            if message.get("input_schema") != self.descriptor.input_schema:
                raise ServiceError("SCHEMA_MISMATCH", "Request input schema mismatch")
            remaining = message["timeout_s"] - (time.monotonic() - started)
            result, timing = await self.scheduler.infer(message["payload"], max(0, remaining))
            response = {"request_id": request_id, "ok": True, "result": result, "timing": timing}
            logger.info("service=%s instance=%s request=%s queue_ms=%.1f infer_ms=%.1f",
                        self.descriptor.service, self.instance_id, request_id,
                        timing["queue_ms"], timing["infer_ms"])
        except TimeoutError:
            response = self._error(request_id, ServiceError("DEADLINE_EXCEEDED", "Inference deadline exceeded; execution may still be running"))
        except ServiceError as exc:
            response = self._error(request_id, exc)
        except (ValueError, TypeError, KeyError) as exc:
            response = self._error(request_id, ServiceError("INVALID_REQUEST", str(exc)))
        except Exception:
            logger.exception("service=%s instance=%s request=%s inference failed", self.descriptor.service, self.instance_id, request_id)
            response = self._error(request_id, ServiceError("INFERENCE_FAILED", "Model inference failed; see server logs"))
        try:
            encoded = pack_message(response)
            if len(encoded) > self.max_message_bytes:
                raise ValueError("Result exceeds message limit")
            return encoded
        except (ValueError, TypeError, OverflowError):
            logger.exception("service=%s request=%s invalid model output", self.descriptor.service, request_id)
            return pack_message(self._error(request_id, ServiceError("INTERNAL", "Invalid or oversized model output")))

    def _error(self, request_id, error):
        logger.warning("service=%s instance=%s request=%s code=%s", self.descriptor.service, self.instance_id, request_id, error.code)
        return {"request_id": request_id, "ok": False, "error": error.to_dict()}

    async def _handle(self, websocket):
        await websocket.send(pack_message({"type": "metadata", **self.description()}))
        closed = asyncio.create_task(websocket.wait_closed())
        operation = None
        try:
            async for raw in websocket:
                operation = asyncio.create_task(self._request(raw))
                done, _ = await asyncio.wait((operation, closed), return_when=asyncio.FIRST_COMPLETED)
                if closed in done:
                    return
                await websocket.send(await operation)
        except ConnectionClosed:
            pass
        finally:
            closed.cancel()
            if operation is not None:
                operation.cancel()
            await asyncio.gather(closed, *([operation] if operation else []), return_exceptions=True)

    def stop(self):
        self._stop.set()

    async def run(self):
        self._executor = None if self.remote else ThreadPoolExecutor(max_workers=1, thread_name_prefix=self.descriptor.service)
        self.scheduler = Scheduler(
            self._execute, batch_key=self.adapter.batch_key if self.batched else None,
            max_batch_size=self.max_batch_size, batch_wait_ms=self.batch_wait_ms,
            queue_capacity=self.queue_capacity, workers=self.concurrency, cancellable=self.remote,
        )
        initialization = None
        try:
            async with serve(self._handle, self.host, self.port, compression=None,
                             max_size=self.max_message_bytes, process_request=self._health,
                             close_timeout=2) as server:
                self.port = server.sockets[0].getsockname()[1]
                self.bound.set()
                initialization = asyncio.create_task(self._initialize())
                try:
                    await self._stop.wait()
                finally:
                    self._stop.set()
                    self.status = "draining"
                    if self.remote:
                        initialization.cancel()
                    cleanup = asyncio.create_task(self._drain(initialization))
                    try:
                        await asyncio.wait_for(asyncio.shield(cleanup), self.shutdown_timeout)
                    except TimeoutError:
                        logger.error("Shutdown deadline exceeded for %s; launcher must terminate the process", self.descriptor.service)
                        # Don't release a model while GPU execution is still running.
                        cleanup.cancel()
                        await asyncio.gather(cleanup, return_exceptions=True)
        finally:
            self.status = "stopped"
            self.scheduler.closed = True
            self.scheduler.available.set()
            self.scheduler.reject_pending(ServiceError("NOT_READY", "Server stopped"))
            if initialization is not None:
                initialization.cancel()
                await asyncio.gather(initialization, return_exceptions=True)
            for worker in self.scheduler.workers:
                worker.cancel()
            await asyncio.gather(*self.scheduler.workers, return_exceptions=True)
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)

    async def _drain(self, initialization):
        await asyncio.gather(initialization, return_exceptions=True)
        await self.scheduler.close()
        if self.remote:
            await self.adapter.close()
        else:
            await self._local(self.adapter.close)

    def serve_forever(self):
        async def main():
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self.stop)
            try:
                await self.run()
            finally:
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.remove_signal_handler(sig)
        asyncio.run(main())


def add_runtime_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--model-id", help="Public model/configuration identifier")
    parser.add_argument("--queue-capacity", type=int, default=64)
    parser.add_argument("--max-message-bytes", type=int, default=MAX_MESSAGE_BYTES)
    parser.add_argument("--shutdown-timeout", type=float, default=30)


def runtime_arguments(args):
    return {"queue_capacity": args.queue_capacity, "max_message_bytes": args.max_message_bytes,
            "shutdown_timeout": args.shutdown_timeout}

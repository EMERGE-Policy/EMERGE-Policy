"""Bounded scheduling; cancellation drops queued work, not running GPU kernels."""

import asyncio
import time
from collections import deque
from dataclasses import dataclass

from .contracts import ServiceError


@dataclass(eq=False)
class Pending:
    payload: dict
    key: object
    submitted: float
    future: asyncio.Future


class Scheduler:
    def __init__(self, execute, *, batch_key=None, max_batch_size=1,
                 batch_wait_ms=0, queue_capacity=64, workers=1, cancellable=False):
        self.execute = execute
        self.batch_key = batch_key
        self.max_batch_size = max_batch_size
        self.batch_wait = batch_wait_ms / 1000
        self.capacity = queue_capacity
        self.cancellable = cancellable
        self.queue = deque()
        self.available = asyncio.Event()
        self.closed = False
        self.workers = [asyncio.create_task(self._work()) for _ in range(workers)]

    async def infer(self, payload, timeout):
        if self.closed:
            raise ServiceError("NOT_READY", "Server is draining")
        if len(self.queue) >= self.capacity:
            raise ServiceError("OVERLOADED", "Inference queue is full", retryable=True)
        try:
            key = self.batch_key(payload) if self.batch_key else None
        except (ValueError, TypeError, KeyError) as exc:
            raise ServiceError("INVALID_REQUEST", str(exc)) from exc
        item = Pending(payload, key, time.monotonic(), asyncio.get_running_loop().create_future())
        self.queue.append(item)
        self.available.set()
        try:
            return await asyncio.wait_for(item.future, timeout)
        finally:
            if item in self.queue:
                self.queue.remove(item)

    def reject_pending(self, error):
        while self.queue:
            item = self.queue.popleft()
            if not item.future.done():
                item.future.set_exception(error)

    async def close(self):
        self.closed = True
        self.reject_pending(ServiceError("NOT_READY", "Server is draining"))
        self.available.set()
        await asyncio.gather(*self.workers)

    async def _work(self):
        while not self.closed:
            if not self.queue:
                self.available.clear()
                await self.available.wait()
                continue
            if self.batch_wait and self.max_batch_size > 1:
                await asyncio.sleep(self.batch_wait)
            if not self.queue:
                continue
            first = self.queue.popleft()
            batch = [first]
            for item in list(self.queue):
                if len(batch) == self.max_batch_size:
                    break
                if item.key == first.key:
                    self.queue.remove(item)
                    batch.append(item)
            active = [item for item in batch if not item.future.done()]
            if not active:
                continue
            started = time.monotonic()
            operation = asyncio.create_task(self.execute([item.payload for item in active]))
            # Only remote I/O can be cancelled safely. Local executor work must finish
            # before another inference or model.close() can run.
            def cancel_remote(_future, active=tuple(active), operation=operation):
                if all(item.future.cancelled() for item in active):
                    operation.cancel()
            if self.cancellable:
                for item in active:
                    item.future.add_done_callback(cancel_remote)
            try:
                results = await operation
                if len(results) != len(active):
                    raise RuntimeError("Adapter returned the wrong number of batch results")
                finished = time.monotonic()
                for item, result in zip(active, results, strict=True):
                    if not item.future.done():
                        item.future.set_result((result, {
                            "queue_ms": (started - item.submitted) * 1000,
                            "infer_ms": (finished - started) * 1000,
                            "batch_size": len(active),
                        }))
            except asyncio.CancelledError:
                if not self.cancellable or self.closed:
                    raise
            except Exception as exc:
                for item in active:
                    if not item.future.done():
                        item.future.set_exception(exc)
            finally:
                if self.cancellable:
                    for item in active:
                        item.future.remove_done_callback(cancel_remote)

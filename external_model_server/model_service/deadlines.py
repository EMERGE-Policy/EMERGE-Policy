"""Timeout contexts for the Python 3.10 Cosmos and newer perception environments."""

import sys

if sys.version_info >= (3, 11):
    from asyncio import timeout as timeout
else:
    import asyncio
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def timeout(delay):
        """Small ``asyncio.timeout`` backport for the service's Python 3.10 env."""
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("timeout must run inside an asyncio task")
        expired = False

        def cancel_task():
            nonlocal expired
            expired = True
            task.cancel()

        handle = asyncio.get_running_loop().call_later(delay, cancel_task)
        try:
            yield
        except asyncio.CancelledError as exc:
            if expired:
                raise TimeoutError from exc
            raise
        finally:
            handle.cancel()

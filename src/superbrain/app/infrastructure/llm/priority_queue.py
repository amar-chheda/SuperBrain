"""Single-worker asyncio priority queue for serialising Ollama LLM calls.

Ollama processes requests sequentially; without this gate, background
classification floods the queue and stalls latency-sensitive QA calls.

Priority numbers follow convention: lower = more urgent.
  1 = QA (user waiting on a response)
  2 = Ingestion (user-triggered, background)
  3 = Classification / digest (fully background)
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

import structlog

log = structlog.get_logger(__name__)
T = TypeVar("T")


class LLMPriorityQueue:
    """Serialises all LLM calls through a single asyncio.PriorityQueue."""

    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[tuple[int, int, Callable[[], Awaitable[Any]], asyncio.Future[Any]]] = asyncio.PriorityQueue()
        self._seq = 0
        self._worker_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._worker_task = asyncio.get_event_loop().create_task(
            self._worker(), name="llm-priority-worker"
        )
        log.info("llm_queue.started")

    def stop(self) -> None:
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
        log.info("llm_queue.stopped")

    async def submit(self, call: Callable[[], Awaitable[T]], *, priority: int) -> T:
        """Enqueue `call` at `priority` and suspend until the worker runs it."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()
        self._seq += 1
        await self._queue.put((priority, self._seq, call, future))
        remaining = self._queue.qsize()
        if remaining:
            log.debug("llm_queue.enqueued", priority=priority, queue_depth=remaining)
        return await future

    async def _worker(self) -> None:
        while True:
            priority, _seq, call, future = await self._queue.get()
            try:
                if future.cancelled():
                    continue
                result = await call()
                if not future.done():
                    future.set_result(result)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            finally:
                self._queue.task_done()
            depth = self._queue.qsize()
            if depth:
                log.debug("llm_queue.drained_one", priority=priority, remaining=depth)

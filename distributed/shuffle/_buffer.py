from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict
from collections.abc import Iterator, Sized
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, NamedTuple, TypeVar

from distributed.metrics import time
from distributed.shuffle._limiter import ResourceLimiter
from distributed.sizeof import sizeof

logger = logging.getLogger("distributed.shuffle")
if TYPE_CHECKING:
    # TODO import from collections.abc (requires Python >=3.12)
    # TODO import from typing (requires Python >= 3.10)
    from typing_extensions import Buffer, TypeAlias
else:
    Buffer = Sized

ShardType = TypeVar("ShardType", bound=Buffer)

T = TypeVar("T")


BufferState: TypeAlias = Literal["open", "flushing", "flushed", "erred", "closed"]


class SizedShard(NamedTuple):
    shard: Any
    size: int


class BaseBuffer(Generic[ShardType]):
    """Base class for buffers in P2P

    The buffer allows for concurrent writes and buffers shard to reduce overhead.
    Writes are non-blocking while the buffer's memory limit is not full. Once full,
    buffers become blocking to apply back pressure and limit memory consumption.

    The shards are typically provided in a format like::

        {
            "bucket-0": [<Shard1>, <Shard2>, <Shard3>],
            "bucket-1": [<Shard4>, <Shard5>],
        }

    If exceptions occur during writing, the buffer automatically errs and stores the
    exception. Subsequent attempts to write and to flush will raise this exception.
    To ensure that the buffer finished successfully, please call
    `BaseBuffer.raise_if_erred`.
    """

    #: Whether or not to drain the buffer when flushing
    drain: ClassVar[bool] = True

    #: List of buffered shards per output key with their approximate in-memory size
    shards: defaultdict[str, list[SizedShard]]
    #: Total size of in-memory data per flushable key
    flushable_sizes: defaultdict[str, int]
    #: Total size of in-memory data per currently flushing key
    flushing_sizes: dict[str, int]
    #: Limit of concurrent tasks flushing data
    concurrency_limit: int
    #: Plugin-wide resource limiter used to apply back-pressure
    memory_limiter: ResourceLimiter
    #: Diagnostic data
    diagnostics: dict[str, float]
    #: Maximum size of data per key when flushing
    max_message_size: int

    bytes_total: int
    bytes_memory: int
    bytes_written: int
    bytes_read: int

    #: State of the buffer
    _state: BufferState
    #: Exception that occurred while flushing (if exists)
    _exception: Exception | None
    #: Async condition used for coordination
    _flush_condition: asyncio.Condition
    #: Background tasks flushing data
    _tasks: list[asyncio.Task]

    def __init__(
        self,
        memory_limiter: ResourceLimiter,
        concurrency_limit: int = 2,
        max_message_size: int = -1,
    ) -> None:
        self.shards = defaultdict(list[SizedShard])
        self.flushable_sizes = defaultdict(int)
        self.flushing_sizes = {}
        self.memory_limiter = memory_limiter
        self.diagnostics: dict[str, float] = defaultdict(float)
        self.max_message_size = max_message_size

        self.bytes_total = 0
        self.bytes_memory = 0
        self.bytes_written = 0
        self.bytes_read = 0
        self._state = "open"
        self._exception = None
        self._flush_condition = asyncio.Condition()
        if self.memory_limiter.limit or self.drain:
            self._tasks = [
                asyncio.create_task(self._background_task())
                for _ in range(concurrency_limit)
            ]
        else:
            self._tasks = []

    def heartbeat(self) -> dict[str, Any]:
        return {
            "memory": self.bytes_memory,
            "total": self.bytes_total,
            "buckets": len(self.shards),
            "written": self.bytes_written,
            "read": self.bytes_read,
            "diagnostics": self.diagnostics,
            "memory_limit": self.memory_limiter.limit,
        }

    async def write(self, data: dict[str, list[Any]]) -> None:
        self.raise_if_erred()

        if self._state != "open":
            raise RuntimeError(
                f"{self} is no longer open for new data, it is {self._state}."
            )

        if not data:
            return

        async with self._flush_condition:
            for worker, shards in data.items():
                for shard in shards:
                    size = sizeof(shard)
                    self.shards[worker].append(SizedShard(shard=shard, size=size))
                    if worker in self.flushing_sizes:
                        self.flushing_sizes[worker] += size
                    else:
                        self.flushable_sizes[worker] += size
                    self.bytes_memory += size
                    self.bytes_total += size
                    self.memory_limiter.increase(size)
            del data
            self._flush_condition.notify_all()
        await self.memory_limiter.wait_for_available()

    async def _background_task(self) -> None:
        def _continue() -> bool:
            return bool(self.flushable_sizes or self._state != "open")

        while True:
            async with self._flush_condition:
                await self._flush_condition.wait_for(_continue)
                if self._state != "open" and not (self.drain and self.flushable_sizes):
                    break
            await self.flush_largest()

    async def flush_largest(self) -> None:
        if not self.flushable_sizes or self._state not in {"open", "flushing"}:
            return
        largest_partition = max(
            self.flushable_sizes, key=self.flushable_sizes.__getitem__
        )

        partition_size = self.flushable_sizes.pop(largest_partition)
        shards: list[ShardType]

        if self.max_message_size > 0:
            shards = []
            message_size = 0
            while message_size < self.max_message_size:
                try:
                    shard, size = self.shards[largest_partition].pop()
                    message_size += size
                    shards.append(shard)
                except IndexError:
                    break
            if message_size == partition_size:
                assert not self.shards[largest_partition]
                del self.shards[largest_partition]
        else:
            shards = [shard for shard, _ in self.shards.pop(largest_partition)]
            message_size = partition_size

        self.flushing_sizes[largest_partition] = partition_size - message_size

        start = time()
        try:
            bytes_written = await self._flush(largest_partition, shards)
            if bytes_written is None:
                bytes_written = message_size

            self.bytes_memory -= message_size
            self.bytes_written += bytes_written
        except Exception as e:
            if not self._state == "erred":
                self._exception = e
                self._state = "erred"
                self.shards.clear()
                flushable_size = sum(self.flushable_sizes.values())
                self.flushable_sizes.clear()
                # flushing_size = sum(self.flushing_sizes.values())
                # self.flushing_sizes.clear()
                await self.memory_limiter.decrease(flushable_size)
        finally:
            stop = time()
            self.diagnostics["avg_size"] = (
                0.98 * self.diagnostics["avg_size"] + 0.02 * message_size
            )
            self.diagnostics["avg_duration"] = 0.98 * self.diagnostics[
                "avg_duration"
            ] + 0.02 * (stop - start)

            async with self._flush_condition:
                size_since_flush = self.flushing_sizes.pop(largest_partition)
                if self._state in {"open", "flushing"}:
                    if size_since_flush > 0:
                        self.flushable_sizes[largest_partition] = size_since_flush
                else:
                    assert self._state == "erred"
                    await self.memory_limiter.decrease(size_since_flush)
                await self.memory_limiter.decrease(message_size)
                self._flush_condition.notify_all()

    async def flush(self) -> None:
        if self._state in {"flushed", "closed"}:
            return

        if self._state == "flushing":
            async with self._flush_condition:
                await self._flush_condition.wait_for(
                    lambda: self._state in {"erred", "flushed", "closed"}
                )
            self.raise_if_erred()
            return
        self.raise_if_erred()
        assert self._state == "open", self._state
        logger.debug(f"Flushing {self}")
        self._state = "flushing"
        try:
            async with self._flush_condition:
                self._flush_condition.notify_all()
            await asyncio.gather(*self._tasks)
            self.raise_if_erred()
            assert self._state == "flushing"
            self._state = "flushed"
            logger.debug(f"Successfully flushed {self}")
        except Exception:
            logger.debug(f"Failed to flush {self}, now in {self._state}")
            raise
        finally:
            async with self._flush_condition:
                self._flush_condition.notify_all()

    async def close(self) -> None:
        if self._state == "closed":
            return

        try:
            await self.flush()
        except Exception:
            assert self._state == "erred"
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._state = "closed"
        self.shards.clear()
        self.flushable_sizes.clear()
        self.bytes_memory = 0
        assert not self.flushing_sizes

    async def __aenter__(self) -> BaseBuffer:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    def raise_if_erred(self) -> None:
        if self._exception:
            assert self._state == "erred"
            raise self._exception

    async def _flush(self, id: str, shards: list[ShardType]) -> int | None:
        raise NotImplementedError()

    @contextlib.contextmanager
    def time(self, name: str) -> Iterator[None]:
        start = time()
        yield
        stop = time()
        self.diagnostics[name] += stop - start

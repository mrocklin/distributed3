from __future__ import annotations

import asyncio
import contextlib
import logging
from collections import defaultdict
from collections.abc import Iterator, Sized
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar

from typing_extensions import TypeAlias

from distributed.metrics import time
from distributed.shuffle._limiter import ResourceLimiter
from distributed.sizeof import sizeof

logger = logging.getLogger("distributed.shuffle")
if TYPE_CHECKING:
    # TODO import from collections.abc (requires Python >=3.12)
    from typing_extensions import Buffer
else:
    Buffer = Sized

ShardType = TypeVar("ShardType", bound=Buffer)

T = TypeVar("T")


BufferState: TypeAlias = Literal["open", "flushing", "flushed", "erred", "closed"]


class _List(list[T]):
    # This ensures that the distributed.protocol will not iterate over this collection
    pass


class ShardsBuffer(Generic[ShardType]):
    """A buffer for P2P shuffle

    The objects to buffer are typically bytes belonging to certain shards.
    Typically the buffer is implemented on sending and receiving end.

    The buffer allows for concurrent writing and buffers shards to reduce overhead of writing.

    The shards are typically provided in a format like::

        {
            "bucket-0": [b"shard1", b"shard2"],
            "bucket-1": [b"shard1", b"shard2"],
        }

    Buckets typically correspond to output partitions.

    If exceptions occur during writing, the buffer is automatically closed. Subsequent attempts to write will raise the same exception.
    Flushing will not raise an exception. To ensure that the buffer finished successfully, please call `ShardsBuffer.raise_on_exception`
    """

    shards: defaultdict[str, list[ShardType]]
    sizes: defaultdict[str, int]
    sizes_detail: defaultdict[str, list[int]]
    concurrency_limit: int
    memory_limiter: ResourceLimiter
    diagnostics: dict[str, float]
    max_message_size: int

    bytes_total: int
    bytes_memory: int
    bytes_written: int
    bytes_read: int

    _accepts_input: bool
    _inputs_done: bool
    _exception: None | Exception
    _tasks: list[asyncio.Task]
    _shards_available: asyncio.Condition
    _flush_lock: asyncio.Lock

    def __init__(
        self,
        memory_limiter: ResourceLimiter,
        concurrency_limit: int = 2,
        max_message_size: int = -1,
    ) -> None:
        self._accepts_input = True
        self.shards = defaultdict(list)
        self.sizes = defaultdict(int)
        self.sizes_detail = defaultdict(list)
        self._exception = None
        self.concurrency_limit = concurrency_limit
        self._inputs_done = False
        self.memory_limiter = memory_limiter
        self.diagnostics: dict[str, float] = defaultdict(float)
        self._tasks = [
            asyncio.create_task(self._background_task())
            for _ in range(concurrency_limit)
        ]
        self._shards_available = asyncio.Condition()
        self._flush_lock = asyncio.Lock()
        self.max_message_size = max_message_size

        self.bytes_total = 0
        self.bytes_memory = 0
        self.bytes_written = 0
        self.bytes_read = 0

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

    async def process(self, id: str, shards: list[ShardType], size: int) -> None:
        try:
            start = time()
            try:
                await self._process(id, shards)
                self.bytes_written += size

            except Exception as e:
                self._exception = e
                self._inputs_done = True
            stop = time()

            self.diagnostics["avg_size"] = (
                0.98 * self.diagnostics["avg_size"] + 0.02 * size
            )
            self.diagnostics["avg_duration"] = 0.98 * self.diagnostics[
                "avg_duration"
            ] + 0.02 * (stop - start)
        finally:
            await self.memory_limiter.decrease(size)
            self.bytes_memory -= size

    async def _process(self, id: str, shards: list[ShardType]) -> None:
        raise NotImplementedError()

    def read(self, id: str) -> ShardType:
        raise NotImplementedError()

    @property
    def empty(self) -> bool:
        return not self.shards

    async def _background_task(self) -> None:
        def _continue() -> bool:
            return bool(self.shards or self._inputs_done)

        while True:
            async with self._shards_available:
                await self._shards_available.wait_for(_continue)
                if self._inputs_done and not self.shards:
                    break
                part_id = max(self.sizes, key=self.sizes.__getitem__)
                if self.max_message_size > 0:
                    size = 0
                    shards = []
                    while size < self.max_message_size:
                        try:
                            shard = self.shards[part_id].pop()
                            shards.append(shard)
                            s = self.sizes_detail[part_id].pop()
                            size += s
                            self.sizes[part_id] -= s
                        except IndexError:
                            break
                        finally:
                            if not self.shards[part_id]:
                                del self.shards[part_id]
                                assert not self.sizes[part_id]
                                del self.sizes[part_id]
                                assert not self.sizes_detail[part_id]
                                del self.sizes_detail[part_id]
                else:
                    shards = self.shards.pop(part_id)
                    size = self.sizes.pop(part_id)
                self._shards_available.notify_all()
            await self.process(part_id, shards, size)

    async def write(self, data: dict[str, ShardType]) -> None:
        """
        Writes objects into the local buffers, blocks until ready for more

        Parameters
        ----------
        data: dict
            A dictionary mapping destinations to the object that should
            be written to that destination

        Notes
        -----
        If this buffer has a memory limiter configured, then it will
        apply back-pressure to the sender (blocking further receives)
        if local resource usage hits the limit, until such time as the
        resource usage drops.

        """

        if self._exception:
            raise self._exception
        if not self._accepts_input or self._inputs_done:
            raise RuntimeError(f"Trying to put data in closed {self}.")

        if not data:
            return

        sizes = {worker: sizeof(shard) for worker, shard in data.items()}
        total_batch_size = sum(sizes.values())
        self.bytes_memory += total_batch_size
        self.bytes_total += total_batch_size

        self.memory_limiter.increase(total_batch_size)
        async with self._shards_available:
            for worker, shard in data.items():
                self.shards[worker].append(shard)
                self.sizes_detail[worker].append(sizes[worker])
                self.sizes[worker] += sizes[worker]
            self._shards_available.notify()
        await self.memory_limiter.wait_for_available()
        del data
        assert total_batch_size

    def raise_on_exception(self) -> None:
        """Raises an exception if something went wrong during writing"""
        if self._exception:
            raise self._exception

    async def flush(self) -> None:
        """Wait until all writes are finished.

        This closes the buffer such that no new writes are allowed
        """
        async with self._flush_lock:
            self._accepts_input = False
            async with self._shards_available:
                self._shards_available.notify_all()
                await self._shards_available.wait_for(
                    lambda: not self.shards or self._exception or self._inputs_done
                )
                self._inputs_done = True
                self._shards_available.notify_all()

            await asyncio.gather(*self._tasks)
            if not self._exception:
                assert not self.bytes_memory, (type(self), self.bytes_memory)

    async def close(self) -> None:
        """Flush and close the buffer.

        This cleans up all allocated resources.
        """
        await self.flush()
        if not self._exception:
            assert not self.bytes_memory, (type(self), self.bytes_memory)
        for t in self._tasks:
            t.cancel()
        self._accepts_input = False
        self._inputs_done = True
        self.shards.clear()
        self.bytes_memory = 0
        async with self._shards_available:
            self._shards_available.notify_all()
        await asyncio.gather(*self._tasks)

    async def __aenter__(self) -> ShardsBuffer:
        return self

    async def __aexit__(self, exc: Any, typ: Any, traceback: Any) -> None:
        await self.close()

    @contextlib.contextmanager
    def time(self, name: str) -> Iterator[None]:
        start = time()
        yield
        stop = time()
        self.diagnostics[name] += stop - start


class AsyncShardsBuffer(Generic[ShardType]):
    """A buffer for P2P shuffle

    The objects to buffer are typically bytes belonging to certain shards.
    Typically the buffer is implemented on sending and receiving end.

    The buffer allows for concurrent writing and buffers shards to reduce overhead of writing.

    The shards are typically provided in a format like::

        {
            "bucket-0": [b"shard1", b"shard2"],
            "bucket-1": [b"shard1", b"shard2"],
        }

    Buckets typically correspond to output partitions.

    If exceptions occur during writing, the buffer is automatically closed. Subsequent attempts to write will raise the same exception.
    Flushing will not raise an exception. To ensure that the buffer finished successfully, please call `ShardsBuffer.raise_on_exception`
    """

    shards: defaultdict[str, _List[ShardType]]
    sizes: defaultdict[str, int]
    memory_limiter: ResourceLimiter
    diagnostics: dict[str, float]
    max_message_size: int

    bytes_total: int
    bytes_memory: int
    bytes_written: int
    bytes_read: int

    _state: BufferState
    _exception: Exception | None
    _active_flushes: int
    _flush_condition: asyncio.Condition
    # FIXME: This is ugly but it is used to avoid flushing the same key multiple times without
    # blocking adding to it
    _flushing_sizes: dict[str, int]

    def __init__(
        self,
        memory_limiter: ResourceLimiter,
        max_message_size: int = -1,
    ) -> None:
        self.shards = defaultdict(_List)
        self.sizes = defaultdict(int)
        self.memory_limiter = memory_limiter
        self.diagnostics: dict[str, float] = defaultdict(float)
        self._flush_condition = asyncio.Condition()
        self._state = "open"
        self._active_flushes = 0
        self._exception = None
        self.max_message_size = max_message_size

        self.bytes_total = 0
        self.bytes_memory = 0
        self.bytes_written = 0
        self.bytes_read = 0
        self._flushing_sizes = {}

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

    async def _write(self, id: str, shards: list[ShardType]) -> int:
        raise NotImplementedError()

    @property
    def empty(self) -> bool:
        return not self.shards

    async def write(self, data: dict[str, tuple[ShardType, int]]) -> None:
        """
        Writes objects into the local buffers, blocks until ready for more

        Parameters
        ----------
        data: dict
            A dictionary mapping destinations to the object that should
            be written to that destination

        Notes
        -----
        If this buffer has a memory limiter configured, then it will
        apply back-pressure to the sender (blocking further receives)
        if local resource usage hits the limit, until such time as the
        resource usage drops.

        """
        self._raise_if_erred()

        if self._state != "open":
            raise RuntimeError(
                f"{self} is no longer open for new data, it is {self._state}."
            )

        if not data:
            return

        for worker, (shard, size) in data.items():
            self.shards[worker].append(shard)
            if worker in self._flushing_sizes:
                self._flushing_sizes[worker] += size
            else:
                self.sizes[worker] += size
            self.bytes_memory += size
            self.bytes_total += size
            self.memory_limiter.increase(size)
        del data

        while self.memory_limiter.full and self.sizes:
            await self.flush_largest()
        await self.memory_limiter.wait_for_available()

    def _raise_if_erred(self) -> None:
        if self._state == "erred":
            assert self._exception
            raise self._exception

    async def flush_largest(self) -> None:
        if not self.sizes:
            return
        largest_key = max(self.sizes, key=self.sizes.__getitem__)
        data = self.shards.pop(largest_key)
        size = self.sizes.pop(largest_key)

        self._flushing_sizes[largest_key] = 0

        bytes_written = 0
        try:
            self._raise_if_erred()
            if self._state not in {"open", "flushing"}:
                raise RuntimeError(
                    f"{self} can no longer flush data, it is {self._state}."
                )
            start = time()
            try:
                self._active_flushes += 1
                bytes_written = await self._write(largest_key, data)
            except Exception as e:
                if not self._state == "erred":
                    self._exception = e
                    self._state = "erred"
                    self.shards.clear()
                    total_size = sum(self.sizes.values())
                    self.sizes.clear()
                    self._flushing_sizes.clear()
                    await self.memory_limiter.decrease(total_size)
            finally:
                async with self._flush_condition:
                    self._active_flushes -= 1
                    self._flush_condition.notify_all()

            stop = time()
            self.diagnostics["avg_size"] = (
                0.98 * self.diagnostics["avg_size"] + 0.02 * size
            )
            self.diagnostics["avg_duration"] = 0.98 * self.diagnostics[
                "avg_duration"
            ] + 0.02 * (stop - start)
        finally:
            await self.memory_limiter.decrease(size)
            self.bytes_memory -= size
            self.bytes_written += bytes_written
            size_since_flush = self._flushing_sizes.pop(largest_key, 0)
            if size_since_flush:
                self.sizes[largest_key] = size_since_flush

    async def flush(self) -> None:
        """Wait until all writes are finished.

        This closes the buffer such that no new writes are allowed
        """
        if self._state in {"flushed", "closed"}:
            return

        if self._state == "flushing":
            async with self._flush_condition:
                await self._flush_condition.wait_for(
                    lambda: self._state in {"erred", "flushed", "closed"}
                )
            self._raise_if_erred()
            return

        self._raise_if_erred()
        assert self._state == "open", self._state
        self._state = "flushing"

        async with self._flush_condition:
            await self._flush_condition.wait_for(lambda: self._active_flushes == 0)
            if self._state == "flushing":
                self._state = "flushed"
            self._flush_condition.notify_all()

        if self._exception:
            raise self._exception

    async def close(self) -> None:
        """Flush and close the buffer.

        This cleans up all allocated resources.
        """
        if self._state == "closed":
            return

        try:
            await self.flush()
        except Exception:
            assert self._state == "erred"
        self._state = "closed"
        self.shards.clear()
        self.bytes_memory = 0

    async def __aenter__(self) -> AsyncShardsBuffer:
        return self

    async def __aexit__(self, exc: Any, typ: Any, traceback: Any) -> None:
        await self.close()

    @contextlib.contextmanager
    def time(self, name: str) -> Iterator[None]:
        start = time()
        yield
        stop = time()
        self.diagnostics[name] += stop - start

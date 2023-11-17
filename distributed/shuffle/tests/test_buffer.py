from __future__ import annotations

import asyncio
import math
from collections import defaultdict

import pytest

from dask.utils import parse_bytes

from distributed.shuffle._buffer import BaseBuffer
from distributed.shuffle._limiter import ResourceLimiter
from distributed.utils import wait_for
from distributed.utils_test import gen_test


def gen_bytes(percentage: float, limit: int) -> bytes:
    num_bytes = int(math.floor(percentage * limit))
    return b"0" * num_bytes


class BufferTest(BaseBuffer):
    def __init__(self, memory_limiter: ResourceLimiter, concurrency_limit: int) -> None:
        self.allow_process = asyncio.Event()
        self.storage: dict[str, bytes] = defaultdict(bytes)
        super().__init__(
            memory_limiter=memory_limiter, concurrency_limit=concurrency_limit
        )

    async def _flush(self, id: str, shards: list[bytes]) -> None:
        await self.allow_process.wait()
        self.storage[id] += b"".join(shards)

    def read(self, id: str) -> bytes:
        return self.storage[id]


limit = parse_bytes("10.0 MiB")


@pytest.mark.parametrize(
    "big_payloads",
    [
        [{"big": [gen_bytes(2, limit)]}],
        [{"big": [gen_bytes(0.5, limit)]}] * 4,
        [{f"big-{ix}": [gen_bytes(0.5, limit)]} for ix in range(4)],
        [{f"big-{ix}": [gen_bytes(0.5, limit)]} for ix in range(2)] * 2,
    ],
)
@gen_test()
async def test_memory_limit(big_payloads):
    small_payload = {"small": [gen_bytes(0.1, limit)]}

    limiter = ResourceLimiter(limit)

    async with BufferTest(
        memory_limiter=limiter,
        concurrency_limit=2,
    ) as buf:
        # It's OK to write nothing
        await buf.write({})

        many_small = [asyncio.create_task(buf.write(small_payload)) for _ in range(9)]
        buf.allow_process.set()
        many_small = asyncio.gather(*many_small)
        # Puts that do not breach the limit do not block
        await many_small
        assert buf.memory_limiter.time_blocked_total == 0
        buf.allow_process.clear()

        many_small = [asyncio.create_task(buf.write(small_payload)) for _ in range(11)]
        assert buf.memory_limiter
        while buf.memory_limiter.available:
            await asyncio.sleep(0.1)

        new_put = asyncio.create_task(buf.write(small_payload))
        with pytest.raises(asyncio.TimeoutError):
            await wait_for(asyncio.shield(new_put), 0.1)
        buf.allow_process.set()
        many_small = asyncio.gather(*many_small)
        await new_put

        while not buf.memory_limiter.empty:
            await asyncio.sleep(0.1)
        buf.allow_process.clear()
        big_tasks = [
            asyncio.create_task(buf.write(big_payload)) for big_payload in big_payloads
        ]
        small = asyncio.create_task(buf.write(small_payload))
        with pytest.raises(asyncio.TimeoutError):
            await wait_for(asyncio.shield(asyncio.gather(*big_tasks)), 0.1)
        with pytest.raises(asyncio.TimeoutError):
            await wait_for(asyncio.shield(small), 0.1)
        # Puts only return once we're below memory limit
        buf.allow_process.set()
        await asyncio.gather(*big_tasks)
        await small
        # Once the big write is through, we can write without blocking again
        before = buf.memory_limiter.time_blocked_total
        await buf.write(small_payload)
        assert before == buf.memory_limiter.time_blocked_total


class BufferShardsBroken(BaseBuffer):
    def __init__(self, memory_limiter: ResourceLimiter, concurrency_limit: int) -> None:
        self.storage: dict[str, bytes] = defaultdict(bytes)
        super().__init__(
            memory_limiter=memory_limiter, concurrency_limit=concurrency_limit
        )

    async def _flush(self, id: str, shards: list[bytes]) -> None:
        if id == "error":
            raise RuntimeError("Error during processing")
        self.storage[id] += b"".join(shards)

    def read(self, id: str) -> bytes:
        return self.storage[id]


@gen_test()
async def test_memory_limit_blocked_exception():
    limit = parse_bytes("10.0 MiB")

    big_payload = {
        "shard-1": [gen_bytes(2, limit)],
    }
    broken_payload = {
        "error": ["not-bytes"],
    }
    limiter = ResourceLimiter(limit)
    async with BufferShardsBroken(
        memory_limiter=limiter,
        concurrency_limit=2,
    ) as mf:
        big_write = asyncio.create_task(mf.write(big_payload))
        small_write = asyncio.create_task(mf.write(broken_payload))
        # The broken write hits the limit and blocks
        await big_write
        await small_write

        # Make sure exception is not dropped
        with pytest.raises(RuntimeError, match="Error during processing"):
            await mf.flush()
        with pytest.raises(RuntimeError, match="Error during processing"):
            mf.raise_if_erred()

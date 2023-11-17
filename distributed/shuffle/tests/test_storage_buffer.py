from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from dask.utils import parse_bytes

from distributed.shuffle._limiter import ResourceLimiter
from distributed.shuffle._storage import StorageBuffer
from distributed.utils_test import gen_test


def write_bytes(data: list[bytes], path: Path) -> int:
    with path.open("ab") as f:
        offset = f.tell()
        f.writelines(data)
        return f.tell() - offset


def read_bytes(path: Path) -> tuple[list[bytes], int]:
    with path.open("rb") as f:
        data = f.read()
        size = f.tell()
    return [data], size


# FIXME: Parametrize all tests here
@pytest.mark.parametrize("memory_limit", ["1 B"])  # , "1 KiB", "1 MiB"])
@gen_test()
async def test_basic(tmp_path, memory_limit):
    with ThreadPoolExecutor(2) as executor:
        async with StorageBuffer(
            directory=tmp_path,
            write=write_bytes,
            read=read_bytes,
            executor=executor,
            memory_limiter=ResourceLimiter(parse_bytes(memory_limit)),
        ) as mf:
            await mf.write({"x": [b"0" * 1000], "y": [b"1" * 500]})
            await mf.write({"x": [b"0" * 1000], "y": [b"1" * 500]})

            await mf.flush()

            x = mf.read("x")
            y = mf.read("y")

            with pytest.raises(KeyError):
                mf.read("z")

            assert x == [b"0" * 2000]
            assert y == [b"1" * 1000]

        assert not os.path.exists(tmp_path)


@gen_test()
async def test_read_before_flush(tmp_path):
    payload = {"1": [b"foo"]}
    with ThreadPoolExecutor(2) as executor:
        async with StorageBuffer(
            directory=tmp_path,
            write=write_bytes,
            read=read_bytes,
            executor=executor,
            memory_limiter=ResourceLimiter(None),
        ) as mf:
            with pytest.raises(RuntimeError):
                mf.read(1)

            await mf.write(payload)

            with pytest.raises(RuntimeError):
                mf.read(1)

            await mf.flush()
            assert mf.read("1") == [b"foo"]
            with pytest.raises(KeyError):
                mf.read(2)


@pytest.mark.parametrize("count", [2, 100, 1000])
@gen_test()
async def test_many(tmp_path, count):
    with ThreadPoolExecutor(2) as executor:
        async with StorageBuffer(
            directory=tmp_path,
            write=write_bytes,
            read=read_bytes,
            executor=executor,
            memory_limiter=ResourceLimiter(1),
        ) as mf:
            d = {i: [str(i).encode() * 100] for i in range(count)}

            for _ in range(10):
                await mf.write(d)

            await mf.flush()

            for i in d:
                out = mf.read(i)
                assert out == [str(i).encode() * 100 * 10]

        assert not os.path.exists(tmp_path)


class BrokenStorageBuffer(StorageBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def _flush(self, *args: Any, **kwargs: Any) -> None:
        raise Exception(123)


@gen_test()
async def test_exceptions(tmp_path):
    with ThreadPoolExecutor(2) as executor:
        async with BrokenStorageBuffer(
            directory=tmp_path,
            write=write_bytes,
            read=read_bytes,
            executor=executor,
            memory_limiter=ResourceLimiter(1),
        ) as mf:
            await mf.write({"x": [b"0" * 1000], "y": [b"1" * 500]})

            while not mf._exception:
                await asyncio.sleep(0.1)

            with pytest.raises(Exception, match="123"):
                await mf.write({"x": [b"0" * 1000], "y": [b"1" * 500]})

            with pytest.raises(Exception, match="123"):
                await mf.flush()


class EventuallyBrokenStorageBuffer(StorageBuffer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0

    async def _flush(self, *args: Any, **kwargs: Any) -> None:
        # We only want to raise if this was queued up before
        if self.counter > self.concurrency_limit:
            raise Exception(123)
        self.counter += 1
        await super()._flush(*args, **kwargs)


@pytest.mark.skip("Partial flush and this test don't work well together")
@gen_test()
async def test_high_pressure_flush_with_exception(tmp_path):
    payload = {f"shard-{ix}": [f"shard-{ix}".encode() * 100] for ix in range(100)}

    with ThreadPoolExecutor(2) as executor:
        async with EventuallyBrokenStorageBuffer(
            directory=tmp_path,
            write=write_bytes,
            read=read_bytes,
            executor=executor,
            memory_limiter=ResourceLimiter(1),
        ) as mf:
            tasks = []
            for _ in range(10):
                tasks.append(asyncio.create_task(mf.write(payload)))

            # Wait until things are actually queued up.
            # This is when there is no slot on the queue available anymore
            # but there are still shards around
            while not mf.shards:
                # Disks are fast, don't give it time to unload the queue...
                # There may only be a few ticks atm so keep this at zero
                await asyncio.sleep(0)

            with pytest.raises(Exception, match="123"):
                await mf.flush()
            with pytest.raises(Exception, match="123"):
                mf.raise_if_erred()

from __future__ import annotations

import asyncio
import math
from collections import defaultdict

import pytest

from dask.utils import parse_bytes

from distributed.shuffle._comms import CommShardsBuffer
from distributed.shuffle._limiter import ResourceLimiter
from distributed.utils_test import gen_test


@gen_test()
async def test_basic(tmp_path):
    d = defaultdict(list)

    async def send(address, shards):
        d[address].extend(shards)

    mc = CommShardsBuffer(
        send=send,
        memory_limiter=ResourceLimiter(None),
        message_bytes_limit=parse_bytes("4 MiB"),
    )
    await mc.write({"x": [b"0" * 1000], "y": [b"1" * 500]})
    await mc.write({"x": [b"0" * 1000], "y": [b"1" * 500]})

    await mc.flush()

    assert b"".join(d["x"]) == b"0" * 2000
    assert b"".join(d["y"]) == b"1" * 1000


@gen_test()
async def test_exceptions(tmp_path):
    d = defaultdict(list)

    async def send(address, shards):
        raise Exception(123)

    mc = CommShardsBuffer(
        send=send,
        memory_limiter=ResourceLimiter(None),
        message_bytes_limit=parse_bytes("4 MiB"),
    )
    await mc.write({"x": [b"0" * 1000], "y": [b"1" * 500]})

    while not mc._exception:
        await asyncio.sleep(0.1)

    with pytest.raises(Exception, match="123"):
        await mc.write({"x": [b"0" * 1000], "y": [b"1" * 500]})

    with pytest.raises(Exception, match="123"):
        await mc.flush()

    with pytest.raises(Exception, match="123"):
        mc.raise_if_erred()

    await mc.close()


@gen_test()
async def test_slow_send(tmp_path):
    block_send = asyncio.Event()
    block_send.set()
    sending_first = asyncio.Event()
    d = defaultdict(list)

    async def send(address, shards):
        await block_send.wait()
        d[address].extend(shards)
        sending_first.set()

    mc = CommShardsBuffer(
        send=send,
        concurrency_limit=1,
        memory_limiter=ResourceLimiter(None),
        message_bytes_limit=parse_bytes("4 MiB"),
    )
    await mc.write({"x": [b"0"], "y": [b"1"]})
    await mc.write({"x": [b"0"], "y": [b"1"]})
    flush_task = asyncio.create_task(mc.flush())
    await sending_first.wait()
    block_send.clear()

    with pytest.raises(RuntimeError):
        await mc.write({"x": [b"2"], "y": [b"2"]})
        await flush_task

    assert [b"2" not in shard for shard in d["x"]]


def gen_bytes(percentage: float, memory_limit: int) -> bytes:
    num_bytes = int(math.floor(percentage * memory_limit))
    return b"0" * num_bytes


@gen_test()
async def test_concurrent_puts():
    d = defaultdict(list)

    async def send(address, shards):
        d[address].extend(shards)

    frac = 0.1
    nshards = 10
    nputs = 20
    comm_buffer = CommShardsBuffer(
        send=send,
        memory_limiter=ResourceLimiter(parse_bytes("1 MiB")),
        message_bytes_limit=parse_bytes("4 KiB"),
    )
    payload = {
        str(x): [gen_bytes(frac, comm_buffer.memory_limiter.limit)]
        for x in range(nshards)
    }

    async with comm_buffer as mc:
        futs = [asyncio.create_task(mc.write(payload)) for _ in range(nputs)]

        await asyncio.gather(*futs)
        await mc.flush()

        assert not mc.shards
        assert not mc.flushable_sizes
        assert not mc.flushing_sizes

    assert not mc.shards
    assert not mc.flushable_sizes
    assert not mc.flushing_sizes
    assert len(d) == 10
    assert (
        sum(map(len, d["0"]))
        == len(gen_bytes(frac, comm_buffer.memory_limiter.limit)) * nputs
    )


@gen_test()
async def test_concurrent_puts_error():
    d = defaultdict(list)

    counter = 0

    async def send(address, shards):
        nonlocal counter
        counter += 1
        if counter == 5:
            raise OSError("error during send")
        d[address].extend(shards)

    frac = 0.1
    nshards = 10
    nputs = 20
    comm_buffer = CommShardsBuffer(
        send=send,
        memory_limiter=ResourceLimiter(parse_bytes("100 MiB")),
        message_bytes_limit=parse_bytes("4 MiB"),
    )
    payload = {
        str(x): [gen_bytes(frac, comm_buffer.memory_limiter.limit)]
        for x in range(nshards)
    }

    async with comm_buffer as mc:
        futs = [asyncio.create_task(mc.write(payload)) for _ in range(nputs)]
        await asyncio.gather(*futs)
        with pytest.raises(OSError, match="error during send"):
            await mc.flush()
        with pytest.raises(OSError, match="error during send"):
            mc.raise_if_erred()

    assert not mc.shards
    assert not mc.flushable_sizes
    assert not mc.flushing_sizes

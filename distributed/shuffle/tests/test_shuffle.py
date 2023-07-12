from __future__ import annotations

import asyncio
import io
import itertools
import os
import random
import shutil
from collections import defaultdict
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack
from itertools import count
from typing import Any
from unittest import mock

import pytest

pd = pytest.importorskip("pandas")
dd = pytest.importorskip("dask.dataframe")

import dask
from dask.distributed import Event, Nanny, Worker
from dask.utils import stringify

from distributed.client import Client
from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.scheduler import Scheduler
from distributed.scheduler import TaskState as SchedulerTaskState
from distributed.shuffle._arrow import serialize_table
from distributed.shuffle._limiter import ResourceLimiter
from distributed.shuffle._scheduler_extension import (
    ShuffleSchedulerExtension,
    get_worker_for_range_sharding,
)
from distributed.shuffle._shuffle import ShuffleId, barrier_key
from distributed.shuffle._worker_extension import (
    DataFrameShuffleRun,
    ShuffleRun,
    ShuffleWorkerExtension,
    convert_partition,
    split_by_partition,
    split_by_worker,
)
from distributed.shuffle.tests.utils import (
    AbstractShuffleTestPool,
    invoke_annotation_chaos,
)
from distributed.utils import Deadline
from distributed.utils_test import (
    cluster,
    gen_cluster,
    gen_test,
    raises_with_cause,
    wait_for_state,
)
from distributed.worker_state_machine import TaskState as WorkerTaskState

try:
    import pyarrow as pa
except ImportError:
    pa = None


@pytest.fixture(params=[0, 0.3, 1], ids=["none", "some", "all"])
def lose_annotations(request):
    return request.param


async def clean_worker(
    worker: Worker, interval: float = 0.01, timeout: int | None = None
) -> None:
    """Assert that the worker has no shuffle state"""
    deadline = Deadline.after(timeout)
    extension = worker.extensions["shuffle"]

    while extension._runs and not deadline.expired:
        await asyncio.sleep(interval)
    for dirpath, dirnames, filenames in os.walk(worker.local_directory):
        assert "shuffle" not in dirpath
        for fn in dirnames + filenames:
            assert "shuffle" not in fn


async def clean_scheduler(
    scheduler: Scheduler, interval: float = 0.01, timeout: int | None = None
) -> None:
    """Assert that the scheduler has no shuffle state"""
    deadline = Deadline.after(timeout)
    extension = scheduler.extensions["shuffle"]
    while extension.states and not deadline.expired:
        await asyncio.sleep(interval)
    assert not extension.states
    assert not extension.heartbeats


@pytest.mark.skipif(
    pa is not None,
    reason="We don't have a CI job that is installing a very old pyarrow version",
)
@gen_cluster(client=True)
async def test_minimal_version(c, s, a, b):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="10 s",
    )
    with pytest.raises(RuntimeError, match="requires pyarrow"):
        await c.compute(dd.shuffle.shuffle(df, "x", shuffle="p2p"))


@pytest.mark.parametrize("npartitions", [None, 1, 20])
@gen_cluster(client=True)
async def test_basic_integration(c, s, a, b, lose_annotations, npartitions):
    await invoke_annotation_chaos(lose_annotations, c)
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="10 s",
    )
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p", npartitions=npartitions)
    if npartitions is None:
        assert out.npartitions == df.npartitions
    else:
        assert out.npartitions == npartitions
    x, y = c.compute([df.x.size, out.x.size])
    x = await x
    y = await y
    assert x == y

    await clean_worker(a)
    await clean_worker(b)
    await clean_scheduler(s)


@pytest.mark.parametrize("npartitions", [None, 1, 20])
@gen_cluster(client=True)
async def test_shuffle_with_array_conversion(c, s, a, b, lose_annotations, npartitions):
    await invoke_annotation_chaos(lose_annotations, c)
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="10 s",
    )
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p", npartitions=npartitions).values

    if npartitions == 1:
        # FIXME: distributed#7816
        with raises_with_cause(
            RuntimeError, "shuffle_transfer failed", RuntimeError, "Barrier task"
        ):
            await c.compute(out)
    else:
        await c.compute(out)

    await clean_worker(a)
    await clean_worker(b)
    await clean_scheduler(s)


def test_shuffle_before_categorize(loop_in_thread):
    """Regression test for https://github.com/dask/distributed/issues/7615"""
    with cluster() as (s, [a, b]), Client(s["address"], loop=loop_in_thread) as c:
        df = dask.datasets.timeseries(
            start="2000-01-01",
            end="2000-01-10",
            dtypes={"x": float, "y": str},
            freq="10 s",
        )
        df = dd.shuffle.shuffle(df, "x", shuffle="p2p")
        df.categorize(columns=["y"])
        c.compute(df)


@gen_cluster(client=True)
async def test_concurrent(c, s, a, b, lose_annotations):
    await invoke_annotation_chaos(lose_annotations, c)
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="10 s",
    )
    x = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    y = dd.shuffle.shuffle(df, "y", shuffle="p2p")
    x, y = c.compute([x.x.size, y.y.size])
    x = await x
    y = await y
    assert x == y

    await clean_worker(a)
    await clean_worker(b)
    await clean_scheduler(s)


@gen_cluster(client=True)
async def test_bad_disk(c, s, a, b):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="10 s",
    )
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    out = out.persist()
    shuffle_id = await wait_until_new_shuffle_is_initialized(s)
    while not a.extensions["shuffle"].shuffles:
        await asyncio.sleep(0.01)
    shutil.rmtree(a.local_directory)

    while not b.extensions["shuffle"].shuffles:
        await asyncio.sleep(0.01)
    shutil.rmtree(b.local_directory)
    with pytest.raises(RuntimeError, match=f"shuffle_transfer failed .* {shuffle_id}"):
        out = await c.compute(out)

    await c.close()
    await clean_worker(a)
    await clean_worker(b)
    await clean_scheduler(s)


async def wait_until_worker_has_tasks(
    prefix: str, worker: str, count: int, scheduler: Scheduler, interval: float = 0.01
) -> None:
    ws = scheduler.workers[worker]
    while (
        len(
            [
                key
                for key, ts in scheduler.tasks.items()
                if prefix in key and ts.state == "memory" and {ws} == ts.who_has
            ]
        )
        < count
    ):
        await asyncio.sleep(interval)


async def wait_for_tasks_in_state(
    prefix: str,
    state: str,
    count: int,
    dask_worker: Worker | Scheduler,
    interval: float = 0.01,
) -> None:
    tasks: Mapping[str, SchedulerTaskState | WorkerTaskState]

    if isinstance(dask_worker, Worker):
        tasks = dask_worker.state.tasks
    elif isinstance(dask_worker, Scheduler):
        tasks = dask_worker.tasks
    else:
        raise TypeError(dask_worker)

    while (
        len([key for key, ts in tasks.items() if prefix in key and ts.state == state])
        < count
    ):
        await asyncio.sleep(interval)


async def wait_until_new_shuffle_is_initialized(
    scheduler: Scheduler, interval: float = 0.01, timeout: int | None = None
) -> ShuffleId:
    deadline = Deadline.after(timeout)
    scheduler_extension = scheduler.extensions["shuffle"]
    while not scheduler_extension.shuffle_ids() and not deadline.expired:
        await asyncio.sleep(interval)
    shuffle_ids = scheduler_extension.shuffle_ids()
    assert len(shuffle_ids) == 1
    return next(iter(shuffle_ids))


@gen_cluster(client=True, nthreads=[("", 1)] * 2)
async def test_closed_worker_during_transfer(c, s, a, b):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-03-01",
        dtypes={"x": float, "y": float},
        freq="10 s",
    )
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    out = out.persist()
    await wait_for_tasks_in_state("shuffle-transfer", "memory", 1, b)
    await b.close()

    with pytest.raises(RuntimeError):
        out = await c.compute(out)

    await c.close()
    await clean_worker(a)
    await clean_worker(b)
    await clean_scheduler(s)


@pytest.mark.slow
@gen_cluster(client=True, nthreads=[("", 1)])
async def test_crashed_worker_during_transfer(c, s, a):
    async with Nanny(s.address, nthreads=1) as n:
        killed_worker_address = n.worker_address
        df = dask.datasets.timeseries(
            start="2000-01-01",
            end="2000-03-01",
            dtypes={"x": float, "y": float},
            freq="10 s",
        )
        out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
        out = out.persist()
        await wait_until_worker_has_tasks(
            "shuffle-transfer", killed_worker_address, 1, s
        )
        await n.process.process.kill()

        with pytest.raises(RuntimeError):
            out = await c.compute(out)

        await c.close()
        await clean_worker(a)
        await clean_scheduler(s)


# TODO: Deduplicate instead of failing: distributed#7324
@gen_cluster(client=True, nthreads=[("", 1)] * 2)
async def test_closed_input_only_worker_during_transfer(c, s, a, b):
    def mock_get_worker_for_range_sharding(
        output_partition: int, workers: list[str], npartitions: int
    ) -> str:
        return a.address

    with mock.patch(
        "distributed.shuffle._scheduler_extension.get_worker_for_range_sharding",
        mock_get_worker_for_range_sharding,
    ):
        df = dask.datasets.timeseries(
            start="2000-01-01",
            end="2000-05-01",
            dtypes={"x": float, "y": float},
            freq="10 s",
        )
        out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
        out = out.persist()
        await wait_for_tasks_in_state("shuffle-transfer", "memory", 1, b, 0.001)
        await b.close()

        with pytest.raises(RuntimeError):
            out = await c.compute(out)

        await c.close()
        await clean_worker(a)
        await clean_worker(b)
        await clean_scheduler(s)


# TODO: Deduplicate instead of failing: distributed#7324
@pytest.mark.slow
@gen_cluster(client=True, nthreads=[("", 1)], clean_kwargs={"processes": False})
async def test_crashed_input_only_worker_during_transfer(c, s, a):
    def mock_mock_get_worker_for_range_sharding(
        output_partition: int, workers: list[str], npartitions: int
    ) -> str:
        return a.address

    with mock.patch(
        "distributed.shuffle._scheduler_extension.get_worker_for_range_sharding",
        mock_mock_get_worker_for_range_sharding,
    ):
        async with Nanny(s.address, nthreads=1) as n:
            killed_worker_address = n.worker_address
            df = dask.datasets.timeseries(
                start="2000-01-01",
                end="2000-03-01",
                dtypes={"x": float, "y": float},
                freq="10 s",
            )
            out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
            out = out.persist()
            await wait_until_worker_has_tasks(
                "shuffle-transfer", n.worker_address, 1, s
            )
            await n.process.process.kill()

            with pytest.raises(RuntimeError):
                out = await c.compute(out)

            await c.close()
            await clean_worker(a)
            await clean_scheduler(s)


@pytest.mark.slow
@gen_cluster(client=True, nthreads=[("", 1)] * 3)
async def test_closed_bystanding_worker_during_shuffle(c, s, w1, w2, w3):
    with dask.annotate(workers=[w1.address, w2.address], allow_other_workers=False):
        df = dask.datasets.timeseries(
            start="2000-01-01",
            end="2000-02-01",
            dtypes={"x": float, "y": float},
            freq="10 s",
        )
        out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
        out = out.persist()
    await wait_for_tasks_in_state("shuffle-transfer", "memory", 1, w1)
    await wait_for_tasks_in_state("shuffle-transfer", "memory", 1, w2)
    await w3.close()

    await c.compute(out)
    del out

    await clean_worker(w1)
    await clean_worker(w2)
    await clean_worker(w3)
    await clean_scheduler(s)


class BlockedInputsDoneShuffle(DataFrameShuffleRun):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_inputs_done = asyncio.Event()
        self.block_inputs_done = asyncio.Event()

    async def inputs_done(self) -> None:
        self.in_inputs_done.set()
        await self.block_inputs_done.wait()
        await super().inputs_done()


@mock.patch(
    "distributed.shuffle._worker_extension.DataFrameShuffleRun",
    BlockedInputsDoneShuffle,
)
@gen_cluster(client=True, nthreads=[("", 1)] * 2)
async def test_closed_worker_during_barrier(c, s, a, b):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="10 s",
    )
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    out = out.persist()
    shuffle_id = await wait_until_new_shuffle_is_initialized(s)
    key = barrier_key(shuffle_id)
    await wait_for_state(key, "processing", s)
    shuffleA = a.extensions["shuffle"].shuffles[shuffle_id]
    shuffleB = b.extensions["shuffle"].shuffles[shuffle_id]
    await shuffleA.in_inputs_done.wait()
    await shuffleB.in_inputs_done.wait()

    ts = s.tasks[key]
    processing_worker = a if ts.processing_on.address == a.address else b
    if processing_worker == a:
        close_worker = a
        alive_shuffle = shuffleB

    else:
        close_worker, alive_worker = b, a
        alive_shuffle = shuffleA
    await close_worker.close()

    alive_shuffle.block_inputs_done.set()

    with pytest.raises(RuntimeError):
        out = await c.compute(out)

    await c.close()
    await clean_worker(a)
    await clean_worker(b)
    await clean_scheduler(s)


@mock.patch(
    "distributed.shuffle._worker_extension.DataFrameShuffleRun",
    BlockedInputsDoneShuffle,
)
@gen_cluster(client=True, nthreads=[("", 1)] * 2)
async def test_closed_other_worker_during_barrier(c, s, a, b):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="10 s",
    )
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    out = out.persist()
    shuffle_id = await wait_until_new_shuffle_is_initialized(s)

    key = barrier_key(shuffle_id)
    await wait_for_state(key, "processing", s, interval=0)

    shuffleA = a.extensions["shuffle"].shuffles[shuffle_id]
    shuffleB = b.extensions["shuffle"].shuffles[shuffle_id]
    await shuffleA.in_inputs_done.wait()
    await shuffleB.in_inputs_done.wait()

    ts = s.tasks[key]
    processing_worker = a if ts.processing_on.address == a.address else b
    if processing_worker == a:
        close_worker = b
        alive_shuffle = shuffleA

    else:
        close_worker = a
        alive_shuffle = shuffleB
    await close_worker.close()

    alive_shuffle.block_inputs_done.set()

    with pytest.raises(RuntimeError, match="shuffle_barrier failed"):
        out = await c.compute(out)

    await c.close()
    await clean_worker(a)
    await clean_worker(b)
    await clean_scheduler(s)


@pytest.mark.slow
@mock.patch(
    "distributed.shuffle._worker_extension.DataFrameShuffleRun",
    BlockedInputsDoneShuffle,
)
@gen_cluster(client=True, nthreads=[("", 1)])
async def test_crashed_other_worker_during_barrier(c, s, a):
    async with Nanny(s.address, nthreads=1) as n:
        df = dask.datasets.timeseries(
            start="2000-01-01",
            end="2000-01-10",
            dtypes={"x": float, "y": float},
            freq="10 s",
        )
        out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
        out = out.persist()
        shuffle_id = await wait_until_new_shuffle_is_initialized(s)
        key = barrier_key(shuffle_id)
        # Ensure that barrier is not executed on the nanny
        s.set_restrictions({key: {a.address}})
        await wait_for_state(key, "processing", s, interval=0)
        shuffle = a.extensions["shuffle"].shuffles[shuffle_id]
        await shuffle.in_inputs_done.wait()
        await n.process.process.kill()
        shuffle.block_inputs_done.set()

        with pytest.raises(RuntimeError, match="shuffle"):
            out = await c.compute(out)

        await c.close()
        await clean_worker(a)
        await clean_scheduler(s)


@gen_cluster(client=True, nthreads=[("", 1)] * 2)
async def test_closed_worker_during_unpack(c, s, a, b):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-03-01",
        dtypes={"x": float, "y": float},
        freq="10 s",
    )
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    out = out.persist()
    await wait_for_tasks_in_state("shuffle-p2p", "memory", 1, b)
    await b.close()

    with pytest.raises(RuntimeError):
        out = await c.compute(out)

    await c.close()
    await clean_worker(a)
    await clean_worker(b)
    await clean_scheduler(s)


@pytest.mark.slow
@gen_cluster(client=True, nthreads=[("", 1)])
async def test_crashed_worker_during_unpack(c, s, a):
    async with Nanny(s.address, nthreads=2) as n:
        killed_worker_address = n.worker_address
        df = dask.datasets.timeseries(
            start="2000-01-01",
            end="2000-03-01",
            dtypes={"x": float, "y": float},
            freq="10 s",
        )
        out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
        out = out.persist()
        await wait_until_worker_has_tasks("shuffle-p2p", killed_worker_address, 1, s)
        await n.process.process.kill()
        with pytest.raises(
            RuntimeError,
        ):
            out = await c.compute(out)

        await c.close()
        await clean_worker(a)
        await clean_scheduler(s)


@gen_cluster(client=True)
async def test_heartbeat(c, s, a, b):
    await a.heartbeat()
    await clean_scheduler(s)
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="10 s",
    )
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    out = out.persist()

    while not s.extensions["shuffle"].heartbeats:
        await asyncio.sleep(0.001)
        await a.heartbeat()

    assert s.extensions["shuffle"].heartbeats.values()
    await out

    await clean_worker(a)
    await clean_worker(b)
    del out
    await clean_scheduler(s)


def test_processing_chain():
    """
    This is a serial version of the entire compute chain

    In practice this takes place on many different workers.
    Here we verify its accuracy in a single threaded situation.
    """
    np = pytest.importorskip("numpy")
    pa = pytest.importorskip("pyarrow")

    class Stub:
        def __init__(self, value: int) -> None:
            self.value = value

    counter = count()
    workers = ["a", "b", "c"]
    npartitions = 5

    # Test the processing chain with a dataframe that contains all supported dtypes
    df = pd.DataFrame(
        {
            # numpy dtypes
            f"col{next(counter)}": pd.array([True, False] * 50, dtype="bool"),
            f"col{next(counter)}": pd.array(range(100), dtype="int8"),
            f"col{next(counter)}": pd.array(range(100), dtype="int16"),
            f"col{next(counter)}": pd.array(range(100), dtype="int32"),
            f"col{next(counter)}": pd.array(range(100), dtype="int64"),
            f"col{next(counter)}": pd.array(range(100), dtype="uint8"),
            f"col{next(counter)}": pd.array(range(100), dtype="uint16"),
            f"col{next(counter)}": pd.array(range(100), dtype="uint32"),
            f"col{next(counter)}": pd.array(range(100), dtype="uint64"),
            f"col{next(counter)}": pd.array(range(100), dtype="float16"),
            f"col{next(counter)}": pd.array(range(100), dtype="float32"),
            f"col{next(counter)}": pd.array(range(100), dtype="float64"),
            f"col{next(counter)}": pd.array(
                [np.datetime64("2022-01-01") + i for i in range(100)],
                dtype="datetime64[ns]",
            ),
            f"col{next(counter)}": pd.array(
                [np.timedelta64(1, "D") + i for i in range(100)],
                dtype="timedelta64[ns]",
            ),
            # FIXME: PyArrow does not support complex numbers: https://issues.apache.org/jira/browse/ARROW-638
            # f"col{next(counter)}": pd.array(range(100), dtype="csingle"),
            # f"col{next(counter)}": pd.array(range(100), dtype="cdouble"),
            # f"col{next(counter)}": pd.array(range(100), dtype="clongdouble"),
            # Nullable dtypes
            f"col{next(counter)}": pd.array([True, False] * 50, dtype="boolean"),
            f"col{next(counter)}": pd.array(range(100), dtype="Int8"),
            f"col{next(counter)}": pd.array(range(100), dtype="Int16"),
            f"col{next(counter)}": pd.array(range(100), dtype="Int32"),
            f"col{next(counter)}": pd.array(range(100), dtype="Int64"),
            f"col{next(counter)}": pd.array(range(100), dtype="UInt8"),
            f"col{next(counter)}": pd.array(range(100), dtype="UInt16"),
            f"col{next(counter)}": pd.array(range(100), dtype="UInt32"),
            f"col{next(counter)}": pd.array(range(100), dtype="UInt64"),
            # pandas dtypes
            f"col{next(counter)}": pd.array(
                [np.datetime64("2022-01-01") + i for i in range(100)],
                dtype=pd.DatetimeTZDtype(tz="Europe/Berlin"),
            ),
            f"col{next(counter)}": pd.array(
                [pd.Period("2022-01-01", freq="D") + i for i in range(100)],
                dtype="period[D]",
            ),
            f"col{next(counter)}": pd.array(
                [pd.Interval(left=i, right=i + 2) for i in range(100)], dtype="Interval"
            ),
            f"col{next(counter)}": pd.array(["x", "y"] * 50, dtype="category"),
            f"col{next(counter)}": pd.array(["lorem ipsum"] * 100, dtype="string"),
            # FIXME: PyArrow does not support sparse data: https://issues.apache.org/jira/browse/ARROW-8679
            # f"col{next(counter)}": pd.array(
            #     [np.nan, np.nan, 1.0, np.nan, np.nan] * 20,
            #     dtype="Sparse[float64]",
            # ),
            # PyArrow dtypes
            f"col{next(counter)}": pd.array([True, False] * 50, dtype="bool[pyarrow]"),
            f"col{next(counter)}": pd.array(range(100), dtype="int8[pyarrow]"),
            f"col{next(counter)}": pd.array(range(100), dtype="int16[pyarrow]"),
            f"col{next(counter)}": pd.array(range(100), dtype="int32[pyarrow]"),
            f"col{next(counter)}": pd.array(range(100), dtype="int64[pyarrow]"),
            f"col{next(counter)}": pd.array(range(100), dtype="uint8[pyarrow]"),
            f"col{next(counter)}": pd.array(range(100), dtype="uint16[pyarrow]"),
            f"col{next(counter)}": pd.array(range(100), dtype="uint32[pyarrow]"),
            f"col{next(counter)}": pd.array(range(100), dtype="uint64[pyarrow]"),
            f"col{next(counter)}": pd.array(range(100), dtype="float32[pyarrow]"),
            f"col{next(counter)}": pd.array(range(100), dtype="float64[pyarrow]"),
            f"col{next(counter)}": pd.array(
                [pd.Timestamp.fromtimestamp(1641034800 + i) for i in range(100)],
                dtype=pd.ArrowDtype(pa.timestamp("ms")),
            ),
            f"col{next(counter)}": pd.array(
                ["lorem ipsum"] * 100,
                dtype="string[pyarrow]",
            ),
            f"col{next(counter)}": pd.array(
                ["lorem ipsum"] * 100,
                dtype=pd.StringDtype("pyarrow"),
            ),
            f"col{next(counter)}": pd.array(
                ["lorem ipsum"] * 100,
                dtype="string[python]",
            ),
            # custom objects
            # FIXME: Serializing custom objects is not supported in P2P shuffling
            # f"col{next(counter)}": pd.array(
            #     [Stub(i) for i in range(100)], dtype="object"
            # ),
        }
    )
    df["_partitions"] = df.col4 % npartitions
    worker_for = {i: random.choice(workers) for i in list(range(npartitions))}
    worker_for = pd.Series(worker_for, name="_worker").astype("category")

    data = split_by_worker(df, "_partitions", worker_for=worker_for)
    assert set(data) == set(worker_for.cat.categories)
    assert sum(map(len, data.values())) == len(df)

    batches = {worker: [serialize_table(t)] for worker, t in data.items()}

    # Typically we communicate to different workers at this stage
    # We then receive them back and reconstute them

    by_worker = {
        worker: list_of_buffers_to_table(list_of_batches)
        for worker, list_of_batches in batches.items()
    }
    assert sum(map(len, by_worker.values())) == len(df)

    # We split them again, and then dump them down to disk

    splits_by_worker = {
        worker: split_by_partition(t, "_partitions") for worker, t in by_worker.items()
    }

    splits_by_worker = {
        worker: {partition: [t] for partition, t in d.items()}
        for worker, d in splits_by_worker.items()
    }

    # No two workers share data from any partition
    assert not any(
        set(a) & set(b)
        for w1, a in splits_by_worker.items()
        for w2, b in splits_by_worker.items()
        if w1 is not w2
    )

    # Our simple file system
    filesystem = defaultdict(io.BytesIO)

    for partitions in splits_by_worker.values():
        for partition, tables in partitions.items():
            for table in tables:
                filesystem[partition].write(serialize_table(table))

    out = {}
    for k, bio in filesystem.items():
        bio.seek(0)
        out[k] = convert_partition(bio.read(), df.head(0))

    shuffled_df = pd.concat(df for df in out.values())
    pd.testing.assert_frame_equal(
        df,
        shuffled_df,
        check_like=True,
        check_exact=True,
    )


@gen_cluster(client=True)
async def test_head(c, s, a, b):
    a_files = list(os.walk(a.local_directory))
    b_files = list(os.walk(b.local_directory))

    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="10 s",
    )
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    out = await out.head(compute=False).persist()  # Only ask for one key

    assert list(os.walk(a.local_directory)) == a_files  # cleaned up files?
    assert list(os.walk(b.local_directory)) == b_files

    await clean_worker(a)
    await clean_worker(b)
    del out
    await clean_scheduler(s)


def test_split_by_worker():
    workers = ["a", "b", "c"]
    npartitions = 5
    df = pd.DataFrame({"x": range(100), "y": range(100)})
    df["_partitions"] = df.x % npartitions
    worker_for = {i: random.choice(workers) for i in range(npartitions)}
    s = pd.Series(worker_for, name="_worker").astype("category")


@gen_cluster(client=True, nthreads=[("", 1)] * 2)
async def test_clean_after_forgotten_early(c, s, a, b):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-03-01",
        dtypes={"x": float, "y": float},
        freq="10 s",
    )
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    out = out.persist()
    await wait_for_tasks_in_state("shuffle-transfer", "memory", 1, a)
    await wait_for_tasks_in_state("shuffle-transfer", "memory", 1, b)
    del out
    await clean_worker(a, timeout=2)
    await clean_worker(b, timeout=2)
    await clean_scheduler(s, timeout=2)


@gen_cluster(client=True)
async def test_tail(c, s, a, b):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="1 s",
    )
    x = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    full = await x.persist()
    ntasks_full = len(s.tasks)
    del full
    while s.tasks:
        await asyncio.sleep(0)
    partial = await x.tail(compute=False).persist()  # Only ask for one key

    assert len(s.tasks) < ntasks_full
    del partial

    await clean_worker(a)
    await clean_worker(b)
    await clean_scheduler(s)


@pytest.mark.parametrize("wait_until_forgotten", [True, False])
@gen_cluster(client=True)
async def test_repeat_shuffle_instance(c, s, a, b, wait_until_forgotten):
    """Tests repeating the same instance of a shuffle-based task graph.

    See Also
    --------
    test_repeat_shuffle_operation
    """
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="100 s",
    )
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p").size
    await c.compute(out)

    if wait_until_forgotten:
        while s.tasks:
            await asyncio.sleep(0)

    await c.compute(out)

    await clean_worker(a, timeout=2)
    await clean_worker(b, timeout=2)
    await clean_scheduler(s, timeout=2)


@pytest.mark.parametrize("wait_until_forgotten", [True, False])
@gen_cluster(client=True)
async def test_repeat_shuffle_operation(c, s, a, b, wait_until_forgotten):
    """Tests repeating the same shuffle operation using two distinct instances of the
    task graph.

    See Also
    --------
    test_repeat_shuffle_instance
    """
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="100 s",
    )
    await c.compute(dd.shuffle.shuffle(df, "x", shuffle="p2p"))

    if wait_until_forgotten:
        while s.tasks:
            await asyncio.sleep(0)

    await c.compute(dd.shuffle.shuffle(df, "x", shuffle="p2p"))

    await clean_worker(a, timeout=2)
    await clean_worker(b, timeout=2)
    await clean_scheduler(s, timeout=2)


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_crashed_worker_after_shuffle(c, s, a):
    in_event = Event()
    block_event = Event()

    @dask.delayed
    def block(df, in_event, block_event):
        in_event.set()
        block_event.wait()
        return df

    async with Nanny(s.address, nthreads=1) as n:
        df = df = dask.datasets.timeseries(
            start="2000-01-01",
            end="2000-03-01",
            dtypes={"x": float, "y": float},
            freq="100 s",
            seed=42,
        )
        out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
        in_event = Event()
        block_event = Event()
        with dask.annotate(workers=[n.worker_address], allow_other_workers=True):
            out = block(out, in_event, block_event)
        out = c.compute(out)

        await wait_until_worker_has_tasks("shuffle-p2p", n.worker_address, 1, s)
        await in_event.wait()
        await n.process.process.kill()
        await block_event.set()

        out = await out
        result = out.x.size
        expected = await c.compute(df.x.size)
        assert result == expected

        await c.close()
        await clean_worker(a)
        await clean_scheduler(s)


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_crashed_worker_after_shuffle_persisted(c, s, a):
    async with Nanny(s.address, nthreads=1) as n:
        df = df = dask.datasets.timeseries(
            start="2000-01-01",
            end="2000-01-10",
            dtypes={"x": float, "y": float},
            freq="10 s",
            seed=42,
        )
        out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
        out = out.persist()

        await wait_until_worker_has_tasks("shuffle-p2p", n.worker_address, 1, s)
        await out

        await n.process.process.kill()

        result, expected = c.compute([out.x.size, df.x.size])
        result = await result
        expected = await expected
        assert result == expected

        await c.close()
        await clean_worker(a)
        await clean_scheduler(s)


@gen_cluster(client=True, nthreads=[("", 1)] * 3)
async def test_closed_worker_between_repeats(c, s, w1, w2, w3):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="100 s",
        seed=42,
    )
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    await c.compute(out.head(compute=False))

    await clean_worker(w1)
    await clean_worker(w2)
    await clean_worker(w3)
    await clean_scheduler(s)

    await w3.close()
    await c.compute(out.tail(compute=False))

    await clean_worker(w1)
    await clean_worker(w2)
    await clean_scheduler(s)

    await w2.close()
    await c.compute(out.head(compute=False))
    await clean_worker(w1)
    await clean_scheduler(s)


@gen_cluster(client=True)
async def test_new_worker(c, s, a, b):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-20",
        dtypes={"x": float, "y": float},
        freq="1 s",
    )
    shuffled = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    persisted = shuffled.persist()
    while not s.extensions["shuffle"].states:
        await asyncio.sleep(0.001)

    async with Worker(s.address) as w:
        await c.compute(persisted)

        await clean_worker(a)
        await clean_worker(b)
        await clean_worker(w)
        del persisted
        await clean_scheduler(s)


@gen_cluster(client=True)
async def test_multi(c, s, a, b):
    left = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-20",
        freq="10s",
        dtypes={"id": float, "x": float},
    )
    right = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        freq="10s",
        dtypes={"id": float, "y": float},
    )
    left["id"] = (left["id"] * 1000000).astype(int)
    right["id"] = (right["id"] * 1000000).astype(int)

    out = left.merge(right, on="id", shuffle="p2p")
    out = await c.compute(out.size)
    assert out

    await clean_worker(a)
    await clean_worker(b)
    await clean_scheduler(s)


@gen_cluster(client=True)
async def test_restrictions(c, s, a, b):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="10 s",
    ).persist(workers=a.address)
    await df
    assert a.data
    assert not b.data

    x = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    x = x.persist(workers=b.address)
    y = dd.shuffle.shuffle(df, "y", shuffle="p2p")
    y = y.persist(workers=a.address)

    await x
    assert all(stringify(key) in b.data for key in x.__dask_keys__())

    await y
    assert all(stringify(key) in a.data for key in y.__dask_keys__())


@gen_cluster(client=True)
async def test_delete_some_results(c, s, a, b):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="10 s",
    )
    x = dd.shuffle.shuffle(df, "x", shuffle="p2p").persist()
    while not s.tasks or not any(ts.state == "memory" for ts in s.tasks.values()):
        await asyncio.sleep(0.01)

    x = x.partitions[: x.npartitions // 2].persist()

    await c.compute(x.size)
    del x
    await clean_worker(a)
    await clean_worker(b)
    await clean_scheduler(s)


@gen_cluster(client=True)
async def test_add_some_results(c, s, a, b):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="10 s",
    )
    x = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    y = x.partitions[: x.npartitions // 2].persist()

    while not s.tasks or not any(ts.state == "memory" for ts in s.tasks.values()):
        await asyncio.sleep(0.01)

    x = x.persist()

    await c.compute(x.size)

    await clean_worker(a)
    await clean_worker(b)
    del x
    del y
    await clean_scheduler(s)


@pytest.mark.slow
@gen_cluster(client=True, nthreads=[("", 1)] * 2)
async def test_clean_after_close(c, s, a, b):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2001-01-01",
        dtypes={"x": float, "y": float},
        freq="100 s",
    )

    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    out = out.persist()

    await wait_for_tasks_in_state("shuffle-transfer", "executing", 1, a)
    await wait_for_tasks_in_state("shuffle-transfer", "memory", 1, b)

    await a.close()
    await clean_worker(a)

    del out
    await clean_worker(b)
    await clean_scheduler(s)


class DataFrameShuffleTestPool(AbstractShuffleTestPool):
    _shuffle_run_id_iterator = itertools.count()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._executor = ThreadPoolExecutor(2)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self._executor.shutdown(cancel_futures=True)
        except Exception:  # pragma: no cover
            self._executor.shutdown()

    def new_shuffle(
        self,
        name,
        worker_for_mapping,
        directory,
        loop,
        Shuffle=DataFrameShuffleRun,
    ):
        s = Shuffle(
            column="_partition",
            worker_for=worker_for_mapping,
            # FIXME: Is output_workers redundant with worker_for?
            output_workers=set(worker_for_mapping.values()),
            directory=directory / name,
            id=ShuffleId(name),
            run_id=next(AbstractShuffleTestPool._shuffle_run_id_iterator),
            local_address=name,
            executor=self._executor,
            rpc=self,
            scheduler=self,
            memory_limiter_disk=ResourceLimiter(10000000),
            memory_limiter_comms=ResourceLimiter(10000000),
        )
        self.shuffles[name] = s
        return s


# 36 parametrizations
# Runtime each ~0.1s
@pytest.mark.parametrize("n_workers", [10])
@pytest.mark.parametrize("n_input_partitions", [5000])
@pytest.mark.parametrize("npartitions", [5000])
@pytest.mark.parametrize("barrier_first_worker", [True])
@gen_test()
async def test_basic_lowlevel_shuffle(
    tmp_path,
    loop_in_thread,
    n_workers,
    n_input_partitions,
    npartitions,
    barrier_first_worker,
):
    pa = pytest.importorskip("pyarrow")

    dfs = []
    rows_per_df = 10
    for ix in range(n_input_partitions):
        df = pd.DataFrame({"x": range(rows_per_df * ix, rows_per_df * (ix + 1))})
        df["_partition"] = df.x % npartitions
        dfs.append(df)

    workers = list("abcdefghijklmn")[:n_workers]

    worker_for_mapping = {}

    for part in range(npartitions):
        worker_for_mapping[part] = get_worker_for_range_sharding(
            npartitions, part, workers
        )
    assert len(set(worker_for_mapping.values())) == min(n_workers, npartitions)
    meta = dfs[0].head(0)

    with DataFrameShuffleTestPool() as local_shuffle_pool:
        shuffles = []
        for ix in range(n_workers):
            shuffles.append(
                local_shuffle_pool.new_shuffle(
                    name=workers[ix],
                    worker_for_mapping=worker_for_mapping,
                    directory=tmp_path,
                    loop=loop_in_thread,
                )
            )
        random.seed(42)
        if barrier_first_worker:
            barrier_worker = shuffles[0]
        else:
            barrier_worker = random.sample(shuffles, k=1)[0]

        try:
            for ix, df in enumerate(dfs):
                s = shuffles[ix % len(shuffles)]
                await s.add_partition(df, ix)

            await barrier_worker.barrier()

            total_bytes_sent = 0
            total_bytes_recvd = 0
            total_bytes_recvd_shuffle = 0
            for s in shuffles:
                metrics = s.heartbeat()
                assert metrics["comm"]["total"] == metrics["comm"]["written"]
                total_bytes_sent += metrics["comm"]["written"]
                total_bytes_recvd += metrics["disk"]["total"]
                total_bytes_recvd_shuffle += s.total_recvd

            assert total_bytes_recvd_shuffle == total_bytes_sent

            all_parts = []
            for part, worker in worker_for_mapping.items():
                s = local_shuffle_pool.shuffles[worker]
                all_parts.append(s.get_output_partition(part, f"key-{part}", meta=meta))

            all_parts = await asyncio.gather(*all_parts)

            df_after = pd.concat(all_parts)
        finally:
            await asyncio.gather(*[s.close() for s in shuffles])
        assert len(df_after) == len(pd.concat(dfs))


@gen_test()
async def test_error_offload(tmp_path, loop_in_thread):
    pa = pytest.importorskip("pyarrow")

    dfs = []
    rows_per_df = 10
    n_input_partitions = 2
    npartitions = 2
    for ix in range(n_input_partitions):
        df = pd.DataFrame({"x": range(rows_per_df * ix, rows_per_df * (ix + 1))})
        df["_partition"] = df.x % npartitions
        dfs.append(df)

    workers = ["A", "B"]

    worker_for_mapping = {}
    partitions_for_worker = defaultdict(list)

    for part in range(npartitions):
        worker_for_mapping[part] = w = get_worker_for_range_sharding(
            npartitions, part, workers
        )
        partitions_for_worker[w].append(part)

    class ErrorOffload(DataFrameShuffleRun):
        async def offload(self, func, *args):
            raise RuntimeError("Error during deserialization")

    with DataFrameShuffleTestPool() as local_shuffle_pool:
        sA = local_shuffle_pool.new_shuffle(
            name="A",
            worker_for_mapping=worker_for_mapping,
            directory=tmp_path,
            loop=loop_in_thread,
            Shuffle=ErrorOffload,
        )
        sB = local_shuffle_pool.new_shuffle(
            name="B",
            worker_for_mapping=worker_for_mapping,
            directory=tmp_path,
            loop=loop_in_thread,
        )
        try:
            await sB.add_partition(dfs[0], 0)
            with pytest.raises(RuntimeError, match="Error during deserialization"):
                await sB.add_partition(dfs[1], 1)
                await sB.barrier()
        finally:
            await asyncio.gather(*[s.close() for s in [sA, sB]])


@gen_test()
async def test_error_send(tmp_path, loop_in_thread):
    pa = pytest.importorskip("pyarrow")

    dfs = []
    rows_per_df = 10
    n_input_partitions = 1
    npartitions = 2
    for ix in range(n_input_partitions):
        df = pd.DataFrame({"x": range(rows_per_df * ix, rows_per_df * (ix + 1))})
        df["_partition"] = df.x % npartitions
        dfs.append(df)

    workers = ["A", "B"]

    worker_for_mapping = {}
    partitions_for_worker = defaultdict(list)

    for part in range(npartitions):
        worker_for_mapping[part] = w = get_worker_for_range_sharding(
            npartitions, part, workers
        )
        partitions_for_worker[w].append(part)

    class ErrorSend(DataFrameShuffleRun):
        async def send(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("Error during send")

    with DataFrameShuffleTestPool() as local_shuffle_pool:
        sA = local_shuffle_pool.new_shuffle(
            name="A",
            worker_for_mapping=worker_for_mapping,
            directory=tmp_path,
            loop=loop_in_thread,
            Shuffle=ErrorSend,
        )
        sB = local_shuffle_pool.new_shuffle(
            name="B",
            worker_for_mapping=worker_for_mapping,
            directory=tmp_path,
            loop=loop_in_thread,
        )
        try:
            await sA.add_partition(dfs[0], 0)
            with pytest.raises(RuntimeError, match="Error during send"):
                await sA.barrier()
        finally:
            await asyncio.gather(*[s.close() for s in [sA, sB]])


@gen_test()
async def test_error_receive(tmp_path, loop_in_thread):
    pa = pytest.importorskip("pyarrow")

    dfs = []
    rows_per_df = 10
    n_input_partitions = 1
    npartitions = 2
    for ix in range(n_input_partitions):
        df = pd.DataFrame({"x": range(rows_per_df * ix, rows_per_df * (ix + 1))})
        df["_partition"] = df.x % npartitions
        dfs.append(df)

    workers = ["A", "B"]

    worker_for_mapping = {}
    partitions_for_worker = defaultdict(list)

    for part in range(npartitions):
        worker_for_mapping[part] = w = get_worker_for_range_sharding(
            npartitions, part, workers
        )
        partitions_for_worker[w].append(part)

    class ErrorReceive(DataFrameShuffleRun):
        async def receive(self, data: list[tuple[int, bytes]]) -> None:
            raise RuntimeError("Error during receive")

    with DataFrameShuffleTestPool() as local_shuffle_pool:
        sA = local_shuffle_pool.new_shuffle(
            name="A",
            worker_for_mapping=worker_for_mapping,
            directory=tmp_path,
            loop=loop_in_thread,
            Shuffle=ErrorReceive,
        )
        sB = local_shuffle_pool.new_shuffle(
            name="B",
            worker_for_mapping=worker_for_mapping,
            directory=tmp_path,
            loop=loop_in_thread,
        )
        try:
            await sB.add_partition(dfs[0], 0)
            with pytest.raises(RuntimeError, match="Error during receive"):
                await sB.barrier()
        finally:
            await asyncio.gather(*[s.close() for s in [sA, sB]])


from distributed.worker import DEFAULT_EXTENSIONS


class BlockedShuffleReceiveShuffleWorkerExtension(ShuffleWorkerExtension):
    def __init__(self, worker: Worker) -> None:
        super().__init__(worker)
        self.in_shuffle_receive = asyncio.Event()
        self.block_shuffle_receive = asyncio.Event()

    async def shuffle_receive(self, *args: Any, **kwargs: Any) -> None:
        self.in_shuffle_receive.set()
        await self.block_shuffle_receive.wait()
        return await super().shuffle_receive(*args, **kwargs)


@pytest.mark.parametrize("wait_until_forgotten", [True, False])
@mock.patch.dict(
    DEFAULT_EXTENSIONS,
    {"shuffle": BlockedShuffleReceiveShuffleWorkerExtension},
)
@gen_cluster(client=True, nthreads=[("", 1)] * 2)
async def test_deduplicate_stale_transfer(c, s, a, b, wait_until_forgotten):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="100 s",
    )
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    out = out.persist()

    shuffle_extA = a.extensions["shuffle"]
    shuffle_extB = b.extensions["shuffle"]
    await asyncio.gather(
        shuffle_extA.in_shuffle_receive.wait(), shuffle_extB.in_shuffle_receive.wait()
    )
    del out

    if wait_until_forgotten:
        while s.tasks or shuffle_extA.shuffles or shuffle_extB.shuffles:
            await asyncio.sleep(0)

    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    x = c.compute(out.x.size)
    await wait_until_new_shuffle_is_initialized(s)
    shuffle_extA.block_shuffle_receive.set()
    shuffle_extB.block_shuffle_receive.set()

    x = await x
    y = await c.compute(df.x.size)
    assert x == y

    await clean_worker(a, timeout=2)
    await clean_worker(b, timeout=2)
    await clean_scheduler(s, timeout=2)


class BlockedBarrierShuffleWorkerExtension(ShuffleWorkerExtension):
    def __init__(self, worker: Worker) -> None:
        super().__init__(worker)
        self.in_barrier = asyncio.Event()
        self.block_barrier = asyncio.Event()

    async def _barrier(self, *args: Any, **kwargs: Any) -> int:
        self.in_barrier.set()
        await self.block_barrier.wait()
        return await super()._barrier(*args, **kwargs)


@pytest.mark.parametrize("wait_until_forgotten", [True, False])
@mock.patch.dict(
    DEFAULT_EXTENSIONS,
    {"shuffle": BlockedBarrierShuffleWorkerExtension},
)
@gen_cluster(client=True, nthreads=[("", 1)] * 2)
async def test_handle_stale_barrier(c, s, a, b, wait_until_forgotten):
    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="100 s",
    )
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    out = out.persist()

    shuffle_extA = a.extensions["shuffle"]
    shuffle_extB = b.extensions["shuffle"]

    wait_for_barrier_on_A_task = asyncio.create_task(shuffle_extA.in_barrier.wait())
    wait_for_barrier_on_B_task = asyncio.create_task(shuffle_extB.in_barrier.wait())

    await asyncio.wait(
        [wait_for_barrier_on_A_task, wait_for_barrier_on_B_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    del out

    if wait_until_forgotten:
        while s.tasks:
            await asyncio.sleep(0)

    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    x, y = c.compute([df.x.size, out.x.size])
    await wait_until_new_shuffle_is_initialized(s)
    shuffle_extA.block_barrier.set()
    shuffle_extB.block_barrier.set()

    x = await x
    y = await y
    assert x == y

    await clean_worker(a, timeout=2)
    await clean_worker(b, timeout=2)
    await clean_scheduler(s, timeout=2)


@mock.patch.dict(
    DEFAULT_EXTENSIONS,
    {"shuffle": BlockedBarrierShuffleWorkerExtension},
)
@gen_cluster(client=True, nthreads=[("", 1)])
async def test_shuffle_run_consistency(c, s, a):
    """This test checks the correct creation of shuffle run IDs through the scheduler
    as well as the correct handling through the workers.

    In particular, newer run IDs for the same shuffle must always be larger than
    previous ones so that we can detect stale runs.

    .. note:
        The P2P implementation relies on the correctness of this behavior,
        but it is an implementation detail that users should not rely upon.
    """
    worker_ext = a.extensions["shuffle"]
    scheduler_ext = s.extensions["shuffle"]

    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="100 s",
    )
    # Initialize first shuffle execution
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    out = out.persist()

    shuffle_id = await wait_until_new_shuffle_is_initialized(s)
    shuffle_dict = scheduler_ext.get(shuffle_id, a.worker_address)

    # Worker extension can fetch the current run
    assert await worker_ext._get_shuffle_run(shuffle_id, shuffle_dict["run_id"])

    # This should never occur, but fetching an ID larger than the ID available on
    # the scheduler should result in an error.
    with pytest.raises(RuntimeError, match="invalid"):
        await worker_ext._get_shuffle_run(shuffle_id, shuffle_dict["run_id"] + 1)

    # Finish first execution
    worker_ext.block_barrier.set()
    await out
    del out
    while s.tasks:
        await asyncio.sleep(0)
    worker_ext.block_barrier.clear()

    # Initialize second shuffle execution
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    out = out.persist()

    new_shuffle_id = await wait_until_new_shuffle_is_initialized(s)
    assert shuffle_id == new_shuffle_id

    new_shuffle_dict = scheduler_ext.get(shuffle_id, a.worker_address)

    # Check invariant that the new run ID is larger than the previous
    assert shuffle_dict["run_id"] < new_shuffle_dict["run_id"]

    # Worker extension can fetch the new shuffle run
    assert await worker_ext._get_shuffle_run(shuffle_id, new_shuffle_dict["run_id"])

    # Fetching a stale run from a worker aware of the new run raises an error
    with pytest.raises(RuntimeError, match="stale"):
        await worker_ext._get_shuffle_run(shuffle_id, shuffle_dict["run_id"])

    worker_ext.block_barrier.set()
    await out
    del out

    await clean_worker(a, timeout=2)
    await clean_scheduler(s, timeout=2)


class BlockedShuffleAccessAndFailWorkerExtension(ShuffleWorkerExtension):
    def __init__(self, worker: Worker) -> None:
        super().__init__(worker)
        self.in_get_or_create_shuffle = asyncio.Event()
        self.block_get_or_create_shuffle = asyncio.Event()
        self.in_get_shuffle_run = asyncio.Event()
        self.block_get_shuffle_run = asyncio.Event()
        self.finished_get_shuffle_run = asyncio.Event()
        self.allow_fail = False

    async def _get_or_create_shuffle(self, *args: Any, **kwargs: Any) -> ShuffleRun:
        self.in_get_or_create_shuffle.set()
        await self.block_get_or_create_shuffle.wait()
        return await super()._get_or_create_shuffle(*args, **kwargs)

    async def _get_shuffle_run(self, *args: Any, **kwargs: Any) -> ShuffleRun:
        self.in_get_shuffle_run.set()
        await self.block_get_shuffle_run.wait()
        result = await super()._get_shuffle_run(*args, **kwargs)
        self.finished_get_shuffle_run.set()
        return result

    def shuffle_fail(self, *args: Any, **kwargs: Any) -> None:
        if self.allow_fail:
            return super().shuffle_fail(*args, **kwargs)


@mock.patch.dict(
    DEFAULT_EXTENSIONS,
    {"shuffle": BlockedShuffleAccessAndFailWorkerExtension},
)
@gen_cluster(client=True, nthreads=[("", 1)] * 2)
async def test_replace_stale_shuffle(c, s, a, b):
    ext_A = a.extensions["shuffle"]
    ext_B = b.extensions["shuffle"]

    # Let A behave normal
    ext_A.allow_fail = True
    ext_A.block_get_shuffle_run.set()
    ext_A.block_get_or_create_shuffle.set()

    # B can accept shuffle transfers
    ext_B.block_get_shuffle_run.set()

    df = dask.datasets.timeseries(
        start="2000-01-01",
        end="2000-01-10",
        dtypes={"x": float, "y": float},
        freq="100 s",
    )
    # Initialize first shuffle execution
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    out = out.persist()

    shuffle_id = await wait_until_new_shuffle_is_initialized(s)

    await wait_for_tasks_in_state("shuffle-transfer", "memory", 1, a)
    await ext_B.finished_get_shuffle_run.wait()
    assert shuffle_id in ext_A.shuffles
    assert shuffle_id in ext_B.shuffles
    stale_shuffle_run = ext_B.shuffles[shuffle_id]

    del out
    while s.tasks:
        await asyncio.sleep(0)

    # A is cleaned
    await clean_worker(a)

    # B is not cleaned
    assert shuffle_id in ext_B.shuffles
    assert not stale_shuffle_run.closed
    ext_B.finished_get_shuffle_run.clear()

    # Initialize second shuffle execution
    out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
    out = out.persist()

    await wait_for_tasks_in_state("shuffle-transfer", "memory", 1, a)
    await ext_B.finished_get_shuffle_run.wait()

    # Stale shuffle run has been replaced
    shuffle_run = ext_B.shuffles[shuffle_id]
    assert shuffle_run != stale_shuffle_run
    assert shuffle_run.run_id > stale_shuffle_run.run_id

    # Stale shuffle gets cleaned up
    await stale_shuffle_run._closed_event.wait()

    # Finish shuffle run
    ext_B.block_get_shuffle_run.set()
    ext_B.block_get_or_create_shuffle.set()
    ext_B.allow_fail = True
    await out
    del out

    await clean_worker(a)
    await clean_worker(b)
    await clean_scheduler(s)


class BlockedRemoveWorkerSchedulerPlugin(SchedulerPlugin):
    def __init__(self, scheduler: Scheduler, *args: Any, **kwargs: Any):
        self.scheduler = scheduler
        super().__init__(*args, **kwargs)
        self.in_remove_worker = asyncio.Event()
        self.block_remove_worker = asyncio.Event()
        self.scheduler.add_plugin(self)

    async def remove_worker(self, *args: Any, **kwargs: Any) -> None:
        self.in_remove_worker.set()
        await self.block_remove_worker.wait()


class BlockedBarrierSchedulerExtension(ShuffleSchedulerExtension):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.in_barrier = asyncio.Event()
        self.block_barrier = asyncio.Event()

    async def barrier(self, *args: Any, **kwargs: Any) -> None:
        self.in_barrier.set()
        await self.block_barrier.wait()
        await super().barrier(*args, **kwargs)


@gen_cluster(
    client=True,
    nthreads=[],
    scheduler_kwargs={
        "extensions": {
            "blocking": BlockedRemoveWorkerSchedulerPlugin,
            "shuffle": BlockedBarrierSchedulerExtension,
        }
    },
)
async def test_closed_worker_returns_before_barrier(c, s):
    async with AsyncExitStack() as stack:
        workers = [await stack.enter_async_context(Worker(s.address)) for _ in range(2)]

        df = dask.datasets.timeseries(
            start="2000-01-01",
            end="2000-01-10",
            dtypes={"x": float, "y": float},
            freq="10 s",
        )
        out = dd.shuffle.shuffle(df, "x", shuffle="p2p")
        out = out.persist()
        shuffle_id = await wait_until_new_shuffle_is_initialized(s)
        key = barrier_key(shuffle_id)
        await wait_for_state(key, "processing", s)
        scheduler_extension = s.extensions["shuffle"]
        await scheduler_extension.in_barrier.wait()

        flushes = [
            w.extensions["shuffle"].shuffles[shuffle_id]._flush_comm() for w in workers
        ]
        await asyncio.gather(*flushes)

        ts = s.tasks[key]
        to_close = None
        for worker in workers:
            if ts.processing_on.address != worker.address:
                to_close = worker
                break
        assert to_close
        closed_port = to_close.port
        await to_close.close()

        blocking_extension = s.extensions["blocking"]
        assert blocking_extension.in_remove_worker.is_set()

        workers.append(
            await stack.enter_async_context(Worker(s.address, port=closed_port))
        )

        scheduler_extension.block_barrier.set()

        with pytest.raises(
            RuntimeError, match=f"shuffle_barrier failed .* {shuffle_id}"
        ):
            await c.compute(out.x.size)

        blocking_extension.block_remove_worker.set()
        await c.close()
        await asyncio.gather(*[clean_worker(w) for w in workers])
        await clean_scheduler(s)


@gen_cluster(client=True)
async def test_handle_null_partitions_p2p_shuffling(c, s, *workers):
    data = [
        {"companies": [], "id": "a", "x": None},
        {"companies": [{"id": 3}, {"id": 5}], "id": "b", "x": None},
        {"companies": [{"id": 3}, {"id": 4}, {"id": 5}], "id": "c", "x": "b"},
        {"companies": [{"id": 9}], "id": "a", "x": "a"},
    ]
    df = pd.DataFrame(data)
    ddf = dd.from_pandas(df, npartitions=2)
    ddf = ddf.shuffle(on="id", shuffle="p2p", ignore_index=True)
    result = await c.compute(ddf)
    dd.assert_eq(result, df)

    await c.close()
    await asyncio.gather(*[clean_worker(w) for w in workers])
    await clean_scheduler(s)

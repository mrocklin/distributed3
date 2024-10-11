from __future__ import annotations

import asyncio
import itertools
import json
import logging
import math
import operator
import pickle
import random
import re
import sys
from collections.abc import Collection
from itertools import product
from textwrap import dedent
from time import sleep
from unittest import mock

import cloudpickle
import psutil
import pytest
from tlz import concat, first, merge
from tornado.ioloop import IOLoop

import dask
from dask import bag, delayed
from dask.core import flatten
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from dask.utils import parse_timedelta, tmpfile, typename

from distributed import (
    CancelledError,
    Client,
    Event,
    Lock,
    Nanny,
    SchedulerPlugin,
    Worker,
    fire_and_forget,
    wait,
)
from distributed.comm.addressing import parse_host_port
from distributed.compatibility import LINUX, MACOS, WINDOWS, PeriodicCallback
from distributed.core import ConnectionPool, Status, clean_exception, connect, rpc
from distributed.metrics import time
from distributed.protocol import serialize
from distributed.protocol.pickle import dumps, loads
from distributed.protocol.serialize import ToPickle
from distributed.scheduler import (
    KilledWorker,
    MemoryState,
    NoValidWorkerError,
    NoWorkerError,
    Scheduler,
    WorkerState,
)
from distributed.utils import TimeoutError, wait_for
from distributed.utils_test import (
    NO_AMM,
    BlockedGatherDep,
    BlockedGetData,
    BlockedKillNanny,
    BrokenComm,
    NoSchedulerDelayWorker,
    assert_story,
    async_poll_for,
    captured_handler,
    captured_logger,
    cluster,
    dec,
    div,
    freeze_batched_send,
    freeze_data_fetching,
    gen_cluster,
    gen_test,
    inc,
    nodebug,
    padded_time,
    raises_with_cause,
    slowadd,
    slowdec,
    slowidentity,
    slowinc,
    tls_only_security,
    varying,
    wait_for_state,
)
from distributed.worker import dumps_function, secede

pytestmark = pytest.mark.ci1

QUEUING_ON_BY_DEFAULT = math.isfinite(
    float(dask.config.get("distributed.scheduler.worker-saturation"))
)
alice = "alice:1234"
bob = "bob:1234"


@gen_cluster()
async def test_administration(s, a, b):
    assert isinstance(s.address, str)
    assert s.address in str(s)
    assert str(len(s.workers)) in repr(s)


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)])
async def test_respect_data_in_memory(c, s, a):
    x = delayed(inc)(1, dask_key_name="x")
    y = delayed(inc)(x, dask_key_name="y")
    f = c.persist(y)
    await wait([f])

    assert s.tasks[y.key].who_has == {s.workers[a.address]}

    z = delayed(operator.add)(x, y, dask_key_name="z")
    f2 = c.persist(z)
    while f2.key not in s.tasks or not s.tasks[f2.key]:
        assert s.tasks[y.key].who_has
        await asyncio.sleep(0.0001)


@gen_cluster(client=True)
async def test_recompute_released_results(c, s, a, b):
    x = delayed(inc)(1)
    y = delayed(inc)(x)

    yy = c.persist(y)
    await wait(yy)

    while s.tasks[x.key].who_has or x.key in a.data or x.key in b.data:  # let x go away
        await asyncio.sleep(0.01)

    z = delayed(dec)(x)
    zz = c.compute(z)
    result = await zz
    assert result == 1


@gen_cluster(client=True)
async def test_decide_worker_with_many_independent_leaves(c, s, a, b):
    xs = await asyncio.gather(
        c.scatter(list(range(0, 100, 2)), workers=a.address),
        c.scatter(list(range(1, 100, 2)), workers=b.address),
    )
    xs = list(concat(zip(*xs)))
    ys = [delayed(inc)(x) for x in xs]

    y2s = c.persist(ys)
    await wait(y2s)

    nhits = sum(y.key in a.data for y in y2s[::2]) + sum(
        y.key in b.data for y in y2s[1::2]
    )

    assert nhits > 80


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 3)
async def test_decide_worker_with_restrictions(client, s, a, b, c):
    x = client.submit(inc, 1, workers=[a.address, b.address])
    await x
    assert x.key in a.data or x.key in b.data


@pytest.mark.parametrize("ndeps", [0, 1, 4])
@pytest.mark.parametrize(
    "nthreads",
    [
        [("127.0.0.1", 1)] * 5,
        [("127.0.0.1", 3), ("127.0.0.1", 2), ("127.0.0.1", 1)],
    ],
)
def test_decide_worker_coschedule_order_neighbors(ndeps, nthreads):
    @gen_cluster(
        client=True,
        nthreads=nthreads,
        config={
            "distributed.scheduler.work-stealing": False,
            "distributed.scheduler.worker-saturation": float("inf"),
        },
    )
    async def test_decide_worker_coschedule_order_neighbors_(c, s, *workers):
        r"""
        Ensure that sibling root tasks are scheduled to the same node, reducing future
        data transfer.

        We generate a wide layer of "root" tasks (random NumPy arrays). All of those
        tasks share 0-5 trivial dependencies. The ``ndeps=0`` and ``ndeps=1`` cases are
        most common in real-world use (``ndeps=1`` is basically ``da.from_array(...,
        inline_array=False)`` or ``da.from_zarr``). The graph is structured like this
        (though the number of tasks and workers is different):

            |-W1-|  |-W2-| |-W3-|  |-W4-|   < ---- ideal task scheduling

              q       r       s       t      < --- `sum-aggregate-`
             / \     / \     / \     / \
            i   j   k   l   m   n   o   p    < --- `sum-`
            |   |   |   |   |   |   |   |
            a   b   c   d   e   f   g   h    < --- `random-`
            \   \   \   |   |   /   /   /
                   TRIVIAL * 0..5

        Neighboring `random-` tasks should be scheduled on the same worker. We test that
        generally, only one worker holds each row of the array, that the `random-` tasks
        are never transferred, and that there are few transfers overall.
        """
        np = pytest.importorskip("numpy")
        da = pytest.importorskip("dask.array")

        if ndeps == 0:
            x = da.random.random((100, 100), chunks=(10, 10))
            random_keys = set(flatten(x.__dask_keys__()))
            trivial_deps = {}
        else:

            def random(**kwargs):
                assert len(kwargs) == ndeps
                return np.random.random((10, 10))

            trivial_deps = {
                f"k{i}": delayed(object(), name=f"object-{i}") for i in range(ndeps)
            }

            # TODO is there a simpler (non-blockwise) way to make this sort of graph?
            x = da.blockwise(
                random,
                "yx",
                new_axes={"y": (10,) * 10, "x": (10,) * 10},
                dtype=float,
                **trivial_deps,
            )
            random_keys = set(flatten(x.__dask_keys__()))

        xx, xsum = dask.persist(x, x.sum(axis=1, split_every=20))
        await xsum

        # Check that each chunk-row of the array is (mostly) stored on the same worker
        primary_worker_key_fractions = []
        secondary_worker_key_fractions = []
        for keys in x.__dask_keys__():
            # Iterate along rows of the array.
            keys = set(keys)

            # No more than 2 workers should have any keys
            assert sum(any(k in w.data for k in keys) for w in workers) <= 2

            # What fraction of the keys for this row does each worker hold?
            key_fractions = [
                len(set(w.data).intersection(keys)) / len(keys) for w in workers
            ]
            key_fractions.sort()
            # Primary worker: holds the highest percentage of keys
            # Secondary worker: holds the second highest percentage of keys
            primary_worker_key_fractions.append(key_fractions[-1])
            secondary_worker_key_fractions.append(key_fractions[-2])

        # There may be one or two rows that were poorly split across workers,
        # but the vast majority of rows should only be on one worker.
        assert np.mean(primary_worker_key_fractions) >= 0.9
        assert np.median(primary_worker_key_fractions) == 1.0
        assert np.mean(secondary_worker_key_fractions) <= 0.1
        assert np.median(secondary_worker_key_fractions) == 0.0

        # Check that there were few transfers
        unexpected_transfers = []
        for worker in workers:
            for log in worker.transfer_incoming_log:
                keys = log["keys"]
                # The root-ish tasks should never be transferred
                assert not random_keys.intersection(keys)
                # `object-` keys (the trivial deps of the root random tasks) should be
                # transferred
                if any(
                    (not isinstance(k, str) or not k.startswith("object")) for k in keys
                ):
                    # But not many other things should be
                    unexpected_transfers.append(list(keys))

        # A transfer at the very end to move aggregated results is fine (necessary with
        # unbalanced workers in fact), but generally there should be very very few
        # transfers.
        assert len(unexpected_transfers) <= 3, unexpected_transfers

    test_decide_worker_coschedule_order_neighbors_()


@pytest.mark.skipif(
    QUEUING_ON_BY_DEFAULT,
    reason="Not relevant with queuing on; see https://github.com/dask/distributed/issues/7204",
)
@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    config={"distributed.scheduler.work-stealing": False},
)
async def test_decide_worker_rootish_while_last_worker_is_retiring(c, s, a):
    """https://github.com/dask/distributed/issues/7063"""
    # Put a task in memory on the worker to be retired and prevent the other from
    # acquiring a replica. This will cause a to be stuck in closing_gracefully later on,
    # until we set b.block_gather_dep.
    m = (await c.scatter({"m": 1}, workers=[a.address]))["m"]

    evx = [Event() for _ in range(3)]
    evy = Event()

    async with BlockedGatherDep(s.address, nthreads=1) as b:
        xs = [
            c.submit(lambda ev: ev.wait(), evx[0], key="x-0", workers=[a.address]),
            c.submit(lambda ev: ev.wait(), evx[1], key="x-1", workers=[a.address]),
            c.submit(lambda ev: ev.wait(), evx[2], key="x-2", workers=[b.address]),
        ]
        ys = [
            c.submit(lambda x, ev: ev.wait(), xs[0], evy, key="y-0"),
            c.submit(lambda x, ev: ev.wait(), xs[0], evy, key="y-1"),
            c.submit(lambda x, ev: ev.wait(), xs[1], evy, key="y-2"),
            c.submit(lambda x, ev: ev.wait(), xs[2], evy, key="y-3"),
            c.submit(lambda x, ev: ev.wait(), xs[2], evy, key="y-4"),
            c.submit(lambda x, ev: ev.wait(), xs[2], evy, key="y-5"),
        ]

        while a.state.executing_count != 1 or b.state.executing_count != 1:
            await asyncio.sleep(0.01)

        # Rootish is a dynamic property as it is defined right now. Since the
        # above submit calls are individual update_graph calls, waiting for
        # tasks to be in executing state on the worker is not sufficient to
        # guarantee that all the y tasks are already on the scheduler. Only
        # after at least 5 have been registered, will the task be flagged as
        # rootish
        while "y-2" not in s.tasks or not s.is_rootish(s.tasks["y-2"]):
            await asyncio.sleep(0.01)

        # - y-2 has no restrictions
        # - TaskGroup(y) has more than 4 tasks (total_nthreads * 2)
        # - TaskGroup(y) has less than 5 dependency groups
        # - TaskGroup(y) has less than 5 dependency tasks
        assert s.is_rootish(s.tasks["y-2"])

        await evx[0].set()
        await wait_for_state("y-0", "processing", s)
        await wait_for_state("y-1", "processing", s)
        assert s.tasks["y-2"].group.last_worker == s.workers[a.address]
        assert s.tasks["y-2"].group.last_worker_tasks_left == 1

        # Take a out of the running pool, but without removing it from the cluster
        # completely
        retire_task = asyncio.create_task(c.retire_workers([a.address]))
        # Wait until AMM sends AcquireReplicasEvent to b to move away m
        await b.in_gather_dep.wait()
        assert s.workers[a.address].status == Status.closing_gracefully

        # Transition y-2 to processing. Normally, it would be scheduled on a, but it's
        # not a running worker, so we must choose b
        await evx[1].set()
        await wait_for_state("y-2", "processing", s)
        await wait_for_state("y-2", "waiting", b)  # x-1 is in memory on a

        # Cleanup
        b.block_gather_dep.set()
        await evx[2].set()
        await evy.set()
        await retire_task
        await wait(xs + ys)


from distributed import WorkerPlugin


class CountData(WorkerPlugin):
    def __init__(self, keys):
        self.keys = keys
        self.worker = None
        self.count = 0

    def setup(self, worker):
        self.worker = worker

    def transition(self, start, finish, *args, **kwargs):
        count = 0
        for k in self.worker.data:
            if k in self.keys:
                count += 1
        self.count = max(self.count, count)


@gen_cluster(
    nthreads=[("", 2)] * 4,
    client=True,
    config={"distributed.scheduler.worker-saturation": 1.0},
)
async def test_graph_execution_width(c, s, *workers):
    """
    Test that we don't execute the graph more breadth-first than necessary.

    We shouldn't start loading extra data if we're not going to use it immediately.
    The number of parallel work streams match the number of threads.
    """

    roots = [delayed(inc)(ix) for ix in range(32)]
    passthrough1 = [delayed(slowidentity)(r, delay=0) for r in roots]
    passthrough2 = [delayed(slowidentity)(r, delay=0) for r in passthrough1]
    done = [delayed(lambda r: None)(r) for r in passthrough2]
    await c.register_plugin(CountData(keys=[f.key for f in roots]), name="count-roots")
    fs = c.compute(done)
    await wait(fs)

    res = await c.run(lambda dask_worker: dask_worker.plugins["count-roots"].count)
    assert all(0 < count <= 2 for count in res.values())


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_forget_tasks_while_processing(c, s, a, b):
    events = [Event() for _ in range(10)]

    futures = c.map(Event.wait, events)
    await events[0].set()
    await futures[0]
    await c.close()
    assert not s.tasks


@pytest.mark.slow
@gen_cluster(client=True, nthreads=[("", 1)], Worker=Nanny)
async def test_restart_while_processing(c, s, a, b):
    events = [Event() for _ in range(10)]

    futures = c.map(Event.wait, events)
    await events[0].set()
    await futures[0]
    # TODO slow because worker waits a while for the task to finish
    await c.restart()
    assert not s.tasks


@gen_cluster(
    client=True,
    nthreads=[("", 1)] * 3,
    config={"distributed.scheduler.worker-saturation": 1.0},
)
async def test_queued_release_multiple_workers(c, s, *workers):
    async with Client(s.address, asynchronous=True) as c2:
        event = Event(client=c2)

        rootish_threshold = s.total_nthreads * 2 + 1

        first_batch = c.map(
            lambda i: event.wait(),
            range(rootish_threshold),
            key=[f"first-{i}" for i in range(rootish_threshold)],
        )
        await async_poll_for(lambda: s.queued, 5)

        second_batch = c2.map(
            lambda i: event.wait(),
            range(rootish_threshold),
            key=[f"second-{i}" for i in range(rootish_threshold)],
            fifo_timeout=0,
        )
        await async_poll_for(lambda: second_batch[0].key in s.tasks, 5)

        # All of the second batch should be queued after the first batch
        assert [ts.key for ts in s.queued.sorted()] == [
            f.key
            for f in itertools.chain(first_batch[s.total_nthreads :], second_batch)
        ]

        # Cancel the first batch.
        # Use `Client.close` instead of `del first_batch` because deleting futures sends cancellation
        # messages one at a time. We're testing here that when multiple workers have open slots, we don't
        # recommend the same queued tasks for every worker, so we need a bulk cancellation operation.
        await c.close()
        del c, first_batch

        await async_poll_for(lambda: len(s.tasks) == len(second_batch), 5)

        # Second batch should move up the queue and start processing
        assert len(s.queued) == len(second_batch) - s.total_nthreads, list(
            s.queued.sorted()
        )

        await event.set()
        await c2.gather(second_batch)


@gen_cluster(
    client=True,
    nthreads=[("", 2)] * 2,
    config={
        "distributed.worker.memory.pause": False,
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": False,
        "distributed.scheduler.work-stealing": False,
        "distributed.scheduler.worker-saturation": 1.0,
    },
)
async def test_queued_paused_new_worker(c, s, a, b):
    f1s = c.map(slowinc, range(16))
    f2s = c.map(slowinc, f1s)
    final = c.submit(sum, *f2s)
    del f1s, f2s

    while not a.data or not b.data:
        await asyncio.sleep(0.01)

    # manually pause the workers
    a.status = Status.paused
    b.status = Status.paused

    while s.running:
        # wait for workers pausing to hit the scheduler
        await asyncio.sleep(0.01)

    assert all(ws.processing for ws in s.workers.values())
    assert not s.running

    async with Worker(s.address, nthreads=2) as w:
        # Tasks are successfully scheduled onto a new worker
        while not w.state.data:
            await asyncio.sleep(0.01)

        del final
        while s.tasks:
            await asyncio.sleep(0.01)
        assert not s.queued


@pytest.mark.parametrize("queue", [True, False])
@gen_cluster(
    client=True,
    nthreads=[("", 2)] * 2,
    config={
        "distributed.worker.memory.pause": False,
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": False,
        "distributed.scheduler.work-stealing": False,
    },
)
async def test_queued_paused_unpaused(c, s, a, b, queue):
    if queue:
        s.WORKER_SATURATION = 1.0
    else:
        s.WORKER_SATURATION = float("inf")

    f1s = c.map(slowinc, range(16))
    f2s = c.map(slowinc, f1s)
    final = c.submit(sum, *f2s)
    del f1s, f2s

    while not a.data or not b.data:
        await asyncio.sleep(0.01)

    # manually pause the workers
    a.status = Status.paused
    b.status = Status.paused

    while s.running:
        # wait for workers pausing to hit the scheduler
        await asyncio.sleep(0.01)

    assert not s.running
    assert all(ws.processing for ws in s.workers.values())

    # un-pause
    a.status = Status.running
    b.status = Status.running
    while not s.running:
        await asyncio.sleep(0.01)

    if queue:
        assert all(ws.processing for ws in s.workers.values())

    await wait(final)


@gen_cluster(
    client=True,
    nthreads=[("", 2)] * 2,
    config={"distributed.scheduler.worker-saturation": 1.0},
)
async def test_queued_remove_add_worker(c, s, a, b):
    event = Event()
    fs = c.map(lambda i: event.wait(), range(10))

    await async_poll_for(lambda: len(s.queued) == 6, timeout=5)
    await s.remove_worker(a.address, stimulus_id="fake")
    assert len(s.queued) == 8

    # Add a new worker
    async with Worker(s.address, nthreads=2) as w:
        await async_poll_for(lambda: len(s.queued) == 6, timeout=5)

        await event.set()
        await wait(fs)


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_secede_opens_slot(c, s, a):
    first = Event()
    second = Event()

    def func(first, second):
        first.wait()
        secede()
        second.wait()

    fs = c.map(func, [first] * 5, [second] * 5, key=[f"x{i}" for i in range(5)])
    await async_poll_for(lambda: a.state.executing, timeout=5)

    await first.set()
    await async_poll_for(lambda: len(a.state.long_running) == len(fs), timeout=5)

    await second.set()
    await c.gather(fs)


@pytest.mark.parametrize(
    "saturation_config, expected_task_counts",
    [
        (2.5, (5, 3)),
        (2.0, (4, 2)),
        (1.1, (3, 2)),
        (1.0, (2, 1)),
        (0.1, (1, 1)),
        # This is necessary because there's no way to parse a float infinite from
        # a DASK_* environment variable
        ("inf", (6, 4)),
        (float("inf"), (6, 4)),
        # ^ depends on root task assignment logic; ok if changes, just needs to add up to 10
    ],
)
def test_saturation_factor(
    saturation_config: int | float | str, expected_task_counts: tuple[int, int]
) -> None:
    @gen_cluster(
        client=True,
        nthreads=[("", 2), ("", 1)],
        config={
            "distributed.scheduler.worker-saturation": saturation_config,
        },
    )
    async def _test_saturation_factor(c, s, a, b):
        saturation = float(saturation_config)
        event = Event()
        fs = c.map(
            lambda _: event.wait(), range(10), key=[f"wait-{i}" for i in range(10)]
        )
        while a.state.executing_count < min(
            a.state.nthreads, expected_task_counts[0]
        ) or b.state.executing_count < min(b.state.nthreads, expected_task_counts[1]):
            await asyncio.sleep(0.01)

        if math.isfinite(saturation):
            assert len(a.state.tasks) == expected_task_counts[0]
            assert len(b.state.tasks) == expected_task_counts[1]
        else:
            # Assignment is nondeterministic for some reason without queuing
            assert len(a.state.tasks) > len(b.state.tasks)

        await event.set()
        await c.gather(fs)

    _test_saturation_factor()


@gen_test()
async def test_bad_saturation_factor():
    with pytest.raises(ValueError, match="foo"):
        with dask.config.set({"distributed.scheduler.worker-saturation": "foo"}):
            async with Scheduler(dashboard_address=":0"):
                pass


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 3)
async def test_move_data_over_break_restrictions(client, s, a, b, c):
    [x] = await client.scatter([1], workers=b.address)
    y = client.submit(inc, x, workers=[a.address, b.address])
    await wait(y)
    assert y.key in a.data or y.key in b.data


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 3)
async def test_balance_with_restrictions(client, s, a, b, c):
    [x], [y] = await asyncio.gather(
        client.scatter([[1, 2, 3]], workers=a.address),
        client.scatter([1], workers=c.address),
    )
    z = client.submit(inc, 1, workers=[a.address, c.address])
    await wait(z)

    assert s.tasks[z.key].who_has == {s.workers[c.address]}


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 3)
async def test_no_valid_workers(client, s, a, b, c):
    x = client.submit(inc, 1, workers="127.0.0.5:9999")
    while not s.tasks:
        await asyncio.sleep(0.01)

    assert s.tasks[x.key] in s.unrunnable

    with pytest.raises(TimeoutError):
        await wait_for(x, 0.05)


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 3)
async def test_no_valid_workers_loose_restrictions(client, s, a, b, c):
    x = client.submit(inc, 1, workers="127.0.0.5:9999", allow_other_workers=True)
    result = await x
    assert result == 2


@pytest.mark.parametrize("queue", [False, True])
@gen_cluster(client=True, nthreads=[])
async def test_no_workers(client, s, queue):
    if queue:
        s.WORKER_SATURATION = 1.0
    else:
        s.WORKER_SATURATION = float("inf")

    x = client.submit(inc, 1)
    while not s.tasks:
        await asyncio.sleep(0.01)

    ts = s.tasks[x.key]
    if queue:
        assert ts in s.queued
        assert ts.state == "queued"
    else:
        assert ts in s.unrunnable
        assert ts.state == "no-worker"

    with pytest.raises(TimeoutError):
        await wait_for(x, 0.05)

    async with Worker(s.address, nthreads=1):
        await wait(x)


@gen_cluster(nthreads=[])
async def test_retire_workers_empty(s):
    await s.retire_workers(workers=[])


@gen_cluster(client=True, nthreads=[("", 1)] * 3)
async def test_retire_workers_lock(c, s, *workers):
    """Test that replicate(), rebalance(), and retire_workers() can be called at the
    same time

    See also
    --------
    test_retire_workers_concurrently
    test_active_memory_manager.py::test_RetireWorker_stress
    """
    futs = c.map(inc, range(100))
    await wait(futs)
    before = s.transition_counter
    await asyncio.gather(
        *(
            s.replicate(keys=[fut.key for fut in futs], n=2),
            s.rebalance(),
            s.retire_workers(workers=[workers[0].address]),
            s.retire_workers(workers=[workers[1].address]),
        )
    )
    await c.gather(futs)
    assert s.transition_counter == before


@gen_cluster(
    client=True,
    config={"distributed.scheduler.active-memory-manager.measure": "managed"},
)
async def test_retire_workers_concurrently(c, s, w1, w2):
    """Test that multiple calls to retire_workers can run at the same time

    See also
    --------
    test_retire_workers_lock
    test_active_memory_manager.py::test_RetireWorker_stress
    """
    async with BlockedGetData(s.address) as w3:
        x = c.submit(inc, 1, key="x", workers=[w2.address])
        y = c.submit(inc, 2, key="y", workers=[w3.address])
        await wait([x, y])

        # AMM will choose w1 as it's emptier than w2
        w3_fut = asyncio.create_task(c.retire_workers([w3.address]))
        await w3.in_get_data.wait()

        # Test that we can retire w2 even if w3 has not finished retiring yet
        await c.retire_workers([w2.address])
        assert set(s.workers) == {w1.address, w3.address}
        assert dict(w1.data) == {"x": 2}

        w3.block_get_data.set()
        await w3_fut

        assert set(s.workers) == {w1.address}
        assert dict(w1.data) == {"x": 2, "y": 3}


@gen_cluster()
async def test_server_listens_to_other_ops(s, a, b):
    async with rpc(s.address) as r:
        ident = await r.identity()
        assert ident["type"] == "Scheduler"
        assert ident["id"].lower().startswith("scheduler")


@gen_cluster(client=True)
async def test_remove_worker_from_scheduler(c, s, a, b):
    """see also test_ready_remove_worker"""
    ev = Event()
    futs = c.map(lambda x, ev: ev.wait(), range(20), ev=ev)
    while len(s.tasks) != len(futs):
        await asyncio.sleep(0.01)

    assert a.address in s.stream_comms
    await s.remove_worker(address=a.address, stimulus_id="test")
    assert a.address not in s.workers
    assert len(s.workers[b.address].processing) + len(s.queued) == len(futs)
    await ev.set()
    await c.gather(futs)


@gen_cluster(client=True)
async def test_remove_worker_from_scheduler_warns_on_safe(c, s, a, b):
    with pytest.warns(FutureWarning, match="expected"):
        await s.remove_worker(address=a.address, safe=True, stimulus_id="test")
    assert a.address not in s.workers


@gen_cluster()
async def test_remove_worker_by_name_from_scheduler(s, a, b):
    assert a.address in s.stream_comms
    assert await s.remove_worker(address=a.name, stimulus_id="test") == "OK"
    assert a.address not in s.workers
    assert (
        await s.remove_worker(address=a.address, stimulus_id="test")
        == "already-removed"
    )


@gen_cluster(config={"distributed.scheduler.events-cleanup-delay": "500 ms"})
async def test_clear_events_worker_removal(s, a, b):
    assert a.address in s._broker._topics
    assert a.address in s.workers
    assert b.address in s._broker._topics
    assert b.address in s.workers

    await s.remove_worker(address=a.address, stimulus_id="test")
    # Shortly after removal, the events should still be there
    assert s.get_events(a.address)
    assert a.address not in s.workers
    s.validate_state()

    start = time()
    while s.get_events(a.address):
        await asyncio.sleep(0.01)
        assert time() < start + 2
    assert b.address in s._broker._topics


@gen_cluster(
    config={"distributed.scheduler.events-cleanup-delay": "10 ms"}, client=True
)
async def test_clear_events_client_removal(c, s, a, b):
    assert s.get_events(c.id)
    s.remove_client(c.id)

    assert s.get_events(c.id)
    assert c.id not in s.clients
    assert c not in s.clients

    s.remove_client(c.id)
    # If it doesn't reconnect after a given time, the events log should be cleared
    start = time()
    while s.get_events(c.id):
        await asyncio.sleep(0.01)
        assert time() < start + 2


@gen_cluster(client=True, nthreads=[])
async def test_add_worker(c, s):
    x = c.submit(inc, 1, key="x")
    await wait_for_state("x", ("queued", "no-worker"), s)
    s.validate_state()

    async with Worker(s.address) as w:
        s.validate_state()
        assert w.ip in s.host_info
        assert s.host_info[w.ip]["addresses"] == {w.address}
        assert await x == 2


@gen_cluster(scheduler_kwargs={"blocked_handlers": ["feed"]})
async def test_blocked_handlers_are_respected(s, a, b):
    def func(scheduler):
        return dumps(dict(scheduler.worker_info))

    comm = await connect(s.address)
    await comm.write({"op": "feed", "function": dumps(func), "interval": 0.01})

    response = await comm.read()

    _, exc, _ = clean_exception(response["exception"], response["traceback"])
    assert isinstance(exc, ValueError)
    assert "'feed' handler has been explicitly disallowed" in repr(exc)

    await comm.close()


@gen_cluster(
    nthreads=[], config={"distributed.scheduler.blocked-handlers": ["test-handler"]}
)
async def test_scheduler_init_pulls_blocked_handlers_from_config(s):
    assert s.blocked_handlers == ["test-handler"]


@gen_cluster()
async def test_feed(s, a, b):
    def func(scheduler):
        return dumps({addr: ws.clean() for addr, ws in scheduler.workers.items()})

    comm = await connect(s.address)
    await comm.write({"op": "feed", "function": dumps(func), "interval": 0.01})

    for _ in range(5):
        response = await comm.read()
        expected = {addr: ws.clean() for addr, ws in s.workers.items()}
        assert cloudpickle.loads(response) == expected

    await comm.close()


@gen_cluster()
async def test_feed_setup_teardown(s, a, b):
    def setup(scheduler):
        return 1

    def func(scheduler, state):
        assert state == 1
        return "OK"

    def teardown(scheduler, state):
        scheduler.flag = "done"

    comm = await connect(s.address)
    await comm.write(
        {
            "op": "feed",
            "function": dumps(func),
            "setup": dumps(setup),
            "teardown": dumps(teardown),
            "interval": 0.01,
        }
    )

    for _ in range(5):
        response = await comm.read()
        assert response == "OK"

    await comm.close()
    start = time()
    while not hasattr(s, "flag"):
        await asyncio.sleep(0.01)
        assert time() - start < 5


@gen_cluster()
async def test_feed_large_bytestring(s, a, b):
    np = pytest.importorskip("numpy")

    x = np.ones(10000000)

    def func(scheduler):
        y = x
        return True

    comm = await connect(s.address)
    await comm.write({"op": "feed", "function": dumps(func), "interval": 0.05})

    for _ in range(5):
        response = await comm.read()
        assert response is True

    await comm.close()


@gen_cluster(client=True)
async def test_delete_data(c, s, a, b):
    d = await c.scatter({"x": 1, "y": 2, "z": 3})

    assert {ts.key for ts in s.tasks.values() if ts.who_has} == {"x", "y", "z"}
    assert set(a.data) | set(b.data) == {"x", "y", "z"}
    assert merge(a.data, b.data) == {"x": 1, "y": 2, "z": 3}

    del d["x"]
    del d["y"]

    start = time()
    while set(a.data) | set(b.data) != {"z"}:
        await asyncio.sleep(0.01)
        assert time() < start + 5


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)])
async def test_delete(c, s, a):
    x = c.submit(inc, 1)
    await x
    assert x.key in s.tasks
    assert x.key in a.data

    await c._cancel(x)

    start = time()
    while x.key in a.data:
        await asyncio.sleep(0.01)
        assert time() < start + 5

    assert x.key not in s.tasks

    s.report_on_key(key=x.key)


def test_dumps_function():
    a = dumps_function(inc)
    assert cloudpickle.loads(a)(10) == 11

    b = dumps_function(inc)
    assert a is b

    c = dumps_function(dec)
    assert a != c


@pytest.mark.parametrize("worker_saturation", [1.0, float("inf")])
@gen_cluster(client=True)
async def test_ready_remove_worker(c, s, a, b, worker_saturation):
    """see also test_remove_worker_from_scheduler"""
    s.WORKER_SATURATION = worker_saturation

    ev = Event()
    futs = c.map(lambda x, ev: ev.wait(), range(20), ev=ev)
    while len(s.tasks) != len(futs):
        await asyncio.sleep(0.01)
    if s.WORKER_SATURATION == 1:
        cmp = operator.eq
    elif math.isinf(s.WORKER_SATURATION):
        cmp = operator.gt
    else:
        pytest.fail(f"{s.WORKER_SATURATION=}, must be 1 or inf")
    assert all(cmp(len(w.processing), w.nthreads) for w in s.workers.values()), (
        list(s.workers.values()),
        s.WORKER_SATURATION,
    )
    assert sum(len(w.processing) for w in s.workers.values()) + len(s.queued) == len(
        s.tasks
    )
    await s.remove_worker(address=a.address, stimulus_id="test")
    assert set(s.workers) == {b.address}
    assert all(cmp(len(w.processing), w.nthreads) for w in s.workers.values()), (
        list(s.workers.values()),
        s.WORKER_SATURATION,
    )
    assert sum(len(w.processing) for w in s.workers.values()) + len(s.queued) == len(
        s.tasks
    )
    await ev.set()


@pytest.mark.slow
@gen_cluster(client=True, Worker=Nanny, timeout=60)
async def test_restart(c, s, a, b):
    with captured_logger("distributed.scheduler") as caplog:
        futures = c.map(inc, range(20))
        await wait(futures)
        with captured_logger("distributed.nanny") as nanny_logger:
            await s.restart(stimulus_id="test123")
        assert "Reason: test123" in nanny_logger.getvalue()

        assert not s.computations
        assert not s.task_prefixes
        assert not s.task_groups

        assert len(s.workers) == 2

        for ws in s.workers.values():
            assert not ws.occupancy
            assert not ws.processing

        assert not s.tasks

        assert all(f.status == "cancelled" for f in futures)
        x = c.submit(inc, 1)
        assert await x == 2
    assert "restart" in caplog.getvalue().lower()


@pytest.mark.slow
@gen_cluster(client=True, Worker=Nanny, nthreads=[("", 1)] * 5)
async def test_restart_waits_for_new_workers(c, s, *workers):
    original_procs = {n.process.process for n in workers}
    original_workers = dict(s.workers)

    await c.restart()
    assert len(s.workers) == len(original_workers)
    for w in workers:
        assert w.address not in s.workers

    # Confirm they restarted
    # NOTE: == for `psutil.Process` compares PID and creation time
    new_procs = {n.process.process for n in workers}
    assert new_procs != original_procs
    # Most times, the workers should have new addresses
    assert s.workers.keys() ^ original_workers.keys()
    # The old WorkerState instances should have been replaced
    assert set(s.workers.values()).isdisjoint(original_workers.values())


@pytest.mark.slow
@gen_cluster(client=True, Worker=BlockedKillNanny, nthreads=[("", 1)] * 2)
async def test_restart_nanny_timeout_exceeded(c, s, a, b):
    try:
        f = c.submit(div, 1, 0)
        fr = c.submit(inc, 1, resources={"FOO": 1})
        await wait(f)
        assert s.erred_tasks
        assert s.computations
        assert s.unrunnable
        assert s.tasks

        with pytest.raises(
            TimeoutError, match=r"2/2 nanny worker\(s\) did not shut down within 1s"
        ):
            await c.restart(timeout="1s")
        assert a.in_kill.is_set()
        assert b.in_kill.is_set()

        assert not s.workers
        assert not s.erred_tasks
        assert not s.computations
        assert not s.unrunnable
        assert not s.tasks

        assert not c.futures
        assert f.status == "cancelled"
        assert fr.status == "cancelled"
    finally:
        a.wait_kill.set()
        b.wait_kill.set()


@gen_cluster(client=True, nthreads=[("", 1)] * 2)
async def test_restart_not_all_workers_return(c, s, a, b):
    with pytest.raises(TimeoutError, match="Waited for 2 worker"):
        await c.restart(timeout="1s")

    assert not s.workers
    assert a.status in (Status.closed, Status.closing)
    assert b.status in (Status.closed, Status.closing)


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_restart_worker_rejoins_after_timeout_expired(c, s, a):
    """
    We don't want to see an error message like:

    ``Waited for 1 worker(s) to reconnect after restarting, but after 0s, 0 have not returned.``

    If a worker rejoins after our last poll for new workers, but before we raise the error,
    we shouldn't raise the error.
    """
    # We'll use a 0s timeout on the restart, so it always expires.
    # And we'll use a plugin to block the restart process, and spin up a new worker
    # in the middle of it.

    class Plugin(SchedulerPlugin):
        removed = asyncio.Event()
        proceed = asyncio.Event()

        async def remove_worker(self, *args, **kwargs):
            self.removed.set()
            await self.proceed.wait()

    s.add_plugin(Plugin())

    task = asyncio.create_task(c.restart(timeout=0))
    await Plugin.removed.wait()
    assert not s.workers

    async with Worker(s.address, nthreads=1):
        assert len(s.workers) == 1
        Plugin.proceed.set()

        # New worker has joined, but the timeout has expired (since it was 0).
        # Still, we should not time out.
        await task


@gen_cluster(client=True, nthreads=[("", 1)] * 2)
async def test_restart_no_wait_for_workers(c, s, a, b):
    await c.restart(timeout="1s", wait_for_workers=False)

    assert not s.workers
    # Workers are not immediately closed because of
    # https://github.com/dask/distributed/issues/6390
    # (the message is still waiting in the BatchedSend)
    await a.finished()
    await b.finished()


@pytest.mark.slow
@gen_cluster(client=True, Worker=Nanny)
async def test_restart_some_nannies_some_not(c, s, a, b):
    original_addrs = set(s.workers)
    async with Worker(s.address, nthreads=1) as w:
        await c.wait_for_workers(3)

        # FIXME how to make this not always take 20s if the nannies do restart quickly?
        with pytest.raises(TimeoutError, match=r"The 1 worker\(s\) not using Nannies"):
            await c.restart(timeout="20s")

        assert w.status == Status.closed

        assert len(s.workers) == 2
        assert set(s.workers).isdisjoint(original_addrs)
        assert w.address not in s.workers


@pytest.mark.slow
@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    Worker=BlockedKillNanny,
    worker_kwargs={"heartbeat_interval": "1ms"},
)
async def test_restart_heartbeat_before_closing(c, s, n):
    """
    Ensure that if workers heartbeat in the middle of `Scheduler.restart`, they don't close themselves.
    https://github.com/dask/distributed/issues/6494
    """
    prev_workers = dict(s.workers)
    restart_task = asyncio.create_task(s.restart(stimulus_id="test"))

    await n.in_kill.wait()
    await asyncio.sleep(0.5)  # significantly longer than the heartbeat interval

    # WorkerState should not be removed yet, because the worker hasn't been told to close
    assert s.workers

    n.wait_kill.set()
    # Wait until the worker has left (possibly until it's come back too)
    while s.workers == prev_workers:
        await asyncio.sleep(0.01)

    await restart_task
    await c.wait_for_workers(1)


@gen_cluster()
async def test_broadcast(s, a, b):
    result = await s.broadcast(msg={"op": "ping"})
    assert result == {a.address: b"pong", b.address: b"pong"}

    result = await s.broadcast(msg={"op": "ping"}, workers=[a.address])
    assert result == {a.address: b"pong"}

    result = await s.broadcast(msg={"op": "ping"}, hosts=[a.ip])
    assert result == {a.address: b"pong", b.address: b"pong"}


@gen_cluster(security=tls_only_security())
async def test_broadcast_tls(s, a, b):
    result = await s.broadcast(msg={"op": "ping"})
    assert result == {a.address: b"pong", b.address: b"pong"}

    result = await s.broadcast(msg={"op": "ping"}, workers=[a.address])
    assert result == {a.address: b"pong"}

    result = await s.broadcast(msg={"op": "ping"}, hosts=[a.ip])
    assert result == {a.address: b"pong", b.address: b"pong"}


@gen_cluster(Worker=Nanny)
async def test_broadcast_nanny(s, a, b):
    result1 = await s.broadcast(msg={"op": "identity"}, nanny=True)
    assert all(d["type"] == "Nanny" for d in result1.values())

    result2 = await s.broadcast(
        msg={"op": "identity"}, workers=[a.worker_address], nanny=True
    )
    assert len(result2) == 1
    assert first(result2.values())["id"] == a.id

    result3 = await s.broadcast(msg={"op": "identity"}, hosts=[a.ip], nanny=True)
    assert result1 == result3


@gen_cluster(config={"distributed.comm.timeouts.connect": "200ms"})
async def test_broadcast_on_error(s, a, b):
    a.stop()

    with pytest.raises(OSError):
        await s.broadcast(msg={"op": "ping"}, on_error="raise")
    with pytest.raises(ValueError, match="on_error must be"):
        await s.broadcast(msg={"op": "ping"}, on_error="invalid")

    out = await s.broadcast(msg={"op": "ping"}, on_error="return")
    assert isinstance(out[a.address], OSError)
    assert out[b.address] == b"pong"

    out = await s.broadcast(msg={"op": "ping"}, on_error="return_pickle")
    assert isinstance(loads(out[a.address]), OSError)
    assert out[b.address] == b"pong"

    out = await s.broadcast(msg={"op": "ping"}, on_error="ignore")
    assert out == {b.address: b"pong"}


@gen_cluster()
async def test_broadcast_deprecation(s, a, b):
    out = await s.broadcast(msg={"op": "ping"})
    assert out == {a.address: b"pong", b.address: b"pong"}


@gen_cluster(nthreads=[])
async def test_worker_name(s):
    async with Worker(s.address, name="alice") as w:
        assert s.workers[w.address].name == "alice"
        assert s.aliases["alice"] == w.address

        with raises_with_cause(RuntimeError, None, ValueError, None):
            async with Worker(s.address, name="alice"):
                pass


@gen_cluster(nthreads=[])
async def test_coerce_address(s):
    print("scheduler:", s.address, s.listen_address)
    a = Worker(s.address, name="alice")
    b = Worker(s.address, name=123)
    c = Worker("127.0.0.1", s.port, name="charlie")
    await asyncio.gather(a, b, c)

    assert s.coerce_address("127.0.0.1:8000") == "tcp://127.0.0.1:8000"
    assert s.coerce_address("[::1]:8000") == "tcp://[::1]:8000"
    assert s.coerce_address("tcp://127.0.0.1:8000") == "tcp://127.0.0.1:8000"
    assert s.coerce_address("tcp://[::1]:8000") == "tcp://[::1]:8000"
    assert s.coerce_address("localhost:8000") in (
        "tcp://127.0.0.1:8000",
        "tcp://[::1]:8000",
    )
    assert s.coerce_address("localhost:8000") in (
        "tcp://127.0.0.1:8000",
        "tcp://[::1]:8000",
    )
    assert s.coerce_address(a.address) == a.address
    # Aliases
    assert s.coerce_address("alice") == a.address
    assert s.coerce_address(123) == b.address
    assert s.coerce_address("charlie") == c.address

    assert s.coerce_hostname("127.0.0.1") == "127.0.0.1"
    assert s.coerce_hostname("alice") == a.ip
    assert s.coerce_hostname(123) == b.ip
    assert s.coerce_hostname("charlie") == c.ip
    assert s.coerce_hostname("jimmy") == "jimmy"

    assert s.coerce_address("zzzt:8000", resolve=False) == "tcp://zzzt:8000"
    await asyncio.gather(a.close(), b.close(), c.close())


@gen_cluster(nthreads=[], config={"distributed.scheduler.work-stealing": True})
async def test_config_stealing(s):
    """Regression test for https://github.com/dask/distributed/issues/3409"""
    assert "stealing" in s.extensions


@gen_cluster(nthreads=[], config={"distributed.scheduler.work-stealing": False})
async def test_config_no_stealing(s):
    assert "stealing" not in s.extensions


@pytest.mark.skipif(WINDOWS, reason="num_fds not supported on windows")
@gen_cluster(nthreads=[])
async def test_file_descriptors_dont_leak(s):
    proc = psutil.Process()
    before = proc.num_fds()

    async with Worker(s.address):
        assert proc.num_fds() > before

    while proc.num_fds() > before:
        await asyncio.sleep(0.01)


@gen_cluster()
async def test_update_graph_culls(s, a, b):
    # This is a rather low level API but the fact that update_graph actually
    # culls is worth testing and hard to do so with high level user API. Most
    # but not all HLGs are implementing culling themselves already, i.e. a graph
    # like the one written here will rarely exist in reality. It's worth to
    # consider dropping this from the scheduler iff graph materialization
    # actually ensure this
    dsk = HighLevelGraph(
        layers={
            "foo": MaterializedLayer(
                {
                    "x": (inc, 1),
                    "y": (inc, "x"),
                    "z": (inc, 2),
                }
            )
        },
        dependencies={"foo": set()},
    )

    header, frames = serialize(ToPickle(dsk), on_error="raise")
    await s.update_graph(
        graph_header=header,
        graph_frames=frames,
        keys=["y"],
        client="client",
        internal_priority={k: 0 for k in "xyz"},
        submitting_task=None,
        span_metadata={},
    )
    assert "z" not in s.tasks


@gen_test()
async def test_io_loop(loop):
    with pytest.warns(
        DeprecationWarning, match=r"the loop kwarg to Scheduler is deprecated"
    ):
        s = Scheduler(loop=loop, dashboard_address=":0", validate=True)
    assert s.io_loop is IOLoop.current()


@gen_cluster(client=True)
async def test_story(c, s, a, b):
    x = delayed(inc)(1)
    y = delayed(inc)(x)
    f = c.persist(y)
    await wait([f])

    assert s.transition_log

    story = s.story(x.key)
    assert all(line in s.transition_log for line in story)
    assert len(story) < len(s.transition_log)
    assert all(x.key == line[0] or x.key in line[3] for line in story)

    assert len(s.story(x.key, y.key)) > len(story)

    assert s.story(x.key) == s.story(s.tasks[x.key])


@pytest.mark.parametrize("direct", [False, True])
@gen_cluster(client=True, nthreads=[])
async def test_scatter_no_workers(c, s, direct):
    with pytest.raises(TimeoutError):
        await s.scatter(data={"x": 1}, client="alice", timeout=0.1)

    start = time()
    with pytest.raises(TimeoutError):
        await c.scatter(123, timeout=0.1, direct=direct)
    assert time() < start + 5

    fut = c.scatter({"y": 2}, timeout=5, direct=direct)
    await asyncio.sleep(0.1)
    async with Worker(s.address) as w:
        await fut
        assert w.data["y"] == 2

    # Test race condition between worker init and scatter
    w = Worker(s.address)
    await asyncio.gather(c.scatter({"z": 3}, timeout=5, direct=direct), w)
    assert w.data["z"] == 3
    await w.close()


@gen_cluster(nthreads=[])
async def test_scheduler_sees_memory_limits(s):
    async with Worker(s.address, nthreads=3, memory_limit=12345) as w:
        assert s.workers[w.address].memory_limit == 12345


@gen_cluster(client=True)
async def test_retire_workers(c, s, a, b):
    [x] = await c.scatter([1], workers=a.address)
    [y] = await c.scatter([list(range(1000))], workers=b.address)

    assert s.workers_to_close() == [a.address]

    workers = await s.retire_workers()
    assert list(workers) == [a.address]
    assert workers[a.address]["nthreads"] == a.state.nthreads
    assert list(s.workers) == [b.address]

    assert s.workers_to_close() == []

    assert s.workers[b.address].has_what == {s.tasks[x.key], s.tasks[y.key]}

    workers = await s.retire_workers()
    assert not workers


@gen_cluster(client=True)
async def test_retire_workers_n(c, s, a, b):
    await s.retire_workers(n=1, close_workers=True)
    assert len(s.workers) == 1

    await s.retire_workers(n=0, close_workers=True)
    assert len(s.workers) == 1

    await s.retire_workers(n=1, close_workers=True)
    assert len(s.workers) == 0

    await s.retire_workers(n=0, close_workers=True)
    assert len(s.workers) == 0

    while not (
        a.status in (Status.closed, Status.closing, Status.closing_gracefully)
        and b.status in (Status.closed, Status.closing, Status.closing_gracefully)
    ):
        await asyncio.sleep(0.01)


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 4)
async def test_workers_to_close(cl, s, *workers):
    with dask.config.set(
        {"distributed.scheduler.default-task-durations": {"a": 4, "b": 4, "c": 1}}
    ):
        futures = cl.map(slowinc, [1, 1, 1], key=["a-4", "b-4", "c-1"])
        while sum(len(w.processing) for w in s.workers.values()) < 3:
            await asyncio.sleep(0.001)

        wtc = s.workers_to_close()
        assert all(not s.workers[w].processing for w in wtc)
        assert len(wtc) == 1


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 4)
async def test_workers_to_close_grouped(c, s, *workers):
    groups = {
        workers[0].address: "a",
        workers[1].address: "a",
        workers[2].address: "b",
        workers[3].address: "b",
    }

    def key(ws):
        return groups[ws.address]

    assert set(s.workers_to_close(key=key)) == {w.address for w in workers}

    # Assert that job in one worker blocks closure of group
    future = c.submit(slowinc, 1, delay=0.2, workers=workers[0].address)
    while not any(ws.processing for ws in s.workers.values()):
        await asyncio.sleep(0.001)

    assert set(s.workers_to_close(key=key)) == {workers[2].address, workers[3].address}

    del future

    while any(ws.processing for ws in s.workers.values()):
        await asyncio.sleep(0.001)

    # Assert that *total* byte count in group determines group priority
    av = await c.scatter("a" * 100, workers=workers[0].address)
    bv = await c.scatter("b" * 75, workers=workers[2].address)
    cv = await c.scatter("c" * 75, workers=workers[3].address)

    assert set(s.workers_to_close(key=key)) == {workers[0].address, workers[1].address}


@pytest.mark.parametrize("reverse", [True, False])
@gen_cluster(client=True)
async def test_workers_to_close_never_close_long_running(c, s, a, b, reverse):
    if reverse:
        a, b = b, a
    wait_evt = Event()

    def executing(evt):
        evt.wait()

    def long_running_secede(evt):
        secede()
        evt.wait()

    assert a.address in s.workers_to_close()
    assert b.address in s.workers_to_close()
    long_fut = c.submit(long_running_secede, wait_evt, workers=[a.address])
    wsA = s.workers[a.address]
    while not wsA.long_running:
        await asyncio.sleep(0.01)
    assert s.workers_to_close() == [b.address]
    futs = [c.submit(executing, wait_evt, workers=[b.address]) for _ in range(10)]
    assert a.address not in s.workers_to_close(n=2)
    while not b.state.tasks:
        await asyncio.sleep(0.01)
    assert s.workers_to_close() == []
    assert s.workers_to_close(n=1) == [b.address]
    assert s.workers_to_close(n=2) == [b.address]

    await wait_evt.set()


@gen_cluster(client=True)
async def test_retire_workers_no_suspicious_tasks(c, s, a, b):
    future = c.submit(
        slowinc, 100, delay=0.5, workers=a.address, allow_other_workers=True
    )
    await asyncio.sleep(0.2)
    await s.retire_workers(workers=[a.address])

    assert all(ts.suspicious == 0 for ts in s.tasks.values())
    assert all(tp.suspicious == 0 for tp in s.task_prefixes.values())


@pytest.mark.slow
@pytest.mark.skipif(WINDOWS, reason="num_fds not supported on windows")
@gen_cluster(client=True, nthreads=[], timeout=120)
async def test_file_descriptors(c, s):
    await asyncio.sleep(0.1)
    pytest.importorskip("numpy")
    da = pytest.importorskip("dask.array")

    proc = psutil.Process()
    num_fds_1 = proc.num_fds()

    N = 20
    nannies = await asyncio.gather(*(Nanny(s.address) for _ in range(N)))

    while len(s.workers) < N:
        await asyncio.sleep(0.1)

    num_fds_2 = proc.num_fds()

    await asyncio.sleep(0.2)

    num_fds_3 = proc.num_fds()
    assert num_fds_3 <= num_fds_2 + N  # add some heartbeats

    x = da.random.random(size=(1000, 1000), chunks=(25, 25))
    x = c.persist(x)
    await wait(x)

    num_fds_4 = proc.num_fds()
    assert num_fds_4 <= num_fds_2 + 2 * N

    y = c.persist(x + x.T)
    await wait(y)

    num_fds_5 = proc.num_fds()
    assert num_fds_5 < num_fds_4 + N

    await asyncio.sleep(1)

    num_fds_6 = proc.num_fds()
    assert num_fds_6 < num_fds_5 + N

    await asyncio.gather(*(n.close() for n in nannies))
    await c.close()

    assert not s.rpc.open
    for occ in c.rpc.occupied.values():
        for comm in occ:
            assert comm.closed() or comm.peer_address != s.address, comm
    assert not s.stream_comms

    while proc.num_fds() > num_fds_1 + N:
        await asyncio.sleep(0.01)


@pytest.mark.slow
@nodebug
@gen_cluster(client=True)
async def test_learn_occupancy(c, s, a, b):
    futures = c.map(slowinc, range(1000), delay=0.2)
    while sum(len(ts.who_has or ()) for ts in s.tasks.values()) < 10:
        await asyncio.sleep(0.01)

    nproc = sum(ts.state == "processing" for ts in s.tasks.values())
    assert nproc * 0.1 < s.total_occupancy < nproc * 0.4
    for w in [a, b]:
        ws = s.workers[w.address]
        occ = ws.occupancy
        proc = len(ws.processing)
        assert proc * 0.1 < occ < proc * 0.4


@pytest.mark.slow
@nodebug
@gen_cluster(client=True)
async def test_learn_occupancy_2(c, s, a, b):
    future = c.map(slowinc, range(1000), delay=0.2)
    while not any(ts.who_has for ts in s.tasks.values()):
        await asyncio.sleep(0.01)

    nproc = sum(ts.state == "processing" for ts in s.tasks.values())
    assert nproc * 0.1 < s.total_occupancy < nproc * 0.4


@nodebug
@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 30)
async def test_balance_many_workers(c, s, *workers):
    futures = c.map(slowinc, range(20), delay=0.2)
    await wait(futures)
    assert {len(w.has_what) for w in s.workers.values()} == {0, 1}


# FIXME test is very timing-based; if some threads are consistently slower than others,
# they'll receive fewer tasks from the queue (a good thing).
@pytest.mark.skipif(
    MACOS
    and math.isfinite(
        float(dask.config.get("distributed.scheduler.worker-saturation"))
    ),
    reason="flaky on macOS with queuing active",
)
@nodebug
@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 1)] * 30,
    config={"distributed.scheduler.work-stealing": False},
)
async def test_balance_many_workers_2(c, s, *workers):
    futures = c.map(slowinc, range(90), delay=0.2)
    await wait(futures)
    assert {len(w.has_what) for w in s.workers.values()} == {3}


@gen_cluster(
    client=True,
    worker_kwargs={
        "heartbeat_interval": "10s",  # prevent worker from updating executing task durations
    },
)
async def test_include_communication_in_occupancy(c, s, a, b):
    x = c.submit(operator.mul, b"0", int(s.bandwidth) * 2, workers=a.address)
    y = c.submit(operator.mul, b"1", int(s.bandwidth * 3), workers=b.address)
    event = Event()

    def add_blocked(x, y, event):
        event.wait()
        return x + y

    with freeze_data_fetching(b):
        z = c.submit(add_blocked, x, y, event=event, pure=False)
        while z.key not in s.tasks or not s.tasks[z.key].processing_on:
            await asyncio.sleep(0.01)

        ts = s.tasks[z.key]
        ws = s.workers[b.address]
        assert ts.processing_on == ws
        # Occ should be 0.5s (CPU, unknown) + 2s (network)
        occ = ws.occupancy
        assert occ == 2.5
        z2 = c.submit(add_blocked, x, y, event=event, pure=False, workers=b.address)
        while z2.key not in s.tasks or not s.tasks[z2.key].processing_on:
            await asyncio.sleep(0.01)
        # Occ should be 2 * 0.5 (CPU, unknown) + 2s (network)
        # Network cost for the same key should only cost once
        occ2 = ws.occupancy
        assert occ2 == 3
    while s.tasks[x.key] not in ws.has_what:
        await asyncio.sleep(0.01)
    occ3 = ws.occupancy
    # Occ should be 2 * 0.5 (CPU, unknown)
    assert occ3 == 1
    await event.set()
    await wait(z)


@gen_test()
async def test_nonempty_data_is_rejected():
    with pytest.raises(ValueError, match="Worker.data must be empty"):
        await Worker("localhost:12345", nthreads=1, data={"x": 1})


@gen_cluster(client=True)
async def test_worker_arrives_with_data_is_rejected(c, s, a, b):
    # A worker arriving with data we need should still be rejected,
    # and not affect other computations
    x = delayed(slowinc)(1, delay=0.4)
    y = delayed(slowinc)(x, delay=0.4)
    z = delayed(slowinc)(y, delay=0.4)

    yy, zz = c.persist([y, z])

    while not any(w.processing for w in s.workers.values()):
        await asyncio.sleep(0.01)

    w = Worker(s.address, nthreads=1)
    w.update_data(data={y.key: 3})
    assert w.state.tasks[y.key].state == "memory"
    assert w.data == {y.key: 3}

    with pytest.raises(RuntimeError, match="Worker failed to start"):
        await w
    assert w.status == Status.failed
    assert len(s.workers) == 2

    await wait([yy, zz])


def test_run_on_scheduler_sync(loop):
    def f(dask_scheduler=None):
        return dask_scheduler.address

    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop) as c:
            address = c.run_on_scheduler(f)
            assert address == s["address"]

            with pytest.raises(ZeroDivisionError):
                c.run_on_scheduler(div, 1, 0)


@gen_cluster(client=True)
async def test_run_on_scheduler(c, s, a, b):
    def f(dask_scheduler=None):
        return dask_scheduler.address

    response = await c._run_on_scheduler(f)
    assert response == s.address


@gen_test()
async def test_allow_pickle_false():
    with dask.config.set({"distributed.scheduler.pickle": False}):
        with pytest.raises(RuntimeError, match="Pickling can no longer be disabled"):
            await Scheduler()


@gen_cluster()
async def test_close_worker(s, a, b):
    assert len(s.workers) == 2

    s.close_worker(a.address)
    while len(s.workers) != 1:
        await asyncio.sleep(0.01)
    assert a.address not in s.workers

    await asyncio.sleep(0.2)
    assert len(s.workers) == 1
    events = s.get_events(a.address)
    assert any(
        "reason" in msg
        for _, msg in events
        if "closing-worker" in msg.get("action", "")
    )


@pytest.mark.slow
@gen_cluster(Worker=Nanny)
async def test_close_nanny(s, a, b):
    assert len(s.workers) == 2

    assert a.process.is_alive()
    a_worker_address = a.worker_address

    await s.remove_worker(a_worker_address, stimulus_id="test")

    assert len(s.workers) == 1
    assert a_worker_address not in s.workers

    start = time()
    while a.is_alive():
        await asyncio.sleep(0.1)
        assert time() < start + 5

    assert not a.is_alive()
    assert a.pid is None

    for _ in range(10):
        await asyncio.sleep(0.1)
        assert len(s.workers) == 1
        assert not a.is_alive()
        assert a.pid is None

    while a.status != Status.closed:
        await asyncio.sleep(0.05)
        assert time() < start + 10


@gen_cluster(client=True)
async def test_retire_workers_close(c, s, a, b):
    await s.retire_workers(close_workers=True)
    assert not s.workers
    while a.status != Status.closed and b.status != Status.closed:
        await asyncio.sleep(0.01)


@pytest.mark.slow
@gen_cluster(client=True, Worker=Nanny)
async def test_retire_nannies_close(c, s, a, b):
    nannies = [a, b]
    await s.retire_workers(close_workers=True, remove=True)
    assert not s.workers

    start = time()

    while any(n.status != Status.closed for n in nannies):
        await asyncio.sleep(0.05)
        assert time() < start + 10

    assert not any(n.is_alive() for n in nannies)
    assert not s.workers


@gen_cluster(client=True, nthreads=[("127.0.0.1", 2)])
async def test_fifo_submission(c, s, w):
    futures = []
    for i in range(20):
        future = c.submit(slowinc, i, delay=0.1, key="inc-%02d" % i, fifo_timeout=0.01)
        futures.append(future)
        await asyncio.sleep(0.02)
    await wait(futures[-1])
    assert futures[10].status == "finished"


@gen_test()
async def test_scheduler_file():
    with tmpfile() as fn:
        async with Scheduler(scheduler_file=fn, dashboard_address=":0") as s:
            with open(fn) as f:
                data = json.load(f)
            assert data["address"] == s.address

            async with Client(scheduler_file=fn, loop=s.loop, asynchronous=True):
                pass


@pytest.mark.parametrize(
    "host", ["tcp://0.0.0.0", "tcp://127.0.0.1", "tcp://127.0.0.1:38275"]
)
@pytest.mark.parametrize(
    "dashboard_address,expect",
    [
        (None, ("::", "0.0.0.0", "127.0.0.1")),
        ("127.0.0.1:0", ("127.0.0.1",)),
    ],
)
@gen_test()
async def test_dashboard_host(host, dashboard_address, expect):
    """Dashboard is accessible from any host by default, but it can be also bound to
    localhost.
    """
    async with Scheduler(host=host, dashboard_address=dashboard_address) as s:
        sock = first(s.http_server._sockets.values())
        assert sock.getsockname()[0] in expect


@gen_cluster(client=True, worker_kwargs={"profile_cycle_interval": "100ms"})
async def test_profile_metadata(c, s, a, b):
    start = time() - 1
    futures = c.map(slowinc, range(10), delay=0.05, workers=a.address)
    await wait(futures)
    await asyncio.sleep(0.200)

    meta = await s.get_profile_metadata(profile_cycle_interval=0.100)
    now = time() + 1
    assert meta
    assert all(start < t < now for t, count in meta["counts"])
    assert all(0 <= count < 30 for t, count in meta["counts"][:4])
    assert not meta["counts"][-1][1]


@gen_cluster(
    client=True,
    config={
        "distributed.worker.profile.enabled": True,
        "distributed.worker.profile.cycle": "100ms",
    },
)
async def test_profile_metadata_timeout(c, s, a, b):
    start = time() - 1

    def raise_timeout(*args, **kwargs):
        raise TimeoutError

    b.handlers["profile_metadata"] = raise_timeout

    futures = c.map(slowinc, range(10), delay=0.05, workers=a.address)
    await wait(futures)
    await asyncio.sleep(0.200)

    meta = await s.get_profile_metadata(profile_cycle_interval=0.100)
    now = time() + 1
    assert meta
    assert all(start < t < now for t, count in meta["counts"])
    assert all(0 <= count < 30 for t, count in meta["counts"][:4])
    assert not meta["counts"][-1][1]


@gen_cluster(
    client=True,
    config={
        "distributed.worker.profile.enabled": True,
        "distributed.worker.profile.cycle": "100ms",
    },
)
async def test_profile_metadata_keys(c, s, a, b):
    x = c.map(slowinc, range(10), delay=0.05)
    y = c.map(slowdec, range(10), delay=0.05)
    await wait(x + y)

    meta = await s.get_profile_metadata(profile_cycle_interval=0.100)
    assert set(meta["keys"]) == {"slowinc", "slowdec"}
    assert (
        len(meta["counts"]) - 3 <= len(meta["keys"]["slowinc"]) <= len(meta["counts"])
    )


@gen_cluster(
    client=True,
    config={
        "distributed.worker.profile.enabled": True,
        "distributed.worker.profile.interval": "1ms",
        "distributed.worker.profile.cycle": "100ms",
    },
)
async def test_statistical_profiling(c, s, a, b):
    futures = c.map(slowinc, range(10), delay=0.1)

    await wait(futures)

    profile = await s.get_profile()
    assert profile["count"]


@gen_cluster(
    client=True,
    config={
        "distributed.worker.profile.enabled": True,
        "distributed.worker.profile.interval": "1ms",
        "distributed.worker.profile.cycle": "100ms",
    },
)
async def test_statistical_profiling_failure(c, s, a, b):
    futures = c.map(slowinc, range(10), delay=0.1)

    def raise_timeout(*args, **kwargs):
        raise TimeoutError

    b.handlers["profile"] = raise_timeout
    await wait(futures)

    profile = await s.get_profile()
    assert profile["count"]


@gen_cluster(client=True)
async def test_cancel_fire_and_forget(c, s, a, b):
    ev1 = Event()
    ev2 = Event()

    @delayed
    def f(_):
        pass

    @delayed
    def g(_, ev1, ev2):
        ev1.set()
        ev2.wait()

    x = f(None, dask_key_name="x")
    y = g(x, ev1, ev2, dask_key_name="y")
    z = f(y, dask_key_name="z")
    future = c.compute(z)

    fire_and_forget(future)
    await ev1.wait()
    # Cancel the future for z when
    # - x is in memory
    # - y is processing
    # - z is pending
    await future.cancel(force=True)
    assert future.status == "cancelled"
    while s.tasks:
        await asyncio.sleep(0.01)
    await ev2.set()


@pytest.mark.slow
@gen_cluster(
    client=True, Worker=Nanny, clean_kwargs={"processes": False, "threads": False}
)
async def test_log_tasks_during_restart(c, s, a, b):
    future = c.submit(sys.exit, 0)
    await wait(future)
    assert "exit" in str(
        {name: topic.events for name, topic in s._broker._topics.items()}
    )


@gen_cluster(client=True)
async def test_get_task_status(c, s, a, b):
    future = c.submit(inc, 1)
    await wait(future)

    result = await a.scheduler.get_task_status(keys=[future.key])
    assert result == {future.key: "memory"}


@gen_cluster(nthreads=[])
async def test_deque_handler(s):
    from distributed.scheduler import logger

    deque_handler = s._deque_handler
    logger.info("foo123")
    assert len(deque_handler.deque) >= 1
    msg = deque_handler.deque[-1]
    assert "distributed.scheduler" in deque_handler.format(msg)
    assert any(msg.msg == "foo123" for msg in deque_handler.deque)


@gen_cluster(client=True)
async def test_retries(c, s, a, b):
    args = [ZeroDivisionError("one"), ZeroDivisionError("two"), 42]

    future = c.submit(varying(args), retries=3)
    result = await future
    assert result == 42
    assert s.tasks[future.key].retries == 1
    assert not s.tasks[future.key].exception

    future = c.submit(varying(args), retries=2, pure=False)
    result = await future
    assert result == 42
    assert s.tasks[future.key].retries == 0
    assert not s.tasks[future.key].exception

    future = c.submit(varying(args), retries=1, pure=False)
    with pytest.raises(ZeroDivisionError) as exc_info:
        await future
    exc_info.match("two")

    future = c.submit(varying(args), retries=0, pure=False)
    with pytest.raises(ZeroDivisionError) as exc_info:
        await future
    exc_info.match("one")


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 3, config=NO_AMM)
async def test_missing_data_errant_worker(c, s, w1, w2, w3):
    with dask.config.set({"distributed.comm.timeouts.connect": "1s"}):
        np = pytest.importorskip("numpy")

        x = c.submit(np.random.random, 10000000, workers=w1.address)
        await wait(x)
        await c.replicate(x, workers=[w1.address, w2.address])

        y = c.submit(len, x, workers=w3.address)
        while not w3.state.tasks:
            await asyncio.sleep(0.001)
        await w1.close()
        await wait(y)


@gen_cluster(client=True)
async def test_dont_recompute_if_persisted(c, s, a, b):
    x = delayed(inc)(1, dask_key_name="x")
    y = delayed(inc)(x, dask_key_name="y")

    yy = y.persist()
    await wait(yy)

    old = list(s.transition_log)

    yyy = y.persist()
    await wait(yyy)

    await asyncio.sleep(0.100)
    assert list(s.transition_log) == old


@gen_cluster(client=True)
async def test_dont_recompute_if_persisted_2(c, s, a, b):
    x = delayed(inc)(1, dask_key_name="x")
    y = delayed(inc)(x, dask_key_name="y")
    z = delayed(inc)(y, dask_key_name="z")

    yy = y.persist()
    await wait(yy)

    old = s.story("x", "y")

    zz = z.persist()
    await wait(zz)

    await asyncio.sleep(0.100)
    assert s.story("x", "y") == old


@gen_cluster(client=True)
async def test_dont_recompute_if_persisted_3(c, s, a, b):
    x = delayed(inc)(1, dask_key_name="x")
    y = delayed(inc)(2, dask_key_name="y")
    z = delayed(inc)(y, dask_key_name="z")
    w = delayed(operator.add)(x, z, dask_key_name="w")

    ww = w.persist()
    await wait(ww)

    old = list(s.transition_log)

    www = w.persist()
    await wait(www)
    await asyncio.sleep(0.100)
    assert list(s.transition_log) == old


@gen_cluster(client=True)
async def test_dont_recompute_if_persisted_4(c, s, a, b):
    x = delayed(inc)(1, dask_key_name="x")
    y = delayed(inc)(x, dask_key_name="y")
    z = delayed(inc)(x, dask_key_name="z")

    yy = y.persist()
    await wait(yy)

    old = s.story("x")

    while s.tasks["x"].state == "memory":
        await asyncio.sleep(0.01)

    yyy, zzz = dask.persist(y, z)
    await wait([yyy, zzz])

    new = s.story("x")
    assert len(new) > len(old)


@gen_cluster(client=True)
async def test_dont_forget_released_keys(c, s, a, b):
    x = c.submit(inc, 1, key="x")
    y = c.submit(inc, x, key="y")
    z = c.submit(dec, x, key="z")
    del x
    await wait([y, z])
    del z

    while "z" in s.tasks:
        await asyncio.sleep(0.01)

    assert "x" in s.tasks


@gen_cluster(client=True)
async def test_dont_recompute_if_erred(c, s, a, b):
    x = delayed(inc)(1, dask_key_name="x")
    y = delayed(div)(x, 0, dask_key_name="y")

    yy = y.persist()
    await wait(yy)

    old = list(s.transition_log)

    yyy = y.persist()
    await wait(yyy)

    await asyncio.sleep(0.100)
    assert list(s.transition_log) == old


@gen_cluster()
async def test_closing_scheduler_closes_workers(s, a, b):
    await s.close()

    start = time()
    while a.status != Status.closed or b.status != Status.closed:
        await asyncio.sleep(0.01)
        assert time() < start + 2


@gen_cluster(client=True, nthreads=[("", 1)], worker_kwargs={"resources": {"A": 1}})
async def test_resources_reset_after_cancelled_task(c, s, a):
    lock = Lock()
    await lock.acquire()

    def block(lock):
        with lock:
            return

    assert s.workers[a.address].used_resources == {"A": 0}
    assert a.state.available_resources == {"A": 1}

    future = c.submit(block, lock, key="x", resources={"A": 1})
    await wait_for_state("x", "executing", a)
    assert s.workers[a.address].used_resources == {"A": 1}
    assert a.state.available_resources == {"A": 0}

    await future.cancel()
    await wait_for_state("x", "cancelled", a)
    assert s.workers[a.address].used_resources == {"A": 0}
    assert a.state.available_resources == {"A": 0}

    await lock.release()
    await async_poll_for(lambda: not a.state.tasks, timeout=5)
    assert s.workers[a.address].used_resources == {"A": 0}
    assert a.state.available_resources == {"A": 1}

    assert await c.submit(inc, 1, resources={"A": 1}) == 2
    assert s.workers[a.address].used_resources == {"A": 0}
    assert a.state.available_resources == {"A": 1}


@gen_cluster(client=True)
async def test_gh2187(c, s, a, b):
    def foo():
        return "foo"

    def bar(x):
        return x + "bar"

    def baz(x):
        return x + "baz"

    def qux(x):
        sleep(0.1)
        return x + "qux"

    w = c.submit(foo, key="w")
    x = c.submit(bar, w, key="x")
    y = c.submit(baz, x, key="y")
    await y
    z = c.submit(qux, y, key="z")
    del y
    await asyncio.sleep(0.1)
    f = c.submit(bar, x, key="y")
    await f


@gen_cluster(client=True)
async def test_collect_versions(c, s, a, b):
    cs = s.clients[c.id]
    (w1, w2) = s.workers.values()
    assert cs.versions
    assert w1.versions
    assert w2.versions
    assert "dask" in str(cs.versions)
    assert cs.versions == w1.versions == w2.versions


@pytest.mark.slow
@gen_cluster(client=True)
async def test_idle_timeout(c, s, a, b):
    beginning = time()
    assert s.check_idle() is not None
    assert s.check_idle() is not None  # Repeated calls should still not be None
    s.idle_timeout = 0.500
    pc = PeriodicCallback(s.check_idle, 10)
    future = c.submit(slowinc, 1)
    while not s.tasks:
        await asyncio.sleep(0.01)
    assert s.check_idle() is None
    pc.start()
    await future
    assert s.idle_since is None or s.idle_since > beginning
    _idle_since = s.check_idle()
    assert _idle_since == s.idle_since

    with captured_logger("distributed.scheduler") as caplog:
        start = time()
        while s.status != Status.closed:
            await asyncio.sleep(0.01)
            assert time() < start + 3

        start = time()
        while not (a.status == Status.closed and b.status == Status.closed):
            await asyncio.sleep(0.01)
            assert time() < start + 1

    logs = caplog.getvalue()
    assert "idle" in logs
    assert "500" in logs
    assert "ms" in logs
    assert "idle-timeout-exceeded" in logs
    assert s.idle_since > beginning
    pc.stop()


@gen_cluster(client=True)
async def test_idle_during_update_graph(c, s, a, b):
    class UpdateGraphTrackerPlugin(SchedulerPlugin):
        def start(self, scheduler):
            self.scheduler = scheduler
            self.idle_during_update_graph = None

        def update_graph(self, *args, **kwargs):
            self.idle_during_update_graph = self.scheduler.check_idle() is not None

    await c.register_plugin(UpdateGraphTrackerPlugin(), name="tracker")
    plugin = s.plugins["tracker"]
    # The cluster is idle because no work ever existed
    assert s.check_idle() is not None
    beginning = time()
    assert s.idle_since < beginning
    await c.submit(lambda x: x, 1)
    # The cluster may be considered not idle because of the unit of work
    s.check_idle()
    # Now the cluster must be idle
    assert s.check_idle() is not None
    end = time()
    assert beginning <= s.idle_since
    assert s.idle_since <= end
    # Ensure the cluster isn't idle while `Scheduler.update_graph` was being run
    assert plugin.idle_during_update_graph is False


@gen_cluster(client=True, nthreads=[])
async def test_idle_timeout_no_workers(c, s):
    """Test that idle-timeout is not triggered if there are no workers available
    but there are tasks queued
    """
    # Cancel the idle check periodic timeout so we can step through manually
    s.periodic_callbacks["idle-timeout"].stop()

    s.idle_timeout = 0.1
    future = c.submit(inc, 1)
    while not s.tasks:
        await asyncio.sleep(0.1)

    assert not s.check_idle()

    for _ in range(10):
        await asyncio.sleep(0.01)
        assert not s.check_idle()
        assert s.tasks

    async with Worker(s.address):
        await future
    assert not s.check_idle()
    del future

    while s.tasks:
        await asyncio.sleep(0.1)

    # We only set idleness once nothing happened between two consecutive
    # check_idle calls
    assert not s.check_idle()

    assert s.check_idle()


@gen_cluster(
    client=True,
    nthreads=[],
    config={"distributed.scheduler.no-workers-timeout": None},
)
async def test_no_workers_timeout_disabled(c, s):
    """no-workers-timeout has been disabled"""
    future = c.submit(inc, 1, key="x")
    await wait_for_state("x", ("queued", "no-worker"), s)

    s._check_no_workers()
    await asyncio.sleep(0.2)
    s._check_no_workers()
    await asyncio.sleep(0.2)

    async with Worker(s.address):
        await future

    assert all(
        event["action"] != "no-workers-timeout-exceeded"
        for _, event in s.get_events("scheduler")
    )


@pytest.mark.slow
@gen_cluster(
    client=True,
    nthreads=[],
    config={"distributed.scheduler.no-workers-timeout": "100ms"},
)
async def test_no_workers_timeout_without_workers(c, s):
    """Trip no-workers-timeout when there are no workers available"""
    future = c.submit(inc, 1, key="x")
    await wait_for_state("x", ("queued", "no-worker"), s)
    s._check_no_workers()
    await asyncio.sleep(0.2)
    s._check_no_workers()
    await asyncio.sleep(0.2)

    with pytest.raises(NoWorkerError if QUEUING_ON_BY_DEFAULT else NoValidWorkerError):
        await future

    events = [
        event
        for _, event in s.get_events("scheduler")
        if event["action"] == "no-workers-timeout-exceeded"
    ]
    assert len(events) == 1
    assert events[0]["keys"] == {"x"}


@pytest.mark.slow
@gen_cluster(
    client=True,
    config={"distributed.scheduler.no-workers-timeout": "100ms"},
)
async def test_no_workers_timeout_bad_restrictions(c, s, a, b):
    """Trip no-workers-timeout when there are workers available but none satisfies
    task restrictions
    """
    future = c.submit(inc, 1, key="x", workers=["127.0.0.2:1234"])
    with pytest.raises(NoValidWorkerError):
        await future

    events = [
        event
        for _, event in s.get_events("scheduler")
        if event["action"] == "no-workers-timeout-exceeded"
    ]
    assert len(events) == 1
    assert events[0]["keys"] == {"x"}


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    config={"distributed.scheduler.no-workers-timeout": "100ms"},
)
async def test_no_workers_timeout_queued(c, s, a):
    """Don't trip no-workers-timeout when there are queued tasks AND processing tasks"""
    ev = Event()
    futures = [c.submit(lambda ev: ev.wait(), ev, pure=False) for _ in range(3)]
    await async_poll_for(lambda: len(s.tasks) == 3 and a.state.tasks, timeout=5)
    assert s.queued or math.isinf(s.WORKER_SATURATION)

    s._check_no_workers()
    await asyncio.sleep(0.2)
    s._check_no_workers()
    await asyncio.sleep(0.2)

    await ev.set()
    await c.gather(futures)

    assert all(
        event["action"] != "no-workers-timeout-exceeded"
        for _, event in s.get_events("scheduler")
    )


@pytest.mark.slow
@gen_cluster(
    client=True,
    config={"distributed.scheduler.no-workers-timeout": "100ms"},
)
async def test_no_workers_timeout_processing(c, s, a, b):
    """Don't trip no-workers-timeout when there are tasks processing"""
    ev = Event()
    x = c.submit(lambda ev: ev.wait(), ev, key="x")
    y = c.submit(inc, 1, key="y", workers=["127.0.0.2:1234"])
    await wait_for_state("x", "processing", s)
    await wait_for_state("y", "no-worker", s)

    # Scheduler won't shut down for as long as f1 is running
    s._check_no_workers()
    await asyncio.sleep(0.2)
    s._check_no_workers()
    await asyncio.sleep(0.2)

    with pytest.raises(NoValidWorkerError):
        await y

    events = [
        event
        for _, event in s.get_events("scheduler")
        if event["action"] == "no-workers-timeout-exceeded"
    ]
    assert len(events) == 1
    assert events[0]["keys"] == {"y"}

    await ev.set()
    await x


@gen_cluster(client=True, config={"distributed.scheduler.bandwidth": "100 GB"})
async def test_bandwidth(c, s, a, b):
    start = s.bandwidth
    x = c.submit(operator.mul, b"0", 1000001, workers=a.address)
    y = c.submit(lambda x: x, x, workers=b.address)
    await y
    await b.heartbeat()
    assert s.bandwidth < start  # we've learned that we're slower
    assert b.latency
    assert typename(bytes) in s.bandwidth_types
    assert (b.address, a.address) in s.bandwidth_workers

    await a.close()
    assert not s.bandwidth_workers


@pytest.mark.slow
@gen_cluster(client=True, Worker=Nanny, timeout=60)
async def test_bandwidth_clear(c, s, a, b):
    np = pytest.importorskip("numpy")
    x = c.submit(np.arange, 1000000, workers=[a.worker_address], pure=False)
    y = c.submit(np.arange, 1000000, workers=[b.worker_address], pure=False)
    z = c.submit(operator.add, x, y)  # force communication
    await z

    async def f(dask_worker):
        await dask_worker.heartbeat()

    await c.run(f)

    assert s.bandwidth_workers

    await s.restart(stimulus_id="test")
    assert not s.bandwidth_workers


@gen_cluster()
async def test_workerstate_clean(s, a, b):
    ws = s.workers[a.address].clean()
    assert ws.address == a.address
    b = pickle.dumps(ws)
    assert len(b) < 1000


@gen_cluster(client=True)
async def test_result_type(c, s, a, b):
    x = c.submit(lambda: 1)
    await x

    assert "int" in s.tasks[x.key].type


@gen_cluster()
async def test_close_workers(s, *workers):
    await s.close()

    for w in workers:
        if not w.status == Status.closed:
            await asyncio.sleep(0.1)


@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
@gen_test()
async def test_host_address():
    s = await Scheduler(host="127.0.0.2", dashboard_address=":0")
    assert "127.0.0.2" in s.address
    await s.close()


@gen_test()
async def test_dashboard_address():
    pytest.importorskip("bokeh")
    async with Scheduler(dashboard_address="127.0.0.1:8901") as s:
        assert s.services["dashboard"].port == 8901

    async with Scheduler(dashboard_address="127.0.0.1") as s:
        assert s.services["dashboard"].port

    async with Scheduler(dashboard_address="127.0.0.1:8901,127.0.0.1:8902") as s:
        assert s.services["dashboard"].port == 8901

    async with Scheduler(dashboard_address=":8901,:8902") as s:
        assert s.services["dashboard"].port == 8901

    async with Scheduler(dashboard_address=[8901, 8902]) as s:
        assert s.services["dashboard"].port == 8901


@gen_cluster(client=True)
async def test_adaptive_target(c, s, a, b):
    with dask.config.set(
        {"distributed.scheduler.default-task-durations": {"slowinc": 10}}
    ):
        assert s.adaptive_target() == 0
        x = c.submit(inc, 1)
        await x
        assert s.adaptive_target() == 1

        # Long task
        x = c.submit(slowinc, 1, delay=0.4)
        while x.key not in s.tasks:
            await asyncio.sleep(0.01)
        assert s.adaptive_target(target_duration=".1s") == 1  # still one

        L = c.map(slowinc, range(100), delay=0.5)
        while len(s.tasks) < 100:
            await asyncio.sleep(0.01)
        assert 10 < s.adaptive_target(target_duration=".1s") <= 100
        del x, L
        while s.tasks:
            await asyncio.sleep(0.01)
        assert s.adaptive_target(target_duration=".1s") == 0


@pytest.mark.parametrize("queue", [True, False])
@gen_cluster(
    client=True,
    nthreads=[],
)
async def test_adaptive_target_empty_cluster(c, s, queue):
    if queue:
        s.WORKER_SATURATION = 1.0
    else:
        s.WORKER_SATURATION = float("inf")

    assert s.adaptive_target() == 0

    f = c.submit(inc, -1)
    await async_poll_for(lambda: s.tasks, timeout=5)
    assert s.adaptive_target() == 1
    del f

    if queue:
        # only queuing supports fast scale-up for empty clusters https://github.com/dask/distributed/issues/6962
        fs = c.map(inc, range(100))
        await async_poll_for(lambda: len(s.tasks) == len(fs), timeout=5)
        assert s.adaptive_target() > 1


@gen_test()
async def test_async_context_manager():
    async with Scheduler(dashboard_address=":0") as s:
        assert s.status == Status.running
        async with Worker(s.address) as w:
            assert w.status == Status.running
            assert s.workers
        assert not s.workers


@gen_test()
async def test_allowed_failures_config():
    async with Scheduler(dashboard_address=":0", allowed_failures=10) as s:
        assert s.allowed_failures == 10

    with dask.config.set({"distributed.scheduler.allowed_failures": 100}):
        async with Scheduler(dashboard_address=":0") as s:
            assert s.allowed_failures == 100

    with dask.config.set({"distributed.scheduler.allowed_failures": 0}):
        async with Scheduler(dashboard_address=":0") as s:
            assert s.allowed_failures == 0


@gen_test()
async def test_finished():
    async with Scheduler(dashboard_address=":0") as s:
        async with Worker(s.address) as w:
            pass

    await s.finished()
    await w.finished()


@gen_cluster(nthreads=[], client=True)
async def test_retire_names_str(c, s):
    async with Worker(s.address, name="0") as a, Worker(s.address, name="1") as b:
        futures = c.map(inc, range(5), workers=[a.address])
        futures.extend(c.map(inc, range(5, 10), workers=[b.address]))
        await wait(futures)
        assert a.data and b.data
        await s.retire_workers(names=[0])
        assert all(f.done() for f in futures)
        assert len(b.data) == 10


@gen_cluster(client=True)
async def test_retire_workers_bad_params(c, s, a, b):
    with pytest.raises(TypeError, match="mutually exclusive"):
        await s.retire_workers([a.address], names=[a.name])
    with pytest.raises(TypeError, match="mutually exclusive"):
        await s.retire_workers([a.address], n=1)
    with pytest.raises(TypeError, match="mutually exclusive"):
        await s.retire_workers(names=[a.name], n=1)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        await s.retire_workers(foo=1)


@gen_cluster(
    client=True, config={"distributed.scheduler.default-task-durations": {"inc": 100}}
)
async def test_get_task_duration(c, s, a, b):
    future = c.submit(inc, 1)
    await future
    assert 10 < s.task_prefixes["inc"].duration_average < 100

    ts_pref1 = s.new_task("inc-abcdefab", None, "released")
    assert 10 < s.get_task_duration(ts_pref1) < 100

    # make sure get_task_duration adds TaskStates to unknown dict
    assert len(s.unknown_durations) == 0
    x = c.submit(slowinc, 1, delay=0.5)
    while len(s.tasks) < 3:
        await asyncio.sleep(0.01)

    ts = s.tasks[x.key]
    assert s.get_task_duration(ts) == 0.5  # default
    assert len(s.unknown_durations) == 1
    assert len(s.unknown_durations["slowinc"]) == 1


@gen_cluster(client=True)
async def test_default_task_duration_splits(c, s, a, b):
    """Ensure that the default task durations for shuffle split tasks are
    aligned with the task names of dask.dask
    """
    pd = pytest.importorskip("pandas")
    dd = pytest.importorskip("dask.dataframe")

    # We don't care about the actual computation here but we'll schedule one anyway to
    # verify that we're looking for the correct key
    npart = 10
    df = dd.from_pandas(pd.DataFrame({"A": range(100), "B": 1}), npartitions=npart)
    with dask.config.set({"dataframe.shuffle.method": "tasks"}):
        graph = df.shuffle(
            "A",
            # If we don't have enough partitions, we'll fall back to a simple shuffle
            max_branch=npart - 1,
        ).sum()
    fut = c.compute(graph)
    await wait(fut)

    split_prefix = [pre for pre in s.task_prefixes.keys() if "split" in pre]
    # dask-expr enabled: ['split-taskshuffle', 'split-stage']
    # dask-expr disabled: ['split-shuffle']
    assert split_prefix
    default_times = dask.config.get("distributed.scheduler.default-task-durations")
    for p in split_prefix:
        default_time = parse_timedelta(default_times[p])
        assert default_time <= 1e-6


@gen_test()
async def test_no_dangling_asyncio_tasks():
    start = asyncio.all_tasks()
    async with Scheduler(dashboard_address=":0") as s:
        async with Worker(s.address, name="0"):
            async with Client(s.address, asynchronous=True) as c:
                await c.submit(lambda: 1)

    tasks = asyncio.all_tasks()
    assert tasks == start


@gen_cluster(client=True, Worker=NoSchedulerDelayWorker, config=NO_AMM)
async def test_task_group_and_prefix_statistics(c, s, a, b, no_time_resync):
    pytest.importorskip("numpy")
    da = pytest.importorskip("dask.array")

    start = time()
    x = da.arange(100, chunks=(20,))
    y = (x + 1).persist(optimize_graph=False)
    y = await y
    stop = time()

    tg = s.task_groups[x.name]
    tp = s.task_prefixes["arange"]
    repr(tg)
    repr(tp)
    assert tg.states["memory"] == 0
    assert tg.states["released"] == 5
    assert sum(tg.states.values()) == 5
    assert len(tg) == 5
    assert len(tp) == 5
    assert tg.nbytes_total == sum(
        ts.get_nbytes() for ts in s.tasks.values() if ts.group is tg
    )

    assert tg.prefix is tp
    assert tp.groups == {tg}
    with pytest.warns(FutureWarning, match="active"):
        assert tp.groups == tp.active
    # these must be true since in this simple case there is a 1to1 mapping
    # between prefix and group
    assert tg.states == tp.states
    with pytest.warns(FutureWarning, match="active_states"):
        assert tp.states == tp.active_states
    assert tg.duration == tp.duration
    assert tg.all_durations == tp.all_durations
    assert tg.nbytes_total == tp.nbytes_total
    assert tg.types == tp.types

    # It should map down to individual tasks
    tg = s.task_groups[y.name]
    assert tg.states["memory"] == 5
    assert sum(tg.states.values()) == 5
    assert len(tg) == 5
    assert len(tp) == 5

    tp = s.task_prefixes["add"]
    assert tg.prefix is tp
    assert tp.groups == {tg}
    with pytest.warns(FutureWarning, match="active"):
        assert tp.groups == tp.active
    # these must be true since in this simple case there is a 1to1 mapping
    # between prefix and group
    assert tg.states == tp.states
    with pytest.warns(FutureWarning, match="active_states"):
        assert tp.states == tp.active_states
    assert tg.duration == tp.duration
    assert tg.all_durations == tp.all_durations
    assert tg.nbytes_total == tp.nbytes_total
    assert tg.types == tp.types

    assert s.task_groups[y.name].dependencies == {s.task_groups[x.name]}

    await c.replicate(y)
    # TODO: Are we supposed to track replicated memory here? See also Scheduler.add_keys
    assert "array" in str(tg.types)
    assert "array" in str(tp.types)

    z = y[:20].persist(optimize_graph=False)
    z = await z
    del y

    while len(s.tasks) > 3:
        await asyncio.sleep(0.01)

    assert tg.prefix is tp
    assert tp.groups == {tg}
    with pytest.warns(FutureWarning, match="active"):
        assert tp.groups == tp.active
    assert tg.states["forgotten"] == 4
    assert tg.states["released"] == 1
    assert sum(tg.states.values()) == 5
    assert len(tg) == 5
    assert len(tp) == 5

    assert tg.states == tp.states
    with pytest.warns(FutureWarning, match="active_states"):
        assert tp.states == tp.active_states
    assert tg.duration == tp.duration
    assert tg.all_durations == tp.all_durations
    assert tg.nbytes_total == tp.nbytes_total
    assert tg.types == tp.types

    del z
    while s.tasks:
        await asyncio.sleep(0.01)

    assert tg.states["forgotten"] == 5
    assert sum(tg.states.values()) == 5
    assert len(tg) == 5

    assert tg.states["forgotten"] == 5
    assert tg.name not in s.task_groups
    assert tg.start > start
    assert tg.stop < stop
    assert "compute" in tg.all_durations

    assert tg.prefix is tp
    # all_durations is cumulative
    assert tg.all_durations == tp.all_durations
    # these must be zero because we remove fully-forgotten task groups
    # from the prefixes
    assert tp.groups == set()
    with pytest.warns(FutureWarning, match="active"):
        assert tp.groups == tp.active
    assert all(count == 0 for count in tp.states.values())
    with pytest.warns(FutureWarning, match="active_states"):
        assert tp.states == tp.active_states
    assert len(tp) == 0
    assert tp.duration == 0
    assert tp.nbytes_total == 0
    assert tp.types == set()


@gen_cluster(client=True, nthreads=[("", 2)], Worker=NoSchedulerDelayWorker)
async def test_task_groups_update_start_stop(c, s, a, no_time_resync):
    """TaskGroup.stop increases as the tasks in the group finish.
    TaskGroup.start can move backwards in the following use case:

    - task (x, 0) starts
    - task (x, 1) starts
    - task (x, 1) finishes
    - task (x, 0) finishes
    """
    ev = Event()
    t0 = await padded_time(before=0)
    x0 = c.submit(ev.wait, key=("x", 0))
    await wait_for_state(x0.key, "executing", a)
    tg = s.task_groups["x"]
    assert tg.start == tg.stop == 0
    t1 = await padded_time()
    x1 = c.submit(inc, 1, key=("x", 1))
    await x1
    t2 = await padded_time()
    assert t0 < t1 < tg.start < tg.stop < t2

    await ev.set()
    await x0
    t3 = await padded_time()
    assert t0 < tg.start < t1 < t2 < tg.stop < t3

    x2 = c.submit(inc, 1, key=("x", 2))
    await x2
    t4 = await padded_time(after=0)
    assert t0 < tg.start < t1 < t2 < t3 < tg.stop < t4


@gen_cluster(client=True)
async def test_task_group_done(c, s, a, b):
    """TaskGroup.done is True iff all of its tasks are in memory, erred, released, or
    forgotten state
    """
    x0 = c.submit(inc, 1, key=("x", 0))  # memory
    x1 = c.submit(lambda: 1 / 0, key=("x", 1))  # erred
    x2 = c.submit(inc, 3, key=("x", 2))

    x3 = c.submit(inc, 3, key=("x", 3))  # released
    y = c.submit(inc, x3, key="y")
    del x3

    await wait([x0, x1, x2, y])
    del x2  # forgotten
    await async_poll_for(lambda: ("x", 2) not in s.tasks, timeout=5)

    tg = s.task_groups["x"]
    assert tg.states == {
        "erred": 1,
        "forgotten": 1,
        "memory": 1,
        "no-worker": 0,
        "processing": 0,
        "queued": 0,
        "released": 1,
        "waiting": 0,
    }
    assert tg.done


@gen_cluster(client=True)
async def test_task_group_not_done_waiting(c, s, a, b):
    """TaskGroup.done is False if any of its tasks are in waiting state"""
    ev = Event()
    x = c.submit(ev.wait, key="x")
    y0 = c.submit(lambda x: x, x, key=("y", 0))
    y1 = c.submit(inc, 1, key=("y", 1))
    await wait_for_state(y0.key, "waiting", s)
    await y1

    tg = s.task_groups["y"]
    assert not tg.done
    await ev.set()


@gen_cluster(client=True)
async def test_task_group_not_done_noworker(c, s, a, b):
    """TaskGroup.done is False if any of its tasks are in no-worker state"""
    x0 = c.submit(inc, 1, key=("x", 0), resources={"X": 1})
    x1 = c.submit(inc, 1, key=("x", 1))
    await wait_for_state(x0.key, "no-worker", s)
    await x1

    tg = s.task_groups["x"]
    assert not tg.done


@gen_cluster(
    client=True,
    nthreads=[],
    config={"distributed.scheduler.worker-saturation": 1.0},
    timeout=3,
)
async def test_task_group_not_done_queued(c, s):
    """TaskGroup.done is False if any of its tasks are in queued state"""
    futs = c.map(inc, range(4))
    await wait_for_state(futs[0].key, "queued", s)
    tg = s.task_groups["inc"]
    assert not tg.done


@gen_cluster(client=True)
async def test_task_group_not_done_processing(c, s, a, b):
    """TaskGroup.done is False if any of its tasks are in processing state"""
    ev = Event()
    x0 = c.submit(ev.wait, key=("x", 0))
    x1 = c.submit(inc, 1, key=("x", 1))
    await wait_for_state(x0.key, "processing", s)
    await x1

    tg = s.task_groups["x"]
    assert not tg.done
    await ev.set()


@gen_cluster(client=True)
async def test_task_prefix(c, s, a, b):
    pytest.importorskip("numpy")
    da = pytest.importorskip("dask.array")
    x = da.arange(100, chunks=(20,))
    y = (x + 1).sum().persist()
    y = await y

    assert s.task_prefixes["sum-aggregate"].states["memory"] == 1

    a = da.arange(101, chunks=(20,))
    b = (a + 1).sum().persist()
    b = await b

    assert s.task_prefixes["sum-aggregate"].states["memory"] == 2


@pytest.mark.slow
@gen_cluster(
    client=True, Worker=Nanny, config={"distributed.scheduler.allowed-failures": 0}
)
async def test_failing_task_increments_suspicious(client, s, a, b):
    future = client.submit(sys.exit, 0)
    await wait(future)

    assert s.task_prefixes["exit"].suspicious == 1
    assert sum(tp.suspicious for tp in s.task_prefixes.values()) == sum(
        ts.suspicious for ts in s.tasks.values()
    )


@gen_cluster(client=True)
async def test_task_group_non_tuple_key(c, s, a, b):
    np = pytest.importorskip("numpy")
    da = pytest.importorskip("dask.array")
    x = da.arange(100, chunks=(20,))
    y = (x + 1).sum().persist()
    y = await y

    assert s.task_prefixes["sum"].states["released"] == 4
    assert "sum" not in s.task_groups

    f = c.submit(np.sum, [1, 2, 3])
    await f

    assert s.task_prefixes["sum"].states["released"] == 4
    assert s.task_prefixes["sum"].states["memory"] == 1
    assert "sum" in s.task_groups


@gen_cluster(client=True)
async def test_task_unique_groups(c, s, a, b):
    """This test ensure that task groups remain unique when using submit"""
    x = c.submit(sum, [1, 2])
    y = c.submit(len, [1, 2])
    z = c.submit(sum, [3, 4])
    await asyncio.gather(x, y, z)

    assert s.task_prefixes["len"].states["memory"] == 1
    assert s.task_prefixes["sum"].states["memory"] == 2


@gen_cluster(client=True)
async def test_task_group_on_fire_and_forget(c, s, a, b):
    # Regression test for https://github.com/dask/distributed/issues/3465
    with captured_logger("distributed.scheduler") as logs:
        x = await c.scatter(list(range(10)))
        fire_and_forget([c.submit(slowadd, i, x[i]) for i in range(len(x))])
        await asyncio.sleep(1)

    assert "Error transitioning" not in logs.getvalue()


class FlakyConnectionPool(ConnectionPool):
    def __init__(self, *args, failing_connections=0, **kwargs):
        self.cnn_count = 0
        self.failing_connections = failing_connections
        super().__init__(*args, **kwargs)

    async def connect(self, *args, **kwargs):
        self.cnn_count += 1
        if self.cnn_count > self.failing_connections:
            return await super().connect(*args, **kwargs)
        else:
            return BrokenComm()


@gen_cluster(client=True)
async def test_gather_failing_can_recover(c, s, a, b):
    x = await c.scatter({"x": 1}, workers=a.address)
    rpc = await FlakyConnectionPool(failing_connections=1)
    with (
        mock.patch.object(s, "rpc", rpc),
        dask.config.set({"distributed.comm.retry.count": 1}),
        captured_handler(
            logging.getLogger("distributed").handlers[0]
        ) as distributed_log,
    ):
        res = await s.gather(keys=["x"])
    assert re.match(
        r"\A\d+-\d+-\d+ \d+:\d+:\d+,\d+ - distributed.utils_comm - INFO - "
        r"Retrying get_data_from_worker after exception in attempt 0/1: \n\Z",
        distributed_log.getvalue(),
    )
    assert res["status"] == "OK"


@gen_cluster(client=True)
async def test_gather_failing_cnn_error(c, s, a, b):
    x = await c.scatter({"x": 1}, workers=a.address)
    rpc = await FlakyConnectionPool(failing_connections=10)
    with mock.patch.object(s, "rpc", rpc):
        res = await s.gather(keys=["x"])
    assert res["status"] == "error"
    assert list(res["keys"]) == ["x"]


@gen_cluster(client=True)
async def test_gather_no_workers(c, s, a, b):
    await asyncio.sleep(1)
    x = await c.scatter({"x": 1}, workers=a.address)

    await a.close()
    await b.close()

    res = await s.gather(keys=["x"])
    assert res["status"] == "error"
    assert list(res["keys"]) == ["x"]


@pytest.mark.parametrize("direct", [False, True])
@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    # This behaviour is independent of retries.
    # Disable them to reduce the complexity of this test.
    config={"distributed.comm.retry.count": 0},
)
async def test_gather_bad_worker(c, s, a, direct):
    """Upon connection failure, gather() tries again indefinitely and transparently,
    for as long as the batched comms channel is active.
    """
    x = c.submit(inc, 1, key="x")
    c.rpc = await FlakyConnectionPool(failing_connections=3)
    s.rpc = await FlakyConnectionPool(failing_connections=1)

    with captured_logger("distributed.scheduler") as sched_logger:
        with captured_logger("distributed.client") as client_logger:
            assert await c.gather(x, direct=direct) == 2

    assert "Couldn't gather keys: {'x': 'memory'}" in sched_logger.getvalue()
    assert "Couldn't gather 1 keys, rescheduling ('x',)" in client_logger.getvalue()

    if direct:
        # 1. try direct=True; fail
        # 2. fall back to direct=False; fail
        # 3. try direct=True again; fail
        # 4. fall back to direct=False again; success
        assert c.rpc.cnn_count == 2
        assert s.rpc.cnn_count == 2
    else:
        # 1. try direct=False; fail
        # 2. try again direct=False; success
        assert c.rpc.cnn_count == 0
        assert s.rpc.cnn_count == 2


@gen_cluster(client=True)
async def test_too_many_groups(c, s, a, b):
    x = dask.delayed(inc)(1)
    y = dask.delayed(dec)(2)
    z = dask.delayed(operator.add)(x, y)

    await c.compute(z)

    while s.tasks:
        await asyncio.sleep(0.01)

    assert len(s.task_groups) < 3


@gen_test()
@pytest.mark.parametrize(
    "dashboard_link_template,expected_dashboard_link",
    (
        ("{scheme}://{host}:{port}/status", r"dashboard at:\s*http://"),
        ("{ENV_VAR_MISSING}", r"dashboard at:\s*:\d*"),
    ),
)
async def test_multiple_listeners(dashboard_link_template, expected_dashboard_link):
    with dask.config.set({"distributed.dashboard.link": dashboard_link_template}):
        with captured_logger("distributed.scheduler") as log:
            async with Scheduler(
                dashboard_address=":0", protocol=["inproc", "tcp"]
            ) as s:
                async with Worker(s.listeners[0].contact_address) as a:
                    async with Worker(s.listeners[1].contact_address) as b:
                        assert a.address.startswith("inproc")
                        assert a.scheduler.address.startswith("inproc")
                        assert b.address.startswith("tcp")
                        assert b.scheduler.address.startswith("tcp")

                        async with Client(s.address, asynchronous=True) as c:
                            futures = c.map(inc, range(20))
                            await wait(futures)

                            # Force inter-worker communication both ways
                            await c.submit(sum, futures, workers=[a.address])
                            await c.submit(len, futures, workers=[b.address])

    log = log.getvalue()
    assert re.search(r"Scheduler at:\s*tcp://", log)
    assert re.search(r"Scheduler at:\s*inproc://", log)

    # Dashboard link formatting can fail if template contains env vars which aren't
    # present. Don't kill scheduler, but revert to outputting the port and helpful msg
    assert re.search(expected_dashboard_link, log)
    if "ENV_VAR_MISSING" in dashboard_link_template:
        msg = r"Failed to format dashboard link, unknown value: 'ENV_VAR_MISSING'"
        assert re.search(msg, log)


@gen_cluster(nthreads=[("127.0.0.1", 1)])
async def test_worker_name_collision(s, a):
    # test that a name collision for workers produces the expected response
    # and leaves the data structures of Scheduler in a good state
    # is not updated by the second worker
    with captured_logger("distributed.scheduler") as log:
        with raises_with_cause(
            RuntimeError, None, ValueError, f"name taken, {a.name!r}"
        ):
            await Worker(s.address, name=a.name, host="127.0.0.1")

    s.validate_state()
    assert set(s.workers) == {a.address}
    assert s.aliases == {a.name: a.address}

    log = log.getvalue()
    assert "duplicate" in log
    assert str(a.name) in log


@gen_cluster(client=True, config={"distributed.scheduler.unknown-task-duration": "1h"})
async def test_unknown_task_duration_config(client, s, a, b):
    future = client.submit(slowinc, 1)
    while not s.tasks:
        await asyncio.sleep(0.001)
    assert sum(s.get_task_duration(ts) for ts in s.tasks.values()) == 3600
    assert len(s.unknown_durations) == 1
    await wait(future)
    assert len(s.unknown_durations) == 0


@gen_cluster()
async def test_unknown_task_duration_config_2(s, a, b):
    assert s.idle_since == s.time_started


@gen_cluster(client=True)
async def test_retire_state_change(c, s, a, b):
    np = pytest.importorskip("numpy")
    y = c.map(lambda x: x**2, range(10))
    await c.scatter(y)
    coros = []
    for _ in range(2):
        v = c.map(lambda i: i * np.random.randint(1000), y)
        k = c.map(lambda i: i * np.random.randint(1000), v)
        foo = c.map(lambda j: j * 6, k)
        step = c.compute(foo)
        coros.append(c.gather(step))
    await c.retire_workers(workers=[a.address])
    await asyncio.gather(*coros)


@gen_cluster()
async def test_get_worker_monitor_info(s, a, b):
    res = await s.get_worker_monitor_info()
    ms = ["cpu", "time", "host_net_io.read_bps", "host_net_io.write_bps"]
    if not WINDOWS:
        ms += ["num_fds"]
    for w in (a, b):
        assert all(res[w.address]["range_query"][m] is not None for m in ms)
        assert res[w.address]["count"] is not None
        assert res[w.address]["last_time"] is not None


@gen_cluster(client=True)
async def test_quiet_cluster_round_robin(c, s, a, b):
    await c.submit(inc, 1)
    await c.submit(inc, 2)
    await c.submit(inc, 3)
    assert a.state.log and b.state.log


def test_memorystate():
    m = MemoryState(
        process=100,
        unmanaged_old=15,
        managed=68,
        spilled=12,
    )
    assert m.process == 100
    assert m.managed_total == 80
    assert m.managed == 68
    assert m.spilled == 12
    assert m.unmanaged == 32
    assert m.unmanaged_old == 15
    assert m.unmanaged_recent == 17
    assert m.optimistic == 83

    with pytest.warns(FutureWarning):
        assert m.managed_spilled == m.spilled
    with pytest.warns(FutureWarning):
        assert m.managed_in_memory == m.managed

    assert (
        repr(m)
        == dedent(
            """
            Process memory (RSS)  : 100 B
              - managed by Dask   : 68 B
              - unmanaged (old)   : 15 B
              - unmanaged (recent): 17 B
            Spilled to disk       : 12 B
            """
        ).lstrip()
    )


def test_memorystate_sum():
    m1 = MemoryState(
        process=100,
        unmanaged_old=15,
        managed=68,
        spilled=12,
    )
    m2 = MemoryState(
        process=80,
        unmanaged_old=10,
        managed=58,
        spilled=2,
    )
    m3 = MemoryState.sum(m1, m2)
    assert m3.process == 180
    assert m3.unmanaged_old == 25
    assert m3.managed_total == 140
    assert m3.spilled == 14


@pytest.mark.parametrize(
    "process,unmanaged_old,managed,spilled",
    list(product(*[[0, 1, 2, 3]] * 4)),
)
def test_memorystate_adds_up(process, unmanaged_old, managed, spilled):
    """Input data is massaged by __init__ so that everything adds up by construction"""
    m = MemoryState(
        process=process,
        unmanaged_old=unmanaged_old,
        managed=managed,
        spilled=spilled,
    )
    assert m.managed + m.unmanaged == m.process
    assert m.managed + m.spilled == m.managed_total
    assert m.unmanaged_old + m.unmanaged_recent == m.unmanaged
    assert m.optimistic + m.unmanaged_recent == m.process


def test_memorystate__to_dict():
    m = MemoryState(process=11, unmanaged_old=2, managed=3, spilled=1)
    assert m._to_dict() == {
        "managed": 3,
        "managed_total": 4,
        "optimistic": 5,
        "process": 11,
        "spilled": 1,
        "unmanaged": 8,
        "unmanaged_old": 2,
        "unmanaged_recent": 6,
    }


_test_leak = []


def leaking(out_mib, leak_mib, sleep_time):
    out = "x" * (out_mib * 2**20)
    _test_leak.append("x" * (leak_mib * 2**20))
    sleep(sleep_time)
    return out


def clear_leak():
    _test_leak.clear()


async def assert_memory(
    scheduler_or_workerstate: Scheduler | WorkerState,
    attr: str,
    /,
    min_mib: float,
    max_mib: float,
    *,
    timeout: float = 10,
) -> None:
    t0 = time()
    while True:
        minfo = scheduler_or_workerstate.memory
        nmib = getattr(minfo, attr) / 2**20
        if min_mib <= nmib <= max_mib:
            return
        if time() - t0 > timeout:
            raise AssertionError(
                f"Expected {min_mib} MiB <= {attr} <= {max_mib} MiB; got:\n{minfo!r}"
            )
        await asyncio.sleep(0.01)


@pytest.mark.slow
@gen_cluster(
    client=True,
    Worker=Nanny,
    config={
        "distributed.worker.memory.recent-to-old-time": "4s",
        "distributed.worker.memory.spill": 0.7,
        "distributed.worker.memory.spill-compression": "zlib",
    },
    worker_kwargs={
        "heartbeat_interval": "20ms",
        "memory_limit": "700 MiB",
    },
)
async def test_memory(c, s, *nannies):
    # WorkerState objects, as opposed to the Nanny objects passed by gen_cluster
    a, b = s.workers.values()

    def print_memory_info(msg: str) -> None:
        print(f"==== {msg} ====")
        print(f"---- a ----\n{a.memory}")
        print(f"---- b ----\n{b.memory}")
        print(f"---- s ----\n{s.memory}")

    s_m0 = s.memory
    assert s_m0.process == a.memory.process + b.memory.process
    assert s_m0.managed == 0
    assert a.memory.managed == 0
    assert b.memory.managed == 0

    # Trigger potential imports inside WorkerPlugin.transition
    await c.submit(inc, 0, workers=[a.address])
    await c.submit(inc, 1, workers=[b.address])
    # Wait for the memory readings to stabilize after workers go online
    await asyncio.sleep(2)
    await asyncio.gather(
        assert_memory(a, "unmanaged_recent", 0, 5, timeout=10),
        assert_memory(b, "unmanaged_recent", 0, 5, timeout=10),
        assert_memory(s, "unmanaged_recent", 0, 10, timeout=10.1),
    )

    print()
    print_memory_info("Starting memory")

    # 50 MiB heap + 100 MiB leak
    # Note that runtime=2s is less than recent-to-old-time=4s
    f1 = c.submit(leaking, 50, 100, 2, key="f1", workers=[a.name])
    f2 = c.submit(leaking, 50, 100, 2, key="f2", workers=[b.name])

    await asyncio.gather(
        assert_memory(a, "unmanaged_recent", 150, 170, timeout=1.8),
        assert_memory(b, "unmanaged_recent", 150, 170, timeout=1.8),
        assert_memory(s, "unmanaged_recent", 300, 340, timeout=1.9),
    )
    await wait([f1, f2])

    # On each worker, we now have 50 MiB managed + 100 MiB fresh leak
    await asyncio.gather(
        assert_memory(a, "managed", 50, 51, timeout=0),
        assert_memory(b, "managed", 50, 51, timeout=0),
        assert_memory(s, "managed", 100, 101, timeout=0),
        assert_memory(a, "unmanaged_recent", 100, 120, timeout=0),
        assert_memory(b, "unmanaged_recent", 100, 120, timeout=0),
        assert_memory(s, "unmanaged_recent", 200, 240, timeout=0),
    )

    # Force the output of f1 and f2 to spill to disk
    print_memory_info("Before spill")
    a_leak = round(700 * 0.7 - a.memory.process / 2**20)
    b_leak = round(700 * 0.7 - b.memory.process / 2**20)
    assert a_leak > 50 and b_leak > 50
    a_leak += 10
    b_leak += 10
    print(f"Leaking additional memory: {a_leak=}; {b_leak=}")
    await wait(
        [
            c.submit(leaking, 0, a_leak, 0, pure=False, workers=[a.name]),
            c.submit(leaking, 0, b_leak, 0, pure=False, workers=[b.name]),
        ]
    )

    # dask serialization compresses ("x" * 50 * 2**20) from 50 MiB to ~50 kiB.
    # Test that spilled reports the actual size on disk and not the output of
    # sizeof().
    await asyncio.gather(
        assert_memory(a, "spilled", 0.04, 0.08, timeout=3),
        assert_memory(b, "spilled", 0.04, 0.08, timeout=3),
        assert_memory(s, "spilled", 0.08, 0.16, timeout=3.1),
    )

    # FIXME on Windows and MacOS we occasionally observe managed = 49 bytes
    await asyncio.gather(
        assert_memory(a, "managed", 0, 0.1, timeout=0),
        assert_memory(b, "managed", 0, 0.1, timeout=0),
        assert_memory(s, "managed", 0, 0.1, timeout=0),
    )

    print_memory_info("After spill")

    # Delete spilled keys
    del f1
    del f2
    await asyncio.gather(
        assert_memory(a, "spilled", 0, 0, timeout=3),
        assert_memory(b, "spilled", 0, 0, timeout=3),
        assert_memory(s, "spilled", 0, 0, timeout=3.1),
    )

    print_memory_info("After clearing spilled keys")

    # Wait until 4s have passed since the spill to observe unmanaged_recent
    # transition into unmanaged_old
    await asyncio.gather(
        assert_memory(a, "unmanaged_recent", 0, 5, timeout=4.5),
        assert_memory(b, "unmanaged_recent", 0, 5, timeout=4.5),
        assert_memory(s, "unmanaged_recent", 0, 10, timeout=4.6),
    )

    # When the leaked memory is cleared, unmanaged and unmanaged_old drop.
    # On MacOS and Windows, the process memory of the Python interpreter does not shrink
    # as fast as on Linux. Note that this behaviour is heavily impacted by OS tweaks,
    # meaning that what you observe on your local host may behave differently on CI.
    if not LINUX:
        return

    print_memory_info("Before clearing memory leak")

    prev_unmanaged_a = a.memory.unmanaged / 2**20
    prev_unmanaged_b = b.memory.unmanaged / 2**20
    await c.run(clear_leak)

    await asyncio.gather(
        assert_memory(a, "unmanaged", 0, prev_unmanaged_a - 50, timeout=10),
        assert_memory(b, "unmanaged", 0, prev_unmanaged_b - 50, timeout=10),
    )
    await asyncio.gather(
        assert_memory(a, "unmanaged_recent", 0, 5, timeout=0),
        assert_memory(b, "unmanaged_recent", 0, 5, timeout=0),
    )


@gen_cluster(client=True, worker_kwargs={"memory_limit": 0})
async def test_memory_no_zict(c, s, a, b):
    """When Worker.data is not a SpillBuffer, test that querying spilled
    defaults to 0 and doesn't raise KeyError
    """
    await c.wait_for_workers(2)
    assert isinstance(a.data, dict)
    assert isinstance(b.data, dict)
    f = c.submit(leaking, 10, 0, 0)
    await f
    assert 10 * 2**20 < s.memory.managed < 11 * 2**20
    assert s.memory.spilled == 0


@gen_cluster(nthreads=[])
async def test_memory_no_workers(s):
    assert s.memory.process == 0
    assert s.memory.managed == 0


@gen_cluster(config={"distributed.admin.system-monitor.interval": "999s"})
async def test_infrequent_sysmon(s, a, b):
    """It doesn't matter how infrequently SystemMonitor.update() is called; there's
    always one invocation before the first heartbeat.
    """
    assert s.memory.process > 0


@gen_cluster()
async def test_close_scheduler__close_workers_Worker(s, a, b):
    with captured_logger("distributed.comm", level=logging.DEBUG) as log:
        await s.close()
        while not a.status == Status.closed:
            await asyncio.sleep(0.05)
    log = log.getvalue()
    assert "retry" not in log


@pytest.mark.slow
@gen_cluster(Worker=Nanny)
async def test_close_scheduler__close_workers_Nanny(s, a, b):
    with captured_logger("distributed.comm", level=logging.DEBUG) as log:
        await s.close()
        while not a.status == Status.closed:
            await asyncio.sleep(0.05)
    log = log.getvalue()
    assert "retry" not in log


async def assert_ndata(client, by_addr, total=None):
    """Test that the number of elements in Worker.data is as expected.
    To be used when the worker is wrapped by a nanny.

    by_addr: dict of either exact numbers or (min, max) tuples
    total: optional exact match on the total number of keys (with duplicates) across all
    workers
    """
    out = await client.run(lambda dask_worker: len(dask_worker.data))
    try:
        for k, v in by_addr.items():
            if isinstance(v, tuple):
                assert v[0] <= out[k] <= v[1]
            else:
                assert out[k] == v
        if total is not None:
            assert sum(out.values()) == total
    except AssertionError:
        raise AssertionError(f"Expected {by_addr}; {total=}; got {out}")


@pytest.mark.slow
@gen_cluster(
    client=True,
    Worker=Nanny,
    worker_kwargs={"memory_limit": "1 GiB"},
    config=merge(NO_AMM, {"distributed.worker.memory.rebalance.sender-min": 0.3}),
)
async def test_rebalance(c, s, a, b):
    # We used nannies to have separate processes for each worker
    # Generate 500 buffers worth 512 MiB total on worker a. This sends its memory
    # utilisation slightly above 50% (after counting unmanaged) which is above the
    # distributed.worker.memory.rebalance.sender-min threshold.
    futures = c.map(
        lambda _: "x" * (2**29 // 500), range(500), workers=[a.worker_address]
    )
    await wait(futures)
    # Wait for heartbeats
    await assert_memory(s, "process", 512, 1024)
    await assert_ndata(c, {a.worker_address: 500, b.worker_address: 0})
    await s.rebalance()
    # Allow for some uncertainty as the unmanaged memory is not stable
    await assert_ndata(
        c, {a.worker_address: (50, 450), b.worker_address: (50, 450)}, total=500
    )

    # rebalance() when there is nothing to do
    await s.rebalance()
    await assert_ndata(
        c, {a.worker_address: (50, 450), b.worker_address: (50, 450)}, total=500
    )


# Set rebalance() to work predictably on small amounts of managed memory. By default, it
# uses optimistic memory, which would only be possible to test by allocating very large
# amounts of managed memory, so that they would hide variations in unmanaged memory.
REBALANCE_MANAGED_CONFIG = merge(
    NO_AMM,
    {
        "distributed.worker.memory.rebalance.measure": "managed",
        "distributed.worker.memory.rebalance.sender-min": 0,
        "distributed.worker.memory.rebalance.sender-recipient-gap": 0,
    },
)


@gen_cluster(client=True, config=REBALANCE_MANAGED_CONFIG)
async def test_rebalance_managed_memory(c, s, a, b):
    futures = await c.scatter(range(100), workers=[a.address])
    assert len(a.data) == 100
    assert len(b.data) == 0
    await s.rebalance()
    assert len(a.data) == 50
    assert len(b.data) == 50


@gen_cluster(nthreads=[("", 1)] * 3, client=True, config=REBALANCE_MANAGED_CONFIG)
async def test_rebalance_workers_and_keys(client, s, a, b, c):
    futures = await client.scatter(range(100), workers=[a.address])
    assert (len(a.data), len(b.data), len(c.data)) == (100, 0, 0)

    # Passing empty iterables is not the same as omitting the arguments
    await s.rebalance(keys=[])
    await s.rebalance(workers=[])
    assert (len(a.data), len(b.data), len(c.data)) == (100, 0, 0)

    # Limit rebalancing to two arbitrary keys and two arbitrary workers.
    await s.rebalance(
        keys=[futures[3].key, futures[7].key], workers=[a.address, b.address]
    )
    assert (len(a.data), len(b.data), len(c.data)) == (98, 2, 0)

    with pytest.raises(KeyError):
        await s.rebalance(workers=["notexist"])


@gen_cluster(config=NO_AMM)
async def test_rebalance_missing_data1(s, a, b):
    """key never existed"""
    out = await s.rebalance(keys=["notexist"])
    assert out == {"status": "partial-fail", "keys": ["notexist"]}


@gen_cluster(client=True, config=NO_AMM)
async def test_rebalance_missing_data2(c, s, a, b):
    """keys exist but belong to unfinished futures. Unlike Client.rebalance(),
    Scheduler.rebalance() does not wait for unfinished futures.
    """
    futures = c.map(slowinc, range(10), delay=0.05, workers=a.address)
    await asyncio.sleep(0.1)
    out = await s.rebalance(keys=[f.key for f in futures])
    assert out["status"] == "partial-fail"
    assert 8 <= len(out["keys"]) <= 10


@pytest.mark.parametrize("explicit", [False, True])
@gen_cluster(client=True, config=REBALANCE_MANAGED_CONFIG)
async def test_rebalance_raises_missing_data3(c, s, a, b, explicit):
    """keys exist when the sync part of rebalance runs, but are gone by the time the
    actual data movement runs.
    There is an error message only if the keys are explicitly listed in the API call.
    """
    futures = await c.scatter(range(100), workers=[a.address])

    if explicit:
        pytest.xfail(
            reason="""Freeing keys and gathering data is using different
                   channels (stream vs explicit RPC). Therefore, the
                   partial-fail is very timing sensitive and subject to a race
                   condition. This test assumes that the data is freed before
                   the rebalance get_data requests come in but merely deleting
                   the futures is not sufficient to guarantee this"""
        )
        keys = [f.key for f in futures]
        del futures
        out = await s.rebalance(keys=keys)
        assert out["status"] == "partial-fail"
        assert 1 <= len(out["keys"]) <= 100
    else:
        del futures
        out = await s.rebalance()
        assert out == {"status": "OK"}


@gen_cluster(nthreads=[])
async def test_rebalance_no_workers(s):
    await s.rebalance()


@gen_cluster(
    client=True,
    worker_kwargs={"memory_limit": 0},
    config=merge(NO_AMM, {"distributed.worker.memory.rebalance.measure": "managed"}),
)
async def test_rebalance_no_limit(c, s, a, b):
    futures = await c.scatter(range(100), workers=[a.address])
    assert len(a.data) == 100
    assert len(b.data) == 0
    await s.rebalance()
    # Disabling memory_limit made us ignore all % thresholds set in the config
    assert len(a.data) == 50
    assert len(b.data) == 50


@pytest.mark.slow
@gen_cluster(
    client=True,
    Worker=Nanny,
    worker_kwargs={"memory_limit": "1000 MiB"},
    config=merge(
        NO_AMM,
        {
            "distributed.worker.memory.rebalance.measure": "managed",
            "distributed.worker.memory.rebalance.sender-min": 0.2,
            "distributed.worker.memory.rebalance.recipient-max": 0.1,
        },
    ),
)
async def test_rebalance_no_recipients(c, s, a, b):
    """There are sender workers, but no recipient workers"""
    # Fill 25% of the memory of a and 10% of the memory of b
    fut_a = c.map(lambda _: "x" * (2**20), range(250), workers=[a.worker_address])
    fut_b = c.map(lambda _: "x" * (2**20), range(100), workers=[b.worker_address])
    await wait(fut_a + fut_b)
    await assert_memory(s, "managed", 350, 351)
    await assert_ndata(c, {a.worker_address: 250, b.worker_address: 100})
    await s.rebalance()
    await assert_ndata(c, {a.worker_address: 250, b.worker_address: 100})


@gen_cluster(
    nthreads=[("", 1)] * 3,
    client=True,
    worker_kwargs={"memory_limit": 0},
    config=merge(NO_AMM, {"distributed.worker.memory.rebalance.measure": "managed"}),
)
async def test_rebalance_skip_recipient(client, s, a, b, c):
    """A recipient is skipped because it already holds a copy of the key to be sent"""
    futures = await client.scatter(range(10), workers=[a.address])
    await client.replicate(futures[0:2], workers=[a.address, b.address])
    await client.replicate(futures[2:4], workers=[a.address, c.address])
    assert (len(a.data), len(b.data), len(c.data)) == (10, 2, 2)
    await client.rebalance(futures[:2])
    assert (len(a.data), len(b.data), len(c.data)) == (8, 2, 4)


@gen_cluster(
    client=True,
    worker_kwargs={"memory_limit": 0},
    config=merge(NO_AMM, {"distributed.worker.memory.rebalance.measure": "managed"}),
)
async def test_rebalance_skip_all_recipients(c, s, a, b):
    """All recipients are skipped because they already hold copies"""
    futures = await c.scatter(range(10), workers=[a.address])
    await wait(futures)
    await c.replicate([futures[0]])
    assert (len(a.data), len(b.data)) == (10, 1)
    await c.rebalance(futures[:2])
    assert (len(a.data), len(b.data)) == (9, 2)


@pytest.mark.slow
@gen_cluster(
    client=True,
    Worker=Nanny,
    worker_kwargs={"memory_limit": "1000 MiB"},
    config=merge(NO_AMM, {"distributed.worker.memory.rebalance.measure": "managed"}),
)
async def test_rebalance_sender_below_mean(c, s, *_):
    """A task remains on the sender because moving it would send it below the mean"""
    a, b = s.workers
    f1 = c.submit(lambda: "x" * (400 * 2**20), workers=[a])
    await wait([f1])
    f2 = c.submit(lambda: "x" * (10 * 2**20), workers=[a])
    await wait([f2])
    await assert_memory(s, "managed", 410, 411)
    await assert_ndata(c, {a: 2, b: 0})
    await s.rebalance()
    assert await c.has_what() == {a: (f1.key,), b: (f2.key,)}


@pytest.mark.slow
@gen_cluster(
    client=True,
    Worker=Nanny,
    worker_kwargs={"memory_limit": "1000 MiB"},
    config=merge(
        NO_AMM,
        {
            "distributed.worker.memory.rebalance.measure": "managed",
            "distributed.worker.memory.rebalance.sender-min": 0.3,
        },
    ),
)
async def test_rebalance_least_recently_inserted_sender_min(c, s, *_):
    """
    1. keys are picked using a least recently inserted policy
    2. workers below sender-min are never senders
    """
    a, b = s.workers
    small_futures = c.map(lambda _: "x", range(10), workers=[a])
    await wait(small_futures)
    await assert_ndata(c, {a: 10, b: 0})
    await s.rebalance()
    await assert_ndata(c, {a: 10, b: 0})

    large_future = c.submit(lambda: "x" * (300 * 2**20), workers=[a])
    await wait([large_future])
    await assert_memory(s, "managed", 300, 301)
    await assert_ndata(c, {a: 11, b: 0})
    await s.rebalance()
    await assert_ndata(c, {a: 1, b: 10})
    has_what = await c.has_what()
    assert has_what[a] == (large_future.key,)
    assert sorted(has_what[b]) == sorted(f.key for f in small_futures)


@gen_cluster(client=True)
async def test_gather_on_worker(c, s, a, b):
    x = await c.scatter("x", workers=[a.address])
    x_ts = s.tasks[x.key]
    a_ws = s.workers[a.address]
    b_ws = s.workers[b.address]

    assert a_ws.nbytes > 0
    assert b_ws.nbytes == 0
    assert x_ts in a_ws.has_what
    assert x_ts not in b_ws.has_what
    assert x_ts.who_has == {a_ws}

    out = await s.gather_on_worker(b.address, {x.key: [a.address]})
    assert out == set()
    assert a.data[x.key] == "x"
    assert b.data[x.key] == "x"

    assert b_ws.nbytes == a_ws.nbytes
    assert x_ts in b_ws.has_what
    assert x_ts.who_has == {a_ws, b_ws}


@gen_cluster(client=True, scheduler_kwargs={"timeout": "100ms"})
async def test_gather_on_worker_bad_recipient(c, s, a, b):
    """The recipient is missing"""
    x = await c.scatter("x")
    await b.close()
    await async_poll_for(lambda: s.workers.keys() == {a.address}, timeout=5)
    out = await s.gather_on_worker(b.address, {x.key: [a.address]})
    assert out == {x.key}


@gen_cluster(client=True, worker_kwargs={"timeout": "100ms"})
async def test_gather_on_worker_bad_sender(c, s, a, b):
    """The only sender for a key is missing"""
    out = await s.gather_on_worker(a.address, {"x": ["tcp://127.0.0.1:12345"]})
    assert out == {"x"}


@pytest.mark.parametrize("missing_first", [False, True])
@gen_cluster(client=True, worker_kwargs={"timeout": "100ms"})
async def test_gather_on_worker_bad_sender_replicated(c, s, a, b, missing_first):
    """One of the senders for a key is missing, but the key is available somewhere else"""
    x = await c.scatter("x", workers=[a.address])
    bad_addr = "tcp://127.0.0.1:12345"
    # Order matters; test both
    addrs = [bad_addr, a.address] if missing_first else [a.address, bad_addr]
    out = await s.gather_on_worker(b.address, {x.key: addrs})
    assert out == set()
    assert a.data[x.key] == "x"
    assert b.data[x.key] == "x"


@gen_cluster(client=True)
async def test_gather_on_worker_key_not_on_sender(c, s, a, b):
    """The only sender for a key does not actually hold it"""
    out = await s.gather_on_worker(a.address, {"x": [b.address]})
    assert out == {"x"}


@pytest.mark.parametrize("missing_first", [False, True])
@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 3)
async def test_gather_on_worker_key_not_on_sender_replicated(
    client, s, a, b, c, missing_first
):
    """One of the senders for a key does not actually hold it, but the key is available
    somewhere else
    """
    x = await client.scatter("x", workers=[a.address])
    # Order matters; test both
    addrs = [b.address, a.address] if missing_first else [a.address, b.address]
    out = await s.gather_on_worker(c.address, {x.key: addrs})
    assert out == set()
    assert a.data[x.key] == "x"
    assert c.data[x.key] == "x"


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 3, config=NO_AMM)
async def test_gather_on_worker_duplicate_task(client, s, a, b, c):
    """Race condition where the recipient worker receives the same task twice.
    Test that the task nbytes are not double-counted on the recipient.
    """
    x = await client.scatter("x", workers=[a.address, b.address], broadcast=True)
    assert a.data[x.key] == "x"
    assert b.data[x.key] == "x"
    assert x.key not in c.data

    out = await asyncio.gather(
        s.gather_on_worker(c.address, {x.key: [a.address]}),
        s.gather_on_worker(c.address, {x.key: [b.address]}),
    )
    assert out == [set(), set()]
    assert c.data[x.key] == "x"

    a_ws = s.workers[a.address]
    b_ws = s.workers[b.address]
    c_ws = s.workers[c.address]
    assert a_ws.nbytes > 0
    assert c_ws.nbytes == b_ws.nbytes == a_ws.nbytes


@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 1)] * 3,
    scheduler_kwargs={"timeout": "100ms"},
    config=NO_AMM,
)
async def test_rebalance_dead_recipient(client, s, a, b, c):
    """A key fails to be rebalanced due to recipient failure.
    The key is not deleted from the sender.
    Unrelated, successful keys are deleted from the senders.
    """
    x, y = await client.scatter(["x", "y"], workers=[a.address])
    a_ws = s.workers[a.address]
    b_ws = s.workers[b.address]
    c_ws = s.workers[c.address]
    x_ts = s.tasks[x.key]
    y_ts = s.tasks[y.key]
    await c.close()
    assert s.workers.keys() == {a.address, b.address}

    out = await s._rebalance_move_data(
        [(a_ws, b_ws, x_ts), (a_ws, c_ws, y_ts)], stimulus_id="test"
    )
    assert out == {"status": "partial-fail", "keys": [y.key]}
    assert a.data == {y.key: "y"}
    assert b.data == {x.key: "x"}
    assert await client.has_what() == {a.address: (y.key,), b.address: (x.key,)}


@gen_cluster(client=True, config=NO_AMM)
async def test_delete_worker_data(c, s, a, b):
    # delete only copy of x
    # delete one of the copies of y
    # don't touch z
    x, y, z = await c.scatter(["x", "y", "z"], workers=[a.address])
    await c.replicate(y)

    assert a.data == {x.key: "x", y.key: "y", z.key: "z"}
    assert b.data == {y.key: "y"}
    assert s.tasks.keys() == {x.key, y.key, z.key}

    await s.delete_worker_data(a.address, [x.key, y.key], stimulus_id="test")
    assert a.data == {z.key: "z"}
    assert b.data == {y.key: "y"}
    assert s.tasks.keys() == {y.key, z.key}
    assert s.workers[a.address].nbytes == s.tasks[z.key].nbytes


@gen_cluster(nthreads=[("127.0.0.1", 1)], client=True)
async def test_delete_worker_data_double_delete(c, s, a):
    """_delete_worker_data race condition where the same key is deleted twice.
    WorkerState.nbytes is not double-decreased.
    """
    x, y = await c.scatter(["x", "y"])
    await asyncio.gather(
        s.delete_worker_data(a.address, [x.key], stimulus_id="test"),
        s.delete_worker_data(a.address, [x.key], stimulus_id="test"),
    )
    assert a.data == {y.key: "y"}
    a_ws = s.workers[a.address]
    y_ts = s.tasks[y.key]
    assert a_ws.nbytes == y_ts.nbytes


@gen_cluster(scheduler_kwargs={"timeout": "100ms"})
async def test_delete_worker_data_bad_worker(s, a, b):
    """_delete_worker_data gracefully handles a non-existing worker;
    e.g. a sender died in the middle of rebalance()
    """
    await a.close()
    assert s.workers.keys() == {b.address}
    await s.delete_worker_data(a.address, ["x"], stimulus_id="test")


@pytest.mark.parametrize("bad_first", [False, True])
@gen_cluster(nthreads=[("127.0.0.1", 1)], client=True)
async def test_delete_worker_data_bad_task(c, s, a, bad_first):
    """_delete_worker_data gracefully handles a non-existing key;
    e.g. a task was stolen by work stealing in the middle of a rebalance().
    Other tasks on the same worker are deleted.
    """
    x, y = await c.scatter(["x", "y"])
    assert a.data == {x.key: "x", y.key: "y"}
    assert s.tasks.keys() == {x.key, y.key}

    keys = ["notexist", x.key] if bad_first else [x.key, "notexist"]
    await s.delete_worker_data(a.address, keys, stimulus_id="test")
    assert a.data == {y.key: "y"}
    assert s.tasks.keys() == {y.key}
    assert s.workers[a.address].nbytes == s.tasks[y.key].nbytes


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_transition_counter(c, s, a):
    assert s.transition_counter == 0
    assert a.state.transition_counter == 0
    await c.submit(inc, 1)
    assert s.transition_counter > 1
    assert a.state.transition_counter > 1


@pytest.mark.slow
@gen_cluster(client=True)
async def test_transition_counter_max_scheduler(c, s, a, b):
    # This is set by @gen_cluster; it's False in production
    assert s.transition_counter_max > 0
    s.transition_counter_max = 1
    with captured_logger("distributed.scheduler") as logger:
        with pytest.raises(CancelledError):
            await c.submit(inc, 2)
    assert s.transition_counter > 1
    with pytest.raises(AssertionError):
        s.validate_state()
    assert "transition_counter_max" in logger.getvalue()
    # Scheduler state is corrupted. Avoid test failure on gen_cluster teardown.
    s.validate = False


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_transition_counter_max_worker(c, s, a):
    # This is set by @gen_cluster; it's False in production
    assert s.transition_counter_max > 0
    a.state.transition_counter_max = 1
    fut = c.submit(inc, 2)
    with captured_logger("distributed.worker") as logger:
        await async_poll_for(lambda: a.state.transition_counter > 0, timeout=5)

    assert "TransitionCounterMaxExceeded" in logger.getvalue()
    # Worker state is corrupted. Avoid test failure on gen_cluster teardown.
    a.state.validate = False


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    scheduler_kwargs={"transition_counter_max": False},
    worker_kwargs={"transition_counter_max": False},
)
async def test_disable_transition_counter_max(c, s, a, b):
    """Test that the cluster can run indefinitely if transition_counter_max is disabled.
    This is the default outside of @gen_cluster.
    """
    assert s.transition_counter_max is False
    assert a.state.transition_counter_max is False
    assert await c.submit(inc, 1) == 2
    assert s.transition_counter > 1
    assert a.state.transition_counter > 1
    s.validate_state()
    a.validate_state()


@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 1) for _ in range(10)],
)
async def test_worker_heartbeat_after_cancel(c, s, *workers):
    """This test is intended to ensure that after cancellation of a graph, the
    worker heartbeat is always successful. The heartbeat may not be successful if
    the worker and scheduler state drift and the scheduler doesn't handle
    unknown information gracefully. One example would be a released/cancelled
    computation where the worker returns metrics about duration, type, etc. and
    the scheduler doesn't handle the forgotten task gracefully.

    See also https://github.com/dask/distributed/issues/4587
    """
    for w in workers:
        w.periodic_callbacks["heartbeat"].stop()

    futs = c.map(slowinc, range(100), delay=0.1)

    while sum(w.state.executing_count for w in workers) < len(workers):
        await asyncio.sleep(0.001)

    await c.cancel(futs)

    while any(w.state.tasks for w in workers):
        await asyncio.gather(*(w.heartbeat() for w in workers))


@gen_cluster(client=True, nthreads=[("", 1)] * 2)
async def test_set_restrictions(c, s, a, b):
    f = c.submit(inc, 1, key="f", workers=[b.address])
    await f
    s.set_restrictions(worker={f.key: a.address})
    assert s.tasks[f.key].worker_restrictions == {a.address}
    await b.close()
    await f


@gen_cluster(
    client=True,
    nthreads=[("", 1)] * 3,
    config={"distributed.worker.memory.pause": False},
)
async def test_avoid_paused_workers(c, s, w1, w2, w3):
    w2.status = Status.paused
    while s.workers[w2.address].status != Status.paused:
        await asyncio.sleep(0.01)
    futures = c.map(slowinc, range(8), delay=0.1)
    await wait(futures)
    assert w1.data
    assert not w2.data
    assert w3.data
    assert len(w1.data) + len(w3.data) == 8


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_Scheduler__to_dict(c, s, a):
    futs = c.map(inc, range(2))

    await c.gather(futs)
    d = s._to_dict()
    assert d.keys() == {
        "type",
        "id",
        "address",
        "extensions",
        "services",
        "started",
        "workers",
        "status",
        "thread_id",
        "transition_log",
        "transition_counter",
        "memory",
        "tasks",
        "task_groups",
        "events",
        "clients",
    }
    # TaskStates are serialized as dicts under tasks and as strings under
    # workers.*.has_what and under clients.*.wants_what
    # WorkerStates are serialized s dicts under workers and as
    # strings under tasks.*.who_has
    assert d["tasks"][futs[0].key]["who_has"] == [
        f"<WorkerState '{a.address}', "
        "name: 0, status: running, memory: 2, processing: 0>"
    ]
    assert sorted(d["workers"][a.address]["has_what"]) == sorted(
        [
            f"<TaskState '{futs[0].key}' memory>",
            f"<TaskState '{futs[1].key}' memory>",
        ]
    )
    assert sorted(d["clients"][c.id]["wants_what"]) == sorted(
        [
            f"<TaskState '{futs[0].key}' memory>",
            f"<TaskState '{futs[1].key}' memory>",
        ]
    )

    # TaskGroups are serialized as dicts under task_groups and as strings under
    # tasks.*.group
    assert d["tasks"][futs[0].key]["group"] == "<inc: memory: 2>"
    assert d["task_groups"]["inc"]["prefix"] == "<inc: memory: 2>"

    # ClientStates are serialized as dicts under clients and as strings under
    # tasks.*.who_wants
    assert d["clients"][c.id]["client_key"] == c.id
    assert d["tasks"][futs[0].key]["who_wants"] == [f"<Client '{c.id}'>"]

    # Test MemoryState dump
    assert isinstance(d["memory"]["process"], int)
    assert isinstance(d["workers"][a.address]["memory"]["process"], int)


@gen_cluster(
    client=True, nthreads=[], config={"distributed.scheduler.worker-saturation": 1.0}
)
async def test_TaskState__to_dict(c, s):
    """tasks that are listed as dependencies of other tasks are dumped as a short repr
    and always appear in full under Scheduler.tasks
    """
    x = c.submit(inc, 1, key="x")
    y = c.submit(inc, x, key="y")
    z = c.submit(inc, 2, key="z")
    while len(s.tasks) < 3:
        await asyncio.sleep(0.01)

    tasks = s._to_dict()["tasks"]

    assert isinstance(tasks["x"], dict)
    assert isinstance(tasks["y"], dict)
    assert isinstance(tasks["z"], dict)
    assert tasks["x"]["dependents"] == ["<TaskState 'y' waiting>"]
    assert tasks["y"]["dependencies"] == ["<TaskState 'x' queued>"]


def _verify_cluster_state(
    state: dict,
    workers: Collection[Worker],
    allow_missing: bool = False,
) -> None:
    addrs = {w.address for w in workers}
    assert state.keys() == {"scheduler", "workers", "versions"}
    assert state["workers"].keys() == addrs
    if allow_missing:
        assert state["versions"]["workers"].keys() <= addrs
    else:
        assert state["versions"]["workers"].keys() == addrs


@gen_cluster(nthreads=[("", 1)] * 2)
async def test_get_cluster_state(s, *workers):
    state = await s.get_cluster_state([])
    _verify_cluster_state(state, workers)

    await asyncio.gather(*(w.close() for w in workers))

    while s.workers:
        await asyncio.sleep(0.01)

    state_no_workers = await s.get_cluster_state([])
    _verify_cluster_state(state_no_workers, [])


@gen_cluster(
    nthreads=[("", 1)] * 2,
    config={"distributed.comm.timeouts.connect": "200ms"},
)
async def test_get_cluster_state_worker_error(s, a, b):
    a.stop()
    state = await s.get_cluster_state([])
    _verify_cluster_state(state, [a, b], allow_missing=True)
    assert state["workers"][a.address] == (
        f"OSError('Timed out trying to connect to {a.address} after 0.2 s')"
    )
    assert isinstance(state["workers"][b.address], dict)
    assert state["versions"]["workers"].keys() == {b.address}


def _verify_cluster_dump(url: str, format: str, workers: Collection[Worker]) -> dict:
    import fsspec

    if format == "msgpack":
        import msgpack

        url += ".msgpack.gz"
        loader = msgpack.unpack

    else:
        import yaml

        url += ".yaml"
        loader = yaml.safe_load

    with fsspec.open(url, mode="rb", compression="infer") as f:
        state = loader(f)

    _verify_cluster_state(state, workers)
    return state


@pytest.mark.parametrize("format", ["msgpack", "yaml"])
@gen_cluster(nthreads=[("", 1)] * 2)
async def test_dump_cluster_state(s, *workers, format):
    fsspec = pytest.importorskip("fsspec")
    try:
        await s.dump_cluster_state_to_url(
            "memory://state-dumps/two-workers", [], format
        )
        _verify_cluster_dump("memory://state-dumps/two-workers", format, workers)

        await asyncio.gather(*(w.close() for w in workers))

        while s.workers:
            await asyncio.sleep(0.01)

        await s.dump_cluster_state_to_url("memory://state-dumps/no-workers", [], format)
        _verify_cluster_dump("memory://state-dumps/no-workers", format, [])
    finally:
        fs = fsspec.filesystem("memory")
        fs.rm("state-dumps", recursive=True)


@gen_cluster(nthreads=[("", 1)])
async def test_repr(s, a):
    async with Worker(s.address, nthreads=2) as b:  # name = address by default
        ws_a = s.workers[a.address]
        ws_b = s.workers[b.address]
        while ws_b.status != Status.running:
            await asyncio.sleep(0.01)
        assert repr(s) == f"<Scheduler {s.address!r}, workers: 2, cores: 3, tasks: 0>"
        assert (
            repr(a)
            == f"<Worker {a.address!r}, name: 0, status: running, stored: 0, running: 0/1, ready: 0, comm: 0, waiting: 0>"
        )
        assert (
            repr(b)
            == f"<Worker {b.address!r}, status: running, stored: 0, running: 0/2, ready: 0, comm: 0, waiting: 0>"
        )
        assert (
            repr(ws_a)
            == f"<WorkerState {a.address!r}, name: 0, status: running, memory: 0, processing: 0>"
        )
        assert (
            repr(ws_b)
            == f"<WorkerState {b.address!r}, status: running, memory: 0, processing: 0>"
        )


@gen_cluster(client=True, config={"distributed.comm.timeouts.connect": "2s"})
async def test_ensure_events_dont_include_taskstate_objects(c, s, a, b):
    event = Event()

    def block(x, event):
        event.wait()
        return x

    futs = c.map(block, range(100), event=event)
    while not a.state.tasks:
        await asyncio.sleep(0.1)

    await a.close(executor_wait=False)
    await event.set()
    await c.gather(futs)

    assert not any("TaskState" in str(event) for event in s.get_events())


@gen_cluster(nthreads=[("", 1)])
async def test_worker_state_unique_regardless_of_address(s, w):
    ws1 = s.workers[w.address]
    host, port = parse_host_port(ws1.address)
    await w.close()
    while s.workers:
        await asyncio.sleep(0.1)

    async with Worker(s.address, port=port, host=host) as w2:
        ws2 = s.workers[w2.address]

    assert ws1 is not ws2
    assert ws1 != ws2
    assert hash(ws1) != ws2


@gen_cluster(nthreads=[("", 1)])
async def test_scheduler_close_fast_deprecated(s, w):
    with pytest.warns(FutureWarning):
        await s.close(fast=True)


def test_runspec_regression_sync(loop):
    # https://github.com/dask/distributed/issues/6624
    np = pytest.importorskip("numpy")
    da = pytest.importorskip("dask.array")
    with Client(loop=loop):
        v = da.random.random((20, 20), chunks=(5, 5))

        overlapped = da.map_overlap(np.sum, v, depth=2, boundary="reflect")
        # This computation is somehow broken but we want to avoid catching any
        # serialization errors that result in KilledWorker
        with pytest.raises(IndexError):
            overlapped.compute()


@gen_cluster(config={"distributed.scheduler.allowed-failures": 666})
async def test_KilledWorker_informative_message(s, a, b):
    ws = s.workers[a.address].clean()
    ex = KilledWorker("foo-bar", ws, s.allowed_failures)
    with pytest.raises(KilledWorker) as excinfo:
        raise ex
    msg = str(excinfo.value)
    assert "Attempted to run task 'foo-bar' on 667 different workers" in msg
    assert a.address in msg
    assert "worker logs" in msg
    assert "https://distributed.dask.org/en/stable/killed.html" in msg


@gen_cluster(client=True)
async def test_count_task_prefix(c, s, a, b):
    futures = c.map(inc, range(10))
    await c.gather(futures)

    assert s.task_prefixes["inc"].state_counts["memory"] == 10
    assert s.task_prefixes["inc"].state_counts["erred"] == 0

    futures = c.map(inc, range(10, 20))
    await c.gather(futures)

    assert s.task_prefixes["inc"].state_counts["memory"] == 20
    assert s.task_prefixes["inc"].state_counts["erred"] == 0


@gen_cluster(client=True)
async def test_transition_waiting_memory(c, s, a, b):
    """Test race condition where a task transitions to memory while its state on the
    scheduler is waiting:

    1. worker a finishes x
    2. y transitions to processing and is assigned to worker b
    3. b fetches x and sends an add_keys message to the scheduler
    4. In the meantime, a dies and causes x to be scheduled back to released/waiting.
    5. Scheduler queues up a free-keys intended for b to cancel both x and y
    6. Before free-keys arrives to b, the worker runs and completes y, sending a
       finished-task message to the scheduler
    7. {op: add-keys, keys=[x]} from b finally arrives to the scheduler. This triggers
       a {op: remove-replicas, keys=[x]} message from the scheduler to worker b, because
       add-keys when the task state is not memory triggers a cleanup of redundant
       replicas (see Scheduler.add_keys) - in this, add-keys differs from task-finished!
    8. {op: task-finished, key=y} from b arrives to the scheduler and it is ignored.
    """
    x = c.submit(inc, 1, key="x", workers=[a.address])
    y = c.submit(inc, x, key="y", workers=[b.address])
    await wait_for_state("x", "memory", b, interval=0)
    # Note interval=0 above. It means that x has just landed on b this instant and the
    # scheduler doesn't know yet.
    assert b.state.tasks["y"].state == "executing"
    assert s.tasks["x"].who_has == {s.workers[a.address]}

    with freeze_batched_send(b.batched_stream):
        with freeze_batched_send(s.stream_comms[b.address]):
            await s.remove_worker(a.address, stimulus_id="remove_a")
            assert s.tasks["x"].state == "no-worker"
            assert s.tasks["y"].state == "waiting"
            await wait_for_state("y", "memory", b)

    await async_poll_for(lambda: not b.state.tasks, timeout=5)

    assert s.tasks["x"].state == "no-worker"
    assert s.tasks["y"].state == "waiting"
    assert_story(s.story("y"), [("y", "waiting", "waiting", {})])


@pytest.mark.parametrize(
    "rootish",
    [
        pytest.param(
            True,
            marks=pytest.mark.skipif(
                not QUEUING_ON_BY_DEFAULT,
                reason="Nothing will be classified as root-ish",
            ),
        ),
        False,
    ],
)
@gen_cluster(client=True, nthreads=[("", 1)])
async def test_deadlock_resubmit_queued_tasks_fast(c, s, a, rootish):
    # See https://github.com/dask/distributed/issues/7200
    block = Event()
    block2 = Event()
    executing = Event()
    executing2 = Event()

    def block_on_event(*args, block, executing):
        executing.set()
        block.wait()

    if rootish:
        ntasks = s.total_nthreads * 2 + 1
    else:
        ntasks = 1
    keys = [f"fut-{i}" for i in range(ntasks)]

    def submit_tasks():
        # Use case would be a client rescheduling the same or a similar graph
        # multiple times, closely followed
        # df.head()
        # df.size.compute()
        # We're emulating this by submitting the sames *keys*
        return c.map(
            block_on_event, range(len(keys)), block=block, executing=executing, key=keys
        )

    def assert_rootish():
        # Just to verify our assumptions in case the definition changes. This is
        # currently a bit brittle
        if rootish:
            assert all(s.is_rootish(s.tasks[k]) for k in keys)
        else:
            assert not any(s.is_rootish(s.tasks[k]) for k in keys)

    f1 = submit_tasks()
    # Make sure that the worker is properly saturated
    nblocking_tasks = 5

    # This set of tasks is there to guarantee that the worker is saturated after
    # releasing the first set of tasks s.t. a subsequent submission would run
    # into queuing
    fut2 = c.map(
        block_on_event, range(nblocking_tasks), block=block2, executing=executing2
    )

    # Once the task is on the threadpool, the client/scheduler may start its
    # release chain
    await executing.wait()

    assert len(a.state.tasks)
    # To trigger this condition, the scheduler needs to receive the
    # `task-finished` message after it performed the client release transitions
    # Therefore, the worker must not receive the `free-keys`` signal before it
    # can finish the task since otherwise the worker would recognize it as
    # cancelled and would forget about it. We emulate this behavior by blocking
    # the outgoing scheduler stream until that happens, i.e. this introduces
    # artifical latency
    with freeze_batched_send(s.stream_comms[a.address]):
        del f1
        while any(k in s.tasks for k in keys):
            await asyncio.sleep(0.005)

        assert len(s.tasks) == nblocking_tasks
        fut3 = submit_tasks()
        while len(s.tasks) == nblocking_tasks:
            await asyncio.sleep(0.005)
        assert_rootish()
        if rootish:
            assert all(s.tasks[k] in s.queued for k in keys), [s.tasks[k] for k in keys]
        await block.set()
        # At this point we need/want to wait for the task-finished message to
        # arrive on the scheduler. There is no proper hook to wait, therefore we
        # sleep
        await asyncio.sleep(0.2)
    # Everything should finish properly after this
    await block2.set()
    await c.gather(fut2)
    await c.gather(fut3)


@gen_test()
async def test_transition_failure_triggers_log_event():
    def block_on_event(input, block, executing):
        executing.set()
        block.wait()
        return input

    # Manually spin up cluster to avoid state validation on cluster shutdown in gen_cluster
    async with (
        Scheduler(dashboard_address=":0") as s,
        Worker(s.address) as w,
        Client(s.address, asynchronous=True) as c,
    ):
        block = Event()
        executing = Event()

        fut = c.submit(block_on_event, 0, block, executing)
        await executing.wait()

        # Manually corrupt the state of the processing task
        s.tasks[fut.key].processing_on = None

        await block.set()
        await async_poll_for(
            lambda: sum(
                event["action"] == "scheduler-transition-failed"
                for _, event in s.get_events("transitions")
            )
            == 1,
            timeout=5,
        )


@pytest.mark.skipif(
    not QUEUING_ON_BY_DEFAULT,
    reason="The situation handled in this test requires queueing.",
)
@pytest.mark.parametrize("validate", [True, False])
@gen_cluster(client=True, nthreads=[("", 1)])
async def test_deadlock_dependency_of_queued_released_when_worker_replaced(
    c, s, a, validate
):
    @delayed
    def inc(input):
        return input + 1

    @delayed
    def block_on_event(input, block, executing):
        executing.set()
        block.wait()
        return input

    block = Event()
    executing = Event()

    dep = inc(0)
    futs = [
        block_on_event(dep, block, executing, dask_key_name=("rootish", i))
        for i in range(s.total_nthreads * 2 + 1)
    ]
    del dep
    futs = c.compute(futs)
    await executing.wait()
    assert s.queued
    await s.remove_worker(address=a.address, stimulus_id="test")

    if validate:
        s.validate_state()

    await block.set()
    await executing.clear()

    async with Worker(s.address) as b:
        if validate:
            s.validate_state()
        await c.gather(futs)
        if validate:
            s.validate_state()


@pytest.mark.skipif(
    not QUEUING_ON_BY_DEFAULT,
    reason="The situation handled in this test requires queueing.",
)
@pytest.mark.parametrize("validate", [True, False])
@gen_cluster(client=True)
async def test_deadlock_dependency_of_queued_released_when_worker_removed(
    c, s, a, b, validate
):
    @delayed
    def inc(input):
        return input + 1

    @delayed
    def block_on_event(input, block):
        block.wait()
        return input

    block = Event()

    with dask.annotate(workers=a.address, allow_other_workers=True):
        dep = inc(0)
    futs = [
        block_on_event(dep, block, dask_key_name=("rootish", i))
        for i in range(s.total_nthreads * 2 + 1)
    ]
    dep.release()
    futs = c.compute(futs)
    with freeze_batched_send(b.batched_stream):
        await async_poll_for(
            lambda: b.state.tasks.get(dep.key) is not None
            and b.state.tasks.get(dep.key).state == "memory",
            timeout=5,
        )
        assert s.queued
        await s.remove_worker(address=a.address, stimulus_id="test")

    if validate:
        s.validate_state()

    await block.set()

    if validate:
        s.validate_state()
    await c.gather(futs)
    if validate:
        s.validate_state()


@gen_cluster(client=True)
async def test_submit_dependency_of_erred_task(c, s, a, b):
    x = c.submit(lambda: 1 / 0, key="x")
    await wait(x)
    y = c.submit(inc, x, key="y")
    with pytest.raises(ZeroDivisionError):
        await y


@gen_cluster(client=True)
async def test_tell_workers_when_peers_have_left(c, s, a, b):
    f = (await c.scatter({"f": 1}, workers=[a.address, b.address], broadcast=True))["f"]

    workers = {a.address: a, b.address: b}
    connect_timeout = parse_timedelta(
        dask.config.get("distributed.comm.timeouts.connect"), default="seconds"
    )

    class BrokenGatherDep(Worker):
        async def gather_dep(self, worker, *args, **kwargs):
            w = workers.pop(worker, None)
            if w is not None and workers:
                w.listener.stop()
                s.stream_comms[worker].abort()

            return await super().gather_dep(worker, *args, **kwargs)

    async with BrokenGatherDep(s.address, nthreads=1) as w3:
        start = time()
        g = await c.submit(inc, f, key="g", workers=[w3.address])
        # fails over to the second worker in less than the connect timeout
        assert time() < start + connect_timeout


@gen_cluster(client=True)
async def test_scatter_creates_ts(c, s, a, b):
    """A TaskState object is created by scatter, and only later becomes runnable

    See also
    --------
    test_scheduler.py::test_client_desires_keys_creates_ts
    test_spans.py::test_scatter_creates_ts
    """
    x1 = (await c.scatter({"x": 1}, workers=[a.address]))["x"]
    await wait_for_state("x", "memory", s)
    assert s.tasks["x"].run_spec is None
    async with Client(s.address, asynchronous=True) as c2:
        x2 = c2.submit(inc, 1, key="x")
        assert await x2 == 1
        await a.close()
        assert await x2 == 2
    assert s.tasks["x"].run_spec is not None


@pytest.mark.parametrize("finalize", [False, True])
@gen_cluster(
    client=True,
    nthreads=[("", 1)] * 4,
    worker_kwargs={"memory_limit": "100 kB"},
    config={
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": False,
        "distributed.worker.memory.pause": False,
    },
)
async def test_refuse_to_schedule_huge_task(c, s, *workers, finalize):
    """If the total size of a task's input grossly exceed the memory available on the
    worker, the scheduler must refuse to compute it
    """
    bg = bag.from_sequence(
        [random.randbytes(30_000) for _ in range(4)],
        npartitions=4,
    )
    match = r"worth of input dependencies, but worker .* has memory_limit set to"
    if finalize:
        fut = c.compute(bg)
        match += r".* you called client.compute()"
    else:
        bg = bg.repartition(npartitions=1).persist()
        fut = list(c.futures_of(bg))[0]

    with pytest.raises(MemoryError, match=match):
        await fut

    # The task never reached the workers
    for w in workers:
        for ev in w.state.log:
            assert fut.key not in ev


@gen_cluster(client=True)
async def test_html_repr(c, s, a, b):
    futs = c.map(slowinc, range(10), delay=0.1, key=[("slowinc", i) for i in range(10)])
    f = c.submit(sum, futs)

    while not f.done():
        assert isinstance(s._repr_html_(), str)
        assert isinstance(s.workers[a.address]._repr_html_(), str)
        assert isinstance(s.workers[b.address]._repr_html_(), str)
        for ts in s.tasks.values():
            assert isinstance(ts._repr_html_(), str)
        await asyncio.sleep(0.01)

    await f


@pytest.mark.parametrize("deps", ["same", "less", "more"])
@gen_cluster(client=True, nthreads=[])
async def test_resubmit_different_task_same_key_before_previous_is_done(c, s, deps):
    """If an intermediate key has a different run_spec (either the callable function or
    the dependencies / arguments) that will conflict with what was previously defined,
    it should raise an error since this can otherwise break in many different places and
    cause either spurious exceptions or even deadlocks.

    In this specific test, the previous run_spec has not been computed yet.
    See also test_resubmit_different_task_same_key_after_previous_is_done.

    For a real world example where this can trigger, see
    https://github.com/dask/dask/issues/9888
    """
    seen = False

    def _match(event):
        _, msg = event
        return (
            isinstance(msg, dict)
            and msg.get("action", None) == "update_graph"
            and msg["key-collisions"] > 0
        )

    def handler(ev):
        if _match(ev):
            nonlocal seen
            seen = True

    c.subscribe_topic("all", handler)

    x1 = c.submit(inc, 1, key="x1")
    y_old = c.submit(inc, x1, key="y")

    x1b = x1 if deps != "less" else 2
    x2 = delayed(inc)(10, dask_key_name="x2") if deps == "more" else 11
    y_new = delayed(sum)([x1b, x2], dask_key_name="y")
    z = delayed(inc)(y_new, dask_key_name="z")

    with captured_logger("distributed.scheduler", level=logging.WARNING) as log:
        fut = c.compute(z)
        await wait_for_state("z", "waiting", s)

    assert "Detected different `run_spec` for key 'y'" in log.getvalue()

    await async_poll_for(lambda: seen, timeout=5)

    async with Worker(s.address):
        # Used old run_spec
        assert await y_old == 3
        assert await fut == 4


@pytest.mark.parametrize("deps", ["same", "less", "more"])
@pytest.mark.parametrize("release_previous", [False, True])
@gen_cluster(client=True)
async def test_resubmit_different_task_same_key_after_previous_is_done(
    c, s, a, b, deps, release_previous
):
    """Same as test_resubmit_different_task_same_key, but now the replaced task has
    already been computed and is either in memory or released, and so are its old
    dependencies, so they may need to be recomputed.
    """
    x1 = delayed(inc)(1, dask_key_name="x1")
    x1fut = c.compute(x1)
    y_old = c.submit(inc, x1fut, key="y")
    z1 = c.submit(inc, y_old, key="z1")
    await wait(z1)
    if release_previous:
        del x1fut, y_old
        await wait_for_state("x1", "released", s)
        await wait_for_state("y", "released", s)

    x1b = x1 if deps != "less" else 2
    x2 = delayed(inc)(10, dask_key_name="x2") if deps == "more" else 11
    y_new = delayed(sum)([x1b, x2], dask_key_name="y")
    z2 = delayed(inc)(y_new, dask_key_name="z2")

    with captured_logger("distributed.scheduler", level=logging.WARNING) as log:
        fut = c.compute(z2)
        # Used old run_spec
        assert await fut == 4
        assert "x2" not in s.tasks

    # _generate_taskstates won't run for a dependency that's already in memory
    has_warning = "Detected different `run_spec` for key 'y'" in log.getvalue()
    assert has_warning is (release_previous or deps == "less")


@gen_cluster(client=True, nthreads=[])
async def test_resubmit_different_task_same_key_many_clients(c, s):
    """Two different clients submit a task with the same key but different run_spec's."""
    async with Client(s.address, asynchronous=True) as c2:
        with captured_logger("distributed.scheduler", level=logging.WARNING) as log:
            x1 = c.submit(inc, 1, key="x")
            x2 = c2.submit(inc, 2, key="x")

            await wait_for_state("x", ("no-worker", "queued"), s)
            who_wants = s.tasks["x"].who_wants
            await async_poll_for(
                lambda: {cs.client_key for cs in who_wants} == {c.id, c2.id}, timeout=5
            )

        assert "Detected different `run_spec` for key 'x'" in log.getvalue()

        async with Worker(s.address):
            assert await x1 == 2
            assert await x2 == 2  # kept old run_spec


@pytest.mark.parametrize(
    "before,after,expect_msg",
    [
        (object(), 123, True),
        (123, object(), True),
        (o := object(), o, False),
    ],
)
@gen_cluster(client=True, nthreads=[])
async def test_resubmit_nondeterministic_task_same_deps(
    c, s, before, after, expect_msg
):
    """Some run_specs can't be tokenized deterministically. Silently skip comparison on
    the run_spec when both lhs and rhs are nondeterministic.
    Dependencies must be the same.
    """
    x1 = c.submit(lambda x: x, before, key="x")
    x2 = delayed(lambda x: x)(after, dask_key_name="x")
    y = delayed(lambda x: x)(x2, dask_key_name="y")

    with captured_logger("distributed.scheduler", level=logging.WARNING) as log:
        fut = c.compute(y)
        await async_poll_for(lambda: "y" in s.tasks, timeout=5)

    has_msg = "Detected different `run_spec` for key 'x'" in log.getvalue()
    assert has_msg == expect_msg

    async with Worker(s.address):
        assert type(await fut) is type(before)


@pytest.mark.parametrize("add_deps", [False, True])
@gen_cluster(client=True, nthreads=[])
async def test_resubmit_nondeterministic_task_different_deps(c, s, add_deps):
    """Some run_specs can't be tokenized deterministically. Silently skip comparison on
    the run_spec in those cases. However, fail anyway if dependencies have changed.
    """
    o = object()
    x1 = c.submit(inc, 1, key="x1") if not add_deps else 2
    x2 = c.submit(inc, 2, key="x2")
    y1 = delayed(lambda i, j: i)(x1, o, dask_key_name="y").persist()
    y2 = delayed(lambda i, j: i)(x2, o, dask_key_name="y")
    z = delayed(inc)(y2, dask_key_name="z")

    with captured_logger("distributed.scheduler", level=logging.WARNING) as log:
        fut = c.compute(z)
        await wait_for_state("z", "waiting", s)
    assert "Detected different `run_spec` for key 'y'" in log.getvalue()

    async with Worker(s.address):
        assert await fut == 3


@pytest.mark.parametrize(
    "loglevel,expect_loglines", [(logging.DEBUG, 2), (logging.WARNING, 1)]
)
@gen_cluster(client=True, nthreads=[])
async def test_resubmit_different_task_same_key_warns_only_once(
    c, s, loglevel, expect_loglines
):
    """If all tasks of a layer are affected by the same run_spec collision, warn
    only once.
    """
    y1s = c.map(inc, [0, 1, 2], key=[("y", 0), ("y", 1), ("y", 2)])
    dsk = {
        "x": 3,
        ("y", 0): (inc, "x"),  # run_spec and dependencies change
        ("y", 1): (inc, 4),  # run_spec changes, dependencies don't
        ("y", 2): (inc, 2),  # Doesn't change
        ("z", 0): (inc, ("y", 0)),
        ("z", 1): (inc, ("y", 1)),
        ("z", 2): (inc, ("y", 2)),
    }
    with captured_logger("distributed.scheduler", level=loglevel) as log:
        zs = c.get(dsk, [("z", 0), ("z", 1), ("z", 2)], sync=False)
        await wait_for_state(("z", 2), "waiting", s)

    actual_loglines = len(
        re.findall("Detected different `run_spec` for key ", log.getvalue())
    )
    assert actual_loglines == expect_loglines

    async with Worker(s.address):
        assert await c.gather(zs) == [2, 3, 4]  # Kept old ys


def block(x, in_event, block_event):
    in_event.set()
    block_event.wait()
    return x


@gen_cluster(
    client=True,
    nthreads=[("", 1, {"resources": {"a": 1}})],
    config={"distributed.scheduler.allowed-failures": 0},
)
async def test_fan_out_pattern_deadlock(c, s, a):
    """Regression test for https://github.com/dask/distributed/issues/8548

    This test heavily uses resources to force scheduling decisions.
    """
    in_f, block_f = Event(), Event()
    in_ha, block_ha = Event(), Event()
    in_hb, block_hb = Event(), Event()

    # Input task to 'g' that we can fail
    with dask.annotate(resources={"b": 1}):
        f = delayed(block)(1, in_f, block_f, dask_key_name="f")
        g = delayed(inc)(f, dask_key_name="g")

        # Fan-out from 'g' and run h1 and h2 on different workers
        hb = delayed(block)(g, in_hb, block_hb, dask_key_name="hb")
    with dask.annotate(resources={"a": 1}):
        ha = delayed(block)(g, in_ha, block_ha, dask_key_name="ha")

    f, ha, hb = c.compute([f, ha, hb])
    with captured_logger("distributed.scheduler", level=logging.ERROR) as logger:
        async with Worker(s.address, nthreads=1, resources={"b": 1}) as b:
            await block_f.set()
            await in_ha.wait()
            await in_hb.wait()
            await in_f.clear()

            # Make sure that the scheduler knows that both workers hold 'g' in memory
            await async_poll_for(lambda: len(s.tasks["g"].who_has) == 2, timeout=5)
            # Remove worker 'b' while it's processing h1
            await s.remove_worker(b.address, stimulus_id="remove_b1")
            await block_hb.set()
        await block_f.clear()

        # Remove the new instance of the 'b' worker while it processes 'f'
        # to trigger an transition for 'f' to 'erred'
        async with Worker(s.address, nthreads=1, resources={"b": 1}) as b:
            await in_f.wait()
            await in_f.clear()
            await s.remove_worker(b.address, stimulus_id="remove_b2")
            await block_f.set()
        await block_f.clear()

        await block_ha.set()
        await ha

        with pytest.raises(KilledWorker, match="Attempted to run task 'hb'"):
            await hb

        del ha, hb
        # Make sure that h2 gets forgotten on worker 'a'
        await async_poll_for(lambda: not a.state.tasks, timeout=5)
    # Ensure that no other errors including transition failures were logged
    assert (
        logger.getvalue()
        == "Task hb marked as failed because 1 workers died while trying to run it\nTask f marked as failed because 1 workers died while trying to run it\n"
    )


@gen_cluster(
    client=True,
    nthreads=[("", 1, {"resources": {"a": 1}})],
    config={"distributed.scheduler.allowed-failures": 0},
)
async def test_stimulus_from_erred_task(c, s, a):
    """This test heavily uses resources to force scheduling decisions."""
    in_f, block_f = Event(), Event()
    in_g, block_g = Event(), Event()

    with dask.annotate(resources={"b": 1}):
        f = delayed(block)(1, in_f, block_f, dask_key_name="f")

    with dask.annotate(resources={"a": 1}):
        g = delayed(block)(f, in_g, block_g, dask_key_name="g")

    f, g = c.compute([f, g])
    with captured_logger("distributed.scheduler", level=logging.ERROR) as logger:
        frozen_stream_from_a_ctx = freeze_batched_send(a.batched_stream)
        frozen_stream_from_a_ctx.__enter__()

        async with Worker(s.address, nthreads=1, resources={"b": 1}) as b1:
            await block_f.set()
            await in_g.wait()
            await in_f.clear()
            frozen_stream_to_a_ctx = freeze_batched_send(s.stream_comms[a.address])
            frozen_stream_to_a_ctx.__enter__()
            await s.remove_worker(b1.address, stimulus_id="remove_b1")
            await block_f.clear()

        # Remove the new instance of the 'b' worker while it processes 'f'
        # to trigger a transition for 'f' to 'erred'
        async with Worker(s.address, nthreads=1, resources={"b": 1}) as b2:
            await in_f.wait()
            await in_f.clear()
            await s.remove_worker(b2.address, stimulus_id="remove_b2")
            await block_f.set()

        with pytest.raises(KilledWorker, match="Attempted to run task 'f'"):
            await f

        # g has already been transitioned to 'erred' because 'f' failed
        with pytest.raises(KilledWorker, match="Attempted to run task 'f'"):
            await g

        # Finish 'g' and let the scheduler know so it can trigger cleanup
        await block_g.set()
        with mock.patch.object(
            s, "stimulus_task_finished", wraps=s.stimulus_task_finished
        ) as wrapped_stimulus:
            frozen_stream_from_a_ctx.__exit__(None, None, None)
            # Make sure the `stimulus_task_finished` gets processed
            await async_poll_for(lambda: wrapped_stimulus.call_count == 1, timeout=5)

        # Allow the scheduler to talk to the worker again
        frozen_stream_to_a_ctx.__exit__(None, None, None)
        # Make sure all data gets forgotten on worker 'a'
        await async_poll_for(lambda: not a.state.tasks, timeout=5)

    # Ensure that no other errors including transition failures were logged
    assert (
        logger.getvalue()
        == "Task f marked as failed because 1 workers died while trying to run it\n"
    )


@gen_cluster(client=True)
async def test_concurrent_close_requests(c, s, *workers):
    class BeforeCloseCounterPlugin(SchedulerPlugin):
        async def start(self, scheduler):
            self.call_count = 0

        async def before_close(self):
            self.call_count += 1

    await c.register_plugin(BeforeCloseCounterPlugin(), name="before_close")
    with captured_logger("distributed.scheduler", level=logging.INFO) as caplog:
        await asyncio.gather(*[s.close(reason="test-reason") for _ in range(5)])
    assert s.plugins["before_close"].call_count == 1
    lines = caplog.getvalue().split("\n")
    assert sum("Closing scheduler" in line for line in lines) == 1

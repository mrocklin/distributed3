from __future__ import annotations

import asyncio
import glob
import logging
import os
import signal
from collections import Counter, UserDict
from time import sleep

import psutil
import pytest
from tlz import merge

import dask.config
from dask.utils import format_bytes, parse_bytes

import distributed.system
from distributed import Client, Event, KilledWorker, Nanny, Scheduler, Worker, wait
from distributed.compatibility import MACOS, WINDOWS
from distributed.core import Status
from distributed.metrics import monotonic
from distributed.utils import RateLimiterFilter
from distributed.utils_test import (
    NO_AMM,
    async_poll_for,
    captured_logger,
    gen_cluster,
    inc,
    wait_for_state,
)
from distributed.worker_memory import parse_memory_limit
from distributed.worker_state_machine import (
    ComputeTaskEvent,
    ExecuteSuccessEvent,
    GatherDep,
    GatherDepSuccessEvent,
    TaskErredMsg,
)


def memory_monitor_running(dask_worker: Worker | Nanny) -> bool:
    return "memory_monitor" in dask_worker.periodic_callbacks


def test_parse_memory_limit_zero():
    logger = logging.getLogger(__name__)
    assert parse_memory_limit(0, 1, logger=logger) is None
    assert parse_memory_limit("0", 1, logger=logger) is None
    assert parse_memory_limit(None, 1, logger=logger) is None


def test_resource_limit(monkeypatch):
    logger = logging.getLogger(__name__)
    assert parse_memory_limit("250MiB", 1, 1, logger=logger) == 1024 * 1024 * 250

    new_limit = 1024 * 1024 * 200
    monkeypatch.setattr(distributed.system, "MEMORY_LIMIT", new_limit)
    assert parse_memory_limit("250MiB", 1, 1, logger=logger) == new_limit


@gen_cluster(nthreads=[("", 1)], worker_kwargs={"memory_limit": "2e3 MB"})
async def test_parse_memory_limit_worker(s, w):
    assert w.memory_manager.memory_limit == 2e9


@gen_cluster(nthreads=[("", 1)], worker_kwargs={"memory_limit": "0.5"})
async def test_parse_memory_limit_worker_relative(s, w):
    assert w.memory_manager.memory_limit > 0.5
    assert w.memory_manager.memory_limit == pytest.approx(
        distributed.system.MEMORY_LIMIT * 0.5
    )


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    Worker=Nanny,
    worker_kwargs={"memory_limit": "2e3 MB"},
)
async def test_parse_memory_limit_nanny(c, s, n):
    assert n.memory_manager.memory_limit == 2e9
    out = await c.run(lambda dask_worker: dask_worker.memory_manager.memory_limit)
    assert out[n.worker_address] == 2e9


@gen_cluster(
    nthreads=[("127.0.0.1", 1)],
    config={
        "distributed.worker.memory.spill": False,
        "distributed.worker.memory.target": False,
    },
)
async def test_dict_data_if_no_spill_to_disk(s, w):
    assert type(w.data) is dict


class WorkerData(dict):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs


class WorkerDataLocalDirectory(dict):
    def __init__(self, worker_local_directory, **kwargs):
        super().__init__()
        self.local_directory = worker_local_directory
        self.kwargs = kwargs


@gen_cluster(
    nthreads=[("", 1)], Worker=Worker, worker_kwargs={"data": WorkerDataLocalDirectory}
)
async def test_worker_data_callable_local_directory(s, w):
    assert type(w.memory_manager.data) is WorkerDataLocalDirectory
    assert w.memory_manager.data.local_directory == w.local_directory


@gen_cluster(
    nthreads=[("", 1)],
    Worker=Worker,
    worker_kwargs={"data": (WorkerDataLocalDirectory, {"a": "b"})},
)
async def test_worker_data_callable_local_directory_kwargs(s, w):
    assert type(w.memory_manager.data) is WorkerDataLocalDirectory
    assert w.memory_manager.data.local_directory == w.local_directory
    assert w.memory_manager.data.kwargs == {"a": "b"}


@gen_cluster(
    nthreads=[("", 1)], Worker=Worker, worker_kwargs={"data": (WorkerData, {"a": "b"})}
)
async def test_worker_data_callable_kwargs(s, w):
    assert type(w.memory_manager.data) is WorkerData
    assert w.memory_manager.data.kwargs == {"a": "b"}


class CustomError(Exception):
    pass


class FailToPickle:
    def __init__(self, *, reported_size=0):
        self.reported_size = int(reported_size)

    def __getstate__(self):
        raise CustomError()

    def __sizeof__(self):
        return self.reported_size


async def assert_basic_futures(c: Client) -> None:
    futures = c.map(inc, range(10))
    results = await c.gather(futures)
    assert results == list(map(inc, range(10)))


@gen_cluster(client=True)
async def test_fail_to_pickle_execute_1(c, s, a, b):
    """Test failure to serialize triggered by computing a key which is individually
    larger than target. The data is lost and the task is marked as failed; the worker
    remains in usable condition.

    See also
    --------
    test_workerstate_fail_to_pickle_execute_1
    test_workerstate_fail_to_pickle_flight
    test_fail_to_pickle_execute_2
    test_fail_to_pickle_spill
    """
    x = c.submit(FailToPickle, reported_size=100e9, key="x")
    await wait(x)

    assert x.status == "error"

    with pytest.raises(TypeError, match="Failed to pickle 'x'") as e:
        await x
    assert isinstance(e.value.__cause__.__cause__.__cause__, CustomError)

    await assert_basic_futures(c)


class FailStoreDict(UserDict):
    def __setitem__(self, key, value):
        raise CustomError()


def test_workerstate_fail_to_pickle_execute_1(ws_with_running_task):
    """Same as test_fail_to_pickle_target_execute_1

    See also
    --------
    test_fail_to_pickle_execute_1
    test_workerstate_fail_to_pickle_flight
    test_fail_to_pickle_execute_2
    test_fail_to_pickle_spill
    """
    ws = ws_with_running_task
    assert not ws.data
    ws.data = FailStoreDict()

    instructions = ws.handle_stimulus(
        ExecuteSuccessEvent.dummy("x", None, stimulus_id="s1")
    )
    assert instructions == [
        TaskErredMsg.match(key="x", stimulus_id="s1"),
    ]
    assert ws.tasks["x"].state == "error"


def test_workerstate_fail_to_pickle_flight(ws):
    """Same as test_workerstate_fail_to_pickle_execute_1, but the task was
    computed on another host and for whatever reason it did not fail to pickle when it
    was sent over the network.

    See also
    --------
    test_fail_to_pickle_execute_1
    test_workerstate_fail_to_pickle_execute_1
    test_fail_to_pickle_execute_2
    test_fail_to_pickle_spill

    See also test_worker_state_machine.py::test_gather_dep_failure, where the task
    instead fails to unpickle when leaving the network stack.
    """
    assert not ws.data
    ws.data = FailStoreDict()
    ws.total_resources = {"R": 1}
    ws.available_resources = {"R": 1}
    ws2 = "127.0.0.1:2"

    instructions = ws.handle_stimulus(
        ComputeTaskEvent.dummy(
            "y", who_has={"x": [ws2]}, resource_restrictions={"R": 1}, stimulus_id="s1"
        ),
        GatherDepSuccessEvent(
            worker=ws2, total_nbytes=1, data={"x": 123}, stimulus_id="s2"
        ),
    )
    assert instructions == [
        GatherDep(worker=ws2, to_gather={"x"}, total_nbytes=1, stimulus_id="s1"),
        TaskErredMsg.match(key="x", stimulus_id="s2"),
    ]
    assert ws.tasks["x"].state == "error"
    assert ws.tasks["y"].state == "waiting"  # Not constrained

    # FIXME https://github.com/dask/distributed/issues/6705
    ws.validate = False


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    worker_kwargs={"memory_limit": "1 kiB"},
    config={
        "distributed.worker.memory.target": 0.5,
        "distributed.worker.memory.spill": False,
        "distributed.worker.memory.pause": False,
    },
)
async def test_fail_to_pickle_execute_2(c, s, a):
    """Test failure to spill triggered by computing a key which is individually smaller
    than target, so it is not spilled immediately. The data is retained and the task is
    NOT marked as failed; the worker remains in usable condition.

    See also
    --------
    test_fail_to_pickle_execute_1
    test_workerstate_fail_to_pickle_execute_1
    test_workerstate_fail_to_pickle_flight
    test_fail_to_pickle_spill
    """
    x = c.submit(FailToPickle, reported_size=256, key="x")
    await wait(x)
    assert x.status == "finished"
    assert set(a.data.memory) == {"x"}

    y = c.submit(lambda: "y" * 256, key="y")
    await wait(y)
    assert set(a.data.memory) == {"x", "y"}
    assert not a.data.disk
    await assert_basic_futures(c)


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    worker_kwargs={"memory_limit": "1 kB"},
    config={
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": 0.7,
        "distributed.worker.memory.monitor-interval": "100ms",
    },
)
async def test_fail_to_pickle_spill(c, s, a):
    """Test failure to evict a key, triggered by the spill threshold.

    See also
    --------
    test_fail_to_pickle_execute_1
    test_workerstate_fail_to_pickle_execute_1
    test_workerstate_fail_to_pickle_flight
    test_fail_to_pickle_execute_2
    """
    a.monitor.get_process_memory = lambda: 701 if a.data.fast else 0

    with captured_logger("distributed.spill") as logs:
        bad = c.submit(FailToPickle, key="bad")
        await wait(bad)

        # Must wait for memory monitor to kick in
        while True:
            logs_value = logs.getvalue()
            if logs_value:
                break
            await asyncio.sleep(0.01)

    assert "Failed to pickle" in logs_value
    assert "Traceback" in logs_value

    # key is in fast
    assert bad.status == "finished"
    assert bad.key in a.data.fast

    await assert_basic_futures(c)


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    worker_kwargs={"memory_limit": 1200 / 0.6},
    config={
        "distributed.worker.memory.target": 0.6,
        "distributed.worker.memory.spill": False,
        "distributed.worker.memory.pause": False,
    },
)
async def test_spill_target_threshold(c, s, a):
    """Test distributed.worker.memory.target threshold. Note that in this test we
    disabled spill and pause thresholds, which work on the process memory, and just left
    the target threshold, which works on managed memory so it is unperturbed by the
    several hundreds of MB of unmanaged memory that are typical of the test suite.
    """
    assert not memory_monitor_running(a)

    x = c.submit(lambda: "x" * 500, key="x")
    await wait(x)
    y = c.submit(lambda: "y" * 500, key="y")
    await wait(y)

    assert set(a.data) == {"x", "y"}
    assert set(a.data.memory) == {"x", "y"}

    z = c.submit(lambda: "z" * 500, key="z")
    await wait(z)
    assert set(a.data) == {"x", "y", "z"}
    assert set(a.data.memory) == {"y", "z"}
    assert set(a.data.disk) == {"x"}

    await x
    assert set(a.data.memory) == {"x", "z"}
    assert set(a.data.disk) == {"y"}


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    worker_kwargs={"memory_limit": 1600},
    config={
        "distributed.worker.memory.target": 0.6,
        "distributed.worker.memory.spill": False,
        "distributed.worker.memory.pause": False,
        "distributed.worker.memory.max-spill": 600,
    },
)
async def test_spill_constrained(c, s, w):
    """Test distributed.worker.memory.max-spill parameter"""
    # spills starts at 1600*0.6=960 bytes of managed memory

    # size in memory ~200; size on disk ~400
    x = c.submit(lambda: "x" * 200, key="x")
    await wait(x)
    # size in memory ~500; size on disk ~700
    y = c.submit(lambda: "y" * 500, key="y")
    await wait(y)

    assert set(w.data) == {x.key, y.key}
    assert set(w.data.memory) == {x.key, y.key}

    z = c.submit(lambda: "z" * 500, key="z")
    await wait(z)

    assert set(w.data) == {x.key, y.key, z.key}

    # max_spill has not been reached
    assert set(w.data.memory) == {y.key, z.key}
    assert set(w.data.disk) == {x.key}

    # zb is individually larger than max_spill
    zb = c.submit(lambda: "z" * 1700, key="zb")
    await wait(zb)

    assert set(w.data.memory) == {y.key, z.key, zb.key}
    assert set(w.data.disk) == {x.key}

    del zb
    while "zb" in w.data:
        await asyncio.sleep(0.01)

    # zc is individually smaller than max_spill, but the evicted key together with
    # x it exceeds max_spill
    zc = c.submit(lambda: "z" * 500, key="zc")
    await wait(zc)
    assert set(w.data.memory) == {y.key, z.key, zc.key}
    assert set(w.data.disk) == {x.key}


@gen_cluster(
    nthreads=[("", 1)],
    client=True,
    worker_kwargs={"memory_limit": "1000 MB"},
    config={
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": 0.7,
        "distributed.worker.memory.pause": False,
        "distributed.worker.memory.monitor-interval": "10ms",
    },
)
async def test_spill_spill_threshold(c, s, a):
    """Test distributed.worker.memory.spill threshold.
    Test that the spill threshold uses the process memory and not the managed memory
    reported by sizeof(), which may be inaccurate.
    """
    assert memory_monitor_running(a)
    a.monitor.get_process_memory = lambda: 800_000_000 if a.data.fast else 0
    x = c.submit(inc, 0, key="x")
    while not a.data.disk:
        await asyncio.sleep(0.01)
    assert await x == 1


@pytest.mark.parametrize(
    "target,managed,expect_spilled",
    [
        # no target -> no hysteresis
        # Over-report managed memory to test that the automated LRU eviction based on
        # target is never triggered
        (False, int(10e9), 1),
        # Under-report managed memory, so that we reach the spill threshold for process
        # memory without first reaching the target threshold for managed memory
        # target == spill -> no hysteresis
        (0.7, 0, 1),
        # target < spill -> hysteresis from spill to target
        (0.4, 0, 7),
    ],
)
@gen_cluster(
    nthreads=[],
    client=True,
    config={
        "distributed.worker.memory.spill": 0.7,
        "distributed.worker.memory.pause": False,
        "distributed.worker.memory.monitor-interval": "10ms",
    },
)
async def test_spill_hysteresis(c, s, target, managed, expect_spilled):
    """
    1. Test that you can enable the spill threshold while leaving the target threshold
       to False
    2. Test the hysteresis system where, once you reach the spill threshold, the worker
       won't stop spilling until the target threshold is reached
    """

    class C:
        def __sizeof__(self):
            return managed

    with dask.config.set({"distributed.worker.memory.target": target}):
        async with Worker(s.address, memory_limit="1000 MB") as a:
            a.monitor.get_process_memory = lambda: 50_000_000 * len(a.data.fast)

            # Add 500MB (reported) process memory. Spilling must not happen.
            futures = [c.submit(C, pure=False) for _ in range(10)]
            await wait(futures)
            await asyncio.sleep(0.1)
            assert not a.data.disk

            # Add another 250MB unmanaged memory. This must trigger the spilling.
            futures += [c.submit(C, pure=False) for _ in range(5)]
            await wait(futures)

            # Wait until spilling starts. Then, wait until it stops.
            prev_n = 0
            while not a.data.disk or len(a.data.disk) > prev_n:
                prev_n = len(a.data.disk)
                await asyncio.sleep(0)

            assert len(a.data.disk) == expect_spilled


@gen_cluster(
    nthreads=[("", 1)],
    client=True,
    config={
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": False,
        "distributed.worker.memory.pause": False,
    },
)
async def test_pause_executor_manual(c, s, a):
    assert not memory_monitor_running(a)

    # Task that is running when the worker pauses
    ev_x = Event()

    def f(ev):
        ev.wait()
        return 1

    # Task that is running on the worker when the worker pauses
    x = c.submit(f, ev_x, key="x")
    while a.state.executing_count != 1:
        await asyncio.sleep(0.01)

    # Task that is queued on the worker when the worker pauses
    y = c.submit(inc, 1, key="y")
    while "y" not in a.state.tasks:
        await asyncio.sleep(0.01)

    a.status = Status.paused
    # Wait for sync to scheduler
    while s.workers[a.address].status != Status.paused:
        await asyncio.sleep(0.01)

    # Task that is queued on the scheduler when the worker pauses.
    # It is not sent to the worker.
    z = c.submit(inc, 2, key="z")
    while "z" not in s.tasks or s.tasks["z"].state != "no-worker":
        await asyncio.sleep(0.01)
    assert s.unrunnable.keys() == {s.tasks["z"]}

    # Test that a task that already started when the worker paused can complete
    # and its output can be retrieved. Also test that the now free slot won't be
    # used by other tasks.
    await ev_x.set()
    assert await x == 1
    await asyncio.sleep(0.05)

    assert a.state.executing_count == 0
    assert len(a.state.ready) == 1
    assert a.state.tasks["y"].state == "ready"
    assert "z" not in a.state.tasks

    # Unpause. Tasks that were queued on the worker are executed.
    # Tasks that were stuck on the scheduler are sent to the worker and executed.
    a.status = Status.running
    assert await y == 2
    assert await z == 3


@gen_cluster(
    nthreads=[("", 1)],
    client=True,
    worker_kwargs={"memory_limit": "10 GB"},
    config={
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": False,
        "distributed.worker.memory.pause": 0.8,
        "distributed.worker.memory.monitor-interval": "10ms",
    },
)
async def test_pause_executor_with_memory_monitor(c, s, a):
    assert memory_monitor_running(a)
    mocked_rss = 0
    a.monitor.get_process_memory = lambda: mocked_rss

    # Task that is running when the worker pauses
    ev_x = Event()

    def f(ev):
        ev.wait()
        return 1

    # Task that is running on the worker when the worker pauses
    x = c.submit(f, ev_x, key="x")
    while a.state.executing_count != 1:
        await asyncio.sleep(0.01)

    with captured_logger("distributed.worker.memory") as logger:
        # Task that is queued on the worker when the worker pauses
        y = c.submit(inc, 1, key="y")
        while "y" not in a.state.tasks:
            await asyncio.sleep(0.01)

        # Hog the worker with 900GB unmanaged memory
        mocked_rss = 900 * 1000**3
        while s.workers[a.address].status != Status.paused:
            await asyncio.sleep(0.01)

        assert "Pausing worker" in logger.getvalue()

        # Task that is queued on the scheduler when the worker pauses.
        # It is not sent to the worker.
        z = c.submit(inc, 2, key="z")
        while "z" not in s.tasks or s.tasks["z"].state != "no-worker":
            await asyncio.sleep(0.01)
        assert s.unrunnable.keys() == {s.tasks["z"]}

        # Test that a task that already started when the worker paused can complete
        # and its output can be retrieved. Also test that the now free slot won't be
        # used by other tasks.
        await ev_x.set()
        assert await x == 1
        await asyncio.sleep(0.05)

        assert a.state.executing_count == 0
        assert len(a.state.ready) == 1
        assert a.state.tasks["y"].state == "ready"
        assert "z" not in a.state.tasks

        # Release the memory. Tasks that were queued on the worker are executed.
        # Tasks that were stuck on the scheduler are sent to the worker and executed.
        mocked_rss = 0
        assert await y == 2
        assert await z == 3

        assert a.status == Status.running
        assert "Resuming worker" in logger.getvalue()


@gen_cluster(
    client=True,
    nthreads=[("", 1), ("", 1)],
    config=merge(
        NO_AMM,
        {
            "distributed.worker.memory.target": False,
            "distributed.worker.memory.spill": False,
            "distributed.worker.memory.pause": False,
            "distributed.worker.memory.terminate": False,
        },
    ),
)
async def test_pause_prevents_deps_fetch(c, s, a, b):
    """A worker is paused while there are dependencies ready to fetch, but all other
    workers are in flight
    """
    a_addr = a.address

    class X:
        def __sizeof__(self):
            return 2**40  # Disable clustering in select_keys_for_gather

        def __reduce__(self):
            return X.pause_on_unpickle, ()

        @staticmethod
        def pause_on_unpickle():
            # Note: outside of task execution, distributed.get_worker()
            # returns a random worker running in the process
            for w in Worker._instances:
                if w.address == a_addr:
                    w.status = Status.paused
                    return X()
            assert False

    x = c.submit(X, key="x", workers=[b.address])
    y = c.submit(inc, 1, key="y", workers=[b.address])
    await wait([x, y])
    w = c.submit(lambda _: None, x, key="w", priority=1, workers=[a.address])
    z = c.submit(inc, y, key="z", priority=0, workers=[a.address])

    # - w and z reach worker a within the same message
    # - w and z respectively make x and y go into fetch state.
    #   w has a higher priority than z, therefore w's dependency x has a higher priority
    #   than z's dependency y.
    #   a.state.data_needed[b.address] = ["x", "y"]
    # - ensure_communicating decides to fetch x but not to fetch y together with it, as
    #   it thinks x is 1TB in size
    # - x fetch->flight; a is added to in_flight_workers
    # - y is skipped by ensure_communicating since all workers that hold a replica are
    #   in flight
    # - x reaches a and sends a into paused state
    # - x flight->memory; a is removed from in_flight_workers
    # - ensure_communicating is triggered again
    # - ensure_communicating refuses to fetch y because the worker is paused

    await wait_for_state("y", "fetch", a)
    await asyncio.sleep(0.1)
    assert a.state.tasks["y"].state == "fetch"
    assert "y" not in a.data
    assert [ts.key for ts in a.state.data_needed[b.address]] == ["y"]

    # Unpausing kicks off ensure_communicating again
    a.status = Status.running
    assert await z == 3
    assert a.state.tasks["y"].state == "memory"
    assert "y" in a.data


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    worker_kwargs={"memory_limit": 0},
    config={"distributed.worker.memory.monitor-interval": "10ms"},
)
async def test_avoid_memory_monitor_if_zero_limit_worker(c, s, a):
    assert type(a.data) is dict
    assert not memory_monitor_running(a)

    future = c.submit(inc, 1)
    assert await future == 2
    await asyncio.sleep(0.05)
    assert await c.submit(inc, 2) == 3  # worker doesn't pause


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    Worker=Nanny,
    worker_kwargs={"memory_limit": 0},
    config={"distributed.worker.memory.monitor-interval": "10ms"},
)
async def test_avoid_memory_monitor_if_zero_limit_nanny(c, s, nanny):
    typ = await c.run(lambda dask_worker: type(dask_worker.data))
    assert typ == {nanny.worker_address: dict}
    assert not memory_monitor_running(nanny)
    assert not (await c.run(memory_monitor_running))[nanny.worker_address]

    future = c.submit(inc, 1)
    assert await future == 2
    await asyncio.sleep(0.02)
    assert await c.submit(inc, 2) == 3  # worker doesn't pause


@gen_cluster(nthreads=[])
async def test_override_data_worker(s):
    # Use a UserDict to sidestep potential special case handling for dict
    async with Worker(s.address, data=UserDict) as w:
        assert type(w.data) is UserDict

    data = UserDict()
    async with Worker(s.address, data=data) as w:
        assert w.data is data


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    Worker=Nanny,
    worker_kwargs={"data": UserDict},
)
async def test_override_data_nanny(c, s, n):
    r = await c.run(lambda dask_worker: type(dask_worker.data))
    assert r[n.worker_address] is UserDict


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    worker_kwargs={"memory_limit": "10 GB", "data": UserDict},
    config={"distributed.worker.memory.monitor-interval": "10ms"},
)
async def test_override_data_vs_memory_monitor(c, s, a):
    a.monitor.get_process_memory = lambda: 8_100_000_000 if a.data else 0
    assert memory_monitor_running(a)

    # Push a key that would normally trip both the target and the spill thresholds
    class C:
        def __sizeof__(self):
            return 8_100_000_000

    # Capture output of log_errors()
    with captured_logger("distributed.utils") as logger:
        x = c.submit(C)
        await wait(x)

        # The pause subsystem of the memory monitor has been tripped.
        # The spill subsystem hasn't.
        while a.status != Status.paused:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)

    # This would happen if memory_monitor() tried to blindly call SpillBuffer.evict()
    assert "Traceback" not in logger.getvalue()

    assert type(a.data) is UserDict
    assert a.data.keys() == {x.key}


class ManualEvictDict(UserDict):
    """A MutableMapping which implements distributed.spill.ManualEvictProto"""

    def __init__(self):
        super().__init__()
        self.evicted = set()

    @property
    def fast(self):
        # Any Sized of bool will do
        return self.keys() - self.evicted

    def evict(self):
        # Evict a random key
        k = next(iter(self.fast))
        self.evicted.add(k)
        return 1


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    worker_kwargs={"memory_limit": "1 GB", "data": ManualEvictDict},
    config={
        "distributed.worker.memory.pause": False,
        "distributed.worker.memory.monitor-interval": "10ms",
    },
)
async def test_manual_evict_proto(c, s, a):
    """data is a third-party dict-like which respects the ManualEvictProto duck-type
    API. spill threshold is respected.
    """
    a.monitor.get_process_memory = lambda: 701_000_000 if a.data else 0
    assert memory_monitor_running(a)
    assert isinstance(a.data, ManualEvictDict)

    futures = await c.scatter({"x": None, "y": None, "z": None})
    while a.data.evicted != {"x", "y", "z"}:
        await asyncio.sleep(0.01)


async def leak_until_restart(c: Client, s: Scheduler) -> None:
    s.allowed_failures = 0

    def leak():
        L = []
        while True:
            L.append(b"0" * 5_000_000)
            sleep(0.01)

    (addr,) = s.workers
    pid = (await c.run(os.getpid))[addr]

    future = c.submit(leak, key="leak")

    # Wait until the worker is restarted
    while len(s.workers) != 1 or set(s.workers) == {addr}:
        await asyncio.sleep(0.01)

    # Test that the process has been properly waited for and not just left there
    with pytest.raises(psutil.NoSuchProcess):
        psutil.Process(pid)

    with pytest.raises(KilledWorker):
        await future
    assert s.tasks["leak"].suspicious == 1
    assert not any(
        (await c.run(lambda dask_worker: "leak" in dask_worker.state.tasks)).values()
    )
    future.release()
    while "leak" in s.tasks:
        await asyncio.sleep(0.01)


@pytest.mark.slow
@gen_cluster(
    nthreads=[("", 1)],
    client=True,
    Worker=Nanny,
    worker_kwargs={"memory_limit": "400 MiB"},
    config={"distributed.worker.memory.monitor-interval": "10ms"},
)
async def test_nanny_terminate(c, s, a):
    await leak_until_restart(c, s)


@pytest.mark.slow
@pytest.mark.parametrize(
    "ignore_sigterm",
    [
        False,
        pytest.param(True, marks=pytest.mark.skipif(WINDOWS, reason="Needs SIGKILL")),
    ],
)
@gen_cluster(
    nthreads=[("", 1)],
    client=True,
    Worker=Nanny,
    worker_kwargs={"memory_limit": "400 MiB"},
    config={"distributed.worker.memory.monitor-interval": "10ms"},
)
async def test_disk_cleanup_on_terminate(c, s, a, ignore_sigterm):
    """Test that the spilled data on disk is cleaned up when the nanny kills the worker.

    Unlike in a regular worker shutdown, where the worker deletes its own spill
    directory, the cleanup in case of termination from the monitor is performed by the
    nanny.

    The worker may be slow to accept SIGTERM, for whatever reason.
    At the next iteration of the memory manager, if the process is still alive, the
    nanny sends SIGKILL.
    """

    def do_ignore_sigterm():
        # ignore the return value of signal.signal:  it may not be serializable
        signal.signal(signal.SIGTERM, signal.SIG_IGN)

    if ignore_sigterm:
        await c.run(do_ignore_sigterm)

    fut = c.submit(inc, 1, key="myspill")
    await wait(fut)
    await c.run(lambda dask_worker: dask_worker.data.evict())
    glob_out = await c.run(
        # zict <= 2.2.0: myspill
        # zict >= 2.3.0: myspill#0
        lambda dask_worker: glob.glob(dask_worker.local_directory + "/**/myspill*")
    )
    spill_fname = glob_out[a.worker_address][0]
    assert os.path.exists(spill_fname)

    await leak_until_restart(c, s)
    assert not os.path.exists(spill_fname)


@gen_cluster(
    nthreads=[("", 1)],
    client=True,
    worker_kwargs={"memory_limit": "2 GiB"},
    # ^ must be smaller than system memory limit, otherwise that will take precedence
    config={
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": 0.5,
        "distributed.worker.memory.pause": 0.8,
        "distributed.worker.memory.monitor-interval": "10ms",
    },
)
async def test_pause_while_spilling(c, s, a):
    N_PAUSE = 3
    N_TOTAL = 5

    if a.memory_manager.memory_limit < parse_bytes("2 GiB"):
        pytest.fail(
            f"Set 2 GiB memory limit, got {format_bytes(a.memory_manager.memory_limit)}."
        )

    def get_process_memory():
        if len(a.data) < N_PAUSE:
            # Don't trigger spilling until after some tasks have completed
            return 0
        elif a.data.fast and not a.data.slow:
            # Trigger spilling
            return parse_bytes("1.6 GiB")
        else:
            # Trigger pause, but only after we started spilling
            return parse_bytes("1.9 GiB")

    a.monitor.get_process_memory = get_process_memory

    class SlowSpill:
        def __init__(self):
            # We need to record the worker while we are inside a task; can't do it in
            # __reduce__ or it will pick up an arbitrary one among all running workers
            self.worker = distributed.get_worker()
            while len(self.worker.data.fast) >= N_PAUSE:
                sleep(0.01)

        def __reduce__(self):
            paused = self.worker.status == Status.paused
            if not paused:
                sleep(0.1)
            return bool, (paused,)

    futs = [c.submit(SlowSpill, pure=False) for _ in range(N_TOTAL)]

    await async_poll_for(lambda: len(a.data.slow) >= N_PAUSE, timeout=5, period=0)
    assert a.status == Status.paused
    # Worker should have become paused after the first `SlowSpill` was evicted, because
    # the spill to disk took longer than the memory monitor interval.
    assert len(a.data.fast) == 0
    # With queuing enabled, after the 3rd `SlowSpill` has been created, there's a race
    # between the scheduler sending the worker a new task, and the memory monitor
    # running and pausing the worker. If the worker gets paused before the 4th task
    # lands, only 3 will be in memory. If after, the 4th will block on the semaphore
    # until one of the others is spilled.
    assert len(a.data.slow) in (N_PAUSE, N_PAUSE + 1)
    n_spilled_while_not_paused = sum(not paused for paused in a.data.slow.values())
    assert n_spilled_while_not_paused == 1


@pytest.mark.slow
@pytest.mark.skipif(
    condition=MACOS, reason="https://github.com/dask/distributed/issues/6233"
)
@gen_cluster(
    nthreads=[("", 1)],
    client=True,
    worker_kwargs={"memory_limit": "10 GiB"},
    config={
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": 0.6,
        "distributed.worker.memory.pause": False,
        "distributed.worker.memory.monitor-interval": "10ms",
    },
)
async def test_release_evloop_while_spilling(c, s, a):
    N = 100

    def get_process_memory():
        if len(a.data) < N:
            # Don't trigger spilling until after all tasks have completed
            return 0
        return 10 * 2**30

    a.monitor.get_process_memory = get_process_memory

    class SlowSpill:
        def __reduce__(self):
            sleep(0.01)
            return SlowSpill, ()

    futs = [c.submit(SlowSpill, pure=False) for _ in range(N)]
    while len(a.data) < N:
        await asyncio.sleep(0)

    ts = [monotonic()]
    while a.data.fast:
        await asyncio.sleep(0)
        ts.append(monotonic())

    # 100 tasks taking 0.01s to pickle each = 2s to spill everything
    # (this is because everything is pickled twice:
    # https://github.com/dask/distributed/issues/1371).
    # We should regain control of the event loop every 0.5s.
    c = Counter(round(t1 - t0, 1) for t0, t1 in zip(ts, ts[1:]))
    # Depending on the implementation of WorkerMemoryMonitor._maybe_spill:
    # if it calls sleep(0) every 0.5s:
    #   {0.0: 315, 0.5: 4}
    # if it calls sleep(0) after spilling each key:
    #   {0.0: 233}
    # if it never yields:
    #   {0.0: 359, 2.0: 1}
    # Make sure we remain in the first use case.
    assert 1 < sum(v for k, v in c.items() if 0.5 <= k <= 1.9), dict(c)
    assert not any(v for k, v in c.items() if k >= 2.0), dict(c)


@pytest.mark.parametrize(
    "cls,name,value",
    [
        (Worker, "memory_limit", 123e9),
        (Worker, "memory_target_fraction", 0.789),
        (Worker, "memory_spill_fraction", 0.789),
        (Worker, "memory_pause_fraction", 0.789),
        (Nanny, "memory_limit", 123e9),
        (Nanny, "memory_terminate_fraction", 0.789),
    ],
)
@gen_cluster(nthreads=[])
async def test_deprecated_attributes(s, cls, name, value):
    async with cls(s.address) as a:
        with pytest.warns(FutureWarning, match=name):
            setattr(a, name, value)
        with pytest.warns(FutureWarning, match=name):
            assert getattr(a, name) == value
        assert getattr(a.memory_manager, name) == value


@gen_cluster(nthreads=[("", 1)])
async def test_deprecated_memory_monitor_method_worker(s, a):
    with pytest.warns(FutureWarning, match="memory_monitor"):
        await a.memory_monitor()


@gen_cluster(nthreads=[("", 1)], Worker=Nanny)
async def test_deprecated_memory_monitor_method_nanny(s, a):
    with pytest.warns(FutureWarning, match="memory_monitor"):
        a.memory_monitor()


@pytest.mark.parametrize(
    "name",
    ["memory_target_fraction", "memory_spill_fraction", "memory_pause_fraction"],
)
@gen_cluster(nthreads=[])
async def test_deprecated_params(s, name):
    with pytest.warns(FutureWarning, match=name):
        async with Worker(s.address, **{name: 0.789}) as a:
            assert getattr(a.memory_manager, name) == 0.789


@gen_cluster(config={"distributed.worker.memory.monitor-interval": "10ms"})
async def test_pause_while_idle(s, a, b):
    sa = s.workers[a.address]
    assert a.address in s.idle
    assert sa in s.running

    a.monitor.get_process_memory = lambda: 2**40
    await async_poll_for(lambda: sa.status == Status.paused, timeout=5)
    assert a.address not in s.idle
    assert sa not in s.running

    a.monitor.get_process_memory = lambda: 0
    await async_poll_for(lambda: sa.status == Status.running, timeout=5)
    assert a.address in s.idle
    assert sa in s.running


@gen_cluster(client=True, config={"distributed.worker.memory.monitor-interval": "10ms"})
async def test_pause_while_saturated(c, s, a, b):
    sa = s.workers[a.address]
    ev = Event()
    futs = c.map(lambda i, ev: ev.wait(), range(3), ev=ev, workers=[a.address])
    await async_poll_for(lambda: len(a.state.tasks) == 3, timeout=5)
    assert sa in s.saturated
    assert sa in s.running

    a.monitor.get_process_memory = lambda: 2**40
    await async_poll_for(lambda: sa.status == Status.paused, timeout=5)
    assert sa not in s.saturated
    assert sa not in s.running

    a.monitor.get_process_memory = lambda: 0
    await async_poll_for(lambda: sa.status == Status.running, timeout=5)
    assert sa in s.saturated
    assert sa in s.running

    await ev.set()


@gen_cluster(nthreads=[])
async def test_worker_log_memory_limit_too_high(s):
    async with Worker(s.address, memory_limit="1 PB") as worker:
        assert any(
            "Ignoring provided memory limit" in record.msg for record in worker.logs
        )


@gen_cluster(
    nthreads=[],
    config={
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": 0.0001,
        "distributed.worker.memory.pause": False,
        "distributed.worker.memory.monitor-interval": "10ms",
    },
)
async def test_high_unmanaged_memory_warning(s):
    RateLimiterFilter.reset_timer("distributed.worker.memory")
    async with Worker(s.address) as worker:
        await asyncio.sleep(0.1)  # Enough for 10 runs of the memory monitors
    assert (
        sum("Unmanaged memory use is high" in record.msg for record in worker.logs) == 1
    )  # Message is rate limited


class WriteOnlyBuffer(UserDict):
    def __getitem__(self, k):
        raise AssertionError()


@gen_cluster(client=True, nthreads=[("", 1)], worker_kwargs={"data": WriteOnlyBuffer})
async def test_delete_spilled_keys(c, s, a):
    """Test that freeing an in-memory key that has been spilled to disk does not
    accidentally unspill it
    """
    x = c.submit(inc, 1, key="x")
    await wait_for_state("x", "memory", a)
    assert a.data.keys() == {"x"}
    with pytest.raises(AssertionError):
        a.data["x"]

    x.release()
    await async_poll_for(lambda: not a.data, timeout=2)
    assert not a.state.tasks

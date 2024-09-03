from __future__ import annotations

import asyncio
import os
import random
from contextlib import suppress
from time import sleep
from unittest import mock

import pytest
from tlz import first, merge, partition_all

import dask.config
from dask import delayed
from dask.utils import parse_bytes

from distributed import Client, KilledWorker, Nanny, get_worker, profile, wait
from distributed.comm import CommClosedError
from distributed.compatibility import MACOS
from distributed.metrics import time
from distributed.utils import CancelledError, sync
from distributed.utils_test import (
    NO_AMM,
    BlockedGatherDep,
    BlockedGetData,
    async_poll_for,
    captured_logger,
    cluster,
    div,
    gen_cluster,
    inc,
    slowadd,
    slowinc,
)
from distributed.worker_state_machine import FreeKeysEvent

pytestmark = pytest.mark.ci1


@pytest.mark.slow()
def test_submit_after_failed_worker_sync(loop):
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop) as c:
            L = c.map(inc, range(10))
            wait(L)
            a["proc"]().terminate()
            total = c.submit(sum, L)
            assert total.result() == sum(map(inc, range(10)))


@pytest.mark.parametrize("when", ["closing", "closed"])
@pytest.mark.parametrize("y_on_failed", [False, True])
@pytest.mark.parametrize("x_on_failed", [False, True])
@gen_cluster(
    client=True,
    nthreads=[("", 1)] * 2,
    config={"distributed.comm.timeouts.connect": "1s"},
)
async def test_submit_after_failed_worker_async(
    c, s, a, b, x_on_failed, y_on_failed, when, monkeypatch
):
    a_ws = s.workers[a.address]

    x = c.submit(
        inc,
        1,
        key="x",
        workers=[b.address if x_on_failed else a.address],
        allow_other_workers=True,
    )
    await wait(x)

    if when == "closed":
        await b.close()
        await async_poll_for(lambda: b.address not in s.workers, timeout=5)
    elif when == "closing":
        orig_remove_worker = s.remove_worker
        in_remove_worker = asyncio.Event()
        wait_remove_worker = asyncio.Event()

        async def remove_worker(*args, **kwargs):
            in_remove_worker.set()
            await wait_remove_worker.wait()
            return await orig_remove_worker(*args, **kwargs)

        monkeypatch.setattr(s, "remove_worker", remove_worker)
        await b.close()
        await in_remove_worker.wait()
        assert s.workers[b.address].status.name == "closing"

    y = c.submit(
        inc,
        x,
        key="y",
        workers=[b.address if y_on_failed else a.address],
        allow_other_workers=True,
    )
    await async_poll_for(lambda: "y" in s.tasks, timeout=5)

    if when == "closing":
        wait_remove_worker.set()
    assert await y == 3
    assert s.tasks["y"].who_has == {a_ws}


@gen_cluster(client=True, timeout=60)
async def test_submit_after_failed_worker(c, s, a, b):
    L = c.map(inc, range(10))
    await wait(L)

    await a.close()
    total = c.submit(sum, L)
    assert await total == sum(range(1, 11))


@pytest.mark.slow
def test_gather_after_failed_worker(loop):
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop) as c:
            L = c.map(inc, range(10))
            wait(L)
            a["proc"]().terminate()
            result = c.gather(L)
            assert result == list(map(inc, range(10)))


@pytest.mark.slow
@gen_cluster(client=True, Worker=Nanny, nthreads=[("127.0.0.1", 1)] * 4, timeout=60)
async def test_gather_then_submit_after_failed_workers(c, s, w, x, y, z):
    L = c.map(inc, range(20))
    await wait(L)

    w.process.process._process.terminate()
    total = c.submit(sum, L)

    for _ in range(3):
        await wait(total)
        addr = first(s.tasks[total.key].who_has).address
        for worker in [x, y, z]:
            if worker.worker_address == addr:
                worker.process.process._process.terminate()
                break

        result = await c.gather([total])
        assert result == [sum(map(inc, range(20)))]


@gen_cluster(Worker=Nanny, client=True, timeout=60)
async def test_restart(c, s, a, b):
    x = c.submit(inc, 1)
    y = c.submit(inc, x)
    z = c.submit(div, 1, 0)
    await y

    assert s.tasks[x.key].state == "memory"
    assert s.tasks[y.key].state == "memory"
    assert s.tasks[z.key].state != "memory"

    await c.restart()

    assert len(s.workers) == 2
    assert not any(ws.occupancy for ws in s.workers.values())

    assert not s.tasks

    assert x.cancelled()
    assert y.cancelled()
    assert z.cancelled()

    assert not s.tasks
    assert not any(cs.wants_what for cs in s.clients.values())


@gen_cluster(Worker=Nanny, client=True, timeout=60)
async def test_restart_cleared(c, s, a, b):
    x = 2 * delayed(1) + 1
    f = c.compute(x)
    await wait([f])

    await c.restart()

    for coll in [s.tasks, s.unrunnable]:
        assert not coll


def test_restart_sync(loop):
    with cluster(nanny=True) as (s, [a, b]):
        with Client(s["address"], loop=loop) as c:
            x = c.submit(div, 1, 2)
            x.result()

            assert sync(loop, c.scheduler.who_has)
            c.restart()
            assert not sync(loop, c.scheduler.who_has)
            assert x.cancelled()
            assert len(c.nthreads()) == 2

            with pytest.raises(CancelledError):
                x.result()

            y = c.submit(div, 1, 3)
            assert y.result() == 1 / 3


def test_worker_doesnt_await_task_completion(loop):
    with cluster(nanny=True, nworkers=1) as (s, [w]):
        with Client(s["address"], loop=loop) as c:
            future = c.submit(sleep, 100)
            sleep(0.1)
            start = time()
            c.restart(timeout="5s", wait_for_workers=False)
            stop = time()
            assert stop - start < 10


@gen_cluster(Worker=Nanny, timeout=60)
async def test_multiple_clients_restart(s, a, b):
    async with (
        Client(s.address, asynchronous=True) as c1,
        Client(s.address, asynchronous=True) as c2,
    ):
        x = c1.submit(inc, 1)
        y = c2.submit(inc, 2)
        xx = await x
        yy = await y
        assert xx == 2
        assert yy == 3

        await c1.restart()

        assert x.cancelled()
        start = time()
        while not y.cancelled():
            await asyncio.sleep(0.01)
            assert time() < start + 5

        assert not c1.futures
        assert not c2.futures

        # Ensure both clients still work after restart.
        # Reusing a previous key has no effect.
        x2 = c1.submit(inc, 1, key=x.key)
        y2 = c2.submit(inc, 2, key=y.key)

        assert x2._generation != x._generation
        assert y2._generation != y._generation

        assert await x2 == 2
        assert await y2 == 3

        del x2, y2
        await async_poll_for(lambda: not s.tasks, timeout=5)


@gen_cluster(Worker=Nanny, timeout=60)
async def test_restart_scheduler(s, a, b):
    assert len(s.workers) == 2
    pids = (a.pid, b.pid)
    assert pids[0]
    assert pids[1]

    await s.restart(stimulus_id="test")

    assert len(s.workers) == 2
    pids2 = (a.pid, b.pid)
    assert pids2[0]
    assert pids2[1]
    assert pids != pids2


@gen_cluster(Worker=Nanny, client=True, timeout=60)
async def test_forgotten_futures_dont_clean_up_new_futures(c, s, a, b):
    x = c.submit(inc, 1)
    await c.restart()
    y = c.submit(inc, 1)
    del x

    # Ensure that the profiler has stopped and released all references to x so that it
    # can be garbage-collected
    with profile.lock:
        pass
    await asyncio.sleep(0.1)
    await y


@pytest.mark.slow
@pytest.mark.flaky(condition=MACOS, reruns=10, reruns_delay=5)
@gen_cluster(client=True, timeout=60, active_rpc_timeout=10)
async def test_broken_worker_during_computation(c, s, a, b):
    s.allowed_failures = 100
    async with Nanny(s.address, nthreads=2) as n:
        await c.wait_for_workers(3)

        N = 256
        expected_result = N * (N + 1) // 2
        i = 0
        L = c.map(inc, range(N), key=["inc-%d-%d" % (i, j) for j in range(N)])
        while len(L) > 1:
            i += 1
            L = c.map(
                slowadd,
                *zip(*partition_all(2, L)),
                key=["add-%d-%d" % (i, j) for j in range(len(L) // 2)],
            )

        old_worker_address = n.worker_address
        await asyncio.sleep(random.random() / 20)
        with suppress(CommClosedError):  # comm will be closed abrupty
            await c.run(os._exit, 1, workers=[n.worker_address])

        while (
            not n.process.is_alive()
            or n.worker_address == old_worker_address
            or n.worker_address is None
        ):
            await asyncio.sleep(0.01)
        await c.wait_for_workers(3)

        await asyncio.sleep(random.random() / 20)

        with suppress(
            CommClosedError, EnvironmentError
        ):  # perhaps new worker can't be contacted yet
            await c.run(os._exit, 1, workers=[n.worker_address])

        [result] = await c.gather(L)
        assert isinstance(result, int)
        assert result == expected_result


@gen_cluster(client=True, Worker=Nanny, timeout=60)
async def test_restart_during_computation(c, s, a, b):
    xs = [delayed(slowinc)(i, delay=0.01) for i in range(50)]
    ys = [delayed(slowinc)(i, delay=0.01) for i in xs]
    zs = [delayed(slowadd)(x, y, delay=0.01) for x, y in zip(xs, ys)]
    total = delayed(sum)(zs)
    result = c.compute(total)

    await asyncio.sleep(0.5)
    assert any(ws.processing for ws in s.workers.values())
    await c.restart()
    assert not any(ws.processing for ws in s.workers.values())

    assert not s.tasks


class SlowTransmitData:
    def __init__(self, data, delay=0.1):
        self.delay = delay
        self.data = data

    def __reduce__(self):
        sleep(self.delay)
        return SlowTransmitData, (self.data, self.delay)

    def __sizeof__(self) -> int:
        # Ensure this is offloaded to avoid blocking loop
        return parse_bytes(dask.config.get("distributed.comm.offload")) + 1


@pytest.mark.slow
@gen_cluster(client=True, config={"distributed.scheduler.work-stealing": False})
async def test_worker_who_has_clears_after_failed_connection(c, s, a, b):
    """This test is very sensitive to cluster state consistency. Timeouts often
    indicate subtle deadlocks. Be mindful when marking flaky/repeat/etc."""
    async with Nanny(s.address, nthreads=2, worker_class=BlockedGetData) as n:
        while len(s.workers) < 3:
            await asyncio.sleep(0.01)

        n_worker_address = n.worker_address
        futures = c.map(
            inc,
            range(20),
            key=["f%d" % i for i in range(20)],
            workers=[n_worker_address],
            allow_other_workers=True,
        )

        def sink(*args):
            pass

        await wait(futures)
        result_fut = c.submit(sink, futures, workers=a.address)

        await n.kill(timeout=1)
        while len(s.workers) > 2:
            await asyncio.sleep(0.01)

        await result_fut

        assert not a.state.has_what.get(n_worker_address)
        assert not any(
            n_worker_address in s for ts in a.state.tasks.values() for s in ts.who_has
        )


@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 1), ("127.0.0.1", 2), ("127.0.0.1", 3)],
    config=NO_AMM,
)
async def test_worker_same_host_replicas_missing(c, s, a, b, x):
    # See GH4784
    def mock_address_host(addr):
        # act as if A and X are on the same host
        nonlocal a, b, x
        if addr in [a.address, x.address]:
            return "A"
        else:
            return "B"

    with mock.patch("distributed.worker.get_address_host", mock_address_host):
        futures = c.map(
            slowinc,
            range(20),
            delay=0.1,
            key=["f%d" % i for i in range(20)],
            workers=[a.address],
            allow_other_workers=True,
        )
        await wait(futures)

        # replicate data to avoid the scheduler retriggering the computation
        # retriggering cleans up the state nicely but doesn't reflect real world
        # scenarios where there may be replicas on the cluster, e.g. they are
        # replicated as a dependency somewhere else
        await c.replicate(futures, n=2, workers=[a.address, b.address])

        def sink(*args):
            pass

        # Since A and X are mocked to be co-located, X will consistently pick A
        # to fetch data from. It will never succeed since we're removing data
        # artificially, without notifying the scheduler.
        # This can only succeed if B handles the missing data properly by
        # removing A from the known sources of keys
        a.handle_stimulus(
            FreeKeysEvent(keys=["f1"], stimulus_id="Am I evil?")
        )  # Yes, I am!
        result_fut = c.submit(sink, futures, workers=x.address)

        await result_fut


@pytest.mark.slow
@gen_cluster(client=True, timeout=60, Worker=Nanny, nthreads=[("127.0.0.1", 1)])
async def test_restart_timeout_on_long_running_task(c, s, a):
    with captured_logger("distributed.scheduler") as sio:
        future = c.submit(sleep, 3600)
        await asyncio.sleep(0.1)
        await c.restart()

    text = sio.getvalue()
    assert "timeout" not in text.lower()


@pytest.mark.slow
@gen_cluster(client=True, config={"distributed.scheduler.worker-ttl": "500ms"})
async def test_worker_time_to_live(c, s, a, b):
    # Note that this value is ignored because is less than 10x heartbeat_interval
    assert s.worker_ttl == 0.5
    assert set(s.workers) == {a.address, b.address}
    assert all(
        event["action"] != "worker-ttl-timed-out"
        for _, event in s.get_events("scheduler")
    )

    a.periodic_callbacks["heartbeat"].stop()

    start = time()
    while set(s.workers) == {a.address, b.address}:
        await asyncio.sleep(0.01)
    assert set(s.workers) == {b.address}
    events = [
        event
        for _, event in s.get_events("scheduler")
        if event["action"] == "worker-ttl-timed-out"
    ]
    assert len(events) == 1
    # This event includes the actual TTL that we applied, i.e, 10 * heartbeat.
    assert events[0] == {
        "action": "worker-ttl-timed-out",
        "workers": [a.address],
        "ttl": 5.0,
    }

    # Worker removal is triggered after 10 * heartbeat
    # This is 10 * 0.5s at the moment of writing.
    # Currently observing an extra 0.3~0.6s on top of the interval.
    # Adding some padding to prevent flakiness.
    assert time() - start < 7


@pytest.mark.slow
@pytest.mark.parametrize("block_on", [None, "event_loop", "threadpool"])
@gen_cluster(
    client=True,
    Worker=Nanny,
    nthreads=[("", 1)],
    scheduler_kwargs={"worker_ttl": "100ms", "allowed_failures": 0},
    clean_kwargs={"threads": False},
)
async def test_worker_ttl_restarts_worker(c, s, a, block_on, monkeypatch):
    """If the event loop of a worker becomes completely unresponsive, the scheduler will
    restart it through the nanny.
    """
    ev = asyncio.Event()

    def wait_for_restart(event):
        _, msg = event
        if msg.get("action") == "worker-ttl-restart":
            ev.set()

    c.subscribe_topic("all", wait_for_restart)

    def f():
        w = get_worker()
        w.periodic_callbacks["heartbeat"].stop()
        if block_on is None:
            return
        elif block_on == "event_loop":

            async def _():
                sleep(9999)  # Block event loop indefinitely

            w.loop.add_callback(_)
        elif block_on == "threadpool":
            sleep(9999)

    fut = c.submit(f, key="x")

    # TTL is set to at least 10 heartbeats
    import distributed.scheduler

    monkeypatch.setattr(distributed.scheduler, "heartbeat_interval", lambda n: 0.001)

    await ev.wait()

    if block_on:
        # The nanny killed the worker with SIGKILL.
        # The restart has increased the suspicious count.
        with pytest.raises(KilledWorker):
            await fut
        assert s.tasks["x"].suspicious == 1
    else:
        # The nanny sent to the WorkerProcess a {op: stop} through IPC, which in turn
        # successfully invoked Worker.close(nanny=False).
        # This behaviour makes sense as the worker-ttl timeout was most likely caused
        # by a failure in networking, rather than a hung process.
        await fut
        assert s.tasks["x"].suspicious == 0


@pytest.mark.slow
@gen_cluster(
    client=True,
    Worker=Nanny,
    nthreads=[("", 2)],
    scheduler_kwargs={"allowed_failures": 0},
)
async def test_restart_hung_worker(c, s, a):
    """Test restart_workers() to restart a worker whose event loop has become completely
    unresponsive.
    """
    ws = s.workers[a.worker_address]

    async def f():
        w = get_worker()
        w.periodic_callbacks["heartbeat"].stop()
        sleep(9999)  # Block event loop indefinitely

    fut = c.submit(f)
    # Wait for worker to hang
    with pytest.raises(asyncio.TimeoutError):
        while True:
            await wait(c.submit(inc, 1, pure=False), timeout=0.2)

    await c.restart_workers([a.worker_address])
    assert len(s.workers) == 1
    assert next(iter(s.workers.values())) is not ws


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_forget_data_not_supposed_to_have(c, s, a):
    """If a dependency fetch finishes on a worker after the scheduler already released
    everything, the worker might be stuck with a redundant replica which is never
    cleaned up.
    """
    async with BlockedGatherDep(s.address) as b:
        x = c.submit(inc, 1, key="x", workers=[a.address])
        y = c.submit(inc, x, key="y", workers=[b.address])

        await b.in_gather_dep.wait()
        assert b.state.tasks["x"].state == "flight"

        x.release()
        y.release()
        while s.tasks:
            await asyncio.sleep(0.01)

        b.block_gather_dep.set()
        while b.state.tasks:
            await asyncio.sleep(0.01)


@gen_cluster(
    client=True,
    nthreads=[("", 1)] * 2,
    config=merge(NO_AMM, {"distributed.comm.timeouts.connect": "1s"}),
)
async def test_failing_worker_with_additional_replicas_on_cluster(c, s, w0, w2):
    """
    If a worker detects a missing dependency, the scheduler is notified. If no
    other replica is available, the dependency is rescheduled. A reschedule
    typically causes a lot of state to be reset. However, if another replica is
    available, we'll need to ensure that the worker can detect outdated state
    and correct its state.
    """

    def dummy(*args, **kwargs):
        return

    async with BlockedGatherDep(s.address) as w1:
        f1 = c.submit(
            inc,
            1,
            key="f1",
            workers=[w0.worker_address],
        )

        # We'll schedule tasks on two workers, s.t. f1 is replicated. We will
        # suspend one of the workers and kill the origin worker of f1 such that a
        # comm failure causes the worker to handle a missing dependency. It will ask
        # the schedule such that it knows that a replica is available on f2 and
        # reschedules the fetch
        f2 = c.submit(dummy, f1, key="f2", workers=[w1.worker_address])
        f3 = c.submit(dummy, f1, key="f3", workers=[w2.worker_address])

        await w1.in_gather_dep.wait()

        await wait(f3)
        # Because of this line we need to disable AMM; otherwise it could choose to delete
        # the replicas of f1 on w1 and w2 and keep the one on w0.
        await w0.close()

        w1.block_gather_dep.set()
        await c.gather([f1, f2, f3])

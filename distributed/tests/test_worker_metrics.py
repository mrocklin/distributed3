from __future__ import annotations

import asyncio
import time
from collections.abc import Collection, Hashable
from concurrent.futures import ProcessPoolExecutor
from time import sleep

import pytest

import dask

import distributed
from distributed import Event, Reschedule, Scheduler, Worker, get_worker, wait
from distributed.compatibility import WINDOWS
from distributed.metrics import context_meter, meter
from distributed.utils_test import (
    BlockedGatherDep,
    BlockedGetData,
    async_poll_for,
    gen_cluster,
    inc,
    slowinc,
    wait_for_state,
)


def get_digests(
    w: Worker | Scheduler, allow: str | Collection[str] = ()
) -> dict[Hashable, float]:
    if isinstance(allow, str):
        allow = (allow,)
    d = w.digests_total if isinstance(w, Worker) else w.cumulative_worker_metrics
    digests = {
        k: v
        for k, v in d.items()
        if k
        not in {"latency", "tick-duration", "transfer-bandwidth", "transfer-duration"}
        and (any(a in k for a in allow) or not allow)
    }
    assert all(v >= 0 for v in digests.values()), digests
    return digests


def span_id(s: Scheduler) -> str | None:
    ext = s.extensions["spans"]
    defaults = ext.spans_search_by_name["default",]
    # This is an arbitrary constraint enforced by the tests below
    assert len(defaults) == 1
    return defaults[0].id


@gen_cluster(client=True, config={"distributed.worker.memory.target": 1e-9})
async def test_task_lifecycle(c, s, a, b):
    x = (await c.scatter({"x": "x" * 20_000}, workers=[a.address]))["x"]
    y = (await c.scatter({"y": "y" * 20_000}, workers=[b.address]))["y"]
    assert a.state.tasks["x"].state == "memory"
    assert b.state.tasks["y"].state == "memory"
    with meter() as m:
        z = c.submit("".join, [x, y], key=("z-123", 0), workers=[a.address])
        assert (await z) == "x" * 20_000 + "y" * 20_000
        # The call to Worker.get_data will terminate after the fetch of z returns
        await async_poll_for(
            lambda: ("get-data", "network", "seconds") in a.digests_total, timeout=5
        )

    del x, y, z
    await async_poll_for(lambda: not a.state.tasks, timeout=5)  # For hygene only

    # Note: use set instead of list to account for rare, but harmless, race conditions
    expect = {
        # a.gather_dep(worker=b.address, keys=["z"])
        ("gather-dep", "decompress", "seconds"),
        ("gather-dep", "deserialize", "seconds"),
        ("gather-dep", "network", "seconds"),
        # Spill output; added by _transition_to_memory
        ("gather-dep", "serialize", "seconds"),
        ("gather-dep", "compress", "seconds"),
        ("gather-dep", "disk-write", "seconds"),
        ("gather-dep", "disk-write", "count"),
        ("gather-dep", "disk-write", "bytes"),
        # Delta to end-to-end runtime as seen from the worker state machine
        ("gather-dep", "other", "seconds"),
        # a.execute()
        # -> Deserialize run_spec
        ("execute", span_id(s), "z", "deserialize", "seconds"),
        # -> Unspill inputs
        # (There's also another execute-deserialize-seconds entry)
        ("execute", span_id(s), "z", "disk-read", "seconds"),
        ("execute", span_id(s), "z", "disk-read", "count"),
        ("execute", span_id(s), "z", "disk-read", "bytes"),
        ("execute", span_id(s), "z", "decompress", "seconds"),
        # -> Run in thread
        ("execute", span_id(s), "z", "thread-cpu", "seconds"),
        ("execute", span_id(s), "z", "thread-noncpu", "seconds"),
        ("execute", span_id(s), "z", "executor", "seconds"),
        # Spill output; added by _transition_to_memory
        ("execute", span_id(s), "z", "serialize", "seconds"),
        ("execute", span_id(s), "z", "compress", "seconds"),
        ("execute", span_id(s), "z", "disk-write", "seconds"),
        ("execute", span_id(s), "z", "disk-write", "count"),
        ("execute", span_id(s), "z", "disk-write", "bytes"),
        # Delta to end-to-end runtime as seen from the worker state machine
        ("execute", span_id(s), "z", "other", "seconds"),
        # a.get_data() (triggered by the client retrieving the Future for z)
        # Unspill
        ("get-data", "disk-read", "seconds"),
        ("get-data", "disk-read", "count"),
        ("get-data", "disk-read", "bytes"),
        ("get-data", "decompress", "seconds"),
        ("get-data", "deserialize", "seconds"),
        # Send over the network
        ("get-data", "serialize", "seconds"),
        ("get-data", "compress", "seconds"),
        ("get-data", "network", "seconds"),
    }
    assert set(get_digests(a)) == expect

    assert get_digests(a, allow="count") == {
        ("execute", span_id(s), "z", "disk-read", "count"): 2,
        ("execute", span_id(s), "z", "disk-write", "count"): 1,
        ("gather-dep", "disk-write", "count"): 1,
        ("get-data", "disk-read", "count"): 1,
    }
    if not WINDOWS:  # Fiddly rounding; see distributed.metrics._WindowsTime
        assert sum(get_digests(a, allow="seconds").values()) <= m.delta


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_async_task(c, s, a):
    """Test that async tasks are metered"""
    await c.submit(asyncio.sleep, 0.1, key=("x-123", 0))
    assert a.digests_total["execute", span_id(s), "x" "thread-cpu", "seconds"] == 0
    assert (
        0 < a.digests_total["execute", span_id(s), "x", "thread-noncpu", "seconds"] < 1
    )


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_custom_executor(c, s, a):
    """Don't try to acquire in-thread metrics when the executor is a ProcessPoolExecutor
    or a custom, arbitrary executor.
    """
    with ProcessPoolExecutor(1) as e:
        # Warm up executor - this can take up to 2s in Windows and MacOSX
        e.submit(inc, 1).result()

        a.executors["processes"] = e
        with dask.annotate(executor="processes"):
            await c.submit(sleep, 0.1)

    assert list(get_digests(a, "execute")) == [
        ("execute", span_id(s), "sleep", "executor", "seconds"),
        ("execute", span_id(s), "sleep", "other", "seconds"),
    ]

    assert (
        0 < a.digests_total["execute", span_id(s), "sleep", "executor", "seconds"] < 1
    )


@gen_cluster(client=True)
async def test_offload(c, s, a, b, monkeypatch):
    """Test that functions wrapped by offload() are metered"""
    monkeypatch.setattr(distributed.comm.utils, "OFFLOAD_THRESHOLD", 1)

    x = c.submit(inc, 1, key="x", workers=[a.address])
    y = c.submit(lambda x: None, x, key="y", workers=[b.address])
    await y

    assert list(get_digests(b, {"offload", "serialize", "deserialize"})) == [
        ("gather-dep", "offload", "seconds"),
        ("gather-dep", "deserialize", "seconds"),
        ("get-data", "offload", "seconds"),
        ("get-data", "serialize", "seconds"),
    ]


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_execute_failed(c, s, a):
    """Tasks that failed to execute are metered as a separate lump total"""
    x = c.submit(lambda: 1 / 0, key="x")
    await wait(x)

    assert list(get_digests(a)) == [("execute", span_id(s), "x", "failed", "seconds")]


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_cancelled_execute(c, s, a):
    """cancelled(execute) tasks are metered as a separate lump total"""
    ev = await Event()
    x = c.submit(lambda ev: ev.wait(), ev, key="x")
    await wait_for_state("x", "executing", a)
    del x
    await wait_for_state("x", "cancelled", a)
    await ev.set()
    await async_poll_for(lambda: not a.state.tasks, timeout=5)

    assert list(get_digests(a)) == [
        ("execute", span_id(s), "x", "cancelled", "seconds")
    ]


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_cancelled_flight(c, s, a):
    """cancelled(flight) tasks are metered as a separate lump total"""
    async with BlockedGetData(s.address) as b:
        x = c.submit(inc, 1, key="x", workers=[b.address])
        y = c.submit(inc, x, key="y", workers=[a.address])
        await b.in_get_data.wait()

        del y
        await wait_for_state("x", "cancelled", a)
        b.block_get_data.set()

    assert list(get_digests(a)) == [("gather-dep", "cancelled", "seconds")]


@gen_cluster(client=True)
async def test_gather_dep_busy(c, s, a, b):
    """gather_dep() calls that failed because the remote peer is busy
    are metered as a separate lump total
    """
    # We will block A for any outgoing communication. This simulates an
    # overloaded worker which will always return "busy" for get_data requests.
    a.transfer_outgoing_count = 10000000

    x = c.submit(inc, 1, key="x", workers=[a.address])
    y = c.submit(inc, x, key="y", workers=[b.address])

    await wait_for_state(y.key, "waiting", b)
    assert b.state.tasks[x.key].state in ("flight", "fetch")
    with pytest.raises(asyncio.TimeoutError):
        await y.result(timeout=0.5)

    assert list(get_digests(b)) == [("gather-dep", "busy", "seconds")]


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    config={"distributed.scheduler.active-memory-manager.start": False},
)
async def test_gather_dep_no_task(c, s, w1):
    """gather_dep() calls where the remote peer answers that it doesn't have any of the
    requested keys are metered as a separate lump total
    """
    x = c.submit(inc, 1, key="x", workers=[w1.address])

    async with BlockedGetData(s.address) as w2, BlockedGatherDep(s.address) as w3:
        y = c.submit(inc, x, key="y", workers=[w3.address])
        await w3.in_gather_dep.wait()  # Gather from w1

        # Move x from w1 to w2
        s.request_acquire_replicas(w2.address, ["x"], stimulus_id="ar")
        await async_poll_for(lambda: len(s.tasks["x"].who_has) == 2, timeout=5)
        s.request_remove_replicas(w1.address, ["x"], stimulus_id="rr")
        await async_poll_for(lambda: len(s.tasks["x"].who_has) == 1, timeout=5)

        w3.block_gather_dep.set()
        # 1. w1 will now answer that it does not have the key
        # 2. x will transition to missing on w3
        # 3. w3 will ask the scheduler if there are other replicas
        # 4. the scheduler will answer that now w2 holds a replica
        # 5. w3 will try fetching the key from w2. We block again so that we don't
        #    pollute the metrics with a successful attempt.
        await w2.in_get_data.wait()

        assert list(get_digests(w3)) == [("gather-dep", "missing", "seconds")]

        w2.block_get_data.set()
        assert await y == 3


@gen_cluster(client=True)
async def test_gather_dep_failed(c, s, a, b):
    """gather_dep() calls where the task fails to deserialize are metered as a
    separate lump total
    """

    class C:
        def __reduce__(self):
            def expand():
                raise Exception()

            return expand, ()

    x = c.submit(C, key="x", workers=[a.address])
    y = c.submit(lambda x: None, x, key="y", workers=[b.address])
    await wait_for_state("x", "error", b)
    assert list(get_digests(b)) == [("gather-dep", "failed", "seconds")]

    # FIXME https://github.com/dask/distributed/issues/6705
    b.state.validate = False


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    config={"distributed.comm.timeouts.connect": "500ms"},
)
async def test_gather_dep_network_error(c, s, a):
    """gather_dep() calls where the remote peer fails to respond are metered as a
    separate lump total
    """
    x = c.submit(inc, 1, key="x")
    await wait(x)
    async with BlockedGatherDep(s.address) as b:
        y = c.submit(inc, x, key="y", workers=[b.address])
        await b.in_gather_dep.wait()
        await a.close()
        b.block_gather_dep.set()
        await wait(y)
        assert list(get_digests(b, "gather-dep")) == [
            ("gather-dep", "failed", "seconds")
        ]


@gen_cluster(
    nthreads=[("", 1)],
    client=True,
    worker_kwargs={"memory_limit": "10 GiB"},
    config={
        "distributed.worker.memory.target": False,
        "distributed.worker.memory.spill": 0.7,
        "distributed.worker.memory.pause": False,
        "distributed.worker.memory.monitor-interval": "10ms",
    },
)
async def test_memory_monitor(c, s, a):
    a.monitor.get_process_memory = lambda: 800_000_000_000 if a.data.fast else 0
    x = c.submit(inc, 1, key="x")
    await async_poll_for(lambda: a.data.disk, timeout=5)

    assert list(get_digests(a, "memory-monitor")) == [
        ("memory-monitor", "serialize", "seconds"),
        ("memory-monitor", "compress", "seconds"),
        ("memory-monitor", "disk-write", "seconds"),
        ("memory-monitor", "disk-write", "count"),
        ("memory-monitor", "disk-write", "bytes"),
    ]


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_user_metrics_sync(c, s, a):
    def f():
        t1 = time.perf_counter() + 0.1
        while time.perf_counter() < t1:
            pass
        sleep(0.1)
        context_meter.digest_metric("I/O", 5, "seconds")

    await wait(c.submit(f, key="x"))

    assert list(get_digests(a)) == [
        ("execute", span_id(s), "x", "I/O", "seconds"),
        ("execute", span_id(s), "x", "thread-cpu", "seconds"),
        ("execute", span_id(s), "x", "thread-noncpu", "seconds"),
        ("execute", span_id(s), "x", "executor", "seconds"),
        ("execute", span_id(s), "x", "other", "seconds"),
    ]
    assert get_digests(a)["execute", span_id(s), "x", "I/O", "seconds"] == 5
    assert get_digests(a)["execute", span_id(s), "x", "thread-cpu", "seconds"] == 0
    assert get_digests(a)["execute", span_id(s), "x", "thread-noncpu", "seconds"] == 0
    assert get_digests(a)["execute", span_id(s), "x", "executor", "seconds"] == 0
    assert get_digests(a)["execute", span_id(s), "x", "other", "seconds"] == 0


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_user_metrics_async(c, s, a):
    async def f():
        await asyncio.sleep(0.1)
        context_meter.digest_metric("I/O", 5, "seconds")

    await wait(c.submit(f, key="x"))

    assert list(get_digests(a)) == [
        ("execute", span_id(s), "x", "I/O", "seconds"),
        ("execute", span_id(s), "x", "thread-noncpu", "seconds"),
        ("execute", span_id(s), "x", "other", "seconds"),
    ]
    assert get_digests(a)["execute", span_id(s), "x", "I/O", "seconds"] == 5
    assert get_digests(a)["execute", span_id(s), "x", "thread-noncpu", "seconds"] == 0
    assert get_digests(a)["execute", span_id(s), "x", "other", "seconds"] == 0


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_user_metrics_fail(c, s, a):
    def f():
        context_meter.digest_metric("I/O", 5, "seconds")
        context_meter.digest_metric("I/O", 100, "bytes")
        raise ValueError("foo")

    await wait(c.submit(f, key="x"))

    assert list(get_digests(a)) == [
        ("execute", span_id(s), "x", "I/O", "bytes"),
        ("execute", span_id(s), "x", "failed", "seconds"),
    ]
    assert get_digests(a)["execute", span_id(s), "x", "I/O", "bytes"] == 100
    assert get_digests(a)["execute", span_id(s), "x", "failed", "seconds"] < 1


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_user_metrics_weird(c, s, a):
    """label can be any msgpack-serializable Hashable
    unit can be any string
    """

    def f():
        context_meter.digest_metric(("foo", 1), 1, "seconds")
        context_meter.digest_metric(None, 2, "custom")

    await wait(c.submit(f, key="x"))
    await a.heartbeat()

    s_metrics = get_digests(s)
    a_metrics = get_digests(a)

    assert list(s_metrics) == [
        ("execute", "x", ("foo", 1), "seconds"),
        ("execute", "x", None, "custom"),
        ("execute", "x", "thread-cpu", "seconds"),
        ("execute", "x", "thread-noncpu", "seconds"),
        ("execute", "x", "executor", "seconds"),
        ("execute", "x", "other", "seconds"),
    ]
    for (context, prefix, activity, unit), v in s_metrics.items():
        assert a_metrics[context, span_id(s), prefix, activity, unit] == pytest.approx(
            v
        )


@gen_cluster(client=True, nthreads=[("", 3)])
async def test_do_not_leak_metrics(c, s, a, b):
    def f(k):
        for _ in range(10):
            sleep(0.05)
            context_meter.digest_metric(f"ping-{k}", 1, "count")

    async def g(k):
        for _ in range(10):
            await asyncio.sleep(0.05)
            context_meter.digest_metric(f"ping-{k}", 1, "count")

    x = c.submit(f, "x", key="x")
    y = c.submit(f, "y", key="y")
    z = c.submit(g, "z", key="z")
    await c.run(g, "w")  # Metrics are discarded
    await wait([x, y, z])

    assert get_digests(a, "count") == {
        ("execute", span_id(s), "x", "ping-x", "count"): 10,
        ("execute", span_id(s), "y", "ping-y", "count"): 10,
        ("execute", span_id(s), "z", "ping-z", "count"): 10,
    }


@gen_cluster(
    client=True,
    nthreads=[("", 1)] * 2,
    config={"distributed.scheduler.work-stealing": False},
)
async def test_reschedule(c, s, a, b):
    """A task raises Reschedule()

    See also
    --------
    test_reschedule.py
    """
    a_address = a.address

    def f(x):
        sleep(0.1)
        if get_worker().address == a_address:
            raise Reschedule()

    futures = c.map(f, range(4), key=["x-1", "x-2", "x-3", "x-4"])
    futures2 = c.map(slowinc, range(10), delay=0.1, key="clog", workers=[a.address])
    await wait(futures)
    assert all(f.key in b.data for f in futures)

    evs = get_digests(a, "x")
    k = ("execute", span_id(s), "x", "cancelled", "seconds")
    assert list(evs) == [k]
    assert evs[k] > 0


@gen_cluster(
    client=True, scheduler_kwargs={"extensions": {}}, worker_kwargs={"extensions": {}}
)
async def test_send_metrics_to_scheduler(c, s, a, b):
    """Test that Worker.digests_total are sync'ed by the heartbeat to
    Scheduler.cumulative_worker_metrics

    See also
    --------
    test_spans.py::test_worker_metrics
    """
    # Race condition: metrics that are updated while idle may or may not be there already
    assert s.cumulative_worker_metrics.keys() in (set(), {"latency"})

    x0 = c.submit(inc, 1, key=("x", 0), workers=[a.address])
    x1 = c.submit(inc, 1, key=("x", 1), workers=[b.address])
    # Trigger gather_dep and get_data
    x2 = c.submit(inc, x0, key=("x", 2), workers=[b.address])
    x3 = c.submit(inc, x1, key=("x", 3), workers=[a.address])
    await wait([x2, x3])
    # Flush metrics from workers to scheduler
    await a.heartbeat()
    await b.heartbeat()

    # Make sure that cumulative sync over multiple heartbeats works as expected
    x4 = c.submit(inc, 1, key=("x", 4))
    await wait(x4)
    await a.heartbeat()
    await b.heartbeat()

    # Metrics from workers have been summed up on scheduler
    a_metrics = get_digests(a)
    b_metrics = get_digests(b)
    s_metrics = get_digests(s)

    expect_worker = [
        ("execute", None, "x", "thread-cpu", "seconds"),
        ("execute", None, "x", "thread-noncpu", "seconds"),
        ("execute", None, "x", "executor", "seconds"),
        ("execute", None, "x", "other", "seconds"),
        ("get-data", "memory-read", "count"),
        ("get-data", "memory-read", "bytes"),
        ("get-data", "serialize", "seconds"),
        ("get-data", "compress", "seconds"),
        ("gather-dep", "decompress", "seconds"),
        ("gather-dep", "deserialize", "seconds"),
        ("gather-dep", "network", "seconds"),
        ("gather-dep", "other", "seconds"),
        ("get-data", "network", "seconds"),
        ("execute", None, "x", "memory-read", "count"),
        ("execute", None, "x", "memory-read", "bytes"),
    ]
    expect_scheduler = [
        k[:1] + k[2:] if k[0] == "execute" else k for k in expect_worker
    ]

    # Note: use set instead of list to account for rare, but harmless, race conditions
    assert set(a_metrics) == set(b_metrics) == set(expect_worker)
    assert set(s_metrics) == set(expect_scheduler)

    for wk, sk in zip(expect_worker, expect_scheduler):
        assert a_metrics[wk] >= 0
        assert b_metrics[wk] >= 0
        assert s_metrics[sk] == pytest.approx(a_metrics[wk] + b_metrics[wk])


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    scheduler_kwargs={"extensions": {}},
    worker_kwargs={"extensions": {}},
)
async def test_no_spans_extension(c, s, a):
    await wait(c.submit(lambda: 1 / 0, key="x"))
    await wait(c.submit(inc, 1, key="y"))
    await a.heartbeat()

    w_metrics = get_digests(a)
    s_metrics = get_digests(s)
    expect_worker = [
        ("execute", None, "x", "failed", "seconds"),
        ("execute", None, "y", "thread-cpu", "seconds"),
        ("execute", None, "y", "thread-noncpu", "seconds"),
        ("execute", None, "y", "executor", "seconds"),
        ("execute", None, "y", "other", "seconds"),
    ]
    expect_scheduler = [
        k[:1] + k[2:] if k[0] == "execute" else k for k in expect_worker
    ]
    assert list(w_metrics) == expect_worker
    assert list(s_metrics) == expect_scheduler

    for wk, sk in zip(expect_worker, expect_scheduler):
        assert w_metrics[wk] >= 0
        assert s_metrics[sk] == pytest.approx(w_metrics[wk])


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_new_metrics_during_heartbeat(c, s, a):
    """Make sure that metrics generated during the heartbeat don't get lost"""
    # Create default span
    await c.submit(inc, 1)
    span = s.extensions["spans"].spans_search_by_name["default",][0]

    hb_task = asyncio.create_task(a.heartbeat())
    n = 0
    while not hb_task.done():
        n += 1
        a.digest_metric(("execute", span.id, "x", "test", "test"), 1)
        await asyncio.sleep(0)
    await hb_task
    assert n > 1
    await a.heartbeat()

    assert a.digests_total["execute", span.id, "x", "test", "test"] == n
    assert s.cumulative_worker_metrics["execute", "x", "test", "test"] == n
    assert span.cumulative_worker_metrics["execute", "x", "test", "test"] == n


@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    config={"distributed.scheduler.worker-saturation": float("inf")},
)
async def test_delayed_ledger_is_not_reentrant(c, s, a):
    """https://github.com/dask/distributed/issues/7949

    Test that, when there's a long chain of task done -> task start events,
    the callbacks added by the delayed ledger don't pile up on top of each other.
    """

    def f(_):
        return len(context_meter._callbacks.get())

    out = await c.gather(c.map(f, range(1000)))
    assert max(out) < 10


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_int_metrics(c, s, a):
    """Test that int metrics are not cast to float"""

    def f():
        context_meter.digest_metric("foo", 1, "u")

    await c.submit(f, key="x")
    await a.heartbeat()

    span = s.extensions["spans"].spans_search_by_name["default",][0]

    def assert_int(d, k):
        assert d[k] == 1
        assert isinstance(d[k], int)

    assert_int(a.digests_total, ("execute", span.id, "x", "foo", "u"))
    assert_int(a.digests_max, ("execute", span.id, "x", "foo", "u"))
    assert_int(s.cumulative_worker_metrics, ("execute", "x", "foo", "u"))
    assert_int(span.cumulative_worker_metrics, ("execute", "x", "foo", "u"))

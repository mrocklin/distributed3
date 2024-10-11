from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import math
import random
import weakref
from collections import defaultdict
from collections.abc import Callable, Coroutine, Iterable, Mapping, Sequence
from operator import mul
from time import sleep

import pytest
from tlz import merge, sliding_window

import dask
from dask.utils import key_split, parse_bytes

from distributed import (
    Client,
    Event,
    Lock,
    Nanny,
    Scheduler,
    Worker,
    profile,
    wait,
    worker_client,
)
from distributed.client import Future
from distributed.compatibility import LINUX
from distributed.core import Status
from distributed.metrics import time
from distributed.system import MEMORY_LIMIT
from distributed.utils import wait_for
from distributed.utils_test import (
    NO_AMM,
    BlockedGatherDep,
    BlockedGetData,
    captured_logger,
    freeze_batched_send,
    gen_cluster,
    gen_nbytes,
    inc,
    nodebug_setup_module,
    nodebug_teardown_module,
    slowadd,
    slowidentity,
    slowinc,
    wait_for_state,
)
from distributed.worker_state_machine import (
    ExecuteSuccessEvent,
    FreeKeysEvent,
    StealRequestEvent,
)

pytestmark = pytest.mark.ci1

# Most tests here are timing-dependent
setup_module = nodebug_setup_module
teardown_module = nodebug_teardown_module


@gen_cluster(client=True, nthreads=[("", 2), ("", 2)])
async def test_work_stealing(c, s, a, b):
    [x] = await c._scatter([1], workers=a.address)
    futures = c.map(slowadd, range(50), [x] * 50)
    await wait(futures)
    assert len(a.data) > 10
    assert len(b.data) > 10


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 2)
async def test_dont_steal_expensive_data_fast_computation(c, s, a, b):
    np = pytest.importorskip("numpy")

    x = c.submit(np.arange, 1000000, workers=a.address)
    await wait([x])
    future = c.submit(np.sum, [1], workers=a.address)  # learn that sum is fast
    await wait([future])

    cheap = [
        c.submit(np.sum, x, pure=False, workers=a.address, allow_other_workers=True)
        for i in range(10)
    ]
    await wait(cheap)
    assert len(s.tasks[x.key].who_has) == 1
    assert len(b.data) == 0
    assert len(a.data) == 12


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 2)
async def test_steal_cheap_data_slow_computation(c, s, a, b):
    x = c.submit(slowinc, 100, delay=0.1)  # learn that slowinc is slow
    await wait(x)

    futures = c.map(
        slowinc, range(10), delay=0.1, workers=a.address, allow_other_workers=True
    )
    await wait(futures)
    assert abs(len(a.data) - len(b.data)) <= 5


@pytest.mark.slow
@gen_cluster(client=True, nthreads=[("", 1)] * 2, config=NO_AMM)
async def test_steal_expensive_data_slow_computation(c, s, a, b):
    np = pytest.importorskip("numpy")

    x = c.submit(slowinc, 1, delay=0.2, workers=a.address)
    await wait(x)  # learn that slowinc is slow

    x = c.submit(np.arange, 1_000_000, workers=a.address)  # put expensive data
    await wait(x)

    slow = [c.submit(slowinc, x, delay=0.1, pure=False) for _ in range(20)]
    await wait(slow)
    assert len(s.tasks[x.key].who_has) > 1

    assert b.data  # not empty


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 10, config=NO_AMM)
async def test_worksteal_many_thieves(c, s, *workers):
    x = c.submit(slowinc, -1, delay=0.1)
    await x

    xs = c.map(slowinc, [x] * 100, pure=False, delay=0.1)

    await wait(xs)

    for ws in s.workers.values():
        assert 2 < len(ws.has_what) < 30

    assert len(s.tasks[x.key].who_has) > 1
    assert sum(len(ws.has_what) for ws in s.workers.values()) < 150


@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 1)] * 2,
    config={"distributed.scheduler.work-stealing-interval": "10ms"},
)
async def test_stop_plugin(c, s, a, b):
    steal = s.extensions["stealing"]

    await steal.stop()
    futs = c.map(slowinc, range(10), workers=[a.address], allow_other_workers=True)
    await c.gather(futs)
    assert len(a.data) == 10

    # nothing happens
    for _ in range(10):
        await steal.stop()


@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 1)] * 2,
    config={"distributed.scheduler.work-stealing-interval": "1ms"},
)
async def test_stop_in_flight(c, s, a, b):
    steal = s.extensions["stealing"]
    num_tasks = 10
    futs = c.map(
        slowinc, range(num_tasks), workers=[a.address], allow_other_workers=True
    )
    while not steal.in_flight:
        await asyncio.sleep(0)

    assert steal.in_flight
    await steal.stop()
    assert not steal.in_flight

    assert len(a.data) != num_tasks
    del futs
    while s.tasks or a.state.tasks or b.state.tasks:
        await asyncio.sleep(0.1)

    futs = c.map(
        slowinc, range(num_tasks), workers=[a.address], allow_other_workers=True
    )
    await c.gather(futs)
    assert len(a.data) == num_tasks

    del futs
    while s.tasks or a.state.tasks or b.state.tasks:
        await asyncio.sleep(0.1)
    event = Event()

    def block(x, event):
        event.wait()
        return x + 1

    futs = c.map(
        block,
        range(num_tasks),
        event=event,
        workers=[a.address],
        allow_other_workers=True,
    )
    while not len(a.state.tasks) == num_tasks:
        await asyncio.sleep(0.01)
    assert len(b.state.tasks) == 0
    await steal.start()
    await event.set()
    await c.gather(futs)
    assert len(a.state.tasks) != num_tasks
    assert len(b.state.tasks) != 0


@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 1)] * 2,
    config={"distributed.scheduler.work-stealing-interval": "10ms"},
)
async def test_allow_tasks_stolen_before_first_completes(c, s, a, b):
    # https://github.com/dask/distributed/issues/5564
    from distributed import Semaphore

    steal = s.extensions["stealing"]
    await steal.stop()
    lock = await Semaphore(max_leases=1)

    # We will reuse the same function such that multiple dispatches have the
    # same task prefix. This ensures that we have tasks queued up but all of
    # them are still classified as unknown.
    # The lock allows us to control the duration of the first task without
    # delaying test runtime or flakyness
    def blocked_task(x, lock):
        if x == 0:
            with lock:
                return x
        return x

    async with lock:
        first = c.submit(blocked_task, 0, lock, workers=[a.address], key="f-0")
        while first.key not in a.state.tasks:
            await asyncio.sleep(0.001)
        # Ensure the task is indeed blocked
        with pytest.raises(asyncio.TimeoutError):
            await wait_for(first, 0.01)

        more_tasks = c.map(
            blocked_task,
            # zero is a sentinel for using the lock.
            # Start counting at one for non-blocking funcs
            range(1, 11),
            lock=lock,
            workers=[a.address],
            key=[f"f-{ix}" for ix in range(1, 11)],
            allow_other_workers=True,
        )
        # All tasks are put on A since this is what we asked for. Only work
        # stealing should rebalance the tasks once we allow for it
        while not len(a.state.tasks) == 11:
            await asyncio.sleep(0.1)

        assert len(b.state.tasks) == 0

        await steal.start()
        # A is still blocked by executing task f-1 so this can only pass if
        # workstealing moves the tasks to B
        await c.gather(more_tasks)
        assert len(b.data) == 10
    await first


@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 1)] * 2,
    config={"distributed.scheduler.work-stealing-interval": "10ms"},
)
async def test_eventually_steal_unknown_functions(c, s, a, b):
    futures = c.map(
        slowinc, range(10), delay=0.1, workers=a.address, allow_other_workers=True
    )
    await wait(futures)
    assert not s.unknown_durations
    assert len(a.data) >= 3, [len(a.data), len(b.data)]
    assert len(b.data) >= 3, [len(a.data), len(b.data)]


@pytest.mark.skip(reason="")
@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 3)
async def test_steal_related_tasks(e, s, a, b, c):
    futures = e.map(
        slowinc, range(20), delay=0.05, workers=a.address, allow_other_workers=True
    )

    await wait(futures)

    nearby = 0
    for f1, f2 in sliding_window(2, futures):
        if s.tasks[f1.key].who_has == s.tasks[f2.key].who_has:
            nearby += 1

    assert nearby > 10


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 10)
async def test_dont_steal_fast_tasks_compute_time(c, s, *workers):
    def do_nothing(x, y=None):
        pass

    xs = c.map(do_nothing, range(10), workers=workers[0].address)
    await wait(xs)

    futures = c.map(do_nothing, range(100), y=xs)

    await wait(futures)

    assert len(set.union(*(s.tasks[x.key].who_has for x in xs))) == 1
    assert len(s.workers[workers[0].address].has_what) == len(xs) + len(futures)


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_dont_steal_fast_tasks_blocklist(c, s, a):
    async with BlockedGetData(s.address) as b:
        # create a dependency
        x = c.submit(inc, 1, workers=[b.address], key="x")
        await wait(x)

        # If the blocklist of fast tasks is tracked somewhere else, this needs to be
        # changed. This test requires *any* key which is blocked.
        from distributed.stealing import fast_tasks

        blocked_key = next(iter(fast_tasks))

        def fast_blocked(i, x):
            # The task should observe a certain computation time such that we can
            # ensure that it is not stolen due to the blocking. If it is too
            # fast, the standard mechanism shouldn't allow stealing
            sleep(0.01)

        futures = c.map(
            fast_blocked,
            range(50),
            x=x,
            # Submit the task to one worker but allow it to be distributed elsewhere,
            # i.e. this is not a task restriction
            workers=[a.address],
            allow_other_workers=True,
            key=blocked_key,
        )

        while len(s.tasks) < 51:
            await asyncio.sleep(0.01)
        b.block_get_data.set()
        await wait(futures)

        # Note: x may now be on a, b, or both, depending if the Active Memory Manager
        # got to run or not
        ws_a = s.workers[a.address]
        for ts in s.tasks.values():
            if ts.key.startswith(blocked_key):
                assert ts.who_has == {ws_a}


@gen_cluster(client=True, nthreads=[("", 1)], config=NO_AMM)
async def test_new_worker_steals(c, s, a):
    await wait(c.submit(slowinc, 1, delay=0.01))

    futures = c.map(slowinc, range(100), delay=0.05)
    total = c.submit(sum, futures)
    while len(a.state.tasks) < 10:
        await asyncio.sleep(0.01)

    async with Worker(s.address, nthreads=1) as b:
        result = await total
        assert result == sum(map(inc, range(100)))

        for w in (a, b):
            assert all(isinstance(v, int) for v in w.data.values())

        # This requires AMM to be off. Otherwise, if b reports higher optimistic memory
        # than a and `total` happens to be computed on a, then all keys on b will be
        # replicated onto a and then deleted by the AMM.
        assert b.data


@gen_cluster(client=True)
async def test_work_steal_allow_other_workers(c, s, a, b):
    # Note: this test also verifies the baseline for all other tests below
    futures = c.map(
        slowinc, range(100), workers=a.address, allow_other_workers=True, delay=0.05
    )
    await c.gather(futures)

    assert 20 < len(a.data) < 80
    assert 20 < len(b.data) < 80

    total = c.submit(sum, futures)
    result = await total

    assert result == sum(map(inc, range(100)))


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1), ("127.0.0.1", 2)])
async def test_dont_steal_worker_restrictions(c, s, a, b):
    futures = c.map(slowinc, range(100), delay=0.05, workers=a.address)

    while len(a.state.tasks) + len(b.state.tasks) < 100:
        await asyncio.sleep(0.01)

    assert len(a.state.tasks) == 100
    assert len(b.state.tasks) == 0

    s.extensions["stealing"].balance()
    await asyncio.sleep(0.1)

    assert len(a.state.tasks) == 100
    assert len(b.state.tasks) == 0


@gen_cluster(client=True, nthreads=[("", 1), ("", 2), ("", 2)])
async def test_steal_worker_restrictions(c, s, wa, wb, wc):
    futures = c.map(slowinc, range(100), delay=0.05, workers={wa.address, wb.address})
    while sum(len(w.state.tasks) for w in [wa, wb, wc]) < 100:
        await asyncio.sleep(0.01)

    assert 20 < len(wa.state.tasks) < 80
    assert 20 < len(wb.state.tasks) < 80
    assert len(wc.state.tasks) == 0

    s.extensions["stealing"].balance()
    await asyncio.sleep(0.1)

    assert 20 < len(wa.state.tasks) < 80
    assert 20 < len(wb.state.tasks) < 80
    assert len(wc.state.tasks) == 0


@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
@gen_cluster(client=True, nthreads=[("127.0.0.1", 1), ("127.0.0.2", 1)])
async def test_dont_steal_host_restrictions(c, s, a, b):
    futures = c.map(slowinc, range(100), delay=0.05, workers="127.0.0.1")
    while len(a.state.tasks) + len(b.state.tasks) < 100:
        await asyncio.sleep(0.01)

    assert len(a.state.tasks) == 100
    assert len(b.state.tasks) == 0

    s.extensions["stealing"].balance()
    await asyncio.sleep(0.1)

    assert len(a.state.tasks) == 100
    assert len(b.state.tasks) == 0


@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
@gen_cluster(client=True, nthreads=[("127.0.0.1", 1), ("127.0.0.2", 2)])
async def test_steal_host_restrictions(c, s, wa, wb):
    futures = c.map(slowinc, range(100), delay=0.05, workers="127.0.0.1")
    while len(wa.state.tasks) + len(wb.state.tasks) < 100:
        await asyncio.sleep(0.01)

    assert len(wa.state.tasks) == 100
    assert len(wb.state.tasks) == 0

    async with Worker(s.address, nthreads=1) as wc:
        while s.workers[wc.address].status != Status.running:
            await asyncio.sleep(0.01)

        s.extensions["stealing"].balance()
        await asyncio.sleep(0.1)

        assert 20 < len(wa.state.tasks) < 95
        assert len(wb.state.tasks) == 0
        assert 5 < len(wc.state.tasks) < 80


@gen_cluster(client=True, nthreads=[("", 1, {"resources": {"A": 2}}), ("", 1)])
async def test_dont_steal_resource_restrictions(c, s, a, b):
    futures = c.map(slowinc, range(100), delay=0.05, resources={"A": 1})
    while len(a.state.tasks) + len(b.state.tasks) < 100:
        await asyncio.sleep(0.01)

    assert len(a.state.tasks) == 100
    assert len(b.state.tasks) == 0

    s.extensions["stealing"].balance()
    await asyncio.sleep(0.1)

    assert len(a.state.tasks) == 100
    assert len(b.state.tasks) == 0


@gen_cluster(client=True, nthreads=[("", 1, {"resources": {"A": 2}})])
async def test_steal_resource_restrictions(c, s, a):
    futures = c.map(slowinc, range(100), delay=0.05, resources={"A": 1})
    while len(a.state.tasks) < 100:
        await asyncio.sleep(0.01)

    async with Worker(s.address, nthreads=1, resources={"A": 4}) as b:
        while s.workers[b.address].status != Status.running:
            await asyncio.sleep(0.01)

        s.extensions["stealing"].balance()
        await asyncio.sleep(0.1)

        assert 20 < len(b.state.tasks) < 80
        assert 20 < len(a.state.tasks) < 80


@gen_cluster(client=True, nthreads=[("", 1, {"resources": {"A": 2, "C": 1}})])
async def test_steal_resource_restrictions_asym_diff(c, s, a):
    # See https://github.com/dask/distributed/issues/5565
    future = c.submit(slowinc, 1, delay=0.1, workers=a.address)
    await future

    futures = c.map(slowinc, range(100), delay=0.2, resources={"A": 1})
    while len(a.state.tasks) < 101:
        await asyncio.sleep(0.01)
    assert len(a.state.tasks) == 101

    async with Worker(s.address, nthreads=1, resources={"A": 4, "B": 5}) as b:
        while not b.state.tasks or len(a.state.tasks) == 101:
            await asyncio.sleep(0.01)

        assert len(b.state.tasks) > 0
        assert len(a.state.tasks) < 101


@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 1)] * 5,
    config={"distributed.scheduler.work-stealing-interval": "20ms"},
)
async def test_balance_without_dependencies(c, s, *workers):
    def slow(x):
        y = random.random() * 0.1
        sleep(y)
        return y

    futures = c.map(slow, range(100))
    await wait(futures)

    durations = [sum(w.data.values()) for w in workers]
    assert max(durations) / min(durations) < 3


@gen_cluster(client=True, nthreads=[("127.0.0.1", 4)] * 2)
async def test_dont_steal_executing_tasks(c, s, a, b):
    futures = c.map(
        slowinc, range(4), delay=0.1, workers=a.address, allow_other_workers=True
    )

    await wait(futures)
    assert len(a.data) == 4
    assert len(b.data) == 0


@gen_cluster(client=True)
async def test_dont_steal_executing_tasks_2(c, s, a, b):
    steal = s.extensions["stealing"]

    future = c.submit(slowinc, 1, delay=0.5, workers=a.address)
    while not a.state.executing_count:
        await asyncio.sleep(0.01)

    steal.move_task_request(
        s.tasks[future.key], s.workers[a.address], s.workers[b.address]
    )
    await asyncio.sleep(0.1)
    assert a.state.tasks[future.key].state == "executing"
    assert not b.state.executing_count


@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 1)] * 10,
    config={
        "distributed.scheduler.default-task-durations": {"slowidentity": 0.2},
        "distributed.scheduler.work-stealing-interval": "20ms",
    },
)
async def test_dont_steal_few_saturated_tasks_many_workers(c, s, a, *rest):
    x = c.submit(mul, b"0", 100000000, workers=a.address)  # 100 MB
    await wait(x)

    futures = [c.submit(slowidentity, x, pure=False, delay=0.2) for i in range(2)]

    await wait(futures)

    assert len(a.data) == 3
    assert not any(w.state.tasks for w in rest)


@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 1)] * 10,
    worker_kwargs={"memory_limit": MEMORY_LIMIT},
    config={
        "distributed.scheduler.default-task-durations": {"slowidentity": 0.2},
        "distributed.scheduler.work-stealing-interval": "20ms",
    },
)
async def test_steal_when_more_tasks(c, s, a, *rest):
    x = c.submit(mul, b"0", 50000000, workers=a.address)  # 50 MB
    await wait(x)

    futures = [c.submit(slowidentity, x, pure=False, delay=0.2) for i in range(20)]

    start = time()
    while not any(w.state.tasks for w in rest):
        await asyncio.sleep(0.01)
        assert time() < start + 1


@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 1)] * 10,
    config={
        "distributed.scheduler.default-task-durations": {
            "slowidentity": 0.2,
            "slow2": 1,
        },
    },
)
async def test_steal_more_attractive_tasks(c, s, a, *rest):
    ext = s.extensions["stealing"]

    def slow2(x):
        sleep(1)
        return x

    x = c.submit(mul, b"0", 100000000, workers=a.address)  # 100 MB
    await wait(x)

    # The submits below are all individual update_graph calls which are very
    # likely submitted in the same batch.
    # Prior to https://github.com/dask/distributed/pull/8049, the entire batch
    # would be processed by the scheduler in the same event loop tick.
    # Therefore, the first PC `stealing.balance` call would be guaranteed to see
    # all the tasks and make the correct decision.
    # After the PR, the batch is processed in multiple event loop ticks, so the
    # first PC `stealing.balance` call would potentially only see the first
    # tasks and would try to rebalance them instead of the slow and heavy one.
    # To guarantee that the stealing extension sees all tasks, we're stopping
    # the callback and are calling balance ourselves once we are certain the
    # tasks are all on the scheduler.
    # Related https://github.com/dask/distributed/pull/5443
    await ext.stop()
    futures = [c.submit(slowidentity, x, pure=False, delay=0.2) for i in range(10)]
    future = c.submit(slow2, x, priority=-1)

    while future.key not in s.tasks:
        await asyncio.sleep(0.01)
    # Now call it once explicitly to move the heavy task
    ext.balance()
    while not any(w.state.tasks for w in rest):
        await asyncio.sleep(0.01)

    # good future moves first
    assert any(future.key in w.state.tasks for w in rest)


async def assert_balanced(inp, expected, c, s, *workers):
    steal = s.extensions["stealing"]
    await steal.stop()
    ev = Event()

    def block(*args, event, **kwargs):
        event.wait()

    counter = itertools.count()

    futures_per_worker = defaultdict(list)
    for w, tasks in zip(workers, inp):
        for t in sorted(tasks, reverse=True):
            if t:
                [dat] = await c.scatter(
                    [gen_nbytes(int(t * s.bandwidth))], workers=w.address
                )
            else:
                dat = 123
            i = next(counter)
            f = c.submit(
                block,
                dat,
                event=ev,
                key="%d-%d" % (int(t), i),
                workers=w.address,
                allow_other_workers=True,
                pure=False,
                priority=-i,
            )
            futures_per_worker[w].append(f)

    # Make sure all tasks are scheduled on the workers
    # We are relying on the futures not to be rootish (and thus not to remain in the
    # scheduler-side queue) because they have worker restrictions
    wait_for_states = []
    for w, fs in futures_per_worker.items():
        for i, f in enumerate(fs):
            # Make sure the first task is executing, all others are ready
            state = "executing" if i == 0 else "ready"
            wait_for_states.append(wait_for_state(f.key, state, w))
    await asyncio.gather(*wait_for_states)

    # Balance several times since stealing might attempt to steal the already executing task
    # for each saturated worker and will need a chance to correct its mistake
    previous = steal.count
    print("# balance", steal.count)
    steal.balance()
    await steal.stop()
    print(steal.count)
    print([msg[1] for msg in s.get_events("stealing")])
    while steal.count != previous:
        previous = steal.count
        print("# balance", steal.count)
        steal.balance()
        # steal.stop() ensures that all in-flight stealing requests have been resolved
        await steal.stop()
        print(steal.count)
        print([msg[1] for msg in s.get_events("stealing")])
    await ev.set()
    for w in workers:
        w.block_gather_dep.set()
    await c.gather([f for fs in futures_per_worker.values() for f in fs])

    result = [
        sorted(
            # The name of input data starts with ``SizeOf``
            (int(key_split(t)) for t in w.data.keys() if not t.startswith("SizeOf")),
            reverse=True,
        )
        for w in workers
    ]

    result2 = sorted(result, reverse=True)
    expected2 = sorted(expected, reverse=True)

    assert result2 == expected2


@pytest.mark.parametrize(
    "inp,expected",
    [
        pytest.param([[1], []], [[1], []], id="don't move unnecessarily"),
        pytest.param([[0, 0], []], [[0], [0]], id="balance"),
        pytest.param(
            [[0.1, 0.1], []], [[0], [0]], id="balance even if results in even"
        ),
        pytest.param([[0, 0, 0], []], [[0, 0], [0]], id="don't over balance"),
        pytest.param(
            [[0, 0], [0, 0, 0], []], [[0, 0], [0, 0], [0]], id="move from larger"
        ),
        pytest.param([[0, 0, 0], [0], []], [[0, 0], [0], [0]], id="move to smaller"),
        pytest.param([[0, 1], []], [[1], [0]], id="choose easier first"),
        pytest.param([[0, 0, 0, 0], [], []], [[0, 0], [0], [0]], id="spread evenly"),
        pytest.param([[1, 0, 2, 0], [], []], [[2, 1], [0], [0]], id="move easier"),
        pytest.param(
            [[1, 1, 1], []], [[1, 1], [1]], id="be willing to move costly items"
        ),
        pytest.param(
            [[1, 1, 1, 1], []], [[1, 1, 1], [1]], id="but don't move too many"
        ),
        pytest.param(
            [[0, 0], [0, 0], [0, 0], []],
            [[0, 0], [0, 0], [0], [0]],
            id="no one clearly saturated",
        ),
        pytest.param(
            [[4, 2, 2, 2, 2, 1, 1], [4, 2, 1, 1], [], [], []],
            [[4, 2, 2, 2, 2], [4, 2, 1], [1], [1], [1]],
            id="balance multiple saturated workers",
        ),
    ],
)
def test_balance(inp, expected):
    async def test_balance_(*args, **kwargs):
        await assert_balanced(inp, expected, *args, **kwargs)

    config = {
        "distributed.scheduler.default-task-durations": {str(i): 1 for i in range(10)}
    }
    gen_cluster(
        client=True,
        Worker=BlockedGatherDep,
        nthreads=[("", 1)] * len(inp),
        config=config,
    )(test_balance_)()


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 2, Worker=Nanny, timeout=60)
async def test_restart(c, s, a, b):
    futures = c.map(
        slowinc, range(100), delay=0.01, workers=a.address, allow_other_workers=True
    )
    while not s.workers[b.worker_address].processing:
        await asyncio.sleep(0.01)

    # Unknown tasks are never stolen therefore wait for a measurement
    while not any(s.tasks[f.key].state == "memory" for f in futures):
        await asyncio.sleep(0.01)

    steal = s.extensions["stealing"]
    assert any(x for L in steal.stealable.values() for x in L)

    await c.restart()

    assert not any(x for L in steal.stealable.values() for x in L)


@gen_cluster(client=True)
async def test_steal_twice(c, s, a, b):
    x = c.submit(inc, 1, workers=a.address)
    await wait(x)

    futures = [c.submit(slowadd, x, i, delay=0.2) for i in range(100)]

    while len(s.tasks) < 100:  # tasks are all allocated
        await asyncio.sleep(0.01)
    if math.isinf(s.WORKER_SATURATION):
        # Wait for b to start stealing tasks
        while len(b.state.tasks) < 30:
            await asyncio.sleep(0.01)
    else:
        # Wait for b to complete some tasks
        while len(b.data) < 8:
            await asyncio.sleep(0.01)

    # Army of new workers arrives to help
    async with contextlib.AsyncExitStack() as stack:
        # This is pretty timing sensitive
        workers = [stack.enter_async_context(Worker(s.address)) for _ in range(10)]
        workers = await asyncio.gather(*workers)

        await wait(futures)

        # Note: this includes a and b
        empty_workers = [ws for ws in s.workers.values() if not ws.has_what]
        assert (
            len(empty_workers) < 3
        ), f"Too many workers without keys ({len(empty_workers)} out of {len(s.workers)})"
        # This also tests that some tasks were stolen from b
        # (see `while len(b.state.tasks) < 30` above)
        # If queuing is enabled, then there was nothing to steal from b,
        # so this just tests the queue was balanced not-terribly.
        assert max(len(ws.has_what) for ws in s.workers.values()) < 30

        assert a.state.in_flight_tasks_count == 0
        assert b.state.in_flight_tasks_count == 0


@gen_cluster(
    client=True,
    nthreads=[("", 1)] * 3,
    config={"distributed.worker.memory.pause": False},
)
async def test_paused_workers_must_not_steal(c, s, w1, w2, w3):
    w2.status = Status.paused
    while s.workers[w2.address].status != Status.paused:
        await asyncio.sleep(0.01)

    x = c.submit(inc, 1, workers=w1.address)
    await wait(x)

    futures = [c.submit(slowadd, x, i, delay=0.1) for i in range(10)]
    await wait(futures)

    assert w1.data
    assert not w2.data
    assert w3.data


@gen_cluster(client=True)
async def test_dont_steal_already_released(c, s, a, b):
    future = c.submit(slowinc, 1, delay=0.05, workers=a.address)
    key = future.key
    while key not in a.state.tasks:
        await asyncio.sleep(0.05)

    del future

    while key in a.state.tasks and a.state.tasks[key].state != "released":
        await asyncio.sleep(0.05)

    a.handle_stimulus(StealRequestEvent(key=key, stimulus_id="test"))
    assert len(a.batched_stream.buffer) == 1
    msg = a.batched_stream.buffer[0]
    assert msg["op"] == "steal-response"
    assert msg["key"] == key
    assert msg["state"] in [None, "released"]

    with captured_logger(
        logging.getLogger("distributed.stealing"), level=logging.DEBUG
    ) as stealing_logs:
        msg = f"Key released between request and confirm: {key}"
        while msg not in stealing_logs.getvalue():
            await asyncio.sleep(0.05)


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 2)
async def test_dont_steal_long_running_tasks(c, s, a, b):
    def long(delay):
        with worker_client() as c:
            sleep(delay)

    await c.submit(long, 0.1)  # learn duration
    await c.submit(inc, 1)  # learn duration

    long_tasks = c.map(long, [0.5, 0.6], workers=a.address, allow_other_workers=True)
    while sum(len(ws.long_running) for ws in s.workers.values()) < 2:  # let them start
        await asyncio.sleep(0.01)

    start = time()
    while any(t.key in s.extensions["stealing"].key_stealable for t in long_tasks):
        await asyncio.sleep(0.01)
        assert time() < start + 1

    na = a.state.executing_count
    nb = b.state.executing_count

    incs = c.map(inc, range(100), workers=a.address, allow_other_workers=True)

    await asyncio.sleep(0.2)
    await wait(long_tasks)

    for t in long_tasks:
        assert (
            sum(log[1] == "executing" for log in a.state.story(t))
            + sum(log[1] == "executing" for log in b.state.story(t))
        ) <= 1


@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 5)] * 2,
    config={"distributed.scheduler.work-stealing-interval": "20ms"},
)
async def test_cleanup_repeated_tasks(c, s, a, b):
    class Foo:
        pass

    await c.submit(slowidentity, -1, delay=0.1)
    objects = [c.submit(Foo, pure=False, workers=a.address) for _ in range(50)]

    x = c.map(
        slowidentity, objects, workers=a.address, allow_other_workers=True, delay=0.05
    )
    del objects
    await wait(x)
    assert a.data and b.data
    assert len(a.data) + len(b.data) > 10
    ws = weakref.WeakSet()
    ws.update(a.data.values())
    ws.update(b.data.values())
    del x

    start = time()
    while a.data or b.data:
        await asyncio.sleep(0.01)
        assert time() < start + 1

    assert not s.tasks

    with profile.lock:
        assert not list(ws)


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 2)
async def test_lose_task(c, s, a, b):
    with captured_logger("distributed.stealing") as log:
        s.periodic_callbacks["stealing"].interval = 1
        for _ in range(100):
            futures = c.map(
                slowinc,
                range(10),
                delay=0.01,
                pure=False,
                workers=a.address,
                allow_other_workers=True,
            )
            await asyncio.sleep(0.01)
            del futures

    out = log.getvalue()
    assert "Error" not in out


@pytest.mark.parametrize("interval, expected", [(None, 100), ("500ms", 500), (2, 2)])
@gen_cluster(nthreads=[], config={"distributed.scheduler.work-stealing": False})
async def test_parse_stealing_interval(s, interval, expected):
    from distributed.scheduler import WorkStealing

    if interval:
        ctx = dask.config.set(
            {"distributed.scheduler.work-stealing-interval": interval}
        )
    else:
        ctx = contextlib.nullcontext()
    with ctx:
        ws = WorkStealing(s)
        await ws.start()
        assert s.periodic_callbacks["stealing"].callback_time == expected


@gen_cluster(client=True)
async def test_balance_with_longer_task(c, s, a, b):
    np = pytest.importorskip("numpy")

    await c.submit(slowinc, 0, delay=0)  # scheduler learns that slowinc is very fast
    x = await c.scatter(np.arange(10000), workers=[a.address])
    y = c.submit(
        slowinc, 1, delay=5, workers=[a.address], priority=1
    )  # a surprisingly long task
    z = c.submit(
        slowadd, x, 1, workers=[a.address], allow_other_workers=True, priority=0
    )  # a task after y, suggesting a, but open to b

    # Allow task to be learned, otherwise it will not be stolen
    _ = c.submit(slowadd, x, 2, workers=[b.address])
    await z
    assert not y.done()
    assert z.key in b.data


@gen_cluster(client=True)
async def test_blocklist_shuffle_split(c, s, a, b):
    pd = pytest.importorskip("pandas")
    dd = pytest.importorskip("dask.dataframe")
    npart = 10
    df = dd.from_pandas(pd.DataFrame({"A": range(100), "B": 1}), npartitions=npart)
    with dask.config.set({"dataframe.shuffle.method": "tasks"}):
        graph = df.shuffle(
            "A",
            # If we don't have enough partitions, we'll fall back to a simple shuffle
            max_branch=npart - 1,
        ).sum()
    res = c.compute(graph)

    while not s.tasks:
        await asyncio.sleep(0.005)

    from distributed.stealing import fast_tasks

    blocked = fast_tasks & s.task_prefixes.keys()
    assert blocked
    assert any(["split" in prefix for prefix in blocked])

    stealable = s.extensions["stealing"].stealable
    while not res.done():
        for tasks_per_level in stealable.values():
            for tasks in tasks_per_level:
                for ts in tasks:
                    assert ts.prefix.name not in fast_tasks
                    assert "split" not in ts.prefix.name
        await asyncio.sleep(0.001)
    await res


@gen_cluster(
    client=True,
    nthreads=[("", 1)] * 3,
    config={
        "distributed.scheduler.work-stealing-interval": 1_000_000,
    },
)
async def test_steal_concurrent_simple(c, s, *workers):
    steal = s.extensions["stealing"]
    w0 = workers[0]
    w1 = workers[1]
    w2 = workers[2]
    futs1 = c.map(
        slowinc,
        range(10),
        key=[f"f1-{ix}" for ix in range(10)],
        workers=[w0.address],
    )

    while not w0.state.tasks:
        await asyncio.sleep(0.1)

    # ready is a heap but we don't need last, just not the next
    victim_key = w0.state.ready.peekright().key
    victim_ts = s.tasks[victim_key]

    ws0 = s.workers[w0.address]
    ws1 = s.workers[w1.address]
    ws2 = s.workers[w2.address]
    steal.move_task_request(victim_ts, ws0, ws1)
    steal.move_task_request(victim_ts, ws0, ws2)

    await c.gather(futs1)

    # First wins
    assert ws1.has_what
    assert not ws2.has_what


@gen_cluster(client=True)
async def test_steal_reschedule_reset_in_flight_occupancy(c, s, *workers):
    # https://github.com/dask/distributed/issues/5370
    steal = s.extensions["stealing"]
    await steal.stop()
    w0 = workers[0]
    roots = c.map(
        inc,
        range(6),
        key=[f"r-{ix}" for ix in range(6)],
    )

    def block(x, event):
        event.wait()
        return x + 1

    event = Event()
    futs1 = [
        c.submit(block, r, event=event, key=f"f{ir}-{ix}")
        for ir, r in enumerate(roots)
        for ix in range(4)
    ]
    while not w0.state.ready:
        await asyncio.sleep(0.01)

    # ready is a heap but we don't need last, just not the next
    victim_key = w0.state.ready.peekright().key
    victim_ts = s.tasks[victim_key]

    wsA = victim_ts.processing_on
    other_workers = [ws for ws in s.workers.values() if ws != wsA]
    wsB = other_workers[0]

    steal.move_task_request(victim_ts, wsA, wsB)

    s._reschedule(victim_key, stimulus_id="test")
    await event.set()
    await c.gather(futs1)

    del futs1

    assert all(v == 0 for v in steal.in_flight_occupancy.values())


@gen_cluster(
    client=True,
    config={
        "distributed.scheduler.work-stealing-interval": 10,
    },
)
async def test_get_story(c, s, a, b):
    steal = s.extensions["stealing"]
    futs = c.map(slowinc, range(100), workers=[a.address], allow_other_workers=True)
    collect = c.submit(sum, futs)
    await collect
    key = next(iter(b.state.tasks))
    ts = s.tasks[key]
    msgs = steal.story(key)
    msgs_ts = steal.story(ts)
    assert msgs
    assert msgs == msgs_ts
    assert all(isinstance(m, tuple) for m in msgs)


@gen_cluster(
    client=True,
    config={
        "distributed.scheduler.work-stealing-interval": 1_000_000,
    },
)
async def test_steal_worker_dies_same_ip(c, s, w0, w1):
    # https://github.com/dask/distributed/issues/5370
    steal = s.extensions["stealing"]
    ev = Event()
    futs1 = c.map(
        lambda _, ev: ev.wait(),
        range(10),
        ev=ev,
        key=[f"f1-{ix}" for ix in range(10)],
        workers=[w0.address],
        allow_other_workers=True,
    )
    while not w0.active_keys:
        await asyncio.sleep(0.01)

    # ready is a heap but we don't need last, just not the next
    victim_key = w0.state.ready.peekright().key
    victim_ts = s.tasks[victim_key]

    wsA = victim_ts.processing_on
    assert wsA.address == w0.address
    wsB = s.workers[w1.address]

    steal.move_task_request(victim_ts, wsA, wsB)
    len_before = len(s.get_events("stealing"))
    with freeze_batched_send(w0.batched_stream):
        while not any(
            isinstance(event, StealRequestEvent) for event in w0.state.stimulus_log
        ):
            await asyncio.sleep(0.1)
        async with contextlib.AsyncExitStack() as stack:
            # Block batched stream of w0 to ensure the steal-confirmation doesn't
            # arrive at the scheduler before we want it to
            await w1.close()
            # Kill worker wsB
            # Restart new worker with same IP, name, etc.
            while w1.address in s.workers:
                await asyncio.sleep(0.1)

            w_new = await stack.enter_async_context(
                Worker(
                    s.address,
                    host=w1.host,
                    port=w1.port,
                    name=w1.name,
                )
            )
            wsB2 = s.workers[w_new.address]
            assert wsB2.address == wsB.address
            assert wsB2 is not wsB
            assert wsB2 != wsB
            assert hash(wsB2) != hash(wsB)

    # Wait for the steal response to arrive
    while len_before == len(s.get_events("stealing")):
        await asyncio.sleep(0.1)

    assert victim_ts.processing_on != wsB

    await w_new.close(executor_wait=False)
    await ev.set()
    await c.gather(futs1)


@gen_cluster(
    client=True,
    nthreads=[("", 1)] * 3,
    config={
        "distributed.scheduler.work-stealing-interval": 1_000_000,
    },
)
async def test_reschedule_concurrent_requests_deadlock(c, s, *workers):
    # https://github.com/dask/distributed/issues/5370
    steal = s.extensions["stealing"]
    w0 = workers[0]
    ev = Event()
    futs1 = c.map(
        lambda _, ev: ev.wait(),
        range(10),
        ev=ev,
        key=[f"f1-{ix}" for ix in range(10)],
        workers=[w0.address],
        allow_other_workers=True,
    )
    while not w0.active_keys:
        await asyncio.sleep(0.01)

    # ready is a heap but we don't need last, just not the next
    victim_key = list(w0.active_keys)[0]

    victim_ts = s.tasks[victim_key]

    wsA = victim_ts.processing_on
    other_workers = [ws for ws in s.workers.values() if ws != wsA]
    wsB = other_workers[0]
    wsC = other_workers[1]

    steal.move_task_request(victim_ts, wsA, wsB)

    s.set_restrictions(worker={victim_key: [wsB.address]})
    s._reschedule(victim_key, stimulus_id="test")
    assert wsB == victim_ts.processing_on
    # move_task_request is not responsible for respecting worker restrictions
    steal.move_task_request(victim_ts, wsB, wsC)

    # Let tasks finish
    await ev.set()
    await c.gather(futs1)

    assert victim_ts.who_has != {wsC}
    msgs = steal.story(victim_ts)
    msgs = [msg[:-1] for msg in msgs]  # Remove random IDs

    # There are three possible outcomes
    expect1 = [
        ("stale-response", victim_key, "executing", wsA.address),
        ("already-computing", victim_key, "executing", wsB.address, wsC.address),
    ]
    expect2 = [
        ("already-computing", victim_key, "executing", wsB.address, wsC.address),
        ("already-aborted", victim_key, "executing", wsA.address),
    ]
    # This outcome appears only in ~2% of the runs
    expect3 = [
        ("already-computing", victim_key, "executing", wsB.address, wsC.address),
        ("already-aborted", victim_key, "memory", wsA.address),
    ]
    assert msgs in (expect1, expect2, expect3)


@pytest.mark.skip("executing heartbeats not considered yet")
@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 3)
async def test_correct_bad_time_estimate(c, s, *workers):
    """Initial time estimation causes the task to not be considered for
    stealing. Following occupancy readjustments will re-enlist the keys since
    the duration estimate is now significant.

    This is done during reevaluate occupancy
    """
    steal = s.extensions["stealing"]
    future = c.submit(slowinc, 1, delay=0)
    await wait(future)
    futures = [c.submit(slowinc, future, delay=0.1, pure=False) for _ in range(20)]
    while len(s.tasks) < 21:
        await asyncio.sleep(0)
    assert not any(s.tasks[f.key] in steal.key_stealable for f in futures)
    await asyncio.sleep(0.5)
    assert any(s.tasks[f.key] in steal.key_stealable for f in futures)
    await wait(futures)
    assert all(w.data for w in workers), [sorted(w.data) for w in workers]


@gen_cluster(client=True)
async def test_steal_stimulus_id_unique(c, s, a, b):
    steal = s.extensions["stealing"]
    num_futs = 1_000
    async with Lock() as lock:

        def blocked(x, lock):
            lock.acquire()

        # Setup all tasks on worker 0 such that victim/thief relation is the
        # same for all tasks.
        futures = c.map(blocked, range(num_futs), lock=lock, workers=[a.address])
        # Ensure all tasks are assigned to the worker since otherwise the
        # move_task_request fails.
        while len(a.state.tasks) != num_futs:
            await asyncio.sleep(0.1)
        tasks = [s.tasks[f.key] for f in futures]
        w0 = s.workers[a.address]
        w1 = s.workers[b.address]
        # Generating the move task requests as fast as possible increases the
        # chance of duplicates if the uniqueness is not guaranteed.
        for ts in tasks:
            steal.move_task_request(ts, w0, w1)
        # Values stored in in_flight are used for response verification.
        # Therefore all stimulus IDs are stored here and must be unique
        stimulus_ids = {dct["stimulus_id"] for dct in steal.in_flight.values()}
        assert len(stimulus_ids) == num_futs
        await c.cancel(futures)


def test_steal_worker_state(ws_with_running_task):
    ws = ws_with_running_task

    ws.handle_stimulus(FreeKeysEvent(keys=["x"], stimulus_id="s1"))
    assert ws.available_resources == {"R": 0}
    assert ws.tasks["x"].state == "cancelled"

    instructions = ws.handle_stimulus(ExecuteSuccessEvent.dummy("x", stimulus_id="s2"))
    assert not instructions
    assert "x" not in ws.tasks
    assert "x" not in ws.data
    assert ws.available_resources == {"R": 1}


@pytest.mark.slow()
@gen_cluster(nthreads=[("", 1)] * 4, client=True)
async def test_steal_very_fast_tasks(c, s, *workers):
    np = pytest.importorskip("numpy")

    # Ensure that very fast tasks are allowed to be stolen
    root = dask.delayed(lambda n: "x" * n)(
        dask.utils.parse_bytes("1MiB"), dask_key_name="root"
    )

    @dask.delayed
    def func(*args):
        import time

        time.sleep(0.002)

    ntasks = 1000
    results = [func(root, i) for i in range(ntasks)]
    futs = c.compute(results)
    await c.gather(futs)

    ntasks_per_worker = np.array([len(w.data) for w in workers])
    ideal = ntasks / len(workers)
    assert (ntasks_per_worker > ideal * 0.5).all(), (ideal, ntasks_per_worker)
    assert (ntasks_per_worker < ideal * 1.5).all(), (ideal, ntasks_per_worker)


@pytest.mark.parametrize(
    "cost, ntasks, expect_steal",
    [
        pytest.param(10, 10, False, id="not enough work to steal"),
        pytest.param(10, 12, True, id="enough work to steal"),
        pytest.param(20, 12, False, id="not enough work for increased cost"),
    ],
)
def test_balance_expensive_tasks(cost, ntasks, expect_steal):
    dependencies = {"a": cost}
    dependency_placement = [["a"], []]
    task_placement = [[["a"]] * ntasks, []]

    def _correct_placement(actual):
        actual_task_counts = [len(placed) for placed in actual]
        return sum(actual_task_counts) == ntasks and (
            (actual_task_counts[1] > 0) == expect_steal
        )

    _run_dependency_balance_test(
        dependencies,
        dependency_placement,
        task_placement,
        _correct_placement,
    )


def test_balance_uneven_without_replica():
    dependencies = {"a": 1}
    dependency_placement = [["a"], []]
    task_placement = [[["a"], ["a"]], []]

    def _correct_placement(actual):
        actual_task_counts = [len(placed) for placed in actual]
        return actual_task_counts == [2, 0]

    _run_dependency_balance_test(
        dependencies,
        dependency_placement,
        task_placement,
        _correct_placement,
    )


def test_balance_eventually_steals_large_dependency_without_replica():
    dependencies = {"a": 10}
    dependency_placement = [["a"], []]
    task_placement = [[["a"]] * 20, []]

    def _correct_placement(actual):
        actual_task_counts = [len(placed) for placed in actual]
        return sum(actual_task_counts) == 20 and actual_task_counts[1] > 0

    _run_dependency_balance_test(
        dependencies,
        dependency_placement,
        task_placement,
        _correct_placement,
    )


def test_balance_even_with_replica():
    dependencies = {"a": 1}
    dependency_placement = [["a"], ["a"]]
    task_placement = [[["a"], ["a"]], []]

    def _correct_placement(actual):
        actual_task_counts = [len(placed) for placed in actual]
        return actual_task_counts == [
            1,
            1,
        ]

    _run_dependency_balance_test(
        dependencies,
        dependency_placement,
        task_placement,
        _correct_placement,
    )


def test_balance_to_replica():
    dependencies = {"a": 2}
    dependency_placement = [["a"], ["a"], []]
    task_placement = [[["a"], ["a"]], [], []]

    def _correct_placement(actual):
        actual_task_counts = [len(placed) for placed in actual]
        return actual_task_counts == [
            1,
            1,
            0,
        ]

    _run_dependency_balance_test(
        dependencies,
        dependency_placement,
        task_placement,
        _correct_placement,
    )


def test_balance_multiple_to_replica():
    dependencies = {"a": 6}
    dependency_placement = [["a"], ["a"], []]
    task_placement = [[["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"], ["a"]], [], []]

    def _correct_placement(actual):
        actual_task_counts = [len(placed) for placed in actual]
        # FIXME: A better task placement would be even but the current balancing
        # logic only steals when a worker has an idle thread which makes it a
        # little less aggressive
        return actual_task_counts == [
            6,
            2,
            0,
        ]

    _run_dependency_balance_test(
        dependencies,
        dependency_placement,
        task_placement,
        _correct_placement,
    )


def test_balance_to_larger_dependency():
    dependencies = {"a": 2, "b": 1}
    dependency_placement = [["a", "b"], ["a"], ["b"]]
    task_placement = [[["a", "b"], ["a", "b"], ["a", "b"]], [], []]

    def _correct_placement(actual):
        actual_task_counts = [len(placed) for placed in actual]
        return actual_task_counts == [
            2,
            1,
            0,
        ]

    _run_dependency_balance_test(
        dependencies,
        dependency_placement,
        task_placement,
        _correct_placement,
    )


def test_balance_prefers_busier_with_dependency():
    dependencies = {"a": 5, "b": 1}
    dependency_placement = [["a"], ["a", "b"], []]
    task_placement = [
        [["a"], ["a"], ["a"], ["a"], ["a"], ["a"]],
        [["b"]],
        [],
    ]

    def _correct_placement(actual):
        actual_task_placements = [sorted(placed) for placed in actual]
        # FIXME: A better task placement would be even but the current balancing
        # logic aborts as soon as a worker is no longer classified as idle
        # return actual_task_placements == [
        #     [["a"], ["a"], ["a"], ["a"]],
        #     [["a"], ["a"], ["b"]],
        #     [],
        # ]
        return actual_task_placements == [
            [["a"], ["a"], ["a"], ["a"], ["a"]],
            [["a"], ["b"]],
            [],
        ]

    _run_dependency_balance_test(
        dependencies,
        dependency_placement,
        task_placement,
        _correct_placement,
        # This test relies on disabling queueing to flag workers as idle
        config={
            "distributed.scheduler.worker-saturation": float("inf"),
        },
    )


def _run_dependency_balance_test(
    dependencies: Mapping[str, int],
    dependency_placement: list[list[str]],
    task_placement: list[list[list[str]]],
    correct_placement_fn: Callable[[list[list[list[str]]]], bool],
    config: dict | None = None,
) -> None:
    """Run a test for balancing with task dependencies according to the provided
    specifications.

    This method executes the test logic for all permutations of worker placements
    and generates a new cluster for each one.

    Parameters
    ----------
    dependencies
        Mapping of task dependencies to their weight.
    dependency_placement
        List of list of dependencies to be placed on the worker corresponding
        to the index of the outer list.
    task_placement
        List of list of tasks to be placed on the worker corresponding to the
        index of the outer list. Each task is a list of names of dependencies.
    correct_placement_fn
        Callable used to determine if stealing placed the tasks as expected.
    config
        Optional configuration to apply to the test.
    See Also
    --------
    _dependency_balance_test_permutation
    """
    nworkers = len(task_placement)
    for permutation in itertools.permutations(range(nworkers)):

        async def _run(
            *args,
            permutation=permutation,
            **kwargs,
        ):
            await _dependency_balance_test_permutation(
                dependencies,
                dependency_placement,
                task_placement,
                correct_placement_fn,
                permutation,
                *args,
                **kwargs,
            )

        default_task_durations = {
            compose_task_prefix(deps): "1s"
            for tasks in task_placement
            for deps in tasks
        }

        gen_cluster(
            client=True,
            nthreads=[("", 1)] * len(task_placement),
            config=merge(
                NO_AMM,
                config or {},
                {
                    "distributed.scheduler.default-task-durations": default_task_durations,
                },
            ),
            # Avoid heartbeats since comm costs are sensitive to bandwidth updates
            worker_kwargs={"heartbeat_interval": "100s"},
        )(_run)()


def compose_task_prefix(dependencies: list[str]) -> str:
    dep_key = "".join(sorted(dependencies))
    return f"task-{dep_key}"


async def _dependency_balance_test_permutation(
    dependencies: Mapping[str, int],
    dependency_placement: list[list[str]],
    task_placement: list[list[list[str]]],
    correct_placement_fn: Callable[[list[list[list[str]]]], bool],
    permutation: list[int],
    c: Client,
    s: Scheduler,
    *workers: Worker,
) -> None:
    """Run a test for balancing with task dependencies according to the provided
    specifications and worker permutations.

    Parameters
    ----------
    dependencies
        Mapping of task dependencies to their weight.
    dependency_placement
        List of list of dependencies to be placed on the worker corresponding
        to the index of the outer list.
    task_placement
        List of list of tasks to be placed on the worker corresponding to the
        index of the outer list. Each task is a list of names of dependencies.
    correct_placement_fn
        Callable used to determine if stealing placed the tasks as expected.
    permutation
        Permutation of workers to use for this run.

    See Also
    --------
    _run_dependency_balance_test
    """
    steal = s.extensions["stealing"]
    await steal.stop()

    inverse = [permutation.index(i) for i in range(len(permutation))]
    permutated_dependency_placement = [dependency_placement[i] for i in permutation]
    permutated_task_placement = [task_placement[i] for i in permutation]

    dependency_futures = await _place_dependencies(
        dependencies, permutated_dependency_placement, c, s, workers
    )

    ev, futures_per_worker = await _place_tasks(
        permutated_task_placement,
        permutated_dependency_placement,
        dependency_futures,
        c,
        s,
        workers,
    )

    # Balance several since stealing might attempt to steal the already executing task
    # for each saturated worker and will need a chance to correct its mistake
    for _ in workers:
        steal.balance()
        # steal.stop() ensures that all in-flight stealing requests have been resolved
        await steal.stop()

    await ev.set()
    await c.gather([f for fs in futures_per_worker.values() for f in fs])

    permutated_actual_placement = _get_task_placement(s, workers)
    actual_placement = [permutated_actual_placement[i] for i in inverse]

    assert correct_placement_fn(actual_placement), (actual_placement, permutation)


async def _place_dependencies(
    dependencies: Mapping[str, int],
    placement: list[list[str]],
    c: Client,
    s: Scheduler,
    workers: Sequence[Worker],
) -> dict[str, Future]:
    """Places the dependencies on the workers as specified.

    Parameters
    ----------
    dependencies
        Mapping of task dependencies to their weight.
    placement
        List of list of dependencies to be placed on the worker corresponding to the
        index of the outer list.

    Returns
    -------
    Dictionary of futures matching the input dependencies.

    See Also
    --------
    _run_dependency_balance_test
    """
    dependencies_to_workers = defaultdict(set)
    for worker_idx, placed in enumerate(placement):
        for dependency in placed:
            dependencies_to_workers[dependency].add(workers[worker_idx].address)

    futures = {}
    for name, multiplier in dependencies.items():
        key = f"dep-{name}"
        worker_addresses = dependencies_to_workers[name]
        futs = await c.scatter(
            {key: gen_nbytes(int(multiplier * s.bandwidth))},
            workers=worker_addresses,
            broadcast=True,
        )
        futures[name] = futs[key]

    await c.gather(futures.values())

    return futures


async def _place_tasks(
    placement: list[list[list[str]]],
    dependency_placement: list[list[str]],
    dependency_futures: Mapping[str, Future],
    c: Client,
    s: Scheduler,
    workers: Sequence[Worker],
) -> tuple[Event, dict[Worker, list[Future]]]:
    """Places the tasks on the workers as specified.

    Parameters
    ----------
    placement
        List of list of tasks to be placed on the worker corresponding to the
        index of the outer list. Each task is a list of names of dependencies.
    dependency_placement
        List of list of dependencies to be placed on the worker corresponding to the
        index of the outer list.
    dependency_futures
        Mapping of dependency names to their corresponding futures.

    Returns
    -------
    Tuple of the event blocking the placed tasks and list of futures matching
    the input task placement.

    See Also
    --------
    _run_dependency_balance_test
    """
    ev = Event()

    def block(*args, event, **kwargs):
        event.wait()

    counter = itertools.count()
    futures_per_worker = defaultdict(list)
    for worker, tasks, placed_dependencies in zip(
        workers, placement, dependency_placement
    ):
        for dependencies in tasks:
            for dependency in dependencies:
                assert dependency in placed_dependencies, (
                    f"Dependency {dependency} of task {dependencies} not found "
                    "on worker {worker}. Make sure that workers already hold all "
                    "dependencies of their tasks to avoid transfers and skewing "
                    "bandwidth measurements"
                )
            i = next(counter)
            key = f"{compose_task_prefix(dependencies)}-{i}"
            f = c.submit(
                block,
                [dependency_futures[dependency] for dependency in dependencies],
                event=ev,
                key=key,
                workers=worker.address,
                allow_other_workers=True,
                pure=False,
                priority=-i,
            )
            futures_per_worker[worker].append(f)

    # Make sure all tasks are scheduled on the workers
    # We are relying on the futures not to be rootish (and thus not to remain in the
    # scheduler-side queue) because they have worker restrictions
    waits_for_state: list[Coroutine] = []
    for w, fs in futures_per_worker.items():
        waits_for_executing_state = []
        for f in fs:
            # Every task should be either ready or executing
            waits_for_state.append(wait_for_state(f.key, ["executing", "ready"], w))
            waits_for_executing_state.append(
                asyncio.create_task(wait_for_state(f.key, "executing", w))
            )
        # Ensure that each worker has started executing a task
        waits_for_state.append(
            asyncio.wait(waits_for_executing_state, return_when="FIRST_COMPLETED")
        )
    await asyncio.gather(
        *waits_for_state,
    )

    return ev, futures_per_worker


def _get_task_placement(
    s: Scheduler, workers: Iterable[Worker]
) -> list[list[list[str]]]:
    """Return the placement of tasks on this worker"""
    actual = []
    for w in workers:
        actual.append(
            [
                list(key_split(key[5:]))  # Remove "task-" prefix
                for key in w.data
                if isinstance(key, str) and key.startswith("task-")
            ]
        )
    return _deterministic_placement(actual)


def _deterministic_placement(placement):
    """Return a deterministic ordering of the tasks or dependencies on each worker."""
    return [sorted(placed) for placed in placement]


# Reproducer from https://github.com/dask/distributed/issues/6573
@gen_cluster(
    client=True,
    nthreads=[("", 1)] * 4,
)
async def test_trivial_workload_should_not_cause_work_stealing(c, s, *workers):
    root = dask.delayed(lambda n: "x" * n)(parse_bytes("1MiB"), dask_key_name="root")
    results = [dask.delayed(lambda *args: None)(root, i) for i in range(1000)]
    futs = c.compute(results)
    await c.gather(futs)
    events = s.get_events("stealing")
    assert len(events) == 0

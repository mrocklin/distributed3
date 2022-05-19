import asyncio

import distributed
from distributed import Event, Lock, Worker
from distributed.utils_test import (
    _LockedCommPool,
    assert_story,
    gen_cluster,
    inc,
    slowinc,
)


async def wait_for_state(key, state, dask_worker):
    while key not in dask_worker.tasks or dask_worker.tasks[key].state != state:
        await asyncio.sleep(0.005)


async def wait_for_cancelled(key, dask_worker):
    while key in dask_worker.tasks:
        if dask_worker.tasks[key].state == "cancelled":
            return
        await asyncio.sleep(0.005)
    assert False


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_abort_execution_release(c, s, a):
    fut = c.submit(slowinc, 1, delay=1)
    await wait_for_state(fut.key, "executing", a)
    fut.release()
    await wait_for_cancelled(fut.key, a)


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_abort_execution_reschedule(c, s, a):
    fut = c.submit(slowinc, 1, delay=1)
    await wait_for_state(fut.key, "executing", a)
    fut.release()
    await wait_for_cancelled(fut.key, a)
    fut = c.submit(slowinc, 1, delay=0.1)
    await fut


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_abort_execution_add_as_dependency(c, s, a):
    fut = c.submit(slowinc, 1, delay=1)
    await wait_for_state(fut.key, "executing", a)
    fut.release()
    await wait_for_cancelled(fut.key, a)

    fut = c.submit(slowinc, 1, delay=1)
    fut = c.submit(slowinc, fut, delay=1)
    await fut


@gen_cluster(client=True)
async def test_abort_execution_to_fetch(c, s, a, b):
    ev = Event()

    def f(ev):
        ev.wait()
        return 123

    fut = c.submit(f, ev, key="f1", workers=[a.worker_address])
    await wait_for_state(fut.key, "executing", a)
    fut.release()
    await wait_for_cancelled(fut.key, a)

    # While the first worker is still trying to compute f1, we'll resubmit it to
    # another worker with a smaller delay. The key is still the same
    fut = c.submit(inc, 1, key="f1", workers=[b.worker_address])
    # then, a must switch the execute to fetch. Instead of doing so, it will
    # simply re-use the currently computing result.
    fut = c.submit(inc, fut, workers=[a.worker_address], key="f2")
    await wait_for_state("f2", "waiting", a)
    await ev.set()
    assert await fut == 124  # It would be 3 if the result was copied from b
    del fut
    while "f1" in a.tasks:
        await asyncio.sleep(0.01)

    assert_story(
        a.story("f1"),
        [
            ("f1", "compute-task", "released"),
            ("f1", "released", "waiting", "waiting", {"f1": "ready"}),
            ("f1", "waiting", "ready", "ready", {"f1": "executing"}),
            ("f1", "ready", "executing", "executing", {}),
            ("free-keys", ("f1",)),
            ("f1", "executing", "released", "cancelled", {}),
            ("f1", "cancelled", "fetch", "resumed", {}),
            ("f1", "resumed", "memory", "memory", {"f2": "ready"}),
            ("free-keys", ("f1",)),
            ("f1", "memory", "released", "released", {}),
            ("f1", "released", "forgotten", "forgotten", {}),
        ],
    )


@gen_cluster(client=True)
async def test_worker_find_missing(c, s, a, b):
    fut = c.submit(inc, 1, workers=[a.address])
    await fut
    # We do not want to use proper API since it would ensure that the cluster is
    # informed properly
    del a.data[fut.key]
    del a.tasks[fut.key]

    # Actually no worker has the data; the scheduler is supposed to reschedule
    assert await c.submit(inc, fut, workers=[b.address]) == 3


@gen_cluster(client=True)
async def test_worker_stream_died_during_comm(c, s, a, b):
    write_queue = asyncio.Queue()
    write_event = asyncio.Event()
    b.rpc = _LockedCommPool(
        b.rpc,
        write_queue=write_queue,
        write_event=write_event,
    )
    fut = c.submit(inc, 1, workers=[a.address], allow_other_workers=True)
    await fut
    # Actually no worker has the data; the scheduler is supposed to reschedule
    res = c.submit(inc, fut, workers=[b.address])

    await write_queue.get()
    await a.close()
    write_event.set()

    await res
    assert any("receive-dep-failed" in msg for msg in b.log)


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_flight_to_executing_via_cancelled_resumed(c, s, b):

    block_get_data = asyncio.Lock()
    block_compute = Lock()
    enter_get_data = asyncio.Event()
    await block_get_data.acquire()

    class BrokenWorker(Worker):
        async def get_data(self, comm, *args, **kwargs):
            enter_get_data.set()
            async with block_get_data:
                comm.abort()

    def blockable_compute(x, lock):
        with lock:
            return x + 1

    async with BrokenWorker(s.address) as a:
        await c.wait_for_workers(2)
        fut1 = c.submit(
            blockable_compute,
            1,
            lock=block_compute,
            workers=[a.address],
            allow_other_workers=True,
            key="fut1",
        )
        fut2 = c.submit(inc, fut1, workers=[b.address], key="fut2")

        await enter_get_data.wait()
        await block_compute.acquire()

        # Close in scheduler to ensure we transition and reschedule task properly
        await s.close_worker(worker=a.address, stimulus_id="test")
        await wait_for_state(fut1.key, "resumed", b)

        block_get_data.release()
        await block_compute.release()
        assert await fut2 == 3

        b_story = b.story(fut1.key)
        assert any("receive-dep-failed" in msg for msg in b_story)
        assert any("cancelled" in msg for msg in b_story)
        assert any("resumed" in msg for msg in b_story)


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_executing_cancelled_error(c, s, w):
    """One worker with one thread. We provoke an executing->cancelled transition
    and let the task err. This test ensures that there is no residual state
    (e.g. a semaphore) left blocking the thread"""
    lock = distributed.Lock()
    await lock.acquire()

    async def wait_and_raise(*args, **kwargs):
        async with lock:
            raise RuntimeError()

    fut = c.submit(wait_and_raise)
    await wait_for_state(fut.key, "executing", w)
    # Queue up another task to ensure this is not affected by our error handling
    fut2 = c.submit(inc, 1)
    await wait_for_state(fut2.key, "ready", w)

    fut.release()
    await wait_for_state(fut.key, "cancelled", w)
    await lock.release()

    # At this point we do not fetch the result of the future since the future
    # itself would raise a cancelled exception. At this point we're concerned
    # about the worker. The task should transition over error to be eventually
    # forgotten since we no longer hold a ref.
    while fut.key in w.tasks:
        await asyncio.sleep(0.01)

    assert await fut2 == 2
    # Everything should still be executing as usual after this
    await c.submit(sum, c.map(inc, range(10))) == sum(map(inc, range(10)))

    # Everything above this line should be generically true, regardless of
    # refactoring. Below verifies some implementation specific test assumptions

    story = w.story(fut.key)
    start_finish = [(msg[1], msg[2], msg[3]) for msg in story if len(msg) == 7]
    assert ("executing", "released", "cancelled") in start_finish
    assert ("cancelled", "error", "error") in start_finish
    assert ("error", "released", "released") in start_finish


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_flight_cancelled_error(c, s, b):
    """One worker with one thread. We provoke an flight->cancelled transition
    and let the task err."""
    lock = asyncio.Lock()
    await lock.acquire()

    class BrokenWorker(Worker):
        block_get_data = True

        async def get_data(self, comm, *args, **kwargs):
            if self.block_get_data:
                async with lock:
                    comm.abort()
            return await super().get_data(comm, *args, **kwargs)

    async with BrokenWorker(s.address) as a:
        await c.wait_for_workers(2)
        fut1 = c.submit(inc, 1, workers=[a.address], allow_other_workers=True)
        fut2 = c.submit(inc, fut1, workers=[b.address])
        await wait_for_state(fut1.key, "flight", b)
        fut2.release()
        fut1.release()
        await wait_for_state(fut1.key, "cancelled", b)
        lock.release()
        # At this point we do not fetch the result of the future since the
        # future itself would raise a cancelled exception. At this point we're
        # concerned about the worker. The task should transition over error to
        # be eventually forgotten since we no longer hold a ref.
        while fut1.key in b.tasks:
            await asyncio.sleep(0.01)
        a.block_get_data = False
        # Everything should still be executing as usual after this
        assert await c.submit(sum, c.map(inc, range(10))) == sum(map(inc, range(10)))


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_in_flight_lost_after_resumed(c, s, b):
    block_get_data = asyncio.Lock()
    in_get_data = asyncio.Event()

    lock_executing = Lock()

    def block_execution(lock):
        with lock:
            return

    class BlockedGetData(Worker):
        async def get_data(self, comm, *args, **kwargs):
            in_get_data.set()
            async with block_get_data:
                return await super().get_data(comm, *args, **kwargs)

    async with BlockedGetData(s.address, name="blocked-get-dataworker") as a:
        fut1 = c.submit(
            block_execution,
            lock_executing,
            workers=[a.address],
            allow_other_workers=True,
            key="fut1",
        )
        # Ensure fut1 is in memory but block any further execution afterwards to
        # ensure we control when the recomputation happens
        await fut1
        await lock_executing.acquire()
        in_get_data.clear()
        await block_get_data.acquire()
        fut2 = c.submit(inc, fut1, workers=[b.address], key="fut2")

        # This ensures that B already fetches the task, i.e. after this the task
        # is guaranteed to be in flight
        await in_get_data.wait()
        assert fut1.key in b.tasks
        assert b.tasks[fut1.key].state == "flight"

        # It is removed, i.e. get_data is guaranteed to fail and f1 is scheduled
        # to be recomputed on B
        await s.remove_worker(a.address, "foo", close=False, safe=True)

        while not b.tasks[fut1.key].state == "resumed":
            await asyncio.sleep(0.01)

        fut1.release()
        fut2.release()

        while not b.tasks[fut1.key].state == "cancelled":
            await asyncio.sleep(0.01)

        block_get_data.release()
        while b.tasks:
            await asyncio.sleep(0.01)

    assert_story(
        b.story(fut1.key),
        expect=[
            # The initial free-keys is rejected
            ("free-keys", (fut1.key,)),
            (fut1.key, "resumed", "released", "cancelled", {}),
            # After gather_dep receives the data, the task is forgotten
            (fut1.key, "cancelled", "memory", "released", {fut1.key: "forgotten"}),
        ],
    )


@gen_cluster(client=True)
async def test_cancelled_error(c, s, a, b):
    executing = Event()
    lock_executing = Lock()
    await lock_executing.acquire()

    def block_execution(event, lock):
        event.set()
        with lock:
            raise RuntimeError()

    fut1 = c.submit(
        block_execution,
        executing,
        lock_executing,
        workers=[b.address],
        allow_other_workers=True,
        key="fut1",
    )

    await executing.wait()
    assert b.tasks[fut1.key].state == "executing"
    fut1.release()
    while b.tasks[fut1.key].state == "executing":
        await asyncio.sleep(0.01)
    await lock_executing.release()
    while b.tasks:
        await asyncio.sleep(0.01)

    assert_story(
        b.story(fut1.key),
        [
            (fut1.key, "executing", "released", "cancelled", {}),
            (fut1.key, "cancelled", "error", "error", {fut1.key: "released"}),
        ],
    )


@gen_cluster(client=True, nthreads=[("", 1, {"resources": {"A": 1}})])
async def test_cancelled_error_with_resources(c, s, a):
    executing = Event()
    lock_executing = Lock()
    await lock_executing.acquire()

    def block_execution(event, lock):
        event.set()
        with lock:
            raise RuntimeError()

    fut1 = c.submit(
        block_execution,
        executing,
        lock_executing,
        resources={"A": 1},
        key="fut1",
    )

    await executing.wait()
    fut2 = c.submit(inc, 1, resources={"A": 1}, key="fut2")

    while fut2.key not in a.tasks:
        await asyncio.sleep(0.01)

    fut1.release()
    while a.tasks[fut1.key].state == "executing":
        await asyncio.sleep(0.01)
    await lock_executing.release()

    assert await fut2 == 2

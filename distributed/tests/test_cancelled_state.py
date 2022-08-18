from __future__ import annotations

import asyncio

import pytest

import distributed
from distributed import Event, Lock, Worker
from distributed.client import wait
from distributed.utils_test import (
    BlockedExecute,
    BlockedGatherDep,
    BlockedGetData,
    _LockedCommPool,
    assert_story,
    gen_cluster,
    inc,
    lock_inc,
    slowinc,
    wait_for_state,
    wait_for_stimulus,
)
from distributed.worker_state_machine import (
    ComputeTaskEvent,
    Execute,
    ExecuteFailureEvent,
    ExecuteSuccessEvent,
    FreeKeysEvent,
    GatherDep,
    GatherDepNetworkFailureEvent,
    GatherDepSuccessEvent,
    TaskFinishedMsg,
)


async def wait_for_cancelled(key, dask_worker):
    while key in dask_worker.state.tasks:
        if dask_worker.state.tasks[key].state == "cancelled":
            return
        await asyncio.sleep(0.005)
    assert False


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_abort_execution_release(c, s, a):
    lock = Lock()
    async with lock:
        fut = c.submit(lock_inc, 1, lock=lock)
        await wait_for_state(fut.key, "executing", a)
        fut.release()
        await wait_for_cancelled(fut.key, a)


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_abort_execution_reschedule(c, s, a):
    lock = Lock()
    async with lock:
        fut = c.submit(lock_inc, 1, lock=lock)
        await wait_for_state(fut.key, "executing", a)
        fut.release()
        await wait_for_cancelled(fut.key, a)
        fut = c.submit(lock_inc, 1, lock=lock)
    await fut


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_abort_execution_add_as_dependency(c, s, a):
    lock = Lock()
    async with lock:
        fut = c.submit(lock_inc, 1, lock=lock)
        await wait_for_state(fut.key, "executing", a)
        fut.release()
        await wait_for_cancelled(fut.key, a)

        fut = c.submit(lock_inc, 1, lock=lock)
        fut = c.submit(inc, fut)
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
    while "f1" in a.state.tasks:
        await asyncio.sleep(0.01)

    assert_story(
        a.state.story("f1"),
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
    assert any("receive-dep-failed" in msg for msg in b.state.log)


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
        await wait(fut1)
        await block_compute.acquire()
        fut2 = c.submit(inc, fut1, workers=[b.address], key="fut2")

        await enter_get_data.wait()

        # Close in scheduler to ensure we transition and reschedule task properly
        await s.remove_worker(a.address, stimulus_id="test", close=False)
        await wait_for_stimulus(ComputeTaskEvent, b, key=fut1.key)

        block_get_data.release()
        await block_compute.release()
        assert await fut2 == 3

        b_story = b.state.story(fut1.key)
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

    f1 = c.submit(wait_and_raise, key="f1")
    await wait_for_state(f1.key, "executing", w)
    # Queue up another task to ensure this is not affected by our error handling
    f2 = c.submit(inc, 1, key="f2")
    await wait_for_state(f2.key, "ready", w)

    f1.release()
    await wait_for_state(f1.key, "cancelled", w)
    await lock.release()

    # At this point we do not fetch the result of the future since the future
    # itself would raise a cancelled exception. At this point we're concerned
    # about the worker. The task should transition over error to be eventually
    # forgotten since we no longer hold a ref.
    while f1.key in w.state.tasks:
        await asyncio.sleep(0.01)

    assert await f2 == 2
    # Everything should still be executing as usual after this
    assert await c.submit(sum, c.map(inc, range(10))) == sum(map(inc, range(10)))

    # Everything above this line should be generically true, regardless of
    # refactoring. Below verifies some implementation specific test assumptions

    assert_story(
        w.state.story(f1.key),
        [
            (f1.key, "executing", "released", "cancelled", {}),
            (
                f1.key,
                "cancelled",
                "error",
                "error",
                {f2.key: "executing", f1.key: "released"},
            ),
            (f1.key, "error", "released", "released", {f1.key: "forgotten"}),
            (f1.key, "released", "forgotten", "forgotten", {}),
        ],
    )


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
        while fut1.key in b.state.tasks:
            await asyncio.sleep(0.01)
        a.block_get_data = False
        # Everything should still be executing as usual after this
        assert await c.submit(sum, c.map(inc, range(10))) == sum(map(inc, range(10)))


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_in_flight_lost_after_resumed(c, s, b):
    lock_executing = Lock()

    def block_execution(lock):
        lock.acquire()
        return 1

    async with BlockedGetData(s.address) as a:
        fut1 = c.submit(
            block_execution,
            lock_executing,
            workers=[a.address],
            key="fut1",
        )
        # Ensure fut1 is in memory but block any further execution afterwards to
        # ensure we control when the recomputation happens
        await wait(fut1)
        fut2 = c.submit(inc, fut1, workers=[b.address], key="fut2")

        # This ensures that B already fetches the task, i.e. after this the task
        # is guaranteed to be in flight
        await a.in_get_data.wait()
        assert fut1.key in b.state.tasks
        assert b.state.tasks[fut1.key].state == "flight"

        s.set_restrictions({fut1.key: [a.address, b.address]})
        # It is removed, i.e. get_data is guaranteed to fail and f1 is scheduled
        # to be recomputed on B
        await s.remove_worker(a.address, stimulus_id="foo", close=False, safe=True)

        while not b.state.tasks[fut1.key].state == "resumed":
            await asyncio.sleep(0.01)

        fut1.release()
        fut2.release()

        while not b.state.tasks[fut1.key].state == "cancelled":
            await asyncio.sleep(0.01)

        a.block_get_data.set()
        while b.state.tasks:
            await asyncio.sleep(0.01)

    assert_story(
        b.state.story(fut1.key),
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
    assert b.state.tasks[fut1.key].state == "executing"
    fut1.release()
    while b.state.tasks[fut1.key].state == "executing":
        await asyncio.sleep(0.01)
    await lock_executing.release()
    while b.state.tasks:
        await asyncio.sleep(0.01)

    assert_story(
        b.state.story(fut1.key),
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

    while fut2.key not in a.state.tasks:
        await asyncio.sleep(0.01)

    fut1.release()
    while a.state.tasks[fut1.key].state == "executing":
        await asyncio.sleep(0.01)
    await lock_executing.release()

    assert await fut2 == 2


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_cancelled_resumed_after_flight_with_dependencies(c, s, a):
    """A task is in flight from b to a.
    While a is waiting, b dies. The scheduler notices before a and reschedules the
    task on a itself (as the only surviving replica was just lost).
    Test that the worker eventually computes the task.

    See https://github.com/dask/distributed/pull/6327#discussion_r872231090
    See test_cancelled_resumed_after_flight_with_dependencies_workerstate below.
    """
    async with await BlockedGetData(s.address) as b:
        x = c.submit(inc, 1, key="x", workers=[b.address], allow_other_workers=True)
        y = c.submit(inc, x, key="y", workers=[a.address])
        await b.in_get_data.wait()

        # Make b dead to s, but not to a
        await s.remove_worker(b.address, stimulus_id="stim-id")

        # Wait for the scheduler to reschedule x on a.
        # We want the comms from the scheduler to reach a before b closes the RPC
        # channel, causing a.gather_dep() to raise OSError.
        await wait_for_stimulus(ComputeTaskEvent, a, key="x")

    # b closed; a.gather_dep() fails. Note that, in the current implementation, x won't
    # be recomputed on a until this happens.
    assert await y == 3


def test_cancelled_resumed_after_flight_with_dependencies_workerstate(ws):
    """Same as test_cancelled_resumed_after_flight_with_dependencies, but testing the
    WorkerState in isolation
    """
    ws2 = "127.0.0.1:2"
    instructions = ws.handle_stimulus(
        # Create task x and put it in flight from ws2
        ComputeTaskEvent.dummy("y", who_has={"x": [ws2]}, stimulus_id="s1"),
        # The scheduler realises that ws2 is unresponsive, although ws doesn't know yet.
        # Having lost the last surviving replica of x, the scheduler cancels all of its
        # dependents. This also cancels x.
        FreeKeysEvent(keys=["y"], stimulus_id="s2"),
        # The scheduler reschedules x on another worker, which just happens to be one
        # that was previously fetching it. This does not generate an Execute
        # instruction, because the GatherDep instruction isn't complete yet.
        ComputeTaskEvent.dummy("x", stimulus_id="s3"),
        # After ~30s, the TCP socket with ws2 finally times out and collapses.
        # This triggers the Execute instruction.
        GatherDepNetworkFailureEvent(worker=ws2, total_nbytes=1, stimulus_id="s4"),
    )
    assert instructions == [
        GatherDep(worker=ws2, to_gather={"x"}, total_nbytes=1, stimulus_id="s1"),
        Execute(key="x", stimulus_id="s4"),  # Note the stimulus_id!
    ]
    assert ws.tasks["x"].state == "executing"


@pytest.mark.parametrize("wait_for_processing", [True, False])
@pytest.mark.parametrize("raise_error", [True, False])
@gen_cluster(client=True)
async def test_resumed_cancelled_handle_compute(
    c, s, a, b, raise_error, wait_for_processing
):
    """
    Given the history of a task
    executing -> cancelled -> resumed(fetch)

    Setup
    -----
    A task is running on the threadpool while a client is cancelling the
    computations. In this case we cannot cancel the task on the threadpool but
    need to transition the worker task to cancelled instead. While the task on
    the threadpool is still running, the client resubmits the graph. This is
    relatively common in interactive workflows in which a user cancels a
    computation and resubmits shortly after.

    This resubmission can decide to distribute tasks onto different workers than the initial submission

    Parameters
    ----------
    raise_error:
        Both successful and erred results should be properly handled.

    wait_for_processing:
        Is the scheduler properly handling the task-done message of the worker
        even if the task is not, yet, transitioned to processing again?

    Expectation
    -----------
    A `handle_compute` should properly restore state `executing` even after the
    task has transitioned through `cancelled` and `resumed(fetch)`.


    See also
    --------
    test_worker_state_machine.py::test_executing_cancelled_fetch_executing
    for minimal example
    """

    lock_compute = Lock()
    await lock_compute.acquire()
    enter_compute = Event()

    def block(x, lock, enter_event):
        enter_event.set()
        with lock:
            if raise_error:
                raise RuntimeError("test error")
            else:
                return x + 1

    f1 = c.submit(inc, 1, key="f1", workers=[a.address])
    f2 = c.submit(inc, f1, key="f2", workers=[a.address])
    f3 = c.submit(
        block,
        f2,
        lock=lock_compute,
        enter_event=enter_compute,
        key="f3",
        workers=[b.address],
    )

    f4 = c.submit(sum, [f1, f3], key="f4", workers=[b.address])

    await enter_compute.wait()

    async def release_all_futures():
        futs = [f1, f2, f3, f4]
        for fut in futs:
            fut.release()

        while any(fut.key in s.tasks for fut in futs):
            await asyncio.sleep(0.05)

    await release_all_futures()
    await wait_for_state(f3.key, "cancelled", b)

    # At the second iteration, the scheduler chooses to distribute the tasks to
    # different workers
    # Particularly for our "stuck" task f3 that is still cancelled(executing) on
    # worker B this will transition the task to a resumed(fetch) since it is no
    # longer supposed to be computed on B. However, B will still require the
    # data of f3 and is supposed to fetch it in case the compute fails
    f1 = c.submit(inc, 1, key="f1", workers=[a.address])
    f2 = c.submit(inc, f1, key="f2", workers=[a.address])
    f3 = c.submit(inc, f2, key="f3", workers=[a.address])
    f4 = c.submit(sum, [f1, f3], key="f4", workers=[b.address])

    await wait_for_state(f3.key, "resumed", b)
    await release_all_futures()

    # We're again cancelling, forcing f3 to transition back to executing from a
    # resumed(fetch) state

    f1 = c.submit(inc, 1, key="f1", workers=[a.address])
    f2 = c.submit(inc, f1, key="f2", workers=[a.address])
    f3 = c.submit(inc, f2, key="f3", workers=[b.address])
    f4 = c.submit(sum, [f1, f3], key="f4", workers=[b.address])

    if wait_for_processing:
        await wait_for_state(f3.key, "processing", s)

    await lock_compute.release()

    if not raise_error:
        assert await f4 == 4 + 2

        assert_story(
            b.state.story(f3.key),
            expect=[
                (f3.key, "ready", "executing", "executing", {}),
                (f3.key, "executing", "released", "cancelled", {}),
                (f3.key, "cancelled", "fetch", "resumed", {}),
                (f3.key, "resumed", "memory", "memory", {}),
            ],
        )

    else:
        with pytest.raises(RuntimeError, match="test error"):
            await f3

        assert_story(
            b.state.story(f3.key),
            expect=[
                (f3.key, "ready", "executing", "executing", {}),
                (f3.key, "executing", "released", "cancelled", {}),
                (f3.key, "cancelled", "fetch", "resumed", {}),
                (f3.key, "resumed", "error", "error", {}),
            ],
        )


@pytest.mark.parametrize("intermediate_state", ["resumed", "cancelled"])
@pytest.mark.parametrize("close_worker", [False, True])
@gen_cluster(client=True, config={"distributed.comm.timeouts.connect": "500ms"})
async def test_deadlock_cancelled_after_inflight_before_gather_from_worker(
    c, s, a, x, intermediate_state, close_worker
):
    """If a task was transitioned to in-flight, the gather_dep coroutine was scheduled
    but a cancel request came in before gather_data_from_worker was issued. This might
    corrupt the state machine if the cancelled key is not properly handled.

    See also
    --------
    test_workerstate_deadlock_cancelled_after_inflight_before_gather_from_worker
    """
    fut1 = c.submit(slowinc, 1, workers=[a.address], key="f1")
    fut1B = c.submit(slowinc, 2, workers=[x.address], key="f1B")
    fut2 = c.submit(sum, [fut1, fut1B], workers=[x.address], key="f2")
    await fut2

    async with BlockedGatherDep(s.address, name="b") as b:
        fut3 = c.submit(inc, fut2, workers=[b.address], key="f3")

        await wait_for_state(fut2.key, "flight", b)

        s.set_restrictions(worker={fut1B.key: a.address, fut2.key: b.address})

        await b.in_gather_dep.wait()

        await s.remove_worker(
            address=x.address,
            safe=True,
            close=close_worker,
            stimulus_id="remove-worker",
        )

        await wait_for_state(fut2.key, intermediate_state, b, interval=0)

        b.block_gather_dep.set()
        await fut3


def test_workerstate_executing_to_executing(ws_with_running_task):
    """Test state loops:

    - executing -> cancelled -> executing
    - executing -> long-running -> cancelled -> long-running

    Test that the task immediately reverts to its original state.
    """
    ws = ws_with_running_task
    ts = ws.tasks["x"]
    prev_state = ts.state

    instructions = ws.handle_stimulus(
        FreeKeysEvent(keys=["x"], stimulus_id="s1"),
        ComputeTaskEvent.dummy("x", resource_restrictions={"R": 1}, stimulus_id="s2"),
    )
    assert not instructions
    assert ws.tasks["x"] is ts
    assert ts.state == prev_state


def test_workerstate_flight_to_flight(ws):
    """Test state loop:

    flight -> cancelled -> flight

    Test that the task immediately reverts to its original state.
    """
    ws2 = "127.0.0.1:2"
    instructions = ws.handle_stimulus(
        ComputeTaskEvent.dummy("y", who_has={"x": [ws2]}, stimulus_id="s1"),
        FreeKeysEvent(keys=["y", "x"], stimulus_id="s2"),
        ComputeTaskEvent.dummy("z", who_has={"x": [ws2]}, stimulus_id="s3"),
    )
    assert instructions == [
        GatherDep(worker=ws2, to_gather={"x"}, total_nbytes=1, stimulus_id="s1")
    ]
    assert ws.tasks["x"].state == "flight"


def test_workerstate_executing_skips_fetch_on_success(ws_with_running_task):
    """Test state loops:

    - executing -> cancelled -> resumed(fetch) -> memory
    - executing -> long-running -> cancelled -> resumed(fetch) -> memory

    The task execution later terminates successfully.
    Test that the task is never fetched and that dependents are unblocked.

    See also: test_workerstate_executing_failure_to_fetch
    """
    ws = ws_with_running_task
    ws2 = "127.0.0.1:2"
    instructions = ws.handle_stimulus(
        FreeKeysEvent(keys=["x"], stimulus_id="s1"),
        ComputeTaskEvent.dummy("y", who_has={"x": [ws2]}, stimulus_id="s2"),
        ExecuteSuccessEvent.dummy("x", 123, stimulus_id="s3"),
    )
    assert instructions == [
        TaskFinishedMsg.match(key="x", stimulus_id="s3"),
        Execute(key="y", stimulus_id="s3"),
    ]
    assert ws.tasks["x"].state == "memory"
    assert ws.data["x"] == 123


@pytest.mark.xfail(reason="distributed#6689")
def test_workerstate_executing_failure_to_fetch(ws_with_running_task):
    """Test state loops:

    - executing -> cancelled -> resumed(fetch)
    - executing -> long-running -> cancelled -> resumed(fetch)

    The task execution later terminates with a failure.
    This is an edge case interaction between work stealing and a task that does not
    deterministically succeed or fail when run multiple times or on different workers.

    Test that the task is fetched from the other worker. This is to avoid having to deal
    with cancelling the dependent, which would require interaction with the scheduler
    and increase the complexity of the use case.

    See also: test_workerstate_executing_success_to_fetch
    """
    ws = ws_with_running_task
    ws2 = "127.0.0.1:2"
    instructions = ws.handle_stimulus(
        FreeKeysEvent(keys=["x"], stimulus_id="s1"),
        ComputeTaskEvent.dummy("y", who_has={"x": [ws2]}, stimulus_id="s2"),
        ExecuteFailureEvent.dummy("x", stimulus_id="s3"),
    )
    assert instructions == [
        GatherDep(worker=ws2, to_gather={"x"}, total_nbytes=1, stimulus_id="s3")
    ]
    assert ws.tasks["x"].state == "flight"


def test_workerstate_flight_skips_executing_on_success(ws):
    """Test state loop

    flight -> cancelled -> resumed(waiting) -> memory

    gather_dep later terminates successfully.
    Test that the task is not executed and is in memory.
    """
    ws2 = "127.0.0.1:2"
    instructions = ws.handle_stimulus(
        ComputeTaskEvent.dummy("y", who_has={"x": [ws2]}, stimulus_id="s1"),
        FreeKeysEvent(keys=["y", "x"], stimulus_id="s2"),
        ComputeTaskEvent.dummy("x", stimulus_id="s3"),
        GatherDepSuccessEvent(
            worker=ws2, total_nbytes=1, data={"x": 123}, stimulus_id="s4"
        ),
    )
    assert instructions == [
        GatherDep(worker=ws2, to_gather={"x"}, total_nbytes=1, stimulus_id="s1"),
        TaskFinishedMsg.match(key="x", stimulus_id="s4"),
    ]
    assert ws.tasks["x"].state == "memory"
    assert ws.data["x"] == 123


@pytest.mark.parametrize("block_queue", [False, True])
def test_workerstate_flight_failure_to_executing(ws, block_queue):
    """Test state loop

    flight -> cancelled -> resumed(waiting)

    gather_dep later terminates with a failure.
    Test that the task is executed.
    """
    ws2 = "127.0.0.1:2"
    instructions = ws.handle_stimulus(
        ComputeTaskEvent.dummy("y", who_has={"x": [ws2]}, stimulus_id="s1"),
        FreeKeysEvent(keys=["y", "x"], stimulus_id="s2"),
        ComputeTaskEvent.dummy("x", stimulus_id="s3"),
    )
    assert instructions == [
        GatherDep(stimulus_id="s1", worker=ws2, to_gather={"x"}, total_nbytes=1),
    ]
    assert ws.tasks["x"].state == "resumed"

    if block_queue:
        instructions = ws.handle_stimulus(
            ComputeTaskEvent.dummy(key="z", stimulus_id="s4"),
            GatherDepNetworkFailureEvent(worker=ws2, total_nbytes=1, stimulus_id="s5"),
            ExecuteSuccessEvent.dummy(key="z", stimulus_id="s6"),
        )
        assert instructions == [
            Execute(key="z", stimulus_id="s4"),
            TaskFinishedMsg.match(key="z", stimulus_id="s6"),
            Execute(key="x", stimulus_id="s6"),
        ]
        assert_story(
            ws.story("x"),
            [
                ("x", "resumed", "fetch", "released", {"x": "waiting"}),
                ("x", "released", "waiting", "waiting", {"x": "ready"}),
                ("x", "waiting", "ready", "ready", {}),
            ],
        )

    else:
        instructions = ws.handle_stimulus(
            GatherDepNetworkFailureEvent(worker=ws2, total_nbytes=1, stimulus_id="s5"),
        )
        assert instructions == [
            Execute(key="x", stimulus_id="s5"),
        ]
        assert_story(
            ws.story("x"),
            [
                ("x", "resumed", "fetch", "released", {"x": "waiting"}),
                ("x", "released", "waiting", "waiting", {"x": "ready"}),
                ("x", "waiting", "ready", "ready", {"x": "executing"}),
            ],
        )

    assert ws.tasks["x"].state == "executing"


def test_workerstate_resumed_fetch_to_executing(ws_with_running_task):
    """Test state loops:

    - executing -> cancelled -> resumed(fetch) -> cancelled -> executing
    - executing -> long-running -> cancelled -> resumed(fetch) -> long-running

    See also: test_workerstate_resumed_waiting_to_flight
    """
    ws = ws_with_running_task
    ws2 = "127.0.0.1:2"
    prev_state = ws.tasks["x"].state

    instructions = ws.handle_stimulus(
        FreeKeysEvent(keys=["x"], stimulus_id="s1"),
        ComputeTaskEvent.dummy("y", who_has={"x": [ws2]}, stimulus_id="s2"),
        FreeKeysEvent(keys=["y", "x"], stimulus_id="s3"),
        ComputeTaskEvent.dummy("x", resource_restrictions={"R": 1}, stimulus_id="s4"),
    )
    assert not instructions
    assert ws.tasks["x"].state == prev_state


def test_workerstate_resumed_waiting_to_flight(ws):
    """Test state loop:

    - flight -> cancelled -> resumed(waiting) -> cancelled -> flight

    See also: test_workerstate_resumed_fetch_to_executing
    """
    ws2 = "127.0.0.1:2"
    instructions = ws.handle_stimulus(
        ComputeTaskEvent.dummy("y", who_has={"x": [ws2]}, stimulus_id="s1"),
        FreeKeysEvent(keys=["y", "x"], stimulus_id="s2"),
        ComputeTaskEvent.dummy("x", resource_restrictions={"R": 1}, stimulus_id="s3"),
        FreeKeysEvent(keys=["x"], stimulus_id="s4"),
        ComputeTaskEvent.dummy("y", who_has={"x": [ws2]}, stimulus_id="s5"),
    )
    assert instructions == [
        GatherDep(worker=ws2, to_gather={"x"}, stimulus_id="s1", total_nbytes=1),
    ]
    assert ws.tasks["x"].state == "flight"


@pytest.mark.parametrize("critical_section", ["execute", "deserialize_task"])
@pytest.mark.parametrize("resume_inside_critical_section", [False, True])
@pytest.mark.parametrize("resumed_status", ["executing", "resumed"])
@gen_cluster(client=True, nthreads=[("", 1)])
async def test_execute_preamble_early_cancel(
    c, s, b, critical_section, resume_inside_critical_section, resumed_status
):
    """Test multiple race conditions in the preamble of Worker.execute(), which used to
    cause a task to remain permanently in resumed state or to crash the worker through
    `fail_hard` in case of very tight timings when resuming a task.

    See also
    --------
    https://github.com/dask/distributed/issues/6869
    https://github.com/dask/dask/issues/9330
    test_worker.py::test_execute_preamble_abort_retirement
    """
    async with BlockedExecute(s.address, validate=True) as a:
        if critical_section == "execute":
            in_ev = a.in_execute
            block_ev = a.block_execute
            a.block_deserialize_task.set()
        else:
            assert critical_section == "deserialize_task"
            in_ev = a.in_deserialize_task
            block_ev = a.block_deserialize_task
            a.block_execute.set()

        async def resume():
            if resumed_status == "executing":
                x = c.submit(inc, 1, key="x", workers=[a.address])
                await wait_for_state("x", "executing", a)
                return x, 2
            else:
                assert resumed_status == "resumed"
                x = c.submit(inc, 1, key="x", workers=[b.address])
                y = c.submit(inc, x, key="y", workers=[a.address])
                await wait_for_state("x", "resumed", a)
                return y, 3

        x = c.submit(inc, 1, key="x", workers=[a.address])
        await in_ev.wait()

        x.release()
        await wait_for_state("x", "cancelled", a)

        if resume_inside_critical_section:
            fut, expect = await resume()

        # Unblock Worker.execute. At the moment of writing this test, the method
        # would detect the cancelled status and perform an early exit.
        block_ev.set()
        await a.in_execute_exit.wait()

        if not resume_inside_critical_section:
            fut, expect = await resume()

        # Finally let the done_callback of Worker.execute run
        a.block_execute_exit.set()

        # Test that x does not get stuck.
        assert await fut == expect

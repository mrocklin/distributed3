from __future__ import annotations

import asyncio
import os
import pathlib
import signal
import socket
import subprocess
import sys
import textwrap
import threading
from contextlib import contextmanager
from multiprocessing.synchronize import Barrier
from time import sleep
from unittest import mock

import pytest
import yaml
from tornado import gen

import dask.config
from dask.sizeof import sizeof

from distributed import Client, Event, Nanny, Scheduler, Worker, config, default_client
from distributed.batched import BatchedSend
from distributed.comm.core import connect
from distributed.compatibility import WINDOWS
from distributed.core import ConnectionPool, Server, Status
from distributed.metrics import time
from distributed.tests.test_batched import EchoServer
from distributed.utils import get_mp_context
from distributed.utils_test import (
    SizeOf,
    _LockedCommPool,
    _UnhashableCallable,
    assert_story,
    captured_logger,
    check_process_leak,
    check_thread_leak,
    cluster,
    dump_cluster_state,
    freeze_batched_send,
    gen_cluster,
    gen_nbytes,
    gen_test,
    inc,
    new_config,
    popen,
    raises_with_cause,
    tls_only_security,
    wait_for_state,
    wait_for_stimulus,
)
from distributed.worker import fail_hard
from distributed.worker_state_machine import (
    ComputeTaskEvent,
    InvalidTaskState,
    InvalidTransition,
    PauseEvent,
    WorkerState,
)


def test_bare_cluster(loop):
    with cluster(nworkers=10) as (s, _):
        pass


def test_cluster(cleanup):
    async def identity():
        async with ConnectionPool() as rpc:
            async with rpc(s["address"]) as scheduler_rpc:
                return await scheduler_rpc.identity()

    with cluster() as (s, [a, b]):
        ident = asyncio.run(identity())
        assert ident["type"] == "Scheduler"
        assert len(ident["workers"]) == 2


@gen_cluster(client=True)
async def test_gen_cluster(c, s, a, b):
    assert isinstance(c, Client)
    assert isinstance(s, Scheduler)
    for w in [a, b]:
        assert isinstance(w, Worker)
    assert set(s.workers) == {w.address for w in [a, b]}
    assert await c.submit(lambda: 123) == 123


@gen_cluster(client=True)
async def test_gen_cluster_pytest_fixture(c, s, a, b, tmp_path):
    assert isinstance(tmp_path, pathlib.Path)
    assert isinstance(c, Client)
    assert isinstance(s, Scheduler)
    for w in [a, b]:
        assert isinstance(w, Worker)


@pytest.mark.parametrize("foo", [True])
@gen_cluster(client=True)
async def test_gen_cluster_parametrized(c, s, a, b, foo):
    assert foo is True
    assert isinstance(c, Client)
    assert isinstance(s, Scheduler)
    for w in [a, b]:
        assert isinstance(w, Worker)


@pytest.mark.parametrize("foo", [True])
@pytest.mark.parametrize("bar", ["a", "b"])
@gen_cluster(client=True)
async def test_gen_cluster_multi_parametrized(c, s, a, b, foo, bar):
    assert foo is True
    assert bar in ("a", "b")
    assert isinstance(c, Client)
    assert isinstance(s, Scheduler)
    for w in [a, b]:
        assert isinstance(w, Worker)


@pytest.mark.parametrize("foo", [True])
@gen_cluster(client=True)
async def test_gen_cluster_parametrized_variadic_workers(c, s, *workers, foo):
    assert foo is True
    assert isinstance(c, Client)
    assert isinstance(s, Scheduler)
    for w in workers:
        assert isinstance(w, Worker)


@gen_cluster(
    client=True,
    Worker=Nanny,
    config={"distributed.comm.timeouts.connect": "1s", "new.config.value": "foo"},
)
async def test_gen_cluster_set_config_nanny(c, s, a, b):
    def assert_config():
        assert dask.config.get("distributed.comm.timeouts.connect") == "1s"
        assert dask.config.get("new.config.value") == "foo"

    await c.run(assert_config)
    await c.run_on_scheduler(assert_config)


@pytest.mark.skip(reason="This hangs on travis")
def test_gen_cluster_cleans_up_client(loop):
    import dask.context

    assert not dask.config.get("get", None)

    @gen_cluster(client=True)
    async def f(c, s, a, b):
        assert dask.config.get("get", None)
        await c.submit(inc, 1)

    f()

    assert not dask.config.get("get", None)


@gen_cluster()
async def test_gen_cluster_without_client(s, a, b):
    assert isinstance(s, Scheduler)
    for w in [a, b]:
        assert isinstance(w, Worker)
    assert set(s.workers) == {w.address for w in [a, b]}

    async with Client(s.address, asynchronous=True) as c:
        future = c.submit(lambda x: x + 1, 1)
        result = await future
        assert result == 2


@gen_cluster(
    client=True,
    scheduler="tls://127.0.0.1",
    nthreads=[("tls://127.0.0.1", 1), ("tls://127.0.0.1", 2)],
    security=tls_only_security(),
)
async def test_gen_cluster_tls(e, s, a, b):
    assert isinstance(e, Client)
    assert isinstance(s, Scheduler)
    assert s.address.startswith("tls://")
    for w in [a, b]:
        assert isinstance(w, Worker)
        assert w.address.startswith("tls://")
    assert set(s.workers) == {w.address for w in [a, b]}


@pytest.mark.xfail(
    reason="Test should always fail to ensure the body of the test function was run",
    strict=True,
)
@gen_test()
async def test_gen_test():
    await asyncio.sleep(0.01)
    assert False


@pytest.mark.xfail(
    reason="Test should always fail to ensure the body of the test function was run",
    strict=True,
)
@gen_test()
def test_gen_test_legacy_implicit():
    yield asyncio.sleep(0.01)
    assert False


@pytest.mark.xfail(
    reason="Test should always fail to ensure the body of the test function was run",
    strict=True,
)
@gen_test()
@gen.coroutine
def test_gen_test_legacy_explicit():
    yield asyncio.sleep(0.01)
    assert False


@pytest.mark.parametrize("foo", [True])
@gen_test()
async def test_gen_test_parametrized(foo):
    assert foo is True


@pytest.mark.parametrize("foo", [True])
@pytest.mark.parametrize("bar", [False])
@gen_test()
async def test_gen_test_double_parametrized(foo, bar):
    assert foo is True
    assert bar is False


@gen_test()
async def test_gen_test_pytest_fixture(tmp_path):
    assert isinstance(tmp_path, pathlib.Path)


@contextmanager
def _listen(delay=0):
    serv = socket.socket()
    serv.bind(("127.0.0.1", 0))
    e = threading.Event()

    def do_listen():
        e.set()
        sleep(delay)
        serv.listen(5)
        ret = serv.accept()
        if ret is not None:
            cli, _ = ret
            cli.close()
        serv.close()

    t = threading.Thread(target=do_listen)
    t.daemon = True
    t.start()
    try:
        e.wait()
        sleep(0.01)
        yield serv
    finally:
        t.join(5.0)


def test_new_config():
    c = config.copy()
    with new_config({"xyzzy": 5}):
        assert config["xyzzy"] == 5

    assert config == c
    assert "xyzzy" not in config


def test_lingering_client():
    @gen_cluster()
    async def f(s, a, b):
        # TODO: force async-with here?
        # see https://github.com/dask/distributed/issues/6616
        await Client(s.address, asynchronous=True)

    f()

    with pytest.raises(ValueError):
        default_client()


def test_lingering_client_2(loop):
    # TODO: assert where the client went
    with cluster() as (s, [a, b]):
        client = Client(s["address"], loop=loop)


def test_tls_cluster(tls_client):
    assert tls_client.submit(lambda x: x + 1, 10).result() == 11
    assert tls_client.security


@gen_test()
async def test_tls_scheduler(security):
    async with Scheduler(
        security=security, host="localhost", dashboard_address=":0"
    ) as s:
        assert s.address.startswith("tls")


def test__UnhashableCallable():
    func = _UnhashableCallable()
    assert func(1) == 2
    with pytest.raises(TypeError, match="unhashable"):
        hash(func)


class MyServer(Server):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handlers["ping"] = self.pong
        self.counter = 0

    def pong(self, comm):
        self.counter += 1
        return "pong"


@gen_test()
async def test_locked_comm_drop_in_replacement(loop):

    async with MyServer({}) as a, await MyServer({}) as b:
        await a.listen(0)

        read_event = asyncio.Event()
        read_event.set()
        read_queue = asyncio.Queue()
        original_pool = a.rpc
        a.rpc = _LockedCommPool(
            original_pool, read_event=read_event, read_queue=read_queue
        )

        await b.listen(0)
        # Event is set, the pool works like an ordinary pool
        res = await a.rpc(b.address).ping()
        assert await read_queue.get() == (b.address, "pong")
        assert res == "pong"
        assert b.counter == 1

        read_event.clear()
        # Can also be used without a lock to intercept network traffic
        a.rpc = _LockedCommPool(original_pool, read_queue=read_queue)
        a.rpc.remove(b.address)
        res = await a.rpc(b.address).ping()
        assert await read_queue.get() == (b.address, "pong")


@gen_test()
async def test_locked_comm_intercept_read(loop):

    async with MyServer({}) as a, MyServer({}) as b:
        await a.listen(0)
        await b.listen(0)

        read_event = asyncio.Event()
        read_queue = asyncio.Queue()
        a.rpc = _LockedCommPool(a.rpc, read_event=read_event, read_queue=read_queue)

        async def ping_pong():
            return await a.rpc(b.address).ping()

        fut = asyncio.create_task(ping_pong())

        # We didn't block the write but merely the read. The remove should have
        # received the message and responded already
        while not b.counter:
            await asyncio.sleep(0.001)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(fut), 0.01)

        assert await read_queue.get() == (b.address, "pong")
        read_event.set()
        assert await fut == "pong"


@gen_test()
async def test_locked_comm_intercept_write(loop):

    async with MyServer({}) as a, MyServer({}) as b:
        await a.listen(0)
        await b.listen(0)

        write_event = asyncio.Event()
        write_queue = asyncio.Queue()
        a.rpc = _LockedCommPool(a.rpc, write_event=write_event, write_queue=write_queue)

        async def ping_pong():
            return await a.rpc(b.address).ping()

        fut = asyncio.create_task(ping_pong())

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(fut), 0.01)
        # Write was blocked. The remote hasn't received the message, yet
        assert b.counter == 0
        assert await write_queue.get() == (b.address, {"op": "ping", "reply": True})
        write_event.set()
        assert await fut == "pong"


@pytest.mark.slow()
def test_dump_cluster_state_timeout(tmp_path):
    sleep_time = 30

    async def inner_test(c, s, a, b):
        await asyncio.sleep(sleep_time)

    # This timeout includes cluster startup and teardown which sometimes can
    # take a significant amount of time. For this particular test we would like
    # to keep the _test timeout_ small because we intend to trigger it but the
    # overall timeout large.
    test = gen_cluster(client=True, timeout=5, cluster_dump_directory=tmp_path)(
        inner_test
    )
    try:
        with pytest.raises(asyncio.TimeoutError) as exc:
            test()
        assert "inner_test" in str(exc)
        assert "await asyncio.sleep(sleep_time)" in str(exc)
    except gen.TimeoutError:
        pytest.xfail("Cluster startup or teardown took too long")

    _, dirs, files = next(os.walk(tmp_path))
    assert not dirs
    assert files == [inner_test.__name__ + ".yaml"]
    import yaml

    with open(tmp_path / files[0], "rb") as fd:
        state = yaml.load(fd, Loader=yaml.Loader)

    assert "scheduler" in state
    assert "workers" in state


def test_assert_story():
    now = time()
    story = [
        ("foo", "id1", now - 600),
        ("bar", "id2", now),
        ("baz", {1: 2}, "id2", now),
    ]
    # strict=False
    assert_story(story, [("foo",), ("bar",), ("baz", {1: 2})])
    assert_story(story, [])
    assert_story(story, [("foo",)])
    assert_story(story, [("foo",), ("bar",)])
    assert_story(story, [("baz", lambda d: d[1] == 2)])
    with pytest.raises(AssertionError):
        assert_story(story, [("foo", "nomatch")])
    with pytest.raises(AssertionError):
        assert_story(story, [("baz",)])
    with pytest.raises(AssertionError):
        assert_story(story, [("baz", {1: 3})])
    with pytest.raises(AssertionError):
        assert_story(story, [("foo",), ("bar",), ("baz", "extra"), ("+1",)])
    with pytest.raises(AssertionError):
        assert_story(story, [("baz", lambda d: d[1] == 3)])
    with pytest.raises(KeyError):  # Faulty lambda
        assert_story(story, [("baz", lambda d: d[2] == 1)])
    assert_story([], [])
    assert_story([("foo", "id1", now)], [("foo",)])
    with pytest.raises(AssertionError):
        assert_story([], [("foo",)])

    # strict=True
    assert_story([], [], strict=True)
    assert_story([("foo", "id1", now)], [("foo",)])
    assert_story(story, [("foo",), ("bar",), ("baz", {1: 2})], strict=True)
    with pytest.raises(AssertionError):
        assert_story(story, [("foo",), ("bar",)], strict=True)
    with pytest.raises(AssertionError):
        assert_story(story, [("foo",), ("baz", {1: 2})], strict=True)
    with pytest.raises(AssertionError):
        assert_story(story, [], strict=True)


@pytest.mark.parametrize(
    "story_factory",
    [
        pytest.param(lambda: [()], id="Missing payload, stimulus_id, ts"),
        pytest.param(lambda: [("foo",)], id="Missing (stimulus_id, ts)"),
        pytest.param(lambda: [("foo", "bar")], id="Missing ts"),
        pytest.param(lambda: [("foo", "bar", "baz")], id="ts is not a float"),
        pytest.param(lambda: [("foo", "bar", time() + 3600)], id="ts is in the future"),
        pytest.param(lambda: [("foo", "bar", time() - 7200)], id="ts is too old"),
        pytest.param(lambda: [("foo", 123, time())], id="stimulus_id is not a str"),
        pytest.param(lambda: [("foo", "", time())], id="stimulus_id is an empty str"),
        pytest.param(lambda: [("", time())], id="no payload"),
        pytest.param(
            lambda: [("foo", "id", time()), ("foo", "id", time() - 10)],
            id="timestamps out of order",
        ),
    ],
)
def test_assert_story_malformed_story(story_factory):
    # defer the calls to time() to when the test runs rather than collection
    story = story_factory()
    with pytest.raises(AssertionError, match="Malformed story event"):
        assert_story(story, [])


@pytest.mark.parametrize("strict", [True, False])
@gen_cluster(client=True, nthreads=[("", 1)])
async def test_assert_story_identity(c, s, a, strict):
    f1 = c.submit(inc, 1, key="f1")
    f2 = c.submit(inc, f1, key="f2")
    assert await f2 == 3
    scheduler_story = s.story(f2.key)
    assert scheduler_story
    worker_story = a.state.story(f2.key)
    assert worker_story
    assert_story(worker_story, worker_story, strict=strict)
    assert_story(scheduler_story, scheduler_story, strict=strict)
    with pytest.raises(AssertionError):
        assert_story(scheduler_story, worker_story, strict=strict)
    with pytest.raises(AssertionError):
        assert_story(worker_story, scheduler_story, strict=strict)


@gen_cluster()
async def test_dump_cluster_state(s, a, b, tmpdir):
    await dump_cluster_state(s, [a, b], str(tmpdir), "dump")
    with open(f"{tmpdir}/dump.yaml") as fh:
        out = yaml.safe_load(fh)

    assert out.keys() == {"scheduler", "workers", "versions"}
    assert out["workers"].keys() == {a.address, b.address}


@gen_cluster(nthreads=[])
async def test_dump_cluster_state_no_workers(s, tmpdir):
    await dump_cluster_state(s, [], str(tmpdir), "dump")
    with open(f"{tmpdir}/dump.yaml") as fh:
        out = yaml.safe_load(fh)

    assert out.keys() == {"scheduler", "workers", "versions"}
    assert out["workers"] == {}


@gen_cluster(Worker=Nanny)
async def test_dump_cluster_state_nannies(s, a, b, tmpdir):
    await dump_cluster_state(s, [a, b], str(tmpdir), "dump")
    with open(f"{tmpdir}/dump.yaml") as fh:
        out = yaml.safe_load(fh)

    assert out.keys() == {"scheduler", "workers", "versions"}
    assert out["workers"].keys() == s.workers.keys()


@gen_cluster()
async def test_dump_cluster_state_unresponsive_local_worker(s, a, b, tmpdir):
    a.stop()
    await dump_cluster_state(s, [a, b], str(tmpdir), "dump")
    with open(f"{tmpdir}/dump.yaml") as fh:
        out = yaml.safe_load(fh)

    assert out.keys() == {"scheduler", "workers", "versions"}
    assert isinstance(out["workers"][a.address], dict)
    assert isinstance(out["workers"][b.address], dict)


@pytest.mark.slow
@gen_cluster(
    client=True,
    Worker=Nanny,
    config={"distributed.comm.timeouts.connect": "600ms"},
)
async def test_dump_cluster_unresponsive_remote_worker(c, s, a, b, tmpdir):
    clog_fut = asyncio.create_task(
        c.run(lambda dask_scheduler: dask_scheduler.stop(), workers=[a.worker_address])
    )
    await asyncio.sleep(0.2)

    await dump_cluster_state(s, [a, b], str(tmpdir), "dump")
    with open(f"{tmpdir}/dump.yaml") as fh:
        out = yaml.safe_load(fh)

    assert out.keys() == {"scheduler", "workers", "versions"}
    assert isinstance(out["workers"][b.worker_address], dict)
    assert out["workers"][a.worker_address].startswith(
        "OSError('Timed out trying to connect to"
    )

    clog_fut.cancel()


# Note: WINDOWS constant doesn't work with `mypy --platform win32`
if sys.platform == "win32":
    TERM_SIGNALS = (signal.SIGTERM, signal.SIGINT)
else:
    TERM_SIGNALS = (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)


def garbage_process(
    barrier: Barrier, ignore_sigterm: bool = False, t: float = 3600
) -> None:
    if ignore_sigterm:
        for signum in TERM_SIGNALS:
            signal.signal(signum, signal.SIG_IGN)
    barrier.wait()
    sleep(t)


def test_check_process_leak():
    barrier = get_mp_context().Barrier(parties=2)
    with pytest.raises(AssertionError):
        with check_process_leak(check=True, check_timeout=0.01):
            p = get_mp_context().Process(target=garbage_process, args=(barrier,))
            p.start()
            barrier.wait()
    assert not p.is_alive()


def test_check_process_leak_slow_cleanup():
    """check_process_leak waits a bit for processes to terminate themselves"""
    barrier = get_mp_context().Barrier(parties=2)
    with check_process_leak(check=True):
        p = get_mp_context().Process(target=garbage_process, args=(barrier, False, 0.2))
        p.start()
        barrier.wait()
    assert not p.is_alive()


@pytest.mark.parametrize(
    "ignore_sigterm",
    [False, pytest.param(True, marks=pytest.mark.skipif(WINDOWS, reason="no SIGKILL"))],
)
def test_check_process_leak_pre_cleanup(ignore_sigterm):
    barrier = get_mp_context().Barrier(parties=2)
    p = get_mp_context().Process(target=garbage_process, args=(barrier, ignore_sigterm))
    p.start()
    barrier.wait()

    with check_process_leak(term_timeout=0.2):
        assert not p.is_alive()


@pytest.mark.parametrize(
    "ignore_sigterm",
    [False, pytest.param(True, marks=pytest.mark.skipif(WINDOWS, reason="no SIGKILL"))],
)
def test_check_process_leak_post_cleanup(ignore_sigterm):
    barrier = get_mp_context().Barrier(parties=2)
    with check_process_leak(check=False, term_timeout=0.2):
        p = get_mp_context().Process(
            target=garbage_process, args=(barrier, ignore_sigterm)
        )
        p.start()
        barrier.wait()
    assert not p.is_alive()


@pytest.mark.parametrize("nanny", [True, False])
def test_start_failure_worker(nanny):
    if nanny:
        ctx = raises_with_cause(RuntimeError, None, TypeError, None)
    else:
        ctx = pytest.raises(TypeError)
    with ctx:
        with cluster(nanny=nanny, worker_kwargs={"foo": "bar"}):
            return


def test_start_failure_scheduler():
    with pytest.raises(TypeError):
        with cluster(scheduler_kwargs={"foo": "bar"}):
            return


def test_invalid_transitions(capsys):
    @gen_cluster(client=True, nthreads=[("", 1)])
    async def test_log_invalid_transitions(c, s, a):
        x = c.submit(inc, 1, key="task-name")
        await x
        ts = a.state.tasks["task-name"]
        ev = PauseEvent(stimulus_id="test")
        with mock.patch.object(
            WorkerState, "_handle_event", return_value=({ts: "foo"}, [])
        ):
            with pytest.raises(InvalidTransition):
                a.handle_stimulus(ev)

        while not s.events["invalid-worker-transition"]:
            await asyncio.sleep(0.01)

    with pytest.raises(Exception) as info:
        test_log_invalid_transitions()

    assert "invalid" in str(info).lower()
    assert "worker" in str(info).lower()
    assert "transition" in str(info).lower()

    out, err = capsys.readouterr()

    assert "foo" in out + err
    assert "task-name" in out + err


def test_invalid_worker_state(capsys):
    @gen_cluster(client=True, nthreads=[("127.0.0.1", 1)])
    async def test_log_invalid_worker_task_state(c, s, a):
        x = c.submit(inc, 1, key="task-name")
        await x
        a.state.tasks[x.key].state = "released"
        with pytest.raises(InvalidTaskState):
            a.validate_state()

        while not s.events["invalid-worker-task-state"]:
            await asyncio.sleep(0.01)

    with pytest.raises(Exception) as info:
        test_log_invalid_worker_task_state()

    out, err = capsys.readouterr()

    assert "released" in out + err
    assert "task-name" in out + err


def test_raises_with_cause():
    with raises_with_cause(RuntimeError, "exception", ValueError, "cause"):
        raise RuntimeError("exception") from ValueError("cause")

    with raises_with_cause(RuntimeError, "exception", ValueError, "tial mat"):
        raise RuntimeError("exception") from ValueError("partial match")

    with raises_with_cause(RuntimeError, None, ValueError, "cause"):
        raise RuntimeError("exception") from ValueError("cause")

    with raises_with_cause(RuntimeError, "exception", ValueError, None):
        raise RuntimeError("exception") from ValueError("bar")

    with raises_with_cause(RuntimeError, None, ValueError, None):
        raise RuntimeError("foo") from ValueError("bar")

    # we're trying to stick to pytest semantics
    # If the exception types don't match, raise the original exception
    # If the text doesn't match, raise an assert

    with pytest.raises(RuntimeError):
        with raises_with_cause(RuntimeError, "exception", ValueError, "cause"):
            raise RuntimeError("exception") from OSError("cause")

    with pytest.raises(ValueError):
        with raises_with_cause(RuntimeError, "exception", ValueError, "cause"):
            raise ValueError("exception") from ValueError("cause")

    with pytest.raises(AssertionError):
        with raises_with_cause(RuntimeError, "exception", ValueError, "foo"):
            raise RuntimeError("exception") from ValueError("cause")

    with pytest.raises(AssertionError):
        with raises_with_cause(RuntimeError, "foo", ValueError, "cause"):
            raise RuntimeError("exception") from ValueError("cause")


@pytest.mark.slow
def test_check_thread_leak():
    event = threading.Event()

    t1 = threading.Thread(target=lambda: (event.wait(), "one"))
    t1.start()

    t2 = t3 = None
    try:
        with pytest.raises(
            pytest.fail.Exception, match=r"2 thread\(s\) were leaked"
        ) as exc:
            with check_thread_leak():
                t2 = threading.Thread(target=lambda: (event.wait(), "two"))
                t2.start()
                t3 = threading.Thread(target=lambda: (event.wait(), "three"))
                t3.start()

        msg = exc.value.msg
        assert msg
        print(msg)  # For reference, if test fails

        # First, outer thread is ignored
        assert msg.count("Call stack of leaked thread") == 2
        assert "one" not in msg

        # Make sure we can see the full traceback, not just the last line
        assert msg.count(__file__) == 2
        assert 'target=lambda: (event.wait(), "two")' in msg
        assert 'target=lambda: (event.wait(), "three")' in msg

        # Ensure there aren't too many or too few newlines
        exc.match(r'event.wait\(\), "three"\)\)\n +File')
    finally:
        # Clean up
        event.set()
        t1.join(5)
        if t2:
            t2.join(5)
        if t3:
            t3.join(5)


@pytest.mark.parametrize("sync", [True, False])
def test_fail_hard(sync):
    """@fail_hard is a last resort when error handling for everything that we foresaw
    could possibly go wrong failed.
    Instead of trying to force a crash here, we'll write custom methods which do crash.
    """

    class CustomError(Exception):
        pass

    class FailWorker(Worker):
        @fail_hard
        def fail_sync(self):
            raise CustomError()

        @fail_hard
        async def fail_async(self):
            raise CustomError()

    test_done = False

    @gen_cluster(nthreads=[])
    async def test(s):
        nonlocal test_done
        with captured_logger("distributed.worker") as logger:
            async with FailWorker(s.address) as a:
                with pytest.raises(CustomError):
                    if sync:
                        a.fail_sync()
                    else:
                        await a.fail_async()

                while a.status != Status.closed:
                    await asyncio.sleep(0.01)
            method_name = "fail_sync" if sync else "fail_async"
            assert f"worker-{method_name}-fail-hard" in logger.getvalue()

        test_done = True

    with pytest.raises(CustomError):
        test()
    assert test_done


def test_popen_write_during_terminate_deadlock():
    # Fabricate a command which, when terminated, tries to write more than the pipe
    # buffer can hold (OS specific: on Linux it's typically 65536 bytes; on Windows it's
    # less). This would deadlock if `proc.wait()` was called, since the process will be
    # trying to write to stdout, but stdout isn't being cleared because our process is
    # blocked in `proc.wait()`. `proc.communicate()` is necessary:
    # https://docs.python.org/3/library/subprocess.html#subprocess.Popen.wait
    with popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import signal
                import threading

                e = threading.Event()

                def cb(signum, frame):
                    # 131072 is 2x the size of the default Linux pipe buffer
                    print('x' * 131072)
                    e.set()

                signal.signal(signal.SIGINT, cb)
                print('ready', flush=True)
                e.wait()
                """
            ),
        ],
        capture_output=True,
    ) as proc:
        assert proc.stdout.readline().strip() == b"ready"
    # Exiting the context manager (terminating the subprocess) will raise
    # `subprocess.TimeoutExpired` if this test breaks.


def test_popen_timeout(capsys):
    with pytest.raises(subprocess.TimeoutExpired):
        with popen(
            [
                sys.executable,
                "-c",
                textwrap.dedent(
                    """
                    import signal
                    import sys
                    import time

                    if sys.platform == "win32":
                        signal.signal(signal.SIGBREAK, signal.default_int_handler)
                        # ^ Cause `CTRL_BREAK_EVENT` on Windows to raise `KeyboardInterrupt`

                    print('ready', flush=True)
                    while True:
                        try:
                            time.sleep(0.1)
                            print("slept", flush=True)
                        except KeyboardInterrupt:
                            print("interrupted", flush=True)
                    """
                ),
            ],
            capture_output=True,
            terminate_timeout=1,
        ) as proc:
            assert proc.stdout
            assert proc.stdout.readline().strip() == b"ready"
    # Exiting contextmanager sends SIGINT, waits 1s for shutdown.
    # Our script ignores SIGINT, so after 1s it sends SIGKILL.
    # The contextmanager raises `TimeoutExpired` once the process is killed,
    # because it failed the 1s timeout
    captured = capsys.readouterr()
    assert "stdout: returncode" in captured.out
    assert "interrupted" in captured.out
    assert "slept" in captured.out


def test_popen_always_prints_output(capsys):
    # We always print stdout even if there was no error, in case some other assertion
    # later in the test fails and the output would be useful.
    with popen([sys.executable, "-c", "print('foo')"], capture_output=True) as proc:
        proc.communicate(timeout=5)

    captured = capsys.readouterr()
    assert "stdout: returncode 0" in captured.out
    assert "foo" in captured.out


@gen_test()
async def test_freeze_batched_send():
    async with EchoServer() as e:
        comm = await connect(e.address)
        b = BatchedSend(interval=0)
        b.start(comm)

        b.send("hello")
        assert await comm.read() == ("hello",)

        with freeze_batched_send(b) as locked_comm:
            b.send("foo")
            b.send("bar")

            # Sent messages are available on the write queue
            msg = await locked_comm.write_queue.get()
            assert msg == (comm.peer_address, ["foo", "bar"])

            # Sent messages will not reach the echo server
            await asyncio.sleep(0.01)
            assert e.count == 1

            # Now we let messages send to the echo server
            locked_comm.write_event.set()
            assert await comm.read() == ("foo", "bar")
            assert e.count == 2

            locked_comm.write_event.clear()
            b.send("baz")
            await asyncio.sleep(0.01)
            assert e.count == 2

        assert b.comm is comm
        assert await comm.read() == ("baz",)
        assert e.count == 3


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_wait_for_state(c, s, a, capsys):
    ev = Event()
    x = c.submit(lambda ev: ev.wait(), ev, key="x")

    await asyncio.gather(
        wait_for_state("x", "processing", s),
        wait_for_state("x", "executing", a),
        c.run(wait_for_state, "x", "executing"),
    )

    await ev.set()

    await asyncio.gather(
        wait_for_state("x", "memory", s),
        wait_for_state("x", {"memory", "other"}, a),
        c.run(wait_for_state, "x", "memory"),
    )

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wait_for_state("x", "bad_state", s), timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wait_for_state("x", ("this", "that"), s), timeout=0.1)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(wait_for_state("y", "memory", s), timeout=0.1)
    assert capsys.readouterr().out == (
        f"tasks[x].state='memory' on {s.address}; expected state='bad_state'\n"
        f"tasks[x].state='memory' on {s.address}; expected state=('this', 'that')\n"
        f"tasks[y] not found on {s.address}\n"
    )


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_wait_for_stimulus(c, s, a):
    t1 = asyncio.create_task(wait_for_stimulus(ComputeTaskEvent, a))
    t2 = asyncio.create_task(wait_for_stimulus(ComputeTaskEvent, a, key="y"))
    await asyncio.sleep(0.05)
    assert not t1.done()
    assert not t2.done()

    x = c.submit(inc, 1, key="x")
    ev = await t1
    assert isinstance(ev, ComputeTaskEvent)
    await wait_for_stimulus(ComputeTaskEvent, a, key="x")
    await c.run(wait_for_stimulus, ComputeTaskEvent, key="x")
    assert not t2.done()

    y = c.submit(inc, 1, key="y")
    await t2


def test_ws_with_running_task(ws_with_running_task):
    ws = ws_with_running_task
    ts = ws.tasks["x"]
    assert ts.resource_restrictions == {"R": 1}
    assert ws.available_resources == {"R": 0}
    assert ws.total_resources == {"R": 1}
    assert ts.state in ("executing", "long-running")


def test_sizeof():
    assert sizeof(SizeOf(100)) == 100
    assert isinstance(gen_nbytes(100), SizeOf)
    assert sizeof(gen_nbytes(100)) == 100


@pytest.mark.parametrize(
    "input, exc, msg",
    [
        (12345.0, TypeError, "Expected integer"),
        (-1, ValueError, "larger than"),
        (0, ValueError, "larger than"),
        (10, ValueError, "larger than"),
    ],
)
def test_sizeof_error(input, exc, msg):
    with pytest.raises(exc, match=msg):
        SizeOf(input)

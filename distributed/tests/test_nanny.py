from __future__ import annotations

import asyncio
import gc
import logging
import multiprocessing as mp
import os
import random
import sys
import warnings
import weakref
from contextlib import suppress
from unittest import mock

import psutil
import pytest
from tlz import first
from tornado.ioloop import IOLoop

import dask
from dask.utils import tmpfile

from distributed import Nanny, Scheduler, Worker, profile, rpc, wait, worker
from distributed.compatibility import LINUX, WINDOWS
from distributed.core import CommClosedError, ConnectionPool, Status
from distributed.diagnostics import SchedulerPlugin
from distributed.diagnostics.plugin import NannyPlugin, WorkerPlugin
from distributed.metrics import time
from distributed.protocol.pickle import dumps
from distributed.utils import TimeoutError, get_mp_context, parse_ports
from distributed.utils_test import (
    BlockedInstantiateNanny,
    async_poll_for,
    captured_logger,
    gen_cluster,
    gen_test,
    raises_with_cause,
)

pytestmark = [pytest.mark.ci1, pytest.mark.gpu]


@gen_cluster(Worker=Nanny)
async def test_str(s, a, b):
    assert a.worker_address in str(a)
    assert a.worker_address in repr(a)
    assert str(a.nthreads) in str(a)
    assert str(a.nthreads) in repr(a)


@gen_cluster(nthreads=[], client=True)
async def test_nanny_process_failure(c, s):
    async with Nanny(s.address, nthreads=2) as n:
        first_dir = n.worker_dir

        assert os.path.exists(first_dir)

        pid = n.pid
        assert pid is not None
        with suppress(CommClosedError):
            await c.run(os._exit, 0, workers=[n.worker_address])

        # Wait while process dies
        await async_poll_for(lambda: n.pid != pid, timeout=5)
        # Wait while process comes back
        await async_poll_for(lambda: n.is_alive(), timeout=5)

        # assert n.worker_address != original_address  # most likely

        await async_poll_for(
            lambda: n.worker_address in s.workers and n.worker_dir is not None,
            timeout=5,
        )

        second_dir = n.worker_dir

    await n.close()
    assert not os.path.exists(second_dir)
    assert not os.path.exists(first_dir)
    assert first_dir != n.worker_dir
    s.stop()


@gen_cluster(nthreads=[])
async def test_run(s):
    async with Nanny(s.address, nthreads=2) as n:
        async with rpc(n.address) as nn:
            response = await nn.run(function=dumps(lambda: 1))
            assert response["status"] == "OK"
            assert response["result"] == 1


@pytest.mark.slow
@gen_cluster(config={"distributed.comm.timeouts.connect": "1s"}, timeout=120)
async def test_no_hang_when_scheduler_closes(s, a, b):
    # https://github.com/dask/distributed/issues/2880
    with captured_logger("tornado.application", logging.ERROR) as logger:
        await s.close()
        await asyncio.sleep(1.2)
        assert a.status == Status.closed
        assert b.status == Status.closed

    out = logger.getvalue()
    assert "Timed out trying to connect" not in out


@pytest.mark.slow
@gen_cluster(
    Worker=Nanny, nthreads=[("127.0.0.1", 1)], worker_kwargs={"reconnect": False}
)
async def test_close_on_disconnect(s, w):
    with captured_logger("distributed.nanny") as logger:
        await s.close()

        start = time()
        while w.status != Status.closed:
            await asyncio.sleep(0.05)
            assert time() < start + 9
    assert "Reason: scheduler-close" in logger.getvalue()


class Something(Worker):
    # a subclass of Worker which is not Worker
    pass


@gen_cluster(client=True, Worker=Nanny)
async def test_nanny_worker_class(c, s, w1, w2):
    out = await c._run(lambda dask_worker=None: str(dask_worker.__class__))
    assert "Worker" in list(out.values())[0]
    assert w1.Worker is Worker


@gen_cluster(client=True, Worker=Nanny, worker_kwargs={"worker_class": Something})
async def test_nanny_alt_worker_class(c, s, w1, w2):
    out = await c._run(lambda dask_worker=None: str(dask_worker.__class__))
    assert "Something" in list(out.values())[0]
    assert w1.Worker is Something


@pytest.mark.slow
@gen_cluster(nthreads=[])
async def test_nanny_death_timeout(s):
    await s.close()
    w = Nanny(s.address, death_timeout=1)
    with pytest.raises(TimeoutError):
        await w

    assert w.status == Status.failed


@gen_cluster(client=True, Worker=Nanny)
async def test_random_seed(c, s, a, b):
    async def check_func(func):
        x = c.submit(func, 0, 2**31, pure=False, workers=a.worker_address)
        y = c.submit(func, 0, 2**31, pure=False, workers=b.worker_address)
        assert x.key != y.key
        x = await x
        y = await y
        assert x != y

    await check_func(lambda a, b: random.randint(a, b))
    np = pytest.importorskip("numpy")
    await check_func(lambda a, b: np.random.randint(a, b))


@pytest.mark.skipif(WINDOWS, reason="num_fds not supported on windows")
@gen_cluster(nthreads=[])
async def test_num_fds(s):
    proc = psutil.Process()

    # Warm up
    async with Nanny(s.address):
        pass
    with profile.lock:
        gc.collect()

    before = proc.num_fds()

    for _ in range(3):
        async with Nanny(s.address):
            await asyncio.sleep(0.1)

    while proc.num_fds() > before:
        print("fds:", before, proc.num_fds())
        await asyncio.sleep(0.1)


@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
@gen_cluster(client=True, nthreads=[])
async def test_worker_uses_same_host_as_nanny(c, s):
    for host in ["tcp://0.0.0.0", "tcp://127.0.0.2"]:
        async with Nanny(s.address, host=host):

            def func(dask_worker):
                return dask_worker.listener.listen_address

            result = await c.run(func)
            assert host in first(result.values())


@gen_test()
async def test_scheduler_file():
    with tmpfile() as fn:
        s = await Scheduler(scheduler_file=fn, dashboard_address=":0")
        async with Nanny(scheduler_file=fn) as n:
            assert set(s.workers) == {n.worker_address}
        s.stop()


@pytest.mark.xfail(
    os.environ.get("MINDEPS") == "true",
    reason="Timeout errors with mindeps environment",
)
@gen_cluster(client=True, Worker=Nanny, nthreads=[("127.0.0.1", 2)])
async def test_nanny_timeout(c, s, a):
    x = await c.scatter(123)
    await a.restart(timeout=0.1)

    start = time()
    while x.status != "cancelled":
        await asyncio.sleep(0.1)
        assert time() < start + 7


@gen_cluster(
    nthreads=[("", 1)] * 8,
    client=True,
    clean_kwargs={"threads": False},
    config={"distributed.worker.memory.pause": False},
)
async def test_throttle_outgoing_transfers(c, s, a, *other_workers):
    # Put a bunch of small data on worker a
    logging.getLogger("distributed.worker").setLevel(logging.DEBUG)
    remote_data = c.map(
        lambda x: b"0" * 10000, range(10), pure=False, workers=[a.address]
    )
    await wait(remote_data)

    a.status = Status.paused
    a.transfer_outgoing_count = 2

    requests = [
        await a.get_data(await w.rpc.connect(w.address), keys=[f.key], who=w.address)
        for w in other_workers
        for f in remote_data
    ]
    await wait(requests)
    wlogs = await c.get_worker_logs(workers=[a.address])
    wlogs = "\n".join(x[1] for x in wlogs[a.address])
    assert "throttling" in wlogs.lower()


@gen_cluster(nthreads=[])
async def test_scheduler_address_config(s):
    with dask.config.set({"scheduler-address": s.address}):
        async with Nanny() as nanny:
            assert nanny.scheduler.address == s.address
            while not s.workers:
                await asyncio.sleep(0.01)


@pytest.mark.slow
@gen_test()
async def test_wait_for_scheduler():
    with captured_logger("distributed") as log:
        w = Nanny("127.0.0.1:44737")
        IOLoop.current().add_callback(w.start)
        await asyncio.sleep(6)
        await w.close()

    log = log.getvalue()
    assert "error" not in log.lower(), log
    assert "restart" not in log.lower(), log


@gen_cluster(nthreads=[], client=True)
async def test_environment_variable(c, s):
    a = Nanny(s.address, memory_limit=0, env={"FOO": "123"})
    b = Nanny(s.address, memory_limit=0, env={"FOO": "456"})
    await asyncio.gather(a, b)
    results = await c.run(lambda: os.environ["FOO"])
    assert results == {a.worker_address: "123", b.worker_address: "456"}
    await asyncio.gather(a.close(), b.close())


@gen_cluster(nthreads=[], client=True)
async def test_environment_variable_by_config(c, s, monkeypatch):
    with dask.config.set({"distributed.nanny.environ": "456"}):
        with pytest.raises(TypeError, match="configuration must be of type dict"):
            Nanny(s.address, memory_limit=0)

    with dask.config.set({"distributed.nanny.environ": {"FOO": "456"}}):
        # precedence
        # kwargs > env var > config

        with mock.patch.dict(os.environ, {"FOO": "BAR"}, clear=True):
            a = Nanny(s.address, memory_limit=0, env={"FOO": "123"})
            x = Nanny(s.address, memory_limit=0)

        b = Nanny(s.address, memory_limit=0)

        await asyncio.gather(a, b, x)
        results = await c.run(lambda: os.environ["FOO"])
        assert results == {
            a.worker_address: "123",
            b.worker_address: "456",
            x.worker_address: "BAR",
        }
        await asyncio.gather(a.close(), b.close(), x.close())


@gen_cluster(
    nthreads=[],
    client=True,
    config={"distributed.nanny.environ": {"A": 1, "B": 2, "D": 4}},
)
async def test_environment_variable_config(c, s, monkeypatch):
    monkeypatch.setenv("D", "123")
    async with Nanny(s.address, env={"B": 3, "C": 4}) as n:
        results = await c.run(lambda: os.environ)
        assert results[n.worker_address]["A"] == "1"
        assert results[n.worker_address]["B"] == "3"
        assert results[n.worker_address]["C"] == "4"
        assert results[n.worker_address]["D"] == "123"


@gen_cluster(
    nthreads=[("", 1)],
    client=True,
    Worker=Nanny,
    config={
        "distributed.nanny.pre-spawn-environ": {"PRE-SPAWN": 1},
        "distributed.nanny.environ": {"POST-SPAWN": 2},
    },
)
async def test_environment_variable_pre_post_spawn(c, s, n):
    assert n.env == {"PRE-SPAWN": "1", "POST-SPAWN": "2", "PYTHONHASHSEED": "6640"}
    results = await c.run(lambda: os.environ)
    assert results[n.worker_address]["PRE-SPAWN"] == "1"
    assert results[n.worker_address]["POST-SPAWN"] == "2"
    # if unset in pre-spawn-environ config, PYTHONHASHSEED defaults to "6640" to ensure
    # consistent hashing across workers; https://github.com/dask/distributed/issues/4141
    assert results[n.worker_address]["PYTHONHASHSEED"] == "6640"

    del os.environ["PRE-SPAWN"]
    assert "POST-SPAWN" not in os.environ


@gen_cluster(
    nthreads=[],
    client=True,
    config={
        "distributed.nanny.pre-spawn-environ.PRE1": 1,
        "distributed.nanny.pre-spawn-environ.PRE2": 2,
        "distributed.nanny.pre-spawn-environ.PRE3": 3,
        "distributed.nanny.environ.POST1": 4,
        "distributed.nanny.environ.POST2": 5,
        "distributed.nanny.environ.POST3": 6,
    },
)
async def test_environment_variable_overlay(c, s):
    """You can set a value to None to unset a variable in a config overlay"""
    # Not the same as running Nanny(config=...), which would not work for pre-spawn
    # variables
    with dask.config.set(
        {
            "distributed.nanny.pre-spawn-environ.PRE2": 7,
            "distributed.nanny.pre-spawn-environ.PRE3": None,
            "distributed.nanny.environ.POST2": 8,
            "distributed.nanny.environ.POST3": None,
        },
    ):
        async with Nanny(s.address):
            env = await c.submit(lambda: os.environ)
            assert env["PRE1"] == "1"
            assert env["PRE2"] == "7"
            assert "PRE3" not in env
            assert env["POST1"] == "4"
            assert env["POST2"] == "8"
            assert "POST3" not in env


@gen_cluster(client=True, nthreads=[])
async def test_config_param_overlays(c, s):
    with dask.config.set({"test123.foo": 1, "test123.bar": 2}):
        async with Nanny(s.address, config={"test123.bar": 3, "test123.baz": 4}) as n:
            out = await c.submit(lambda: dask.config.get("test123"))

    assert out == {"foo": 1, "bar": 3, "baz": 4}


@gen_cluster(nthreads=[])
async def test_local_directory(s):
    with tmpfile() as fn:
        with dask.config.set(temporary_directory=fn):
            async with Nanny(s.address) as n:
                assert n.local_directory.startswith(fn)
                assert "dask-scratch-space" in n.local_directory
                assert n.process.worker_dir.count("dask-scratch-space") == 1


@pytest.mark.skipif(WINDOWS, reason="Need POSIX filesystem permissions and UIDs")
@gen_cluster(nthreads=[])
async def test_unwriteable_dask_worker_space(s, tmp_path):
    os.mkdir(f"{tmp_path}/dask-scratch-space", mode=0o500)
    with pytest.raises(PermissionError):
        open(f"{tmp_path}/dask-scratch-space/tryme", "w")

    with dask.config.set(temporary_directory=tmp_path):
        async with Nanny(s.address) as n:
            assert n.local_directory == os.path.join(
                tmp_path, f"dask-scratch-space-{os.getuid()}"
            )
            assert n.process.worker_dir.count(f"dask-scratch-space-{os.getuid()}") == 1


def _noop(x):
    """Define here because closures aren't pickleable."""
    pass


@gen_cluster(
    nthreads=[("127.0.0.1", 1)],
    client=True,
    Worker=Nanny,
    config={"distributed.worker.daemon": False},
)
async def test_mp_process_worker_no_daemon(c, s, a):
    def multiprocessing_worker():
        p = mp.Process(target=_noop, args=(None,))
        p.start()
        p.join()

    await c.submit(multiprocessing_worker)


@gen_cluster(
    nthreads=[("127.0.0.1", 1)],
    client=True,
    Worker=Nanny,
    config={"distributed.worker.daemon": False},
)
async def test_mp_pool_worker_no_daemon(c, s, a):
    def pool_worker(world_size):
        with mp.Pool(processes=world_size) as p:
            p.map(_noop, range(world_size))

    await c.submit(pool_worker, 4)


@gen_cluster(nthreads=[])
async def test_nanny_closes_cleanly(s):
    async with Nanny(s.address) as n:
        assert n.process.pid
        proc = n.process.process
    assert not n.process.is_alive()
    assert not proc.is_alive()
    assert proc.exitcode == 0


@pytest.mark.slow
@gen_cluster(nthreads=[], timeout=60)
async def test_lifetime(s):
    counter = 0
    event = asyncio.Event()

    class Plugin(SchedulerPlugin):
        def add_worker(self, **kwargs):
            pass

        def remove_worker(self, **kwargs):
            nonlocal counter
            counter += 1
            if counter == 2:  # wait twice, then trigger closing event
                event.set()

    s.add_plugin(Plugin())
    async with Nanny(s.address):
        async with Nanny(s.address, lifetime="500 ms", lifetime_restart=True):
            await event.wait()


@gen_cluster(client=True, nthreads=[])
async def test_nanny_closes_cleanly_if_worker_is_terminated(c, s):
    async with Nanny(s.address) as n:
        async with c.rpc(n.worker_address) as w:
            IOLoop.current().add_callback(w.terminate)
            start = time()
            while n.status != Status.closed:
                await asyncio.sleep(0.01)
                assert time() < start + 5

            assert n.status == Status.closed


@gen_cluster(client=True, nthreads=[])
async def test_config(c, s):
    async with Nanny(s.address, config={"foo": "bar"}) as n:
        config = await c.run(dask.config.get, "foo")
        assert config[n.worker_address] == "bar"


@gen_cluster(client=True, nthreads=[])
async def test_nanny_port_range(c, s):
    nanny_port = "9867:9868"
    worker_port = "9869:9870"
    async with Nanny(s.address, port=nanny_port, worker_port=worker_port) as n1:
        assert n1.port == 9867  # Selects first port in range
        async with Nanny(s.address, port=nanny_port, worker_port=worker_port) as n2:
            assert n2.port == 9868  # Selects next port in range
            with raises_with_cause(
                RuntimeError,
                "Nanny failed to start.",
                ValueError,
                "with port 9867:9868",
            ):  # No more ports left
                async with Nanny(s.address, port=nanny_port, worker_port=worker_port):
                    pass

            # Ensure Worker ports are in worker_port range
            def get_worker_port(dask_worker):
                return dask_worker.port

            worker_ports = await c.run(get_worker_port)
            assert list(worker_ports.values()) == parse_ports(worker_port)


class KeyboardInterruptWorker(worker.Worker):
    """A Worker that raises KeyboardInterrupt almost immediately"""

    async def heartbeat(self):
        def raise_err():
            raise KeyboardInterrupt()

        self.loop.add_callback(raise_err)


@pytest.mark.parametrize("protocol", ["tcp", "ucx"])
@gen_test()
async def test_nanny_closed_by_keyboard_interrupt(ucx_loop, protocol):
    if protocol == "ucx":  # Skip if UCX isn't available
        pytest.importorskip("ucp")

    async with Scheduler(protocol=protocol, dashboard_address=":0") as s:
        async with Nanny(
            s.address, nthreads=1, worker_class=KeyboardInterruptWorker
        ) as n:
            await n.process.stopped.wait()
            # Check that the scheduler has been notified about the closed worker
            assert "remove-worker" in str(s.get_events())


class BrokenWorker(worker.Worker):
    async def start_unsafe(self):
        raise ValueError("broken")


@gen_cluster(nthreads=[])
async def test_worker_start_exception(s):
    nanny = Nanny(s.address, worker_class=BrokenWorker)
    with captured_logger(logger="distributed.nanny", level=logging.WARNING) as logs:
        with raises_with_cause(
            RuntimeError,
            "Nanny failed to start",
            RuntimeError,
            "BrokenWorker failed to start",
        ):
            async with nanny:
                pass
    assert nanny.status == Status.failed
    # ^ NOTE: `Nanny.close` sets it to `closed`, then `Server.start._close_on_failure` sets it to `failed`
    assert not nanny.process.is_alive()
    assert "Restarting worker" not in logs.getvalue()
    # Avoid excessive spewing. (It's also printed once extra within the subprocess, which is okay.)
    assert logs.getvalue().count("ValueError: broken") == 2, logs.getvalue()


@gen_cluster(nthreads=[])
async def test_failure_during_worker_initialization(s):
    with captured_logger(logger="distributed.nanny", level=logging.WARNING) as logs:
        with pytest.raises(RuntimeError):
            await Nanny(s.address, foo="bar")
    assert "Restarting worker" not in logs.getvalue()


@gen_cluster(client=True, Worker=Nanny)
async def test_environ_plugin(c, s, a, b):
    from dask.distributed import Environ

    await c.register_plugin(Environ({"ABC": 123}))

    async with Nanny(s.address, name="new") as n:
        results = await c.run(os.getenv, "ABC")
        assert results[a.worker_address] == "123"
        assert results[b.worker_address] == "123"
        assert results[n.worker_address] == "123"


@pytest.mark.parametrize(
    "modname",
    [
        # numpy is always imported, and for a good reason:
        # https://github.com/dask/distributed/issues/5729
        "scipy",
        pytest.param("pandas", marks=pytest.mark.xfail(reason="distributed#5723")),
    ],
)
@gen_cluster(client=True, Worker=Nanny, nthreads=[("", 1)])
async def test_no_unnecessary_imports_on_worker(c, s, a, modname):
    """
    Regression test against accidentally importing unnecessary modules at worker startup.

    Importing modules like pandas slows down worker startup, especially if workers are
    loading their software environment from NFS or other non-local filesystems.
    It also slightly increases memory footprint.
    """

    def assert_no_import(dask_worker):
        assert modname not in sys.modules

    await c.wait_for_workers(1)
    await c.run(assert_no_import)


@pytest.mark.slow
@gen_cluster(client=True, Worker=Nanny)
async def test_repeated_restarts(c, s, a, b):
    for _ in range(3):
        await c.restart()
        assert len(s.workers) == 2


@pytest.mark.slow
@gen_cluster(
    client=True,
    Worker=Nanny,
    worker_kwargs={"memory_limit": "1 GiB"},
    nthreads=[("127.0.0.1", 1)],
)
async def test_restart_memory(c, s, n):
    # First kill nanny with restart
    await c.restart()

    # then kill nanny with memory
    from dask.distributed import KilledWorker

    np = pytest.importorskip("numpy")
    s.allowed_failures = 1
    future = c.submit(np.ones, 300_000_000, dtype="f8")
    with pytest.raises(KilledWorker):
        await future

    while not s.workers:
        await asyncio.sleep(0.1)

    msgs = s.get_events("worker-restart-memory")
    assert len(msgs)
    msg = msgs[0][1]
    assert isinstance(msg, dict)
    assert {"worker", "pid", "rss"}.issubset(set(msg))


class BlockClose(WorkerPlugin):
    def __init__(self, close_happened):
        self.close_happened = close_happened

    async def teardown(self, worker):
        # Never let the worker cleanly shut down, so it has to be killed
        self.close_happened.set()
        while True:
            await asyncio.sleep(10)


@pytest.mark.slow
@gen_cluster(nthreads=[])
async def test_close_joins(s):
    close_happened = get_mp_context().Event()

    nanny = Nanny(s.address, plugins=[BlockClose(close_happened)])
    async with nanny:
        p = nanny.process
        assert p
        close_t = asyncio.create_task(nanny.close())

        while not close_happened.wait(0):
            await asyncio.sleep(0.01)

        assert not close_t.done()
        assert nanny.status == Status.closing
        assert nanny.process and nanny.process.status == Status.stopping

        await close_t

        assert nanny.status == Status.closed
        assert not nanny.process.is_alive()

        assert p.status == Status.stopped
        assert not p.process


@gen_cluster(Worker=Nanny, nthreads=[("", 1)])
async def test_scheduler_crash_doesnt_restart(s, a):
    # Simulate a scheduler crash by disconnecting it first
    # (`s.close()` would tell workers to cleanly shut down)
    bcomm = next(iter(s.stream_comms.values()))
    bcomm.abort()
    await s.close()

    while a.status not in {Status.closing_gracefully, Status.closed}:
        await asyncio.sleep(0.01)

    await a.finished()
    assert a.status == Status.closed
    assert not a.process.is_alive()


@pytest.mark.slow
@pytest.mark.skipif(not LINUX, reason="Requires GNU libc")
@gen_cluster(
    client=True,
    Worker=Nanny,
    nthreads=[("", 2)],
    worker_kwargs={"memory_limit": "2GiB"},
)
async def test_malloc_trim_threshold(c, s, a):
    """Test that the nanny sets the MALLOC_TRIM_THRESHOLD_ environment variable before
    starting the worker process.

    This test relies on these settings to work:

        distributed.nanny.pre-spawn-environ.MALLOC_TRIM_THRESHOLD_: 65536
        distributed.worker.multiprocessing-method: spawn

    We're deliberately not setting them explicitly in @gen_cluster above, as we want
    this test to trip if somebody changes distributed.yaml.

    Note
    ----
    This test may start failing in a future Python version if CPython switches to
    using mimalloc by default. If it does, a thorough benchmarking exercise is needed.
    """
    pytest.importorskip("numpy")
    da = pytest.importorskip("dask.array")

    arr = da.random.random(2**29 // 8, chunks="512 kiB")  # 0.5 GiB
    arr = arr.persist()
    await wait(arr)
    # Wait for heartbeat
    await async_poll_for(lambda: s.memory.process > 2**29, timeout=5)
    del arr

    # This is the delicate bit, as it relies on
    # 1. PyMem_Free() to be quick to invoke glibc free() when memory becomes available
    # 2. glibc free() to be quick to invoke the kernel's sbrk() when the same happens
    #
    # At the moment of writing, the readings are:
    # - 132        MiB after starting a new worker
    # - 670~725[1] MiB after arr has been fully computed
    # - 156~210[1] MiB at the end of this test, with MALLOC_TRIM_THRESHOLD_=65536
    # - 620~670[1] MiB at the end of this test, without MALLOC_TRIM_THRESHOLD_
    # [1] depends on distributed.scheduler.worker-saturation
    await async_poll_for(lambda: s.memory.process < 300 * 2**20, timeout=5)


@gen_cluster(client=True, nthreads=[])
async def test_default_client_does_not_propagate_to_subprocess(c, s):
    @dask.delayed
    def run_in_thread():
        return

    def func():
        with warnings.catch_warnings(record=True) as rec:
            warnings.filterwarnings(
                "once",
                message="Running on a single-machine scheduler",
                category=UserWarning,
            )
            # If no scheduler kwarg is provided, this will
            # automatically transition to long-running
            dask.compute(run_in_thread(), scheduler="single-threaded")
        return rec

    async with Nanny(s.address):
        rec = await c.submit(func)
        assert not rec


@gen_cluster(client=True, nthreads=[], config={"test123": 456})
async def test_worker_inherits_temp_config(c, s):
    with dask.config.set(test123=123):
        async with Nanny(s.address):
            out = await c.submit(lambda: dask.config.get("test123"))
            assert out == 123


@gen_cluster(client=True, nthreads=[("", 1)], Worker=Nanny)
async def test_nanny_plugin_simple(c, s, a):
    """A plugin should be registered to already existing workers but also to new ones."""
    plugin = DummyNannyPlugin("foo")
    await c.register_plugin(plugin)
    assert a._plugin_registered
    async with Nanny(s.address) as n:
        assert n._plugin_registered


class DummyNannyPlugin(NannyPlugin):
    def __init__(self, name, restart=False):
        self.restart = restart
        self.name = name
        self.nanny = None

    def setup(self, nanny):
        print(f"Setup on {nanny}")
        self.nanny = weakref.ref(nanny)
        nanny._plugin_registered = True

    def teardown(self, nanny):
        nanny._plugin_registered = False


@pytest.mark.parametrize("restart", [True, False])
@gen_cluster(client=True, nthreads=[])
async def test_nanny_plugin_register_during_start_success(c, s, restart):
    plugin = DummyNannyPlugin("foo", restart=restart)
    n = BlockedInstantiateNanny(s.address)
    assert not hasattr(n, "_plugin_registered")
    start = asyncio.create_task(n.start())
    try:
        await n.in_instantiate.wait()

        register = asyncio.create_task(c.register_plugin(plugin))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(register), timeout=0.1)
        n.wait_instantiate.set()
        assert await register
        await start
        assert n._plugin_registered
    finally:
        start.cancel()
        await n.close()


class SlowBrokenNanny(Nanny):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_instantiate = asyncio.Event()
        self.wait_instantiate = asyncio.Event()

    async def start_worker_process(self):
        self.in_instantiate.set()
        await self.wait_instantiate.wait()
        raise RuntimeError("Nope")


@pytest.mark.parametrize("restart", [True, False])
@gen_cluster(client=True, nthreads=[])
async def test_nanny_plugin_register_during_start_failure(c, s, restart):
    plugin = DummyNannyPlugin("foo", restart=restart)
    n = SlowBrokenNanny(s.address)
    assert not hasattr(n, "_plugin_registered")
    start = asyncio.create_task(n.start())
    await n.in_instantiate.wait()

    register = asyncio.create_task(c.register_plugin(plugin))
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(register), timeout=0.1)
    n.wait_instantiate.set()
    with pytest.raises(RuntimeError):
        await start
    assert not await register


class SlowDistNanny(Nanny):
    def __init__(self, *args, in_instantiate, wait_instantiate, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_instantiate = in_instantiate
        self.wait_instantiate = wait_instantiate

    async def start_worker_process(self):
        self.in_instantiate.set()
        self.wait_instantiate.wait()
        return await super().start_worker_process()


def run_nanny(scheduler_addr, in_instantiate, wait_instantiate):
    async def _():
        worker = await SlowDistNanny(
            scheduler_addr,
            wait_instantiate=wait_instantiate,
            in_instantiate=in_instantiate,
        )
        await worker.finished()

    asyncio.run(_())


@pytest.mark.parametrize("restart", [True, False])
@gen_cluster(client=True, nthreads=[])
async def test_nanny_plugin_register_nanny_killed(c, s, restart):
    in_instantiate = get_mp_context().Event()
    wait_instantiate = get_mp_context().Event()
    proc = get_mp_context().Process(
        name="run_nanny",
        target=run_nanny,
        kwargs={
            "in_instantiate": in_instantiate,
            "wait_instantiate": wait_instantiate,
        },
        args=(s.address,),
    )
    proc.start()
    try:
        plugin = DummyNannyPlugin("foo", restart=restart)
        await asyncio.to_thread(in_instantiate.wait)
        register = asyncio.create_task(c.register_plugin(plugin))
    finally:
        proc.kill()
    assert await register == {}


@pytest.mark.parametrize("api", ["restart", "kill"])
@gen_cluster(client=True, nthreads=[("", 1)], Worker=Nanny)
async def test_restart_stress(c, s, a, api):
    async def keep_killing():
        pool = await ConnectionPool()
        try:
            rpc = pool(a.address)
            for _ in range(2):
                try:
                    import uuid

                    meth = getattr(rpc, api)
                    uid = uuid.uuid4().hex
                    print(api, uid)
                    await meth(reason="scheduler-restart")
                    print(api, uid, "done")

                except OSError:
                    break

                await asyncio.sleep(0.1)
        finally:
            print("closing pool")
            await pool.close()

    kill_tasks = [asyncio.create_task(keep_killing()) for _ in range(2)]
    await asyncio.gather(*kill_tasks)
    assert a.status == Status.running


@pytest.mark.slow
@pytest.mark.parametrize(
    "api",
    [
        "kill",
        "restart",
    ],
)
@gen_cluster(nthreads=[])
async def test_worker_start_exception_after_restart(s, api):
    async with Nanny(s.address, death_timeout="2s") as nanny:
        # Stop the listener on the scheduler, i.e. do not allow any new incoming
        # connections. The restarting workers will fail while trying to attempt
        # connection
        s.stop()
        if api == "kill":
            await nanny.kill()
        else:
            await nanny.restart()
        await nanny.finished()
        assert nanny.status in (Status.failed, Status.closed)

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import copy
import errno
import functools
import gc
import inspect
import io
import logging
import logging.config
import multiprocessing
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import warnings
import weakref
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager, nullcontext, suppress
from itertools import count
from time import sleep
from typing import IO, Any, Generator, Iterator, Literal

import pytest
import yaml
from tlz import assoc, memoize, merge
from tornado.ioloop import IOLoop

import dask

from distributed import Scheduler, system
from distributed import versions as version_module
from distributed.batched import BatchedSend
from distributed.client import Client, _global_clients, default_client
from distributed.comm import Comm
from distributed.comm.tcp import TCP
from distributed.compatibility import MACOS, WINDOWS
from distributed.config import initialize_logging
from distributed.core import (
    CommClosedError,
    ConnectionPool,
    Status,
    clean_exception,
    connect,
    rpc,
)
from distributed.deploy import SpecCluster
from distributed.diagnostics.plugin import WorkerPlugin
from distributed.metrics import time
from distributed.nanny import Nanny
from distributed.node import ServerNode
from distributed.proctitle import enable_proctitle_on_children
from distributed.protocol import deserialize
from distributed.scheduler import TaskState as SchedulerTaskState
from distributed.security import Security
from distributed.utils import (
    DequeHandler,
    _offload_executor,
    get_ip,
    get_ipv6,
    get_mp_context,
    iscoroutinefunction,
    log_errors,
    reset_logger_locks,
    sync,
)
from distributed.worker import WORKER_ANY_RUNNING, Worker
from distributed.worker_state_machine import InvalidTransition, StateMachineEvent
from distributed.worker_state_machine import TaskState as WorkerTaskState
from distributed.worker_state_machine import WorkerState

try:
    import ssl
except ImportError:
    ssl = None  # type: ignore

try:
    import dask.array  # register config
except ImportError:
    pass

try:
    from pytest_timeout import is_debugging
except ImportError:

    def is_debugging() -> bool:
        # The pytest_timeout logic is more sophisticated. Not only debuggers
        # attach a trace callback but vendoring the entire logic is not worth it
        return sys.gettrace() is not None


logger = logging.getLogger(__name__)


logging_levels = {
    name: logger.level
    for name, logger in logging.root.manager.loggerDict.items()
    if isinstance(logger, logging.Logger)
}

_TEST_TIMEOUT = 30
_offload_executor.submit(lambda: None).result()  # create thread during import


@pytest.fixture(scope="session")
def valid_python_script(tmpdir_factory):
    local_file = tmpdir_factory.mktemp("data").join("file.py")
    local_file.write("print('hello world!')")
    return local_file


@pytest.fixture(scope="session")
def client_contract_script(tmpdir_factory):
    local_file = tmpdir_factory.mktemp("data").join("distributed_script.py")
    lines = (
        "from distributed import Client",
        "e = Client('127.0.0.1:8989')",
        "print(e)",
    )
    local_file.write("\n".join(lines))
    return local_file


@pytest.fixture(scope="session")
def invalid_python_script(tmpdir_factory):
    local_file = tmpdir_factory.mktemp("data").join("file.py")
    local_file.write("a+1")
    return local_file


async def cleanup_global_workers():
    for worker in Worker._instances:
        await worker.close(executor_wait=False)


@pytest.fixture
def loop(cleanup):
    with check_instances():
        with pristine_loop() as loop:
            # Monkey-patch IOLoop.start to wait for loop stop
            orig_start = loop.start
            is_stopped = threading.Event()
            is_stopped.set()

            def start():
                is_stopped.clear()
                try:
                    orig_start()
                finally:
                    is_stopped.set()

            loop.start = start

            yield loop

            # Stop the loop in case it's still running
            try:
                sync(loop, cleanup_global_workers, callback_timeout=0.500)
                loop.add_callback(loop.stop)
            except RuntimeError as e:
                if not re.match("IOLoop is clos(ed|ing)", str(e)):
                    raise
            except asyncio.TimeoutError:
                pass
            else:
                is_stopped.wait()


@pytest.fixture
def loop_in_thread(cleanup):
    loop_started = concurrent.futures.Future()
    with concurrent.futures.ThreadPoolExecutor(
        1, thread_name_prefix="test IOLoop"
    ) as tpe:

        async def run():
            io_loop = IOLoop.current()
            stop_event = asyncio.Event()
            loop_started.set_result((io_loop, stop_event))
            await stop_event.wait()

        # run asyncio.run in a thread and collect exceptions from *either*
        # the loop failing to start, or failing to close
        ran = tpe.submit(_run_and_close_tornado, run)
        for f in concurrent.futures.as_completed((loop_started, ran)):
            if f is loop_started:
                io_loop, stop_event = loop_started.result()
                try:
                    yield io_loop
                finally:
                    io_loop.add_callback(stop_event.set)

            elif f is ran:
                # if this is the first iteration the loop failed to start
                # if it's the second iteration the loop has finished or
                # the loop failed to close and we need to raise the exception
                ran.result()
                return


@pytest.fixture
def zmq_ctx():
    import zmq

    ctx = zmq.Context.instance()
    yield ctx
    ctx.destroy(linger=0)


@contextmanager
def pristine_loop():
    IOLoop.clear_instance()
    IOLoop.clear_current()
    loop = IOLoop()
    loop.make_current()
    assert IOLoop.current() is loop
    try:
        yield loop
    finally:
        try:
            loop.close(all_fds=True)
        except (KeyError, ValueError):
            pass
        IOLoop.clear_instance()
        IOLoop.clear_current()


original_config = copy.deepcopy(dask.config.config)


def reset_config():
    dask.config.config.clear()
    dask.config.config.update(copy.deepcopy(original_config))


def nodebug(func):
    """
    A decorator to disable debug facilities during timing-sensitive tests.
    Warning: this doesn't affect already created IOLoops.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        old_asyncio_debug = os.environ.get("PYTHONASYNCIODEBUG")
        if old_asyncio_debug is not None:
            del os.environ["PYTHONASYNCIODEBUG"]
        try:
            return func(*args, **kwargs)
        finally:
            if old_asyncio_debug is not None:
                os.environ["PYTHONASYNCIODEBUG"] = old_asyncio_debug

    return wrapped


def nodebug_setup_module(module):
    """
    A setup_module() that you can install in a test module to disable
    debug facilities.
    """
    module._old_asyncio_debug = os.environ.get("PYTHONASYNCIODEBUG")
    if module._old_asyncio_debug is not None:
        del os.environ["PYTHONASYNCIODEBUG"]


def nodebug_teardown_module(module):
    """
    A teardown_module() that you can install in a test module to reenable
    debug facilities.
    """
    if module._old_asyncio_debug is not None:
        os.environ["PYTHONASYNCIODEBUG"] = module._old_asyncio_debug


def inc(x):
    return x + 1


def dec(x):
    return x - 1


def mul(x, y):
    return x * y


def div(x, y):
    return x / y


def deep(n):
    if n > 0:
        return deep(n - 1)
    else:
        return True


def throws(x):
    raise RuntimeError("hello!")


def double(x):
    return x * 2


def slowinc(x, delay=0.02):
    sleep(delay)
    return x + 1


def slowdec(x, delay=0.02):
    sleep(delay)
    return x - 1


def slowdouble(x, delay=0.02):
    sleep(delay)
    return 2 * x


def randominc(x, scale=1):
    from random import random

    sleep(random() * scale)
    return x + 1


def slowadd(x, y, delay=0.02):
    sleep(delay)
    return x + y


def slowsum(seq, delay=0.02):
    sleep(delay)
    return sum(seq)


def slowidentity(*args, **kwargs):
    delay = kwargs.get("delay", 0.02)
    sleep(delay)
    if len(args) == 1:
        return args[0]
    else:
        return args


class _UnhashableCallable:
    # FIXME https://github.com/python/mypy/issues/4266
    __hash__ = None  # type: ignore

    def __call__(self, x):
        return x + 1


def run_for(duration, timer=time):
    """
    Burn CPU for *duration* seconds.
    """
    deadline = timer() + duration
    while timer() <= deadline:
        pass


# This dict grows at every varying() invocation
_varying_dict: defaultdict[str, int] = defaultdict(int)
_varying_key_gen = count()


class _ModuleSlot:
    def __init__(self, modname, slotname):
        self.modname = modname
        self.slotname = slotname

    def get(self):
        return getattr(sys.modules[self.modname], self.slotname)


def varying(items):
    """
    Return a function that returns a result (or raises an exception)
    from *items* at each call.
    """
    # cloudpickle would serialize the *values* of all globals
    # used by *func* below, so we can't use `global <something>`.
    # Instead look up the module by name to get the original namespace
    # and not a copy.
    slot = _ModuleSlot(__name__, "_varying_dict")
    key = next(_varying_key_gen)

    def func():
        dct = slot.get()
        i = dct[key]
        if i == len(items):
            raise IndexError
        else:
            x = items[i]
            dct[key] = i + 1
            if isinstance(x, Exception):
                raise x
            else:
                return x

    return func


def map_varying(itemslists):
    """
    Like *varying*, but return the full specification for a map() call
    on multiple items lists.
    """

    def apply(func, *args, **kwargs):
        return func(*args, **kwargs)

    return apply, list(map(varying, itemslists))


async def geninc(x, delay=0.02):
    await asyncio.sleep(delay)
    return x + 1


async def asyncinc(x, delay=0.02):
    await asyncio.sleep(delay)
    return x + 1


_readone_queues: dict[Any, asyncio.Queue] = {}


async def readone(comm):
    """
    Read one message at a time from a comm that reads lists of
    messages.
    """
    try:
        q = _readone_queues[comm]
    except KeyError:
        q = _readone_queues[comm] = asyncio.Queue()

        async def background_read():
            while True:
                try:
                    messages = await comm.read()
                except CommClosedError:
                    break
                for msg in messages:
                    q.put_nowait(msg)
            q.put_nowait(None)
            del _readone_queues[comm]

        background_read()

    msg = await q.get()
    if msg is None:
        raise CommClosedError
    else:
        return msg


def _run_and_close_tornado(async_fn, /, *args, **kwargs):
    tornado_loop = None

    async def inner_fn():
        nonlocal tornado_loop
        tornado_loop = IOLoop.current()
        return await async_fn(*args, **kwargs)

    try:
        return asyncio.run(inner_fn())
    finally:
        tornado_loop.close(all_fds=True)


def run_scheduler(q, nputs, config, port=0, **kwargs):
    with dask.config.set(config):

        async def _():
            try:
                scheduler = await Scheduler(
                    validate=True, host="127.0.0.1", port=port, **kwargs
                )
            except Exception as exc:
                for i in range(nputs):
                    q.put(exc)
            else:
                for i in range(nputs):
                    q.put(scheduler.address)
                await scheduler.finished()

        _run_and_close_tornado(_)


def run_worker(q, scheduler_q, config, **kwargs):
    with dask.config.set(config):
        from distributed import Worker

        reset_logger_locks()
        with log_errors():
            scheduler_addr = scheduler_q.get()

            async def _():
                pid = os.getpid()
                try:
                    worker = await Worker(scheduler_addr, validate=True, **kwargs)
                except Exception as exc:
                    q.put((pid, exc))
                else:
                    q.put((pid, worker.address))
                    await worker.finished()

            # Scheduler might've failed
            if isinstance(scheduler_addr, str):
                _run_and_close_tornado(_)


@log_errors
def run_nanny(q, scheduler_q, config, **kwargs):
    with dask.config.set(config):
        scheduler_addr = scheduler_q.get()

        async def _():
            pid = os.getpid()
            try:
                worker = await Nanny(scheduler_addr, validate=True, **kwargs)
            except Exception as exc:
                q.put((pid, exc))
            else:
                q.put((pid, worker.address))
                await worker.finished()

        # Scheduler might've failed
        if isinstance(scheduler_addr, str):
            _run_and_close_tornado(_)


@contextmanager
def check_active_rpc(loop, active_rpc_timeout=1):
    warnings.warn(
        "check_active_rpc is deprecated - use gen_test()",
        DeprecationWarning,
        stacklevel=2,
    )
    active_before = set(rpc.active)
    yield
    # Some streams can take a bit of time to notice their peer
    # has closed, and keep a coroutine (*) waiting for a CommClosedError
    # before calling close_rpc() after a CommClosedError.
    # This would happen especially if a non-localhost address is used,
    # as Nanny does.
    # (*) (example: gather_from_workers())

    def fail():
        pytest.fail(
            "some RPCs left active by test: %s" % (set(rpc.active) - active_before)
        )

    async def wait():
        await async_wait_for(
            lambda: len(set(rpc.active) - active_before) == 0,
            timeout=active_rpc_timeout,
            fail_func=fail,
        )

    loop.run_sync(wait)


@contextlib.asynccontextmanager
async def _acheck_active_rpc(active_rpc_timeout=1):
    active_before = set(rpc.active)
    yield
    # Some streams can take a bit of time to notice their peer
    # has closed, and keep a coroutine (*) waiting for a CommClosedError
    # before calling close_rpc() after a CommClosedError.
    # This would happen especially if a non-localhost address is used,
    # as Nanny does.
    # (*) (example: gather_from_workers())

    def fail():
        pytest.fail(
            "some RPCs left active by test: %s" % (set(rpc.active) - active_before)
        )

    await async_wait_for(
        lambda: len(set(rpc.active) - active_before) == 0,
        timeout=active_rpc_timeout,
        fail_func=fail,
    )


@pytest.fixture
def cluster_fixture(loop):
    with cluster() as (scheduler, workers):
        yield (scheduler, workers)


@pytest.fixture
def s(cluster_fixture):
    scheduler, workers = cluster_fixture
    return scheduler


@pytest.fixture
def a(cluster_fixture):
    scheduler, workers = cluster_fixture
    return workers[0]


@pytest.fixture
def b(cluster_fixture):
    scheduler, workers = cluster_fixture
    return workers[1]


@pytest.fixture
def client(loop, cluster_fixture):
    scheduler, workers = cluster_fixture
    with Client(scheduler["address"], loop=loop) as client:
        yield client


# Compatibility. A lot of tests simply use `c` as fixture name
c = client


@pytest.fixture
def client_secondary(loop, cluster_fixture):
    scheduler, workers = cluster_fixture
    with Client(scheduler["address"], loop=loop) as client:
        yield client


@contextmanager
def tls_cluster_context(
    worker_kwargs=None, scheduler_kwargs=None, security=None, **kwargs
):
    security = security or tls_only_security()
    worker_kwargs = assoc(worker_kwargs or {}, "security", security)
    scheduler_kwargs = assoc(scheduler_kwargs or {}, "security", security)

    with cluster(
        worker_kwargs=worker_kwargs, scheduler_kwargs=scheduler_kwargs, **kwargs
    ) as (s, workers):
        yield s, workers


@pytest.fixture
def tls_cluster(loop, security):
    with tls_cluster_context(security=security) as (scheduler, workers):
        yield (scheduler, workers)


@pytest.fixture
def tls_client(tls_cluster, loop, security):
    s, workers = tls_cluster
    with Client(s["address"], security=security, loop=loop) as client:
        yield client


@pytest.fixture
def security():
    return tls_only_security()


def _terminate_join(proc):
    proc.terminate()
    proc.join()
    proc.close()


def _close_queue(q):
    q.close()
    q.join_thread()
    q._writer.close()  # https://bugs.python.org/issue42752


class _SafeTemporaryDirectory(tempfile.TemporaryDirectory):
    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        except (PermissionError, NotADirectoryError):
            # It appears that we either have a process still interacting with
            # the tmpdirs of the workers or that win process are not releasing
            # their lock in time. We are receiving PermissionErrors during
            # teardown
            # See also https://github.com/dask/distributed/pull/5825
            pass


@contextmanager
def cluster(
    nworkers=2,
    nanny=False,
    worker_kwargs={},
    active_rpc_timeout=10,
    disconnect_timeout=20,
    scheduler_kwargs={},
    config={},
):
    ws = weakref.WeakSet()
    enable_proctitle_on_children()

    with check_process_leak(check=True), check_instances(), _reconfigure(config):
        if nanny:
            _run_worker = run_nanny
        else:
            _run_worker = run_worker

        with contextlib.ExitStack() as stack:
            # The scheduler queue will receive the scheduler's address
            scheduler_q = get_mp_context().Queue()
            stack.callback(_close_queue, scheduler_q)

            # Launch scheduler
            scheduler = get_mp_context().Process(
                name="Dask cluster test: Scheduler",
                target=run_scheduler,
                args=(scheduler_q, nworkers + 1, config),
                kwargs=scheduler_kwargs,
                daemon=True,
            )
            ws.add(scheduler)
            scheduler.start()
            stack.callback(_terminate_join, scheduler)

            # Launch workers
            workers_by_pid = {}
            q = get_mp_context().Queue()
            stack.callback(_close_queue, q)
            for _ in range(nworkers):
                tmpdirname = stack.enter_context(
                    _SafeTemporaryDirectory(prefix="_dask_test_worker")
                )
                kwargs = merge(
                    {
                        "nthreads": 1,
                        "local_directory": tmpdirname,
                        "memory_limit": system.MEMORY_LIMIT,
                    },
                    worker_kwargs,
                )
                proc = get_mp_context().Process(
                    name="Dask cluster test: Worker",
                    target=_run_worker,
                    args=(q, scheduler_q, config),
                    kwargs=kwargs,
                )
                ws.add(proc)
                proc.start()
                stack.callback(_terminate_join, proc)
                workers_by_pid[proc.pid] = {"proc": proc}

            saddr_or_exception = scheduler_q.get()
            if isinstance(saddr_or_exception, Exception):
                raise saddr_or_exception
            saddr = saddr_or_exception

            for _ in range(nworkers):
                pid, addr_or_exception = q.get()
                if isinstance(addr_or_exception, Exception):
                    raise addr_or_exception
                workers_by_pid[pid]["address"] = addr_or_exception

            start = time()
            try:
                try:
                    security = scheduler_kwargs["security"]
                    rpc_kwargs = {
                        "connection_args": security.get_connection_args("client")
                    }
                except KeyError:
                    rpc_kwargs = {}

                async def wait_for_workers():
                    async with rpc(saddr, **rpc_kwargs) as s:
                        while True:
                            nthreads = await s.ncores_running()
                            if len(nthreads) == nworkers:
                                break
                            if time() - start > 5:
                                raise Exception("Timeout on cluster creation")

                _run_and_close_tornado(wait_for_workers)

                # avoid sending processes down to function
                yield {"address": saddr}, [
                    {"address": w["address"], "proc": weakref.ref(w["proc"])}
                    for w in workers_by_pid.values()
                ]
            finally:

                async def close():
                    logger.debug("Closing out test cluster")
                    alive_workers = [
                        w["address"]
                        for w in workers_by_pid.values()
                        if w["proc"].is_alive()
                    ]
                    await disconnect_all(
                        alive_workers,
                        timeout=disconnect_timeout,
                        rpc_kwargs=rpc_kwargs,
                    )
                    if scheduler.is_alive():
                        await disconnect(
                            saddr, timeout=disconnect_timeout, rpc_kwargs=rpc_kwargs
                        )

                _run_and_close_tornado(close)
        try:
            client = default_client()
        except ValueError:
            pass
        else:
            client.close()


async def disconnect(addr, timeout=3, rpc_kwargs=None):
    rpc_kwargs = rpc_kwargs or {}

    async def do_disconnect():
        async with rpc(addr, **rpc_kwargs) as w:
            # If the worker was killed hard (e.g. sigterm) during test runtime,
            # we do not know at this point and may not be able to connect
            with suppress(EnvironmentError, CommClosedError):
                # Do not request a reply since comms will be closed by the
                # worker before a reply can be made and we will always trigger
                # the timeout
                await w.terminate(reply=False)

    await asyncio.wait_for(do_disconnect(), timeout=timeout)


async def disconnect_all(addresses, timeout=3, rpc_kwargs=None):
    await asyncio.gather(*(disconnect(addr, timeout, rpc_kwargs) for addr in addresses))


def gen_test(
    timeout: float = _TEST_TIMEOUT,
    clean_kwargs: dict[str, Any] = {},
) -> Callable[[Callable], Callable]:
    """Coroutine test

    @pytest.mark.parametrize("param", [1, 2, 3])
    @gen_test(timeout=5)
    async def test_foo(param)
        await ... # use tornado coroutines


    @gen_test(timeout=5)
    async def test_foo():
        await ...  # use tornado coroutines
    """
    assert timeout, (
        "timeout should always be set and it should be smaller than the global one from"
        "pytest-timeout"
    )
    if is_debugging():
        timeout = 3600

    async def async_fn_outer(async_fn, /, *args, **kwargs):
        async with _acheck_active_rpc():
            return await asyncio.wait_for(
                asyncio.create_task(async_fn(*args, **kwargs)), timeout
            )

    def _(func):
        @functools.wraps(func)
        @clean(**clean_kwargs)
        def test_func(*args, **kwargs):
            if not iscoroutinefunction(func):
                raise RuntimeError("gen_test only works for coroutine functions.")

            return _run_and_close_tornado(async_fn_outer, func, *args, **kwargs)

        # Patch the signature so pytest can inject fixtures
        test_func.__signature__ = inspect.signature(func)
        return test_func

    return _


async def start_cluster(
    nthreads: list[tuple[str, int] | tuple[str, int, dict]],
    scheduler_addr: str,
    security: Security | dict[str, Any] | None = None,
    Worker: type[ServerNode] = Worker,
    scheduler_kwargs: dict[str, Any] = {},
    worker_kwargs: dict[str, Any] = {},
) -> tuple[Scheduler, list[ServerNode]]:
    s = await Scheduler(
        validate=True,
        security=security,
        port=0,
        host=scheduler_addr,
        **scheduler_kwargs,
    )

    workers = [
        Worker(
            s.address,
            nthreads=ncore[1],
            name=i,
            security=security,
            validate=True,
            host=ncore[0],
            **(
                merge(worker_kwargs, ncore[2])  # type: ignore
                if len(ncore) > 2
                else worker_kwargs
            ),
        )
        for i, ncore in enumerate(nthreads)
    ]

    await asyncio.gather(*workers)

    start = time()
    while (
        len(s.workers) < len(nthreads)
        or any(ws.status != Status.running for ws in s.workers.values())
        or any(comm.comm is None for comm in s.stream_comms.values())
    ):
        await asyncio.sleep(0.01)
        if time() > start + 30:
            await asyncio.gather(*(w.close(timeout=1) for w in workers))
            await s.close()
            check_invalid_worker_transitions(s)
            check_invalid_task_states(s)
            check_worker_fail_hard(s)
            raise TimeoutError("Cluster creation timeout")
    return s, workers


def check_invalid_worker_transitions(s: Scheduler) -> None:
    if not s.events.get("invalid-worker-transition"):
        return

    for timestamp, msg in s.events["invalid-worker-transition"]:
        worker = msg.pop("worker")
        print("Worker:", worker)
        print(InvalidTransition(**msg))

    raise ValueError(
        "Invalid worker transitions found", len(s.events["invalid-worker-transition"])
    )


def check_invalid_task_states(s: Scheduler) -> None:
    if not s.events.get("invalid-worker-task-state"):
        return

    for timestamp, msg in s.events["invalid-worker-task-state"]:
        print("Worker:", msg["worker"])
        print("State:", msg["state"])
        for line in msg["story"]:
            print(line)

    raise ValueError("Invalid worker task state")


def check_worker_fail_hard(s: Scheduler) -> None:
    if not s.events.get("worker-fail-hard"):
        return

    for timestamp, msg in s.events["worker-fail-hard"]:
        msg = msg.copy()
        worker = msg.pop("worker")
        msg["exception"] = deserialize(msg["exception"].header, msg["exception"].frames)
        msg["traceback"] = deserialize(msg["traceback"].header, msg["traceback"].frames)
        print("Failed worker", worker)
        _, exc, tb = clean_exception(**msg)
        assert exc
        raise exc.with_traceback(tb)


async def end_cluster(s, workers):
    logger.debug("Closing out test cluster")

    async def end_worker(w):
        with suppress(asyncio.TimeoutError, CommClosedError, EnvironmentError):
            await w.close()

    await asyncio.gather(*(end_worker(w) for w in workers))
    await s.close()  # wait until scheduler stops completely
    s.stop()
    check_invalid_worker_transitions(s)
    check_invalid_task_states(s)
    check_worker_fail_hard(s)


def gen_cluster(
    nthreads: list[tuple[str, int] | tuple[str, int, dict]] = [
        ("127.0.0.1", 1),
        ("127.0.0.1", 2),
    ],
    scheduler: str = "127.0.0.1",
    timeout: float = _TEST_TIMEOUT,
    security: Security | dict[str, Any] | None = None,
    Worker: type[ServerNode] = Worker,
    client: bool = False,
    scheduler_kwargs: dict[str, Any] = {},
    worker_kwargs: dict[str, Any] = {},
    client_kwargs: dict[str, Any] = {},
    active_rpc_timeout: float = 1,
    config: dict[str, Any] = {},
    clean_kwargs: dict[str, Any] = {},
    allow_unclosed: bool = False,
    cluster_dump_directory: str | Literal[False] = "test_cluster_dump",
) -> Callable[[Callable], Callable]:
    from distributed import Client

    """ Coroutine test with small cluster

    @gen_cluster()
    async def test_foo(scheduler, worker1, worker2):
        await ...  # use tornado coroutines

    @pytest.mark.parametrize("param", [1, 2, 3])
    @gen_cluster()
    async def test_foo(scheduler, worker1, worker2, param):
        await ...  # use tornado coroutines

    @gen_cluster()
    async def test_foo(scheduler, worker1, worker2, pytest_fixture_a, pytest_fixture_b):
        await ...  # use tornado coroutines

    See also:
        start
        end
    """
    assert timeout, (
        "timeout should always be set and it should be smaller than the global one from"
        "pytest-timeout"
    )
    if is_debugging():
        timeout = 3600

    scheduler_kwargs = merge(
        dict(
            dashboard=False,
            dashboard_address=":0",
            transition_counter_max=50_000,
        ),
        scheduler_kwargs,
    )
    worker_kwargs = merge(
        dict(
            memory_limit=system.MEMORY_LIMIT,
            death_timeout=15,
            transition_counter_max=50_000,
        ),
        worker_kwargs,
    )

    def _(func):
        if not iscoroutinefunction(func):
            raise RuntimeError("gen_cluster only works for coroutine functions.")

        @functools.wraps(func)
        @clean(**clean_kwargs)
        def test_func(*outer_args, **kwargs):
            async def async_fn():
                result = None
                with _SafeTemporaryDirectory() as tmpdir:
                    config2 = merge({"temporary-directory": tmpdir}, config)
                    with dask.config.set(config2):
                        workers = []
                        s = False

                        for _ in range(60):
                            try:
                                s, ws = await start_cluster(
                                    nthreads,
                                    scheduler,
                                    security=security,
                                    Worker=Worker,
                                    scheduler_kwargs=scheduler_kwargs,
                                    worker_kwargs=worker_kwargs,
                                )
                            except Exception as e:
                                logger.error(
                                    "Failed to start gen_cluster: "
                                    f"{e.__class__.__name__}: {e}; retrying",
                                    exc_info=True,
                                )
                                await asyncio.sleep(1)
                            else:
                                workers[:] = ws
                                args = [s] + workers
                                break
                        if s is False:
                            raise Exception("Could not start cluster")
                        if client:
                            c = await Client(
                                s.address,
                                security=security,
                                asynchronous=True,
                                **client_kwargs,
                            )
                            args = [c] + args

                        try:
                            coro = func(*args, *outer_args, **kwargs)
                            task = asyncio.create_task(coro)
                            coro2 = asyncio.wait_for(asyncio.shield(task), timeout)
                            result = await coro2
                            validate_state(s, *workers)

                        except asyncio.TimeoutError:
                            assert task
                            buffer = io.StringIO()
                            # This stack indicates where the coro/test is suspended
                            task.print_stack(file=buffer)

                            if cluster_dump_directory:
                                await dump_cluster_state(
                                    s,
                                    ws,
                                    output_dir=cluster_dump_directory,
                                    func_name=func.__name__,
                                )

                            task.cancel()
                            while not task.cancelled():
                                await asyncio.sleep(0.01)

                            # Hopefully, the hang has been caused by inconsistent
                            # state, which should be much more meaningful than the
                            # timeout
                            validate_state(s, *workers)

                            # Remove as much of the traceback as possible; it's
                            # uninteresting boilerplate from utils_test and asyncio
                            # and not from the code being tested.
                            raise asyncio.TimeoutError(
                                f"Test timeout after {timeout}s.\n"
                                "========== Test stack trace starts here ==========\n"
                                f"{buffer.getvalue()}"
                            ) from None

                        except pytest.xfail.Exception:
                            raise

                        except Exception:
                            if cluster_dump_directory and not has_pytestmark(
                                test_func, "xfail"
                            ):
                                await dump_cluster_state(
                                    s,
                                    ws,
                                    output_dir=cluster_dump_directory,
                                    func_name=func.__name__,
                                )
                            raise

                        finally:
                            if client and c.status not in ("closing", "closed"):
                                await c._close(fast=s.status == Status.closed)
                            await end_cluster(s, workers)
                            await asyncio.wait_for(cleanup_global_workers(), 1)

                        try:
                            c = await default_client()
                        except ValueError:
                            pass
                        else:
                            await c._close(fast=True)

                        def get_unclosed():
                            return [c for c in Comm._instances if not c.closed()] + [
                                c
                                for c in _global_clients.values()
                                if c.status != "closed"
                            ]

                        try:
                            start = time()
                            while time() < start + 60:
                                gc.collect()
                                if not get_unclosed():
                                    break
                                await asyncio.sleep(0.05)
                            else:
                                if allow_unclosed:
                                    print(f"Unclosed Comms: {get_unclosed()}")
                                else:
                                    raise RuntimeError("Unclosed Comms", get_unclosed())
                        finally:
                            Comm._instances.clear()
                            _global_clients.clear()

                            for w in workers:
                                if getattr(w, "data", None):
                                    try:
                                        w.data.clear()
                                    except OSError:
                                        # zict backends can fail if their storage directory
                                        # was already removed
                                        pass

                        return result

            async def async_fn_outer():
                async with _acheck_active_rpc(active_rpc_timeout=active_rpc_timeout):
                    if timeout:
                        return await asyncio.wait_for(async_fn(), timeout=timeout * 2)
                    return await async_fn()

            return _run_and_close_tornado(async_fn_outer)

        # Patch the signature so pytest can inject fixtures
        orig_sig = inspect.signature(func)
        args = [None] * (1 + len(nthreads))  # scheduler, *workers
        if client:
            args.insert(0, None)

        bound = orig_sig.bind_partial(*args)
        test_func.__signature__ = orig_sig.replace(
            parameters=[
                p
                for name, p in orig_sig.parameters.items()
                if name not in bound.arguments
            ]
        )

        return test_func

    return _


async def dump_cluster_state(
    s: Scheduler, ws: list[ServerNode], output_dir: str, func_name: str
) -> None:
    """A variant of Client.dump_cluster_state, which does not rely on any of the below
    to work:

    - Having a client at all
    - Client->Scheduler comms
    - Scheduler->Worker comms (unless using Nannies)
    """
    scheduler_info = s._to_dict()
    workers_info: dict[str, Any]
    versions_info = version_module.get_versions()

    if not ws or isinstance(ws[0], Worker):
        workers_info = {w.address: w._to_dict() for w in ws}
    else:
        workers_info = await s.broadcast(msg={"op": "dump_state"}, on_error="return")
        workers_info = {
            k: repr(v) if isinstance(v, Exception) else v
            for k, v in workers_info.items()
        }

    state = {
        "scheduler": scheduler_info,
        "workers": workers_info,
        "versions": versions_info,
    }
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, func_name) + ".yaml"
    with open(fname, "w") as fh:
        yaml.safe_dump(state, fh)  # Automatically convert tuples to lists
    print(f"Dumped cluster state to {fname}")


def validate_state(*servers: Scheduler | Worker | Nanny) -> None:
    """Run validate_state() on the Scheduler and all the Workers of the cluster.
    Excludes workers wrapped by Nannies and workers manually started by the test.
    """
    for s in servers:
        if isinstance(s, Scheduler) and s.validate:
            s.validate_state()
        elif isinstance(s, Worker) and s.state.validate:
            s.validate_state()


def raises(func, exc=Exception):
    try:
        func()
        return False
    except exc:
        return True


def _terminate_process(proc: subprocess.Popen, terminate_timeout: float) -> None:
    if proc.poll() is None:
        if sys.platform.startswith("win"):
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
        try:
            proc.communicate(timeout=terminate_timeout)
        finally:
            # Make sure we don't leave the process lingering around
            with suppress(OSError):
                proc.kill()


@contextmanager
def popen(
    args: list[str],
    capture_output: bool = False,
    terminate_timeout: float = 30,
    kill_timeout: float = 10,
    **kwargs: Any,
) -> Iterator[subprocess.Popen[bytes]]:
    """Start a shell command in a subprocess.
    Yields a subprocess.Popen object.

    On exit, the subprocess is terminated.

    Parameters
    ----------
    args: list[str]
        Command line arguments
    capture_output: bool, default False
        Set to True if you need to read output from the subprocess.
        Stdout and stderr will both be piped to ``proc.stdout``.

        If False, the subprocess will write to stdout/stderr normally.

        When True, the test could deadlock if the stdout pipe's buffer gets full
        (Linux default is 65536 bytes; macOS and Windows may be smaller).
        Therefore, you may need to periodically read from ``proc.stdout``, or
        use ``proc.communicate``. All the deadlock warnings apply from
        https://docs.python.org/3/library/subprocess.html#subprocess.Popen.stderr.

        Note that ``proc.communicate`` is called automatically when the
        contextmanager exits. Calling code must not call ``proc.communicate``
        in a separate thread, since it's not thread-safe.

        When captured, the stdout/stderr of the process is always printed
        when the process exits for easier test debugging.
    terminate_timeout: optional, default 30
        When the contextmanager exits, SIGINT is sent to the subprocess.
        ``terminate_timeout`` sets how many seconds to wait for the subprocess
        to terminate after that. If the timeout expires, SIGKILL is sent to
        the subprocess (which cannot be blocked); see ``kill_timeout``.
        If this timeout expires, `subprocess.TimeoutExpired` is raised.
    kill_timeout: optional, default 10
        When the contextmanger exits, if the subprocess does not shut down
        after ``terminate_timeout`` seconds in response to SIGINT, SIGKILL
        is sent to the subprocess (which cannot be blocked). ``kill_timeout``
        controls how long to wait after SIGKILL to join the process.
        If this timeout expires, `subprocess.TimeoutExpired` is raised.
    kwargs: optional
        optional arguments to subprocess.Popen
    """
    with new_config_file(dask.config.config):
        if capture_output:
            kwargs["stdout"] = subprocess.PIPE
            kwargs["stderr"] = subprocess.STDOUT
        if sys.platform.startswith("win"):
            # Allow using CTRL_C_EVENT / CTRL_BREAK_EVENT
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

        environ = kwargs.pop("env", os.environ.copy())
        args = list(args)
        if sys.platform.startswith("win"):
            args[0] = os.path.join(sys.prefix, "Scripts", args[0])
        else:
            args[0] = os.path.join(
                os.environ.get("DESTDIR", "") + sys.prefix, "bin", args[0]
            )
        with subprocess.Popen(args, env=environ, **kwargs) as proc:
            try:
                yield proc
            finally:
                try:
                    _terminate_process(proc, terminate_timeout)
                finally:
                    out, err = proc.communicate(timeout=kill_timeout)
                    if out:
                        print(
                            f"------ stdout: returncode {proc.returncode}, {args} ------"
                        )
                        print(out.decode() if isinstance(out, bytes) else out)
                    if err:
                        print(
                            f"------ stderr: returncode {proc.returncode}, {args} ------"
                        )
                        print(err.decode() if isinstance(err, bytes) else err)


def wait_for(predicate, timeout, fail_func=None, period=0.05):
    deadline = time() + timeout
    while not predicate():
        sleep(period)
        if time() > deadline:
            if fail_func is not None:
                fail_func()
            pytest.fail(f"condition not reached until {timeout} seconds")


async def async_wait_for(predicate, timeout, fail_func=None, period=0.05):
    deadline = time() + timeout
    while not predicate():
        await asyncio.sleep(period)
        if time() > deadline:
            if fail_func is not None:
                fail_func()
            pytest.fail(f"condition not reached until {timeout} seconds")


@memoize
def has_ipv6():
    """
    Return whether IPv6 is locally functional.  This doesn't guarantee IPv6
    is properly configured outside of localhost.
    """
    if os.getenv("DISABLE_IPV6") == "1":
        return False

    serv = cli = None
    try:
        serv = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        serv.bind(("::", 0))
        serv.listen(5)
        cli = socket.create_connection(serv.getsockname()[:2])
        return True
    except OSError:
        return False
    finally:
        if cli is not None:
            cli.close()
        if serv is not None:
            serv.close()


if has_ipv6():

    def requires_ipv6(test_func):
        return test_func

else:
    requires_ipv6 = pytest.mark.skip("ipv6 required")


async def assert_can_connect(addr, timeout=0.5, **kwargs):
    """
    Check that it is possible to connect to the distributed *addr*
    within the given *timeout*.
    """
    comm = await connect(addr, timeout=timeout, **kwargs)
    comm.abort()


async def assert_cannot_connect(
    addr, timeout=0.5, exception_class=EnvironmentError, **kwargs
):
    """
    Check that it is impossible to connect to the distributed *addr*
    within the given *timeout*.
    """
    with pytest.raises(exception_class):
        comm = await connect(addr, timeout=timeout, **kwargs)
        comm.abort()


async def assert_can_connect_from_everywhere_4_6(port, protocol="tcp", **kwargs):
    """
    Check that the local *port* is reachable from all IPv4 and IPv6 addresses.
    """
    futures = [
        assert_can_connect("%s://127.0.0.1:%d" % (protocol, port), **kwargs),
        assert_can_connect("%s://%s:%d" % (protocol, get_ip(), port), **kwargs),
    ]
    if has_ipv6():
        futures += [
            assert_can_connect("%s://[::1]:%d" % (protocol, port), **kwargs),
            assert_can_connect("%s://[%s]:%d" % (protocol, get_ipv6(), port), **kwargs),
        ]
    await asyncio.gather(*futures)


async def assert_can_connect_from_everywhere_4(port, protocol="tcp", **kwargs):
    """
    Check that the local *port* is reachable from all IPv4 addresses.
    """
    futures = [
        assert_can_connect("%s://127.0.0.1:%d" % (protocol, port), **kwargs),
        assert_can_connect("%s://%s:%d" % (protocol, get_ip(), port), **kwargs),
    ]
    if has_ipv6():
        futures += [
            assert_cannot_connect("%s://[::1]:%d" % (protocol, port), **kwargs),
            assert_cannot_connect(
                "%s://[%s]:%d" % (protocol, get_ipv6(), port), **kwargs
            ),
        ]
    await asyncio.gather(*futures)


async def assert_can_connect_locally_4(port, **kwargs):
    """
    Check that the local *port* is only reachable from local IPv4 addresses.
    """
    futures = [assert_can_connect("tcp://127.0.0.1:%d" % port, **kwargs)]
    if get_ip() != "127.0.0.1":  # No outside IPv4 connectivity?
        futures += [assert_cannot_connect("tcp://%s:%d" % (get_ip(), port), **kwargs)]
    if has_ipv6():
        futures += [
            assert_cannot_connect("tcp://[::1]:%d" % port, **kwargs),
            assert_cannot_connect("tcp://[%s]:%d" % (get_ipv6(), port), **kwargs),
        ]
    await asyncio.gather(*futures)


async def assert_can_connect_from_everywhere_6(port, **kwargs):
    """
    Check that the local *port* is reachable from all IPv6 addresses.
    """
    assert has_ipv6()
    futures = [
        assert_cannot_connect("tcp://127.0.0.1:%d" % port, **kwargs),
        assert_cannot_connect("tcp://%s:%d" % (get_ip(), port), **kwargs),
        assert_can_connect("tcp://[::1]:%d" % port, **kwargs),
        assert_can_connect("tcp://[%s]:%d" % (get_ipv6(), port), **kwargs),
    ]
    await asyncio.gather(*futures)


async def assert_can_connect_locally_6(port, **kwargs):
    """
    Check that the local *port* is only reachable from local IPv6 addresses.
    """
    assert has_ipv6()
    futures = [
        assert_cannot_connect("tcp://127.0.0.1:%d" % port, **kwargs),
        assert_cannot_connect("tcp://%s:%d" % (get_ip(), port), **kwargs),
        assert_can_connect("tcp://[::1]:%d" % port, **kwargs),
    ]
    if get_ipv6() != "::1":  # No outside IPv6 connectivity?
        futures += [
            assert_cannot_connect("tcp://[%s]:%d" % (get_ipv6(), port), **kwargs)
        ]
    await asyncio.gather(*futures)


@contextmanager
def captured_logger(logger, level=logging.INFO, propagate=None):
    """Capture output from the given Logger."""
    if isinstance(logger, str):
        logger = logging.getLogger(logger)
    orig_level = logger.level
    orig_handlers = logger.handlers[:]
    if propagate is not None:
        orig_propagate = logger.propagate
        logger.propagate = propagate
    sio = io.StringIO()
    logger.handlers[:] = [logging.StreamHandler(sio)]
    logger.setLevel(level)
    try:
        yield sio
    finally:
        logger.handlers[:] = orig_handlers
        logger.setLevel(orig_level)
        if propagate is not None:
            logger.propagate = orig_propagate


@contextmanager
def captured_handler(handler):
    """Capture output from the given logging.StreamHandler."""
    assert isinstance(handler, logging.StreamHandler)
    orig_stream = handler.stream
    handler.stream = io.StringIO()
    try:
        yield handler.stream
    finally:
        handler.stream = orig_stream


@contextmanager
def new_config(new_config):
    """
    Temporarily change configuration dictionary.
    """
    from distributed.config import defaults

    config = dask.config.config
    orig_config = copy.deepcopy(config)
    try:
        config.clear()
        config.update(copy.deepcopy(defaults))
        dask.config.update(config, new_config)
        initialize_logging(config)
        yield
    finally:
        config.clear()
        config.update(orig_config)
        initialize_logging(config)


@contextmanager
def new_environment(changes):
    saved_environ = os.environ.copy()
    os.environ.update(changes)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved_environ)


@contextmanager
def new_config_file(c):
    """
    Temporarily change configuration file to match dictionary *c*.
    """
    import yaml

    old_file = os.environ.get("DASK_CONFIG")
    fd, path = tempfile.mkstemp(prefix="dask-config")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(yaml.dump(c))
        os.environ["DASK_CONFIG"] = path
        try:
            yield
        finally:
            if old_file:
                os.environ["DASK_CONFIG"] = old_file
            else:
                del os.environ["DASK_CONFIG"]
    finally:
        os.remove(path)


certs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "tests"))


def get_cert(filename):
    """
    Get the path to one of the test TLS certificates.
    """
    path = os.path.join(certs_dir, filename)
    assert os.path.exists(path), path
    return path


def tls_config():
    """
    A functional TLS configuration with our test certs.
    """
    ca_file = get_cert("tls-ca-cert.pem")
    keycert = get_cert("tls-key-cert.pem")

    return {
        "distributed": {
            "comm": {
                "tls": {
                    "ca-file": ca_file,
                    "client": {"cert": keycert},
                    "scheduler": {"cert": keycert},
                    "worker": {"cert": keycert},
                }
            }
        }
    }


def tls_only_config():
    """
    A functional TLS configuration with our test certs, disallowing
    plain TCP communications.
    """
    c = tls_config()
    c["distributed"]["comm"]["require-encryption"] = True
    return c


def tls_security():
    """
    A Security object with proper TLS configuration.
    """
    with new_config(tls_config()):
        sec = Security()
    return sec


def tls_only_security():
    """
    A Security object with proper TLS configuration and disallowing plain
    TCP communications.
    """
    with new_config(tls_only_config()):
        sec = Security()
    assert sec.require_encryption
    return sec


def get_server_ssl_context(
    certfile="tls-cert.pem", keyfile="tls-key.pem", ca_file="tls-ca-cert.pem"
):
    ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH, cafile=get_cert(ca_file))
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_cert_chain(get_cert(certfile), get_cert(keyfile))
    return ctx


def get_client_ssl_context(
    certfile="tls-cert.pem", keyfile="tls-key.pem", ca_file="tls-ca-cert.pem"
):
    ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=get_cert(ca_file))
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_REQUIRED
    ctx.load_cert_chain(get_cert(certfile), get_cert(keyfile))
    return ctx


def bump_rlimit(limit, desired):
    resource = pytest.importorskip("resource")
    try:
        soft, hard = resource.getrlimit(limit)
        if soft < desired:
            resource.setrlimit(limit, (desired, max(hard, desired)))
    except Exception as e:
        pytest.skip(f"rlimit too low ({soft}) and can't be increased: {e}")


def gen_tls_cluster(**kwargs):
    kwargs.setdefault("nthreads", [("tls://127.0.0.1", 1), ("tls://127.0.0.1", 2)])
    return gen_cluster(
        scheduler="tls://127.0.0.1", security=tls_only_security(), **kwargs
    )


@contextmanager
def save_sys_modules():
    old_modules = sys.modules
    old_path = sys.path
    try:
        yield
    finally:
        for i, elem in enumerate(sys.path):
            if elem not in old_path:
                del sys.path[i]
        for elem in sys.modules.keys():
            if elem not in old_modules:
                del sys.modules[elem]


@contextmanager
def check_thread_leak():
    """Context manager to ensure we haven't leaked any threads"""
    active_threads_start = threading.enumerate()

    yield

    start = time()
    while True:
        bad_threads = [
            thread
            for thread in threading.enumerate()
            if thread not in active_threads_start
            # FIXME this looks like a genuine leak that needs fixing
            and "watch message queue" not in thread.name
        ]
        if not bad_threads:
            break
        else:
            sleep(0.01)
        if time() > start + 5:
            # Raise an error with information about leaked threads
            from distributed import profile

            bad_thread = bad_threads[0]
            call_stacks = profile.call_stack(sys._current_frames()[bad_thread.ident])
            assert False, (bad_thread, call_stacks)


def wait_active_children(timeout: float) -> list[multiprocessing.Process]:
    """Wait until timeout for get_mp_context().active_children() to terminate.
    Return list of active subprocesses after the timeout expired.
    """
    t0 = time()
    while True:
        # Do not sample the subprocesses once at the beginning with
        # `for proc in get_mp_context().active_children: ...`, assume instead that new
        # children processes may be spawned before the timeout expires.
        children = get_mp_context().active_children()
        if not children:
            return []
        join_timeout = timeout - time() + t0
        if join_timeout <= 0:
            return children
        children[0].join(timeout=join_timeout)


def term_or_kill_active_children(timeout: float) -> None:
    """Send SIGTERM to get_mp_context().active_children(), wait up to 3 seconds for processes
    to die, then send SIGKILL to the survivors
    """
    children = get_mp_context().active_children()
    for proc in children:
        proc.terminate()

    children = wait_active_children(timeout=timeout)
    for proc in children:
        proc.kill()

    children = wait_active_children(timeout=30)
    if children:  # pragma: nocover
        logger.warning("Leaked unkillable children processes: %s", children)
        # It should be impossible to ignore SIGKILL on Linux/MacOSX
        assert WINDOWS


@contextmanager
def check_process_leak(
    check: bool = True, check_timeout: float = 40, term_timeout: float = 3
) -> Iterator[None]:
    """Terminate any currently-running subprocesses at both the beginning and end of this context

    Parameters
    ----------
    check : bool, optional
        If True, raise AssertionError if any processes survive at the exit
    check_timeout: float, optional
        Wait up to these many seconds for subprocesses to terminate before failing
    term_timeout: float, optional
        After sending SIGTERM to a subprocess, wait up to these many seconds before
        sending SIGKILL
    """
    term_or_kill_active_children(timeout=term_timeout)
    try:
        yield
        if check:
            children = wait_active_children(timeout=check_timeout)
            assert not children, f"Test leaked subprocesses: {children}"
    finally:
        term_or_kill_active_children(timeout=term_timeout)


@contextmanager
def check_instances():
    Client._instances.clear()
    Worker._instances.clear()
    Scheduler._instances.clear()
    SpecCluster._instances.clear()
    Worker._initialized_clients.clear()
    SchedulerTaskState._instances.clear()
    WorkerTaskState._instances.clear()
    Nanny._instances.clear()
    _global_clients.clear()
    Comm._instances.clear()

    yield

    start = time()
    while set(_global_clients):
        sleep(0.1)
        assert time() < start + 10

    _global_clients.clear()

    for w in Worker._instances:
        with suppress(RuntimeError):  # closed IOLoop
            w.loop.add_callback(w.close, executor_wait=False)
            if w.status in WORKER_ANY_RUNNING:
                w.loop.add_callback(w.close)
    Worker._instances.clear()

    start = time()
    while any(c.status != "closed" for c in Worker._initialized_clients):
        sleep(0.1)
        assert time() < start + 10
    Worker._initialized_clients.clear()

    for i in range(5):
        if all(c.closed() for c in Comm._instances):
            break
        else:
            sleep(0.1)
    else:
        L = [c for c in Comm._instances if not c.closed()]
        Comm._instances.clear()
        raise ValueError("Unclosed Comms", L)

    assert all(
        n.status in {Status.closed, Status.init, Status.failed}
        for n in Nanny._instances
    ), {n: n.status for n in Nanny._instances}

    # assert not list(SpecCluster._instances)  # TODO
    assert all(c.status == Status.closed for c in SpecCluster._instances), list(
        SpecCluster._instances
    )
    SpecCluster._instances.clear()

    Nanny._instances.clear()
    DequeHandler.clear_all_instances()


def _get_default_config():
    # Define this in case a function calls clean as a ctxmanager only
    return {
        "distributed.comm.timeouts.connect": "5s",
        "distributed.admin.tick.interval": "500 ms",
        "distributed.worker.profile.enabled": False,
        "distributed.scheduler.dashboard.default-port": ":0",
    }


@pytest.fixture(scope="module")
def default_dashboard_port():
    """The default dashboard port that will be set in the dask
    configuration for the test run when using the cleanup fixture
    """
    return ":0"


@pytest.fixture(scope="module")
def default_dask_config(default_dashboard_port):
    config = _get_default_config()
    config["distributed.scheduler.dashboard.default-port"] = default_dashboard_port
    return config


@contextmanager
def _reconfigure(config=None):
    reset_config()
    if not config:
        config = _get_default_config()
    with dask.config.set(config):
        # Restore default logging levels
        # XXX use pytest hooks/fixtures instead?
        for name, level in logging_levels.items():
            logging.getLogger(name).setLevel(level)

        yield


@contextmanager
def clean(threads=True, instances=True, processes=True, config=None):
    asyncio.set_event_loop(None)
    with check_thread_leak() if threads else nullcontext():
        with check_process_leak(check=processes):
            with check_instances() if instances else nullcontext():
                with _reconfigure(config=config):
                    yield


@pytest.fixture
def cleanup(default_dask_config):
    with clean(config=default_dask_config):
        yield


class TaskStateMetadataPlugin(WorkerPlugin):
    """WorkPlugin to populate TaskState.metadata"""

    def setup(self, worker):
        self.tasks = worker.state.tasks

    def transition(self, key, start, finish, **kwargs):
        ts = self.tasks[key]

        if start == "ready" and finish == "executing":
            ts.metadata["start_time"] = time()
        elif start == "executing" and finish == "memory":
            ts.metadata["stop_time"] = time()


class LockedComm(TCP):
    def __init__(self, comm, read_event, read_queue, write_event, write_queue):
        self.write_event = write_event
        self.write_queue = write_queue
        self.read_event = read_event
        self.read_queue = read_queue
        self.comm = comm
        assert isinstance(comm, TCP)

    def __getattr__(self, name):
        return getattr(self.comm, name)

    async def write(self, msg, serializers=None, on_error="message"):
        if self.write_queue:
            await self.write_queue.put((self.comm.peer_address, msg))
        if self.write_event:
            await self.write_event.wait()
        return await self.comm.write(msg, serializers=serializers, on_error=on_error)

    async def read(self, deserializers=None):
        msg = await self.comm.read(deserializers=deserializers)
        if self.read_queue:
            await self.read_queue.put((self.comm.peer_address, msg))
        if self.read_event:
            await self.read_event.wait()
        return msg

    async def close(self):
        await self.comm.close()


class _LockedCommPool(ConnectionPool):
    """A ConnectionPool wrapper to intercept network traffic between servers

    This wrapper can be attached to a running server to intercept outgoing read or write requests in test environments.

    Examples
    --------
    >>> w = await Worker(...)
    >>> read_event = asyncio.Event()
    >>> read_queue = asyncio.Queue()
    >>> w.rpc = _LockedCommPool(
            w.rpc,
            read_event=read_event,
            read_queue=read_queue,
        )
    # It might be necessary to remove all existing comms
    # if the wrapped pool has been used before
    >>> w.rpc.remove(remote_address)

    >>> async def ping_pong():
            return await w.rpc(remote_address).ping()
    >>> with pytest.raises(asyncio.TimeoutError):
    >>>     await asyncio.wait_for(ping_pong(), 0.01)
    >>> read_event.set()
    >>> await ping_pong()
    """

    def __init__(
        self, pool, read_event=None, read_queue=None, write_event=None, write_queue=None
    ):
        self.write_event = write_event
        self.write_queue = write_queue
        self.read_event = read_event
        self.read_queue = read_queue
        self.pool = pool

    def __getattr__(self, name):
        return getattr(self.pool, name)

    async def connect(self, *args, **kwargs):
        comm = await self.pool.connect(*args, **kwargs)
        return LockedComm(
            comm, self.read_event, self.read_queue, self.write_event, self.write_queue
        )

    async def close(self):
        await self.pool.close()


def xfail_ssl_issue5601():
    """Work around https://github.com/dask/distributed/issues/5601 where any test that
    inits Security.temporary() crashes on MacOS GitHub Actions CI
    """
    pytest.importorskip("cryptography")
    try:
        Security.temporary()
    except ImportError:
        if MACOS:
            pytest.xfail(reason="distributed#5601")
        raise


def assert_valid_story(story, ordered_timestamps=True):
    """Test that a story is well formed.

    Parameters
    ----------
    story: list[tuple]
        Output of Worker.story
    ordered_timestamps: bool, optional
        If False, timestamps are not required to be monotically increasing.
        Useful for asserting stories composed from the scheduler and
        multiple workers
    """

    now = time()
    prev_ts = 0.0
    for ev in story:
        try:
            assert len(ev) > 2, "too short"
            assert isinstance(ev, tuple), "not a tuple"
            assert isinstance(ev[-2], str) and ev[-2], "stimulus_id not a string"
            assert isinstance(ev[-1], float), "Timestamp is not a float"
            if ordered_timestamps:
                assert prev_ts <= ev[-1], "Timestamps are not monotonically ascending"
            # Timestamps are within the last hour. It's been observed that a
            # timestamp generated in a Nanny process can be a few milliseconds
            # in the future.
            assert now - 3600 < ev[-1] <= now + 1, "Timestamps is too old"
            prev_ts = ev[-1]
        except AssertionError as err:
            raise AssertionError(
                f"Malformed story event: {ev}\nProblem: {err}.\nin story:\n{_format_story(story)}"
            )


def assert_story(
    story: list[tuple],
    expect: list[tuple],
    *,
    strict: bool = False,
    ordered_timestamps: bool = True,
) -> None:
    """Test the output of ``Worker.story`` or ``Scheduler.story``

    Warning
    =======

    Tests with overly verbose stories introduce maintenance cost and should
    therefore be used with caution. This should only be used for very specific
    unit tests where the exact order of events is crucial and there is no other
    practical way to assert or observe what happened.
    A typical use case involves testing for race conditions where subtle changes
    of event ordering would cause harm.

    Parameters
    ==========
    story: list[tuple]
        Output of Worker.story
    expect: list[tuple]
        Expected events.
        The expected events either need to be exact matches or are allowed to
        not provide a stimulus_id and timestamp.
        e.g.
        `("log", "entry", "stim-id-9876", 1234)`
        is equivalent to
        `("log", "entry")`

        story (the last two fields are always the stimulus_id and the timestamp).

        Elements of the expect tuples can be

        - callables, which accept a single element of the event tuple as argument and
          return True for match and False for no match;
        - arbitrary objects, which are compared with a == b

        e.g.
        .. code-block:: python

            expect=[
                ("x", "missing", "fetch", "fetch", {}),
                ("gather-dependencies", worker_addr, lambda set_: "x" in set_),
            ]

    strict: bool, optional
        If True, the story must contain exactly as many events as expect.
        If False (the default), the story may contain more events than expect; extra
        events are ignored.
    ordered_timestamps: bool, optional
        If False, timestamps are not required to be monotically increasing.
        Useful for asserting stories composed from the scheduler and
        multiple workers
    """
    assert_valid_story(story, ordered_timestamps=ordered_timestamps)

    def _valid_event(event, ev_expect):
        return len(event) == len(ev_expect) and all(
            ex(ev) if callable(ex) else ev == ex for ev, ex in zip(event, ev_expect)
        )

    try:
        if strict and len(story) != len(expect):
            raise StopIteration()
        story_it = iter(story)
        for ev_expect in expect:
            while True:
                event = next(story_it)

                if (
                    _valid_event(event, ev_expect)
                    # Ignore (stimulus_id, timestamp)
                    or _valid_event(event[:-2], ev_expect)
                ):
                    break
    except StopIteration:
        raise AssertionError(
            f"assert_story({strict=}) failed\n"
            f"story:\n{_format_story(story)}\n"
            f"expect:\n{_format_story(expect)}"
        ) from None


def _format_story(story: list[tuple]) -> str:
    if not story:
        return "(empty story)"
    return "- " + "\n- ".join(str(ev) for ev in story)


class BrokenComm(Comm):
    peer_address = ""
    local_address = ""

    def close(self):
        pass

    def closed(self):
        return True

    def abort(self):
        pass

    def read(self, deserializers=None):
        raise OSError()

    def write(self, msg, serializers=None, on_error=None):
        raise OSError()


def has_pytestmark(test_func: Callable, name: str) -> bool:
    """Return True if the test function is marked by the given @pytest.mark.<name>;
    False otherwise.

    FIXME doesn't work with individually marked parameters inside
          @pytest.mark.parametrize
    """
    marks = getattr(test_func, "pytestmark", [])
    return any(mark.name == name for mark in marks)


@contextmanager
def raises_with_cause(
    expected_exception: type[BaseException] | tuple[type[BaseException], ...],
    match: str | None,
    expected_cause: type[BaseException] | tuple[type[BaseException], ...],
    match_cause: str | None,
) -> Generator[None, None, None]:
    """Contextmanager to assert that a certain exception with cause was raised

    Parameters
    ----------
    exc_type:
    """
    with pytest.raises(expected_exception, match=match) as exc_info:
        yield

    exc = exc_info.value
    assert exc.__cause__
    if not isinstance(exc.__cause__, expected_cause):
        raise exc
    if match_cause:
        assert re.search(
            match_cause, str(exc.__cause__)
        ), f"Pattern ``{match_cause}`` not found in ``{exc.__cause__}``"


def ucx_exception_handler(loop, context):
    """UCX exception handler for `ucx_loop` during test.

    Prints the exception and its message.

    Parameters
    ----------
    loop: object
        Reference to the running event loop
    context: dict
        Dictionary containing exception details.
    """
    msg = context.get("exception", context["message"])
    print(msg)


# Let's make sure that UCX gets time to cancel
# progress tasks before closing the event loop.
@pytest.fixture(scope="function")
def ucx_loop():
    """Allows UCX to cancel progress tasks before closing event loop.

    When UCX tasks are not completed in time (e.g., by unexpected Endpoint
    closure), clean up tasks before closing the event loop to prevent unwanted
    errors from being raised.
    """
    ucp = pytest.importorskip("ucp")

    loop = asyncio.new_event_loop()
    loop.set_exception_handler(ucx_exception_handler)
    ucp.reset()
    yield loop
    ucp.reset()
    loop.close()


def wait_for_log_line(
    match: bytes, stream: IO[bytes] | None, max_lines: int | None = 10
) -> bytes:
    """
    Read lines from an IO stream until the match is found, and return the matching line.

    Prints each line to test stdout for easier debugging of failures.
    """
    assert stream
    i = 0
    while True:
        if max_lines is not None and i == max_lines:
            raise AssertionError(
                f"{match!r} not found in {max_lines} log lines. See test stdout for details."
            )
        line = stream.readline()
        print(line)
        if match in line:
            return line
        i += 1


class BlockedGatherDep(Worker):
    """A Worker that sets event `in_gather_dep` the first time it enters the gather_dep
    method and then does not initiate any comms, thus leaving the task(s) in flight
    indefinitely, until the test sets `block_gather_dep`

    Example
    -------
    .. code-block:: python

        @gen_test()
        async def test1(s, a, b):
            async with BlockedGatherDep(s.address) as x:
                # [do something to cause x to fetch data from a or b]
                await x.in_gather_dep.wait()
                # [do something that must happen while the tasks are in flight]
                x.block_gather_dep.set()
                # [from this moment on, x is a regular worker]

    See also
    --------
    BlockedGetData
    """

    def __init__(self, *args, **kwargs):
        self.in_gather_dep = asyncio.Event()
        self.block_gather_dep = asyncio.Event()
        super().__init__(*args, **kwargs)

    async def gather_dep(self, *args, **kwargs):
        self.in_gather_dep.set()
        await self.block_gather_dep.wait()
        return await super().gather_dep(*args, **kwargs)


class BlockedGetData(Worker):
    """A Worker that sets event `in_get_data` the first time it enters the get_data
    method and then does not answer the comms, thus leaving the task(s) in flight
    indefinitely, until the test sets `block_get_data`

    See also
    --------
    BlockedGatherDep
    """

    def __init__(self, *args, **kwargs):
        self.in_get_data = asyncio.Event()
        self.block_get_data = asyncio.Event()
        super().__init__(*args, **kwargs)

    async def get_data(self, comm, *args, **kwargs):
        self.in_get_data.set()
        await self.block_get_data.wait()
        return await super().get_data(comm, *args, **kwargs)


@contextmanager
def freeze_data_fetching(w: Worker, *, jump_start: bool = False) -> Iterator[None]:
    """Prevent any task from transitioning from fetch to flight on the worker while
    inside the context, simulating a situation where the worker's network comms are
    saturated.

    This is not the same as setting the worker to Status=paused, which would also
    inform the Scheduler and prevent further tasks to be enqueued on the worker.

    Parameters
    ----------
    w: Worker
        The Worker on which tasks will not transition from fetch to flight
    jump_start: bool
        If False, tasks will remain in fetch state after exiting the context, until
        something else triggers ensure_communicating.
        If True, trigger ensure_communicating on exit; this simulates e.g. an unrelated
        worker moving out of in_flight_workers.
    """
    old_out_connections = w.state.total_out_connections
    old_comm_threshold = w.state.comm_threshold_bytes
    w.state.total_out_connections = 0
    w.state.comm_threshold_bytes = 0
    yield
    w.state.total_out_connections = old_out_connections
    w.state.comm_threshold_bytes = old_comm_threshold
    if jump_start:
        w.status = Status.paused
        w.status = Status.running


@contextmanager
def freeze_batched_send(bcomm: BatchedSend) -> Iterator[LockedComm]:
    """
    Contextmanager blocking writes to a `BatchedSend` from sending over the network.

    The returned `LockedComm` object can be used for control flow and inspection via its
    ``read_event``, ``read_queue``, ``write_event``, and ``write_queue`` attributes.

    On exit, any writes that were blocked are un-blocked, and the original comm of the
    `BatchedSend` is restored.
    """
    assert not bcomm.closed()
    assert bcomm.comm
    assert not bcomm.comm.closed()
    orig_comm = bcomm.comm

    write_event = asyncio.Event()
    write_queue: asyncio.Queue = asyncio.Queue()

    bcomm.comm = locked_comm = LockedComm(
        orig_comm, None, None, write_event, write_queue
    )

    try:
        yield locked_comm
    finally:
        write_event.set()
        bcomm.comm = orig_comm


async def wait_for_state(
    key: str, state: str, dask_worker: Worker | Scheduler, *, interval: float = 0.01
) -> None:
    """Wait for a task to appear on a Worker or on the Scheduler and to be in a specific
    state.
    """
    if isinstance(dask_worker, Worker):
        tasks = dask_worker.state.tasks
    elif isinstance(dask_worker, Scheduler):
        tasks = dask_worker.tasks
    else:
        raise TypeError(dask_worker)  # pragma: nocover

    try:
        while key not in tasks or tasks[key].state != state:
            await asyncio.sleep(interval)
    except (asyncio.CancelledError, asyncio.TimeoutError):
        if key in tasks:
            msg = (
                f"tasks[{key}].state={tasks[key].state!r} on {dask_worker.address}; "
                f"expected {state=}"
            )
        else:
            msg = f"tasks[{key}] not found on {dask_worker.address}"
        # 99% of the times this is triggered by @gen_cluster timeout, so raising the
        # message as an exception wouldn't work.
        print(msg)
        raise


async def wait_for_stimulus(
    type_: type[StateMachineEvent] | tuple[type[StateMachineEvent], ...],
    dask_worker: Worker,
    *,
    interval: float = 0.01,
    **matches: Any,
) -> StateMachineEvent:
    """Wait for a specific stimulus to appear in the log of the WorkerState."""
    log = dask_worker.state.stimulus_log
    last_ev = None
    while True:
        if log and log[-1] is not last_ev:
            last_ev = log[-1]
            for ev in log:
                if not isinstance(ev, type_):
                    continue
                if all(getattr(ev, k) == v for k, v in matches.items()):
                    return ev
        await asyncio.sleep(interval)


@pytest.fixture
def ws():
    state = WorkerState(address="127.0.0.1:1", transition_counter_max=50_000)
    yield state
    state.validate_state()


@pytest.fixture()
def name_of_test(request):
    return f"{request.node.nodeid}"


@pytest.fixture()
def requires_default_ports(name_of_test):
    start = time()

    @contextmanager
    def _bind_port(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("", port))
            s.listen(1)
            yield s

    default_ports = [8786]

    while time() - start < _TEST_TIMEOUT:
        try:
            with contextlib.ExitStack() as stack:
                for port in default_ports:
                    stack.enter_context(_bind_port(port=port))
                break
        except OSError as err:
            if err.errno == errno.EADDRINUSE:
                print(
                    f"Address already in use. Waiting before running test {name_of_test}"
                )
                sleep(1)
                continue
    else:
        raise TimeoutError(f"Default ports didn't open up in time for {name_of_test}")

    yield

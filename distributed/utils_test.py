from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import functools
import gc
import inspect
import io
import logging
import logging.config
import multiprocessing
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import uuid
import warnings
import weakref
from collections import defaultdict
from collections.abc import Callable
from contextlib import contextmanager, nullcontext, suppress
from glob import glob
from itertools import count
from time import sleep
from typing import Any, Literal
from unittest.mock import patch

from distributed.compatibility import MACOS
from distributed.scheduler import Scheduler

try:
    import ssl
except ImportError:
    ssl = None  # type: ignore

import pytest
import yaml
from tlz import assoc, memoize, merge
from tornado import gen
from tornado.ioloop import IOLoop

import dask

from distributed import system
from distributed import versions as version_module
from distributed.client import Client, _global_clients, default_client
from distributed.comm import Comm
from distributed.comm.tcp import TCP, BaseTCPConnector
from distributed.compatibility import WINDOWS
from distributed.config import initialize_logging
from distributed.core import CommClosedError, ConnectionPool, Status, connect, rpc
from distributed.deploy import SpecCluster
from distributed.diagnostics.plugin import WorkerPlugin
from distributed.metrics import time
from distributed.nanny import Nanny
from distributed.node import ServerNode
from distributed.proctitle import enable_proctitle_on_children
from distributed.security import Security
from distributed.utils import (
    DequeHandler,
    TimeoutError,
    _offload_executor,
    get_ip,
    get_ipv6,
    iscoroutinefunction,
    log_errors,
    mp_context,
    reset_logger_locks,
    sync,
)
from distributed.worker import Worker

try:
    import dask.array  # register config
except ImportError:
    pass

# https://docs.github.com/en/actions/learn-github-actions/environment-variables#default-environment-variables
ON_CI = bool(os.environ.get("CI", False))


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
        await worker.close(report=False, executor_wait=False)


@pytest.fixture
def loop():
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
            except TimeoutError:
                pass
            else:
                is_stopped.wait()


@pytest.fixture
def loop_in_thread():
    with pristine_loop() as loop:
        thread = threading.Thread(target=loop.start, name="test IOLoop")
        thread.daemon = True
        thread.start()
        loop_started = threading.Event()
        loop.add_callback(loop_started.set)
        loop_started.wait()
        yield loop
        loop.add_callback(loop.stop)
        thread.join(timeout=5)


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


@contextmanager
def mock_ipython():
    from unittest import mock

    from distributed._ipython_utils import remote_magic

    ip = mock.Mock()
    ip.user_ns = {}
    ip.kernel = None

    def get_ip():
        return ip

    with mock.patch("IPython.get_ipython", get_ip), mock.patch(
        "distributed._ipython_utils.get_ipython", get_ip
    ):
        yield ip
    # cleanup remote_magic client cache
    for kc in remote_magic._clients.values():
        kc.stop_channels()
    remote_magic._clients.clear()


original_config = copy.deepcopy(dask.config.config)


def reset_config():
    """Reset the global Dask config."""
    warnings.warn(
        "The function reset_config will be removed soon. Instead, consider using unitest.mock.patch.dict",
        FutureWarning,
    )
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


def run_scheduler(q, nputs, config, port=0, **kwargs):
    with dask.config.set(config):
        from distributed import Scheduler

        # On Python 2.7 and Unix, fork() is used to spawn child processes,
        # so avoid inheriting the parent's IO loop.
        with pristine_loop() as loop:

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

            try:
                loop.run_sync(_)
            finally:
                loop.close(all_fds=True)


def run_worker(q, scheduler_q, config, **kwargs):
    with dask.config.set(config):
        from distributed import Worker

        reset_logger_locks()
        with log_errors():
            with pristine_loop() as loop:
                scheduler_addr = scheduler_q.get()

                async def _():
                    try:
                        worker = await Worker(scheduler_addr, validate=True, **kwargs)
                    except Exception as exc:
                        q.put(exc)
                    else:
                        q.put(worker.address)
                        await worker.finished()

                # Scheduler might've failed
                if isinstance(scheduler_addr, str):
                    try:
                        loop.run_sync(_)
                    finally:
                        loop.close(all_fds=True)


def run_nanny(q, scheduler_q, config, **kwargs):
    with dask.config.set(config):
        with log_errors():
            with pristine_loop() as loop:
                scheduler_addr = scheduler_q.get()

                async def _():
                    try:
                        worker = await Nanny(scheduler_addr, validate=True, **kwargs)
                    except Exception as exc:
                        q.put(exc)
                    else:
                        q.put(worker.address)
                        await worker.finished()

                # Scheduler might've failed
                if isinstance(scheduler_addr, str):
                    try:
                        loop.run_sync(_)
                    finally:
                        loop.close(all_fds=True)


@contextmanager
def check_active_rpc(loop, active_rpc_timeout=1):
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

    with clean(timeout=active_rpc_timeout, threads=False) as loop:
        with dask.config.set(config):
            if nanny:
                _run_worker = run_nanny
            else:
                _run_worker = run_worker

            # The scheduler queue will receive the scheduler's address
            scheduler_q = mp_context.Queue()

            # Launch scheduler
            scheduler = mp_context.Process(
                name="Dask cluster test: Scheduler",
                target=run_scheduler,
                args=(scheduler_q, nworkers + 1, config),
                kwargs=scheduler_kwargs,
            )
            ws.add(scheduler)
            scheduler.daemon = True
            scheduler.start()

            # Launch workers
            workers = []
            for i in range(nworkers):
                q = mp_context.Queue()
                fn = "_test_worker-%s" % uuid.uuid4()
                kwargs = merge(
                    {
                        "nthreads": 1,
                        "local_directory": fn,
                        "memory_limit": system.MEMORY_LIMIT,
                    },
                    worker_kwargs,
                )
                proc = mp_context.Process(
                    name="Dask cluster test: Worker",
                    target=_run_worker,
                    args=(q, scheduler_q, config),
                    kwargs=kwargs,
                )
                ws.add(proc)
                workers.append({"proc": proc, "queue": q, "dir": fn})

            for worker in workers:
                worker["proc"].start()
            saddr_or_exception = scheduler_q.get()
            if isinstance(saddr_or_exception, Exception):
                raise saddr_or_exception
            saddr = saddr_or_exception

            for worker in workers:
                addr_or_exception = worker["queue"].get()
                if isinstance(addr_or_exception, Exception):
                    raise addr_or_exception
                worker["address"] = addr_or_exception

            start = time()
            try:
                try:
                    security = scheduler_kwargs["security"]
                    rpc_kwargs = {
                        "connection_args": security.get_connection_args("client")
                    }
                except KeyError:
                    rpc_kwargs = {}

                with rpc(saddr, **rpc_kwargs) as s:
                    while True:
                        nthreads = loop.run_sync(s.ncores)
                        if len(nthreads) == nworkers:
                            break
                        if time() - start > 5:
                            raise Exception("Timeout on cluster creation")

                # avoid sending processes down to function
                yield {"address": saddr}, [
                    {"address": w["address"], "proc": weakref.ref(w["proc"])}
                    for w in workers
                ]
            finally:
                logger.debug("Closing out test cluster")

                loop.run_sync(
                    lambda: disconnect_all(
                        [w["address"] for w in workers],
                        timeout=disconnect_timeout,
                        rpc_kwargs=rpc_kwargs,
                    )
                )
                loop.run_sync(
                    lambda: disconnect(
                        saddr, timeout=disconnect_timeout, rpc_kwargs=rpc_kwargs
                    )
                )

                scheduler.terminate()
                scheduler_q.close()
                scheduler_q._reader.close()
                scheduler_q._writer.close()

                for w in workers:
                    w["proc"].terminate()
                    w["queue"].close()
                    w["queue"]._reader.close()
                    w["queue"]._writer.close()

                scheduler.join(2)
                del scheduler
                for proc in [w["proc"] for w in workers]:
                    proc.join(timeout=30)

                with suppress(UnboundLocalError):
                    del worker, w, proc
                del workers[:]

                for fn in glob("_test_worker-*"):
                    with suppress(OSError):
                        shutil.rmtree(fn)

            try:
                client = default_client()
            except ValueError:
                pass
            else:
                client.close()

        start = time()
        while any(proc.is_alive() for proc in ws):
            text = str(list(ws))
            sleep(0.2)
            assert time() < start + 5, ("Workers still around after five seconds", text)


async def disconnect(addr, timeout=3, rpc_kwargs=None):
    rpc_kwargs = rpc_kwargs or {}

    async def do_disconnect():
        with rpc(addr, **rpc_kwargs) as w:
            # If the worker was killed hard (e.g. sigterm) during test runtime,
            # we do not know at this point and may not be able to connect
            with suppress(EnvironmentError, CommClosedError):
                # Do not request a reply since comms will be closed by the
                # worker before a reply can be made and we will always trigger
                # the timeout
                await w.terminate(reply=False)

    with suppress(asyncio.TimeoutError):
        await asyncio.wait_for(do_disconnect(), timeout=timeout)


async def disconnect_all(addresses, timeout=3, rpc_kwargs=None):
    await asyncio.gather(*(disconnect(addr, timeout, rpc_kwargs) for addr in addresses))


def gen_test(timeout: float = _TEST_TIMEOUT) -> Callable[[Callable], Callable]:
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

    def _(func):
        def test_func(*args, **kwargs):
            with clean() as loop:
                injected_func = functools.partial(func, *args, **kwargs)
                if iscoroutinefunction(func):
                    cor = injected_func
                else:
                    cor = gen.coroutine(injected_func)

                loop.run_sync(cor, timeout=timeout)

        # Patch the signature so pytest can inject fixtures
        test_func.__signature__ = inspect.signature(func)
        return test_func

    return _


async def start_cluster(
    nthreads: list[tuple[str, int] | tuple[str, int, dict]],
    scheduler_addr: str,
    loop: IOLoop,
    security: Security | dict[str, Any] | None = None,
    Worker: type[ServerNode] = Worker,
    scheduler_kwargs: dict[str, Any] = {},
    worker_kwargs: dict[str, Any] = {},
) -> tuple[Scheduler, list[ServerNode]]:
    s = await Scheduler(
        loop=loop,
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
            loop=loop,
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
            await s.close(fast=True)
            raise TimeoutError("Cluster creation timeout")
    return s, workers


async def end_cluster(s, workers):
    logger.debug("Closing out test cluster")

    async def end_worker(w):
        with suppress(TimeoutError, CommClosedError, EnvironmentError):
            await w.close(report=False)

    await asyncio.gather(*(end_worker(w) for w in workers))
    await s.close()  # wait until scheduler stops completely
    s.stop()


def gen_cluster(
    nthreads: list[tuple[str, int] | tuple[str, int, dict]] = [
        ("127.0.0.1", 1),
        ("127.0.0.1", 2),
    ],
    scheduler="127.0.0.1",
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

    scheduler_kwargs = merge(
        {"dashboard": False, "dashboard_address": ":0"}, scheduler_kwargs
    )
    worker_kwargs = merge(
        {"memory_limit": system.MEMORY_LIMIT, "death_timeout": 15}, worker_kwargs
    )

    def _(func):
        if not iscoroutinefunction(func):
            raise RuntimeError("gen_cluster only works for coroutine functions.")

        @functools.wraps(func)
        def test_func(*outer_args, **kwargs):
            result = None
            workers = []
            with clean(timeout=active_rpc_timeout, **clean_kwargs) as loop:

                async def coro():
                    with dask.config.set(config):
                        s = False
                        for _ in range(60):
                            try:
                                s, ws = await start_cluster(
                                    nthreads,
                                    scheduler,
                                    loop,
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
                                loop=loop,
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
                            if s.validate:
                                s.validate_state()

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

                            # Remove as much of the traceback as possible; it's
                            # uninteresting boilerplate from utils_test and asyncio and
                            # not from the code being tested.
                            raise TimeoutError(
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

                        return result

                result = loop.run_sync(
                    coro, timeout=timeout * 2 if timeout else timeout
                )

            for w in workers:
                if getattr(w, "data", None):
                    try:
                        w.data.clear()
                    except OSError:
                        # zict backends can fail if their storage directory
                        # was already removed
                        pass

            return result

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


def raises(func, exc=Exception):
    try:
        func()
        return False
    except exc:
        return True


def _terminate_process(proc):
    if proc.poll() is None:
        if sys.platform.startswith("win"):
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.send_signal(signal.SIGINT)
        try:
            proc.wait(30)
        finally:
            # Make sure we don't leave the process lingering around
            with suppress(OSError):
                proc.kill()


@contextmanager
def popen(args: list[str], flush_output: bool = True, **kwargs):
    """Start a shell command in a subprocess.
    Yields a subprocess.Popen object.

    stderr is redirected to stdout.
    stdout is redirected to a pipe.

    Parameters
    ----------
    args: list[str]
        Command line arguments
    flush_output: bool, optional
        If True (the default), the stdout/stderr pipe is emptied while it is being
        filled. Set to False if you wish to read the output yourself. Note that setting
        this to False and then failing to periodically read from the pipe may result in
        a deadlock due to the pipe getting full.
    kwargs: optional
        optional arguments to subprocess.Popen
    """
    kwargs["stdout"] = subprocess.PIPE
    kwargs["stderr"] = subprocess.STDOUT
    if sys.platform.startswith("win"):
        # Allow using CTRL_C_EVENT / CTRL_BREAK_EVENT
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    dump_stdout = False

    args = list(args)
    if sys.platform.startswith("win"):
        args[0] = os.path.join(sys.prefix, "Scripts", args[0])
    else:
        args[0] = os.path.join(
            os.environ.get("DESTDIR", "") + sys.prefix, "bin", args[0]
        )
    proc = subprocess.Popen(args, **kwargs)

    if flush_output:
        ex = concurrent.futures.ThreadPoolExecutor(1)
        flush_future = ex.submit(proc.communicate)

    try:
        yield proc

    # asyncio.CancelledError is raised by @gen_test/@gen_cluster timeout
    except (Exception, asyncio.CancelledError):
        dump_stdout = True
        raise

    finally:
        try:
            _terminate_process(proc)
        finally:
            # XXX Also dump stdout if return code != 0 ?
            if flush_output:
                out, err = flush_future.result()
                ex.shutdown()
            else:
                out, err = proc.communicate()
            assert not err

            if dump_stdout:
                print("\n" + "-" * 27 + " Subprocess stdout/stderr" + "-" * 27)
                print(out.decode().rstrip())
                print("-" * 80)


def wait_for(predicate, timeout, fail_func=None, period=0.001):
    deadline = time() + timeout
    while not predicate():
        sleep(period)
        if time() > deadline:
            if fail_func is not None:
                fail_func()
            pytest.fail(f"condition not reached until {timeout} seconds")


async def async_wait_for(predicate, timeout, fail_func=None, period=0.001):
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
    # "TCP-Executor" threads are never stopped once they are started
    BaseTCPConnector.warmup()

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
    """Wait until timeout for mp_context.active_children() to terminate.
    Return list of active subprocesses after the timeout expired.
    """
    t0 = time()
    while True:
        # Do not sample the subprocesses once at the beginning with
        # `for proc in mp_context.active_children: ...`, assume instead that new
        # children processes may be spawned before the timeout expires.
        children = mp_context.active_children()
        if not children:
            return []
        join_timeout = timeout - time() + t0
        if join_timeout <= 0:
            return children
        children[0].join(timeout=join_timeout)


def term_or_kill_active_children(timeout: float) -> None:
    """Send SIGTERM to mp_context.active_children(), wait up to 3 seconds for processes
    to die, then send SIGKILL to the survivors
    """
    children = mp_context.active_children()
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
):
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
    # assert all(n.status == "closed" for n in Nanny._instances), {
    #     n: n.status for n in Nanny._instances
    # }
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
            w.loop.add_callback(w.close, report=False, executor_wait=False)
            if w.status in Status.ANY_RUNNING:
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
        n.status == Status.closed or n.status == Status.init for n in Nanny._instances
    ), {n: n.status for n in Nanny._instances}

    # assert not list(SpecCluster._instances)  # TODO
    assert all(c.status == Status.closed for c in SpecCluster._instances), list(
        SpecCluster._instances
    )
    SpecCluster._instances.clear()

    Nanny._instances.clear()
    DequeHandler.clear_all_instances()


@contextmanager
def clean(threads=not WINDOWS, instances=True, timeout=1, processes=True):
    with check_thread_leak() if threads else nullcontext():
        with pristine_loop() as loop:
            with check_process_leak(check=processes):
                with check_instances() if instances else nullcontext():
                    with check_active_rpc(loop, timeout):
                        with patch.dict(original_config, clear=True):
                            with dask.config.set(
                                {"distributed.comm.timeouts.connect": "300s"}
                            ):
                                # Restore default logging levels
                                # XXX use pytest hooks/fixtures instead?
                                for name, level in logging_levels.items():
                                    logging.getLogger(name).setLevel(level)

                                yield loop


@pytest.fixture
def cleanup():
    with clean():
        yield


class TaskStateMetadataPlugin(WorkerPlugin):
    """WorkPlugin to populate TaskState.metadata"""

    def setup(self, worker):
        self.worker = worker

    def transition(self, key, start, finish, **kwargs):
        ts = self.worker.tasks[key]

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
    >>> w.remove(remote_address)

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


def assert_worker_story(
    story: list[tuple], expect: list[tuple], *, strict: bool = False
) -> None:
    """Test the output of ``Worker.story``

    Parameters
    ==========
    story: list[tuple]
        Output of Worker.story
    expect: list[tuple]
        Expected events. Each expected event must contain exactly 2 less fields than the
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
    """
    now = time()
    prev_ts = 0.0
    for ev in story:
        try:
            assert len(ev) > 2
            assert isinstance(ev, tuple)
            assert isinstance(ev[-2], str) and ev[-2]  # stimulus_id
            assert isinstance(ev[-1], float)  # timestamp
            assert prev_ts <= ev[-1]  # Timestamps are monotonic ascending
            # Timestamps are within the last hour. It's been observed that a timestamp
            # generated in a Nanny process can be a few milliseconds in the future.
            assert now - 3600 < ev[-1] <= now + 1
            prev_ts = ev[-1]
        except AssertionError:
            raise AssertionError(
                f"Malformed story event: {ev}\nin story:\n{_format_story(story)}"
            )

    try:
        if strict and len(story) != len(expect):
            raise StopIteration()
        story_it = iter(story)
        for ev_expect in expect:
            while True:
                event = next(story_it)
                # Ignore (stimulus_id, timestamp)
                event = event[:-2]
                if len(event) == len(ev_expect) and all(
                    ex(ev) if callable(ex) else ev == ex
                    for ev, ex in zip(event, ev_expect)
                ):
                    break
    except StopIteration:
        raise AssertionError(
            f"assert_worker_story({strict=}) failed\n"
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

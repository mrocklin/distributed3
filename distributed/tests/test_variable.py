from __future__ import annotations

import asyncio
import pickle
import random
import sys
from datetime import timedelta
from time import sleep

import pytest

from distributed import Client, Nanny, TimeoutError, Variable, wait, worker_client
from distributed.compatibility import WINDOWS
from distributed.metrics import monotonic, time
from distributed.utils import open_port
from distributed.utils_test import captured_logger, div, gen_cluster, inc, popen


@gen_cluster(client=True)
async def test_variable(c, s, a, b):
    x = Variable("x")
    xx = Variable("x")
    assert x.client is c

    future = c.submit(inc, 1)

    assert not await x.is_set()
    await x.set(future)
    assert await x.is_set()
    future2 = await xx.get()
    assert future.key == future2.key

    del future, future2

    await asyncio.sleep(0.1)
    assert s.tasks  # future still present

    x.delete()
    assert not await x.is_set()

    start = time()
    while s.tasks:
        await asyncio.sleep(0.01)
        assert time() < start + 5


def test_variable_in_task(loop):
    port = open_port()
    # Ensure that we can create a Variable inside a task on a
    # worker in a separate Python process than the client
    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "scheduler",
            "--no-dashboard",
            "--port",
            str(port),
        ]
    ):
        with popen([sys.executable, "-m", "dask", "worker", f"127.0.0.1:{port}"]):
            with Client(f"tcp://127.0.0.1:{port}", loop=loop) as c:
                c.wait_for_workers(1)

                x = Variable("x")
                x.set(123)

                def foo():
                    y = Variable("x")
                    return y.get()

                result = c.submit(foo).result()
                assert result == 123


@gen_cluster(client=True)
async def test_delete_unset_variable(c, s, a, b):
    x = Variable()
    assert x.client is c
    with captured_logger("distributed.utils") as logger:
        x.delete()
        await c.close()
    text = logger.getvalue()
    assert "KeyError" not in text


@gen_cluster(client=True)
async def test_queue_with_data(c, s, a, b):
    x = Variable("x")
    xx = Variable("x")
    assert x.client is c

    await x.set((1, "hello"))
    data = await xx.get()

    assert data == (1, "hello")


def test_sync(client):
    future = client.submit(lambda x: x + 1, 10)
    x = Variable("x")
    xx = Variable("x")
    x.set(future)
    future2 = xx.get()

    assert future2.result() == 11


@gen_cluster()
async def test_hold_futures(s, a, b):
    async with Client(s.address, asynchronous=True) as c1:
        future = c1.submit(lambda x: x + 1, 10)
        x1 = Variable("x")
        await x1.set(future)
        del x1

    await asyncio.sleep(0.1)

    async with Client(s.address, asynchronous=True) as c2:
        x2 = Variable("x")
        future2 = await x2.get()
        result = await future2

        assert result == 11


@gen_cluster(client=True)
async def test_timeout(c, s, a, b):
    v = Variable("v")

    start = monotonic()
    with pytest.raises(TimeoutError):
        await v.get(timeout="200ms")
    stop = monotonic()

    if WINDOWS:  # timing is weird with asyncio and Windows
        assert 0.1 < stop - start < 2.0
    else:
        assert 0.2 < stop - start < 2.0

    with pytest.raises(TimeoutError):
        await v.get(timeout=timedelta(milliseconds=10))


def test_timeout_sync(client):
    v = Variable("v")
    start = time()
    with pytest.raises(TimeoutError):
        v.get(timeout=0.2)
    stop = time()

    if WINDOWS:
        assert 0.1 < stop - start < 2.0
    else:
        assert 0.2 < stop - start < 2.0

    with pytest.raises(TimeoutError):
        v.get(timeout=0.01)


@gen_cluster(client=True)
async def test_cleanup(c, s, a, b):
    v = Variable("v")
    vv = Variable("v")

    x = c.submit(lambda x: x + 1, 10)
    y = c.submit(lambda x: x + 1, 20)
    x_key = x.key

    await v.set(x)
    del x
    await asyncio.sleep(0.1)

    t_future = xx = asyncio.ensure_future(vv._get())
    await asyncio.sleep(0)
    asyncio.ensure_future(v.set(y))

    future = await t_future
    assert future.key == x_key
    result = await future
    assert result == 11


def test_pickleable(client):
    v = Variable("v")

    def f(x):
        v.set(x + 1)

    client.submit(f, 10).result()
    assert v.get() == 11


@gen_cluster(client=True)
async def test_timeout_get(c, s, a, b):
    v = Variable("v")

    tornado_future = v.get()

    vv = Variable("v")
    await vv.set(1)

    result = await tornado_future
    assert result == 1


@pytest.mark.slow
@gen_cluster(client=True, nthreads=[("127.0.0.1", 2)] * 5, Worker=Nanny, timeout=60)
async def test_race(c, s, *workers):
    NITERS = 50

    def f(i):
        with worker_client() as c:
            v = Variable("x", client=c)
            for _ in range(NITERS):
                future = v.get()
                x = future.result()
                y = c.submit(inc, x)
                v.set(y)
                sleep(0.01 * random.random())
            result = v.get().result()
            sleep(0.1)  # allow fire-and-forget messages to clear
            return result

    v = Variable("x", client=c)
    x = await c.scatter(1)
    await v.set(x)

    futures = c.map(f, range(15))
    results = await c.gather(futures)

    while "variable-x" in s.tasks:
        await asyncio.sleep(0.01)


@gen_cluster(client=True)
async def test_Future_knows_status_immediately(c, s, a, b):
    x = await c.scatter(123)
    v = Variable("x")
    await v.set(x)

    async with Client(s.address, asynchronous=True) as c2:
        v2 = Variable("x", client=c2)
        future = await v2.get()
        assert future.status == "finished"

        x = c.submit(div, 1, 0)
        await wait(x)
        await v.set(x)

        future2 = await v2.get()
        assert future2.status == "error"
        with pytest.raises(ZeroDivisionError):
            await future2

        start = time()
        while True:  # we learn about the true error eventually
            try:
                await future2
            except ZeroDivisionError:
                break
            except Exception:
                assert time() < start + 5
                await asyncio.sleep(0.05)


@gen_cluster(client=True)
async def test_erred_future(c, s, a, b):
    future = c.submit(div, 1, 0)
    var = Variable()
    await var.set(future)
    await asyncio.sleep(0.1)
    future2 = await var.get()
    with pytest.raises(ZeroDivisionError):
        await future2.result()

    exc = await future2.exception()
    assert isinstance(exc, ZeroDivisionError)


def test_future_erred_sync(client):
    future = client.submit(div, 1, 0)
    var = Variable()
    var.set(future)

    sleep(0.1)

    future2 = var.get()

    with pytest.raises(ZeroDivisionError):
        future2.result()


@gen_cluster(client=True)
async def test_variables_do_not_leak_client(c, s, a, b):
    # https://github.com/dask/distributed/issues/3899
    clients_pre = set(s.clients)

    # setup variable with future
    x = Variable("x")
    future = c.submit(inc, 1)
    await x.set(future)

    # complete teardown
    x.delete()

    start = time()
    while set(s.clients) != clients_pre:
        await asyncio.sleep(0.01)
        assert time() < start + 5


@gen_cluster(nthreads=[])
async def test_unpickle_without_client(s):
    """Ensure that the object properly pickle roundtrips even if no client, worker, etc. is active in the given context.

    This typically happens if the object is being deserialized on the scheduler.
    """
    async with Client(s.address, asynchronous=True) as c:
        obj = Variable("foo")
        pickled = pickle.dumps(obj)

    # We do not want to initialize a client during unpickling
    with pytest.raises(ValueError):
        Client.current()

    obj2 = pickle.loads(pickled)

    with pytest.raises(ValueError):
        Client.current()

    assert obj2.client is None

    with pytest.raises(RuntimeError, match="not properly initialized"):
        await obj2.set(42)

    async with Client(s.address, asynchronous=True):
        obj3 = pickle.loads(pickled)
        await obj3.set(42)
        assert await obj3.get() == 42


@gen_cluster(client=True, nthreads=[])
async def test_set_cancelled_future(c, s):
    x = c.submit(inc, 1)
    await x.cancel()
    v = Variable("x")
    with pytest.raises(TimeoutError):
        await v.set(x, timeout="5ms")

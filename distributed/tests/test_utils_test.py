import asyncio
import pathlib
import socket
import threading
from contextlib import contextmanager
from time import sleep

import pytest
from tornado import gen
from tornado.ioloop import IOLoop

from distributed import Client, Nanny, Scheduler, Worker, config, default_client
from distributed.core import rpc
from distributed.metrics import time
from distributed.utils import get_ip
from distributed.utils_test import (
    cluster,
    gen_cluster,
    gen_test,
    inc,
    new_config,
    start_cluster,
    tls_only_security,
    wait_for_port,
)


def test_bare_cluster(loop):
    with cluster(nworkers=10) as (s, _):
        pass


def test_cluster(loop):
    with cluster() as (s, [a, b]):
        with rpc(s["address"]) as s:
            ident = loop.run_sync(s.identity)
            assert ident["type"] == "Scheduler"
            assert len(ident["workers"]) == 2


@gen_cluster(client=True)
async def test_gen_cluster(c, s, a, b):
    assert isinstance(c, Client)
    assert isinstance(s, Scheduler)
    for w in [a, b]:
        assert isinstance(w, Worker)
    assert s.nthreads == {w.address: w.nthreads for w in [a, b]}
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
        import dask

        assert dask.config.get("distributed.comm.timeouts.connect") == "1s"
        assert dask.config.get("new.config.value") == "foo"
        return dask.config

    await c.run(assert_config)
    await c.run_on_scheduler(assert_config)


@gen_cluster(client=True)
def test_gen_cluster_legacy_implicit(c, s, a, b):
    assert isinstance(c, Client)
    assert isinstance(s, Scheduler)
    for w in [a, b]:
        assert isinstance(w, Worker)
    assert s.nthreads == {w.address: w.nthreads for w in [a, b]}
    assert (yield c.submit(lambda: 123)) == 123


@gen_cluster(client=True)
@gen.coroutine
def test_gen_cluster_legacy_explicit(c, s, a, b):
    assert isinstance(c, Client)
    assert isinstance(s, Scheduler)
    for w in [a, b]:
        assert isinstance(w, Worker)
    assert s.nthreads == {w.address: w.nthreads for w in [a, b]}
    assert (yield c.submit(lambda: 123)) == 123


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


@gen_cluster(client=False)
async def test_gen_cluster_without_client(s, a, b):
    assert isinstance(s, Scheduler)
    for w in [a, b]:
        assert isinstance(w, Worker)
    assert s.nthreads == {w.address: w.nthreads for w in [a, b]}

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
    assert s.nthreads == {w.address: w.nthreads for w in [a, b]}


@gen_test()
async def test_gen_test():
    await asyncio.sleep(0.01)


@gen_test()
def test_gen_test_legacy_implicit():
    yield asyncio.sleep(0.01)


@gen_test()
@gen.coroutine
def test_gen_test_legacy_explicit():
    yield asyncio.sleep(0.01)


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


def test_wait_for_port():
    t1 = time()
    with pytest.raises(RuntimeError):
        wait_for_port((get_ip(), 9999), 0.5)
    t2 = time()
    assert t2 - t1 >= 0.5

    with _listen(0) as s1:
        t1 = time()
        wait_for_port(s1.getsockname())
        t2 = time()
        assert t2 - t1 <= 1.0

    with _listen(1) as s1:
        t1 = time()
        wait_for_port(s1.getsockname())
        t2 = time()
        assert t2 - t1 <= 2.0


def test_new_config():
    c = config.copy()
    with new_config({"xyzzy": 5}):
        config["xyzzy"] == 5

    assert config == c
    assert "xyzzy" not in config


def test_lingering_client():
    @gen_cluster()
    async def f(s, a, b):
        await Client(s.address, asynchronous=True)

    f()

    with pytest.raises(ValueError):
        default_client()


def test_lingering_client_2(loop):
    with cluster() as (s, [a, b]):
        client = Client(s["address"], loop=loop)


def test_tls_cluster(tls_client):
    tls_client.submit(lambda x: x + 1, 10).result() == 11
    assert tls_client.security


@pytest.mark.asyncio
async def test_tls_scheduler(security, cleanup):
    async with Scheduler(security=security, host="localhost") as s:
        assert s.address.startswith("tls")


from distributed.core import Status


@pytest.mark.asyncio
@pytest.mark.parametrize("w_cls", [Worker, Nanny])
async def test_start_cluster_closes_scheduler_worker_failure(w_cls):
    nthreads = [("127.0.0.1", 0)]
    scheduler = "127.0.0.1"
    loop = IOLoop.current()
    for _ in range(2):
        with pytest.raises(TypeError, match="got an unexpected keyword argument"):
            await start_cluster(
                nthreads,
                scheduler,
                loop,
                security=None,
                Worker=w_cls,
                scheduler_kwargs={},
                worker_kwargs={"dont": "start"},
            )
    assert all([s.status == Status.closed for s in Scheduler._instances])
    assert all([w.status == Status.closed for w in Worker._instances])

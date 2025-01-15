from __future__ import annotations

import asyncio
import subprocess
import sys
from threading import Lock
from time import sleep
from unittest import mock
from urllib.parse import urlparse

import pytest
from tornado.httpclient import AsyncHTTPClient

from dask.system import CPU_COUNT

from distributed import Client, LocalCluster, Nanny, Worker, get_client
from distributed.compatibility import LINUX, asyncio_run
from distributed.config import get_loop_factory
from distributed.core import Status
from distributed.metrics import time
from distributed.system import MEMORY_LIMIT
from distributed.utils import TimeoutError, open_port, sync
from distributed.utils_test import (
    assert_can_connect_from_everywhere_4,
    assert_can_connect_from_everywhere_4_6,
    assert_can_connect_locally_4,
    assert_cannot_connect,
    async_poll_for,
    captured_logger,
    gen_test,
    inc,
    raises_with_cause,
    slowinc,
    tls_only_security,
    xfail_ssl_issue5601,
)


def test_simple(loop):
    with LocalCluster(
        n_workers=4,
        processes=False,
        silence_logs=False,
        dashboard_address=":0",
        loop=loop,
    ) as c:
        with Client(c) as e:
            x = e.submit(inc, 1)
            x.result()
            assert x.key in c.scheduler.tasks
            assert any(w.data == {x.key: 2} for w in c.workers.values())

            assert e.loop is c.loop


def test_local_cluster_supports_blocked_handlers(loop):
    with LocalCluster(
        blocked_handlers=["run_function"],
        n_workers=0,
        loop=loop,
        dashboard_address=":0",
    ) as c:
        with Client(c) as client:
            with pytest.raises(ValueError) as exc:
                client.run_on_scheduler(lambda x: x, 42)

    assert "'run_function' handler has been explicitly disallowed in Scheduler" in str(
        exc.value
    )


def test_close_twice(loop):
    with LocalCluster(dashboard_address=":0", loop=loop) as cluster:
        with Client(cluster.scheduler_address, loop=loop) as client:
            f = client.map(inc, range(100))
            client.gather(f)
        with captured_logger("tornado.application") as log:
            cluster.close()
            cluster.close()
            sleep(0.5)
        log = log.getvalue()
        assert not log


def test_procs(loop_in_thread):
    loop = loop_in_thread
    with LocalCluster(
        n_workers=2,
        processes=False,
        threads_per_worker=3,
        dashboard_address=":0",
        silence_logs=False,
        loop=loop,
    ) as c:
        assert len(c.workers) == 2
        assert all(isinstance(w, Worker) for w in c.workers.values())
        with Client(c.scheduler.address, loop=loop) as e:
            assert all(w.state.nthreads == 3 for w in c.workers.values())
            assert all(isinstance(w, Worker) for w in c.workers.values())
        repr(c)

    with LocalCluster(
        n_workers=2,
        processes=True,
        threads_per_worker=3,
        dashboard_address=":0",
        silence_logs=False,
        loop=loop,
    ) as c:
        assert len(c.workers) == 2
        assert all(isinstance(w, Nanny) for w in c.workers.values())
        with Client(c.scheduler.address, loop=loop) as e:
            assert all(v == 3 for v in e.nthreads().values())

            c.scale(3)
            assert all(isinstance(w, Nanny) for w in c.workers.values())
        repr(c)


def test_move_unserializable_data(loop):
    """
    Test that unserializable data is still fine to transfer over inproc
    transports.
    """
    with LocalCluster(
        processes=False, silence_logs=False, dashboard_address=":0", loop=loop
    ) as cluster:
        assert cluster.scheduler_address.startswith("inproc://")
        assert cluster.workers[0].address.startswith("inproc://")
        with Client(cluster, loop=loop) as client:
            lock = Lock()
            x = client.scatter(lock)
            y = client.submit(lambda x: x, x)
            assert y.result() is lock


def test_transports_inproc(loop):
    """
    Test the transport chosen by LocalCluster depending on arguments.
    """
    with LocalCluster(
        n_workers=1,
        processes=False,
        silence_logs=False,
        dashboard_address=":0",
        loop=loop,
    ) as c:
        assert c.scheduler_address.startswith("inproc://")
        assert c.workers[0].address.startswith("inproc://")
        with Client(c.scheduler.address, loop=loop) as e:
            assert e.submit(inc, 4).result() == 5


def test_transports_tcp(loop):
    # Have nannies => need TCP
    with LocalCluster(
        n_workers=1,
        processes=True,
        silence_logs=False,
        dashboard_address=":0",
        loop=loop,
    ) as c:
        assert c.scheduler_address.startswith("tcp://")
        assert c.workers[0].address.startswith("tcp://")
        with Client(c.scheduler.address, loop=loop) as e:
            assert e.submit(inc, 4).result() == 5


def test_transports_tcp_port(loop):
    port = open_port()
    # Scheduler port specified => need TCP
    with LocalCluster(
        n_workers=1,
        processes=False,
        scheduler_port=port,
        silence_logs=False,
        dashboard_address=":0",
        loop=loop,
    ) as c:
        assert c.scheduler_address == f"tcp://127.0.0.1:{port}"
        assert c.workers[0].address.startswith("tcp://")
        with Client(c.scheduler.address, loop=loop) as e:
            assert e.submit(inc, 4).result() == 5


def test_cores(loop):
    with (
        LocalCluster(
            n_workers=2,
            scheduler_port=0,
            silence_logs=False,
            dashboard_address=":0",
            processes=False,
            loop=loop,
        ) as cluster,
        Client(cluster.scheduler_address, loop=loop) as client,
    ):
        client.scheduler_info()
        assert len(client.nthreads()) == 2


def test_submit(loop):
    with (
        LocalCluster(
            n_workers=2,
            scheduler_port=0,
            silence_logs=False,
            dashboard_address=":0",
            processes=False,
            loop=loop,
        ) as cluster,
        Client(cluster.scheduler_address, loop=loop) as client,
    ):
        future = client.submit(lambda x: x + 1, 1)
        assert future.result() == 2


def test_context_manager(loop):
    with (
        LocalCluster(
            silence_logs=False, dashboard_address=":0", processes=False, loop=loop
        ) as c,
        Client(c) as e,
    ):
        assert e.nthreads()


def test_no_workers_sync(loop):
    with LocalCluster(
        n_workers=0,
        scheduler_port=0,
        silence_logs=False,
        dashboard_address=":0",
        processes=False,
        loop=loop,
    ):
        pass


def test_Client_with_local(loop):
    with LocalCluster(
        n_workers=1, silence_logs=False, dashboard_address=":0", loop=loop
    ) as c:
        with Client(c) as e:
            assert len(e.nthreads()) == len(c.workers)
            assert c.scheduler_address in repr(c)


def test_Client_solo(loop):
    with Client(loop=loop, silence_logs=False, dashboard_address=":0") as c:
        pass
    assert c.cluster.status == Status.closed


@gen_test()
async def test_duplicate_clients():
    pytest.importorskip("bokeh")
    async with Client(
        processes=False, silence_logs=False, dashboard_address=9876, asynchronous=True
    ) as c1:
        c1_services = c1.cluster.scheduler.services
        with pytest.warns(Warning) as info:
            async with Client(
                processes=False,
                silence_logs=False,
                dashboard_address=9876,
                asynchronous=True,
            ) as c2:
                c2_services = c2.cluster.scheduler.services

    assert c1_services == {"dashboard": mock.ANY}
    assert c2_services == {"dashboard": mock.ANY}

    assert any(
        all(
            word in str(msg.message).lower()
            for word in ["9876", "running", "already in use"]
        )
        for msg in info.list
    )


def test_Client_kwargs(loop):
    with Client(
        loop=loop,
        processes=False,
        n_workers=2,
        silence_logs=False,
        dashboard_address=":0",
    ) as c:
        assert len(c.cluster.workers) == 2
        assert all(isinstance(w, Worker) for w in c.cluster.workers.values())
    assert c.cluster.status == Status.closed


def test_Client_unused_kwargs_with_cluster(loop):
    with LocalCluster(dashboard_address=":0", loop=loop) as cluster:
        with pytest.raises(Exception) as argexcept:
            c = Client(cluster, n_workers=2, dashboard_port=8000, silence_logs=None)
        assert (
            str(argexcept.value)
            == "Unexpected keyword arguments: ['dashboard_port', 'n_workers', 'silence_logs']"
        )


def test_Client_unused_kwargs_with_address(loop):
    with pytest.raises(Exception) as argexcept:
        c = Client(
            "127.0.0.1:8786", n_workers=2, dashboard_port=8000, silence_logs=None
        )
    assert (
        str(argexcept.value)
        == "Unexpected keyword arguments: ['dashboard_port', 'n_workers', 'silence_logs']"
    )


def test_Client_twice(loop):
    with Client(loop=loop, silence_logs=False, dashboard_address=":0") as c:
        with Client(loop=loop, silence_logs=False, dashboard_address=":0") as f:
            assert c.cluster.scheduler.port != f.cluster.scheduler.port


@gen_test()
async def test_client_constructor_with_temporary_security():
    xfail_ssl_issue5601()
    pytest.importorskip("cryptography")
    async with Client(
        security=True, silence_logs=False, dashboard_address=":0", asynchronous=True
    ) as c:
        assert c.cluster.scheduler_address.startswith("tls")
        assert c.security == c.cluster.security


@gen_test()
async def test_defaults():
    async with LocalCluster(
        silence_logs=False, dashboard_address=":0", asynchronous=True
    ) as c:
        assert sum(w.nthreads for w in c.workers.values()) == CPU_COUNT
        assert all(isinstance(w, Nanny) for w in c.workers.values())


@gen_test()
async def test_defaults_2():
    async with LocalCluster(
        processes=False,
        silence_logs=False,
        dashboard_address=":0",
        asynchronous=True,
    ) as c:
        assert sum(w.state.nthreads for w in c.workers.values()) == CPU_COUNT
        assert all(isinstance(w, Worker) for w in c.workers.values())
        assert len(c.workers) == 1


@gen_test()
async def test_defaults_3():
    async with LocalCluster(
        n_workers=2,
        silence_logs=False,
        dashboard_address=":0",
        asynchronous=True,
    ) as c:
        if CPU_COUNT % 2 == 0:
            expected_total_threads = max(2, CPU_COUNT)
        else:
            # n_workers not a divisor of _nthreads => threads are overcommitted
            expected_total_threads = max(2, CPU_COUNT + 1)
        assert sum(w.nthreads for w in c.workers.values()) == expected_total_threads


@gen_test()
async def test_defaults_4():
    async with LocalCluster(
        threads_per_worker=CPU_COUNT * 2,
        silence_logs=False,
        dashboard_address=":0",
        asynchronous=True,
    ) as c:
        assert len(c.workers) == 1


@gen_test()
async def test_defaults_5():
    async with LocalCluster(
        n_workers=CPU_COUNT * 2,
        silence_logs=False,
        dashboard_address=":0",
        asynchronous=True,
    ) as c:
        assert all(w.nthreads == 1 for w in c.workers.values())


@gen_test()
async def test_defaults_6():
    async with LocalCluster(
        threads_per_worker=2,
        n_workers=3,
        silence_logs=False,
        dashboard_address=":0",
        asynchronous=True,
    ) as c:
        assert len(c.workers) == 3
        assert all(w.nthreads == 2 for w in c.workers.values())


@gen_test()
async def test_worker_params():
    async with LocalCluster(
        processes=False,
        n_workers=2,
        silence_logs=False,
        dashboard_address=":0",
        memory_limit=500,
        asynchronous=True,
    ) as c:
        assert [w.memory_manager.memory_limit for w in c.workers.values()] == [500] * 2


@gen_test()
async def test_memory_limit_none():
    async with LocalCluster(
        n_workers=2,
        silence_logs=False,
        processes=False,
        dashboard_address=":0",
        memory_limit=None,
        asynchronous=True,
    ) as c:
        w = c.workers[0]
        assert type(w.data) is dict
        assert w.memory_manager.memory_limit is None


def test_cleanup(loop):
    c = LocalCluster(n_workers=2, silence_logs=False, dashboard_address=":0", loop=loop)
    port = c.scheduler.port
    c.close()
    c2 = LocalCluster(
        n_workers=2,
        scheduler_port=port,
        silence_logs=False,
        dashboard_address=":0",
        loop=loop,
    )
    c2.close()


def test_repeated(loop_in_thread):
    loop = loop_in_thread
    with LocalCluster(
        n_workers=0,
        scheduler_port=8448,
        silence_logs=False,
        dashboard_address=":0",
        loop=loop,
    ):
        pass
    with LocalCluster(
        n_workers=0,
        scheduler_port=8448,
        silence_logs=False,
        dashboard_address=":0",
        loop=loop,
    ):
        pass


@pytest.mark.parametrize("processes", [True, False])
def test_bokeh(loop, processes):
    pytest.importorskip("bokeh")
    requests = pytest.importorskip("requests")
    with LocalCluster(
        n_workers=0,
        silence_logs=False,
        loop=loop,
        processes=processes,
        dashboard_address=":0",
    ) as c:
        bokeh_port = c.scheduler.http_server.port
        url = "http://127.0.0.1:%d/status/" % bokeh_port
        start = time()
        while True:
            response = requests.get(url)
            if response.ok:
                break
            assert time() < start + 20
            sleep(0.01)
        # 'localhost' also works
        response = requests.get("http://localhost:%d/status/" % bokeh_port)
        assert response.ok

    with pytest.raises(requests.RequestException):
        requests.get(url, timeout=0.2)


def test_blocks_until_full(loop):
    with Client(loop=loop, dashboard_address=":0") as c:
        assert len(c.nthreads()) > 0


@gen_test()
async def test_scale_up_and_down():
    async with LocalCluster(
        n_workers=0,
        processes=False,
        silence_logs=False,
        dashboard_address=":0",
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as c:
            assert not cluster.workers

            cluster.scale(2)
            await cluster
            assert len(cluster.workers) == 2
            assert len(cluster.scheduler.workers) == 2

            cluster.scale(1)
            await cluster

            assert len(cluster.workers) == 1


def test_silent_startup():
    code = """if 1:
        from time import sleep
        from distributed import LocalCluster

        if __name__ == "__main__":
            with LocalCluster(n_workers=1, dashboard_address=":0"):
                sleep(.1)
        """

    out = subprocess.check_output(
        [sys.executable, "-Wi", "-c", code], stderr=subprocess.STDOUT
    )
    out = out.decode()
    try:
        assert not out
    except AssertionError:
        print("=== Cluster stdout / stderr ===")
        print(out)
        raise


def test_only_local_access(loop):
    with LocalCluster(
        n_workers=0, silence_logs=False, dashboard_address=":0", loop=loop
    ) as c:
        sync(loop, assert_can_connect_locally_4, c.scheduler.port)


def test_remote_access(loop):
    with LocalCluster(
        n_workers=0,
        silence_logs=False,
        dashboard_address=":0",
        host="",
        loop=loop,
    ) as c:
        sync(loop, assert_can_connect_from_everywhere_4_6, c.scheduler.port)


@pytest.mark.parametrize("n_workers", [None, 3])
def test_memory(loop, n_workers):
    with LocalCluster(
        n_workers=n_workers,
        processes=False,
        silence_logs=False,
        dashboard_address=":0",
        loop=loop,
    ) as cluster:
        assert (
            sum(w.memory_manager.memory_limit for w in cluster.workers.values())
            <= MEMORY_LIMIT
        )


@pytest.mark.parametrize("n_workers", [None, 3])
def test_memory_nanny(loop, n_workers):
    with LocalCluster(
        n_workers=n_workers,
        processes=True,
        silence_logs=False,
        dashboard_address=":0",
        loop=loop,
    ) as cluster:
        with Client(cluster.scheduler_address, loop=loop) as c:
            info = c.scheduler_info()
            assert (
                sum(w["memory_limit"] for w in info["workers"].values()) <= MEMORY_LIMIT
            )


def test_death_timeout_raises(loop):
    with pytest.raises(TimeoutError):
        with LocalCluster(
            silence_logs=False,
            death_timeout=1e-10,
            dashboard_address=":0",
            loop=loop,
        ) as cluster:
            pass


@gen_test()
async def test_bokeh_kwargs():
    pytest.importorskip("bokeh")
    async with LocalCluster(
        n_workers=0,
        silence_logs=False,
        dashboard_address=":0",
        asynchronous=True,
        scheduler_kwargs={"http_prefix": "/foo"},
    ) as c:
        client = AsyncHTTPClient()
        response = await client.fetch(
            f"http://localhost:{c.scheduler.http_server.port}/foo/status"
        )
        assert "bokeh" in response.body.decode()


def test_io_loop_periodic_callbacks(loop):
    with LocalCluster(loop=loop, dashboard_address=":0", silence_logs=False) as cluster:
        assert cluster.scheduler.loop is loop
        for pc in cluster.scheduler.periodic_callbacks.values():
            assert pc.io_loop is loop
        for worker in cluster.workers.values():
            for pc in worker.periodic_callbacks.values():
                assert pc.io_loop is loop


def test_logging(loop):
    """
    Workers and scheduler have logs even when silenced
    """
    with LocalCluster(
        n_workers=1, processes=False, dashboard_address=":0", loop=loop
    ) as c:
        assert c.scheduler._deque_handler.deque
        assert c.workers[0]._deque_handler.deque


def test_ipywidgets(loop):
    ipywidgets = pytest.importorskip("ipywidgets")
    with LocalCluster(
        n_workers=0,
        silence_logs=False,
        loop=loop,
        dashboard_address=":0",
        processes=False,
    ) as cluster:
        cluster._ipython_display_()
        box = cluster._cached_widget
        assert isinstance(box, ipywidgets.Widget)


def test_ipywidgets_loop(loop):
    """
    Previously cluster._ipython_display_ attached the PeriodicCallback to the
    currently running loop, See https://github.com/dask/distributed/pull/6444
    """
    ipywidgets = pytest.importorskip("ipywidgets")

    async def get_ioloop(cluster):
        return cluster.periodic_callbacks["cluster-repr"].io_loop

    async def amain():
        # running synchronous code in an async context to setup a
        # IOLoop.current() that's different from cluster.loop
        with LocalCluster(
            n_workers=0,
            silence_logs=False,
            loop=loop,
            dashboard_address=":0",
            processes=False,
        ) as cluster:
            cluster._ipython_display_()
            assert cluster.sync(get_ioloop, cluster) is loop
            box = cluster._cached_widget
            assert isinstance(box, ipywidgets.Widget)

    asyncio_run(amain(), loop_factory=get_loop_factory())


def test_no_ipywidgets(loop, monkeypatch):
    from unittest.mock import MagicMock

    mock_display = MagicMock()

    monkeypatch.setitem(sys.modules, "ipywidgets", None)
    monkeypatch.setitem(sys.modules, "IPython.display", mock_display)

    with LocalCluster(
        n_workers=0,
        silence_logs=False,
        loop=loop,
        dashboard_address=":0",
        processes=False,
    ) as cluster:
        cluster._ipython_display_()
        args, kwargs = mock_display.display.call_args
        res = args[0]
        assert kwargs == {"raw": True}
        assert isinstance(res, dict)
        assert "text/plain" in res
        assert "text/html" in res


def test_scale(loop):
    """Directly calling scale both up and down works as expected"""
    with LocalCluster(
        silence_logs=False,
        loop=loop,
        dashboard_address=":0",
        processes=False,
        n_workers=0,
    ) as cluster:
        assert not cluster.scheduler.workers
        cluster.scale(3)

        start = time()
        while len(cluster.scheduler.workers) != 3:
            sleep(0.01)
            assert time() < start + 5, len(cluster.scheduler.workers)

        sleep(0.2)  # let workers settle # TODO: remove need for this

        cluster.scale(2)

        start = time()
        while len(cluster.scheduler.workers) != 2:
            sleep(0.01)
            assert time() < start + 5, len(cluster.scheduler.workers)


def test_adapt(loop):
    with LocalCluster(
        silence_logs=False,
        loop=loop,
        dashboard_address=":0",
        processes=False,
        n_workers=0,
    ) as cluster:
        cluster.adapt(minimum=0, maximum=2, interval="10ms")
        assert cluster._adaptive.minimum == 0
        assert cluster._adaptive.maximum == 2

        cluster.adapt(minimum=1, maximum=2, interval="10ms")
        assert cluster._adaptive.minimum == 1

        start = time()
        while len(cluster.scheduler.workers) != 1:
            sleep(0.01)
            assert time() < start + 5


@gen_test()
async def test_adapt_then_manual():
    """We can revert from adaptive, back to manual"""
    async with LocalCluster(
        asynchronous=True,
        dashboard_address=":0",
        processes=False,
        n_workers=8,
        threads_per_worker=1,
    ) as cluster:

        def wait_workers(n):
            return async_poll_for(
                lambda: len(cluster.scheduler.workers) == n
                and len(cluster.workers) == n,
                timeout=5,
            )

        await wait_workers(8)
        cluster.adapt(minimum=0, maximum=4, interval="10ms")
        await wait_workers(0)

        async with Client(cluster, asynchronous=True) as client:
            futures = client.map(slowinc, range(1000), delay=0.1)
            await wait_workers(4)

            cluster._adaptive.stop()
            # stop() disables a PeriodicCallback. However, sometimes the callback has
            # already been scheduled or is halfway through (it's asynchronous).
            # Wait for it to finish, otherwise you'd get a race condition with
            # scale() below and flakiness in the test.
            tasks = {t for t in asyncio.all_tasks() if "PeriodicCallback" in str(t)}
            if tasks:
                await asyncio.wait(tasks)

            cluster.scale(2)
            await wait_workers(2)


@pytest.mark.parametrize("temporary", [True, False])
def test_local_tls(loop, temporary):
    port = open_port()
    if temporary:
        xfail_ssl_issue5601()
        pytest.importorskip("cryptography")
        security = True
    else:
        security = tls_only_security()
    with LocalCluster(
        n_workers=0,
        scheduler_port=port,
        silence_logs=False,
        security=security,
        dashboard_address=":0",
        host="tls://0.0.0.0",
        loop=loop,
    ) as c:
        sync(
            loop,
            assert_can_connect_from_everywhere_4,
            c.scheduler.port,
            protocol="tls",
            timeout=3,
            **c.security.get_connection_args("client"),
        )

        # If we connect to a TLS localculster without ssl information we should fail
        sync(
            loop,
            assert_cannot_connect,
            addr="tcp://127.0.0.1:%d" % c.scheduler.port,
            exception_class=RuntimeError,
            **c.security.get_connection_args("client"),
        )


@gen_test()
async def test_scale_retires_workers():
    class MyCluster(LocalCluster):
        def scale_down(self, *args, **kwargs):
            pass

    async with (
        MyCluster(
            n_workers=0,
            processes=False,
            silence_logs=False,
            dashboard_address=":0",
            loop=None,
            asynchronous=True,
        ) as cluster,
        Client(cluster, asynchronous=True) as c,
    ):
        assert not cluster.workers

        await cluster.scale(2)

        start = time()
        while len(cluster.scheduler.workers) != 2:
            await asyncio.sleep(0.01)
            assert time() < start + 3

        await cluster.scale(1)

        start = time()
        while len(cluster.scheduler.workers) != 1:
            await asyncio.sleep(0.01)
            assert time() < start + 3


def test_local_tls_restart(loop):
    from distributed.utils_test import tls_only_security

    security = tls_only_security()
    with LocalCluster(
        n_workers=1,
        scheduler_port=open_port(),
        silence_logs=False,
        security=security,
        dashboard_address=":0",
        host="tls://0.0.0.0",
        loop=loop,
    ) as c:
        with Client(c.scheduler.address, loop=loop, security=security) as client:
            workers_before = set(client.scheduler_info()["workers"])
            assert client.submit(inc, 1).result() == 2
            client.restart()
            workers_after = set(client.scheduler_info()["workers"])
            assert client.submit(inc, 2).result() == 3
            assert workers_before != workers_after


def test_asynchronous_property(loop):
    with LocalCluster(
        n_workers=4,
        processes=False,
        silence_logs=False,
        dashboard_address=":0",
        loop=loop,
    ) as cluster:

        async def _():
            assert cluster.asynchronous

        cluster.sync(_)


def test_protocol_inproc(loop):
    with LocalCluster(
        protocol="inproc://", loop=loop, processes=False, dashboard_address=":0"
    ) as cluster:
        assert cluster.scheduler.address.startswith("inproc://")


def test_protocol_tcp(loop):
    with LocalCluster(
        protocol="tcp", loop=loop, n_workers=0, processes=False, dashboard_address=":0"
    ) as cluster:
        assert cluster.scheduler.address.startswith("tcp://")


@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
def test_protocol_ip(loop):
    with LocalCluster(
        host="tcp://127.0.0.2",
        loop=loop,
        n_workers=0,
        processes=False,
        dashboard_address=":0",
    ) as cluster:
        assert cluster.scheduler.address.startswith("tcp://127.0.0.2")


class MyWorker(Worker):
    pass


def test_worker_class_worker(loop):
    with LocalCluster(
        n_workers=2,
        loop=loop,
        worker_class=MyWorker,
        processes=False,
        dashboard_address=":0",
    ) as cluster:
        assert all(isinstance(w, MyWorker) for w in cluster.workers.values())


def test_worker_class_nanny(loop):
    class MyNanny(Nanny):
        pass

    with LocalCluster(
        n_workers=2,
        loop=loop,
        worker_class=MyNanny,
        dashboard_address=":0",
    ) as cluster:
        assert all(isinstance(w, MyNanny) for w in cluster.workers.values())


@gen_test()
async def test_worker_class_nanny_async():
    class MyNanny(Nanny):
        pass

    async with LocalCluster(
        n_workers=2,
        worker_class=MyNanny,
        dashboard_address=":0",
        asynchronous=True,
    ) as cluster:
        assert all(isinstance(w, MyNanny) for w in cluster.workers.values())


def test_starts_up_sync(loop):
    cluster = LocalCluster(
        n_workers=2,
        loop=loop,
        processes=False,
        dashboard_address=":0",
    )
    try:
        assert len(cluster.scheduler.workers) == 2
    finally:
        cluster.close()


def test_dont_select_closed_worker(loop):
    # Make sure distributed does not try to reuse a client from a
    # closed cluster (https://github.com/dask/distributed/issues/2840).
    cluster = LocalCluster(n_workers=0, dashboard_address=":0", loop=loop)
    c = Client(cluster)
    cluster.scale(2)
    assert c == get_client()

    c.close()
    cluster.close()

    cluster2 = LocalCluster(n_workers=0, dashboard_address=":0", loop=loop)
    c2 = Client(cluster2)
    cluster2.scale(2)

    current_client = get_client()
    assert c2 == current_client

    cluster2.close()
    c2.close()


def test_client_cluster_synchronous(loop):
    with Client(loop=loop, processes=False, dashboard_address=":0") as c:
        assert not c.asynchronous
        assert not c.cluster.asynchronous


@gen_test()
async def test_scale_memory_cores():
    async with LocalCluster(
        n_workers=0,
        processes=False,
        threads_per_worker=2,
        memory_limit="2GB",
        asynchronous=True,
        dashboard_address=":0",
    ) as cluster:
        cluster.scale(cores=4)
        assert len(cluster.worker_spec) == 2

        cluster.scale(memory="6GB")
        assert len(cluster.worker_spec) == 3

        cluster.scale(cores=1)
        assert len(cluster.worker_spec) == 1

        cluster.scale(memory="7GB")
        assert len(cluster.worker_spec) == 4


@pytest.mark.parametrize("memory_limit", ["2 GiB", None])
@gen_test()
async def test_repr(memory_limit):
    async with LocalCluster(
        n_workers=2,
        processes=False,
        threads_per_worker=2,
        memory_limit=memory_limit,
        asynchronous=True,
        dashboard_address=":0",
    ) as cluster:
        # __repr__ uses cluster.scheduler_info, which slightly lags behind
        # cluster.scheduler.workers and client.wait_for_workers.
        while len(cluster.scheduler_info["workers"]) < 2:
            await asyncio.sleep(0.01)

        text = repr(cluster)
        assert cluster.scheduler_address in text
        assert "workers=2, threads=4" in text
        if memory_limit:
            assert "memory=4.00 GiB" in text
        else:
            assert "memory" not in text


@gen_test()
async def test_threads_per_worker_set_to_0():
    with pytest.warns(
        Warning, match="Setting `threads_per_worker` to 0 has been deprecated."
    ):
        async with LocalCluster(
            n_workers=2,
            processes=False,
            threads_per_worker=0,
            asynchronous=True,
            dashboard_address=":0",
        ) as cluster:
            assert len(cluster.workers) == 2
            assert all(w.state.nthreads < CPU_COUNT for w in cluster.workers.values())


@pytest.mark.parametrize("temporary", [True, False])
@gen_test()
async def test_capture_security(temporary):
    if temporary:
        xfail_ssl_issue5601()
        pytest.importorskip("cryptography")
        security = True
    else:
        security = tls_only_security()
    async with LocalCluster(
        n_workers=0,
        silence_logs=False,
        security=security,
        asynchronous=True,
        dashboard_address=":0",
        host="tls://0.0.0.0",
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            assert client.security == cluster.security


@gen_test()
async def test_no_dangling_asyncio_tasks():
    start = asyncio.all_tasks()
    async with LocalCluster(asynchronous=True, processes=False, dashboard_address=":0"):
        await asyncio.sleep(0.01)

    tasks = asyncio.all_tasks()
    assert tasks == start


@gen_test()
async def test_async_with():
    async with LocalCluster(
        processes=False, asynchronous=True, dashboard_address=":0"
    ) as cluster:
        w = cluster.workers
        assert w

    assert not w


@gen_test()
async def test_no_workers():
    async with Client(
        n_workers=0, silence_logs=False, dashboard_address=":0", asynchronous=True
    ) as c:
        pass


@gen_test()
async def test_cluster_names():
    async with LocalCluster(
        processes=False, asynchronous=True, dashboard_address=":0"
    ) as unnamed_cluster:
        async with LocalCluster(
            processes=False, asynchronous=True, name="mycluster", dashboard_address=":0"
        ) as named_cluster:
            assert isinstance(unnamed_cluster.name, str)
            assert isinstance(named_cluster.name, str)
            assert named_cluster.name == "mycluster"
            assert unnamed_cluster == unnamed_cluster
            assert named_cluster == named_cluster
            assert unnamed_cluster != named_cluster

        async with LocalCluster(
            processes=False, asynchronous=True, dashboard_address=":0"
        ) as unnamed_cluster2:
            assert unnamed_cluster2 != unnamed_cluster


@pytest.mark.parametrize("nanny", [True, False])
@gen_test()
async def test_local_cluster_redundant_kwarg(nanny):
    cluster = LocalCluster(
        typo_kwarg="foo",
        processes=nanny,
        n_workers=1,
        dashboard_address=":0",
        asynchronous=True,
    )
    if nanny:
        ctx = raises_with_cause(
            RuntimeError, None, TypeError, "unexpected keyword argument"
        )
    else:
        ctx = pytest.raises(TypeError, match="unexpected keyword argument")
    with ctx:
        # Extra arguments are forwarded to the worker class. Depending on
        # whether we use the nanny or not, the error treatment is quite
        # different and we should assert that an exception is raised
        async with cluster:
            pass


@gen_test()
async def test_cluster_info_sync():
    async with LocalCluster(
        processes=False,
        asynchronous=True,
        scheduler_sync_interval="1ms",
        dashboard_address=":0",
    ) as cluster:
        assert cluster._cluster_info["name"] == cluster.name

        while "name" not in cluster.scheduler.get_metadata(
            keys=["cluster-manager-info"]
        ):
            await asyncio.sleep(0.01)

        info = await cluster.scheduler_comm.get_metadata(keys=["cluster-manager-info"])
        assert info["name"] == cluster.name
        info = cluster.scheduler.get_metadata(keys=["cluster-manager-info"])
        assert info["name"] == cluster.name

        cluster._cluster_info["foo"] = "bar"
        while "foo" not in cluster.scheduler.get_metadata(
            keys=["cluster-manager-info"]
        ):
            await asyncio.sleep(0.01)

        info = cluster.scheduler.get_metadata(keys=["cluster-manager-info"])
        assert info["foo"] == "bar"


@gen_test()
async def test_cluster_info_sync_is_robust_to_network_blips(monkeypatch):
    async with LocalCluster(
        processes=False,
        asynchronous=True,
        scheduler_sync_interval="1ms",
        dashboard_address=":0",
    ) as cluster:
        assert cluster._cluster_info["name"] == cluster.name

        error_called = False

        async def error(*args, **kwargs):
            nonlocal error_called
            await asyncio.sleep(0.001)
            error_called = True
            raise OSError

        # Temporarily patch the `set_metadata` RPC to error
        with monkeypatch.context() as patch:
            patch.setattr(cluster.scheduler_comm, "set_metadata", error)

            # Set a new cluster_info value
            cluster._cluster_info["foo"] = "bar"

            # Wait for the bad method to be called at least once
            while not error_called:
                await asyncio.sleep(0.01)

        # Check that cluster_info is resynced after the error condition is fixed
        while "foo" not in cluster.scheduler.get_metadata(
            keys=["cluster-manager-info"]
        ):
            await asyncio.sleep(0.01)

        info = cluster.scheduler.get_metadata(keys=["cluster-manager-info"])
        assert info["foo"] == "bar"


@pytest.mark.parametrize("host", [None, "127.0.0.1"])
@pytest.mark.parametrize("use_nanny", [True, False])
@gen_test()
async def test_cluster_host_used_throughout_cluster(host, use_nanny):
    """Ensure that the `host` kwarg is propagated through scheduler, nanny, and workers"""
    async with LocalCluster(
        host=host, asynchronous=True, dashboard_address=":0"
    ) as cluster:
        url = urlparse(cluster.scheduler_address)
        assert url.hostname == "127.0.0.1"
        for worker in cluster.workers.values():
            url = urlparse(worker.address)
            assert url.hostname == "127.0.0.1"

            if use_nanny:
                url = urlparse(worker.process.worker_address)
                assert url.hostname == "127.0.0.1"


@gen_test()
async def test_connect_to_closed_cluster():
    async with LocalCluster(
        processes=False, asynchronous=True, dashboard_address=":0"
    ) as cluster:
        async with Client(cluster, asynchronous=True) as c1:
            assert await c1.submit(inc, 1) == 2

    with pytest.raises(
        RuntimeError,
        match="Trying to connect to an already closed or closing Cluster",
    ):
        # Raises during init without actually connecting since we're not
        # awaiting anything
        Client(cluster, asynchronous=True)


class MyPlugin:
    def setup(self, worker=None):
        import my_nonexistent_library  # noqa


@pytest.mark.slow
def test_localcluster_start_exception(loop):
    with raises_with_cause(
        RuntimeError,
        "Nanny failed to start",
        RuntimeError,
        "Worker failed to start",
        ImportError,
        "my_nonexistent_library",
    ):
        with LocalCluster(
            n_workers=1,
            threads_per_worker=1,
            processes=True,
            plugins={MyPlugin()},
            loop=loop,
        ):
            pass


def test_localcluster_scale_exception(loop):
    with raises_with_cause(
        RuntimeError,
        "Nanny failed to start",
        RuntimeError,
        "Worker failed to start",
        ImportError,
        "my_nonexistent_library",
    ):
        with LocalCluster(
            n_workers=0,
            threads_per_worker=1,
            processes=True,
            plugins={MyPlugin()},
            loop=loop,
        ) as cluster:
            cluster.scale(1)
            cluster.sync(cluster._correct_state)


def test_localcluster_get_client(loop):
    with LocalCluster(
        n_workers=0, asynchronous=False, dashboard_address=":0", loop=loop
    ) as cluster:
        with cluster.get_client() as client1:
            assert client1.cluster == cluster

            with Client(cluster) as client2:
                assert client1 != client2
                assert client2 == cluster.get_client()

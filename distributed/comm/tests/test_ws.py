import os
import tempfile
import warnings

import numpy as np
import pytest

import dask

from distributed import Client, Scheduler, Worker
from distributed.comm import connect, listen, ws
from distributed.comm.core import FatalCommClosedError
from distributed.comm.registry import backends, get_backend
from distributed.security import Security
from distributed.utils_test import (
    gen_cluster,
    get_client_ssl_context,
    get_server_ssl_context,
    inc,
)

from .test_comms import check_tls_extra

security = Security.temporary()


def test_registered():
    assert "ws" in backends
    backend = get_backend("ws")
    assert isinstance(backend, ws.WSBackend)


@pytest.mark.asyncio
async def test_listen_connect(cleanup):
    async def handle_comm(comm):
        while True:
            msg = await comm.read()
            await comm.write(msg)

    async with listen("ws://", handle_comm) as listener:
        comm = await connect(listener.contact_address)
        await comm.write(b"Hello!")
        result = await comm.read()
        assert result == b"Hello!"

        await comm.close()


@pytest.mark.asyncio
async def test_listen_connect_wss(cleanup):
    async def handle_comm(comm):
        while True:
            msg = await comm.read()
            await comm.write(msg)

    server_ctx = get_server_ssl_context()
    client_ctx = get_client_ssl_context()

    async with listen("wss://", handle_comm, ssl_context=server_ctx) as listener:
        comm = await connect(listener.contact_address, ssl_context=client_ctx)
        assert comm.peer_address.startswith("wss://")
        check_tls_extra(comm.extra_info)
        await comm.write(b"Hello!")
        result = await comm.read()
        assert result == b"Hello!"
        await comm.close()


@pytest.mark.asyncio
async def test_expect_ssl_context(cleanup):
    server_ctx = get_server_ssl_context()

    async with listen("wss://", lambda comm: comm, ssl_context=server_ctx) as listener:
        with pytest.raises(FatalCommClosedError, match="TLS expects a `ssl_context` *"):
            comm = await connect(listener.contact_address)


@pytest.mark.asyncio
async def test_expect_scheduler_ssl_when_sharing_server(cleanup):
    with tempfile.TemporaryDirectory() as tempdir:
        key_path = os.path.join(tempdir, "dask.pem")
        cert_path = os.path.join(tempdir, "dask.crt")
        with open(key_path, "w") as f:
            f.write(security.tls_scheduler_key)
        with open(cert_path, "w") as f:
            f.write(security.tls_scheduler_cert)
        c = {
            "distributed.scheduler.dashboard.tls.key": key_path,
            "distributed.scheduler.dashboard.tls.cert": cert_path,
        }
        with dask.config.set(c):
            with pytest.raises(RuntimeError):
                async with Scheduler(protocol="ws://", dashboard=True, port=8787) as s:
                    pass


@pytest.mark.asyncio
async def test_roundtrip(cleanup):
    async with Scheduler(protocol="ws://") as s:
        async with Worker(s.address) as w:
            async with Client(s.address, asynchronous=True) as c:
                assert c.scheduler.address.startswith("ws://")
                assert w.address.startswith("ws://")
                future = c.submit(inc, 1)
                result = await future
                assert result == 2


@pytest.mark.asyncio
async def test_collections(cleanup):
    da = pytest.importorskip("dask.array")
    async with Scheduler(protocol="ws://") as s:
        async with Worker(s.address) as a:
            async with Worker(s.address) as b:
                async with Client(s.address, asynchronous=True) as c:
                    x = da.random.random((1000, 1000), chunks=(100, 100))
                    x = x + x.T
                    await x.persist()


@pytest.mark.asyncio
async def test_large_transfer(cleanup):
    np = pytest.importorskip("numpy")
    async with Scheduler(protocol="ws://") as s:
        async with Worker(s.address, protocol="ws://") as w:
            async with Client(s.address, asynchronous=True) as c:
                future = await c.scatter(np.random.random(1000000))


@pytest.mark.asyncio
@pytest.mark.filterwarnings(
    "ignore:Dashboard and Scheduler are using the same server on port"
)
@pytest.mark.parametrize(
    "dashboard,protocol,security,port",
    [
        (True, "ws://", None, 8787),
        (True, "wss://", security, 8787),
        (False, "ws://", None, 8787),
        (False, "wss://", security, 8787),
        (True, "ws://", None, 8786),
        (True, "wss://", security, 8786),
        (False, "ws://", None, 8786),
        (False, "wss://", security, 8786),
    ],
)
async def test_http_and_comm_server(cleanup, dashboard, protocol, security, port):
    async with Scheduler(
        protocol=protocol, dashboard=dashboard, port=port, security=security
    ) as s:
        if port == 8787:
            assert s.http_server is s.listener.server
        else:
            assert s.http_server is not s.listener.server
        async with Worker(s.address, protocol=protocol, security=security) as w:
            async with Client(s.address, asynchronous=True, security=security) as c:
                result = await c.submit(lambda x: x + 1, 10)
                assert result == 11


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "protocol,security",
    [
        (
            "ws://",
            Security(extra_conn_args={"headers": {"Authorization": "Token abcd"}}),
        ),
        (
            "wss://",
            Security.temporary(
                extra_conn_args={"headers": {"Authorization": "Token abcd"}}
            ),
        ),
    ],
)
async def test_connection_made_with_extra_conn_args(cleanup, protocol, security):
    async with Scheduler(protocol=protocol, security=security) as s:
        connection_args = security.get_connection_args("worker")
        comm = await connect(s.address, **connection_args)
        assert comm.sock.request.headers.get("Authorization") == "Token abcd"
        await comm.close()


@pytest.mark.asyncio
async def test_quiet_close(cleanup):
    with warnings.catch_warnings(record=True) as record:
        async with Client(protocol="ws", processes=False, asynchronous=True) as c:
            pass

    # For some reason unrelated @coroutine warnings are showing up
    record = [warning for warning in record if "coroutine" not in str(warning.message)]

    assert not record, record[0].message


@gen_cluster(
    client=True,
    scheduler_kwargs={"protocol": "ws://"},
)
async def test_ws_roundtrip(c, s, a, b):
    x = np.arange(100)
    future = await c.scatter(x)
    y = await future
    assert (x == y).all()


@gen_cluster(
    client=True,
    security=security,
    scheduler_kwargs={"protocol": "wss://"},
)
async def test_wss_roundtrip(c, s, a, b):
    x = np.arange(100)
    future = await c.scatter(x)
    y = await future
    assert (x == y).all()

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import random
import socket
import time as timemod
import weakref
from unittest import mock

import pytest
from tornado.ioloop import IOLoop

import dask

from distributed.batched import BatchedSend
from distributed.comm.core import CommClosedError
from distributed.comm.tcp import TCPBackend, TCPListener
from distributed.core import (
    AsyncTaskGroup,
    AsyncTaskGroupClosedError,
    ConnectionPool,
    Server,
    Status,
    _expects_comm,
    clean_exception,
    coerce_to_address,
    connect,
    pingpong,
    rpc,
    send_recv,
)
from distributed.metrics import time
from distributed.protocol import to_serialize
from distributed.protocol.compression import compressions
from distributed.utils import get_ip, get_ipv6, wait_for
from distributed.utils_test import (
    assert_can_connect,
    assert_can_connect_from_everywhere_4,
    assert_can_connect_from_everywhere_4_6,
    assert_can_connect_from_everywhere_6,
    assert_can_connect_locally_4,
    assert_can_connect_locally_6,
    assert_cannot_connect,
    gen_test,
    has_ipv6,
    inc,
    throws,
    tls_security,
)


class CountedObject:
    """
    A class which counts the number of live instances.
    """

    n_instances = 0

    # Use __new__, as __init__ can be bypassed by pickle.
    def __new__(cls):
        cls.n_instances += 1
        obj = object.__new__(cls)
        weakref.finalize(obj, cls._finalize)
        return obj

    @classmethod
    def _finalize(cls, *args):
        cls.n_instances -= 1


def echo_serialize(comm, x):
    return {"result": to_serialize(x)}


def echo_no_serialize(comm, x):
    return {"result": x}


def test_async_task_group_initialization():
    group = AsyncTaskGroup()
    assert not group.closed
    assert len(group) == 0


async def _wait_for_n_loop_cycles(n):
    for _ in range(n):
        await asyncio.sleep(0)


@gen_test()
async def test_async_task_group_call_soon_executes_task_in_background():
    group = AsyncTaskGroup()
    ev = asyncio.Event()
    flag = False

    async def set_flag():
        nonlocal flag
        await ev.wait()
        flag = True

    assert group.call_soon(set_flag) is None
    assert len(group) == 1
    ev.set()
    await _wait_for_n_loop_cycles(2)
    assert len(group) == 0
    assert flag


@gen_test()
async def test_async_task_group_call_later_executes_delayed_task_in_background():
    group = AsyncTaskGroup()
    ev = asyncio.Event()

    start = timemod.monotonic()
    assert group.call_later(1, ev.set) is None
    assert len(group) == 1
    await ev.wait()
    end = timemod.monotonic()
    # the task must be removed in exactly 1 event loop cycle
    await _wait_for_n_loop_cycles(2)
    assert len(group) == 0
    assert end - start > 1 - timemod.get_clock_info("monotonic").resolution


def test_async_task_group_close_closes():
    group = AsyncTaskGroup()
    group.close()
    assert group.closed

    # Test idempotency
    group.close()
    assert group.closed


@gen_test()
async def test_async_task_group_close_does_not_cancel_existing_tasks():
    group = AsyncTaskGroup()

    ev = asyncio.Event()
    flag = False

    async def set_flag():
        nonlocal flag
        await ev.wait()
        flag = True
        return None

    assert group.call_soon(set_flag) is None

    group.close()

    assert len(group) == 1

    ev.set()
    await _wait_for_n_loop_cycles(2)
    assert len(group) == 0


@gen_test()
async def test_async_task_group_close_prohibits_new_tasks():
    group = AsyncTaskGroup()
    group.close()

    ev = asyncio.Event()
    flag = False

    async def set_flag():
        nonlocal flag
        await ev.wait()
        flag = True
        return True

    with pytest.raises(AsyncTaskGroupClosedError):
        group.call_soon(set_flag)
    assert len(group) == 0

    with pytest.raises(AsyncTaskGroupClosedError):
        group.call_later(1, set_flag)
    assert len(group) == 0

    await asyncio.sleep(0.01)
    assert not flag


@gen_test()
async def test_async_task_group_stop_disallows_shutdown():
    group = AsyncTaskGroup()

    task = None

    async def set_flag():
        nonlocal task
        task = asyncio.current_task()

    assert group.call_soon(set_flag) is None
    assert len(group) == 1
    # tasks are not given a grace period, and are not even allowed to start
    # if the group is closed immediately
    await group.stop()
    assert task is None


@gen_test()
async def test_async_task_group_stop_cancels_long_running():
    group = AsyncTaskGroup()

    task = None
    flag = False
    started = asyncio.Event()

    async def set_flag():
        nonlocal task
        task = asyncio.current_task()
        started.set()
        await asyncio.sleep(10)
        nonlocal flag
        flag = True
        return True

    assert group.call_soon(set_flag) is None
    assert len(group) == 1
    await started.wait()
    await group.stop()
    assert task
    assert task.cancelled()
    assert not flag


@gen_test()
async def test_server():
    """
    Simple Server test.
    """

    async with Server({"ping": pingpong}) as server:
        with pytest.raises(ValueError):
            server.port
        await server.listen(8881)
        assert server.port == 8881
        assert server.address == ("tcp://%s:8881" % get_ip())
        await server

        for addr in ("127.0.0.1:8881", "tcp://127.0.0.1:8881", server.address):
            comm = await connect(addr)

            n = await comm.write({"op": "ping"})
            assert isinstance(n, int)
            assert 4 <= n <= 1000

            response = await comm.read()
            assert response == b"pong"

            await comm.write({"op": "ping", "close": True})
            response = await comm.read()
            assert response == b"pong"

            await comm.close()


@gen_test()
async def test_server_raises_on_blocked_handlers():
    async with Server({"ping": pingpong}, blocked_handlers=["ping"]) as server:
        await server.listen(8881)

        comm = await connect(server.address)
        await comm.write({"op": "ping"})
        msg = await comm.read()

        _, exception, _ = clean_exception(msg["exception"])
        assert isinstance(exception, ValueError)
        assert "'ping' handler has been explicitly disallowed" in repr(exception)

        await comm.close()


class MyServer(Server):
    default_port = 8756


@pytest.mark.slow
@gen_test()
async def test_server_listen():
    """
    Test various Server.listen() arguments and their effect.
    """
    import socket

    try:
        EXTERNAL_IP4 = get_ip()
        if has_ipv6():
            EXTERNAL_IP6 = get_ipv6()
    except socket.gaierror:
        pytest.skip("no network access")

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def listen_on(cls, *args, **kwargs):
        server = cls({})
        await server.listen(*args, **kwargs)
        try:
            yield server
        finally:
            server.stop()

    # Note server.address is the concrete, contactable address

    async with listen_on(Server, 7800) as server:
        assert server.port == 7800
        assert server.address == "tcp://%s:%d" % (EXTERNAL_IP4, server.port)
        await assert_can_connect(server.address)
        await assert_can_connect_from_everywhere_4_6(server.port)

    async with listen_on(Server) as server:
        assert server.port > 0
        assert server.address == "tcp://%s:%d" % (EXTERNAL_IP4, server.port)
        await assert_can_connect(server.address)
        await assert_can_connect_from_everywhere_4_6(server.port)

    async with listen_on(MyServer) as server:
        assert server.port == MyServer.default_port
        assert server.address == "tcp://%s:%d" % (EXTERNAL_IP4, server.port)
        await assert_can_connect(server.address)
        await assert_can_connect_from_everywhere_4_6(server.port)

    async with listen_on(Server, ("", 7801)) as server:
        assert server.port == 7801
        assert server.address == "tcp://%s:%d" % (EXTERNAL_IP4, server.port)
        await assert_can_connect(server.address)
        await assert_can_connect_from_everywhere_4_6(server.port)

    async with listen_on(Server, "tcp://:7802") as server:
        assert server.port == 7802
        assert server.address == "tcp://%s:%d" % (EXTERNAL_IP4, server.port)
        await assert_can_connect(server.address)
        await assert_can_connect_from_everywhere_4_6(server.port)

    # Only IPv4

    async with listen_on(Server, ("0.0.0.0", 7810)) as server:
        assert server.port == 7810
        assert server.address == "tcp://%s:%d" % (EXTERNAL_IP4, server.port)
        await assert_can_connect(server.address)
        await assert_can_connect_from_everywhere_4(server.port)

    async with listen_on(Server, ("127.0.0.1", 7811)) as server:
        assert server.port == 7811
        assert server.address == "tcp://127.0.0.1:%d" % server.port
        await assert_can_connect(server.address)
        await assert_can_connect_locally_4(server.port)

    async with listen_on(Server, "tcp://127.0.0.1:7812") as server:
        assert server.port == 7812
        assert server.address == "tcp://127.0.0.1:%d" % server.port
        await assert_can_connect(server.address)
        await assert_can_connect_locally_4(server.port)

    # Only IPv6

    if has_ipv6():
        async with listen_on(Server, ("::", 7813)) as server:
            assert server.port == 7813
            assert server.address == "tcp://[%s]:%d" % (EXTERNAL_IP6, server.port)
            await assert_can_connect(server.address)
            await assert_can_connect_from_everywhere_6(server.port)

        async with listen_on(Server, ("::1", 7814)) as server:
            assert server.port == 7814
            assert server.address == "tcp://[::1]:%d" % server.port
            await assert_can_connect(server.address)
            await assert_can_connect_locally_6(server.port)

        async with listen_on(Server, "tcp://[::1]:7815") as server:
            assert server.port == 7815
            assert server.address == "tcp://[::1]:%d" % server.port
            await assert_can_connect(server.address)
            await assert_can_connect_locally_6(server.port)

    # TLS

    sec = tls_security()
    async with listen_on(
        Server, "tls://", **sec.get_listen_args("scheduler")
    ) as server:
        assert server.address.startswith("tls://")
        await assert_can_connect(server.address, **sec.get_connection_args("client"))

    # InProc

    async with listen_on(Server, "inproc://") as server:
        inproc_addr1 = server.address
        assert inproc_addr1.startswith("inproc://%s/%d/" % (get_ip(), os.getpid()))
        await assert_can_connect(inproc_addr1)

        async with listen_on(Server, "inproc://") as server2:
            inproc_addr2 = server2.address
            assert inproc_addr2.startswith("inproc://%s/%d/" % (get_ip(), os.getpid()))
            await assert_can_connect(inproc_addr2)

        await assert_can_connect(inproc_addr1)
        await assert_cannot_connect(inproc_addr2)


async def check_rpc(listen_addr, rpc_addr=None, listen_args=None, connection_args=None):
    listen_args = listen_args or {}
    connection_args = connection_args or {}

    async with Server({"ping": pingpong}) as server:
        await server.listen(listen_addr, **listen_args)
        if rpc_addr is None:
            rpc_addr = server.address

        async with rpc(rpc_addr, connection_args=connection_args) as remote:
            response = await remote.ping()
            assert response == b"pong"
            assert remote.comms

            response = await remote.ping(close=True)
            assert response == b"pong"
            response = await remote.ping()
            assert response == b"pong"

        assert not remote.comms
        assert remote.status == Status.closed
        await asyncio.sleep(0)


@gen_test()
async def test_rpc_default():
    await check_rpc(8883, "127.0.0.1:8883")
    await check_rpc(8883)


@gen_test()
async def test_rpc_tcp():
    await check_rpc("tcp://:8883", "tcp://127.0.0.1:8883")
    await check_rpc("tcp://")


@gen_test()
async def test_rpc_tls():
    sec = tls_security()
    await check_rpc(
        "tcp://",
        None,
        sec.get_listen_args("scheduler"),
        sec.get_connection_args("worker"),
    )


@gen_test()
async def test_rpc_inproc():
    await check_rpc("inproc://", None)


@gen_test()
async def test_rpc_inputs():
    L = [rpc("127.0.0.1:8884"), rpc(("127.0.0.1", 8884)), rpc("tcp://127.0.0.1:8884")]

    assert all(r.address == "tcp://127.0.0.1:8884" for r in L), L

    for r in L:
        await r.close_rpc()


async def check_rpc_message_lifetime(*listen_args):
    # Issue #956: rpc arguments and result shouldn't be kept alive longer
    # than necessary
    async with Server({"echo": echo_serialize}) as server:
        await server.listen(*listen_args)

        # Sanity check
        obj = CountedObject()
        assert CountedObject.n_instances == 1
        del obj
        start = time()
        while CountedObject.n_instances != 0:
            await asyncio.sleep(0.01)
            assert time() < start + 1

        async with rpc(server.address) as remote:
            obj = CountedObject()
            res = await remote.echo(x=to_serialize(obj))
            assert isinstance(res["result"], CountedObject)
            # Make sure resource cleanup code in coroutines runs
            await asyncio.sleep(0.05)

            w1 = weakref.ref(obj)
            w2 = weakref.ref(res["result"])
            del obj, res

            assert w1() is None
            assert w2() is None
            # If additional instances were created, they were deleted as well
            assert CountedObject.n_instances == 0


@gen_test()
async def test_rpc_message_lifetime_default():
    await check_rpc_message_lifetime()


@gen_test()
async def test_rpc_message_lifetime_tcp():
    await check_rpc_message_lifetime("tcp://")


@gen_test()
async def test_rpc_message_lifetime_inproc():
    await check_rpc_message_lifetime("inproc://")


async def check_rpc_with_many_connections(listen_arg):
    async def g():
        for _ in range(10):
            await remote.ping()

    server = await Server({"ping": pingpong})
    await server.listen(listen_arg)

    async with rpc(server.address) as remote:
        for _ in range(10):
            await g()

        server.stop()

        remote.close_comms()
        assert all(comm.closed() for comm in remote.comms)


@gen_test()
async def test_rpc_with_many_connections_tcp():
    await check_rpc_with_many_connections("tcp://")


@gen_test()
async def test_rpc_with_many_connections_inproc():
    await check_rpc_with_many_connections("inproc://")


async def check_large_packets(listen_arg):
    """tornado has a 100MB cap by default"""
    async with Server({}) as server:
        await server.listen(listen_arg)

        data = b"0" * int(200e6)  # slightly more than 100MB
        async with rpc(server.address) as conn:
            result = await conn.echo(data=data)
            assert result == data

            d = {"x": data}
            result = await conn.echo(data=d)
            assert result == d


@pytest.mark.slow
@gen_test()
async def test_large_packets_tcp():
    await check_large_packets("tcp://")


@gen_test()
async def test_large_packets_inproc():
    await check_large_packets("inproc://")


async def check_identity(listen_arg):
    async with Server({}) as server:
        await server.listen(listen_arg)

        async with rpc(server.address) as remote:
            a = await remote.identity()
            b = await remote.identity()
            assert a["type"] == "Server"
            assert a["id"] == b["id"]


@gen_test()
async def test_identity_tcp():
    await check_identity("tcp://")


@gen_test()
async def test_identity_inproc():
    await check_identity("inproc://")


@gen_test()
async def test_ports():
    loop = IOLoop.current()
    async with Server({}) as server:
        for port in range(9877, 9887):
            try:
                await server.listen(port)
            except OSError:  # port already taken?
                pass
            else:
                break
        else:
            raise Exception()

        assert server.port == port

        with pytest.raises((OSError, socket.error)):
            async with Server({}) as server2:
                await server2.listen(port)

    async with Server({}) as server3:
        await server3.listen(0)
        assert isinstance(server3.port, int)
        assert server3.port > 1024


def stream_div(comm=None, x=None, y=None):
    return x / y


@gen_test()
async def test_errors():
    async with Server({"div": stream_div}) as server:
        await server.listen(0)

        async with rpc(("127.0.0.1", server.port)) as r:
            with pytest.raises(ZeroDivisionError):
                await r.div(x=1, y=0)


@gen_test()
async def test_connect_raises():
    with pytest.raises(IOError):
        await connect("127.0.0.1:58259", timeout=0.01)


@gen_test()
async def test_send_recv_args():
    async with Server({}) as server:
        await server.listen(0)

        comm = await connect(server.address)
        result = await send_recv(comm, op="echo", data=b"1")
        assert result == b"1"
        assert not comm.closed()
        result = await send_recv(comm, op="echo", data=b"2", reply=False)
        assert result is None
        assert not comm.closed()
        result = await send_recv(comm, op="echo", data=b"3", close=True)
        assert result == b"3"
        assert comm.closed()


@gen_test(timeout=5)
async def test_send_recv_cancelled():
    """Test that the comm channel is closed on CancelledError"""

    async def get_stuck(comm):
        await asyncio.Future()

    async with Server({"get_stuck": get_stuck}) as server:
        await server.listen(0)

        client_comm = await connect(server.address, deserialize=False)
        while not server._comms:
            await asyncio.sleep(0.01)
        server_comm = next(iter(server._comms))

        with pytest.raises(asyncio.TimeoutError):
            await wait_for(send_recv(client_comm, op="get_stuck"), timeout=0.1)
        assert client_comm.closed()
        while not server_comm.closed():
            await asyncio.sleep(0.01)


def test_coerce_to_address():
    for arg in ["127.0.0.1:8786", ("127.0.0.1", 8786), ("127.0.0.1", "8786")]:
        assert coerce_to_address(arg) == "tcp://127.0.0.1:8786"


@gen_test()
async def test_connection_pool():
    async def ping(comm, delay=0.1):
        await asyncio.sleep(delay)
        return "pong"

    servers = [Server({"ping": ping}) for i in range(10)]
    for server in servers:
        await server
        await server.listen(0)

    rpc = await ConnectionPool(limit=5)

    # Reuse connections
    await asyncio.gather(
        *(rpc(ip="127.0.0.1", port=s.port).ping() for s in servers[:5])
    )
    await asyncio.gather(*(rpc(s.address).ping() for s in servers[:5]))
    await asyncio.gather(*(rpc(f"127.0.0.1:{s.port}").ping() for s in servers[:5]))
    await asyncio.gather(
        *(rpc(ip="127.0.0.1", port=s.port).ping() for s in servers[:5])
    )
    await asyncio.gather(*(rpc(("127.0.0.1", s.port)).ping() for s in servers[:5]))

    assert sum(map(len, rpc.available.values())) == 5
    assert sum(map(len, rpc.occupied.values())) == 0
    assert rpc.active == 0
    assert rpc.open == 5

    # Clear out connections to make room for more
    await asyncio.gather(
        *(rpc(ip="127.0.0.1", port=s.port).ping() for s in servers[5:])
    )
    assert rpc.active == 0
    assert rpc.open == 5

    s = servers[0]
    await asyncio.gather(
        *(rpc(ip="127.0.0.1", port=s.port).ping(delay=0.1) for i in range(3))
    )
    assert len(rpc.available["tcp://127.0.0.1:%d" % s.port]) == 3

    # Explicitly clear out connections
    rpc.collect()
    start = time()
    while any(rpc.available.values()):
        await asyncio.sleep(0.01)
        assert time() < start + 2

    await rpc.close()
    with pytest.raises(RuntimeError, match=r"ConnectionPool is closed"):
        await rpc(ip="127.0.0.1", port=s.port).ping()

    await asyncio.gather(*[server.close() for server in servers])


@gen_test()
async def test_connection_pool_close_while_connecting(monkeypatch):
    """
    Ensure a closed connection pool guarantees to have no connections left open
    even if it is closed mid-connecting
    """
    from distributed.comm.registry import backends
    from distributed.comm.tcp import TCPBackend, TCPConnector

    class SlowConnector(TCPConnector):
        async def connect(self, address, deserialize, **connection_args):
            await asyncio.sleep(10000)
            return await super().connect(
                address, deserialize=deserialize, **connection_args
            )

    class SlowBackend(TCPBackend):
        _connector_class = SlowConnector

    monkeypatch.setitem(backends, "tcp", SlowBackend())

    async with Server({}) as server:
        await server.listen("tcp://")

        pool = await ConnectionPool(limit=2)

        async def connect_to_server():
            comm = await pool.connect(server.address)
            pool.reuse(server.address, comm)

        # #tasks > limit
        tasks = [asyncio.create_task(connect_to_server()) for _ in range(5)]

        while not pool._connecting:
            await asyncio.sleep(0.01)

        await pool.close()
        for t in tasks:
            with pytest.raises(CommClosedError):
                await t
        assert not pool.open
        assert not pool._n_connecting


@gen_test()
async def test_connection_pool_outside_cancellation(monkeypatch):
    # Ensure cancellation errors are properly reraised
    from distributed.comm.registry import backends
    from distributed.comm.tcp import TCPBackend, TCPConnector

    class SlowConnector(TCPConnector):
        async def connect(self, address, deserialize, **connection_args):
            await asyncio.sleep(10000)
            return await super().connect(
                address, deserialize=deserialize, **connection_args
            )

    class SlowBackend(TCPBackend):
        _connector_class = SlowConnector

    monkeypatch.setitem(backends, "tcp", SlowBackend())

    async with Server({}) as server:
        await server.listen("tcp://")
        pool = await ConnectionPool(limit=2)

        async def connect_to_server():
            comm = await pool.connect(server.address)
            pool.reuse(server.address, comm)

        # #tasks > limit
        tasks = [asyncio.create_task(connect_to_server()) for _ in range(5)]
        while not pool._connecting:
            await asyncio.sleep(0.01)

        for t in tasks:
            t.cancel()

        done, _ = await asyncio.wait(tasks)
        assert all(t.cancelled() for t in tasks)


@gen_test()
async def test_remove_cancels_connect_attempts():
    loop = asyncio.get_running_loop()
    connect_started = asyncio.Event()
    connect_finished = loop.create_future()

    async def connect(*args, **kwargs):
        connect_started.set()
        await connect_finished

    async def connect_to_server():
        with pytest.raises(CommClosedError, match="Address removed."):
            await rpc.connect("tcp://0.0.0.0")

    async def remove_address():
        await connect_started.wait()
        rpc.remove("tcp://0.0.0.0")

    rpc = await ConnectionPool(limit=1)

    with mock.patch("distributed.core.connect", connect):
        await asyncio.gather(
            connect_to_server(),
            remove_address(),
        )
    assert connect_finished.cancelled()


@gen_test()
async def test_remove_cancels_connect_before_task_running():
    loop = asyncio.get_running_loop()
    connect_finished = loop.create_future()

    async def connect(*args, **kwargs):
        await connect_finished

    async def connect_to_server():
        with pytest.raises(CommClosedError, match="Address removed."):
            await rpc.connect("tcp://0.0.0.0")
        return True

    rpc = await ConnectionPool(limit=1)
    with mock.patch("distributed.core.connect", connect):
        t1 = asyncio.create_task(connect_to_server())
        # Cancel the actual connect task before it can even run
        while not rpc._connecting:
            await asyncio.sleep(0)
        rpc.remove("tcp://0.0.0.0")

        assert await t1


@gen_test()
async def test_connection_pool_respects_limit():
    limit = 5

    async def ping(comm, delay=0.01):
        await asyncio.sleep(delay)
        return "pong"

    async def do_ping(pool, port):
        assert pool.open <= limit
        await pool(ip="127.0.0.1", port=port).ping()
        assert pool.open <= limit

    async with contextlib.AsyncExitStack() as stack:
        servers = [
            await stack.enter_async_context(Server({"ping": ping})) for i in range(10)
        ]
        for server in servers:
            await server.listen(0)

        pool = await ConnectionPool(limit=limit)
        await asyncio.gather(*(do_ping(pool, s.port) for s in servers))
        await pool.close()


@gen_test()
async def test_connection_pool_tls():
    """
    Make sure connection args are supported.
    """
    sec = tls_security()
    connection_args = sec.get_connection_args("client")
    listen_args = sec.get_listen_args("scheduler")

    async def ping(comm, delay=0.01):
        await asyncio.sleep(delay)
        return "pong"

    servers = [Server({"ping": ping}) for i in range(10)]
    for server in servers:
        await server
        await server.listen("tls://", **listen_args)

    rpc = await ConnectionPool(limit=5, connection_args=connection_args)

    await asyncio.gather(*(rpc(s.address).ping() for s in servers[:5]))
    await asyncio.gather(*(rpc(s.address).ping() for s in servers[::2]))
    await asyncio.gather(*(rpc(s.address).ping() for s in servers))
    assert rpc.active == 0

    await rpc.close()
    await asyncio.gather(*[server.close() for server in servers])


@gen_test()
async def test_connection_pool_remove():
    async def ping(comm, delay=0.01):
        await asyncio.sleep(delay)
        return "pong"

    servers = [Server({"ping": ping}) for i in range(5)]
    for server in servers:
        await server
        await server.listen(0)

    rpc = await ConnectionPool(limit=10)
    serv = servers.pop()
    await asyncio.gather(*(rpc(s.address).ping() for s in servers))
    await asyncio.gather(*(rpc(serv.address).ping() for i in range(3)))
    await rpc.connect(serv.address)
    assert sum(map(len, rpc.available.values())) == 6
    assert sum(map(len, rpc.occupied.values())) == 1
    assert rpc.active == 1
    assert rpc.open == 7

    rpc.remove(serv.address)
    assert serv.address not in rpc.available
    assert serv.address not in rpc.occupied
    assert sum(map(len, rpc.available.values())) == 4
    assert sum(map(len, rpc.occupied.values())) == 0
    assert rpc.active == 0
    assert rpc.open == 4

    rpc.collect()

    # this pattern of calls (esp. `reuse` after `remove`)
    # can happen in case of worker failures:
    comm = await rpc.connect(serv.address)
    rpc.remove(serv.address)
    rpc.reuse(serv.address, comm)

    await rpc.close()
    await asyncio.gather(*[server.close() for server in servers])


@pytest.mark.parametrize("compression", list(compressions))
@pytest.mark.parametrize("serialize", [echo_serialize, echo_no_serialize])
@gen_test()
async def test_compression(compression, serialize):
    with dask.config.set({"distributed.comm.compression": compression}):
        async with Server({"echo": serialize}) as server:
            await server.listen("tcp://")

            async with rpc(server.address) as r:
                data = b"1" * 1000000
                result = await r.echo(x=to_serialize(data))
                assert result == {"result": data}


@gen_test()
async def test_rpc_serialization():
    async with Server({"echo": echo_serialize}) as server:
        await server.listen("tcp://")

        async with rpc(server.address, serializers=["msgpack"]) as r:
            with pytest.raises(TypeError):
                await r.echo(x=to_serialize(inc))

        async with rpc(server.address, serializers=["msgpack", "pickle"]) as r:
            result = await r.echo(x=to_serialize(inc))
            assert result == {"result": inc}


@gen_test()
async def test_deserialize_error():
    async with Server({"throws": throws}) as server:
        await server.listen(0)

        comm = await connect(server.address, deserialize=False)
        with pytest.raises(Exception, match=r"RuntimeError\('hello!'\)") as info:
            await send_recv(comm, op="throws", x="foo")

        assert type(info.value) == Exception
        for c in str(info.value):
            assert c.isalpha() or c in "(',!)"  # no crazy bytestrings

        await comm.close()


@gen_test()
async def test_connection_pool_detects_remote_close():
    async with Server({"ping": pingpong}) as server:
        await server.listen("tcp://")

        # open a connection, use it and give it back to the pool
        p = await ConnectionPool(limit=10)
        conn = await p.connect(server.address)
        await send_recv(conn, op="ping")
        p.reuse(server.address, conn)

        # now close this connection on the *server*
        assert len(server._comms) == 1
        server_conn = list(server._comms.keys())[0]
        await server_conn.close()

        # give the ConnectionPool some time to realize that the connection is closed
        await asyncio.sleep(0.1)

        # the connection pool should not hand out `conn` again
        conn2 = await p.connect(server.address)
        assert conn2 is not conn
        p.reuse(server.address, conn2)
        # check that `conn` has ben removed from the internal data structures
        assert p.open == 1 and p.active == 0

        # check connection pool invariants hold even after it detects a closed connection
        # while creating conn2:
        p._validate()
        await p.close()


@gen_test()
async def test_close_properly():
    """
    If the server is closed we should cancel all still ongoing coros and close
    all listeners.
    GH4704
    """

    sleep_started = asyncio.Event()

    async def sleep(comm=None):
        # We want to ensure this is actually canceled therefore don't give it a
        # chance to actually complete
        sleep_started.set()
        await asyncio.sleep(2000000)

    server = await Server({"sleep": sleep})
    assert not server.stopped
    ports = [8881, 8882, 8883]

    # Previously we close *one* listener, therefore ensure we always use more
    # than one for this test
    assert len(ports) > 1
    for port in ports:
        await server.listen(port)
    # We use TCP here for simplicity. If they are closed we should expect other
    # backends to close properly as well
    ip = get_ip()
    rpc_addr = f"tcp://{ip}:{ports[-1]}"
    async with rpc(rpc_addr) as remote:
        comm = await remote.live_comm()
        await comm.write({"op": "sleep"})
        await sleep_started.wait()

        listeners = server.listeners
        assert len(listeners) == len(ports)

        for port in ports:
            await assert_can_connect(f"tcp://{ip}:{port}")

        await server.close()

        for port in ports:
            await assert_cannot_connect(f"tcp://{ip}:{port}")

        # weakref set/dict should be cleaned up
        assert not len(server._handle_comm_tasks)


@gen_test()
async def test_server_redundant_kwarg():
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        await Server({}, typo_kwarg="foo")


@gen_test()
async def test_server_comms_mark_active_handlers():
    """Whether handlers are active can be read off of the self._comms values.
    ensure this is properly reflected and released. The sentinel for
    "open comm but no active handler" is `None`
    """

    async def long_handler(comm):
        await asyncio.sleep(0.2)
        return "done"

    async with Server({"wait": long_handler}) as server:
        await server.listen(0)
        assert server._comms == {}

        comm = await connect(server.address)
        await comm.write({"op": "wait"})
        while not server._comms:
            await asyncio.sleep(0.05)
        assert set(server._comms.values()) == {"wait"}

        assert server.incoming_comms_open == 1
        assert server.incoming_comms_active == 1

        def validate_dict(server):
            assert (
                server.get_connection_counters()["incoming_comms_open"]
                == server.incoming_comms_open
            )
            assert (
                server.get_connection_counters()["incoming_comms_active"]
                == server.incoming_comms_active
            )

            assert (
                server.get_connection_counters()["outgoing_comms_open"]
                == server.rpc.open
            )
            assert (
                server.get_connection_counters()["outgoing_comms_active"]
                == server.rpc.active
            )

        validate_dict(server)

        assert await comm.read() == "done"
        assert set(server._comms.values()) == {None}
        assert server.incoming_comms_open == 1
        assert server.incoming_comms_active == 0
        validate_dict(server)
        await comm.close()

        while server._comms:
            await asyncio.sleep(0.01)
        assert server.incoming_comms_active == 0
        assert server.incoming_comms_open == 0
        validate_dict(server)

        async with Server({}) as server2:
            rpc_ = server2.rpc(server.address)
            task = asyncio.create_task(rpc_.wait())
            while not server.incoming_comms_active:
                await asyncio.sleep(0.1)
            assert server.incoming_comms_active == 1
            assert server.incoming_comms_open == 1
            assert server.outgoing_comms_active == 0
            assert server.outgoing_comms_open == 0

            assert server2.incoming_comms_active == 0
            assert server2.incoming_comms_open == 0
            assert server2.outgoing_comms_active == 1
            assert server2.outgoing_comms_open == 1
            validate_dict(server)

            await task
            assert server.incoming_comms_active == 0
            assert server.incoming_comms_open == 1
            assert server.outgoing_comms_active == 0
            assert server.outgoing_comms_open == 0

            assert server2.incoming_comms_active == 0
            assert server2.incoming_comms_open == 0
            assert server2.outgoing_comms_active == 0
            assert server2.outgoing_comms_open == 1
            validate_dict(server)


@pytest.mark.parametrize("close_via_rpc", [True, False])
@gen_test()
async def test_close_fast_without_active_handlers(close_via_rpc):
    server = await Server({})
    server.handlers["terminate"] = server.close
    await server.listen(0)
    assert server._comms == {}

    if not close_via_rpc:
        fut = server.close()
        await wait_for(fut, 0.5)
    else:
        async with rpc(server.address) as _rpc:
            fut = _rpc.terminate(reply=False)
            await wait_for(fut, 0.5)


@gen_test()
async def test_close_grace_period_for_handlers():
    async def long_handler(comm, delay=10):
        await asyncio.sleep(delay)
        return "done"

    server = await Server({"wait": long_handler})
    await server.listen(0)
    assert server._comms == {}

    comm = await connect(server.address)
    await comm.write({"op": "wait"})
    while not server._comms:
        await asyncio.sleep(0.05)
    task = asyncio.create_task(server.close())
    wait_for_close = asyncio.Event()
    task.add_done_callback(lambda _: wait_for_close.set)
    # since the handler is running for a while, the close will not immediately
    # go through. We'll give the comm about a second to close itself
    with pytest.raises(asyncio.TimeoutError):
        await wait_for(wait_for_close.wait(), 0.5)
    await task


def test_expects_comm():
    class A:
        def empty(self):
            ...

        def one_arg(self, arg):
            ...

        def comm_arg(self, comm):
            ...

        def stream_arg(self, stream):
            ...

        def two_arg(self, arg, other):
            ...

        def comm_arg_other(self, comm, other):
            ...

        def stream_arg_other(self, stream, other):
            ...

        def arg_kwarg(self, arg, other=None):
            ...

        def comm_posarg_only(self, comm, /, other):
            ...

        def comm_not_leading_position(self, other, comm):
            ...

        def stream_not_leading_position(self, other, stream):
            ...

    expected_warning = "first argument of a RPC handler `stream` is deprecated"

    instance = A()

    assert not _expects_comm(instance.empty)
    assert not _expects_comm(instance.one_arg)
    assert _expects_comm(instance.comm_arg)
    with pytest.warns(FutureWarning, match=expected_warning):
        assert _expects_comm(instance.stream_arg)
    assert not _expects_comm(instance.two_arg)
    assert _expects_comm(instance.comm_arg_other)
    with pytest.warns(FutureWarning, match=expected_warning):
        assert _expects_comm(instance.stream_arg_other)
    assert not _expects_comm(instance.arg_kwarg)
    assert _expects_comm(instance.comm_posarg_only)
    assert not _expects_comm(instance.comm_not_leading_position)

    assert not _expects_comm(instance.stream_not_leading_position)


class AsyncStopTCPListener(TCPListener):
    async def stop(self):
        await asyncio.sleep(0)
        super().stop()


class TCPAsyncListenerBackend(TCPBackend):
    _listener_class = AsyncStopTCPListener


@gen_test()
async def test_messages_are_ordered_bsend():
    ledger = []

    async def async_handler(val):
        await asyncio.sleep(0.01 * random.random())
        ledger.append(val)

    def sync_handler(val):
        ledger.append(val)

    async with Server(
        {},
        stream_handlers={
            "sync_handler": sync_handler,
            "async_handler": async_handler,
        },
    ) as s:
        await s.listen()
        comm = await connect(s.address)
        try:
            b = BatchedSend(interval=10)
            try:
                await comm.write({"op": "connection_stream"})
                b.start(comm)
                n = 100
                for ix in range(n):
                    if ix % 2:
                        b.send({"op": "sync_handler", "val": ix})
                    else:
                        b.send({"op": "async_handler", "val": ix})
                while not len(ledger) == n:
                    await asyncio.sleep(0.01)
                assert ledger == list(range(n))
            finally:
                await b.close()
        finally:
            await comm.close()


@gen_test()
async def test_messages_are_ordered_raw():
    ledger = []

    async def async_handler(val):
        await asyncio.sleep(0.01 * random.random())
        ledger.append(val)

    def sync_handler(val):
        ledger.append(val)

    async with Server(
        {},
        stream_handlers={
            "sync_handler": sync_handler,
            "async_handler": async_handler,
        },
    ) as s:
        await s.listen()
        comm = await connect(s.address)
        try:
            await comm.write({"op": "connection_stream"})
            n = 100
            for ix in range(n):
                if ix % 2:
                    await comm.write({"op": "sync_handler", "val": ix})
                else:
                    await comm.write({"op": "async_handler", "val": ix})
            while not len(ledger) == n:
                await asyncio.sleep(0.01)
            assert ledger == list(range(n))
        finally:
            await comm.close()


@pytest.mark.slow
@gen_test(timeout=180)
async def test_large_payload(caplog):
    """See also: protocol/tests/test_protocol.py::test_large_payload"""
    critical_size = 2**31 + 1  # >2 GiB
    data = b"0" * critical_size

    async with Server({"echo": echo_serialize}) as server:
        await server.listen(0)
        comm = await connect(server.address)

        # FIXME https://github.com/dask/distributed/issues/8465
        # At debug level, messages are dumped into the log. By default, pytest captures
        # all logs, which would make this test extremely expensive to run.
        with caplog.at_level(logging.INFO, logger="distributed.core"):
            # Note: if we wrap data in to_serialize, it will be sent as a buffer, which
            # is not encoded by msgpack.
            await comm.write({"op": "echo", "x": data})
            response = await comm.read()

        assert response["result"] == data
        await comm.close()

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest

import dask

from distributed.utils import wait_for

pytestmark = pytest.mark.gpu

ucp = pytest.importorskip("ucp")

from distributed import Client, Scheduler, wait
from distributed.comm import connect, listen, parse_address, ucx
from distributed.comm.core import CommClosedError
from distributed.comm.registry import backends, get_backend
from distributed.deploy.local import LocalCluster
from distributed.diagnostics.nvml import (
    device_get_count,
    get_device_index_and_uuid,
    get_device_mig_mode,
    has_cuda_context,
)
from distributed.protocol import to_serialize
from distributed.protocol.utils_test import get_host_array
from distributed.utils_test import gen_test, inc

try:
    HOST = ucp.get_address()
except Exception:
    HOST = "127.0.0.1"


def test_registered(ucx_loop):
    assert "ucx" in backends
    backend = get_backend("ucx")
    assert isinstance(backend, ucx.UCXBackend)


async def get_comm_pair(
    listen_addr=f"ucx://{HOST}", listen_args=None, connect_args=None, **kwargs
):
    listen_args = listen_args or {}
    connect_args = connect_args or {}
    q = asyncio.queues.Queue()

    async def handle_comm(comm):
        await q.put(comm)

    listener = listen(listen_addr, handle_comm, **listen_args, **kwargs)
    async with listener:
        comm = await connect(listener.contact_address, **connect_args, **kwargs)
        serv_comm = await q.get()
        return (comm, serv_comm)


@gen_test()
async def test_ping_pong(ucx_loop):
    com, serv_com = await get_comm_pair()
    msg = {"op": "ping"}
    await com.write(msg)
    result = await serv_com.read()
    assert result == msg
    result["op"] = "pong"

    await serv_com.write(result)

    result = await com.read()
    assert result == {"op": "pong"}

    await com.close()
    await serv_com.close()


@gen_test()
async def test_comm_objs(ucx_loop):
    comm, serv_comm = await get_comm_pair()

    scheme, loc = parse_address(comm.peer_address)
    assert scheme == "ucx"

    scheme, loc = parse_address(serv_comm.peer_address)
    assert scheme == "ucx"

    assert comm.peer_address == serv_comm.local_address


@gen_test()
async def test_ucx_specific(ucx_loop):
    """
    Test concrete UCX API.
    """
    # TODO:
    # 1. ensure exceptions in handle_comm fail the test
    # 2. Use dict in read / write, put seralization there.
    # 3. Test peer_address
    # 4. Test cleanup
    address = f"ucx://{HOST}:{0}"

    async def handle_comm(comm):
        msg = await comm.read()
        msg["op"] = "pong"
        await comm.write(msg)
        await comm.read()
        await comm.close()
        assert comm.closed() is True

    listener = await ucx.UCXListener(address, handle_comm)
    host, port = listener.get_host_port()
    assert host.count(".") == 3
    assert port > 0

    l = []

    async def client_communicate(key, delay=0):
        addr = "%s:%d" % (host, port)
        comm = await connect(listener.contact_address)
        # TODO: peer_address
        # assert comm.peer_address == 'ucx://' + addr
        assert comm.extra_info == {}
        msg = {"op": "ping", "data": key}
        await comm.write(msg)
        if delay:
            await asyncio.sleep(delay)
        msg = await comm.read()
        assert msg == {"op": "pong", "data": key}
        await comm.write({"op": "client closed"})
        l.append(key)
        return comm

    comm = await client_communicate(key=1234, delay=0.5)

    # Many clients at once
    N = 2
    futures = [client_communicate(key=i, delay=0.05) for i in range(N)]
    await asyncio.gather(*futures)
    assert set(l) == {1234} | set(range(N))

    listener.stop()


@gen_test()
async def test_ping_pong_data(ucx_loop):
    np = pytest.importorskip("numpy")

    data = np.ones((10, 10))

    com, serv_com = await get_comm_pair()
    msg = {"op": "ping", "data": to_serialize(data)}
    await com.write(msg)
    result = await serv_com.read()
    result["op"] = "pong"
    data2 = result.pop("data")
    np.testing.assert_array_equal(data2, data)

    await serv_com.write(result)

    result = await com.read()
    assert result == {"op": "pong"}

    await com.close()
    await serv_com.close()


@gen_test()
async def test_ucx_deserialize(ucx_loop):
    # Note we see this error on some systems with this test:
    # `socket.gaierror: [Errno -5] No address associated with hostname`
    # This may be due to a system configuration issue.
    from distributed.comm.tests.test_comms import check_deserialize

    await check_deserialize("tcp://")


@pytest.mark.parametrize(
    "g",
    [
        lambda cudf: cudf.Series([1, 2, 3]),
        lambda cudf: cudf.Series([], dtype=object),
        lambda cudf: cudf.DataFrame([], dtype=object),
        lambda cudf: cudf.DataFrame([1]).head(0),
        lambda cudf: cudf.DataFrame([1.0]).head(0),
        lambda cudf: cudf.DataFrame({"a": []}),
        lambda cudf: cudf.DataFrame({"a": ["a"]}).head(0),
        lambda cudf: cudf.DataFrame({"a": [1.0]}).head(0),
        lambda cudf: cudf.DataFrame({"a": [1]}).head(0),
        lambda cudf: cudf.DataFrame({"a": [1, 2, None], "b": [1.0, 2.0, None]}),
        lambda cudf: cudf.DataFrame({"a": ["Check", "str"], "b": ["Sup", "port"]}),
    ],
)
@gen_test()
async def test_ping_pong_cudf(ucx_loop, g):
    # if this test appears after cupy an import error arises
    # *** ImportError: /usr/lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.11'
    # not found (required by python3.7/site-packages/pyarrow/../../../libarrow.so.12)
    cudf = pytest.importorskip("cudf")
    from cudf.testing import assert_eq

    cudf_obj = g(cudf)

    com, serv_com = await get_comm_pair()
    msg = {"op": "ping", "data": to_serialize(cudf_obj)}

    await com.write(msg)
    result = await serv_com.read()

    cudf_obj_2 = result.pop("data")
    assert result["op"] == "ping"
    assert_eq(cudf_obj, cudf_obj_2)

    await com.close()
    await serv_com.close()


@pytest.mark.parametrize("shape", [(100,), (10, 10), (4947,)])
@gen_test()
async def test_ping_pong_cupy(ucx_loop, shape):
    cupy = pytest.importorskip("cupy")
    com, serv_com = await get_comm_pair()

    arr = cupy.random.random(shape)
    msg = {"op": "ping", "data": to_serialize(arr)}

    _, result = await asyncio.gather(com.write(msg), serv_com.read())
    data2 = result.pop("data")

    assert result["op"] == "ping"
    cupy.testing.assert_array_equal(arr, data2)
    await com.close()
    await serv_com.close()


@pytest.mark.slow
@pytest.mark.parametrize("n", [int(1e9), int(2.5e9)])
@gen_test()
async def test_large_cupy(ucx_loop, n, cleanup):
    cupy = pytest.importorskip("cupy")
    com, serv_com = await get_comm_pair()

    arr = cupy.ones(n, dtype="u1")
    msg = {"op": "ping", "data": to_serialize(arr)}

    _, result = await asyncio.gather(com.write(msg), serv_com.read())
    data2 = result.pop("data")

    assert result["op"] == "ping"
    assert len(data2) == len(arr)
    await com.close()
    await serv_com.close()


@gen_test()
async def test_ping_pong_numba(ucx_loop):
    np = pytest.importorskip("numpy")
    numba = pytest.importorskip("numba")
    import numba.cuda

    arr = np.arange(10)
    arr = numba.cuda.to_device(arr)

    com, serv_com = await get_comm_pair()
    msg = {"op": "ping", "data": to_serialize(arr)}

    await com.write(msg)
    result = await serv_com.read()
    data2 = result.pop("data")
    assert result["op"] == "ping"


@pytest.mark.parametrize("processes", [True, False])
@gen_test()
async def test_ucx_localcluster(ucx_loop, processes, cleanup):
    async with LocalCluster(
        protocol="ucx",
        host=HOST,
        dashboard_address=":0",
        n_workers=2,
        threads_per_worker=1,
        processes=processes,
        asynchronous=True,
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            x = client.submit(inc, 1)
            await x
            assert x.key in cluster.scheduler.tasks
            if not processes:
                assert any(w.data == {x.key: 2} for w in cluster.workers.values())
            assert len(cluster.scheduler.workers) == 2


@pytest.mark.slow
@gen_test(timeout=60)
async def test_stress(
    ucx_loop,
):
    da = pytest.importorskip("dask.array")

    chunksize = "10 MB"

    async with LocalCluster(
        protocol="ucx",
        dashboard_address=":0",
        asynchronous=True,
        host=HOST,
    ) as cluster:
        async with Client(cluster, asynchronous=True):
            rs = da.random.RandomState()
            x = rs.random((10000, 10000), chunks=(-1, chunksize))
            x = x.persist()
            await wait(x)

            for _ in range(10):
                x = x.rechunk((chunksize, -1))
                x = x.rechunk((-1, chunksize))
                x = x.persist()
                await wait(x)


@gen_test()
async def test_simple(
    ucx_loop,
):
    async with LocalCluster(
        protocol="ucx", n_workers=2, threads_per_worker=2, asynchronous=True
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            assert cluster.scheduler_address.startswith("ucx://")
            assert await client.submit(lambda x: x + 1, 10) == 11


@pytest.mark.xfail(reason="If running on Docker, requires --pid=host")
@gen_test()
async def test_cuda_context(
    ucx_loop,
):
    try:
        device_info = get_device_index_and_uuid(
            next(
                filter(
                    lambda i: get_device_mig_mode(i)[0] == 0, range(device_get_count())
                )
            )
        )
    except StopIteration:
        pytest.skip("No CUDA device in non-MIG mode available")

    with patch.dict(
        os.environ, {"CUDA_VISIBLE_DEVICES": device_info.uuid.decode("utf-8")}
    ):
        with dask.config.set({"distributed.comm.ucx.create-cuda-context": True}):
            async with LocalCluster(
                protocol="ucx", n_workers=1, asynchronous=True
            ) as cluster:
                async with Client(cluster, asynchronous=True) as client:
                    assert cluster.scheduler_address.startswith("ucx://")
                    ctx = has_cuda_context()
                    assert ctx.has_context and ctx.device_info == device_info
                    worker_cuda_context = await client.run(has_cuda_context)
                    assert len(worker_cuda_context) == 1
                    worker_cuda_context = list(worker_cuda_context.values())
                    assert (
                        worker_cuda_context[0].has_context
                        and worker_cuda_context[0].device_info == device_info
                    )


@gen_test()
async def test_transpose(
    ucx_loop,
):
    da = pytest.importorskip("dask.array")

    async with LocalCluster(
        protocol="ucx", n_workers=2, threads_per_worker=2, asynchronous=True
    ) as cluster:
        async with Client(cluster, asynchronous=True):
            assert cluster.scheduler_address.startswith("ucx://")
            x = da.ones((10000, 10000), chunks=(1000, 1000)).persist()
            await x
            y = (x + x.T).sum()
            await y


@pytest.mark.parametrize("port", [0, 1234])
@gen_test()
async def test_ucx_protocol(ucx_loop, cleanup, port):
    async with Scheduler(protocol="ucx", port=port, dashboard_address=":0") as s:
        assert s.address.startswith("ucx://")


@gen_test()
async def test_ucx_listener_ip(ucx_loop, cleanup):
    async with Scheduler(
        protocol="ucx", interface="localhost", dashboard_address=":0"
    ) as s:
        assert s.address.startswith("ucx://127.0.0.1")
        assert s.listener.ucp.server.ip == "127.0.0.1"


@pytest.mark.skipif(
    not hasattr(ucp.exceptions, "UCXUnreachable"),
    reason="Requires UCX-Py support for UCXUnreachable exception",
)
@gen_test()
async def test_ucx_unreachable(
    ucx_loop,
):
    with pytest.raises(OSError, match="Timed out trying to connect to"):
        await Client("ucx://255.255.255.255:12345", timeout=1, asynchronous=True)


@gen_test()
async def test_comm_closed_on_read_error():
    reader, writer = await get_comm_pair()

    # Depending on the UCP protocol selected, it may raise either
    # `asyncio.TimeoutError` or `CommClosedError`, so validate either one.
    with pytest.raises((asyncio.TimeoutError, CommClosedError)):
        await wait_for(reader.read(), 0.01)

    assert reader.closed()


@gen_test()
async def test_embedded_cupy_array(
    ucx_loop,
):
    cupy = pytest.importorskip("cupy")
    da = pytest.importorskip("dask.array")
    np = pytest.importorskip("numpy")

    async with LocalCluster(
        protocol="ucx", n_workers=1, threads_per_worker=1, asynchronous=True
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            assert cluster.scheduler_address.startswith("ucx://")
            a = cupy.arange(10000)
            x = da.from_array(a, chunks=(10000,))
            b = await client.compute(x)
            cupy.testing.assert_array_equal(a, b)


@gen_test()
async def test_do_not_share_buffers(ucx_loop):
    """Test that two objects with buffer interface in the same message do not share
    their buffer upon deserialization.

    See Also
    --------
    test_comms.py::test_do_not_share_buffers
    """
    np = pytest.importorskip("numpy")

    com, serv_com = await get_comm_pair()
    msg = {"data": to_serialize([np.array([1, 2]), np.array([3, 4])])}

    await com.write(msg)
    result = await serv_com.read()
    await com.close()
    await serv_com.close()

    a, b = result["data"]
    ha = get_host_array(a)
    hb = get_host_array(b)
    assert ha is not hb
    assert ha.nbytes == a.nbytes
    assert hb.nbytes == a.nbytes

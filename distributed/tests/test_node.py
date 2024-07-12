from __future__ import annotations

import asyncio
import sys
import threading

import pytest

from distributed.core import Status
from distributed.node import Node
from distributed.utils_test import captured_logger, gen_cluster, gen_test


@gen_test(config={"distributed.admin.system-monitor.gil.enabled": True})
async def test_server_close_stops_gil_monitoring():
    pytest.importorskip("gilknocker")

    node = Node()
    assert node.monitor._gilknocker.is_running
    await node.close()
    assert not node.monitor._gilknocker.is_running


@gen_test()
async def test_server_sys_path_local_directory_cleanup(tmp_path, monkeypatch):
    local_directory = str(tmp_path / "dask-scratch-space")

    # Ensure `local_directory` is removed from `sys.path` as part of the
    # `Server` shutdown process
    assert not any(i.startswith(local_directory) for i in sys.path)
    async with Node(local_directory=local_directory):
        assert sys.path[0].startswith(local_directory)
    assert not any(i.startswith(local_directory) for i in sys.path)

    # Ensure `local_directory` isn't removed from `sys.path` if it
    # was already there before the `Server` started
    monkeypatch.setattr("sys.path", [local_directory] + sys.path)
    assert sys.path[0].startswith(local_directory)
    # NOTE: `needs_workdir=False` is needed to make sure the same path added
    # to `sys.path` above is used by the `Server` (a subdirectory is created
    # by default).
    async with Node(local_directory=local_directory, needs_workdir=False):
        assert sys.path[0].startswith(local_directory)
    assert sys.path[0].startswith(local_directory)


@gen_test()
async def test_server_status_is_always_enum():
    """Assignments with strings is forbidden"""
    server = Node()
    assert isinstance(server.status, Status)
    assert server.status != Status.stopped
    server.status = Status.stopped
    assert server.status == Status.stopped
    with pytest.raises(TypeError):
        server.status = "running"


@gen_test()
async def test_server_assign_assign_enum_is_quiet():
    """That would be the default in user code"""
    node = Node()
    node.status = Status.running


@gen_test()
async def test_server_status_compare_enum_is_quiet():
    """That would be the default in user code"""
    node = Node()
    # Note: We only want to assert that this comparison does not
    # raise an error/warning. We do not want to assert its result.
    node.status == Status.running  # noqa: B015


@gen_cluster(config={"distributed.admin.tick.interval": "20 ms"})
async def test_ticks(s, a, b):
    pytest.importorskip("crick")
    await asyncio.sleep(0.1)
    c = s.digests["tick-duration"]
    assert c.size()
    assert 0.01 < c.components[0].quantile(0.5) < 0.5


@gen_cluster(config={"distributed.admin.tick.interval": "20 ms"})
async def test_tick_logging(s, a, b):
    pytest.importorskip("crick")
    from distributed import node

    old = node.tick_maximum_delay
    node.tick_maximum_delay = 0.001
    try:
        with captured_logger("distributed.core") as sio:
            await asyncio.sleep(0.1)

        text = sio.getvalue()
        assert "unresponsive" in text
        assert "Scheduler" in text or "Worker" in text
    finally:
        node.tick_maximum_delay = old


@gen_cluster()
async def test_thread_id(s, a, b):
    assert s.thread_id == a.thread_id == b.thread_id == threading.get_ident()

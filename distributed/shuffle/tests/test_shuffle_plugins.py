from __future__ import annotations

from asyncio import iscoroutinefunction

import pytest

from distributed.shuffle._scheduler_plugin import ShuffleSchedulerPlugin
from distributed.shuffle._worker_plugin import ShuffleWorkerPlugin
from distributed.utils_test import gen_cluster

pd = pytest.importorskip("pandas")
dd = pytest.importorskip("dask.dataframe")


@gen_cluster([("", 1)])
async def test_installation_on_worker(s, a):
    ext = a.extensions["shuffle"]
    assert isinstance(ext, ShuffleWorkerPlugin)
    assert a.handlers["shuffle_receive"] == ext.shuffle_receive
    assert a.handlers["shuffle_inputs_done"] == ext.shuffle_inputs_done
    assert a.stream_handlers["shuffle-fail"] == ext.shuffle_fail
    # To guarantee the correct order of operations, shuffle_fail must be synchronous.
    # See also https://github.com/dask/distributed/pull/7486#discussion_r1088857185.
    assert not iscoroutinefunction(ext.shuffle_fail)


@gen_cluster([("", 1)])
async def test_installation_on_scheduler(s, a):
    ext = s.extensions["shuffle"]
    assert isinstance(ext, ShuffleSchedulerPlugin)
    assert s.handlers["shuffle_barrier"] == ext.barrier
    assert s.handlers["shuffle_get"] == ext.get

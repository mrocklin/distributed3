from __future__ import annotations

import asyncio
import concurrent.futures
import functools
import gc
import inspect
import logging
import os
import pathlib
import pickle
import random
import subprocess
import sys
import threading
import traceback
import types
import warnings
import weakref
import zipfile
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager, nullcontext
from functools import partial
from operator import add
from threading import Semaphore
from time import sleep
from typing import Any

import psutil
import pytest
import yaml
from tlz import concat, first, identity, isdistinct, merge, pluck, valmap

import dask
import dask.bag as db
from dask import delayed
from dask.optimization import SubgraphCallable
from dask.utils import parse_timedelta, stringify, tmpfile

from distributed import (
    CancelledError,
    Event,
    LocalCluster,
    Lock,
    Nanny,
    TimeoutError,
    Worker,
    fire_and_forget,
    get_client,
    get_worker,
    performance_report,
    profile,
    secede,
)
from distributed.client import (
    Client,
    Future,
    _get_global_client,
    as_completed,
    default_client,
    ensure_default_client,
    futures_of,
    get_task_metadata,
    temp_default_client,
    tokenize,
    wait,
)
from distributed.cluster_dump import load_cluster_dump
from distributed.comm import CommClosedError
from distributed.compatibility import LINUX, WINDOWS
from distributed.core import Server, Status
from distributed.metrics import time
from distributed.scheduler import CollectTaskMetaDataPlugin, KilledWorker, Scheduler
from distributed.sizeof import sizeof
from distributed.utils import get_mp_context, is_valid_xml, sync, tmp_text
from distributed.utils_test import (
    TaskStateMetadataPlugin,
    _UnhashableCallable,
    async_wait_for,
    asyncinc,
    captured_logger,
    cluster,
    dec,
    div,
    double,
    gen_cluster,
    gen_test,
    geninc,
    get_cert,
    inc,
    map_varying,
    nodebug,
    popen,
    pristine_loop,
    randominc,
    save_sys_modules,
    slowadd,
    slowdec,
    slowinc,
    throws,
    tls_only_security,
    varying,
    wait_for,
)

pytestmark = pytest.mark.ci1


@gen_cluster(client=True)
async def test_submit(c, s, a, b):
    x = c.submit(inc, 10, key="x")
    assert not x.done()

    assert isinstance(x, Future)
    assert x.client is c

    result = await x
    assert result == 11
    assert x.done()

    y = c.submit(inc, 20, key="y")
    z = c.submit(add, x, y)

    result = await z
    assert result == 11 + 21
    s.validate_state()


@gen_cluster(client=True)
async def test_map(c, s, a, b):
    L1 = c.map(inc, range(5))
    assert len(L1) == 5
    assert isdistinct(x.key for x in L1)
    assert all(isinstance(x, Future) for x in L1)

    result = await L1[0]
    assert result == inc(0)
    assert len(s.tasks) == 5

    L2 = c.map(inc, L1)

    result = await L2[1]
    assert result == inc(inc(1))
    assert len(s.tasks) == 10
    # assert L1[0].key in s.tasks[L2[0].key]

    total = c.submit(sum, L2)
    result = await total
    assert result == sum(map(inc, map(inc, range(5))))

    L3 = c.map(add, L1, L2)
    result = await L3[1]
    assert result == inc(1) + inc(inc(1))

    L4 = c.map(add, range(3), range(4))
    results = await c.gather(L4)
    assert results == list(map(add, range(3), range(4)))

    def f(x, y=10):
        return x + y

    L5 = c.map(f, range(5), y=5)
    results = await c.gather(L5)
    assert results == list(range(5, 10))

    y = c.submit(f, 10)
    L6 = c.map(f, range(5), y=y)
    results = await c.gather(L6)
    assert results == list(range(20, 25))
    s.validate_state()


@gen_cluster(client=True)
async def test_map_empty(c, s, a, b):
    L1 = c.map(inc, [], pure=False)
    assert len(L1) == 0
    results = await c.gather(L1)
    assert results == []


@gen_cluster(client=True)
async def test_map_keynames(c, s, a, b):
    futures = c.map(inc, range(4), key="INC")
    assert all(f.key.startswith("INC") for f in futures)
    assert isdistinct(f.key for f in futures)

    futures2 = c.map(inc, [5, 6, 7, 8], key="INC")
    assert [f.key for f in futures] != [f.key for f in futures2]

    keys = ["inc-1", "inc-2", "inc-3", "inc-4"]
    futures = c.map(inc, range(4), key=keys)
    assert [f.key for f in futures] == keys


@gen_cluster(client=True)
async def test_map_retries(c, s, a, b):
    args = [
        [ZeroDivisionError("one"), 2, 3],
        [4, 5, 6],
        [ZeroDivisionError("seven"), ZeroDivisionError("eight"), 9],
    ]

    x, y, z = c.map(*map_varying(args), retries=2)
    assert await x == 2
    assert await y == 4
    assert await z == 9

    x, y, z = c.map(*map_varying(args), retries=1, pure=False)
    assert await x == 2
    assert await y == 4
    with pytest.raises(ZeroDivisionError, match="eight"):
        await z

    x, y, z = c.map(*map_varying(args), retries=0, pure=False)
    with pytest.raises(ZeroDivisionError, match="one"):
        await x
    assert await y == 4
    with pytest.raises(ZeroDivisionError, match="seven"):
        await z


@gen_cluster(client=True)
async def test_map_batch_size(c, s, a, b):
    result = c.map(inc, range(100), batch_size=10)
    result = await c.gather(result)
    assert result == list(range(1, 101))

    result = c.map(add, range(100), range(100), batch_size=10)
    result = await c.gather(result)
    assert result == list(range(0, 200, 2))

    # mismatch shape
    result = c.map(add, range(100, 200), range(10), batch_size=2)
    result = await c.gather(result)
    assert result == list(range(100, 120, 2))


@gen_cluster(client=True)
async def test_custom_key_with_batches(c, s, a, b):
    """Test of <https://github.com/dask/distributed/issues/4588>"""

    futs = c.map(
        lambda x: x**2,
        range(10),
        batch_size=5,
        key=[str(x) for x in range(10)],
    )
    assert len(futs) == 10
    await wait(futs)


@gen_cluster(client=True)
async def test_compute_retries(c, s, a, b):
    args = [ZeroDivisionError("one"), ZeroDivisionError("two"), 3]

    # Sanity check for varying() use
    x = c.compute(delayed(varying(args))())
    with pytest.raises(ZeroDivisionError, match="one"):
        await x

    # Same retries for all
    x = c.compute(delayed(varying(args))(), retries=1)
    with pytest.raises(ZeroDivisionError, match="two"):
        await x

    x = c.compute(delayed(varying(args))(), retries=2)
    assert await x == 3

    args.append(4)
    x = c.compute(delayed(varying(args))(), retries=2)
    assert await x == 3


@gen_cluster(client=True)
async def test_compute_retries_annotations(c, s, a, b):
    # Per-future retries
    xargs = [ZeroDivisionError("one"), ZeroDivisionError("two"), 30, 40]
    yargs = [ZeroDivisionError("five"), ZeroDivisionError("six"), 70]
    zargs = [80, 90, 100]

    with dask.annotate(retries=2):
        x = delayed(varying(xargs))()
    y = delayed(varying(yargs))()

    x, y = c.compute([x, y], optimize_graph=False)

    assert await x == 30
    with pytest.raises(ZeroDivisionError, match="five"):
        await y

    x = delayed(varying(xargs))()
    with dask.annotate(retries=2):
        y = delayed(varying(yargs))()
        z = delayed(varying(zargs))()

    x, y, z = c.compute([x, y, z], optimize_graph=False)

    with pytest.raises(ZeroDivisionError, match="one"):
        await x
    assert await y == 70
    assert await z == 80


def test_retries_get(c):
    args = [ZeroDivisionError("one"), ZeroDivisionError("two"), 3]
    x = delayed(varying(args))()
    assert x.compute(retries=5) == 3

    args = [ZeroDivisionError("one"), ZeroDivisionError("two"), 3]
    x = delayed(varying(args))()
    with pytest.raises(ZeroDivisionError):
        x.compute()


@gen_cluster(client=True)
async def test_compute_persisted_retries(c, s, a, b):
    args = [ZeroDivisionError("one"), ZeroDivisionError("two"), 3]

    # Sanity check
    x = c.persist(delayed(varying(args))())
    fut = c.compute(x)
    with pytest.raises(ZeroDivisionError, match="one"):
        await fut

    x = c.persist(delayed(varying(args))())
    fut = c.compute(x, retries=1)
    with pytest.raises(ZeroDivisionError, match="two"):
        await fut

    x = c.persist(delayed(varying(args))())
    fut = c.compute(x, retries=2)
    assert await fut == 3

    args.append(4)
    x = c.persist(delayed(varying(args))())
    fut = c.compute(x, retries=3)
    assert await fut == 3


@gen_cluster(client=True)
async def test_persist_retries(c, s, a, b):
    # Same retries for all
    args = [ZeroDivisionError("one"), ZeroDivisionError("two"), 3]

    x = c.persist(delayed(varying(args))(), retries=1)
    x = c.compute(x)
    with pytest.raises(ZeroDivisionError, match="two"):
        await x

    x = c.persist(delayed(varying(args))(), retries=2)
    x = c.compute(x)
    assert await x == 3


@gen_cluster(client=True)
async def test_persist_retries_annotations(c, s, a, b):
    # Per-key retries
    xargs = [ZeroDivisionError("one"), ZeroDivisionError("two"), 30, 40]
    yargs = [ZeroDivisionError("five"), ZeroDivisionError("six"), 70]
    zargs = [80, 90, 100]

    x = delayed(varying(xargs))()
    with dask.annotate(retries=2):
        y = delayed(varying(yargs))()
        z = delayed(varying(zargs))()

    x, y, z = c.persist([x, y, z], optimize_graph=False)
    x, y, z = c.compute([x, y, z])

    with pytest.raises(ZeroDivisionError, match="one"):
        await x
    assert await y == 70
    assert await z == 80


@gen_cluster(client=True)
async def test_retries_dask_array(c, s, a, b):
    da = pytest.importorskip("dask.array")
    x = da.ones((10, 10), chunks=(3, 3))
    future = c.compute(x.sum(), retries=2)
    y = await future
    assert y == 100


@gen_cluster(client=True)
async def test_future_repr(c, s, a, b):
    pd = pytest.importorskip("pandas")
    x = c.submit(inc, 10)
    y = c.submit(pd.DataFrame, {"x": [1, 2, 3]})
    await x
    await y

    for func in [repr, lambda x: x._repr_html_()]:
        assert str(x.key) in func(x)
        assert str(x.status) in func(x)
        assert str(x.status) in repr(c.futures[x.key])

        assert "int" in func(x)
        assert "pandas" in func(y)
        assert "DataFrame" in func(y)


@gen_cluster(client=True)
async def test_future_tuple_repr(c, s, a, b):
    da = pytest.importorskip("dask.array")
    y = da.arange(10, chunks=(5,)).persist()
    f = futures_of(y)[0]
    for func in [repr, lambda x: x._repr_html_()]:
        for k in f.key:
            assert str(k) in func(f)


@gen_cluster(client=True)
async def test_Future_exception(c, s, a, b):
    x = c.submit(div, 1, 0)
    result = await x.exception()
    assert isinstance(result, ZeroDivisionError)

    x = c.submit(div, 1, 1)
    result = await x.exception()
    assert result is None


def test_Future_exception_sync(c):
    x = c.submit(div, 1, 0)
    assert isinstance(x.exception(), ZeroDivisionError)

    x = c.submit(div, 1, 1)
    assert x.exception() is None


@gen_cluster(client=True)
async def test_Future_release(c, s, a, b):
    # Released Futures should be removed timely from the Client
    x = c.submit(div, 1, 1)
    await x
    x.release()
    await asyncio.sleep(0)
    assert not c.futures

    x = c.submit(slowinc, 1, delay=0.5)
    x.release()
    await asyncio.sleep(0)
    assert not c.futures

    x = c.submit(div, 1, 0)
    await x.exception()
    x.release()
    await asyncio.sleep(0)
    assert not c.futures


def test_Future_release_sync(c):
    # Released Futures should be removed timely from the Client
    x = c.submit(div, 1, 1)
    x.result()
    x.release()
    wait_for(lambda: not c.futures, timeout=0.3)

    x = c.submit(slowinc, 1, delay=0.8)
    x.release()
    wait_for(lambda: not c.futures, timeout=0.3)

    x = c.submit(div, 1, 0)
    x.exception()
    x.release()
    wait_for(lambda: not c.futures, timeout=0.3)


def test_short_tracebacks(loop, c):
    tblib = pytest.importorskip("tblib")
    future = c.submit(div, 1, 0)
    try:
        future.result()
    except Exception:
        _, _, tb = sys.exc_info()
    tb = tblib.Traceback(tb).to_dict()
    n = 0

    while tb is not None:
        n += 1
        tb = tb["tb_next"]

    assert n < 5


@gen_cluster(client=True)
async def test_map_naming(c, s, a, b):
    L1 = c.map(inc, range(5))
    L2 = c.map(inc, range(5))

    assert [x.key for x in L1] == [x.key for x in L2]

    L3 = c.map(inc, [1, 1, 1, 1])
    assert len({x._state for x in L3}) == 1

    L4 = c.map(inc, [1, 1, 1, 1], pure=False)
    assert len({x._state for x in L4}) == 4


@gen_cluster(client=True)
async def test_submit_naming(c, s, a, b):
    a = c.submit(inc, 1)
    b = c.submit(inc, 1)

    assert a._state is b._state

    c = c.submit(inc, 1, pure=False)
    assert c.key != a.key


@gen_cluster(client=True)
async def test_exceptions(c, s, a, b):
    x = c.submit(div, 1, 2)
    result = await x
    assert result == 1 / 2

    x = c.submit(div, 1, 0)
    with pytest.raises(ZeroDivisionError):
        await x

    x = c.submit(div, 10, 2)  # continues to operate
    result = await x
    assert result == 10 / 2


@gen_cluster()
async def test_gc(s, a, b):
    c = await Client(s.address, asynchronous=True)

    x = c.submit(inc, 10)
    await x
    assert s.tasks[x.key].who_has
    x.__del__()
    await async_wait_for(
        lambda: x.key not in s.tasks or not s.tasks[x.key].who_has, timeout=0.3
    )

    await c.close()


def test_thread(c):
    x = c.submit(inc, 1)
    assert x.result() == 2

    x = c.submit(slowinc, 1, delay=0.3)
    with pytest.raises(TimeoutError):
        x.result(timeout="10 ms")
    assert x.result() == 2


def test_sync_exceptions(c):
    x = c.submit(div, 10, 2)
    assert x.result() == 5

    y = c.submit(div, 10, 0)
    try:
        y.result()
        assert False
    except ZeroDivisionError:
        pass

    z = c.submit(div, 10, 5)
    assert z.result() == 2


@gen_cluster(client=True)
async def test_gather(c, s, a, b):
    x = c.submit(inc, 10)
    y = c.submit(inc, x)

    result = await c.gather(x)
    assert result == 11
    result = await c.gather([x])
    assert result == [11]
    result = await c.gather({"x": x, "y": [y]})
    assert result == {"x": 11, "y": [12]}


@gen_cluster(client=True)
async def test_gather_mismatched_client(c, s, a, b):
    c2 = await Client(s.address, asynchronous=True)

    x = c.submit(inc, 10)
    y = c2.submit(inc, 5)

    with pytest.raises(ValueError, match="Futures created by another client"):
        await c.gather([x, y])


@gen_cluster(client=True)
async def test_gather_lost(c, s, a, b):
    [x] = await c.scatter([1], workers=a.address)
    y = c.submit(inc, 1, workers=b.address)

    await a.close()

    with pytest.raises(Exception):
        await c.gather([x, y])


def test_gather_sync(c):
    x = c.submit(inc, 1)
    assert c.gather(x) == 2

    y = c.submit(div, 1, 0)

    with pytest.raises(ZeroDivisionError):
        c.gather([x, y])

    [xx] = c.gather([x, y], errors="skip")
    assert xx == 2


@gen_cluster(client=True)
async def test_gather_strict(c, s, a, b):
    x = c.submit(div, 2, 1)
    y = c.submit(div, 1, 0)

    with pytest.raises(ZeroDivisionError):
        await c.gather([x, y])

    [xx] = await c.gather([x, y], errors="skip")
    assert xx == 2


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)])
async def test_gather_skip(c, s, a):
    x = c.submit(div, 1, 0, priority=10)
    y = c.submit(slowinc, 1, delay=0.5)

    with captured_logger(logging.getLogger("distributed.scheduler")) as sched:
        with captured_logger(logging.getLogger("distributed.client")) as client:
            L = await c.gather([x, y], errors="skip")
            assert L == [2]

    assert not client.getvalue()
    assert not sched.getvalue()


@gen_cluster(client=True)
async def test_limit_concurrent_gathering(c, s, a, b):
    futures = c.map(inc, range(100))
    await c.gather(futures)
    assert len(a.outgoing_transfer_log) + len(b.outgoing_transfer_log) < 100


@gen_cluster(client=True)
async def test_get(c, s, a, b):
    future = c.get({"x": (inc, 1)}, "x", sync=False)
    assert isinstance(future, Future)
    result = await future
    assert result == 2

    futures = c.get({"x": (inc, 1)}, ["x"], sync=False)
    assert isinstance(futures[0], Future)
    result = await c.gather(futures)
    assert result == [2]

    futures = c.get({}, [], sync=False)
    result = await c.gather(futures)
    assert result == []

    result = await c.get(
        {("x", 1): (inc, 1), ("x", 2): (inc, ("x", 1))}, ("x", 2), sync=False
    )
    assert result == 3


def test_get_sync(c):
    assert c.get({"x": (inc, 1)}, "x") == 2


def test_no_future_references(c):
    """Test that there are neither global references to Future objects nor circular
    references that need to be collected by gc
    """
    ws = weakref.WeakSet()
    futures = c.map(inc, range(10))
    ws.update(futures)
    del futures
    with profile.lock:
        assert not list(ws)


def test_get_sync_optimize_graph_passes_through(c):
    bag = db.range(10, npartitions=3).map(inc)
    dask.compute(bag.sum(), optimize_graph=False)


@gen_cluster(client=True)
async def test_gather_errors(c, s, a, b):
    def f(a, b):
        raise TypeError

    def g(a, b):
        raise AttributeError

    future_f = c.submit(f, 1, 2)
    future_g = c.submit(g, 1, 2)
    with pytest.raises(TypeError):
        await c.gather(future_f)
    with pytest.raises(AttributeError):
        await c.gather(future_g)

    await a.close()


@gen_cluster(client=True)
async def test_wait(c, s, a, b):
    x = c.submit(inc, 1)
    y = c.submit(inc, 1)
    z = c.submit(inc, 2)

    done, not_done = await wait([x, y, z])

    assert done == {x, y, z}
    assert not_done == set()
    assert x.status == y.status == "finished"


@gen_cluster(client=True)
async def test_wait_first_completed(c, s, a, b):
    x = c.submit(slowinc, 1)
    y = c.submit(slowinc, 1)
    z = c.submit(inc, 2)

    done, not_done = await wait([x, y, z], return_when="FIRST_COMPLETED")

    assert done == {z}
    assert not_done == {x, y}
    assert z.status == "finished"
    assert x.status == "pending"
    assert y.status == "pending"


@gen_cluster(client=True)
async def test_wait_timeout(c, s, a, b):
    future = c.submit(sleep, 0.3)
    with pytest.raises(TimeoutError):
        await wait(future, timeout=0.01)


def test_wait_sync(c):
    x = c.submit(inc, 1)
    y = c.submit(inc, 2)

    done, not_done = wait([x, y])
    assert done == {x, y}
    assert not_done == set()
    assert x.status == y.status == "finished"

    future = c.submit(sleep, 0.3)
    with pytest.raises(TimeoutError):
        wait(future, timeout=0.01)


def test_wait_informative_error_for_timeouts(c):
    x = c.submit(inc, 1)
    y = c.submit(inc, 2)

    try:
        wait(x, y)
    except Exception as e:
        assert "timeout" in str(e)
        assert "list" in str(e)


@gen_cluster(client=True)
async def test_garbage_collection(c, s, a, b):
    x = c.submit(inc, 1)
    y = c.submit(inc, 1)

    assert c.refcount[x.key] == 2
    x.__del__()
    await asyncio.sleep(0)
    assert c.refcount[x.key] == 1

    z = c.submit(inc, y)
    y.__del__()
    await asyncio.sleep(0)

    result = await z
    assert result == 3

    ykey = y.key
    y.__del__()
    await asyncio.sleep(0)
    assert ykey not in c.futures


@gen_cluster(client=True)
async def test_garbage_collection_with_scatter(c, s, a, b):
    [future] = await c.scatter([1])
    assert future.key in c.futures
    assert future.status == "finished"
    assert {cs.client_key for cs in s.tasks[future.key].who_wants} == {c.id}

    key = future.key
    assert c.refcount[key] == 1
    future.__del__()
    await asyncio.sleep(0)
    assert c.refcount[key] == 0

    while key in s.tasks and s.tasks[key].who_has:
        await asyncio.sleep(0.1)


@gen_cluster(client=True)
async def test_recompute_released_key(c, s, a, b):
    x = c.submit(inc, 100)
    result1 = await x
    xkey = x.key
    del x
    with profile.lock:
        await asyncio.sleep(0)
        assert c.refcount[xkey] == 0

    # 1 second batching needs a second action to trigger
    while xkey in s.tasks and s.tasks[xkey].who_has or xkey in a.data or xkey in b.data:
        await asyncio.sleep(0.1)

    x = c.submit(inc, 100)
    assert x.key in c.futures
    result2 = await x
    assert result1 == result2


@pytest.mark.slow
@gen_cluster(client=True)
async def test_long_tasks_dont_trigger_timeout(c, s, a, b):
    from time import sleep

    x = c.submit(sleep, 3)
    await x


@pytest.mark.skip
@gen_cluster(client=True)
async def test_missing_data_heals(c, s, a, b):
    a.validate = False
    b.validate = False
    x = c.submit(inc, 1)
    y = c.submit(inc, x)
    z = c.submit(inc, y)

    await wait([x, y, z])

    # Secretly delete y's key
    if y.key in a.data:
        del a.data[y.key]
        a.release_key(y.key, stimulus_id="test")
    if y.key in b.data:
        del b.data[y.key]
        b.release_key(y.key, stimulus_id="test")
    await asyncio.sleep(0)

    w = c.submit(add, y, z)

    result = await w
    assert result == 3 + 4


@pytest.mark.skip
@gen_cluster(client=True)
async def test_gather_robust_to_missing_data(c, s, a, b):
    a.validate = False
    b.validate = False
    x, y, z = c.map(inc, range(3))
    await wait([x, y, z])  # everything computed

    for f in [x, y]:
        for w in [a, b]:
            if f.key in w.data:
                del w.data[f.key]
                await asyncio.sleep(0)
                w.release_key(f.key, stimulus_id="test")

    xx, yy, zz = await c.gather([x, y, z])
    assert (xx, yy, zz) == (1, 2, 3)


@pytest.mark.skip
@gen_cluster(client=True)
async def test_gather_robust_to_nested_missing_data(c, s, a, b):
    a.validate = False
    b.validate = False
    w = c.submit(inc, 1)
    x = c.submit(inc, w)
    y = c.submit(inc, x)
    z = c.submit(inc, y)

    await wait([z])

    for worker in [a, b]:
        for datum in [y, z]:
            if datum.key in worker.data:
                del worker.data[datum.key]
                await asyncio.sleep(0)
                worker.release_key(datum.key, stimulus_id="test")

    result = await c.gather([z])

    assert result == [inc(inc(inc(inc(1))))]


@gen_cluster(client=True)
async def test_tokenize_on_futures(c, s, a, b):
    x = c.submit(inc, 1)
    y = c.submit(inc, 1)
    tok = tokenize(x)
    assert tokenize(x) == tokenize(x)
    assert tokenize(x) == tokenize(y)

    c.futures[x.key].finish()

    assert tok == tokenize(y)


@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
@gen_cluster([("127.0.0.1", 1), ("127.0.0.2", 2)], client=True)
async def test_restrictions_submit(c, s, a, b):
    x = c.submit(inc, 1, workers={a.ip})
    y = c.submit(inc, x, workers={b.ip})
    await wait([x, y])

    assert s.tasks[x.key].host_restrictions == {a.ip}
    assert x.key in a.data

    assert s.tasks[y.key].host_restrictions == {b.ip}
    assert y.key in b.data


@gen_cluster(client=True)
async def test_restrictions_ip_port(c, s, a, b):
    x = c.submit(inc, 1, workers={a.address})
    y = c.submit(inc, x, workers={b.address})
    await wait([x, y])

    assert s.tasks[x.key].worker_restrictions == {a.address}
    assert x.key in a.data

    assert s.tasks[y.key].worker_restrictions == {b.address}
    assert y.key in b.data


@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
@gen_cluster([("127.0.0.1", 1), ("127.0.0.2", 2)], client=True)
async def test_restrictions_map(c, s, a, b):
    L = c.map(inc, range(5), workers={a.ip})
    await wait(L)

    assert set(a.data) == {x.key for x in L}
    assert not b.data
    for x in L:
        assert s.tasks[x.key].host_restrictions == {a.ip}


@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
@gen_cluster([("127.0.0.1", 1), ("127.0.0.2", 2)], client=True)
async def test_restrictions_get(c, s, a, b):
    dsk = {"x": 1, "y": (inc, "x"), "z": (inc, "y")}

    futures = c.get(dsk, ["y", "z"], workers=a.ip, sync=False)
    result = await c.gather(futures)
    assert result == [2, 3]
    assert "y" in a.data
    assert "z" in a.data
    assert len(b.data) == 0


@gen_cluster(client=True)
async def test_restrictions_get_annotate(c, s, a, b):
    x = 1
    with dask.annotate(workers=a.address):
        y = delayed(inc)(x)
    with dask.annotate(workers=b.address):
        z = delayed(inc)(y)

    futures = c.get(z.__dask_graph__(), [y.key, z.key], sync=False)
    result = await c.gather(futures)
    assert result == [2, 3]
    assert y.key in a.data
    assert z.key in b.data


@gen_cluster(client=True)
async def dont_test_bad_restrictions_raise_exception(c, s, a, b):
    z = c.submit(inc, 2, workers={"bad-address"})
    try:
        await z
        assert False
    except ValueError as e:
        assert "bad-address" in str(e)
        assert z.key in str(e)


@gen_cluster(client=True)
async def test_remove_worker(c, s, a, b):
    L = c.map(inc, range(20))
    await wait(L)

    await b.close()

    assert b.address not in s.workers

    result = await c.gather(L)
    assert result == list(map(inc, range(20)))


@gen_cluster(nthreads=[("127.0.0.1", 1)], client=True)
async def test_errors_dont_block(c, s, w):
    L = [c.submit(inc, 1), c.submit(throws, 1), c.submit(inc, 2), c.submit(throws, 2)]

    while not (L[0].status == L[2].status == "finished"):
        await asyncio.sleep(0.01)

    result = await c.gather([L[0], L[2]])
    assert result == [2, 3]


@gen_cluster(client=True)
async def test_submit_quotes(c, s, a, b):
    def assert_list(x, z=[]):
        return isinstance(x, list) and isinstance(z, list)

    x = c.submit(assert_list, [1, 2, 3])
    result = await x
    assert result

    x = c.submit(assert_list, [1, 2, 3], z=[4, 5, 6])
    result = await x
    assert result

    x = c.submit(inc, 1)
    y = c.submit(inc, 2)
    z = c.submit(assert_list, [x, y])
    result = await z
    assert result


@gen_cluster(client=True)
async def test_map_quotes(c, s, a, b):
    def assert_list(x, z=[]):
        return isinstance(x, list) and isinstance(z, list)

    L = c.map(assert_list, [[1, 2, 3], [4]])
    result = await c.gather(L)
    assert all(result)

    L = c.map(assert_list, [[1, 2, 3], [4]], z=[10])
    result = await c.gather(L)
    assert all(result)

    L = c.map(assert_list, [[1, 2, 3], [4]], [[]] * 3)
    result = await c.gather(L)
    assert all(result)


@gen_cluster()
async def test_two_consecutive_clients_share_results(s, a, b):
    c = await Client(s.address, asynchronous=True)

    x = c.submit(random.randint, 0, 1000, pure=True)
    xx = await x

    f = await Client(s.address, asynchronous=True)

    y = f.submit(random.randint, 0, 1000, pure=True)
    yy = await y

    assert xx == yy

    await c.close()
    await f.close()


@gen_cluster(client=True)
async def test_submit_then_get_with_Future(c, s, a, b):
    x = c.submit(slowinc, 1)
    dsk = {"y": (inc, x)}

    result = await c.get(dsk, "y", sync=False)
    assert result == 3


@gen_cluster(client=True)
async def test_aliases(c, s, a, b):
    x = c.submit(inc, 1)

    dsk = {"y": x}
    result = await c.get(dsk, "y", sync=False)
    assert result == 2


@gen_cluster(client=True)
async def test_aliases_2(c, s, a, b):
    dsk_keys = [
        ({"x": (inc, 1), "y": "x", "z": "x", "w": (add, "y", "z")}, ["y", "w"]),
        ({"x": "y", "y": 1}, ["x"]),
        ({"x": 1, "y": "x", "z": "y", "w": (inc, "z")}, ["w"]),
    ]
    for dsk, keys in dsk_keys:
        result = await c.gather(c.get(dsk, keys, sync=False))
        assert list(result) == list(dask.get(dsk, keys))
        await asyncio.sleep(0)


@gen_cluster(client=True)
async def test_scatter(c, s, a, b):
    d = await c.scatter({"y": 20})
    assert isinstance(d["y"], Future)
    assert a.data.get("y") == 20 or b.data.get("y") == 20
    y_who_has = s.get_who_has(keys=["y"])["y"]
    assert a.address in y_who_has or b.address in y_who_has
    assert s.get_nbytes(summary=False) == {"y": sizeof(20)}
    yy = await c.gather([d["y"]])
    assert yy == [20]

    [x] = await c.scatter([10])
    assert isinstance(x, Future)
    assert a.data.get(x.key) == 10 or b.data.get(x.key) == 10
    xx = await c.gather([x])
    x_who_has = s.get_who_has(keys=[x.key])[x.key]
    assert s.tasks[x.key].who_has
    assert (
        s.workers[a.address] in s.tasks[x.key].who_has
        or s.workers[b.address] in s.tasks[x.key].who_has
    )
    assert s.get_nbytes(summary=False) == {"y": sizeof(20), x.key: sizeof(10)}
    assert xx == [10]

    z = c.submit(add, x, d["y"])  # submit works on Future
    result = await z
    assert result == 10 + 20
    result = await c.gather([z, x])
    assert result == [30, 10]


@gen_cluster(client=True)
async def test_scatter_types(c, s, a, b):
    d = await c.scatter({"x": 1})
    assert isinstance(d, dict)
    assert list(d) == ["x"]

    for seq in [[1], (1,), {1}, frozenset([1])]:
        L = await c.scatter(seq)
        assert isinstance(L, type(seq))
        assert len(L) == 1
        s.validate_state()

    seq = await c.scatter(range(5))
    assert isinstance(seq, list)
    assert len(seq) == 5
    s.validate_state()


@gen_cluster(client=True)
async def test_scatter_non_list(c, s, a, b):
    x = await c.scatter(1)
    assert isinstance(x, Future)
    result = await x
    assert result == 1


@gen_cluster(client=True)
async def test_scatter_tokenize_local(c, s, a, b):
    from dask.base import normalize_token

    class MyObj:
        pass

    L = []

    @normalize_token.register(MyObj)
    def f(x):
        L.append(x)
        return "x"

    obj = MyObj()

    future = await c.scatter(obj)
    assert L and L[0] is obj


@gen_cluster(client=True)
async def test_scatter_singletons(c, s, a, b):
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    for x in [1, np.ones(5), pd.DataFrame({"x": [1, 2, 3]})]:
        future = await c.scatter(x)
        result = await future
        assert str(result) == str(x)


@gen_cluster(client=True)
async def test_scatter_typename(c, s, a, b):
    future = await c.scatter(123)
    assert future.key.startswith("int")


@gen_cluster(client=True)
async def test_scatter_hash(c, s, a, b):
    x = await c.scatter(123)
    y = await c.scatter(123)
    assert x.key == y.key

    z = await c.scatter(123, hash=False)
    assert z.key != y.key


@gen_cluster(client=True)
async def test_scatter_hash_2(c, s, a, b):
    [a] = await c.scatter([1])
    [b] = await c.scatter([1])

    assert a.key == b.key
    s.validate_state()


@gen_cluster(client=True)
async def test_get_releases_data(c, s, a, b):
    await c.gather(c.get({"x": (inc, 1)}, ["x"], sync=False))
    while c.refcount["x"]:
        await asyncio.sleep(0.01)


def test_current(s, a, b):
    with Client(s["address"]) as c:
        assert Client.current() is c
    with pytest.raises(ValueError):
        Client.current()
    with Client(s["address"]) as c:
        assert Client.current() is c


def test_global_clients(loop):
    assert _get_global_client() is None
    with pytest.raises(ValueError):
        default_client()
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop) as c:
            assert _get_global_client() is c
            assert default_client() is c
            with Client(s["address"], loop=loop) as f:
                assert _get_global_client() is f
                assert default_client() is f
                assert default_client(c) is c
                assert default_client(f) is f

    assert _get_global_client() is None


@gen_cluster(client=True)
async def test_exception_on_exception(c, s, a, b):
    x = c.submit(lambda: 1 / 0)
    y = c.submit(inc, x)

    with pytest.raises(ZeroDivisionError):
        await y

    z = c.submit(inc, y)

    with pytest.raises(ZeroDivisionError):
        await z


@gen_cluster(client=True)
async def test_get_task_prefix_states(c, s, a, b):
    x = await c.submit(inc, 1)
    res = s.get_task_prefix_states()

    data = {
        "inc": {
            "erred": 0,
            "memory": 1,
            "processing": 0,
            "released": 0,
            "waiting": 0,
        }
    }
    assert res == data
    del x

    while s.get_task_prefix_states() == data:
        await asyncio.sleep(0.01)

    res = s.get_task_prefix_states()
    assert res == {}


@gen_cluster(client=True)
async def test_get_nbytes(c, s, a, b):
    [x] = await c.scatter([1])
    assert s.get_nbytes(summary=False) == {x.key: sizeof(1)}

    y = c.submit(inc, x)
    await y

    assert s.get_nbytes(summary=False) == {x.key: sizeof(1), y.key: sizeof(2)}


@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
@gen_cluster([("127.0.0.1", 1), ("127.0.0.2", 2)], client=True)
async def test_nbytes_determines_worker(c, s, a, b):
    x = c.submit(identity, 1, workers=[a.ip])
    y = c.submit(identity, tuple(range(100)), workers=[b.ip])
    await c.gather([x, y])

    z = c.submit(lambda x, y: None, x, y)
    await z
    assert s.tasks[z.key].who_has == {s.workers[b.address]}


@gen_cluster(client=True)
async def test_if_intermediates_clear_on_error(c, s, a, b):
    x = delayed(div, pure=True)(1, 0)
    y = delayed(div, pure=True)(1, 2)
    z = delayed(add, pure=True)(x, y)
    f = c.compute(z)
    with pytest.raises(ZeroDivisionError):
        await f
    s.validate_state()
    assert not any(ts.who_has for ts in s.tasks.values())


@gen_cluster(
    client=True, config={"distributed.scheduler.default-task-durations": {"f": "1ms"}}
)
async def test_pragmatic_move_small_data_to_large_data(c, s, a, b):
    np = pytest.importorskip("numpy")
    lists = c.map(np.ones, [10000] * 10, pure=False)
    sums = c.map(np.sum, lists)
    total = c.submit(sum, sums)

    def f(x, y):
        return None

    results = c.map(f, lists, [total] * 10)

    await wait([total])
    await wait(results)

    assert (
        sum(
            s.tasks[r.key].who_has.issubset(s.tasks[l.key].who_has)
            for l, r in zip(lists, results)
        )
        >= 9
    )


@gen_cluster(client=True)
async def test_get_with_non_list_key(c, s, a, b):
    dsk = {("x", 0): (inc, 1), 5: (inc, 2)}

    x = await c.get(dsk, ("x", 0), sync=False)
    y = await c.get(dsk, 5, sync=False)
    assert x == 2
    assert y == 3


@gen_cluster(client=True)
async def test_get_with_error(c, s, a, b):
    dsk = {"x": (div, 1, 0), "y": (inc, "x")}
    with pytest.raises(ZeroDivisionError):
        await c.get(dsk, "y", sync=False)


def test_get_with_error_sync(c):
    dsk = {"x": (div, 1, 0), "y": (inc, "x")}
    with pytest.raises(ZeroDivisionError):
        c.get(dsk, "y")


@gen_cluster(client=True)
async def test_directed_scatter(c, s, a, b):
    await c.scatter([1, 2, 3], workers=[a.address])
    assert len(a.data) == 3
    assert not b.data

    await c.scatter([4, 5], workers=[b.name])
    assert len(b.data) == 2


@gen_cluster(client=True)
async def test_scatter_direct(c, s, a, b):
    future = await c.scatter(123, direct=True)
    assert future.key in a.data or future.key in b.data
    assert s.tasks[future.key].who_has
    assert future.status == "finished"
    result = await future
    assert result == 123
    assert not s.counters["op"].components[0]["scatter"]

    result = await future
    assert not s.counters["op"].components[0]["gather"]

    result = await c.gather(future)
    assert not s.counters["op"].components[0]["gather"]


@gen_cluster()
async def test_scatter_direct_2(s, a, b):
    c = await Client(s.address, asynchronous=True, heartbeat_interval=10)

    last = s.clients[c.id].last_seen

    while s.clients[c.id].last_seen == last:
        await asyncio.sleep(0.10)

    await c.close()


@gen_cluster(client=True)
async def test_scatter_direct_numpy(c, s, a, b):
    np = pytest.importorskip("numpy")
    x = np.ones(5)
    future = await c.scatter(x, direct=True)
    result = await future
    assert np.allclose(x, result)
    assert not s.counters["op"].components[0]["scatter"]


@gen_cluster(client=True)
async def test_scatter_direct_broadcast(c, s, a, b):
    future2 = await c.scatter(456, direct=True, broadcast=True)
    assert future2.key in a.data
    assert future2.key in b.data
    assert s.tasks[future2.key].who_has == {s.workers[a.address], s.workers[b.address]}
    result = await future2
    assert result == 456
    assert not s.counters["op"].components[0]["scatter"]


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 4)
async def test_scatter_direct_balanced(c, s, *workers):
    futures = await c.scatter([1, 2, 3], direct=True)
    assert sorted(len(w.data) for w in workers) == [0, 1, 1, 1]


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 4)
async def test_scatter_direct_broadcast_target(c, s, *workers):
    futures = await c.scatter([123, 456], direct=True, workers=workers[0].address)
    assert futures[0].key in workers[0].data
    assert futures[1].key in workers[0].data

    futures = await c.scatter(
        [123, 456],
        direct=True,
        broadcast=True,
        workers=[w.address for w in workers[:3]],
    )
    assert (
        f.key in w.data and w.address in s.tasks[f.key].who_has
        for f in futures
        for w in workers[:3]
    )


@gen_cluster(client=True, nthreads=[])
async def test_scatter_direct_empty(c, s):
    with pytest.raises((ValueError, TimeoutError)):
        await c.scatter(123, direct=True, timeout=0.1)


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 5)
async def test_scatter_direct_spread_evenly(c, s, *workers):
    futures = []
    for i in range(10):
        future = await c.scatter(i, direct=True)
        futures.append(future)

    assert all(w.data for w in workers)


@pytest.mark.parametrize("direct", [True, False])
@pytest.mark.parametrize("broadcast", [True, False])
def test_scatter_gather_sync(c, direct, broadcast):
    futures = c.scatter([1, 2, 3], direct=direct, broadcast=broadcast)
    results = c.gather(futures, direct=direct)
    assert results == [1, 2, 3]

    delayed(inc)(1).compute(direct=direct)


@gen_cluster(client=True)
async def test_gather_direct(c, s, a, b):
    futures = await c.scatter([1, 2, 3])

    data = await c.gather(futures, direct=True)
    assert data == [1, 2, 3]


@gen_cluster(client=True)
async def test_many_submits_spread_evenly(c, s, a, b):
    L = [c.submit(inc, i) for i in range(10)]
    await wait(L)

    assert a.data and b.data


@gen_cluster(client=True)
async def test_traceback(c, s, a, b):
    x = c.submit(div, 1, 0)
    tb = await x.traceback()
    assert any("x / y" in line for line in pluck(3, traceback.extract_tb(tb)))


@gen_cluster(client=True)
async def test_get_traceback(c, s, a, b):
    try:
        await c.get({"x": (div, 1, 0)}, "x", sync=False)
    except ZeroDivisionError:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        L = traceback.format_tb(exc_traceback)
        assert any("x / y" in line for line in L)


@gen_cluster(client=True)
async def test_gather_traceback(c, s, a, b):
    x = c.submit(div, 1, 0)
    try:
        await c.gather(x)
    except ZeroDivisionError:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        L = traceback.format_tb(exc_traceback)
        assert any("x / y" in line for line in L)


def test_traceback_sync(c):
    x = c.submit(div, 1, 0)
    tb = x.traceback()
    assert any(
        "x / y" in line
        for line in concat(traceback.extract_tb(tb))
        if isinstance(line, str)
    )

    y = c.submit(inc, x)
    tb2 = y.traceback()

    assert set(pluck(3, traceback.extract_tb(tb2))).issuperset(
        set(pluck(3, traceback.extract_tb(tb)))
    )

    z = c.submit(div, 1, 2)
    tb = z.traceback()
    assert tb is None


@gen_cluster(client=True)
async def test_upload_file(c, s, a, b):
    def g():
        import myfile

        return myfile.f()

    with save_sys_modules():
        for value in [123, 456]:
            with tmp_text("myfile.py", f"def f():\n    return {value}") as fn:
                await c.upload_file(fn)

            x = c.submit(g, pure=False)
            result = await x
            assert result == value


@gen_cluster(client=True)
async def test_upload_file_refresh_delayed(c, s, a, b):
    with save_sys_modules():
        for value in [123, 456]:
            with tmp_text("myfile.py", f"def f():\n    return {value}") as fn:
                await c.upload_file(fn)

            sys.path.append(os.path.dirname(fn))
            from myfile import f

            b = delayed(f)()
            bb = c.compute(b, sync=False)
            result = await c.gather(bb)
            assert result == value


@gen_cluster(client=True)
async def test_upload_file_no_extension(c, s, a, b):
    with tmp_text("myfile", "") as fn:
        await c.upload_file(fn)


@gen_cluster(client=True)
async def test_upload_file_zip(c, s, a, b):
    def g():
        import myfile

        return myfile.f()

    with save_sys_modules():
        try:
            for value in [123, 456]:
                with tmp_text(
                    "myfile.py", f"def f():\n    return {value}"
                ) as fn_my_file:
                    with zipfile.ZipFile("myfile.zip", "w") as z:
                        z.write(fn_my_file, arcname=os.path.basename(fn_my_file))
                    await c.upload_file("myfile.zip")

                    x = c.submit(g, pure=False)
                    result = await x
                    assert result == value
        finally:
            if os.path.exists("myfile.zip"):
                os.remove("myfile.zip")


@pytest.mark.slow
@gen_cluster(client=True)
async def test_upload_file_egg(c, s, a, b):
    pytest.importorskip("setuptools")

    def g():
        import package_1
        import package_2

        return package_1.a, package_2.b

    # c.upload_file tells each worker to
    # - put this file in their local_directory
    # - modify their sys.path to include it
    # we don't care about the local_directory
    # but we do care about restoring the path

    with save_sys_modules():
        for value in [123, 456]:
            with tmpfile() as dirname:
                os.mkdir(dirname)

                with open(os.path.join(dirname, "setup.py"), "w") as f:
                    f.write("from setuptools import setup, find_packages\n")
                    f.write(
                        'setup(name="my_package", packages=find_packages(), '
                        f'version="{value}")\n'
                    )

                # test a package with an underscore in the name
                package_1 = os.path.join(dirname, "package_1")
                os.mkdir(package_1)
                with open(os.path.join(package_1, "__init__.py"), "w") as f:
                    f.write(f"a = {value}\n")

                # test multiple top-level packages
                package_2 = os.path.join(dirname, "package_2")
                os.mkdir(package_2)
                with open(os.path.join(package_2, "__init__.py"), "w") as f:
                    f.write(f"b = {value}\n")

                # compile these into an egg
                subprocess.check_call(
                    [sys.executable, "setup.py", "bdist_egg"], cwd=dirname
                )

                egg_root = os.path.join(dirname, "dist")
                # first file ending with '.egg'
                egg_name = [
                    fname for fname in os.listdir(egg_root) if fname.endswith(".egg")
                ][0]
                egg_path = os.path.join(egg_root, egg_name)

                await c.upload_file(egg_path)
                os.remove(egg_path)

                x = c.submit(g, pure=False)
                result = await x
                assert result == (value, value)


@gen_cluster(client=True)
async def test_upload_large_file(c, s, a, b):
    assert a.local_directory
    assert b.local_directory
    with tmp_text("myfile", "abc") as fn:
        with tmp_text("myfile2", "def") as fn2:
            await c._upload_large_file(fn, remote_filename="x")
            await c._upload_large_file(fn2)

            for w in [a, b]:
                assert os.path.exists(os.path.join(w.local_directory, "x"))
                assert os.path.exists(os.path.join(w.local_directory, "myfile2"))
                with open(os.path.join(w.local_directory, "x")) as f:
                    assert f.read() == "abc"
                with open(os.path.join(w.local_directory, "myfile2")) as f:
                    assert f.read() == "def"


def test_upload_file_sync(c):
    def g():
        import myfile

        return myfile.x

    with tmp_text("myfile.py", "x = 123") as fn:
        c.upload_file(fn)
        x = c.submit(g)
        assert x.result() == 123


@gen_cluster(client=True)
async def test_upload_file_exception(c, s, a, b):
    with tmp_text("myfile.py", "syntax-error!") as fn:
        with pytest.raises(SyntaxError):
            await c.upload_file(fn)


def test_upload_file_exception_sync(c):
    with tmp_text("myfile.py", "syntax-error!") as fn:
        with pytest.raises(SyntaxError):
            c.upload_file(fn)


@gen_cluster(client=True, nthreads=[])
async def test_upload_file_new_worker(c, s):
    def g():
        import myfile

        return myfile.x

    with tmp_text("myfile.py", "x = 123") as fn:
        await c.upload_file(fn)
        async with Worker(s.address):
            x = await c.submit(g)

        assert x == 123


@pytest.mark.skip
@gen_cluster()
async def test_multiple_clients(s, a, b):
    a = await Client(s.address, asynchronous=True)
    b = await Client(s.address, asynchronous=True)

    x = a.submit(inc, 1)
    y = b.submit(inc, 2)
    assert x.client is a
    assert y.client is b
    xx = await x
    yy = await y
    assert xx == 2
    assert yy == 3
    z = a.submit(add, x, y)
    assert z.client is a
    zz = await z
    assert zz == 5

    await a.close()
    await b.close()


@gen_cluster(client=True)
async def test_async_compute(c, s, a, b):
    from dask.delayed import delayed

    x = delayed(1)
    y = delayed(inc)(x)
    z = delayed(dec)(x)

    [yy, zz, aa] = c.compute([y, z, 3], sync=False)
    assert isinstance(yy, Future)
    assert isinstance(zz, Future)
    assert aa == 3

    result = await c.gather([yy, zz])
    assert result == [2, 0]

    assert isinstance(c.compute(y), Future)
    assert isinstance(c.compute([y]), (tuple, list))


@gen_cluster(client=True)
async def test_async_compute_with_scatter(c, s, a, b):
    d = await c.scatter({("x", 1): 1, ("y", 1): 2})
    x, y = d[("x", 1)], d[("y", 1)]

    from dask.delayed import delayed

    z = delayed(add)(delayed(inc)(x), delayed(inc)(y))
    zz = c.compute(z)

    [result] = await c.gather([zz])
    assert result == 2 + 3


def test_sync_compute(c):
    x = delayed(1)
    y = delayed(inc)(x)
    z = delayed(dec)(x)

    yy, zz = c.compute([y, z], sync=True)
    assert (yy, zz) == (2, 0)


@gen_cluster(client=True)
async def test_remote_scatter_gather(c, s, a, b):
    x, y, z = await c.scatter([1, 2, 3])

    assert x.key in a.data or x.key in b.data
    assert y.key in a.data or y.key in b.data
    assert z.key in a.data or z.key in b.data

    xx, yy, zz = await c.gather([x, y, z])
    assert (xx, yy, zz) == (1, 2, 3)


@gen_cluster(client=True)
async def test_remote_submit_on_Future(c, s, a, b):
    x = c.submit(lambda x: x + 1, 1)
    y = c.submit(lambda x: x + 1, x)
    result = await y
    assert result == 3


def test_start_is_idempotent(c):
    c.start()
    c.start()
    c.start()

    x = c.submit(inc, 1)
    assert x.result() == 2


@gen_cluster(client=True)
async def test_client_with_scheduler(c, s, a, b):
    x = c.submit(inc, 1)
    y = c.submit(inc, 2)
    z = c.submit(add, x, y)
    result = await x
    assert result == 1 + 1
    result = await z
    assert result == 1 + 1 + 1 + 2

    A, B, C = await c.scatter([1, 2, 3])
    AA, BB, xx = await c.gather([A, B, x])
    assert (AA, BB, xx) == (1, 2, 2)

    result = await c.get({"x": (inc, 1), "y": (add, "x", 10)}, "y", sync=False)
    assert result == 12


@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
@gen_cluster([("127.0.0.1", 1), ("127.0.0.2", 2)], client=True)
async def test_allow_restrictions(c, s, a, b):
    aws = s.workers[a.address]
    bws = s.workers[a.address]

    x = c.submit(inc, 1, workers=a.ip)
    await x
    assert s.tasks[x.key].who_has == {aws}
    assert not any(ts.loose_restrictions for ts in s.tasks.values())

    x = c.submit(inc, 2, workers=a.ip, allow_other_workers=True)
    await x
    assert s.tasks[x.key].who_has == {aws}
    assert s.tasks[x.key].loose_restrictions

    L = c.map(inc, range(3, 13), workers=a.ip, allow_other_workers=True)
    await wait(L)
    assert all(s.tasks[f.key].who_has == {aws} for f in L)
    for f in L:
        assert s.tasks[f.key].loose_restrictions

    x = c.submit(inc, 15, workers="127.0.0.3", allow_other_workers=True)

    await x
    assert s.tasks[x.key].who_has
    assert s.tasks[x.key].loose_restrictions

    L = c.map(inc, range(15, 25), workers="127.0.0.3", allow_other_workers=True)
    await wait(L)
    assert all(s.tasks[f.key].who_has for f in L)
    for f in L:
        assert s.tasks[f.key].loose_restrictions

    with pytest.raises(ValueError):
        c.submit(inc, 1, allow_other_workers=True)

    with pytest.raises(ValueError):
        c.map(inc, [1], allow_other_workers=True)

    with pytest.raises(TypeError):
        c.submit(inc, 20, workers="127.0.0.1", allow_other_workers="Hello!")

    with pytest.raises(TypeError):
        c.map(inc, [20], workers="127.0.0.1", allow_other_workers="Hello!")


def test_bad_address():
    with pytest.raises(OSError, match="connect"):
        Client("123.123.123.123:1234", timeout=0.1)
    with pytest.raises(OSError, match="connect"):
        Client("127.0.0.1:1234", timeout=0.1)


def test_informative_error_on_cluster_type():
    with pytest.raises(TypeError) as exc_info:
        Client(LocalCluster)

    assert "Scheduler address must be a string or a Cluster instance" in str(
        exc_info.value
    )


@gen_cluster(client=True)
async def test_long_error(c, s, a, b):
    def bad(x):
        raise ValueError("a" * 100000)

    x = c.submit(bad, 10)

    try:
        await x
    except ValueError as e:
        assert len(str(e)) < 100000

    tb = await x.traceback()
    assert all(
        len(line) < 100000
        for line in concat(traceback.extract_tb(tb))
        if isinstance(line, str)
    )


@gen_cluster(client=True)
async def test_map_on_futures_with_kwargs(c, s, a, b):
    def f(x, y=10):
        return x + y

    futures = c.map(inc, range(10))
    futures2 = c.map(f, futures, y=20)
    results = await c.gather(futures2)
    assert results == [i + 1 + 20 for i in range(10)]

    future = c.submit(inc, 100)
    future2 = c.submit(f, future, y=200)
    result = await future2
    assert result == 100 + 1 + 200


class BadlySerializedObject:
    def __getstate__(self):
        return 1

    def __setstate__(self, state):
        raise TypeError("hello!")


class FatallySerializedObject:
    def __getstate__(self):
        return 1

    def __setstate__(self, state):
        print("This should never have been deserialized, closing")
        import sys

        sys.exit(0)


@gen_cluster(client=True)
async def test_badly_serialized_input(c, s, a, b):
    o = BadlySerializedObject()

    future = c.submit(inc, o)
    futures = c.map(inc, range(10))

    L = await c.gather(futures)
    assert list(L) == list(map(inc, range(10)))
    assert future.status == "error"

    with pytest.raises(Exception) as info:
        await future

    assert "hello!" in str(info.value)


@pytest.mark.skip
@gen_test()
async def test_badly_serialized_input_stderr(capsys, c):
    o = BadlySerializedObject()
    future = c.submit(inc, o)

    while True:
        sleep(0.01)
        out, err = capsys.readouterr()
        if "hello!" in err:
            break

    assert future.status == "error"


def test_repr(loop):
    funcs = [str, repr, lambda x: x._repr_html_()]
    with cluster(nworkers=3, worker_kwargs={"memory_limit": "2 GiB"}) as (s, [a, b, c]):
        with Client(s["address"], loop=loop) as c:
            for func in funcs:
                text = func(c)
                assert c.scheduler.address in text
                assert "threads=3" in text or "Total threads: </strong>" in text
                assert "6.00 GiB" in text
                if "<table" not in text:
                    assert len(text) < 80

        for func in funcs:
            text = func(c)
            assert "No scheduler connected" in text


@gen_cluster(client=True)
async def test_repr_async(c, s, a, b):
    c._repr_html_()


@gen_cluster(client=True, worker_kwargs={"memory_limit": None})
async def test_repr_no_memory_limit(c, s, a, b):
    c._repr_html_()


@gen_test()
async def test_repr_localcluster():
    cluster = await LocalCluster(
        processes=False, dashboard_address=":0", asynchronous=True
    )
    client = await Client(cluster, asynchronous=True)
    try:
        text = client._repr_html_()
        assert cluster.scheduler.address in text
        assert is_valid_xml(client._repr_html_())
    finally:
        await client.close()
        await cluster.close()


@gen_cluster(client=True)
async def test_forget_simple(c, s, a, b):
    x = c.submit(inc, 1, retries=2)
    y = c.submit(inc, 2)
    z = c.submit(add, x, y, workers=[a.ip], allow_other_workers=True)

    await wait([x, y, z])
    assert not s.tasks[x.key].waiting_on
    assert not s.tasks[y.key].waiting_on

    assert set(s.tasks) == {x.key, y.key, z.key}

    s.client_releases_keys(keys=[x.key], client=c.id)
    assert x.key in s.tasks
    s.client_releases_keys(keys=[z.key], client=c.id)

    assert x.key not in s.tasks
    assert z.key not in s.tasks
    assert not s.tasks[y.key].dependents

    s.client_releases_keys(keys=[y.key], client=c.id)
    assert not s.tasks


@gen_cluster(client=True)
async def test_forget_complex(e, s, A, B):
    a, b, c, d = await e.scatter(list(range(4)))
    ab = e.submit(add, a, b)
    cd = e.submit(add, c, d)
    ac = e.submit(add, a, c)
    acab = e.submit(add, ac, ab)

    await wait([a, b, c, d, ab, ac, cd, acab])

    assert set(s.tasks) == {f.key for f in [ab, ac, cd, acab, a, b, c, d]}

    s.client_releases_keys(keys=[ab.key], client=e.id)
    assert set(s.tasks) == {f.key for f in [ab, ac, cd, acab, a, b, c, d]}

    s.client_releases_keys(keys=[b.key], client=e.id)
    assert set(s.tasks) == {f.key for f in [ac, cd, acab, a, c, d]}

    s.client_releases_keys(keys=[acab.key], client=e.id)
    assert set(s.tasks) == {f.key for f in [ac, cd, a, c, d]}
    assert b.key not in s.tasks

    while b.key in A.data or b.key in B.data:
        await asyncio.sleep(0.01)

    s.client_releases_keys(keys=[ac.key], client=e.id)
    assert set(s.tasks) == {f.key for f in [cd, a, c, d]}


@gen_cluster(client=True)
async def test_forget_in_flight(e, s, A, B):
    delayed2 = partial(delayed, pure=True)
    a, b, c, d = (delayed2(slowinc)(i) for i in range(4))
    ab = delayed2(slowadd)(a, b, dask_key_name="ab")
    cd = delayed2(slowadd)(c, d, dask_key_name="cd")
    ac = delayed2(slowadd)(a, c, dask_key_name="ac")
    acab = delayed2(slowadd)(ac, ab, dask_key_name="acab")

    x, y = e.compute([ac, acab])
    s.validate_state()

    for i in range(5):
        await asyncio.sleep(0.01)
        s.validate_state()

    s.client_releases_keys(keys=[y.key], client=e.id)
    s.validate_state()

    for k in [acab.key, ab.key, b.key]:
        assert k not in s.tasks


@gen_cluster(client=True)
async def test_forget_errors(c, s, a, b):
    x = c.submit(div, 1, 0)
    y = c.submit(inc, x)
    z = c.submit(inc, y)
    await wait([y])

    assert s.tasks[x.key].exception
    assert s.tasks[x.key].exception_blame
    assert s.tasks[y.key].exception_blame
    assert s.tasks[z.key].exception_blame

    s.client_releases_keys(keys=[z.key], client=c.id)

    assert s.tasks[x.key].exception
    assert s.tasks[x.key].exception_blame
    assert s.tasks[y.key].exception_blame
    assert z.key not in s.tasks

    s.client_releases_keys(keys=[x.key], client=c.id)

    assert s.tasks[x.key].exception
    assert s.tasks[x.key].exception_blame
    assert s.tasks[y.key].exception_blame
    assert z.key not in s.tasks

    s.client_releases_keys(keys=[y.key], client=c.id)

    assert x.key not in s.tasks
    assert y.key not in s.tasks
    assert z.key not in s.tasks


def test_repr_sync(c):
    s = str(c)
    r = repr(c)
    assert c.scheduler.address in s
    assert c.scheduler.address in r
    assert str(2) in s  # nworkers
    assert "cores" in s or "threads" in s


@gen_cluster()
async def test_multi_client(s, a, b):
    c = await Client(s.address, asynchronous=True)
    f = await Client(s.address, asynchronous=True)

    assert set(s.client_comms) == {c.id, f.id}

    x = c.submit(inc, 1)
    y = f.submit(inc, 2)
    y2 = c.submit(inc, 2)

    assert y.key == y2.key

    await wait([x, y])

    assert {ts.key for ts in s.clients[c.id].wants_what} == {x.key, y.key}
    assert {ts.key for ts in s.clients[f.id].wants_what} == {y.key}

    await c.close()

    while c.id in s.clients:
        await asyncio.sleep(0.01)

    assert c.id not in s.clients
    assert c.id not in s.tasks[y.key].who_wants
    assert x.key not in s.tasks

    await f.close()

    while s.tasks:
        await asyncio.sleep(0.01)


def long_running_client_connection(address):
    with pristine_loop():
        c = Client(address)
        x = c.submit(lambda x: x + 1, 10)
        x.result()
        sleep(100)


@gen_cluster()
async def test_cleanup_after_broken_client_connection(s, a, b):
    proc = get_mp_context().Process(
        target=long_running_client_connection, args=(s.address,)
    )
    proc.daemon = True
    proc.start()

    while not s.tasks:
        await asyncio.sleep(0.01)

    proc.terminate()

    while s.tasks:
        await asyncio.sleep(0.01)


@gen_cluster()
async def test_multi_garbage_collection(s, a, b):
    c = await Client(s.address, asynchronous=True)

    f = await Client(s.address, asynchronous=True)

    x = c.submit(inc, 1)
    y = f.submit(inc, 2)
    y2 = c.submit(inc, 2)

    assert y.key == y2.key

    await wait([x, y])

    x.__del__()
    while x.key in a.data or x.key in b.data:
        await asyncio.sleep(0.01)

    assert {ts.key for ts in s.clients[c.id].wants_what} == {y.key}
    assert {ts.key for ts in s.clients[f.id].wants_what} == {y.key}

    y.__del__()
    while x.key in {ts.key for ts in s.clients[f.id].wants_what}:
        await asyncio.sleep(0.01)

    await asyncio.sleep(0.1)
    assert y.key in a.data or y.key in b.data
    assert {ts.key for ts in s.clients[c.id].wants_what} == {y.key}
    assert not s.clients[f.id].wants_what

    y2.__del__()
    while y.key in a.data or y.key in b.data:
        await asyncio.sleep(0.01)

    assert not s.tasks

    await c.close()
    await f.close()


@gen_cluster(client=True)
async def test__broadcast(c, s, a, b):
    x, y = await c.scatter([1, 2], broadcast=True)
    assert a.data == b.data == {x.key: 1, y.key: 2}


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 4)
async def test__broadcast_integer(c, s, *workers):
    x, y = await c.scatter([1, 2], broadcast=2)
    assert len(s.tasks[x.key].who_has) == 2
    assert len(s.tasks[y.key].who_has) == 2


@gen_cluster(client=True)
async def test__broadcast_dict(c, s, a, b):
    d = await c.scatter({"x": 1}, broadcast=True)
    assert a.data == b.data == {"x": 1}


@gen_cluster(client=True)
async def test_proxy(c, s, a, b):
    msg = await c.scheduler.proxy(msg={"op": "identity"}, worker=a.address)
    assert msg["id"] == a.identity()["id"]


@gen_cluster(client=True)
async def test_cancel(c, s, a, b):
    x = c.submit(slowinc, 1)
    y = c.submit(slowinc, x)

    while y.key not in s.tasks:
        await asyncio.sleep(0.01)

    await c.cancel([x])

    assert x.cancelled()
    assert "cancel" in str(x)
    s.validate_state()

    while not y.cancelled():
        await asyncio.sleep(0.01)

    assert not s.tasks
    s.validate_state()


@gen_cluster(client=True)
async def test_cancel_tuple_key(c, s, a, b):
    x = c.submit(inc, 1, key=("x", 0, 1))
    await x
    await c.cancel(x)
    with pytest.raises(CancelledError):
        await x


@gen_cluster()
async def test_cancel_multi_client(s, a, b):
    async with Client(s.address, asynchronous=True, name="c") as c:
        async with Client(s.address, asynchronous=True, name="f") as f:

            x = c.submit(slowinc, 1)
            y = f.submit(slowinc, 1)

            assert x.key == y.key

            # Ensure both clients are known to the scheduler.
            await y
            await x

            await c.cancel([x])

            # Give the scheduler time to pass messages
            await asyncio.sleep(0.1)

            assert x.cancelled()
            assert not y.cancelled()

            out = await y
            assert out == 2

            with pytest.raises(CancelledError):
                await x


@gen_cluster(nthreads=[("", 1)], client=True)
async def test_cancel_before_known_to_scheduler(c, s, a):
    with captured_logger("distributed.scheduler") as slogs:
        f = c.submit(inc, 1)
        await c.cancel([f])

        with pytest.raises(CancelledError):
            await f

        assert "Scheduler cancels key" in slogs.getvalue()


@gen_cluster(client=True)
async def test_cancel_collection(c, s, a, b):
    L = c.map(double, [[1], [2], [3]])
    x = db.Bag({("b", i): f for i, f in enumerate(L)}, "b", 3)

    await c.cancel(x)
    await c.cancel([x])
    assert all(f.cancelled() for f in L)
    while s.tasks:
        await asyncio.sleep(0.01)


def test_cancel_sync(c):
    x = c.submit(slowinc, 1, key="x")
    y = c.submit(slowinc, x, key="y")
    z = c.submit(slowinc, y, key="z")

    c.cancel([y])

    start = time()
    while not z.cancelled():
        sleep(0.01)
        assert time() < start + 30

    assert x.result() == 2

    z.cancel()
    assert z.cancelled()


@gen_cluster(client=True)
async def test_future_type(c, s, a, b):
    x = c.submit(inc, 1)
    await wait([x])
    assert x.type == int
    assert "int" in str(x)


@gen_cluster(client=True)
async def test_traceback_clean(c, s, a, b):
    x = c.submit(div, 1, 0)
    try:
        await x
    except Exception as e:
        f = e
        exc_type, exc_value, tb = sys.exc_info()
        while tb:
            assert "scheduler" not in tb.tb_frame.f_code.co_filename
            assert "worker" not in tb.tb_frame.f_code.co_filename
            tb = tb.tb_next


@gen_cluster(client=True)
async def test_map_differnet_lengths(c, s, a, b):
    assert len(c.map(add, [1, 2], [1, 2, 3])) == 2


def test_Future_exception_sync_2(loop, capsys):
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop) as c:
            assert dask.base.get_scheduler() == c.get

    out, err = capsys.readouterr()
    assert len(out.strip().split("\n")) == 1

    assert dask.base.get_scheduler() != c.get


@gen_cluster(timeout=60, client=True)
async def test_async_persist(c, s, a, b):
    from dask.delayed import Delayed, delayed

    x = delayed(1)
    y = delayed(inc)(x)
    z = delayed(dec)(x)
    w = delayed(add)(y, z)

    yy, ww = c.persist([y, w])
    assert type(yy) == type(y)
    assert type(ww) == type(w)
    assert len(yy.dask) == 1
    assert len(ww.dask) == 1
    assert len(w.dask) > 1
    assert y.__dask_keys__() == yy.__dask_keys__()
    assert w.__dask_keys__() == ww.__dask_keys__()

    while y.key not in s.tasks and w.key not in s.tasks:
        await asyncio.sleep(0.01)

    assert {cs.client_key for cs in s.tasks[y.key].who_wants} == {c.id}
    assert {cs.client_key for cs in s.tasks[w.key].who_wants} == {c.id}

    yyf, wwf = c.compute([yy, ww])
    yyy, www = await c.gather([yyf, wwf])
    assert yyy == inc(1)
    assert www == add(inc(1), dec(1))

    assert isinstance(c.persist(y), Delayed)
    assert isinstance(c.persist([y]), (list, tuple))


@gen_cluster(client=True)
async def test__persist(c, s, a, b):
    pytest.importorskip("dask.array")
    import dask.array as da

    x = da.ones((10, 10), chunks=(5, 10))
    y = 2 * (x + 1)
    assert len(y.dask) == 6
    yy = c.persist(y)

    assert len(y.dask) == 6
    assert len(yy.dask) == 2
    assert all(isinstance(v, Future) for v in yy.dask.values())
    assert yy.__dask_keys__() == y.__dask_keys__()

    g, h = c.compute([y, yy])

    gg, hh = await c.gather([g, h])
    assert (gg == hh).all()


def test_persist(c):
    pytest.importorskip("dask.array")
    import dask.array as da

    x = da.ones((10, 10), chunks=(5, 10))
    y = 2 * (x + 1)
    assert len(y.dask) == 6
    yy = c.persist(y)
    assert len(y.dask) == 6
    assert len(yy.dask) == 2
    assert all(isinstance(v, Future) for v in yy.dask.values())
    assert yy.__dask_keys__() == y.__dask_keys__()

    zz = yy.compute()
    z = y.compute()
    assert (zz == z).all()


@gen_cluster(timeout=60, client=True)
async def test_long_traceback(c, s, a, b):
    from distributed.protocol.pickle import dumps

    def deep(n):
        if n == 0:
            1 / 0
        else:
            return deep(n - 1)

    x = c.submit(deep, 200)
    await wait([x])
    assert len(dumps(c.futures[x.key].traceback)) < 10000
    assert isinstance(c.futures[x.key].exception, ZeroDivisionError)


@gen_cluster(client=True)
async def test_wait_on_collections(c, s, a, b):
    L = c.map(double, [[1], [2], [3]])
    x = db.Bag({("b", i): f for i, f in enumerate(L)}, "b", 3)

    await wait(x)
    assert all(f.key in a.data or f.key in b.data for f in L)


@gen_cluster(client=True)
async def test_futures_of_get(c, s, a, b):
    x, y, z = c.map(inc, [1, 2, 3])

    assert set(futures_of(0)) == set()
    assert set(futures_of(x)) == {x}
    assert set(futures_of([x, y, z])) == {x, y, z}
    assert set(futures_of([x, [y], [[z]]])) == {x, y, z}
    assert set(futures_of({"x": x, "y": [y]})) == {x, y}

    b = db.Bag({("b", i): f for i, f in enumerate([x, y, z])}, "b", 3)
    assert set(futures_of(b)) == {x, y, z}

    sg = SubgraphCallable(
        {"x": x, "y": y, "z": z, "out": (add, (add, (add, x, y), z), "in")},
        "out",
        ("in",),
    )
    assert set(futures_of(sg)) == {x, y, z}


def test_futures_of_class():
    da = pytest.importorskip("dask.array")
    assert futures_of([da.Array]) == []


@gen_cluster(client=True)
async def test_futures_of_cancelled_raises(c, s, a, b):
    x = c.submit(inc, 1)
    await c.cancel([x])

    with pytest.raises(CancelledError):
        await x

    with pytest.raises(CancelledError):
        await c.get({"x": (inc, x), "y": (inc, 2)}, ["x", "y"], sync=False)

    with pytest.raises(CancelledError):
        c.submit(inc, x)

    with pytest.raises(CancelledError):
        c.submit(add, 1, y=x)

    with pytest.raises(CancelledError):
        c.map(add, [1], y=x)

    assert "y" not in s.tasks


@pytest.mark.skip
@gen_cluster(nthreads=[("127.0.0.1", 1)], client=True)
async def test_dont_delete_recomputed_results(c, s, w):
    x = c.submit(inc, 1)  # compute first time
    await wait([x])
    x.__del__()  # trigger garbage collection
    await asyncio.sleep(0)
    xx = c.submit(inc, 1)  # compute second time

    start = time()
    while xx.key not in w.data:  # data shows up
        await asyncio.sleep(0.01)
        assert time() < start + 1

    while time() < start + (s.delete_interval + 100) / 1000:  # and stays
        assert xx.key in w.data
        await asyncio.sleep(0.01)


@gen_cluster(nthreads=[], client=True)
async def test_fatally_serialized_input(c, s):
    o = FatallySerializedObject()

    future = c.submit(inc, o)

    while not s.tasks:
        await asyncio.sleep(0.01)


@pytest.mark.skip(reason="Use fast random selection now")
@gen_cluster(client=True)
async def test_balance_tasks_by_stacks(c, s, a, b):
    x = c.submit(inc, 1)
    await wait(x)

    y = c.submit(inc, 2)
    await wait(y)

    assert len(a.data) == len(b.data) == 1


@gen_cluster(client=True)
async def test_run(c, s, a, b):
    results = await c.run(inc, 1)
    assert results == {a.address: 2, b.address: 2}

    results = await c.run(inc, 1, workers=[a.address])
    assert results == {a.address: 2}

    results = await c.run(inc, 1, workers=[])
    assert results == {}


@gen_cluster(client=True)
async def test_run_handles_picklable_data(c, s, a, b):
    futures = c.map(inc, range(10))
    await wait(futures)

    def func():
        return {}, set(), [], (), 1, "hello", b"100"

    results = await c.run_on_scheduler(func)
    assert results == func()

    results = await c.run(func)
    assert results == {w.address: func() for w in [a, b]}


def test_run_sync(c, s, a, b):
    def func(x, y=10):
        return x + y

    result = c.run(func, 1, y=2)
    assert result == {a["address"]: 3, b["address"]: 3}

    result = c.run(func, 1, y=2, workers=[a["address"]])
    assert result == {a["address"]: 3}


@gen_cluster(client=True)
async def test_run_coroutine(c, s, a, b):
    results = await c.run(geninc, 1, delay=0.05)
    assert results == {a.address: 2, b.address: 2}

    results = await c.run(geninc, 1, delay=0.05, workers=[a.address])
    assert results == {a.address: 2}

    results = await c.run(geninc, 1, workers=[])
    assert results == {}

    with pytest.raises(RuntimeError, match="hello"):
        await c.run(throws, 1)

    results = await c.run(asyncinc, 2, delay=0.01)
    assert results == {a.address: 3, b.address: 3}


def test_run_coroutine_sync(c, s, a, b):
    result = c.run(geninc, 2, delay=0.01)
    assert result == {a["address"]: 3, b["address"]: 3}

    result = c.run(geninc, 2, workers=[a["address"]])
    assert result == {a["address"]: 3}

    t1 = time()
    result = c.run(geninc, 2, delay=10, wait=False)
    t2 = time()
    assert result is None
    assert t2 - t1 <= 1.0


@gen_cluster(client=True)
async def test_run_exception(c, s, a, b):
    class MyError(Exception):
        pass

    def raise_exception(dask_worker, addr):
        if addr == dask_worker.address:
            raise MyError("informative message")
        return 123

    with pytest.raises(MyError, match="informative message"):
        await c.run(raise_exception, addr=a.address)
    with pytest.raises(MyError, match="informative message"):
        await c.run(raise_exception, addr=a.address, on_error="raise")
    with pytest.raises(ValueError, match="on_error must be"):
        await c.run(raise_exception, addr=a.address, on_error="invalid")

    out = await c.run(raise_exception, addr=a.address, on_error="return")
    assert isinstance(out[a.address], MyError)
    assert out[b.address] == 123

    out = await c.run(raise_exception, addr=a.address, on_error="ignore")
    assert out == {b.address: 123}


@gen_cluster(client=True, config={"distributed.comm.timeouts.connect": "200ms"})
async def test_run_rpc_error(c, s, a, b):
    a.stop()
    with pytest.raises(OSError, match="Timed out trying to connect"):
        await c.run(inc, 1)
    with pytest.raises(OSError, match="Timed out trying to connect"):
        await c.run(inc, 1, on_error="raise")

    out = await c.run(inc, 1, on_error="return")
    assert isinstance(out[a.address], OSError)
    assert out[b.address] == 2

    out = await c.run(inc, 1, on_error="ignore")
    assert out == {b.address: 2}


def test_diagnostic_ui(loop):
    with cluster() as (s, [a, b]):
        a_addr = a["address"]
        b_addr = b["address"]
        with Client(s["address"], loop=loop) as c:
            d = c.nthreads()
            assert d == {a_addr: 1, b_addr: 1}

            d = c.nthreads([a_addr])
            assert d == {a_addr: 1}
            d = c.nthreads(a_addr)
            assert d == {a_addr: 1}
            d = c.nthreads(a["address"])
            assert d == {a_addr: 1}

            x = c.submit(inc, 1)
            y = c.submit(inc, 2)
            z = c.submit(inc, 3)
            wait([x, y, z])
            d = c.who_has()
            assert set(d) == {x.key, y.key, z.key}
            assert all(w in [a_addr, b_addr] for v in d.values() for w in v)
            assert all(d.values())

            d = c.who_has([x, y])
            assert set(d) == {x.key, y.key}

            d = c.who_has(x)
            assert set(d) == {x.key}

            d = c.has_what()
            assert set(d) == {a_addr, b_addr}
            assert all(k in [x.key, y.key, z.key] for v in d.values() for k in v)

            d = c.has_what([a_addr])
            assert set(d) == {a_addr}

            d = c.has_what(a_addr)
            assert set(d) == {a_addr}


def test_diagnostic_nbytes_sync(c):
    incs = c.map(inc, [1, 2, 3])
    doubles = c.map(double, [1, 2, 3])
    wait(incs + doubles)

    assert c.nbytes(summary=False) == {k.key: sizeof(1) for k in incs + doubles}
    assert c.nbytes(summary=True) == {"inc": sizeof(1) * 3, "double": sizeof(1) * 3}


@gen_cluster(client=True)
async def test_diagnostic_nbytes(c, s, a, b):
    incs = c.map(inc, [1, 2, 3])
    doubles = c.map(double, [1, 2, 3])
    await wait(incs + doubles)

    assert s.get_nbytes(summary=False) == {k.key: sizeof(1) for k in incs + doubles}
    assert s.get_nbytes(summary=True) == {"inc": sizeof(1) * 3, "double": sizeof(1) * 3}


@gen_cluster(client=True, nthreads=[])
async def test_worker_aliases(c, s):
    a = Worker(s.address, name="alice")
    b = Worker(s.address, name="bob")
    w = Worker(s.address, name=3)
    await asyncio.gather(a, b, w)

    L = c.map(inc, range(10), workers="alice")
    future = await c.scatter(123, workers=3)
    await wait(L)
    assert len(a.data) == 10
    assert len(b.data) == 0
    assert dict(w.data) == {future.key: 123}

    for i, alias in enumerate([3, [3], "alice"]):
        result = await c.submit(lambda x: x + 1, i, workers=alias)
        assert result == i + 1

    await asyncio.gather(a.close(), b.close(), w.close())


def test_persist_get_sync(c):
    x, y = delayed(1), delayed(2)
    xx = delayed(add)(x, x)
    yy = delayed(add)(y, y)
    xxyy = delayed(add)(xx, yy)

    xxyy2 = c.persist(xxyy)
    xxyy3 = delayed(add)(xxyy2, 10)

    assert xxyy3.compute() == ((1 + 1) + (2 + 2)) + 10


@gen_cluster(client=True)
async def test_persist_get(c, s, a, b):
    x, y = delayed(1), delayed(2)
    xx = delayed(add)(x, x)
    yy = delayed(add)(y, y)
    xxyy = delayed(add)(xx, yy)

    xxyy2 = c.persist(xxyy)
    xxyy3 = delayed(add)(xxyy2, 10)

    await asyncio.sleep(0.5)
    result = await c.gather(c.get(xxyy3.dask, xxyy3.__dask_keys__(), sync=False))
    assert result[0] == ((1 + 1) + (2 + 2)) + 10

    result = await c.compute(xxyy3)
    assert result == ((1 + 1) + (2 + 2)) + 10

    result = await c.compute(xxyy3)
    assert result == ((1 + 1) + (2 + 2)) + 10

    result = await c.compute(xxyy3)
    assert result == ((1 + 1) + (2 + 2)) + 10


@pytest.mark.skipif(WINDOWS, reason="num_fds not supported on windows")
def test_client_num_fds(loop):
    with cluster() as (s, [a, b]):
        proc = psutil.Process()
        with Client(s["address"], loop=loop) as c:  # first client to start loop
            before = proc.num_fds()  # measure
            for i in range(4):
                with Client(s["address"], loop=loop):  # start more clients
                    pass
            start = time()
            while proc.num_fds() > before:
                sleep(0.01)
                assert time() < start + 10, (before, proc.num_fds())


@gen_cluster()
async def test_startup_close_startup(s, a, b):
    c = await Client(s.address, asynchronous=True)
    await c.close()

    c = await Client(s.address, asynchronous=True)
    await c.close()


def test_startup_close_startup_sync(loop):
    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop) as c:
            sleep(0.1)
        with Client(s["address"]) as c:
            pass
        with Client(s["address"]) as c:
            pass
        sleep(0.1)
        with Client(s["address"]) as c:
            pass


@gen_cluster(client=True)
async def test_badly_serialized_exceptions(c, s, a, b):
    def f():
        class BadlySerializedException(Exception):
            def __reduce__(self):
                raise TypeError()

        raise BadlySerializedException("hello world")

    x = c.submit(f)
    with pytest.raises(Exception, match="hello world"):
        await x


# Set rebalance() to work predictably on small amounts of managed memory. By default, it
# uses optimistic memory, which would only be possible to test by allocating very large
# amounts of managed memory, so that they would hide variations in unmanaged memory.
REBALANCE_MANAGED_CONFIG = {
    "distributed.worker.memory.rebalance.measure": "managed",
    "distributed.worker.memory.rebalance.sender-min": 0,
    "distributed.worker.memory.rebalance.sender-recipient-gap": 0,
}


@gen_cluster(client=True, config=REBALANCE_MANAGED_CONFIG)
async def test_rebalance(c, s, a, b):
    """Test Client.rebalance(). These are just to test the Client wrapper around
    Scheduler.rebalance(); for more thorough tests on the latter see test_scheduler.py.
    """
    futures = await c.scatter(range(100), workers=[a.address])
    assert len(a.data) == 100
    assert len(b.data) == 0
    await c.rebalance()
    assert len(a.data) == 50
    assert len(b.data) == 50


@gen_cluster(nthreads=[("", 1)] * 3, client=True, config=REBALANCE_MANAGED_CONFIG)
async def test_rebalance_workers_and_keys(client, s, a, b, c):
    """Test Client.rebalance(). These are just to test the Client wrapper around
    Scheduler.rebalance(); for more thorough tests on the latter see test_scheduler.py.
    """
    futures = await client.scatter(range(100), workers=[a.address])
    assert (len(a.data), len(b.data), len(c.data)) == (100, 0, 0)

    # Passing empty iterables is not the same as omitting the arguments
    await client.rebalance([])
    await client.rebalance(workers=[])
    assert (len(a.data), len(b.data), len(c.data)) == (100, 0, 0)

    # Limit rebalancing to two arbitrary keys and two arbitrary workers.
    await client.rebalance([futures[3], futures[7]], [a.address, b.address])
    assert (len(a.data), len(b.data), len(c.data)) == (98, 2, 0)

    with pytest.raises(KeyError):
        await client.rebalance(workers=["notexist"])


def test_rebalance_sync():
    with dask.config.set(REBALANCE_MANAGED_CONFIG):
        with Client(n_workers=2, processes=False, dashboard_address=":0") as c:
            s = c.cluster.scheduler
            a = c.cluster.workers[0]
            b = c.cluster.workers[1]
            futures = c.scatter(range(100), workers=[a.address])

            assert len(a.data) == 100
            assert len(b.data) == 0
            c.rebalance()
            assert len(a.data) == 50
            assert len(b.data) == 50


@gen_cluster(client=True)
async def test_rebalance_unprepared(c, s, a, b):
    """Client.rebalance() internally waits for unfinished futures"""
    futures = c.map(slowinc, range(10), delay=0.05, workers=a.address)
    # Let the futures reach the scheduler
    await asyncio.sleep(0.1)
    # We didn't wait enough for futures to complete. However, Client.rebalance() will
    # block until all futures are completed before invoking Scheduler.rebalance().
    await c.rebalance(futures)
    s.validate_state()


@gen_cluster(client=True)
async def test_rebalance_raises_on_explicit_missing_data(c, s, a, b):
    """rebalance() raises KeyError if explicitly listed futures disappear"""
    f = Future("x", client=c, state="memory")
    with pytest.raises(KeyError, match="Could not rebalance keys:"):
        await c.rebalance(futures=[f])


@gen_cluster(client=True)
async def test_receive_lost_key(c, s, a, b):
    x = c.submit(inc, 1, workers=[a.address])
    await x
    await a.close()

    while x.status == "finished":
        await asyncio.sleep(0.01)


@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
@gen_cluster([("127.0.0.1", 1), ("127.0.0.2", 2)], client=True)
async def test_unrunnable_task_runs(c, s, a, b):
    x = c.submit(inc, 1, workers=[a.ip])
    await x

    await a.close()
    while x.status == "finished":
        await asyncio.sleep(0.01)

    assert s.tasks[x.key] in s.unrunnable
    assert s.get_task_status(keys=[x.key]) == {x.key: "no-worker"}

    w = await Worker(s.address)

    while x.status != "finished":
        await asyncio.sleep(0.01)

    assert s.tasks[x.key] not in s.unrunnable
    result = await x
    assert result == 2
    await w.close()


@gen_cluster(client=True, nthreads=[])
async def test_add_worker_after_tasks(c, s):
    futures = c.map(inc, range(10))
    async with Nanny(s.address, nthreads=2):
        await c.gather(futures)


@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
@gen_cluster([("127.0.0.1", 1), ("127.0.0.2", 2)], client=True)
async def test_workers_register_indirect_data(c, s, a, b):
    [x] = await c.scatter([1], workers=a.address)
    y = c.submit(inc, x, workers=b.ip)
    await y
    assert b.data[x.key] == 1
    assert s.tasks[x.key].who_has == {s.workers[a.address], s.workers[b.address]}
    assert s.workers[b.address].has_what == {s.tasks[x.key], s.tasks[y.key]}
    s.validate_state()


@gen_cluster(client=True)
async def test_submit_on_cancelled_future(c, s, a, b):
    x = c.submit(inc, 1)
    await x

    await c.cancel(x)

    with pytest.raises(CancelledError):
        c.submit(inc, x)


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 10)
async def test_replicate(c, s, *workers):
    [a, b] = await c.scatter([1, 2])
    await s.replicate(keys=[a.key, b.key], n=5)
    s.validate_state()

    assert len(s.tasks[a.key].who_has) == 5
    assert len(s.tasks[b.key].who_has) == 5

    assert sum(a.key in w.data for w in workers) == 5
    assert sum(b.key in w.data for w in workers) == 5


@gen_cluster(client=True)
async def test_replicate_tuple_keys(c, s, a, b):
    x = delayed(inc)(1, dask_key_name=("x", 1))
    f = c.persist(x)
    await c.replicate(f, n=5)
    s.validate_state()
    assert a.data and b.data

    await c.rebalance(f)
    s.validate_state()


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 10)
async def test_replicate_workers(c, s, *workers):

    [a, b] = await c.scatter([1, 2], workers=[workers[0].address])
    await s.replicate(
        keys=[a.key, b.key], n=5, workers=[w.address for w in workers[:5]]
    )

    assert len(s.tasks[a.key].who_has) == 5
    assert len(s.tasks[b.key].who_has) == 5

    assert sum(a.key in w.data for w in workers[:5]) == 5
    assert sum(b.key in w.data for w in workers[:5]) == 5
    assert sum(a.key in w.data for w in workers[5:]) == 0
    assert sum(b.key in w.data for w in workers[5:]) == 0

    await s.replicate(keys=[a.key, b.key], n=1)

    assert len(s.tasks[a.key].who_has) == 1
    assert len(s.tasks[b.key].who_has) == 1
    assert sum(a.key in w.data for w in workers) == 1
    assert sum(b.key in w.data for w in workers) == 1

    s.validate_state()

    await s.replicate(keys=[a.key, b.key], n=None)  # all
    assert len(s.tasks[a.key].who_has) == 10
    assert len(s.tasks[b.key].who_has) == 10
    s.validate_state()

    await s.replicate(
        keys=[a.key, b.key], n=1, workers=[w.address for w in workers[:5]]
    )
    assert sum(a.key in w.data for w in workers[:5]) == 1
    assert sum(b.key in w.data for w in workers[:5]) == 1
    assert sum(a.key in w.data for w in workers[5:]) == 5
    assert sum(b.key in w.data for w in workers[5:]) == 5
    s.validate_state()


class CountSerialization:
    def __init__(self):
        self.n = 0

    def __setstate__(self, n):
        self.n = n + 1

    def __getstate__(self):
        return self.n


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 10)
async def test_replicate_tree_branching(c, s, *workers):
    obj = CountSerialization()
    [future] = await c.scatter([obj])
    await s.replicate(keys=[future.key], n=10)

    max_count = max(w.data[future.key].n for w in workers)
    assert max_count > 1


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 10)
async def test_client_replicate(c, s, *workers):
    x = c.submit(inc, 1)
    y = c.submit(inc, 2)
    await c.replicate([x, y], n=5)

    assert len(s.tasks[x.key].who_has) == 5
    assert len(s.tasks[y.key].who_has) == 5

    await c.replicate([x, y], n=3)

    assert len(s.tasks[x.key].who_has) == 3
    assert len(s.tasks[y.key].who_has) == 3

    await c.replicate([x, y])
    s.validate_state()

    assert len(s.tasks[x.key].who_has) == 10
    assert len(s.tasks[y.key].who_has) == 10


@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 1), ("127.0.0.2", 1), ("127.0.0.2", 1)],
)
async def test_client_replicate_host(client, s, a, b, c):
    aws = s.workers[a.address]
    bws = s.workers[b.address]
    cws = s.workers[c.address]

    x = client.submit(inc, 1, workers="127.0.0.2")
    await wait([x])
    assert s.tasks[x.key].who_has == {bws} or s.tasks[x.key].who_has == {cws}

    await client.replicate([x], workers=["127.0.0.2"])
    assert s.tasks[x.key].who_has == {bws, cws}

    await client.replicate([x], workers=["127.0.0.1"])
    assert s.tasks[x.key].who_has == {aws, bws, cws}


def test_client_replicate_sync(c):
    x = c.submit(inc, 1)
    y = c.submit(inc, 2)
    c.replicate([x, y], n=2)

    who_has = c.who_has()
    assert len(who_has[x.key]) == len(who_has[y.key]) == 2

    with pytest.raises(ValueError):
        c.replicate([x], n=0)

    assert y.result() == 3


@pytest.mark.skipif(WINDOWS, reason="Windows timer too coarse-grained")
@gen_cluster(client=True, nthreads=[("127.0.0.1", 4)] * 1)
async def test_task_load_adapts_quickly(c, s, a):
    future = c.submit(slowinc, 1, delay=0.2)  # slow
    await wait(future)
    assert 0.15 < s.task_prefixes["slowinc"].duration_average < 0.4

    futures = c.map(slowinc, range(10), delay=0)  # very fast
    await wait(futures)

    assert 0 < s.task_prefixes["slowinc"].duration_average < 0.1


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 2)
async def test_even_load_after_fast_functions(c, s, a, b):
    x = c.submit(inc, 1, workers=a.address)  # very fast
    y = c.submit(inc, 2, workers=b.address)  # very fast
    await wait([x, y])

    futures = c.map(inc, range(2, 11))
    await wait(futures)
    assert any(f.key in a.data for f in futures)
    assert any(f.key in b.data for f in futures)

    # assert abs(len(a.data) - len(b.data)) <= 3


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 2)
async def test_even_load_on_startup(c, s, a, b):
    x, y = c.map(inc, [1, 2])
    await wait([x, y])
    assert len(a.data) == len(b.data) == 1


@pytest.mark.skip
@gen_cluster(client=True, nthreads=[("127.0.0.1", 2)] * 2)
async def test_contiguous_load(c, s, a, b):
    w, x, y, z = c.map(inc, [1, 2, 3, 4])
    await wait([w, x, y, z])

    groups = [set(a.data), set(b.data)]
    assert {w.key, x.key} in groups
    assert {y.key, z.key} in groups


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 4)
async def test_balanced_with_submit(c, s, *workers):
    L = [c.submit(slowinc, i) for i in range(4)]
    await wait(L)
    for w in workers:
        assert len(w.data) == 1


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 4)
async def test_balanced_with_submit_and_resident_data(c, s, *workers):
    [x] = await c.scatter([10], broadcast=True)
    L = [c.submit(slowinc, x, pure=False) for i in range(4)]
    await wait(L)
    for w in workers:
        assert len(w.data) == 2


@gen_cluster(client=True, nthreads=[("127.0.0.1", 20)] * 2)
async def test_scheduler_saturates_cores(c, s, a, b):
    for delay in [0, 0.01, 0.1]:
        futures = c.map(slowinc, range(100), delay=delay)
        futures = c.map(slowinc, futures, delay=delay / 10)
        while not s.tasks:
            if s.tasks:
                assert all(
                    len(p) >= 20
                    for w in s.workers.values()
                    for p in w.processing.values()
                )
            await asyncio.sleep(0.01)


@gen_cluster(client=True, nthreads=[("127.0.0.1", 20)] * 2)
async def test_scheduler_saturates_cores_random(c, s, a, b):
    for delay in [0, 0.01, 0.1]:
        futures = c.map(randominc, range(100), scale=0.1)
        while not s.tasks:
            if s.tasks:
                assert all(
                    len(p) >= 20
                    for w in s.workers.values()
                    for p in w.processing.values()
                )
            await asyncio.sleep(0.01)


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 4)
async def test_cancel_clears_processing(c, s, *workers):
    da = pytest.importorskip("dask.array")
    x = c.submit(slowinc, 1, delay=0.2)
    while not s.tasks:
        await asyncio.sleep(0.01)

    await c.cancel(x)

    while any(v for w in s.workers.values() for v in w.processing):
        await asyncio.sleep(0.01)
    s.validate_state()


def test_default_get():
    with cluster() as (s, [a, b]):
        pre_get = dask.base.get_scheduler()
        pytest.raises(KeyError, dask.config.get, "shuffle")
        with Client(s["address"], set_as_default=True) as c:
            assert dask.base.get_scheduler() == c.get
            assert dask.config.get("shuffle") == "tasks"

        assert dask.base.get_scheduler() == pre_get
        pytest.raises(KeyError, dask.config.get, "shuffle")

        c = Client(s["address"], set_as_default=False)
        assert dask.base.get_scheduler() == pre_get
        pytest.raises(KeyError, dask.config.get, "shuffle")
        c.close()

        c = Client(s["address"], set_as_default=True)
        assert dask.config.get("shuffle") == "tasks"
        assert dask.base.get_scheduler() == c.get
        c.close()
        assert dask.base.get_scheduler() == pre_get
        pytest.raises(KeyError, dask.config.get, "shuffle")

        with Client(s["address"]) as c:
            assert dask.base.get_scheduler() == c.get

        with Client(s["address"], set_as_default=False) as c:
            assert dask.base.get_scheduler() != c.get
        assert dask.base.get_scheduler() != c.get

        with Client(s["address"], set_as_default=True) as c1:
            assert dask.base.get_scheduler() == c1.get
            with Client(s["address"], set_as_default=True) as c2:
                assert dask.base.get_scheduler() == c2.get
            assert dask.base.get_scheduler() == c1.get
        assert dask.base.get_scheduler() == pre_get


@gen_cluster(client=True)
async def test_ensure_default_client(c, s, a, b):
    assert c is default_client()

    async with Client(s.address, set_as_default=False, asynchronous=True) as c2:
        assert c is default_client()
        assert c2 is not default_client()
        ensure_default_client(c2)
        assert c is not default_client()
        assert c2 is default_client()


@gen_cluster()
async def test_set_as_default(s, a, b):
    with pytest.raises(ValueError):
        default_client()

    async with Client(s.address, set_as_default=False, asynchronous=True) as c1:
        with pytest.raises(ValueError):
            default_client()
        async with Client(s.address, set_as_default=True, asynchronous=True) as c2:
            assert default_client() is c2
            async with Client(s.address, set_as_default=True, asynchronous=True) as c3:
                assert default_client() is c3
                async with Client(
                    s.address, set_as_default=False, asynchronous=True
                ) as c4:
                    assert default_client() is c3

                    await c4.scheduler_comm.close()
                    while c4.status != "running":
                        await asyncio.sleep(0.01)
                    assert default_client() is c3

    with pytest.raises(ValueError):
        default_client()


@gen_cluster(client=True)
async def test_get_foo(c, s, a, b):
    futures = c.map(inc, range(10))
    await wait(futures)

    x = await c.scheduler.nbytes(summary=False)
    assert x == s.get_nbytes(summary=False)

    x = await c.scheduler.nbytes(keys=[futures[0].key], summary=False)
    assert x == {futures[0].key: s.tasks[futures[0].key].nbytes}

    x = await c.scheduler.who_has()
    assert valmap(set, x) == {
        key: {ws.address for ws in ts.who_has} for key, ts in s.tasks.items()
    }

    x = await c.scheduler.who_has(keys=[futures[0].key])
    assert valmap(set, x) == {
        futures[0].key: {ws.address for ws in s.tasks[futures[0].key].who_has}
    }


def assert_dict_key_equal(expected, actual):
    assert set(expected.keys()) == set(actual.keys())
    for k in actual.keys():
        ev = expected[k]
        av = actual[k]
        assert list(ev) == list(av)


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 3)
async def test_get_foo_lost_keys(c, s, u, v, w):
    x = c.submit(inc, 1, workers=[u.address])
    y = await c.scatter(3, workers=[v.address])
    await wait([x, y])

    ua, va, wa = u.address, v.address, w.address

    d = await c.scheduler.has_what()
    assert_dict_key_equal(d, {ua: [x.key], va: [y.key], wa: []})
    d = await c.scheduler.has_what(workers=[ua, va])
    assert_dict_key_equal(d, {ua: [x.key], va: [y.key]})
    d = await c.scheduler.who_has()
    assert_dict_key_equal(d, {x.key: [ua], y.key: [va]})
    d = await c.scheduler.who_has(keys=[x.key, y.key])
    assert_dict_key_equal(d, {x.key: [ua], y.key: [va]})

    await u.close()
    await v.close()

    d = await c.scheduler.has_what()
    assert_dict_key_equal(d, {wa: []})
    d = await c.scheduler.has_what(workers=[ua, va])
    assert_dict_key_equal(d, {ua: [], va: []})
    # The scattered key cannot be recomputed so it is forgotten
    d = await c.scheduler.who_has()
    assert_dict_key_equal(d, {x.key: []})
    # ... but when passed explicitly, it is included in the result
    d = await c.scheduler.who_has(keys=[x.key, y.key])
    assert_dict_key_equal(d, {x.key: [], y.key: []})


@pytest.mark.slow
@gen_cluster(
    client=True, Worker=Nanny, clean_kwargs={"threads": False, "processes": False}
)
async def test_bad_tasks_fail(c, s, a, b):
    f = c.submit(sys.exit, 0)
    with captured_logger(logging.getLogger("distributed.scheduler")) as logger:
        with pytest.raises(KilledWorker) as info:
            await f

    text = logger.getvalue()
    assert f.key in text

    assert info.value.last_worker.nanny in {a.address, b.address}
    await asyncio.gather(a.close(), b.close())


def test_get_processing_sync(c, s, a, b):
    processing = c.processing()
    assert not any(v for v in processing.values())

    futures = c.map(
        slowinc, range(10), delay=0.1, workers=[a["address"]], allow_other_workers=False
    )

    sleep(0.2)

    aa = a["address"]
    bb = b["address"]
    processing = c.processing()

    assert set(c.processing(aa)) == {aa}
    assert set(c.processing([aa])) == {aa}

    c.cancel(futures)


def test_close_idempotent(c):
    c.close()
    c.close()
    c.close()


def test_get_returns_early(c):
    event = Event()

    def block(ev):
        ev.wait()

    with pytest.raises(RuntimeError):
        result = c.get({"x": (throws, 1), "y": (block, event)}, ["x", "y"])

    # Futures should be released and forgotten
    wait_for(lambda: not c.futures, timeout=1)
    event.set()
    wait_for(lambda: not any(c.processing().values()), timeout=3)

    x = c.submit(inc, 1)
    x.result()

    with pytest.raises(RuntimeError):
        result = c.get({"x": (throws, 1), x.key: (inc, 1)}, ["x", x.key])
    assert x.key in c.futures


@pytest.mark.slow
@gen_cluster(Worker=Nanny, client=True, timeout=60)
async def test_Client_clears_references_after_restart(c, s, a, b):
    x = c.submit(inc, 1)
    assert x.key in c.refcount

    await c.restart()
    assert x.key not in c.refcount

    key = x.key
    del x
    with profile.lock:
        await asyncio.sleep(0)
        assert key not in c.refcount


@gen_cluster(Worker=Nanny, client=True)
async def test_restart_timeout_is_logged(c, s, a, b):
    with captured_logger(logging.getLogger("distributed.client")) as logger:
        await c.restart(timeout="0.5s")
    text = logger.getvalue()
    assert "Restart timed out after 0.50 seconds" in text


def test_get_stops_work_after_error(c):
    with pytest.raises(RuntimeError):
        c.get({"x": (throws, 1), "y": (sleep, 1.5)}, ["x", "y"])

    start = time()
    while any(c.processing().values()):
        sleep(0.01)
        assert time() < start + 0.5


def test_as_completed_list(c):
    seq = c.map(inc, range(5))
    seq2 = list(as_completed(seq))
    assert set(c.gather(seq2)) == {1, 2, 3, 4, 5}


def test_as_completed_results(c):
    seq = c.map(inc, range(5))
    seq2 = list(as_completed(seq, with_results=True))
    assert set(pluck(1, seq2)) == {1, 2, 3, 4, 5}
    assert set(pluck(0, seq2)) == set(seq)


@pytest.mark.parametrize("with_results", [True, False])
def test_as_completed_batches(c, with_results):
    n = 50
    futures = c.map(slowinc, range(n), delay=0.01)
    out = []
    for batch in as_completed(futures, with_results=with_results).batches():
        assert isinstance(batch, (tuple, list))
        sleep(0.05)
        out.extend(batch)

    assert len(out) == n
    if with_results:
        assert set(pluck(1, out)) == set(range(1, n + 1))
    else:
        assert set(out) == set(futures)


def test_as_completed_next_batch(c):
    futures = c.map(slowinc, range(2), delay=0.1)
    ac = as_completed(futures)
    assert not ac.is_empty()
    assert ac.next_batch(block=False) == []
    assert set(ac.next_batch(block=True)).issubset(futures)
    while not ac.is_empty():
        assert set(ac.next_batch(block=True)).issubset(futures)
    assert ac.is_empty()
    assert not ac.has_ready()


@gen_cluster(nthreads=[])
async def test_status(s):
    c = await Client(s.address, asynchronous=True)
    assert c.status == "running"
    x = c.submit(inc, 1)

    await c.close()
    assert c.status == "closed"


@gen_cluster(client=True)
async def test_persist_optimize_graph(c, s, a, b):
    i = 10
    for method in [c.persist, c.compute]:
        b = db.range(i, npartitions=2)
        i += 1
        b2 = b.map(inc)
        b3 = b2.map(inc)

        b4 = method(b3, optimize_graph=False)
        await wait(b4)

        assert set(map(stringify, b3.__dask_keys__())).issubset(s.tasks)

        b = db.range(i, npartitions=2)
        i += 1
        b2 = b.map(inc)
        b3 = b2.map(inc)

        b4 = method(b3, optimize_graph=True)
        await wait(b4)

        assert not any(stringify(k) in s.tasks for k in b2.__dask_keys__())


@gen_cluster(client=True, nthreads=[])
async def test_scatter_raises_if_no_workers(c, s):
    with pytest.raises(TimeoutError):
        await c.scatter(1, timeout=0.5)


@pytest.mark.flaky(reruns=2)  # due to random port
@gen_test()
async def test_reconnect():
    port = random.randint(10000, 50000)

    async def hard_stop(s):
        for pc in s.periodic_callbacks.values():
            pc.stop()

        s.stop_services()
        for comm in list(s.stream_comms.values()):
            comm.abort()
        for comm in list(s.client_comms.values()):
            comm.abort()

        await s.rpc.close()
        s.stop()
        await Server.close(s)

    s = await Scheduler(port=port)
    w = await Worker(f"127.0.0.1:{port}")
    c = await Client(f"127.0.0.1:{port}", asynchronous=True)
    await c.wait_for_workers(1, timeout=10)
    x = c.submit(inc, 1)
    assert (await x) == 2
    await hard_stop(s)

    start = time()
    while c.status != "connecting":
        assert time() < start + 10
        await asyncio.sleep(0.01)

    assert x.status == "cancelled"
    with pytest.raises(CancelledError):
        await x

    s = await Scheduler(port=port)
    start = time()
    while c.status != "running":
        await asyncio.sleep(0.1)
        assert time() < start + 10

    await w.finished()
    w = await Worker(f"127.0.0.1:{port}")

    start = time()
    while len(await c.nthreads()) != 1:
        await asyncio.sleep(0.05)
        assert time() < start + 10

    x = c.submit(inc, 1)
    assert (await x) == 2
    await hard_stop(s)

    start = time()
    while True:
        assert time() < start + 10
        try:
            await x
            assert False
        except CommClosedError:
            continue
        except CancelledError:
            break

    await w.close()
    await c._close(fast=True)


class UnhandledExceptions(Exception):
    pass


@contextmanager
def catch_unhandled_exceptions() -> Generator[None, None, None]:
    loop = asyncio.get_running_loop()
    ctxs: list[dict[str, Any]] = []

    old_handler = loop.get_exception_handler()

    @loop.set_exception_handler
    def _(loop: object, context: dict[str, Any]) -> None:
        ctxs.append(context)

    try:
        yield
    finally:
        loop.set_exception_handler(old_handler)
    if ctxs:
        msgs = []
        for i, ctx in enumerate(ctxs, 1):
            msgs.append(ctx["message"])
            print(
                f"------ Unhandled exception {i}/{len(ctxs)}: {ctx['message']!r} ------"
            )
            print(ctx)
            if exc := ctx.get("exception"):
                traceback.print_exception(type(exc), exc, exc.__traceback__)

        raise UnhandledExceptions(", ".join(msgs))


@gen_cluster(client=True, nthreads=[], client_kwargs={"timeout": 0.5})
async def test_reconnect_timeout(c, s):
    with catch_unhandled_exceptions(), captured_logger(
        logging.getLogger("distributed.client")
    ) as logger:
        await s.close()
        while c.status != "closed":
            await asyncio.sleep(0.05)
    text = logger.getvalue()
    assert "Failed to reconnect" in text


@pytest.mark.avoid_ci(reason="hangs on github actions ubuntu-latest CI")
@pytest.mark.slow
@pytest.mark.skipif(WINDOWS, reason="num_fds not supported on windows")
@pytest.mark.parametrize("worker,count,repeat", [(Worker, 100, 5), (Nanny, 10, 20)])
def test_open_close_many_workers(loop, worker, count, repeat):
    proc = psutil.Process()

    with cluster(nworkers=0, active_rpc_timeout=2) as (s, _):
        gc.collect()
        before = proc.num_fds()
        done = Semaphore(0)
        running = weakref.WeakKeyDictionary()
        workers = set()
        status = True

        async def start_worker(sleep, duration, repeat=1):
            for i in range(repeat):
                await asyncio.sleep(sleep)
                if not status:
                    return
                w = worker(s["address"], loop=loop)
                running[w] = None
                await w
                workers.add(w)
                addr = w.worker_address
                running[w] = addr
                await asyncio.sleep(duration)
                await w.close()
                del w
                await asyncio.sleep(0)
            done.release()

        for i in range(count):
            loop.add_callback(
                start_worker, random.random() / 5, random.random() / 5, repeat=repeat
            )

        with Client(s["address"], loop=loop) as c:
            sleep(1)

            for i in range(count):
                done.acquire(timeout=5)
                gc.collect()
                if not running:
                    break

            start = time()
            while c.nthreads():
                sleep(0.2)
                assert time() < start + 10

            while len(workers) < count * repeat:
                sleep(0.2)

            status = False

            [c.sync(w.close) for w in list(workers)]
            for w in workers:
                assert w.status == Status.closed

    start = time()
    while proc.num_fds() > before:
        print("fds:", before, proc.num_fds())
        sleep(0.1)
        if time() > start + 10:
            if worker == Worker:  # this is an esoteric case
                print("File descriptors did not clean up")
                break
            else:
                raise ValueError("File descriptors did not clean up")


@gen_cluster()
async def test_idempotence(s, a, b):
    c = await Client(s.address, asynchronous=True)
    f = await Client(s.address, asynchronous=True)

    # Submit
    x = c.submit(inc, 1)
    await x
    log = list(s.transition_log)

    len_single_submit = len(log)  # see last assert

    y = f.submit(inc, 1)
    assert x.key == y.key
    await y
    await asyncio.sleep(0.1)
    log2 = list(s.transition_log)
    assert log == log2

    # Error
    a = c.submit(div, 1, 0)
    await wait(a)
    assert a.status == "error"
    log = list(s.transition_log)

    b = f.submit(div, 1, 0)
    assert a.key == b.key
    await wait(b)
    await asyncio.sleep(0.1)
    log2 = list(s.transition_log)
    assert log == log2

    s.transition_log.clear()
    # Simultaneous Submit
    d = c.submit(inc, 2)
    e = c.submit(inc, 2)
    await wait([d, e])

    assert len(s.transition_log) == len_single_submit

    await c.close()
    await f.close()


def test_scheduler_info(c):
    info = c.scheduler_info()
    assert isinstance(info, dict)
    assert len(info["workers"]) == 2
    assert isinstance(info["started"], float)


def test_write_scheduler_file(c):
    info = c.scheduler_info()
    with tmpfile("json") as scheduler_file:
        c.write_scheduler_file(scheduler_file)
        with Client(scheduler_file=scheduler_file) as c2:
            info2 = c2.scheduler_info()
            assert c.scheduler.address == c2.scheduler.address

        # test that a ValueError is raised if the scheduler_file
        # attribute is already set
        with pytest.raises(ValueError):
            c.write_scheduler_file(scheduler_file)


def test_get_versions_sync(c):
    requests = pytest.importorskip("requests")

    v = c.get_versions()
    assert v["scheduler"] is not None
    assert v["client"] is not None
    assert len(v["workers"]) == 2
    for k, v in v["workers"].items():
        assert v is not None

    c.get_versions(check=True)
    # smoke test for versions
    # that this does not raise

    v = c.get_versions(packages=["requests"])
    assert v["client"]["packages"]["requests"] == requests.__version__


@gen_cluster(client=True)
async def test_get_versions_async(c, s, a, b):
    v = await c.get_versions(check=True)
    assert v.keys() == {"scheduler", "client", "workers"}


@gen_cluster(client=True, config={"distributed.comm.timeouts.connect": "200ms"})
async def test_get_versions_rpc_error(c, s, a, b):
    a.stop()
    v = await c.get_versions()
    assert v.keys() == {"scheduler", "client", "workers"}
    assert v["workers"].keys() == {b.address}


def test_threaded_get_within_distributed(c):
    import dask.multiprocessing

    for get in [dask.local.get_sync, dask.multiprocessing.get, dask.threaded.get]:

        def f():
            return get({"x": (lambda: 1,)}, "x")

        future = c.submit(f)
        assert future.result() == 1


@gen_cluster(client=True)
async def test_lose_scattered_data(c, s, a, b):
    [x] = await c.scatter([1], workers=a.address)

    await a.close()
    await asyncio.sleep(0.1)

    assert x.status == "cancelled"
    assert x.key not in s.tasks


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 3)
async def test_partially_lose_scattered_data(e, s, a, b, c):
    x = await e.scatter(1, workers=a.address)
    await e.replicate(x, n=2)

    await a.close()
    await asyncio.sleep(0.1)

    assert x.status == "finished"
    assert s.get_task_status(keys=[x.key]) == {x.key: "memory"}


@gen_cluster(client=True)
async def test_scatter_compute_lose(c, s, a, b):
    [x] = await c.scatter([[1, 2, 3, 4]], workers=a.address)
    y = c.submit(inc, 1, workers=b.address)

    z = c.submit(slowadd, x, y, delay=0.2)
    await asyncio.sleep(0.1)

    await a.close()

    with pytest.raises(CancelledError):
        await wait(z)

    assert x.status == "cancelled"
    assert y.status == "finished"
    assert z.status == "cancelled"


@gen_cluster(client=True)
async def test_scatter_compute_store_lose(c, s, a, b):
    """
    Create irreplaceable data on one machine,
    cause a dependent computation to occur on another and complete

    Kill the machine with the irreplaceable data.  What happens to the complete
    result?  How about after it GCs and tries to come back?
    """
    x = await c.scatter(1, workers=a.address)
    xx = c.submit(inc, x, workers=a.address)
    y = c.submit(inc, 1)

    z = c.submit(slowadd, xx, y, delay=0.2, workers=b.address)
    await wait(z)

    await a.close()

    while x.status == "finished":
        await asyncio.sleep(0.01)

    # assert xx.status == 'finished'
    assert y.status == "finished"
    assert z.status == "finished"

    zz = c.submit(inc, z)
    await wait(zz)

    zkey = z.key
    del z

    while s.get_task_status(keys=[zkey]) != {zkey: "released"}:
        await asyncio.sleep(0.01)

    xxkey = xx.key
    del xx

    while x.key in s.tasks and zkey not in s.tasks and xxkey not in s.tasks:
        await asyncio.sleep(0.01)


@gen_cluster(client=True)
async def test_scatter_compute_store_lose_processing(c, s, a, b):
    """
    Create irreplaceable data on one machine,
    cause a dependent computation to occur on another and complete

    Kill the machine with the irreplaceable data.  What happens to the complete
    result?  How about after it GCs and tries to come back?
    """
    [x] = await c.scatter([1], workers=a.address)

    y = c.submit(slowinc, x, delay=0.2)
    z = c.submit(inc, y)
    await asyncio.sleep(0.1)
    await a.close()

    while x.status == "finished":
        await asyncio.sleep(0.01)

    assert y.status == "cancelled"
    assert z.status == "cancelled"


@gen_cluster()
async def test_serialize_future(s, a, b):
    c1 = await Client(s.address, asynchronous=True)
    c2 = await Client(s.address, asynchronous=True)

    future = c1.submit(lambda: 1)
    result = await future

    for ci in (c1, c2):
        for ctxman in ci.as_current, lambda: temp_default_client(ci):
            with ctxman():
                future2 = pickle.loads(pickle.dumps(future))
                assert future2.client is ci
                assert stringify(future2.key) in ci.futures
                result2 = await future2
                assert result == result2

    await c1.close()
    await c2.close()


@gen_cluster()
async def test_temp_default_client(s, a, b):
    c1 = await Client(s.address, asynchronous=True)
    c2 = await Client(s.address, asynchronous=True)

    with temp_default_client(c1):
        assert default_client() is c1
        assert default_client(c2) is c2

    with temp_default_client(c2):
        assert default_client() is c2
        assert default_client(c1) is c1

    await c1.close()
    await c2.close()


@gen_cluster(client=True)
async def test_as_current(c, s, a, b):
    c1 = await Client(s.address, asynchronous=True)
    c2 = await Client(s.address, asynchronous=True)

    with temp_default_client(c):
        assert Client.current() is c
        with pytest.raises(ValueError):
            Client.current(allow_global=False)
        with c1.as_current():
            assert Client.current() is c1
            assert Client.current(allow_global=True) is c1
        with c2.as_current():
            assert Client.current() is c2
            assert Client.current(allow_global=True) is c2

    await c1.close()
    await c2.close()


def test_as_current_is_thread_local(s, loop):
    parties = 2
    cm_after_enter = threading.Barrier(parties=parties, timeout=5)
    cm_before_exit = threading.Barrier(parties=parties, timeout=5)

    def run():
        with Client(s["address"], loop=loop) as c:
            with c.as_current():
                cm_after_enter.wait()
                try:
                    # This line runs only when all parties are inside the
                    # context manager
                    assert Client.current(allow_global=False) is c
                finally:
                    cm_before_exit.wait()

    with concurrent.futures.ThreadPoolExecutor(max_workers=parties) as tpe:
        for fut in concurrent.futures.as_completed(
            [tpe.submit(run) for _ in range(parties)]
        ):
            fut.result()


@gen_cluster()
async def test_as_current_is_task_local(s, a, b):
    l1 = asyncio.Lock()
    l2 = asyncio.Lock()
    l3 = asyncio.Lock()
    l4 = asyncio.Lock()
    await l1.acquire()
    await l2.acquire()
    await l3.acquire()
    await l4.acquire()

    async def run1():
        async with Client(s.address, asynchronous=True) as c:
            with c.as_current():
                await l1.acquire()
                l2.release()
                try:
                    # This line runs only when both run1 and run2 are inside the
                    # context manager
                    assert Client.current(allow_global=False) is c
                finally:
                    await l3.acquire()
                    l4.release()

    async def run2():
        async with Client(s.address, asynchronous=True) as c:
            with c.as_current():
                l1.release()
                await l2.acquire()
                try:
                    # This line runs only when both run1 and run2 are inside the
                    # context manager
                    assert Client.current(allow_global=False) is c
                finally:
                    l3.release()
                    await l4.acquire()

    await asyncio.gather(run1(), run2())


@nodebug  # test timing is fragile
@gen_cluster(nthreads=[("127.0.0.1", 1)] * 3, client=True)
async def test_persist_workers_annotate(e, s, a, b, c):
    with dask.annotate(workers=a.address, allow_other_workers=False):
        L1 = [delayed(inc)(i) for i in range(4)]
    with dask.annotate(workers=b.address, allow_other_workers=False):
        total = delayed(sum)(L1)
    with dask.annotate(workers=c.address, allow_other_workers=True):
        L2 = [delayed(add)(i, total) for i in L1]
    with dask.annotate(workers=b.address, allow_other_workers=True):
        total2 = delayed(sum)(L2)

    # TODO: once annotations are faithfully forwarded upon graph optimization,
    # we shouldn't need to disable that here.
    out = e.persist(L1 + L2 + [total, total2], optimize_graph=False)

    await wait(out)
    assert all(v.key in a.data for v in L1)
    assert total.key in b.data

    assert s.tasks[total2.key].loose_restrictions
    for v in L2:
        assert s.tasks[v.key].loose_restrictions


@gen_cluster(nthreads=[("127.0.0.1", 1)] * 3, client=True)
async def test_persist_workers_annotate2(e, s, a, b, c):
    def key_to_worker(key):
        return a.address

    L1 = [delayed(inc)(i) for i in range(4)]
    for x in L1:
        assert all(layer.annotations is None for layer in x.dask.layers.values())

    with dask.annotate(workers=key_to_worker):
        out = e.persist(L1, optimize_graph=False)
        await wait(out)

    for x in L1:
        assert all(layer.annotations is None for layer in x.dask.layers.values())

    for v in L1:
        assert s.tasks[v.key].worker_restrictions == {a.address}


@nodebug  # test timing is fragile
@gen_cluster(nthreads=[("127.0.0.1", 1)] * 3, client=True)
async def test_persist_workers(e, s, a, b, c):
    L1 = [delayed(inc)(i) for i in range(4)]
    total = delayed(sum)(L1)
    L2 = [delayed(add)(i, total) for i in L1]
    total2 = delayed(sum)(L2)

    out = e.persist(
        L1 + L2 + [total, total2],
        workers=[a.address, b.address],
        allow_other_workers=True,
    )

    await wait(out)

    for v in L1 + L2 + [total, total2]:
        assert s.tasks[v.key].worker_restrictions == {a.address, b.address}
    assert not any(c.address in ts.worker_restrictions for ts in s.tasks.values())

    assert s.tasks[total.key].loose_restrictions
    for v in L1 + L2:
        assert s.tasks[v.key].loose_restrictions


@gen_cluster(nthreads=[("127.0.0.1", 1)] * 3, client=True)
async def test_compute_workers_annotate(e, s, a, b, c):
    with dask.annotate(workers=a.address, allow_other_workers=True):
        L1 = [delayed(inc)(i) for i in range(4)]
    with dask.annotate(workers=b.address, allow_other_workers=True):
        total = delayed(sum)(L1)
    with dask.annotate(workers=[c.address]):
        L2 = [delayed(add)(i, total) for i in L1]

    # TODO: once annotations are faithfully forwarded upon graph optimization,
    # we shouldn't need to disable that here.
    out = e.compute(L1 + L2 + [total], optimize_graph=False)

    await wait(out)
    for v in L1:
        assert s.tasks[v.key].worker_restrictions == {a.address}
    for v in L2:
        assert s.tasks[v.key].worker_restrictions == {c.address}
    assert s.tasks[total.key].worker_restrictions == {b.address}

    assert s.tasks[total.key].loose_restrictions
    for v in L1:
        assert s.tasks[v.key].loose_restrictions


@gen_cluster(nthreads=[("127.0.0.1", 1)] * 3, client=True)
async def test_compute_workers(e, s, a, b, c):
    L1 = [delayed(inc)(i) for i in range(4)]
    total = delayed(sum)(L1)
    L2 = [delayed(add)(i, total) for i in L1]

    out = e.compute(
        L1 + L2 + [total],
        workers=[a.address, b.address],
        allow_other_workers=True,
    )

    await wait(out)
    for v in L1 + L2 + [total]:
        assert s.tasks[v.key].worker_restrictions == {a.address, b.address}
    assert not any(c.address in ts.worker_restrictions for ts in s.tasks.values())

    assert s.tasks[total.key].loose_restrictions
    for v in L1 + L2:
        assert s.tasks[v.key].loose_restrictions


@gen_cluster(client=True)
async def test_compute_nested_containers(c, s, a, b):
    da = pytest.importorskip("dask.array")
    np = pytest.importorskip("numpy")
    x = da.ones(10, chunks=(5,)) + 1

    future = c.compute({"x": [x], "y": 123})
    result = await future

    assert isinstance(result, dict)
    assert (result["x"][0] == np.ones(10) + 1).all()
    assert result["y"] == 123


@gen_cluster(client=True)
async def test_scatter_type(c, s, a, b):
    [future] = await c.scatter([1])
    assert future.type == int

    d = await c.scatter({"x": 1.0})
    assert d["x"].type == float


@gen_cluster(client=True)
async def test_retire_workers_2(c, s, a, b):
    [x] = await c.scatter([1], workers=a.address)

    await s.retire_workers(workers=[a.address])
    assert b.data == {x.key: 1}

    assert {ws.address for ws in s.tasks[x.key].who_has} == {b.address}
    assert {ts.key for ts in s.workers[b.address].has_what} == {x.key}

    assert a.address not in s.workers


@gen_cluster(client=True, nthreads=[("", 1)] * 10)
async def test_retire_many_workers(c, s, *workers):
    futures = await c.scatter(list(range(100)))

    await s.retire_workers(workers=[w.address for w in workers[:7]])

    results = await c.gather(futures)
    assert results == list(range(100))

    while len(s.workers) != 3:
        await asyncio.sleep(0.01)

    assert len(s.workers) == 3

    assert all(future.done() for future in futures)
    assert all(s.tasks[future.key].state == "memory" for future in futures)
    assert await c.gather(futures) == list(range(100))

    # Don't count how many task landed on each worker.
    # Normally, tasks would be distributed evenly over the surviving workers. However,
    # here all workers share the same process memory, so you'll get an unintuitive
    # distribution of tasks if for any reason one transfer take longer than 2 seconds
    # and as a consequence the Active Memory Manager ends up running for two iterations.
    # This is something that will happen more frequently on low-powered CI machines.
    # See test_active_memory_manager.py for tests that robustly verify the statistical
    # distribution of tasks after worker retirement.


@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 3)] * 2,
    config={"distributed.scheduler.default-task-durations": {"f": "10ms"}},
)
async def test_weight_occupancy_against_data_movement(c, s, a, b):
    await s.extensions["stealing"].stop()

    def f(x, y=0, z=0):
        sleep(0.01)
        return x

    y = await c.scatter([[1, 2, 3, 4]], workers=[a.address])
    z = await c.scatter([1], workers=[b.address])

    futures = c.map(f, [1, 2, 3, 4], y=y, z=z)

    await wait(futures)

    assert sum(f.key in a.data for f in futures) >= 2
    assert sum(f.key in b.data for f in futures) >= 1


@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 1), ("127.0.0.1", 10)],
    config={"distributed.scheduler.default-task-durations": {"f": "10ms"}},
)
async def test_distribute_tasks_by_nthreads(c, s, a, b):
    await s.extensions["stealing"].stop()

    def f(x, y=0):
        sleep(0.01)
        return x

    y = await c.scatter([1], broadcast=True)

    futures = c.map(f, range(20), y=y)

    await wait(futures)

    assert len(b.data) > 2 * len(a.data)


@gen_cluster(client=True, clean_kwargs={"threads": False})
async def test_add_done_callback(c, s, a, b):
    S = set()

    def f(future):
        future.add_done_callback(g)

    def g(future):
        S.add((future.key, future.status))

    u = c.submit(inc, 1, key="u")
    v = c.submit(throws, "hello", key="v")
    w = c.submit(slowinc, 2, delay=0.3, key="w")
    x = c.submit(inc, 3, key="x")
    u.add_done_callback(f)
    v.add_done_callback(f)
    w.add_done_callback(f)

    await wait((u, v, w, x))

    x.add_done_callback(f)

    while len(S) < 4:
        await asyncio.sleep(0.01)

    assert S == {(f.key, f.status) for f in (u, v, w, x)}


@gen_cluster(client=True)
async def test_normalize_collection(c, s, a, b):
    x = delayed(inc)(1)
    y = delayed(inc)(x)
    z = delayed(inc)(y)

    yy = c.persist(y)

    zz = c.normalize_collection(z)
    assert len(z.dask) == len(y.dask) + 1

    assert isinstance(zz.dask[y.key], Future)
    assert len(zz.dask) < len(z.dask)


@gen_cluster(client=True)
async def test_normalize_collection_dask_array(c, s, a, b):
    da = pytest.importorskip("dask.array")

    x = da.ones(10, chunks=(5,))
    y = x + 1
    yy = c.persist(y)

    z = y.sum()
    zdsk = dict(z.dask)
    zz = c.normalize_collection(z)
    assert z.dask == zdsk  # do not mutate input

    assert len(z.dask) > len(zz.dask)
    assert any(isinstance(v, Future) for v in zz.dask.values())

    for k, v in yy.dask.items():
        assert zz.dask[k].key == v.key

    result1 = await c.compute(z)
    result2 = await c.compute(zz)
    assert result1 == result2


@pytest.mark.slow
def test_normalize_collection_with_released_futures(c):
    da = pytest.importorskip("dask.array")

    x = da.arange(2**20, chunks=2**10)
    y = x.persist()
    wait(y)
    sol = y.sum().compute()
    # Start releasing futures
    del y
    # Try to reuse futures. Previously this was a race condition,
    # and the call to `.compute()` would error out due to missing
    # futures on the scheduler at compute time.
    normalized = c.normalize_collection(x)
    res = normalized.sum().compute()
    assert res == sol


@pytest.mark.xfail(reason="https://github.com/dask/distributed/issues/4404")
@gen_cluster(client=True)
async def test_auto_normalize_collection(c, s, a, b):
    da = pytest.importorskip("dask.array")

    x = da.ones(10, chunks=5)
    assert len(x.dask) == 2

    with dask.config.set(optimizations=[c._optimize_insert_futures]):
        y = x.map_blocks(slowinc, delay=1, dtype=x.dtype)
        yy = c.persist(y)

        await wait(yy)

        start = time()
        future = c.compute(y.sum())
        await future
        end = time()
        assert end - start < 1

        start = time()
        z = c.persist(y + 1)
        await wait(z)
        end = time()
        assert end - start < 1


@pytest.mark.xfail(reason="https://github.com/dask/distributed/issues/4404")
def test_auto_normalize_collection_sync(c):
    da = pytest.importorskip("dask.array")
    x = da.ones(10, chunks=5)

    y = x.map_blocks(slowinc, delay=1, dtype=x.dtype)
    yy = c.persist(y)

    wait(yy)

    with dask.config.set(optimizations=[c._optimize_insert_futures]):
        start = time()
        y.sum().compute()
        end = time()
        assert end - start < 1


def assert_no_data_loss(scheduler):
    for key, start, finish, recommendations, _, _ in scheduler.transition_log:
        if start == "memory" and finish == "released":
            for k, v in recommendations.items():
                assert not (k == key and v == "waiting")


@gen_cluster(client=True)
async def test_interleave_computations(c, s, a, b):
    import distributed

    distributed.g = s
    xs = [delayed(slowinc)(i, delay=0.02) for i in range(30)]
    ys = [delayed(slowdec)(x, delay=0.02) for x in xs]
    zs = [delayed(slowadd)(x, y, delay=0.02) for x, y in zip(xs, ys)]

    total = delayed(sum)(zs)

    future = c.compute(total)

    done = ("memory", "released")

    await asyncio.sleep(0.1)

    x_keys = [x.key for x in xs]
    y_keys = [y.key for y in ys]
    z_keys = [z.key for z in zs]

    while not s.tasks or any(w.processing for w in s.workers.values()):
        await asyncio.sleep(0.05)
        x_done = sum(state in done for state in s.get_task_status(keys=x_keys).values())
        y_done = sum(state in done for state in s.get_task_status(keys=y_keys).values())
        z_done = sum(state in done for state in s.get_task_status(keys=z_keys).values())

        assert x_done >= y_done >= z_done
        assert x_done < y_done + 10
        assert y_done < z_done + 10

    assert_no_data_loss(s)


@pytest.mark.skip(reason="Now prefer first-in-first-out")
@gen_cluster(client=True)
async def test_interleave_computations_map(c, s, a, b):
    xs = c.map(slowinc, range(30), delay=0.02)
    ys = c.map(slowdec, xs, delay=0.02)
    zs = c.map(slowadd, xs, ys, delay=0.02)

    done = ("memory", "released")

    x_keys = [x.key for x in xs]
    y_keys = [y.key for y in ys]
    z_keys = [z.key for z in zs]

    while not s.tasks or any(w.processing for w in s.workers.values()):
        await asyncio.sleep(0.05)
        x_done = sum(state in done for state in s.get_task_status(keys=x_keys).values())
        y_done = sum(state in done for state in s.get_task_status(keys=y_keys).values())
        z_done = sum(state in done for state in s.get_task_status(keys=z_keys).values())

        assert x_done >= y_done >= z_done
        assert x_done < y_done + 10
        assert y_done < z_done + 10


@gen_cluster(client=True)
async def test_scatter_dict_workers(c, s, a, b):
    await c.scatter({"a": 10}, workers=[a.address, b.address])
    assert "a" in a.data or "a" in b.data


@pytest.mark.flaky(reruns=2)
@pytest.mark.slow
@gen_test()
async def test_client_timeout():
    """`await Client(...)` keeps retrying for 10 seconds if it can't find the Scheduler
    straight away
    """
    port = random.randint(10000, 50000)
    with dask.config.set({"distributed.comm.timeouts.connect": "10s"}):
        c = Client(f"127.0.0.1:{port}", asynchronous=True)
        client_start_fut = asyncio.ensure_future(c)
        await asyncio.sleep(2)
        async with Scheduler(port=port, dashboard_address=":0"):
            await client_start_fut
            assert await c.run_on_scheduler(lambda: 123) == 123
            await c.close()


@gen_cluster(client=True)
async def test_submit_list_kwargs(c, s, a, b):
    futures = await c.scatter([1, 2, 3])

    def f(L=None):
        return sum(L)

    future = c.submit(f, L=futures)
    result = await future
    assert result == 1 + 2 + 3


@gen_cluster(client=True)
async def test_map_list_kwargs(c, s, a, b):
    futures = await c.scatter([1, 2, 3])

    def f(i, L=None):
        return i + sum(L)

    futures = c.map(f, range(10), L=futures)
    results = await c.gather(futures)
    assert results == [i + 6 for i in range(10)]


@gen_cluster(client=True)
async def test_recreate_error_delayed(c, s, a, b):
    x0 = delayed(dec)(2)
    y0 = delayed(dec)(1)
    x = delayed(div)(1, x0)
    y = delayed(div)(1, y0)
    tot = delayed(sum)(x, y)

    f = c.compute(tot)

    assert f.status == "pending"

    error_f = await c._get_errored_future(f)
    function, args, kwargs = await c._get_components_from_future(error_f)
    assert f.status == "error"
    assert function.__name__ == "div"
    assert args == (1, 0)
    with pytest.raises(ZeroDivisionError):
        function(*args, **kwargs)


@gen_cluster(client=True)
async def test_recreate_error_futures(c, s, a, b):
    x0 = c.submit(dec, 2)
    y0 = c.submit(dec, 1)
    x = c.submit(div, 1, x0)
    y = c.submit(div, 1, y0)
    tot = c.submit(sum, x, y)
    f = c.compute(tot)

    assert f.status == "pending"

    error_f = await c._get_errored_future(f)
    function, args, kwargs = await c._get_components_from_future(error_f)
    assert f.status == "error"
    assert function.__name__ == "div"
    assert args == (1, 0)
    with pytest.raises(ZeroDivisionError):
        function(*args, **kwargs)


@gen_cluster(client=True)
async def test_recreate_error_collection(c, s, a, b):
    b = db.range(10, npartitions=4)
    b = b.map(lambda x: 1 / x)
    b = b.persist()
    f = c.compute(b)

    error_f = await c._get_errored_future(f)
    function, args, kwargs = await c._get_components_from_future(error_f)
    with pytest.raises(ZeroDivisionError):
        function(*args, **kwargs)

    dd = pytest.importorskip("dask.dataframe")
    import pandas as pd

    df = dd.from_pandas(pd.DataFrame({"a": [0, 1, 2, 3, 4]}), chunksize=2)

    def make_err(x):
        # because pandas would happily work with NaN
        if x == 0:
            raise ValueError
        return x

    df2 = df.a.map(make_err)
    f = c.compute(df2)
    error_f = await c._get_errored_future(f)
    function, args, kwargs = await c._get_components_from_future(error_f)
    with pytest.raises(ValueError):
        function(*args, **kwargs)

    # with persist
    df3 = c.persist(df2)
    error_f = await c._get_errored_future(df3)
    function, args, kwargs = await c._get_components_from_future(error_f)
    with pytest.raises(ValueError):
        function(*args, **kwargs)


@gen_cluster(client=True)
async def test_recreate_error_array(c, s, a, b):
    da = pytest.importorskip("dask.array")
    pytest.importorskip("scipy")
    z = (da.linalg.inv(da.zeros((10, 10), chunks=10)) + 1).sum()
    zz = z.persist()
    error_f = await c._get_errored_future(zz)
    function, args, kwargs = await c._get_components_from_future(error_f)
    assert "0.,0.,0." in str(args).replace(" ", "")  # args contain actual arrays


def test_recreate_error_sync(c):
    x0 = c.submit(dec, 2)
    y0 = c.submit(dec, 1)
    x = c.submit(div, 1, x0)
    y = c.submit(div, 1, y0)
    tot = c.submit(sum, x, y)
    f = c.compute(tot)

    with pytest.raises(ZeroDivisionError):
        c.recreate_error_locally(f)
    assert f.status == "error"


def test_recreate_error_not_error(c):
    f = c.submit(dec, 2)
    with pytest.raises(ValueError, match="No errored futures passed"):
        c.recreate_error_locally(f)


@gen_cluster(client=True)
async def test_recreate_task_delayed(c, s, a, b):
    x0 = delayed(dec)(2)
    y0 = delayed(dec)(2)
    x = delayed(div)(1, x0)
    y = delayed(div)(1, y0)
    tot = delayed(sum)([x, y])

    f = c.compute(tot)

    assert f.status == "pending"

    function, args, kwargs = await c._get_components_from_future(f)
    assert f.status == "finished"
    assert function.__name__ == "sum"
    assert args == ([1, 1],)
    assert function(*args, **kwargs) == 2


@gen_cluster(client=True)
async def test_recreate_task_futures(c, s, a, b):
    x0 = c.submit(dec, 2)
    y0 = c.submit(dec, 2)
    x = c.submit(div, 1, x0)
    y = c.submit(div, 1, y0)
    tot = c.submit(sum, [x, y])
    f = c.compute(tot)

    assert f.status == "pending"

    function, args, kwargs = await c._get_components_from_future(f)
    assert f.status == "finished"
    assert function.__name__ == "sum"
    assert args == ([1, 1],)
    assert function(*args, **kwargs) == 2


@gen_cluster(client=True)
async def test_recreate_task_collection(c, s, a, b):
    b = db.range(10, npartitions=4)
    b = b.map(lambda x: int(3628800 / (x + 1)))
    b = b.persist()
    f = c.compute(b)

    function, args, kwargs = await c._get_components_from_future(f)
    assert function(*args, **kwargs) == [
        3628800,
        1814400,
        1209600,
        907200,
        725760,
        604800,
        518400,
        453600,
        403200,
        362880,
    ]

    dd = pytest.importorskip("dask.dataframe")
    import pandas as pd

    df = dd.from_pandas(pd.DataFrame({"a": [0, 1, 2, 3, 4]}), chunksize=2)

    df2 = df.a.map(lambda x: x + 1)
    f = c.compute(df2)

    function, args, kwargs = await c._get_components_from_future(f)
    expected = pd.DataFrame({"a": [1, 2, 3, 4, 5]})["a"]
    assert function(*args, **kwargs).equals(expected)

    # with persist
    df3 = c.persist(df2)
    # recreate_task_locally only works with futures
    with pytest.raises(AttributeError):
        function, args, kwargs = await c._get_components_from_future(df3)

    f = c.compute(df3)
    function, args, kwargs = await c._get_components_from_future(f)
    assert function(*args, **kwargs).equals(expected)


@gen_cluster(client=True)
async def test_recreate_task_array(c, s, a, b):
    da = pytest.importorskip("dask.array")
    z = (da.zeros((10, 10), chunks=10) + 1).sum()
    f = c.compute(z)
    function, args, kwargs = await c._get_components_from_future(f)
    assert function(*args, **kwargs) == 100


def test_recreate_task_sync(c):
    x0 = c.submit(dec, 2)
    y0 = c.submit(dec, 2)
    x = c.submit(div, 1, x0)
    y = c.submit(div, 1, y0)
    tot = c.submit(sum, [x, y])
    f = c.compute(tot)

    assert c.recreate_task_locally(f) == 2


@gen_cluster(client=True)
async def test_retire_workers(c, s, a, b):
    assert set(s.workers) == {a.address, b.address}
    await c.retire_workers(workers=[a.address], close_workers=True)
    assert set(s.workers) == {b.address}

    while a.status != Status.closed:
        await asyncio.sleep(0.01)


class MyException(Exception):
    pass


@gen_cluster(client=True)
async def test_robust_unserializable(c, s, a, b):
    class Foo:
        def __getstate__(self):
            raise MyException()

    with pytest.raises(MyException):
        future = c.submit(identity, Foo())

    futures = c.map(inc, range(10))
    results = await c.gather(futures)

    assert results == list(map(inc, range(10)))
    assert a.data and b.data


@gen_cluster(client=True)
async def test_robust_undeserializable(c, s, a, b):
    class Foo:
        def __getstate__(self):
            return 1

        def __setstate__(self, state):
            raise MyException("hello")

    future = c.submit(identity, Foo())
    with pytest.raises(MyException):
        await future

    futures = c.map(inc, range(10))
    results = await c.gather(futures)

    assert results == list(map(inc, range(10)))
    assert a.data and b.data


@gen_cluster(client=True)
async def test_robust_undeserializable_function(c, s, a, b):
    class Foo:
        def __getstate__(self):
            return 1

        def __setstate__(self, state):
            raise MyException("hello")

        def __call__(self, *args):
            return 1

    future = c.submit(Foo(), 1)
    with pytest.raises(MyException):
        await future

    futures = c.map(inc, range(10))
    results = await c.gather(futures)

    assert results == list(map(inc, range(10)))
    assert a.data and b.data


@gen_cluster(client=True)
async def test_fire_and_forget(c, s, a, b):
    future = c.submit(slowinc, 1, delay=0.1)
    import distributed

    def f(x):
        distributed.foo = 123

    try:
        fire_and_forget(c.submit(f, future))

        while not hasattr(distributed, "foo"):
            await asyncio.sleep(0.01)
        assert distributed.foo == 123
    finally:
        del distributed.foo

    while len(s.tasks) > 1:
        await asyncio.sleep(0.01)

    assert set(s.tasks) == {future.key}
    assert s.tasks[future.key].who_wants


@gen_cluster(client=True)
async def test_fire_and_forget_err(c, s, a, b):
    fire_and_forget(c.submit(div, 1, 0))
    await asyncio.sleep(0.1)

    # erred task should clear out quickly
    start = time()
    while s.tasks:
        await asyncio.sleep(0.01)
        assert time() < start + 1


def test_quiet_client_close(loop):
    with captured_logger(logging.getLogger("distributed")) as logger:
        with Client(
            loop=loop,
            processes=False,
            dashboard_address=":0",
            threads_per_worker=4,
        ) as c:
            futures = c.map(slowinc, range(1000), delay=0.01)
            sleep(0.200)  # stop part-way
        sleep(0.1)  # let things settle

        out = logger.getvalue()
        lines = out.strip().split("\n")
        assert len(lines) <= 2
        for line in lines:
            assert (
                not line
                or "heartbeat from unregistered worker" in line
                or "unaware of this worker" in line
                or "garbage" in line
                or set(line) == {"-"}
            ), line


@pytest.mark.slow
def test_quiet_client_close_when_cluster_is_closed_before_client(loop):
    with captured_logger(logging.getLogger("tornado.application")) as logger:
        cluster = LocalCluster(loop=loop, n_workers=1, dashboard_address=":0")
        client = Client(cluster, loop=loop)
        cluster.close()
        client.close()

    out = logger.getvalue()
    assert "CancelledError" not in out


@gen_cluster()
async def test_close(s, a, b):
    c = await Client(s.address, asynchronous=True)
    future = c.submit(inc, 1)
    await wait(future)
    assert c.id in s.clients
    await c.close()

    while c.id in s.clients or s.tasks:
        await asyncio.sleep(0.01)


def test_threadsafe(c):
    def f(_):
        d = deque(maxlen=50)
        for i in range(100):
            future = c.submit(inc, random.randint(0, 100))
            d.append(future)
            sleep(0.001)
        c.gather(list(d))
        total = c.submit(sum, list(d))
        return total.result()

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(20) as e:
        results = list(e.map(f, range(20)))
        assert results and all(results)
        del results


@pytest.mark.slow
def test_threadsafe_get(c):
    da = pytest.importorskip("dask.array")
    x = da.arange(100, chunks=(10,))

    def f(_):
        total = 0
        for i in range(20):
            total += (x + random.randint(0, 20)).sum().compute()
            sleep(0.001)
        return total

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(30) as e:
        results = list(e.map(f, range(30)))
        assert results and all(results)


@pytest.mark.slow
def test_threadsafe_compute(c):
    da = pytest.importorskip("dask.array")
    x = da.arange(100, chunks=(10,))

    def f(_):
        total = 0
        for i in range(20):
            future = c.compute((x + random.randint(0, 20)).sum())
            total += future.result()
            sleep(0.001)
        return total

    from concurrent.futures import ThreadPoolExecutor

    e = ThreadPoolExecutor(30)
    results = list(e.map(f, range(30)))
    assert results and all(results)


@gen_cluster(client=True)
async def test_identity(c, s, a, b):
    assert c.id.lower().startswith("client")
    assert a.id.lower().startswith("worker")
    assert b.id.lower().startswith("worker")
    assert s.id.lower().startswith("scheduler")


@gen_cluster(client=True, nthreads=[("127.0.0.1", 4)] * 2)
async def test_get_client(c, s, a, b):
    assert get_client() is c
    assert c.asynchronous

    def f(x):
        import distributed

        client = get_client()
        assert not client.asynchronous
        assert client is distributed.tmp_client

        future = client.submit(inc, x)
        return future.result()

    import distributed

    distributed.tmp_client = c
    try:
        futures = c.map(f, range(5))
        results = await c.gather(futures)
        assert results == list(map(inc, range(5)))
    finally:
        del distributed.tmp_client


def test_get_client_no_cluster():
    # Clean up any global workers added by other tests. This test requires that
    # there are no global workers.
    Worker._instances.clear()

    msg = "No global client found and no address provided"
    with pytest.raises(ValueError, match=rf"^{msg}$"):
        get_client()


@gen_cluster(client=True)
async def test_serialize_collections(c, s, a, b):
    da = pytest.importorskip("dask.array")
    x = da.arange(10, chunks=(5,)).persist()

    def f(x):
        assert isinstance(x, da.Array)
        return x.sum().compute()

    future = c.submit(f, x)
    result = await future
    assert result == sum(range(10))


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)] * 1)
async def test_secede_simple(c, s, a):
    def f():
        client = get_client()
        secede()
        return client.submit(inc, 1).result()

    result = await c.submit(f)
    assert result == 2


@gen_cluster(client=True)
async def test_secede_balances(c, s, a, b):
    """Ensure that tasks scheduled from a seceded thread can be scheduled
    elsewhere"""

    def f(x):
        client = get_client()
        secede()
        futures = client.map(inc, range(10), pure=False)
        total = client.submit(sum, futures).result()
        return total

    futures = c.map(f, range(10), workers=[a.address])

    results = await c.gather(futures)
    # We dispatch 10 tasks and every task generates 11 more tasks
    # 10 * 11 + 10
    assert a.state.executed_count + b.state.executed_count == 120
    assert a.state.executed_count >= 10
    assert b.state.executed_count > 0

    assert results == [sum(map(inc, range(10)))] * 10


@pytest.mark.parametrize("raise_exception", [True, False])
@gen_cluster(client=True, nthreads=[("", 1)])
async def test_long_running_not_in_occupancy(c, s, a, raise_exception):
    # https://github.com/dask/distributed/issues/5332
    # See also test_long_running_removal_clean

    l = Lock()
    entered = Event()
    await l.acquire()

    def long_running(lock, entered):
        entered.set()
        secede()
        lock.acquire()
        if raise_exception:
            raise RuntimeError("Exception in task")

    f = c.submit(long_running, l, entered)
    await entered.wait()
    ts = s.tasks[f.key]
    ws = s.workers[a.address]
    assert ws.occupancy == parse_timedelta(
        dask.config.get("distributed.scheduler.unknown-task-duration")
    )

    while ws.occupancy:
        await asyncio.sleep(0.01)
    await a.heartbeat()

    s._set_duration_estimate(ts, ws)
    assert s.workers[a.address].occupancy == 0
    assert s.total_occupancy == 0
    assert ws.occupancy == 0

    s._ongoing_background_tasks.call_soon(s.reevaluate_occupancy, 0)
    assert s.workers[a.address].occupancy == 0
    await l.release()

    with (
        pytest.raises(RuntimeError, match="Exception in task")
        if raise_exception
        else nullcontext()
    ):
        await f

    assert s.total_occupancy == 0
    assert ws.occupancy == 0
    assert not ws.long_running


@pytest.mark.parametrize("ordinary_task", [True, False])
@gen_cluster(client=True, nthreads=[("", 1)])
async def test_long_running_removal_clean(c, s, a, ordinary_task):
    # https://github.com/dask/distributed/issues/5975 which could reduce
    # occupancy to negative values upon finishing long running tasks

    # See also test_long_running_not_in_occupancy

    l = Lock()
    entered = Event()
    l2 = Lock()
    entered2 = Event()
    await l.acquire()
    await l2.acquire()

    def long_running_secede(lock, entered):
        entered.set()
        secede()
        lock.acquire()

    def long_running(lock, entered):
        entered.set()
        lock.acquire()

    f = c.submit(long_running_secede, l, entered)
    await entered.wait()

    if ordinary_task:
        f2 = c.submit(long_running, l2, entered2)
        await entered2.wait()
    await l.release()
    await f

    ws = s.workers[a.address]

    if ordinary_task:
        # Should be exactly 0.5 but if for whatever reason this test runs slow,
        # some approximation may kick in increasing this number
        assert s.total_occupancy >= 0.5
        assert ws.occupancy >= 0.5
        await l2.release()
        await f2

    # In the end, everything should be reset
    assert s.total_occupancy == 0
    assert ws.occupancy == 0
    assert not ws.long_running


@gen_cluster(client=True)
async def test_sub_submit_priority(c, s, a, b):
    def func():
        client = get_client()
        f = client.submit(slowinc, 1, delay=0.5, key="slowinc")
        client.gather(f)

    future = c.submit(func, key="f")
    while len(s.tasks) != 2:
        await asyncio.sleep(0.001)
    # lower values schedule first
    assert s.tasks["f"].priority > s.tasks["slowinc"].priority, (
        s.tasks["f"].priority,
        s.tasks["slowinc"].priority,
    )


def test_get_client_sync(c, s, a, b):
    results = c.run(lambda: get_worker().scheduler.address)
    assert results == {w["address"]: s["address"] for w in [a, b]}

    results = c.run(lambda: get_client().scheduler.address)
    assert results == {w["address"]: s["address"] for w in [a, b]}


@gen_cluster(client=True)
async def test_serialize_collections_of_futures(c, s, a, b):
    pd = pytest.importorskip("pandas")
    dd = pytest.importorskip("dask.dataframe")
    from dask.dataframe.utils import assert_eq

    df = pd.DataFrame({"x": [1, 2, 3]})
    ddf = dd.from_pandas(df, npartitions=2).persist()
    future = await c.scatter(ddf)

    ddf2 = await future
    df2 = await c.compute(ddf2)

    assert_eq(df, df2)


def test_serialize_collections_of_futures_sync(c):
    pd = pytest.importorskip("pandas")
    dd = pytest.importorskip("dask.dataframe")
    from dask.dataframe.utils import assert_eq

    df = pd.DataFrame({"x": [1, 2, 3]})
    ddf = dd.from_pandas(df, npartitions=2).persist()
    future = c.scatter(ddf)

    result = future.result()
    assert_eq(result.compute(), df)

    assert future.type == dd.DataFrame
    assert c.submit(lambda x, y: assert_eq(x.compute(), y), future, df).result()


def _dynamic_workload(x, delay=0.01):
    if delay == "random":
        sleep(random.random() / 2)
    else:
        sleep(delay)
    if x > 4:
        return 4
    secede()
    client = get_client()
    futures = client.map(
        _dynamic_workload, [x + i + 1 for i in range(2)], pure=False, delay=delay
    )
    total = client.submit(sum, futures)
    return total.result()


def test_dynamic_workloads_sync(c):
    future = c.submit(_dynamic_workload, 0, delay=0.02)
    assert future.result(timeout=20) == 52


@pytest.mark.slow
def test_dynamic_workloads_sync_random(c):
    future = c.submit(_dynamic_workload, 0, delay="random")
    assert future.result(timeout=20) == 52


@gen_cluster(client=True)
async def test_bytes_keys(c, s, a, b):
    key = b"inc-123"
    future = c.submit(inc, 1, key=key)
    result = await future
    assert type(future.key) is bytes
    assert set(s.tasks) == {key}
    assert key in a.data or key in b.data
    assert result == 2


@gen_cluster(client=True)
async def test_unicode_ascii_keys(c, s, a, b):
    uni_type = str
    key = "inc-123"
    future = c.submit(inc, 1, key=key)
    result = await future
    assert type(future.key) is uni_type
    assert set(s.tasks) == {key}
    assert key in a.data or key in b.data
    assert result == 2


@gen_cluster(client=True)
async def test_unicode_keys(c, s, a, b):
    uni_type = str
    key = "inc-123\u03bc"
    future = c.submit(inc, 1, key=key)
    result = await future
    assert type(future.key) is uni_type
    assert set(s.tasks) == {key}
    assert key in a.data or key in b.data
    assert result == 2

    future2 = c.submit(inc, future)
    result2 = await future2
    assert result2 == 3

    future3 = await c.scatter({"data-123": 123})
    result3 = await future3["data-123"]
    assert result3 == 123


def test_use_synchronous_client_in_async_context(loop, c):
    async def f():
        x = await c.scatter(123)
        y = c.submit(inc, x)
        z = await c.gather(y)
        return z

    z = sync(loop, f)
    assert z == 124


def test_quiet_quit_when_cluster_leaves(loop_in_thread):
    loop = loop_in_thread
    with LocalCluster(loop=loop, dashboard_address=":0", silence_logs=False) as cluster:
        with captured_logger("distributed.comm") as sio:
            with Client(cluster, loop=loop) as client:
                futures = client.map(lambda x: x + 1, range(10))
                sleep(0.05)
                cluster.close()
                sleep(0.05)

        text = sio.getvalue()
        assert not text


@gen_cluster([("127.0.0.1", 4)] * 2, client=True)
async def test_call_stack_future(c, s, a, b):
    x = c.submit(slowdec, 1, delay=0.5)
    future = c.submit(slowinc, 1, delay=0.5)
    await asyncio.sleep(0.1)
    results = await asyncio.gather(
        c.call_stack(future), c.call_stack(keys=[future.key])
    )
    assert all(list(first(result.values())) == [future.key] for result in results)
    assert results[0] == results[1]
    result = results[0]
    ts = a.state.tasks.get(future.key)
    if ts is not None and ts.state == "executing":
        w = a
    else:
        w = b

    assert list(result) == [w.address]
    assert list(result[w.address]) == [future.key]
    assert "slowinc" in str(result)
    assert "slowdec" not in str(result)


@gen_cluster([("127.0.0.1", 4)] * 2, client=True)
async def test_call_stack_all(c, s, a, b):
    future = c.submit(slowinc, 1, delay=0.8)
    while not a.state.executing_count and not b.state.executing_count:
        await asyncio.sleep(0.01)
    result = await c.call_stack()
    w = a if a.state.executing_count else b
    assert list(result) == [w.address]
    assert list(result[w.address]) == [future.key]
    assert "slowinc" in str(result)


@gen_cluster([("127.0.0.1", 4)] * 2, client=True)
async def test_call_stack_collections(c, s, a, b):
    da = pytest.importorskip("dask.array")
    x = da.random.random(100, chunks=(10,)).map_blocks(slowinc, delay=0.5).persist()
    while not a.state.executing_count and not b.state.executing_count:
        await asyncio.sleep(0.001)
    result = await c.call_stack(x)
    assert result


@gen_cluster([("127.0.0.1", 4)] * 2, client=True)
async def test_call_stack_collections_all(c, s, a, b):
    da = pytest.importorskip("dask.array")
    x = da.random.random(100, chunks=(10,)).map_blocks(slowinc, delay=0.5).persist()
    while not a.state.executing_count and not b.state.executing_count:
        await asyncio.sleep(0.001)
    result = await c.call_stack()
    assert result


@pytest.mark.flaky(condition=WINDOWS, reruns=10, reruns_delay=5)
@gen_cluster(
    client=True,
    config={
        "distributed.worker.profile.enabled": True,
        "distributed.worker.profile.cycle": "100ms",
    },
)
async def test_profile(c, s, a, b):
    futures = c.map(slowinc, range(10), delay=0.05, workers=a.address)
    await wait(futures)

    x = await c.profile(start=time() + 10, stop=time() + 20)
    assert not x["count"]

    x = await c.profile(start=0, stop=time())
    assert (
        x["count"]
        == sum(p["count"] for _, p in a.profile_history) + a.profile_recent["count"]
    )

    y = await c.profile(start=time() - 0.300, stop=time())
    assert 0 < y["count"] < x["count"]

    assert not any(p["count"] for _, p in b.profile_history)
    result = await c.profile(workers=b.address)
    assert not result["count"]


@gen_cluster(
    client=True,
    config={
        "distributed.worker.profile.enabled": False,
        "distributed.worker.profile.cycle": "100ms",
    },
)
async def test_profile_disabled(c, s, a, b):
    futures = c.map(slowinc, range(10), delay=0.05, workers=a.address)
    await wait(futures)

    x = await c.profile(start=time() + 10, stop=time() + 20)
    assert x["count"] == 0

    x = await c.profile(start=0, stop=time())
    assert x["count"] == 0

    y = await c.profile(start=time() - 0.300, stop=time())
    assert 0 == y["count"] == x["count"]


@gen_cluster(
    client=True,
    config={
        "distributed.worker.profile.cycle": "100ms",
    },
)
async def test_profile_keys(c, s, a, b):
    x = c.map(slowinc, range(10), delay=0.05, workers=a.address)
    y = c.map(slowdec, range(10), delay=0.05, workers=a.address)
    await wait(x + y)

    xp = await c.profile("slowinc")
    yp = await c.profile("slowdec")
    p = await c.profile()

    assert p["count"] == xp["count"] + yp["count"]

    with captured_logger(logging.getLogger("distributed")) as logger:
        prof = await c.profile("does-not-exist")
        assert prof == profile.create()
    out = logger.getvalue()
    assert not out


@gen_cluster()
async def test_client_with_name(s, a, b):
    with captured_logger("distributed.scheduler") as sio:
        client = await Client(s.address, asynchronous=True, name="foo")
        assert "foo" in client.id
        await client.close()

    text = sio.getvalue()
    assert "foo" in text


@gen_cluster(client=True)
async def test_future_defaults_to_default_client(c, s, a, b):
    x = c.submit(inc, 1)
    await wait(x)

    future = Future(x.key)
    assert future.client is c


@gen_cluster(client=True)
async def test_future_auto_inform(c, s, a, b):
    x = c.submit(inc, 1)
    await wait(x)

    client = await Client(s.address, asynchronous=True)
    future = Future(x.key, client)

    while future.status != "finished":
        await asyncio.sleep(0.01)

    await client.close()


def test_client_async_before_loop_starts():
    with pristine_loop() as loop:
        client = Client(asynchronous=True, loop=loop)
        assert client.asynchronous
        client.close()


@pytest.mark.skip(
    "FIXME: Not sure how this related to the changes. Need some debugging"
)
@pytest.mark.slow
@gen_cluster(client=True, Worker=Nanny, timeout=60, nthreads=[("127.0.0.1", 3)] * 2)
async def test_nested_compute(c, s, a, b):
    def fib(x):
        assert get_worker().get_current_task()
        if x < 2:
            return x
        a = delayed(fib)(x - 1)
        b = delayed(fib)(x - 2)
        c = a + b
        return c.compute()

    future = c.submit(fib, 8)
    result = await future
    assert result == 21
    assert len(s.transition_log) > 50


@gen_cluster(client=True)
async def test_task_metadata(c, s, a, b):
    await c.set_metadata("x", 1)
    result = await c.get_metadata("x")
    assert result == 1

    future = c.submit(inc, 1)
    key = future.key
    await wait(future)
    await c.set_metadata(key, 123)
    result = await c.get_metadata(key)
    assert result == 123

    del future

    while key in s.tasks:
        await asyncio.sleep(0.01)

    with pytest.raises(KeyError):
        await c.get_metadata(key)

    result = await c.get_metadata(key, None)
    assert result is None

    await c.set_metadata(["x", "a"], 1)
    result = await c.get_metadata("x")
    assert result == {"a": 1}
    await c.set_metadata(["x", "b"], 2)
    result = await c.get_metadata("x")
    assert result == {"a": 1, "b": 2}
    result = await c.get_metadata(["x", "a"])
    assert result == 1

    await c.set_metadata(["x", "a", "c", "d"], 1)
    result = await c.get_metadata("x")
    assert result == {"a": {"c": {"d": 1}}, "b": 2}


@gen_cluster(client=True, Worker=Nanny)
async def test_logs(c, s, a, b):
    await wait(c.map(inc, range(5)))
    logs = await c.get_scheduler_logs(n=5)
    assert logs

    for _, msg in logs:
        assert "distributed.scheduler" in msg

    w_logs = await c.get_worker_logs(n=5)
    assert set(w_logs.keys()) == {a.worker_address, b.worker_address}
    for log in w_logs.values():
        for _, msg in log:
            assert "distributed.worker" in msg

    n_logs = await c.get_worker_logs(nanny=True)
    assert set(n_logs.keys()) == {a.worker_address, b.worker_address}
    for log in n_logs.values():
        for _, msg in log:
            assert "distributed.nanny" in msg

    n_logs = await c.get_worker_logs(nanny=True, workers=[a.worker_address])
    assert set(n_logs.keys()) == {a.worker_address}
    for log in n_logs.values():
        for _, msg in log:
            assert "distributed.nanny" in msg


@gen_cluster(client=True)
async def test_avoid_delayed_finalize(c, s, a, b):
    x = delayed(inc)(1)
    future = c.compute(x)
    result = await future
    assert result == 2
    assert list(s.tasks) == [future.key] == [x.key]


@gen_cluster()
async def test_config_scheduler_address(s, a, b):
    with dask.config.set({"scheduler-address": s.address}):
        with captured_logger("distributed.client") as sio:
            c = await Client(asynchronous=True)
            assert c.scheduler.address == s.address

        text = sio.getvalue()
        assert s.address in text

        await c.close()


@gen_cluster(client=True)
async def test_warn_when_submitting_large_values(c, s, a, b):
    with pytest.warns(
        UserWarning,
        match=r"Large object of size (2\.00 MB|1.91 MiB) detected in task graph:"
        r" \n  \(b'00000000000000000000000000000000000000000000000 \.\.\. 000000000000',\)"
        r"\nConsider scattering large objects ahead of time.*",
    ):
        future = c.submit(lambda x: x + 1, b"0" * 2000000)

    with warnings.catch_warnings(record=True) as record:
        data = b"0" * 2000000
        for i in range(10):
            future = c.submit(lambda x, y: x, data, i)

    assert not record


@gen_cluster(client=True)
async def test_unhashable_function(c, s, a, b):
    func = _UnhashableCallable()
    result = await c.submit(func, 1)
    assert result == 2


@gen_cluster()
async def test_client_name(s, a, b):
    with dask.config.set({"client-name": "hello-world"}):
        c = await Client(s.address, asynchronous=True)
        assert any("hello-world" in name for name in list(s.clients))

    await c.close()


def test_client_doesnt_close_given_loop(loop_in_thread, s, a, b):
    with Client(s["address"], loop=loop_in_thread) as c:
        assert c.submit(inc, 1).result() == 2
    with Client(s["address"], loop=loop_in_thread) as c:
        assert c.submit(inc, 2).result() == 3


@gen_cluster(client=True, nthreads=[])
async def test_quiet_scheduler_loss(c, s):
    c._periodic_callbacks["scheduler-info"].interval = 10
    with captured_logger(logging.getLogger("distributed.client")) as logger:
        await s.close()
    text = logger.getvalue()
    assert "BrokenPipeError" not in text


def test_dashboard_link(loop, monkeypatch):
    monkeypatch.setenv("USER", "myusername")

    with cluster(scheduler_kwargs={"dashboard_address": ":12355"}) as (s, [a, b]):
        with Client(s["address"], loop=loop) as c:
            with dask.config.set(
                {"distributed.dashboard.link": "{scheme}://foo-{USER}:{port}/status"}
            ):
                link = "http://foo-myusername:12355/status"
                assert link == c.dashboard_link
                text = c._repr_html_()
                assert link in text


@gen_test()
async def test_dashboard_link_inproc():
    async with Client(processes=False, asynchronous=True, dashboard_address=":0") as c:
        with dask.config.set({"distributed.dashboard.link": "{host}"}):
            assert "/" not in c.dashboard_link


@pytest.mark.flaky(reruns=2)
@gen_test()
async def test_client_timeout_2():
    port = random.randint(10000, 50000)
    with dask.config.set({"distributed.comm.timeouts.connect": "10ms"}):
        start = time()
        c = Client(f"127.0.0.1:{port}", asynchronous=True)
        with pytest.raises((TimeoutError, IOError)):
            await c
        stop = time()

        assert c.status == "closed"
        await c.close()

        assert stop - start < 1


@gen_test()
async def test_client_active_bad_port():
    import tornado.httpserver
    import tornado.web

    application = tornado.web.Application([(r"/", tornado.web.RequestHandler)])
    http_server = tornado.httpserver.HTTPServer(application)
    http_server.listen(8080)
    with dask.config.set({"distributed.comm.timeouts.connect": "10ms"}):
        c = Client("127.0.0.1:8080", asynchronous=True)
        with pytest.raises((TimeoutError, IOError)):
            await c
        await c._close(fast=True)
    http_server.stop()


@pytest.mark.parametrize("direct", [True, False])
@gen_cluster(client=True, client_kwargs={"serializers": ["dask", "msgpack"]})
async def test_turn_off_pickle(c, s, a, b, direct):
    np = pytest.importorskip("numpy")

    assert (await c.submit(inc, 1)) == 2
    await c.submit(np.ones, 5)
    await c.scatter(1)

    # Can't send complex data
    with pytest.raises(TypeError):
        await c.scatter(inc)

    # can send complex tasks (this uses pickle regardless)
    future = c.submit(lambda x: x, inc)
    await wait(future)

    # but can't receive complex results
    with pytest.raises(TypeError):
        await c.gather(future, direct=direct)

    # Run works
    result = await c.run(lambda: 1)
    assert list(result.values()) == [1, 1]
    result = await c.run_on_scheduler(lambda: 1)
    assert result == 1

    # But not with complex return values
    with pytest.raises(TypeError):
        await c.run(lambda: inc)
    with pytest.raises(TypeError):
        await c.run_on_scheduler(lambda: inc)


@gen_cluster()
async def test_de_serialization(s, a, b):
    np = pytest.importorskip("numpy")

    c = await Client(
        s.address,
        asynchronous=True,
        serializers=["msgpack", "pickle"],
        deserializers=["msgpack"],
    )
    try:
        # Can send complex data
        future = await c.scatter(np.ones(5))

        # But can not retrieve it
        with pytest.raises(TypeError):
            result = await future
    finally:
        await c.close()


@gen_cluster()
async def test_de_serialization_none(s, a, b):
    np = pytest.importorskip("numpy")

    c = await Client(s.address, asynchronous=True, deserializers=["msgpack"])
    try:
        # Can send complex data
        future = await c.scatter(np.ones(5))

        # But can not retrieve it
        with pytest.raises(TypeError):
            result = await future
    finally:
        await c.close()


@gen_cluster()
async def test_client_repr_closed(s, a, b):
    c = await Client(s.address, asynchronous=True)
    await c.close()
    c._repr_html_()


def test_client_repr_closed_sync(loop):
    with Client(loop=loop, processes=False, dashboard_address=":0") as c:
        pass
    c._repr_html_()


@pytest.mark.xfail(reason="https://github.com/dask/dask/pull/6807")
@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)])
async def test_nested_prioritization(c, s, w):
    x = delayed(inc)(1, dask_key_name=("a", 2))
    y = delayed(inc)(2, dask_key_name=("a", 10))

    o = dask.order.order(merge(x.__dask_graph__(), y.__dask_graph__()))

    fx, fy = c.compute([x, y])

    await wait([fx, fy])

    assert (o[x.key] < o[y.key]) == (
        s.tasks[stringify(fx.key)].priority < s.tasks[stringify(fy.key)].priority
    )


@gen_cluster(client=True)
async def test_scatter_error_cancel(c, s, a, b):
    # https://github.com/dask/distributed/issues/2038
    def bad_fn(x):
        raise Exception("lol")

    x = await c.scatter(1)
    y = c.submit(bad_fn, x)
    del x

    await wait(y)
    assert y.status == "error"
    await asyncio.sleep(0.1)
    assert y.status == "error"  # not cancelled


@pytest.mark.parametrize("workers_arg", [False, True])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("broadcast", [False, True, 10])
@gen_cluster(
    client=True,
    nthreads=[("", 1)] * 10,
    config={"distributed.worker.memory.pause": False},
)
async def test_scatter_and_replicate_avoid_paused_workers(
    c, s, *workers, workers_arg, direct, broadcast
):
    paused_workers = [w for i, w in enumerate(workers) if i not in (3, 7)]
    for w in paused_workers:
        w.status = Status.paused
    while any(s.workers[w.address].status != Status.paused for w in paused_workers):
        await asyncio.sleep(0.01)

    f = await c.scatter(
        {"x": 1},
        workers=[w.address for w in workers[1:-1]] if workers_arg else None,
        broadcast=broadcast,
        direct=direct,
    )
    if not broadcast:
        await c.replicate(f, n=10)

    expect = [i in (3, 7) for i in range(10)]
    actual = [("x" in w.data) for w in workers]
    assert actual == expect


@pytest.mark.xfail(reason="GH#5409 Dask-Default-Threads are frequently detected")
def test_no_threads_lingering():
    if threading.active_count() < 40:
        return
    active = dict(threading._active)
    print(f"==== Found {len(active)} active threads: ====")
    for t in active.values():
        print(t)
    assert False


@gen_cluster()
async def test_direct_async(s, a, b):
    c = await Client(s.address, asynchronous=True, direct_to_workers=True)
    assert c.direct_to_workers
    await c.close()

    c = await Client(s.address, asynchronous=True, direct_to_workers=False)
    assert not c.direct_to_workers
    await c.close()


def test_direct_sync(c):
    assert not c.direct_to_workers

    def f():
        return get_client().direct_to_workers

    assert c.submit(f).result()


@gen_cluster()
async def test_mixing_clients(s, a, b):
    c1 = await Client(s.address, asynchronous=True)
    c2 = await Client(s.address, asynchronous=True)

    future = c1.submit(inc, 1)
    with pytest.raises(ValueError):
        c2.submit(inc, future)

    assert not c2.futures  # Don't create Futures on second Client

    await c1.close()
    await c2.close()


@gen_cluster(client=True)
async def test_tuple_keys(c, s, a, b):
    x = dask.delayed(inc)(1, dask_key_name=("x", 1))
    y = dask.delayed(inc)(x, dask_key_name=("y", 1))
    future = c.compute(y)
    assert (await future) == 3


@gen_cluster(client=True)
async def test_multiple_scatter(c, s, a, b):
    futures = await asyncio.gather(*(c.scatter(1, direct=True) for _ in range(5)))

    x = await futures[0]
    x = await futures[0]


@gen_cluster(client=True)
async def test_map_large_kwargs_in_graph(c, s, a, b):
    np = pytest.importorskip("numpy")
    x = np.random.random(100000)
    futures = c.map(lambda a, b: a + b, range(100), b=x)
    while not s.tasks:
        await asyncio.sleep(0.01)

    assert len(s.tasks) == 101
    assert any(k.startswith("ndarray") for k in s.tasks)


@gen_cluster(client=True)
async def test_retry(c, s, a, b):
    def f():
        assert dask.config.get("foo")

    with dask.config.set(foo=False):
        future = c.submit(f)
        with pytest.raises(AssertionError):
            await future

    with dask.config.set(foo=True):
        await future.retry()
        await future


@gen_cluster(client=True)
async def test_retry_dependencies(c, s, a, b):
    def f():
        return dask.config.get("foo")

    x = c.submit(f)
    y = c.submit(inc, x)

    with pytest.raises(KeyError):
        await y

    with dask.config.set(foo=100):
        await y.retry()
        result = await y
        assert result == 101

        await y.retry()
        await x.retry()
        result = await y
        assert result == 101


@gen_cluster(client=True)
async def test_released_dependencies(c, s, a, b):
    def f(x):
        return dask.config.get("foo") + 1

    x = c.submit(inc, 1, key="x")
    y = c.submit(f, x, key="y")
    del x

    with pytest.raises(KeyError):
        await y

    with dask.config.set(foo=100):
        await y.retry()
        result = await y
        assert result == 101


@gen_cluster(client=True, clean_kwargs={"threads": False})
async def test_profile_bokeh(c, s, a, b):
    pytest.importorskip("bokeh.plotting")
    from bokeh.model import Model

    await c.gather(c.map(slowinc, range(10), delay=0.2))
    state, figure = await c.profile(plot=True)
    assert isinstance(figure, Model)

    with tmpfile("html") as fn:
        try:
            await c.profile(filename=fn)
        except PermissionError:
            if WINDOWS:
                pytest.xfail()
        assert os.path.exists(fn)


@gen_cluster(client=True)
async def test_get_mix_futures_and_SubgraphCallable(c, s, a, b):
    future = c.submit(add, 1, 2)

    subgraph = SubgraphCallable(
        {"_2": (add, "_0", "_1"), "_3": (add, future, "_2")}, "_3", ("_0", "_1")
    )
    dsk = {"a": 1, "b": 2, "c": (subgraph, "a", "b"), "d": (subgraph, "c", "b")}

    future2 = c.get(dsk, "d", sync=False)
    result = await future2
    assert result == 11

    # Nested subgraphs
    subgraph2 = SubgraphCallable(
        {
            "_2": (subgraph, "_0", "_1"),
            "_3": (subgraph, "_2", "_1"),
            "_4": (add, "_3", future2),
        },
        "_4",
        ("_0", "_1"),
    )

    dsk2 = {"e": 1, "f": 2, "g": (subgraph2, "e", "f")}

    result = await c.get(dsk2, "g", sync=False)
    assert result == 22


@gen_cluster(client=True)
async def test_get_mix_futures_and_SubgraphCallable_dask_dataframe(c, s, a, b):
    dd = pytest.importorskip("dask.dataframe")
    import pandas as pd

    df = pd.DataFrame({"x": range(1, 11)})
    ddf = dd.from_pandas(df, npartitions=2).persist()
    ddf = ddf.map_partitions(lambda x: x)
    ddf["x"] = ddf["x"].astype("f8")
    ddf = ddf.map_partitions(lambda x: x)
    ddf["x"] = ddf["x"].astype("f8")
    result = await c.compute(ddf)
    assert result.equals(df.astype("f8"))


def test_direct_to_workers(s, loop):
    with Client(s["address"], loop=loop, direct_to_workers=True) as client:
        future = client.scatter(1)
        future.result()
        resp = client.run_on_scheduler(lambda dask_scheduler: dask_scheduler.events)
        assert "gather" not in str(resp)


@gen_cluster(client=True)
async def test_instances(c, s, a, b):
    assert list(Client._instances) == [c]
    assert list(Scheduler._instances) == [s]
    assert set(Worker._instances) == {a, b}


@gen_cluster(client=True)
async def test_wait_for_workers(c, s, a, b):
    future = asyncio.ensure_future(c.wait_for_workers(n_workers=3))
    await asyncio.sleep(0.22)  # 2 chances
    assert not future.done()

    w = await Worker(s.address)
    start = time()
    await future
    assert time() < start + 1
    await w.close()

    with pytest.raises(TimeoutError) as info:
        await c.wait_for_workers(n_workers=10, timeout="1 ms")

    assert "2/10" in str(info.value).replace(" ", "")
    assert "1 ms" in str(info.value)


@pytest.mark.skipif(WINDOWS, reason="num_fds not supported on windows")
@pytest.mark.parametrize("Worker", [Worker, Nanny])
@gen_test()
async def test_file_descriptors_dont_leak(Worker):
    pytest.importorskip("pandas")
    df = dask.datasets.timeseries(freq="10s", dtypes={"x": int, "y": float})

    proc = psutil.Process()
    before = proc.num_fds()
    async with Scheduler(dashboard_address=":0") as s:
        async with Worker(s.address), Worker(s.address), Client(
            s.address, asynchronous=True
        ):
            assert proc.num_fds() > before
            await df.sum().persist()

    start = time()
    while proc.num_fds() > before:
        await asyncio.sleep(0.01)
        assert time() < start + 10, (before, proc.num_fds())


@gen_test()
async def test_dashboard_link_cluster():
    class MyCluster(LocalCluster):
        @property
        def dashboard_link(self):
            return "http://foo.com"

    async with MyCluster(
        processes=False, asynchronous=True, dashboard_address=":0"
    ) as cluster:
        async with Client(cluster, asynchronous=True) as client:
            assert "http://foo.com" in client._repr_html_()


@gen_test()
async def test_shutdown():
    async with Scheduler(dashboard_address=":0") as s:
        async with Worker(s.address) as w:
            async with Client(s.address, asynchronous=True) as c:
                await c.shutdown()

                assert s.status == Status.closed
                assert w.status in {Status.closed, Status.closing}


@gen_test()
async def test_shutdown_localcluster():
    async with LocalCluster(
        n_workers=1, asynchronous=True, processes=False, dashboard_address=":0"
    ) as lc:
        async with Client(lc, asynchronous=True) as c:
            await c.shutdown()

        assert lc.scheduler.status == Status.closed


@gen_test()
async def test_config_inherited_by_subprocess():
    with dask.config.set(foo=100):
        async with LocalCluster(
            n_workers=1,
            asynchronous=True,
            processes=True,
            dashboard_address=":0",
        ) as lc:
            async with Client(lc, asynchronous=True) as c:
                assert await c.submit(dask.config.get, "foo") == 100


@gen_cluster(client=True)
async def test_futures_of_sorted(c, s, a, b):
    pytest.importorskip("dask.dataframe")
    df = await dask.datasets.timeseries(dtypes={"x": int}).persist()
    futures = futures_of(df)
    for k, f in zip(df.__dask_keys__(), futures):
        assert str(k) in str(f)


@pytest.mark.flaky(reruns=10, reruns_delay=5)
@gen_cluster(
    client=True,
    config={
        "distributed.worker.profile.enabled": True,
        "distributed.worker.profile.cycle": "10ms",
    },
)
async def test_profile_server(c, s, a, b):
    for i in range(5):
        try:
            x = c.map(slowinc, range(10), delay=0.01, workers=a.address, pure=False)
            await wait(x)
            await asyncio.gather(
                c.run(slowinc, 1, delay=0.5), c.run_on_scheduler(slowdec, 1, delay=0.5)
            )

            p = await c.profile(server=True)  # All worker servers
            assert "slowinc" in str(p)

            p = await c.profile(scheduler=True)  # Scheduler
            assert "slowdec" in str(p)
        except AssertionError:
            if i == 4:
                raise
            else:
                pass
        else:
            break


@gen_cluster(
    client=True,
    config={
        "distributed.worker.profile.enabled": False,
        "distributed.worker.profile.cycle": "10ms",
    },
)
async def test_profile_server_disabled(c, s, a, b):
    x = c.map(slowinc, range(10), delay=0.01, workers=a.address, pure=False)
    await wait(x)
    await asyncio.gather(
        c.run(slowinc, 1, delay=0.5), c.run_on_scheduler(slowdec, 1, delay=0.5)
    )

    p = await c.profile(server=True)  # All worker servers
    assert "slowinc" not in str(p)

    p = await c.profile(scheduler=True)  # Scheduler
    assert "slowdec" not in str(p)


@gen_cluster(client=True)
async def test_await_future(c, s, a, b):
    future = c.submit(inc, 1)

    async def f():  # flake8: noqa
        result = await future
        assert result == 2

    await f()

    future = c.submit(div, 1, 0)

    async def f():
        with pytest.raises(ZeroDivisionError):
            await future

    await f()


@gen_cluster(client=True)
async def test_as_completed_async_for(c, s, a, b):
    futures = c.map(inc, range(10))
    ac = as_completed(futures)
    results = []

    async def f():
        async for future in ac:
            result = await future
            results.append(result)

    await f()

    assert set(results) == set(range(1, 11))


@gen_cluster(client=True)
async def test_as_completed_async_for_results(c, s, a, b):
    futures = c.map(inc, range(10))
    ac = as_completed(futures, with_results=True)
    results = []

    async def f():
        async for future, result in ac:
            results.append(result)

    await f()

    assert set(results) == set(range(1, 11))
    assert not s.counters["op"].components[0]["gather"]


@gen_cluster(client=True)
async def test_as_completed_async_for_cancel(c, s, a, b):
    x = c.submit(inc, 1)
    ev = Event()
    y = c.submit(lambda ev: ev.wait(), ev)
    ac = as_completed([x, y])

    await x
    await y.cancel()

    futs = [future async for future in ac]
    assert futs == [x, y]
    await ev.set()  # Allow for clean teardown


@gen_test()
async def test_async_with():
    async with Client(processes=False, dashboard_address=":0", asynchronous=True) as c:
        assert await c.submit(lambda x: x + 1, 10) == 11
    assert c.status == "closed"
    assert c.cluster.status == Status.closed


def test_client_sync_with_async_def(loop):
    async def ff():
        await asyncio.sleep(0.01)
        return 1

    with cluster() as (s, [a, b]):
        with Client(s["address"], loop=loop) as c:
            assert sync(loop, ff) == 1
            assert c.sync(ff) == 1


@pytest.mark.skip(reason="known intermittent failure")
@gen_cluster(client=True)
async def test_dont_hold_on_to_large_messages(c, s, a, b):
    np = pytest.importorskip("numpy")
    da = pytest.importorskip("dask.array")
    x = np.random.random(1000000)
    xr = weakref.ref(x)

    d = da.from_array(x, chunks=(100000,))
    d = d.persist()
    del x

    start = time()
    while xr() is not None:
        if time() > start + 5:
            # Help diagnosing
            from types import FrameType

            x = xr()
            if x is not None:
                del x
                rc = sys.getrefcount(xr())
                refs = gc.get_referrers(xr())
                print("refs to x:", rc, refs, gc.isenabled())
                frames = [r for r in refs if isinstance(r, FrameType)]
                for i, f in enumerate(frames):
                    print(
                        "frames #%d:" % i,
                        f.f_code.co_name,
                        f.f_code.co_filename,
                        sorted(f.f_locals),
                    )
            pytest.fail("array should have been destroyed")

        await asyncio.sleep(0.200)


@gen_cluster(client=True)
async def test_run_on_scheduler_async_def(c, s, a, b):
    async def f(dask_scheduler):
        await asyncio.sleep(0.01)
        dask_scheduler.foo = "bar"

    await c.run_on_scheduler(f)

    assert s.foo == "bar"

    async def f(dask_worker):
        await asyncio.sleep(0.01)
        dask_worker.foo = "bar"

    await c.run(f)
    assert a.foo == "bar"
    assert b.foo == "bar"


@gen_cluster(client=True)
async def test_run_on_scheduler_async_def_wait(c, s, a, b):
    async def f(dask_scheduler):
        await asyncio.sleep(0.01)
        dask_scheduler.foo = "bar"

    await c.run_on_scheduler(f, wait=False)

    while not hasattr(s, "foo"):
        await asyncio.sleep(0.01)
    assert s.foo == "bar"

    async def f(dask_worker):
        await asyncio.sleep(0.01)
        dask_worker.foo = "bar"

    await c.run(f, wait=False)

    while not hasattr(a, "foo") or not hasattr(b, "foo"):
        await asyncio.sleep(0.01)

    assert a.foo == "bar"
    assert b.foo == "bar"


@pytest.mark.skipif(WINDOWS, reason="frequently kills off the whole test suite")
@gen_cluster(client=True, nthreads=[("127.0.0.1", 2)] * 2)
async def test_performance_report(c, s, a, b):
    pytest.importorskip("bokeh")
    da = pytest.importorskip("dask.array")

    async def f(stacklevel, mode=None):
        """
        We wrap this in a function so that the assertions aren't in the
        performanace report itself

        Also, we want this comment to appear
        """
        x = da.random.random((1000, 1000), chunks=(100, 100))
        with tmpfile(extension="html") as fn:
            async with performance_report(
                filename=fn, stacklevel=stacklevel, mode=mode
            ):
                await c.compute((x + x.T).sum())

            with open(fn) as f:
                data = f.read()
        return data

    # Ensure default kwarg maintains backward compatability
    data = await f(stacklevel=1)

    assert "Also, we want this comment to appear" in data
    assert "bokeh" in data
    assert "random" in data
    assert "Dask Performance Report" in data
    assert "x = da.random" in data
    assert "Threads: 4" in data
    assert "No logs to report" in data
    assert dask.__version__ in data

    # stacklevel=2 captures code two frames back -- which in this case
    # is the testing function
    data = await f(stacklevel=2)
    assert "async def test_performance_report(c, s, a, b):" in data
    assert "Dask Performance Report" in data

    # stacklevel=0 or lower is overridden to stacklevel=1 so we don't see
    # distributed internals
    data = await f(stacklevel=0)
    assert "Also, we want this comment to appear" in data
    assert "Dask Performance Report" in data

    data = await f(stacklevel=1, mode="inline")
    assert "cdn.bokeh.org" not in data
    data = await f(stacklevel=1, mode="cdn")
    assert "cdn.bokeh.org" in data


@pytest.mark.skipif(
    sys.version_info >= (3, 10),
    reason="On Py3.10+ semaphore._loop is not bound until .acquire() blocks",
)
@gen_cluster(nthreads=[])
async def test_client_gather_semaphore_loop(s):
    async with Client(s.address, asynchronous=True) as c:
        assert c._gather_semaphore._loop is c.loop.asyncio_loop


@gen_cluster(client=True)
async def test_as_completed_condition_loop(c, s, a, b):
    seq = c.map(inc, range(5))
    ac = as_completed(seq)
    # consume the ac so that the ac.condition is bound to the loop on py3.10+
    async for _ in ac:
        pass
    assert ac.condition._loop == c.loop.asyncio_loop


@pytest.mark.skipif(
    sys.version_info >= (3, 10),
    reason="On Py3.10+ semaphore._loop is not bound until .acquire() blocks",
)
def test_client_connectionpool_semaphore_loop(s, a, b):
    with Client(s["address"]) as c:
        assert c.rpc.semaphore._loop is c.loop.asyncio_loop


@pytest.mark.slow
@gen_cluster(nthreads=[], timeout=60)
async def test_mixed_compression(s):
    pytest.importorskip("lz4")
    da = pytest.importorskip("dask.array")
    async with Nanny(
        s.address, nthreads=1, config={"distributed.comm.compression": None}
    ):
        async with Nanny(
            s.address, nthreads=1, config={"distributed.comm.compression": "lz4"}
        ):
            async with Client(s.address, asynchronous=True) as c:
                await c.get_versions()
                x = da.ones((10000, 10000))
                y = x + x.T
                await c.compute(y.sum())


@gen_cluster(client=True)
async def test_futures_in_subgraphs(c, s, a, b):
    """Regression test of <https://github.com/dask/distributed/issues/4145>"""

    dd = pytest.importorskip("dask.dataframe")
    import pandas as pd

    ddf = dd.from_pandas(
        pd.DataFrame(
            dict(
                uid=range(50),
                enter_time=pd.date_range(
                    start="2020-01-01", end="2020-09-01", periods=50, tz="UTC"
                ),
            )
        ),
        npartitions=5,
    )

    ddf = ddf[ddf.uid.isin(range(29))].persist()
    ddf["local_time"] = ddf.enter_time.dt.tz_convert("US/Central")
    ddf["day"] = ddf.enter_time.dt.day_name()
    ddf = await c.submit(dd.categorical.categorize, ddf, columns=["day"], index=False)


@gen_cluster(client=True)
async def test_get_task_metadata(c, s, a, b):

    # Populate task metadata
    await c.register_worker_plugin(TaskStateMetadataPlugin())

    async with get_task_metadata() as tasks:
        f = c.submit(slowinc, 1)
        await f

    metadata = tasks.metadata
    assert f.key in metadata
    assert metadata[f.key] == s.tasks.get(f.key).metadata

    state = tasks.state
    assert f.key in state
    assert state[f.key] == "memory"

    assert not any(isinstance(p, CollectTaskMetaDataPlugin) for p in s.plugins)


@gen_cluster(client=True)
async def test_get_task_metadata_multiple(c, s, a, b):

    # Populate task metadata
    await c.register_worker_plugin(TaskStateMetadataPlugin())

    # Ensure that get_task_metadata only collects metadata for
    # tasks which are submitted and completed within its context
    async with get_task_metadata() as tasks1:
        f1 = c.submit(slowinc, 1)
        await f1
        async with get_task_metadata() as tasks2:
            f2 = c.submit(slowinc, 2)
            await f2

    metadata1 = tasks1.metadata
    metadata2 = tasks2.metadata

    assert len(metadata1) == 2
    assert sorted(metadata1.keys()) == sorted([f1.key, f2.key])
    assert metadata1[f1.key] == s.tasks.get(f1.key).metadata
    assert metadata1[f2.key] == s.tasks.get(f2.key).metadata

    assert len(metadata2) == 1
    assert list(metadata2.keys()) == [f2.key]
    assert metadata2[f2.key] == s.tasks.get(f2.key).metadata


@gen_cluster(client=True)
async def test_register_worker_plugin_exception(c, s, a, b):
    class MyPlugin:
        def setup(self, worker=None):
            raise ValueError("Setup failed")

    with pytest.raises(ValueError, match="Setup failed"):
        await c.register_worker_plugin(MyPlugin())


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_log_event(c, s, a):

    # Log an event from inside a task
    def foo():
        get_worker().log_event("topic1", {"foo": "bar"})

    assert not await c.get_events("topic1")
    await c.submit(foo)
    events = await c.get_events("topic1")
    assert len(events) == 1
    assert events[0][1] == {"foo": "bar", "worker": a.address}

    # Log an event while on the scheduler
    def log_scheduler(dask_scheduler):
        dask_scheduler.log_event("topic2", {"woo": "hoo"})

    await c.run_on_scheduler(log_scheduler)
    events = await c.get_events("topic2")
    assert len(events) == 1
    assert events[0][1] == {"woo": "hoo"}

    # Log an event from the client process
    await c.log_event("topic2", ("alice", "bob"))
    events = await c.get_events("topic2")
    assert len(events) == 2
    assert events[1][1] == ("alice", "bob")


@gen_cluster(client=True)
async def test_annotations_task_state(c, s, a, b):
    da = pytest.importorskip("dask.array")

    with dask.annotate(qux="bar", priority=100):
        x = da.ones(10, chunks=(5,))

    with dask.config.set(optimization__fuse__active=False):
        x = await x.persist()

    assert all(
        {"qux": "bar", "priority": 100} == ts.annotations for ts in s.tasks.values()
    )


@pytest.mark.parametrize("fn", ["compute", "persist"])
@gen_cluster(client=True)
async def test_annotations_compute_time(c, s, a, b, fn):
    da = pytest.importorskip("dask.array")
    x = da.ones(10, chunks=(5,))

    with dask.annotate(foo="bar"):
        # Turn off optimization to avoid rewriting layers and picking up annotations
        # that way. Instead, we want `compute`/`persist` to be able to pick them up.
        fut = getattr(c, fn)(x, optimize_graph=False)

    await wait(fut)
    assert s.tasks
    assert all(ts.annotations == {"foo": "bar"} for ts in s.tasks.values())


@pytest.mark.xfail(reason="https://github.com/dask/dask/issues/7036")
@gen_cluster(client=True)
async def test_annotations_survive_optimization(c, s, a, b):
    da = pytest.importorskip("dask.array")

    with dask.annotate(foo="bar"):
        x = da.ones(10, chunks=(5,))

    ann = x.__dask_graph__().layers[x.name].annotations
    assert ann is not None
    assert ann.get("foo", None) == "bar"

    (xx,) = dask.optimize(x)

    ann = xx.__dask_graph__().layers[x.name].annotations
    assert ann is not None
    assert ann.get("foo", None) == "bar"


@gen_cluster(client=True)
async def test_annotations_priorities(c, s, a, b):
    da = pytest.importorskip("dask.array")

    with dask.annotate(priority=15):
        x = da.ones(10, chunks=(5,))

    with dask.config.set(optimization__fuse__active=False):
        x = await x.persist()

    assert all("15" in str(ts.priority) for ts in s.tasks.values())
    assert all(ts.priority[0] == -15 for ts in s.tasks.values())
    assert all({"priority": 15} == ts.annotations for ts in s.tasks.values())


@gen_cluster(client=True)
async def test_annotations_workers(c, s, a, b):
    da = pytest.importorskip("dask.array")

    with dask.annotate(workers=[a.address]):
        x = da.ones(10, chunks=(5,))

    with dask.config.set(optimization__fuse__active=False):
        x = await x.persist()

    assert all({"workers": (a.address,)} == ts.annotations for ts in s.tasks.values())
    assert all({a.address} == ts.worker_restrictions for ts in s.tasks.values())
    assert a.data
    assert not b.data


@gen_cluster(client=True)
async def test_annotations_retries(c, s, a, b):
    da = pytest.importorskip("dask.array")

    with dask.annotate(retries=2):
        x = da.ones(10, chunks=(5,))

    with dask.config.set(optimization__fuse__active=False):
        x = await x.persist()

    assert all(ts.retries == 2 for ts in s.tasks.values())
    assert all(ts.annotations == {"retries": 2} for ts in s.tasks.values())


@gen_cluster(client=True)
async def test_annotations_blockwise_unpack(c, s, a, b):
    da = pytest.importorskip("dask.array")
    np = pytest.importorskip("numpy")
    from dask.array.utils import assert_eq

    # A flaky doubling function -- need extra args because it is called before
    # application to establish dtype/meta.
    scale = varying([ZeroDivisionError("one"), ZeroDivisionError("two"), 2, 2])

    def flaky_double(x):
        return scale() * x

    # A reliable double function.
    def reliable_double(x):
        return 2 * x

    x = da.ones(10, chunks=(5,))

    # The later annotations should not override the earlier annotations
    with dask.annotate(retries=2):
        y = x.map_blocks(flaky_double, meta=np.array((), dtype=float))
    with dask.annotate(retries=0):
        z = y.map_blocks(reliable_double, meta=np.array((), dtype=float))

    with dask.config.set(optimization__fuse__active=False):
        z = await c.compute(z)

    assert_eq(z, np.ones(10) * 4.0)


@gen_cluster(
    client=True,
    nthreads=[
        ("127.0.0.1", 1),
        ("127.0.0.1", 1, {"resources": {"GPU": 1}}),
    ],
)
async def test_annotations_resources(c, s, a, b):
    da = pytest.importorskip("dask.array")

    with dask.annotate(resources={"GPU": 1}):
        x = da.ones(10, chunks=(5,))

    with dask.config.set(optimization__fuse__active=False):
        x = await x.persist()

    assert all([{"GPU": 1} == ts.resource_restrictions for ts in s.tasks.values()])
    assert all([{"resources": {"GPU": 1}} == ts.annotations for ts in s.tasks.values()])


@gen_cluster(
    client=True,
    nthreads=[
        ("127.0.0.1", 1),
        ("127.0.0.1", 1, {"resources": {"GPU": 1}}),
    ],
)
async def test_annotations_resources_culled(c, s, a, b):
    da = pytest.importorskip("dask.array")

    x = da.ones((2, 2, 2), chunks=1)
    with dask.annotate(resources={"GPU": 1}):
        y = x.map_blocks(lambda x0: x0, meta=x._meta)

    z = y[0, 0, 0]

    (z,) = c.compute([z], optimize_graph=False)
    await z
    # it worked!


@gen_cluster(client=True)
async def test_annotations_loose_restrictions(c, s, a, b):
    da = pytest.importorskip("dask.array")

    # Eventually fails if allow_other_workers=False
    with dask.annotate(workers=["fake"], allow_other_workers=True):
        x = da.ones(10, chunks=(5,))

    with dask.config.set(optimization__fuse__active=False):
        x = await x.persist()

    assert all(not ts.worker_restrictions for ts in s.tasks.values())
    assert all({"fake"} == ts.host_restrictions for ts in s.tasks.values())
    assert all(
        [
            {"workers": ("fake",), "allow_other_workers": True} == ts.annotations
            for ts in s.tasks.values()
        ]
    )


@gen_cluster(client=True)
async def test_workers_collection_restriction(c, s, a, b):
    da = pytest.importorskip("dask.array")

    future = c.compute(da.arange(10), workers=a.address)
    await future
    assert a.data and not b.data


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)])
async def test_get_client_functions_spawn_clusters(c, s, a):
    # see gh4565

    scheduler_addr = c.scheduler.address

    def f(x):
        with LocalCluster(
            n_workers=1,
            processes=False,
            dashboard_address=":0",
            worker_dashboard_address=":0",
        ) as cluster2:
            with Client(cluster2) as c1:
                c2 = get_client()

                c1_scheduler = c1.scheduler.address
                c2_scheduler = c2.scheduler.address
                assert c1_scheduler != c2_scheduler
                assert c2_scheduler == scheduler_addr

    await c.gather(c.map(f, range(2)))
    await a.close()

    c_default = default_client()
    assert c is c_default


def test_computation_code_walk_frames():

    test_function_code = inspect.getsource(test_computation_code_walk_frames)
    code = Client._get_computation_code()

    assert test_function_code == code

    def nested_call():
        return Client._get_computation_code()

    assert nested_call() == inspect.getsource(nested_call)

    with pytest.raises(TypeError, match="Ignored modules must be a list"):
        with dask.config.set(
            {"distributed.diagnostics.computations.ignore-modules": "test_client"}
        ):
            code = Client._get_computation_code()

    with dask.config.set(
        {"distributed.diagnostics.computations.ignore-modules": ["test_client"]}
    ):
        import sys

        upper_frame_code = inspect.getsource(sys._getframe(1))
        code = Client._get_computation_code()
        assert code == upper_frame_code
        assert nested_call() == upper_frame_code


def test_computation_object_code_dask_compute(client):
    da = pytest.importorskip("dask.array")
    x = da.ones((10, 10), chunks=(3, 3))
    future = x.sum().compute()
    y = future

    test_function_code = inspect.getsource(test_computation_object_code_dask_compute)

    def fetch_comp_code(dask_scheduler):
        computations = list(dask_scheduler.computations)
        assert len(computations) == 1
        comp = computations[0]
        assert len(comp.code) == 1
        return comp.code[0]

    code = client.run_on_scheduler(fetch_comp_code)

    assert code == test_function_code


def test_computation_object_code_not_available(client):
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    dd = pytest.importorskip("dask.dataframe")
    df = pd.DataFrame({"a": range(10)})
    ddf = dd.from_pandas(df, npartitions=3)
    result = np.where(ddf.a > 4)

    def fetch_comp_code(dask_scheduler):
        computations = list(dask_scheduler.computations)
        assert len(computations) == 1
        comp = computations[0]
        assert len(comp.code) == 1
        return comp.code[0]

    code = client.run_on_scheduler(fetch_comp_code)
    assert code == "<Code not available>"


@gen_cluster(client=True)
async def test_computation_object_code_dask_persist(c, s, a, b):
    da = pytest.importorskip("dask.array")
    x = da.ones((10, 10), chunks=(3, 3))
    future = x.sum().persist()
    await future

    test_function_code = inspect.getsource(
        test_computation_object_code_dask_persist.__wrapped__
    )
    computations = list(s.computations)
    assert len(computations) == 1
    comp = computations[0]
    assert len(comp.code) == 1

    assert comp.code[0] == test_function_code


@gen_cluster(client=True)
async def test_computation_object_code_client_submit_simple(c, s, a, b):
    def func(x):
        return x

    fut = c.submit(func, 1)

    await fut

    test_function_code = inspect.getsource(
        test_computation_object_code_client_submit_simple.__wrapped__
    )
    computations = list(s.computations)
    assert len(computations) == 1
    comp = computations[0]

    assert len(comp.code) == 1

    assert comp.code[0] == test_function_code


@gen_cluster(client=True)
async def test_computation_object_code_client_submit_list_comp(c, s, a, b):
    def func(x):
        return x

    futs = [c.submit(func, x) for x in range(10)]

    await c.gather(futs)

    test_function_code = inspect.getsource(
        test_computation_object_code_client_submit_list_comp.__wrapped__
    )
    computations = list(s.computations)
    assert len(computations) == 1
    comp = computations[0]

    # Code is deduplicated
    assert len(comp.code) == 1

    assert comp.code[0] == test_function_code


@gen_cluster(client=True)
async def test_computation_object_code_client_submit_dict_comp(c, s, a, b):
    def func(x):
        return x

    futs = {x: c.submit(func, x) for x in range(10)}

    await c.gather(futs)

    test_function_code = inspect.getsource(
        test_computation_object_code_client_submit_dict_comp.__wrapped__
    )
    computations = list(s.computations)
    assert len(computations) == 1
    comp = computations[0]

    # Code is deduplicated
    assert len(comp.code) == 1

    assert comp.code[0] == test_function_code


@gen_cluster(client=True)
async def test_computation_object_code_client_map(c, s, a, b):
    da = pytest.importorskip("dask.array")
    x = da.ones((10, 10), chunks=(3, 3))
    future = c.compute(x.sum(), retries=2)
    y = await future

    test_function_code = inspect.getsource(
        test_computation_object_code_client_map.__wrapped__
    )
    computations = list(s.computations)
    assert len(computations) == 1
    comp = computations[0]
    assert len(comp.code) == 1

    assert comp.code[0] == test_function_code


@gen_cluster(client=True)
async def test_computation_object_code_client_compute(c, s, a, b):
    da = pytest.importorskip("dask.array")
    x = da.ones((10, 10), chunks=(3, 3))
    future = c.compute(x.sum(), retries=2)
    y = await future

    test_function_code = inspect.getsource(
        test_computation_object_code_client_compute.__wrapped__
    )
    computations = list(s.computations)
    assert len(computations) == 1
    comp = computations[0]
    assert len(comp.code) == 1

    assert comp.code[0] == test_function_code


@gen_cluster(client=True, Worker=Nanny)
async def test_upload_directory(c, s, a, b, tmp_path):
    from dask.distributed import UploadDirectory

    # Be sure to exclude code coverage reports
    files_start = {f for f in os.listdir() if not f.startswith(".coverage")}

    with open(tmp_path / "foo.py", "w") as f:
        f.write("x = 123")
    with open(tmp_path / "bar.py", "w") as f:
        f.write("from foo import x")

    plugin = UploadDirectory(tmp_path, restart=True, update_path=True)
    await c.register_worker_plugin(plugin)

    [name] = a.plugins
    assert os.path.split(tmp_path)[-1] in name

    def f():
        import bar

        return bar.x

    results = await c.run(f)
    assert results[a.worker_address] == 123
    assert results[b.worker_address] == 123

    async with Nanny(s.address, local_directory=tmp_path / "foo", name="foo") as n:
        results = await c.run(f)
        assert results[n.worker_address] == 123

    files_end = {f for f in os.listdir() if not f.startswith(".coverage")}
    assert files_start == files_end  # no change


@gen_cluster(client=True)
async def test_exception_text(c, s, a, b):
    def bad(x):
        raise Exception(x)

    future = c.submit(bad, 123)
    await wait(future)

    ts = s.tasks[future.key]

    assert isinstance(ts.exception_text, str)
    assert "123" in ts.exception_text
    assert "Exception(x)" in ts.traceback_text
    assert "bad" in ts.traceback_text


@gen_cluster(client=True)
async def test_async_task(c, s, a, b):
    async def f(x):
        return x + 1

    future = c.submit(f, 10)
    result = await future
    assert result == 11


@gen_cluster(client=True)
async def test_async_task_with_partial(c, s, a, b):
    async def f(x, y):
        return x + y + 1

    future = c.submit(functools.partial(f, 1), 10)
    result = await future
    assert result == 12


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_events_subscribe_topic(c, s, a):

    log = []

    def user_event_handler(event):
        log.append(event)

    c.subscribe_topic("test-topic", user_event_handler)

    while not s.event_subscriber["test-topic"]:
        await asyncio.sleep(0.01)

    a.log_event("test-topic", {"important": "event"})

    while len(log) != 1:
        await asyncio.sleep(0.01)

    time_, msg = log[0]
    assert isinstance(time_, float)
    assert msg == {"important": "event", "worker": a.address}

    c.unsubscribe_topic("test-topic")

    while s.event_subscriber["test-topic"]:
        await asyncio.sleep(0.01)

    a.log_event("test-topic", {"forget": "me"})

    while len(s.events["test-topic"]) == 1:
        await asyncio.sleep(0.01)

    assert len(log) == 1

    async def async_user_event_handler(event):
        log.append(event)
        await asyncio.sleep(0)

    c.subscribe_topic("test-topic", async_user_event_handler)

    while not s.event_subscriber["test-topic"]:
        await asyncio.sleep(0.01)

    a.log_event("test-topic", {"async": "event"})

    while len(log) == 1:
        await asyncio.sleep(0.01)

    assert len(log) == 2
    time_, msg = log[1]
    assert isinstance(time_, float)
    assert msg == {"async": "event", "worker": a.address}

    # Even though the middle event was not subscribed to, the scheduler still
    # knows about all and we can retrieve them
    all_events = await c.get_events(topic="test-topic")
    assert len(all_events) == 3


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_events_subscribe_topic_cancelled(c, s, a):
    event_handler_started = asyncio.Event()
    exc_info = None

    async def user_event_handler(event):
        nonlocal exc_info
        c.unsubscribe_topic("test-topic")
        event_handler_started.set()
        with pytest.raises(asyncio.CancelledError) as exc_info:
            await asyncio.sleep(0.5)

    c.subscribe_topic("test-topic", user_event_handler)
    while not s.event_subscriber["test-topic"]:
        await asyncio.sleep(0.01)

    a.log_event("test-topic", {})
    await event_handler_started.wait()
    await c._close(fast=True)
    assert exc_info is not None


@gen_cluster(client=True, nthreads=[("", 1)])
async def test_events_all_servers_use_same_channel(c, s, a):
    """Ensure that logs from all server types (scheduler, worker, nanny)
    and the clients themselves arrive"""

    log = []

    def user_event_handler(event):
        log.append(event)

    c.subscribe_topic("test-topic", user_event_handler)

    while not s.event_subscriber["test-topic"]:
        await asyncio.sleep(0.01)

    async with Nanny(s.address) as n:
        a.log_event("test-topic", "worker")
        n.log_event("test-topic", "nanny")
        s.log_event("test-topic", "scheduler")
        await c.log_event("test-topic", "client")

    while not len(log) == 4 == len(set(log)):
        await asyncio.sleep(0.1)


@gen_cluster(client=True, nthreads=[])
async def test_events_unsubscribe_raises_if_unknown(c, s):
    with pytest.raises(ValueError, match="No event handler known for topic unknown"):
        c.unsubscribe_topic("unknown")


@gen_cluster(client=True)
async def test_log_event_warn(c, s, a, b):
    def foo():
        get_worker().log_event(["foo", "warn"], "Hello!")

    with pytest.warns(Warning, match="Hello!"):
        await c.submit(foo)


@gen_cluster(client=True)
async def test_log_event_warn_dask_warns(c, s, a, b):
    from dask.distributed import warn

    def foo():
        warn("Hello!")

    with pytest.warns(Warning, match="Hello!"):
        await c.submit(foo)


@gen_cluster(client=True, Worker=Nanny)
async def test_print(c, s, a, b, capsys):
    from dask.distributed import print

    def foo():
        print("Hello!", 123, sep=":")

    await c.submit(foo)

    out, err = capsys.readouterr()
    assert "Hello!:123" in out


@gen_cluster(client=True, Worker=Nanny)
async def test_print_non_msgpack_serializable(c, s, a, b, capsys):
    from dask.distributed import print

    def foo():
        print(object())

    await c.submit(foo)

    out, err = capsys.readouterr()
    assert "<object object at" in out


def test_print_simple(capsys):
    from dask.distributed import print

    print("Hello!", 123, sep=":")

    out, err = capsys.readouterr()
    assert "Hello!:123" in out


def _verify_cluster_dump(
    url: str | pathlib.PosixPath, format: str, addresses: set[str]
) -> dict:
    fsspec = pytest.importorskip("fsspec")  # for load_cluster_dump
    url = str(url) + (".msgpack.gz" if format == "msgpack" else ".yaml")
    state = load_cluster_dump(url)

    assert isinstance(state, dict)
    assert "scheduler" in state
    assert "workers" in state
    assert "versions" in state
    assert state["workers"].keys() == addresses
    return state


def test_dump_cluster_state_write_from_scheduler(c, s, a, b, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    scheduler_dir = tmp_path / "scheduler"
    scheduler_dir.mkdir()
    c.run_on_scheduler(os.chdir, str(scheduler_dir))

    c.dump_cluster_state("not-url")
    assert (tmp_path / "not-url.msgpack.gz").is_file()

    c.dump_cluster_state("file://is-url")
    assert (scheduler_dir / "is-url.msgpack.gz").is_file()

    c.dump_cluster_state("file://local-explicit", write_from_scheduler=False)
    assert (tmp_path / "local-explicit.msgpack.gz").is_file()

    c.dump_cluster_state("scheduler-explicit", write_from_scheduler=True)
    assert (scheduler_dir / "scheduler-explicit.msgpack.gz").is_file()


@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("_format", ["msgpack", "yaml"])
def test_dump_cluster_state_sync(c, s, a, b, tmp_path, _format, local):
    filename = tmp_path / "foo"
    if not local:
        pytest.importorskip("fsspec")
        # Make it look like an fsspec path
        filename = f"file://{filename}"
    c.dump_cluster_state(filename, format=_format)
    _verify_cluster_dump(filename, _format, {a["address"], b["address"]})


@pytest.mark.parametrize("local", [True, False])
@pytest.mark.parametrize("_format", ["msgpack", "yaml"])
@gen_cluster(client=True)
async def test_dump_cluster_state_async(c, s, a, b, tmp_path, _format, local):
    filename = tmp_path / "foo"
    if not local:
        pytest.importorskip("fsspec")
        # Make it look like an fsspec path
        filename = f"file://{filename}"
    await c.dump_cluster_state(filename, format=_format)
    _verify_cluster_dump(filename, _format, {a.address, b.address})


@pytest.mark.parametrize("local", [True, False])
@gen_cluster(client=True)
async def test_dump_cluster_state_json(c, s, a, b, tmp_path, local):
    filename = tmp_path / "foo"
    if not local:
        pytest.importorskip("fsspec")
        # Make it look like an fsspec path
        filename = f"file://{filename}"
    with pytest.raises(ValueError, match="Unsupported format"):
        await c.dump_cluster_state(filename, format="json")


@gen_cluster(client=True)
async def test_dump_cluster_state_exclude_default(c, s, a, b, tmp_path):
    futs = c.map(inc, range(10))
    while len(s.tasks) != len(futs):
        await asyncio.sleep(0.01)
    excluded_by_default = [
        "run_spec",
    ]

    filename = tmp_path / "foo"
    await c.dump_cluster_state(
        filename=filename,
        format="yaml",
    )

    with open(f"{filename}.yaml") as fd:
        state = yaml.safe_load(fd)

    assert "workers" in state
    assert len(state["workers"]) == len(s.workers)
    for worker, worker_dump in state["workers"].items():
        for k, task_dump in worker_dump["tasks"].items():
            assert not any(blocked in task_dump for blocked in excluded_by_default)
            assert k in s.tasks
    assert "scheduler" in state
    assert "tasks" in state["scheduler"]
    tasks = state["scheduler"]["tasks"]
    assert len(tasks) == len(futs)
    for k, task_dump in tasks.items():
        assert not any(blocked in task_dump for blocked in excluded_by_default)
        assert k in s.tasks

    await c.dump_cluster_state(
        filename=filename,
        format="yaml",
        exclude=(),
    )

    with open(f"{filename}.yaml") as fd:
        state = yaml.safe_load(fd)

    assert "workers" in state
    assert len(state["workers"]) == len(s.workers)
    for worker, worker_dump in state["workers"].items():
        for k, task_dump in worker_dump["tasks"].items():
            assert all(blocked in task_dump for blocked in excluded_by_default)
            assert k in s.tasks
    assert "scheduler" in state
    assert "tasks" in state["scheduler"]
    tasks = state["scheduler"]["tasks"]
    assert len(tasks) == len(futs)
    for k, task_dump in tasks.items():
        assert all(blocked in task_dump for blocked in excluded_by_default)
        assert k in s.tasks


class TestClientSecurityLoader:
    @contextmanager
    def config_loader(self, monkeypatch, loader):
        module_name = "totally_fake_module_name_1"
        module = types.ModuleType(module_name)
        module.loader = loader
        with monkeypatch.context() as m:
            m.setitem(sys.modules, module_name, module)
            with dask.config.set(
                {"distributed.client.security-loader": f"{module_name}.loader"}
            ):
                yield

    @gen_test()
    async def test_security_loader(self, monkeypatch):
        security = tls_only_security()

        async with Scheduler(
            dashboard_address=":0", protocol="tls", security=security
        ) as scheduler:

            def loader(info):
                assert info == {"address": scheduler.address}
                return security

            with self.config_loader(monkeypatch, loader):
                async with Client(scheduler.address, asynchronous=True) as client:
                    assert client.security is security

    @gen_test()
    async def test_security_loader_ignored_if_explicit_security_provided(
        self, monkeypatch
    ):
        security = tls_only_security()

        def loader(info):
            assert False

        async with Scheduler(
            dashboard_address=":0", protocol="tls", security=security
        ) as scheduler:
            with self.config_loader(monkeypatch, loader):
                async with Client(
                    scheduler.address, security=security, asynchronous=True
                ) as client:
                    assert client.security is security

    @gen_test()
    async def test_security_loader_ignored_if_returns_none(self, monkeypatch):
        """Test that if a security loader is configured, but it returns `None`,
        then the default security configuration is used"""
        ca_file = get_cert("tls-ca-cert.pem")
        keycert = get_cert("tls-key-cert.pem")

        config = {
            "distributed.comm.require-encryption": True,
            "distributed.comm.tls.ca-file": ca_file,
            "distributed.comm.tls.client.cert": keycert,
            "distributed.comm.tls.scheduler.cert": keycert,
            "distributed.comm.tls.worker.cert": keycert,
        }

        def loader(info):
            loader.called = True
            return None

        with dask.config.set(config):
            async with Scheduler(dashboard_address=":0", protocol="tls") as scheduler:
                # Smoketest to make sure config was picked up (so we're actually testing something)
                assert scheduler.security.tls_client_cert
                assert scheduler.security.tls_scheduler_cert
                with self.config_loader(monkeypatch, loader):
                    async with Client(scheduler.address, asynchronous=True) as client:
                        assert (
                            client.security.tls_client_cert
                            == scheduler.security.tls_client_cert
                        )

        assert loader.called

    @gen_test()
    async def test_security_loader_import_failed(self):
        security = tls_only_security()

        with dask.config.set(
            {"distributed.client.security-loader": "totally_fake_module_name_2.loader"}
        ):
            with pytest.raises(ImportError, match="totally_fake_module_name_2.loader"):
                async with Client("tls://bad-address:8888", asynchronous=True):
                    pass


@pytest.mark.avoid_ci(reason="This is slow and probably not worth the cost")
@pytest.mark.slow
@gen_cluster(client=True)
async def test_benchmark_hardware(c, s, a, b):
    result = await c.benchmark_hardware()
    assert set(result) == {"disk", "memory", "network"}
    assert all(isinstance(v, float) for d in result.values() for v in d.values())


@gen_cluster(client=True, nthreads=[])
async def test_benchmark_hardware_no_workers(c, s):
    assert await c.benchmark_hardware() == {"memory": {}, "disk": {}, "network": {}}


@gen_cluster(client=True, nthreads=[])
async def test_wait_for_workers_updates_info(c, s):
    async with Worker(s.address):
        await c.wait_for_workers(1)
        assert c.scheduler_info()["workers"]


client_script = """
from dask.distributed import Client
if __name__ == "__main__":
    client = Client(processes=%s, n_workers=1, scheduler_port=0, dashboard_address=":0")
"""


@pytest.mark.parametrize("processes", [True, False])
def test_quiet_close_process(processes, tmp_path):
    with open(tmp_path / "script.py", mode="w") as f:
        f.write(client_script % processes)

    with popen([sys.executable, tmp_path / "script.py"], capture_output=True) as proc:
        out, err = proc.communicate(timeout=10)

    assert not out
    assert not err

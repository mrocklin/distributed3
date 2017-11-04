from __future__ import print_function, division, absolute_import

from concurrent.futures import CancelledError
from datetime import timedelta
from operator import add
import random
import sys
import os
from time import sleep

from dask import delayed
import pytest
from toolz import concat, sliding_window
import psutil

from distributed import Client, wait, Nanny
from distributed.config import config
from distributed.metrics import time
from distributed.utils import All
from distributed.utils_test import (gen_cluster, cluster, inc, slowinc,
                                    slowadd, slow, slowsum, bump_rlimit)
from distributed.utils_test import loop # flake8: noqa
from distributed.client import wait
from tornado import gen


def get_folder_size(folder):
    total_size = os.path.getsize(folder)
    for item in os.listdir(folder):
        itempath = os.path.join(folder, item)
        if os.path.isfile(itempath):
            total_size += os.path.getsize(itempath)
        elif os.path.isdir(itempath):
            total_size += get_folder_size(itempath)
    return total_size


@gen_cluster(client=True)
def test_stress_1(c, s, a, b):
    n = 2**6

    seq = c.map(inc, range(n))
    while len(seq) > 1:
        yield gen.sleep(0.1)
        seq = [c.submit(add, seq[i], seq[i + 1])
               for i in range(0, len(seq), 2)]
    result = yield seq[0]
    assert result == sum(map(inc, range(n)))


@pytest.mark.parametrize(('func', 'n'), [(slowinc, 100), (inc, 1000)])
def test_stress_gc(loop, func, n):
    with cluster() as (s, [a, b]):
        with Client(s['address'], loop=loop) as c:
            x = c.submit(func, 1)
            for i in range(n):
                x = c.submit(func, x)

            assert x.result() == n + 2


@pytest.mark.skipif(sys.platform.startswith('win'),
                    reason="test can leave dangling RPC objects")
@gen_cluster(client=True, ncores=[('127.0.0.1', 1)] * 4, timeout=None)
def test_cancel_stress(c, s, *workers):
    da = pytest.importorskip('dask.array')
    x = da.random.random((40, 40), chunks=(1, 1))
    x = c.persist(x)
    yield wait([x])
    y = (x.sum(axis=0) + x.sum(axis=1) + 1).std()
    for i in range(5):
        f = c.compute(y)
        while len(s.waiting) > (len(y.dask) - len(x.dask)) / 2:
            yield gen.sleep(0.01)
        yield c._cancel(f)


def test_cancel_stress_sync(loop):
    da = pytest.importorskip('dask.array')
    x = da.random.random((40, 40), chunks=(1, 1))
    with cluster(active_rpc_timeout=10) as (s, [a, b]):
        with Client(s['address'], loop=loop) as c:
            x = c.persist(x)
            y = (x.sum(axis=0) + x.sum(axis=1) + 1).std()
            wait(x)
            for i in range(5):
                f = c.compute(y)
                sleep(1)
                c.cancel(f)


@gen_cluster(ncores=[], client=True, timeout=None)
def test_stress_creation_and_deletion(c, s):
    # Assertions are handled by the validate mechanism in the scheduler
    s.allowed_failures = 100000
    da = pytest.importorskip('dask.array')

    x = da.random.random(size=(2000, 2000), chunks=(100, 100))
    y = (x + 1).T + (x * 2) - x.mean(axis=1)

    z = c.persist(y)

    @gen.coroutine
    def create_and_destroy_worker(delay):
        start = time()
        while time() < start + 10:
            n = Nanny(s.ip, s.port, ncores=2, loop=s.loop)
            n.start(0)

            yield gen.sleep(delay)

            yield n._close()
            print("Killed nanny")

    yield gen.with_timeout(timedelta(minutes=1),
                           All([create_and_destroy_worker(0.1 * i) for i in
                                range(10)]))


@gen_cluster(ncores=[('127.0.0.1', 1)] * 10, client=True, timeout=60)
def test_stress_scatter_death(c, s, *workers):
    import random
    s.allowed_failures = 1000
    np = pytest.importorskip('numpy')
    L = yield c.scatter([np.random.random(10000) for i in range(len(workers))])
    yield c._replicate(L, n=2)

    adds = [delayed(slowadd, pure=True)(random.choice(L),
                                        random.choice(L),
                                        delay=0.05)
            for i in range(50)]

    adds = [delayed(slowadd, pure=True)(a, b, delay=0.02)
            for a, b in sliding_window(2, adds)]

    futures = c.compute(adds)

    alive = list(workers)

    from distributed.scheduler import logger

    for i in range(7):
        yield gen.sleep(0.1)
        try:
            s.validate_state()
        except Exception as c:
            logger.exception(c)
            if config.get('log-on-err'):
                import pdb
                pdb.set_trace()
            else:
                raise
        w = random.choice(alive)
        yield w._close()
        alive.remove(w)

    try:
        yield gen.with_timeout(timedelta(seconds=25), c._gather(futures))
    except gen.TimeoutError:
        ws = {w.address: w for w in workers if w.status != 'closed'}
        print(s.processing)
        print(ws)
        print(futures)
        try:
            worker = [w for w in ws.values() if w.waiting_for_data][0]
        except Exception:
            pass
        if config.get('log-on-err'):
            import pdb
            pdb.set_trace()
        else:
            raise
    except CancelledError:
        pass


def vsum(*args):
    return sum(args)


@pytest.mark.avoid_travis
@slow
@gen_cluster(client=True, ncores=[('127.0.0.1', 1)] * 80, timeout=1000)
def test_stress_communication(c, s, *workers):
    s.validate = False  # very slow otherwise
    da = pytest.importorskip('dask.array')
    # Test consumes many file descriptors and can hang if the limit is too low
    resource = pytest.importorskip('resource')
    bump_rlimit(resource.RLIMIT_NOFILE, 8192)

    n = 20
    xs = [da.random.random((100, 100), chunks=(5, 5)) for i in range(n)]
    ys = [x + x.T for x in xs]
    z = da.atop(vsum, 'ij', *concat(zip(ys, ['ij'] * n)), dtype='float64')

    future = c.compute(z.sum())

    result = yield future
    assert isinstance(result, float)


@pytest.mark.skip
@gen_cluster(client=True, ncores=[('127.0.0.1', 1)] * 10, timeout=60)
def test_stress_steal(c, s, *workers):
    s.validate = False
    for w in workers:
        w.validate = False

    dinc = delayed(slowinc)
    L = [delayed(slowinc)(i, delay=0.005) for i in range(100)]
    for i in range(5):
        L = [delayed(slowsum)(part, delay=0.005)
             for part in sliding_window(5, L)]

    total = delayed(sum)(L)
    future = c.compute(total)

    while future.status != 'finished':
        yield gen.sleep(0.1)
        for i in range(3):
            a = random.choice(workers)
            b = random.choice(workers)
            if a is not b:
                s.work_steal(a.address, b.address, 0.5)
        if not s.processing:
            break


@slow
@gen_cluster(ncores=[('127.0.0.1', 1)] * 10, client=True, timeout=120)
def test_close_connections(c, s, *workers):
    da = pytest.importorskip('dask.array')
    x = da.random.random(size=(1000, 1000), chunks=(1000, 1))
    for i in range(3):
        x = x.rechunk((1, 1000))
        x = x.rechunk((1000, 1))

    future = c.compute(x.sum())
    while any(s.processing.values()):
        yield gen.sleep(0.5)
        worker = random.choice(list(workers))
        for comm in worker._comms:
            comm.abort()
        # print(frequencies(s.task_state.values()))
        # for w in workers:
        #     print(w)

    yield wait(future)


@pytest.mark.xfail(reason="IOStream._handle_write blocks on large write_buffer"
                          " https://github.com/tornadoweb/tornado/issues/2110")
@gen_cluster(client=True, timeout=20, ncores=[('127.0.0.1', 1)])
def test_no_delay_during_large_transfer(c, s, w):
    pytest.importorskip('crick')
    np = pytest.importorskip('numpy')
    x = np.random.random(100000000)

    # Reset digests
    from distributed.counter import Digest
    from collections import defaultdict
    from functools import partial
    from dask.diagnostics import ResourceProfiler

    for server in [s, w]:
        server.digests = defaultdict(partial(Digest, loop=server.io_loop))
        server._last_tick = time()

    with ResourceProfiler(dt=0.01) as rprof:
        future = yield c.scatter(x, direct=True, hash=False)
        yield gen.sleep(0.5)

    for server in [s, w]:
        assert server.digests['tick-duration'].components[0].max() < 0.5

    nbytes = np.array([t.mem for t in rprof.results])
    nbytes -= nbytes[0]
    assert nbytes.max() < (x.nbytes * 2) / 1e6
    assert nbytes[-1] < (x.nbytes * 1.2) / 1e6


def test_nanny_no_terminate_on_cyclic_ref(tmpdir):
    from time import sleep

    class BigCyclicRef(object):
        def __init__(self, size):
            sleep(0.1)
            self.data = b'0' * size
            self.cyclic_ref = self

    worker_folder = str(tmpdir)
    memory_limit = int(1e9)
    size = int(1e7)
    n_tasks = 200
    w_kwargs = {'memory_limit': memory_limit, 'local_dir': worker_folder}
    with cluster(nworkers=1, nanny=True, worker_kwargs=w_kwargs) as (s, [a]):
        with Client(s['address']) as c:

            def get_worker_pids():
                info = c.scheduler_info()['workers']
                return [w['pid'] for w in info.values()]


            pids = get_worker_pids()
            # 200 x 10 MB should yield 2 GB of data on the worker. Because
            # the memory limit is set to 1 GB, the worker should evict data
            # to its disk-based store and use the GC to free the data with
            # cyclic ref so as to complete the task. The nanny memory check
            # should not be triggered.
            futures = [c.submit(BigCyclicRef, size, pure=False)
                       for _ in range(n_tasks)]
            start = time()
            while True:
                sleep(0.1)
                # Ensure that the worker process has not been restarted by the
                # nanny memory monitor.
                assert get_worker_pids() == pids
                assert time() < start + 30
                if all(f.done() for f in futures):
                    break

            # Check that the worker process does not exceed the use provided
            # memory limit.
            worker_mem = psutil.Process(pids[0]).memory_info().rss
            assert worker_mem < memory_limit

            # The remaining data should have been evicted to disk. We assume
            # that the eviction to disk does not support compression for those
            # types of objects:
            total_size = n_tasks * size
            assert get_folder_size(worker_folder) > total_size - memory_limit
    
            # It's still possible to fetch the data of the results of the
            # computation from the client process if needed.
            for f in futures:
                assert len(f.result().data) == size
            
            # Fetching the data should not cause the worker to restart and
            # its memory usage should still be low enough.
            assert get_worker_pids() == pids
            worker_mem = psutil.Process(pids[0]).memory_info().rss
            assert worker_mem < memory_limit

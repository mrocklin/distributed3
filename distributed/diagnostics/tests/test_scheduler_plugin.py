from __future__ import annotations

import asyncio

import pytest

from distributed import Scheduler, SchedulerPlugin, Worker, get_worker
from distributed.utils_test import captured_logger, gen_cluster, gen_test, inc


@gen_cluster(client=True)
async def test_simple(c, s, a, b):
    class Counter(SchedulerPlugin):
        def start(self, scheduler):
            self.scheduler = scheduler
            scheduler.add_plugin(self, name="counter")
            self.count = 0

        def transition(self, key, start, finish, *args, **kwargs):
            if start == "processing" and finish == "memory":
                self.count += 1

    counter = Counter()
    counter.start(s)
    assert counter in s.plugins.values()

    assert counter.count == 0

    x = c.submit(inc, 1)
    y = c.submit(inc, x)
    z = c.submit(inc, y)

    await z

    assert counter.count == 3
    s.remove_plugin("counter")
    assert counter not in s.plugins

    with pytest.raises(ValueError, match="Could not find plugin 'counter'") as e:
        s.remove_plugin("counter")


@gen_cluster(nthreads=[])
async def test_add_remove_worker(s):
    events = []

    class MyPlugin(SchedulerPlugin):
        name = "MyPlugin"

        def add_worker(self, worker, scheduler):
            assert scheduler is s
            events.append(("add_worker", worker))

        def remove_worker(self, worker, scheduler):
            assert scheduler is s
            events.append(("remove_worker", worker))

    plugin = MyPlugin()
    s.add_plugin(plugin)
    assert events == []

    a = Worker(s.address)
    b = Worker(s.address)
    await a
    await b
    await a.close()
    await b.close()

    assert events == [
        ("add_worker", a.address),
        ("add_worker", b.address),
        ("remove_worker", a.address),
        ("remove_worker", b.address),
    ]

    events[:] = []
    s.remove_plugin(plugin.name)
    async with Worker(s.address):
        pass
    assert events == []


@gen_cluster(nthreads=[])
async def test_async_add_remove_worker(s):
    events = []

    class MyPlugin(SchedulerPlugin):
        name = "MyPlugin"

        async def add_worker(self, worker, scheduler):
            assert scheduler is s
            events.append(("add_worker", worker))

        async def remove_worker(self, worker, scheduler):
            assert scheduler is s
            events.append(("remove_worker", worker))

    plugin = MyPlugin()
    s.add_plugin(plugin)
    assert events == []

    async with Worker(s.address) as a:
        async with Worker(s.address) as b:
            pass

    assert set(events) == {
        ("add_worker", a.address),
        ("add_worker", b.address),
        ("remove_worker", a.address),
        ("remove_worker", b.address),
    }

    events[:] = []
    s.remove_plugin(plugin.name)
    async with Worker(s.address):
        pass
    assert events == []


@gen_cluster(nthreads=[])
async def test_async_and_sync_remove_worker(s):
    events = []

    class MyAsyncPlugin(SchedulerPlugin):
        name = "MyAsyncPlugin"

        def __init__(self) -> None:
            super().__init__()
            self.in_remove_worker = asyncio.Event()
            self.block_remove_worker = asyncio.Event()

        async def remove_worker(self, worker, scheduler):
            assert scheduler is s
            self.in_remove_worker.set()
            await self.block_remove_worker.wait()
            events.append(("async_remove_worker", worker))

    class MySyncPlugin(SchedulerPlugin):
        name = "MySyncPlugin"

        def remove_worker(self, worker, scheduler):
            assert scheduler is s
            events.append(("sync_remove_worker", worker))

    sync_plugin_a = MySyncPlugin()
    sync_plugin_b = MySyncPlugin()

    async_plugin = MyAsyncPlugin()
    s.add_plugin(sync_plugin_a, name="a")
    s.add_plugin(async_plugin)
    s.add_plugin(sync_plugin_b, name="c")
    assert events == []

    async with Worker(s.address) as a:
        async with Worker(s.address) as b:
            pass
    await async_plugin.in_remove_worker.wait()
    assert events == [
        ("sync_remove_worker", b.address),
        ("sync_remove_worker", b.address),
        ("sync_remove_worker", a.address),
        ("sync_remove_worker", a.address),
    ]

    async_plugin.block_remove_worker.set()
    await asyncio.sleep(0)

    assert events == [
        ("sync_remove_worker", b.address),
        ("sync_remove_worker", b.address),
        ("sync_remove_worker", a.address),
        ("sync_remove_worker", a.address),
        ("async_remove_worker", b.address),
        ("async_remove_worker", a.address),
    ]


@gen_cluster(nthreads=[])
async def test_failing_async_and_sync_remove_worker(s):
    class MyAsyncPlugin(SchedulerPlugin):
        name = "MyAsyncPlugin"

        def __init__(self) -> None:
            super().__init__()

        async def remove_worker(self, worker, scheduler):
            assert scheduler is s
            await asyncio.sleep(0)
            raise RuntimeError("Async remove_worker failed")

    class MySyncPlugin(SchedulerPlugin):
        name = "MySyncPlugin"

        def remove_worker(self, worker, scheduler):
            assert scheduler is s
            raise RuntimeError("Sync remove_worker failed")

    sync_plugin = MySyncPlugin()
    async_plugin = MyAsyncPlugin()
    s.add_plugin(sync_plugin)
    s.add_plugin(async_plugin)
    with captured_logger("distributed.scheduler") as logger:
        async with Worker(s.address):
            pass
        while "Async remove_worker failed" not in logger.getvalue():
            await asyncio.sleep(0)
        assert "Sync remove_worker failed" in logger.getvalue()


@gen_test()
async def test_lifecycle():
    class LifeCycle(SchedulerPlugin):
        def __init__(self):
            self.history = []

        async def start(self, scheduler):
            self.scheduler = scheduler
            self.history.append("started")

        async def close(self):
            self.history.append("closed")

    plugin = LifeCycle()
    async with Scheduler(plugins=[plugin], dashboard_address=":0") as s:
        pass

    assert plugin.history == ["started", "closed"]
    assert plugin.scheduler is s


@gen_cluster(client=True)
async def test_register_scheduler_plugin(c, s, a, b):
    class Dummy1(SchedulerPlugin):
        name = "Dummy1"

        def start(self, scheduler):
            scheduler.foo = "bar"

    assert not hasattr(s, "foo")
    await c.register_scheduler_plugin(Dummy1())
    assert s.foo == "bar"

    with pytest.warns(UserWarning) as w:
        await c.register_scheduler_plugin(Dummy1())
    assert "Scheduler already contains" in w[0].message.args[0]

    class Dummy2(SchedulerPlugin):
        name = "Dummy2"

        def start(self, scheduler):
            raise RuntimeError("raising in start method")

    n_plugins = len(s.plugins)
    with pytest.raises(RuntimeError, match="raising in start method"):
        await c.register_scheduler_plugin(Dummy2())
    # total number of plugins should be unchanged
    assert n_plugins == len(s.plugins)


@gen_cluster(client=True, config={"distributed.scheduler.pickle": False})
async def test_register_scheduler_plugin_pickle_disabled(c, s, a, b):
    class Dummy1(SchedulerPlugin):
        def start(self, scheduler):
            scheduler.foo = "bar"

    n_plugins = len(s.plugins)
    with pytest.raises(ValueError) as excinfo:
        await c.register_scheduler_plugin(Dummy1())

    msg = str(excinfo.value)
    assert "disallowed from deserializing" in msg
    assert "distributed.scheduler.pickle" in msg

    assert n_plugins == len(s.plugins)


@gen_cluster(client=True)
async def test_log_event_plugin(c, s, a, b):
    class EventPlugin(SchedulerPlugin):
        async def start(self, scheduler: Scheduler) -> None:
            self.scheduler = scheduler
            self.scheduler._recorded_events = list()  # type: ignore

        def log_event(self, name, msg):
            self.scheduler._recorded_events.append((name, msg))

    await c.register_scheduler_plugin(EventPlugin())

    def f():
        get_worker().log_event("foo", 123)

    await c.submit(f)

    assert ("foo", 123) in s._recorded_events


@gen_cluster(client=True)
async def test_register_plugin_on_scheduler(c, s, a, b):
    class MyPlugin(SchedulerPlugin):
        async def start(self, scheduler: Scheduler) -> None:
            scheduler._foo = "bar"  # type: ignore

    await s.register_scheduler_plugin(MyPlugin())

    assert s._foo == "bar"


@gen_cluster(client=True)
async def test_closing_errors_ok(c, s, a, b, capsys):
    class OK(SchedulerPlugin):
        async def before_close(self):
            print(123)

        async def close(self):
            print(456)

    class Bad(SchedulerPlugin):
        async def before_close(self):
            raise Exception("BEFORE_CLOSE")

        async def close(self):
            raise Exception("AFTER_CLOSE")

    await s.register_scheduler_plugin(OK())
    await s.register_scheduler_plugin(Bad())

    with captured_logger("distributed.scheduler") as logger:
        await s.close()

    out, err = capsys.readouterr()
    assert "123" in out
    assert "456" in out

    text = logger.getvalue()
    assert "BEFORE_CLOSE" in text
    text = logger.getvalue()
    assert "AFTER_CLOSE" in text


@gen_cluster(client=True)
async def test_update_graph_hook_simple(c, s, a, b):
    class UpdateGraph(SchedulerPlugin):
        def __init__(self) -> None:
            self.success = False

        def update_graph(  # type: ignore
            self,
            scheduler,
            client,
            keys,
            tasks,
            annotations,
            priority,
            dependencies,
            **kwargs,
        ) -> None:
            assert scheduler is s
            assert client == c.id
            # If new parameters are added we should add a test
            assert not kwargs
            assert keys == {"foo"}
            assert tasks == ["foo"]
            assert annotations == {}
            assert len(priority) == 1
            assert isinstance(priority["foo"], tuple)
            assert dependencies == {"foo": set()}
            self.success = True

    plugin = UpdateGraph()
    s.add_plugin(plugin, name="update-graph")

    await c.submit(inc, 5, key="foo")
    assert plugin.success


import dask
from dask import delayed


@gen_cluster(client=True)
async def test_update_graph_hook_complex(c, s, a, b):
    class UpdateGraph(SchedulerPlugin):
        def __init__(self) -> None:
            self.success = False

        def update_graph(  # type: ignore
            self,
            scheduler,
            client,
            keys,
            tasks,
            annotations,
            priority,
            dependencies,
            **kwargs,
        ) -> None:
            assert scheduler is s
            assert client == c.id
            # If new parameters are added we should add a test
            assert not kwargs
            assert keys == {"sum"}
            assert set(tasks) == {"sum", "f1", "f3", "f2"}
            assert annotations == {
                "global_annot": {k: 24 for k in tasks},
                "layer": {"f2": "explicit"},
                "len_key": {"f3": 2},
                "priority": {"f2": 13},
            }
            assert len(priority) == len(tasks), priority
            assert priority["f2"][0] == -13
            for k in keys:
                assert k in dependencies
            assert dependencies["f1"] == set()
            assert dependencies["sum"] == {"f1", "f3"}

            self.success = True

    plugin = UpdateGraph()
    s.add_plugin(plugin, name="update-graph")
    del_inc = delayed(inc)
    f1 = del_inc(1, dask_key_name="f1")
    with dask.annotate(layer="explicit", priority=13):
        f2 = del_inc(2, dask_key_name="f2")
    with dask.annotate(len_key=lambda x: len(x)):
        f3 = del_inc(f2, dask_key_name="f3")

    f4 = delayed(sum)([f1, f3], dask_key_name="sum")

    with dask.annotate(global_annot=24):
        await c.compute(f4)
    assert plugin.success

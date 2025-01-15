from __future__ import annotations

import asyncio
import contextvars
import functools
import io
import logging
import multiprocessing
import os
import queue
import socket
import sys
import traceback
import warnings
import xml
from array import array
from collections import deque
from concurrent.futures import Executor, Future, ThreadPoolExecutor
from contextvars import ContextVar
from time import sleep
from unittest import mock

import pytest
from tornado.ioloop import IOLoop

import dask

from distributed.compatibility import MACOS, WINDOWS, asyncio_run
from distributed.config import get_loop_factory
from distributed.metrics import time
from distributed.utils import (
    All,
    Log,
    Logs,
    LoopRunner,
    RateLimiterFilter,
    TimeoutError,
    TupleComparable,
    ensure_ip,
    ensure_memoryview,
    format_dashboard_link,
    get_ip_interface,
    get_mp_context,
    get_traceback,
    is_kernel,
    is_valid_xml,
    iscoroutinefunction,
    json_load_robust,
    log_errors,
    nbytes,
    offload,
    open_port,
    parse_ports,
    read_block,
    recursive_to_dict,
    run_in_executor_with_context,
    seek_delimiter,
    set_thread_state,
    sync,
    thread_state,
    truncate_exception,
    warn_on_duration,
)
from distributed.utils_test import (
    _UnhashableCallable,
    captured_logger,
    div,
    gen_test,
    has_ipv6,
    inc,
    throws,
)


@gen_test()
async def test_All():
    async def throws():
        1 / 0

    async def slow():
        await asyncio.sleep(10)

    async def inc(x):
        return x + 1

    results = await All([inc(i) for i in range(10)])
    assert results == list(range(1, 11))

    start = time()
    for tasks in [[throws(), slow()], [slow(), throws()]]:
        try:
            await All(tasks)
            assert False
        except ZeroDivisionError:
            pass
        end = time()
        assert end - start < 10


def test_sync(loop_in_thread):
    async def f(x, y):
        await asyncio.sleep(0.01)
        return x, y

    result = sync(loop_in_thread, f, 1, y=2)
    assert result == (1, 2)


def test_sync_error(loop_in_thread):
    with pytest.raises(RuntimeError, match="hello!") as exc:
        sync(loop_in_thread, throws, 1)

    L = traceback.format_tb(exc.value.__traceback__)
    assert any("throws" in line for line in L)

    def function1(x):
        return function2(x)

    def function2(x):
        return throws(x)

    with pytest.raises(RuntimeError, match="hello!") as exc:
        sync(loop_in_thread, function1, 1)

    L = traceback.format_tb(exc.value.__traceback__)
    assert any("function1" in line for line in L)
    assert any("function2" in line for line in L)


def test_sync_timeout(loop_in_thread):
    with pytest.raises(TimeoutError):
        sync(loop_in_thread, asyncio.sleep, 0.5, callback_timeout=0.05)

    with pytest.raises(TimeoutError):
        sync(loop_in_thread, asyncio.sleep, 0.5, callback_timeout="50ms")


def test_sync_closed_loop():
    async def get_loop():
        return IOLoop.current()

    loop = asyncio_run(get_loop(), loop_factory=get_loop_factory())
    loop.close()

    with pytest.raises(RuntimeError) as exc_info:
        sync(loop, inc, 1)
    exc_info.match("IOLoop is clos(ed|ing)")


def test_sync_contextvars(loop_in_thread):
    """Test that sync() propagates contextvars - namely,
    distributed.metrics.context_meter callbacks
    """
    v = ContextVar("v", default=0)

    async def f():
        return v.get()

    assert sync(loop_in_thread, f) == 0
    tok = v.set(1)
    assert sync(loop_in_thread, f) == 1
    v.reset(tok)


def test_is_kernel():
    pytest.importorskip("IPython")
    assert is_kernel() is False


# @pytest.mark.leaking('fds')
# def test_zzz_leaks(l=[]):
# import os, subprocess
# l.append(b"x" * (17 * 1024**2))
# os.open(__file__, os.O_RDONLY)
# subprocess.Popen('sleep 100', shell=True, stdin=subprocess.DEVNULL)


def test_ensure_ip():
    assert ensure_ip("localhost") in ("127.0.0.1", "::1")
    assert ensure_ip("123.123.123.123") == "123.123.123.123"
    assert ensure_ip("8.8.8.8") == "8.8.8.8"
    if has_ipv6():
        assert ensure_ip("2001:4860:4860::8888") == "2001:4860:4860::8888"
        assert ensure_ip("::1") == "::1"


@pytest.mark.skipif(WINDOWS, reason="TODO")
def test_get_ip_interface():
    iface = "lo0" if MACOS else "lo"
    assert get_ip_interface(iface) == "127.0.0.1"
    with pytest.raises(ValueError, match=f"'__notexist'.+network interface.+'{iface}'"):
        get_ip_interface("__notexist")


@pytest.mark.skipif(
    WINDOWS, reason="Windows doesn't support different multiprocessing contexts"
)
def test_get_mp_context():
    # this will need updated if the default multiprocessing context changes from spawn.
    assert get_mp_context() is multiprocessing.get_context("spawn")
    with dask.config.set({"distributed.worker.multiprocessing-method": "forkserver"}):
        assert get_mp_context() is multiprocessing.get_context("forkserver")
    with dask.config.set({"distributed.worker.multiprocessing-method": "fork"}):
        assert get_mp_context() is multiprocessing.get_context("fork")


def test_truncate_exception():
    e = ValueError("a" * 1000)
    assert len(str(e)) >= 1000
    f = truncate_exception(e, 100)
    assert type(f) == type(e)
    assert len(str(f)) < 200
    assert "aaaa" in str(f)

    e = ValueError("a")
    assert truncate_exception(e) is e


def test_get_traceback():
    def a(x):
        return div(x, 0)

    def b(x):
        return a(x)

    def c(x):
        return b(x)

    try:
        c(1)
    except Exception as e:
        tb = get_traceback()
        assert type(tb).__name__ == "traceback"


def test_read_block():
    delimiter = b"\n"
    data = delimiter.join([b"123", b"456", b"789"])
    f = io.BytesIO(data)

    assert read_block(f, 1, 2) == b"23"
    assert read_block(f, 0, 1, delimiter=b"\n") == b"123\n"
    assert read_block(f, 0, 2, delimiter=b"\n") == b"123\n"
    assert read_block(f, 0, 3, delimiter=b"\n") == b"123\n"
    assert read_block(f, 0, 5, delimiter=b"\n") == b"123\n456\n"
    assert read_block(f, 0, 8, delimiter=b"\n") == b"123\n456\n789"
    assert read_block(f, 0, 100, delimiter=b"\n") == b"123\n456\n789"
    assert read_block(f, 1, 1, delimiter=b"\n") == b""
    assert read_block(f, 1, 5, delimiter=b"\n") == b"456\n"
    assert read_block(f, 1, 8, delimiter=b"\n") == b"456\n789"

    for ols in [[(0, 3), (3, 3), (6, 3), (9, 2)], [(0, 4), (4, 4), (8, 4)]]:
        out = [read_block(f, o, l, b"\n") for o, l in ols]
        assert b"".join(filter(None, out)) == data


def test_seek_delimiter_endline():
    f = io.BytesIO(b"123\n456\n789")

    # if at zero, stay at zero
    seek_delimiter(f, b"\n", 5)
    assert f.tell() == 0

    # choose the first block
    for bs in [1, 5, 100]:
        f.seek(1)
        seek_delimiter(f, b"\n", blocksize=bs)
        assert f.tell() == 4

    # handle long delimiters well, even with short blocksizes
    f = io.BytesIO(b"123abc456abc789")
    for bs in [1, 2, 3, 4, 5, 6, 10]:
        f.seek(1)
        seek_delimiter(f, b"abc", blocksize=bs)
        assert f.tell() == 6

    # End at the end
    f = io.BytesIO(b"123\n456")
    f.seek(5)
    seek_delimiter(f, b"\n", 5)
    assert f.tell() == 7


@pytest.mark.parametrize(
    "data",
    [
        b"",
        bytearray(),
        b"1",
        bytearray(b"1"),
        memoryview(b"1"),
        memoryview(bytearray(b"1")),
        array("B", b"1"),
        array("I", range(5)),
        memoryview(b"123456")[1:-1],
        memoryview(b"123456")[::2],
        memoryview(array("I", range(5)))[1:-1],
        memoryview(array("I", range(5)))[::2],
        memoryview(b"123456").cast("B", (2, 3)),
        memoryview(b"0123456789").cast("B", (5, 2))[1:-1],
        memoryview(b"0123456789").cast("B", (5, 2))[::2],
    ],
)
def test_ensure_memoryview(data):
    data_mv = memoryview(data)
    result = ensure_memoryview(data)
    assert isinstance(result, memoryview)
    assert result.contiguous
    assert result.ndim == 1
    assert result.format == "B"
    assert result == bytes(data_mv)
    if data_mv.nbytes and data_mv.contiguous:
        assert result.readonly == data_mv.readonly
        if isinstance(data, memoryview):
            if data.ndim == 1 and data.format == "B":
                assert id(result) == id(data)
            else:
                assert id(data) != id(result)
    else:
        assert id(result.obj) != id(data_mv.obj)
        assert not result.readonly


@pytest.mark.parametrize(
    "dt, nitems, shape, strides",
    [
        ("i8", 12, (12,), (8,)),
        ("i8", 12, (3, 4), (32, 8)),
        ("i8", 12, (4, 3), (8, 32)),
        ("i8", 12, (3, 2), (32, 16)),
        ("i8", 12, (2, 3), (16, 32)),
    ],
)
def test_ensure_memoryview_ndarray(dt, nitems, shape, strides):
    np = pytest.importorskip("numpy")
    data = np.ndarray(
        shape, dtype=dt, buffer=np.arange(nitems, dtype=dt), strides=strides
    )
    result = ensure_memoryview(data)
    assert isinstance(result, memoryview)
    assert result.ndim == 1
    assert result.format == "B"
    assert result.contiguous


def test_ensure_memoryview_pyarrow_buffer():
    pa = pytest.importorskip("pyarrow")
    buf = pa.py_buffer(b"123")
    result = ensure_memoryview(buf)
    assert isinstance(result, memoryview)


def test_nbytes():
    np = pytest.importorskip("numpy")

    def check(obj, expected):
        assert nbytes(obj) == expected
        assert nbytes(memoryview(obj)) == expected

    check(b"123", 3)
    check(bytearray(b"4567"), 4)

    multi_dim = np.ones(shape=(10, 10))
    scalar = np.array(1)

    check(multi_dim, multi_dim.nbytes)
    check(scalar, scalar.nbytes)


def test_open_port():
    port = open_port()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", port))
    s.close()


def test_set_thread_state():
    with set_thread_state(x=1):
        assert thread_state.x == 1

    assert not hasattr(thread_state, "x")


def assert_running(loop):
    """
    Raise if the given IOLoop is not running.
    """
    q = queue.Queue()
    loop.add_callback(q.put, 42)
    assert q.get(timeout=1) == 42


def assert_not_running(loop):
    """
    Raise if the given IOLoop is running.
    """
    q = queue.Queue()
    try:
        loop.add_callback(q.put, 42)
    except RuntimeError:
        # On AsyncIOLoop, can't add_callback() after the loop is closed
        pass
    else:
        with pytest.raises(queue.Empty):
            q.get(timeout=0.02)


def test_loop_runner(loop_in_thread):
    # Implicit loop
    async def make_looprunner_in_async_context():
        return IOLoop.current(), LoopRunner()

    loop, runner = asyncio_run(
        make_looprunner_in_async_context(), loop_factory=get_loop_factory()
    )
    with pytest.raises(
        RuntimeError,
        match=r"Accessing the loop property while the loop is not running is not supported",
    ):
        runner.loop
    assert not runner.is_started()
    runner.start()
    assert runner.is_started()
    assert_running(runner.loop)
    assert runner.loop is not loop
    runner.stop()
    assert not runner.is_started()
    with pytest.raises(
        RuntimeError,
        match=r"Accessing the loop property while the loop is not running is not supported",
    ):
        runner.loop

    async def make_io_loop_in_async_context():
        # calling IOLoop() raises DeprecationWarning: There is no current event loop
        return IOLoop.current()

    # Explicit loop
    loop = asyncio_run(make_io_loop_in_async_context(), loop_factory=get_loop_factory())
    with pytest.raises(
        RuntimeError,
        match=r"Constructing LoopRunner\(loop=loop\) without a running loop is not supported",
    ):
        LoopRunner(loop=loop)

    # Explicit loop, already started
    runner = LoopRunner(loop=loop_in_thread)
    assert not runner.is_started()
    assert_running(loop_in_thread)
    runner.start()
    assert runner.is_started()
    assert_running(loop_in_thread)
    runner.stop()
    assert not runner.is_started()
    assert_running(loop_in_thread)

    # Implicit loop, asynchronous=True
    with pytest.raises(
        RuntimeError,
        match=r"Constructing LoopRunner\(asynchronous=True\) without a running loop is not supported",
    ):
        LoopRunner(asynchronous=True)

    # Explicit loop
    loop = asyncio_run(make_io_loop_in_async_context(), loop_factory=get_loop_factory())
    with pytest.raises(
        RuntimeError,
        match=r"Constructing LoopRunner\(loop=loop\) without a running loop is not supported",
    ):
        LoopRunner(loop=loop, asynchronous=True)


def test_two_loop_runners(loop_in_thread):
    # Loop runners tied to the same loop should cooperate

    # ABCCBA
    a = LoopRunner()
    a.start()
    loop = a.loop
    assert_running(loop)
    b = LoopRunner(loop=loop)
    c = LoopRunner(loop=loop)
    b.start()
    assert_running(loop)
    c.start()
    assert_running(loop)
    c.stop()
    assert_running(loop)
    b.stop()
    assert_running(loop)
    a.stop()
    assert_not_running(loop)

    # ABCABC
    a = LoopRunner()
    a.start()
    loop = a.loop
    assert_running(loop)
    b = LoopRunner(loop=loop)
    c = LoopRunner(loop=loop)
    assert_running(loop)
    b.start()
    assert_running(loop)
    c = LoopRunner(loop=loop)
    c.start()
    assert_running(loop)
    a.stop()
    assert_running(loop)
    b.stop()
    assert_running(loop)
    c.stop()
    assert_not_running(loop)

    # Explicit loop, already started
    a = LoopRunner(loop=loop_in_thread)
    b = LoopRunner(loop=loop_in_thread)
    assert_running(loop_in_thread)
    a.start()
    assert_running(loop_in_thread)
    b.start()
    assert_running(loop_in_thread)
    a.stop()
    assert_running(loop_in_thread)
    b.stop()
    assert_running(loop_in_thread)


@gen_test()
async def test_loop_runner_gen():
    runner = LoopRunner(asynchronous=True)
    assert runner.loop is IOLoop.current()
    assert not runner.is_started()
    await asyncio.sleep(0.01)
    runner.start()
    assert runner.is_started()
    await asyncio.sleep(0.01)
    runner.stop()
    assert not runner.is_started()
    await asyncio.sleep(0.01)


def test_loop_runner_exception_in_start(cleanup):
    class MyException(Exception):
        pass

    with (
        mock.patch("tornado.ioloop.IOLoop.current", side_effect=MyException),
        pytest.raises(MyException),
    ):
        LoopRunner().start()


def test_loop_runner_exception_in_teardown(cleanup):
    runner = LoopRunner()
    runner.start()

    async def cancel_all_tasks():
        current_task = asyncio.current_task()
        for task in asyncio.all_tasks():
            if task is not current_task:
                task.cancel()

    runner.run_sync(cancel_all_tasks)
    with pytest.raises(asyncio.CancelledError):
        runner.stop()


@gen_test()
async def test_all_quiet_exceptions():
    class CustomError(Exception):
        pass

    async def throws(msg):
        raise CustomError(msg)

    with captured_logger("") as sio:
        with pytest.raises(CustomError):
            await All([throws("foo") for _ in range(5)])
        with pytest.raises(CustomError):
            await All([throws("bar") for _ in range(5)], quiet_exceptions=CustomError)

    assert "bar" not in sio.getvalue()
    assert "foo" in sio.getvalue()


def test_warn_on_duration():
    with warnings.catch_warnings(record=True) as record:
        with warn_on_duration("10s", "foo"):
            pass
    assert not record

    with pytest.warns(UserWarning, match=r"foo") as record:
        with warn_on_duration("1ms", "foo"):
            sleep(0.100)

    assert record

    with pytest.warns(UserWarning) as record:
        with warn_on_duration("1ms", "{duration:.4f}"):
            start = time()
            sleep(0.100)
            measured = time() - start

    assert record
    assert len(record) == 1
    assert float(str(record[0].message)) >= float(str(f"{measured:.4f}"))


def test_logs():
    log = Log("Hello")
    assert isinstance(log, str)
    d = Logs({"123": log, "456": Log("World!")})
    assert isinstance(d, dict)
    text = d._repr_html_()
    assert is_valid_xml("<div>" + text + "</div>")
    assert "Hello" in text
    assert "456" in text


def test_is_valid_xml():
    assert is_valid_xml("<a>foo</a>")
    with pytest.raises(xml.etree.ElementTree.ParseError):
        assert is_valid_xml("<a>foo")


def test_format_dashboard_link():
    with dask.config.set({"distributed.dashboard.link": "foo"}):
        assert format_dashboard_link("host", 1234) == "foo"

    assert "host" in format_dashboard_link("host", 1234)
    assert "1234" in format_dashboard_link("host", 1234)

    try:
        os.environ["host"] = "hello"
        assert "hello" not in format_dashboard_link("host", 1234)
    finally:
        del os.environ["host"]


def test_parse_ports():
    assert parse_ports(None) == [None]
    assert parse_ports(23) == [23]
    assert parse_ports("45") == [45]
    assert parse_ports("100:103") == [100, 101, 102, 103]
    assert parse_ports([100, 101, 102, 103]) == [100, 101, 102, 103]

    out = parse_ports((100, 101, 102, 103))
    assert out == [100, 101, 102, 103]
    assert isinstance(out, list)

    with pytest.raises(ValueError, match="port_stop must be greater than port_start"):
        parse_ports("103:100")
    with pytest.raises(TypeError):
        parse_ports(100.5)
    with pytest.raises(TypeError):
        parse_ports([100, 100.5])
    with pytest.raises(ValueError):
        parse_ports("foo")
    with pytest.raises(ValueError):
        parse_ports("100.5")


@gen_test()
async def test_run_in_executor_with_context():
    class MyExecutor(Executor):
        call_count = 0

        def submit(self, __fn, *args, **kwargs):
            self.call_count += 1
            f = Future()
            f.set_result(__fn(*args, **kwargs))
            return f

    ex = MyExecutor()
    out = await run_in_executor_with_context(ex, inc, 1)
    assert out == 2
    assert ex.call_count == 1


@gen_test()
async def test_run_in_executor_with_context_preserves_contextvars():
    var = contextvars.ContextVar("var")

    with ThreadPoolExecutor(2) as ex:

        async def set_var(v: str) -> None:
            var.set(v)
            r = await run_in_executor_with_context(ex, var.get)
            assert r == v

        await asyncio.gather(set_var("foo"), set_var("bar"))


@gen_test()
async def test_offload():
    assert (await offload(inc, 1)) == 2
    assert (await offload(lambda x, y: x + y, 1, y=2)) == 3


@gen_test()
async def test_offload_preserves_contextvars():
    var = contextvars.ContextVar("var")

    async def set_var(v: str) -> None:
        var.set(v)
        r = await offload(var.get)
        assert r == v

    await asyncio.gather(set_var("foo"), set_var("bar"))


def test_no_default_deprecated():
    with pytest.warns(FutureWarning, match="no_default is deprecated"):
        from distributed.utils import no_default
    assert no_default is dask.typing.no_default


def test_iscoroutinefunction_unhashable_input():
    # Ensure iscoroutinefunction can handle unhashable callables
    assert not iscoroutinefunction(_UnhashableCallable())


def test_iscoroutinefunction_nested_partial():
    async def my_async_callable(x, y, z):
        pass

    assert iscoroutinefunction(
        functools.partial(functools.partial(my_async_callable, 1), 2)
    )


def test_recursive_to_dict():
    class C:
        def __init__(self, x):
            self.x = x

        def __repr__(self):
            return "<C>"

        def _to_dict(self, *, exclude):
            assert exclude == ["foo"]
            return ["C:", recursive_to_dict(self.x, exclude=exclude)]

    class D:
        def __repr__(self):
            return "<D>"

    class E:
        def __init__(self):
            self.x = 1  # Public attribute; dump
            self._y = 2  # Private attribute; don't dump
            self.foo = 3  # In exclude; don't dump

        @property
        def z(self):  # Public property; dump
            return 4

        def f(self):  # Callable; don't dump
            return 5

        def _to_dict(self, *, exclude):
            # Output: {"x": 1, "z": 4}
            return recursive_to_dict(self, exclude=exclude, members=True)

    inp = [
        1,
        1.1,
        True,
        False,
        None,
        "foo",
        b"bar",
        C,
        C(1),
        D(),
        (1, 2),
        [3, 4],
        {5, 6},
        frozenset([7, 8]),
        deque([9, 10]),
        {3: 4, 1: 2}.keys(),
        {3: 4, 1: 2}.values(),
        E(),
    ]
    expect = [
        1,
        1.1,
        True,
        False,
        None,
        "foo",
        "b'bar'",
        "<class 'test_utils.test_recursive_to_dict.<locals>.C'>",
        ["C:", 1],
        "<D>",
        [1, 2],
        [3, 4],
        list({5, 6}),
        list(frozenset([7, 8])),
        [9, 10],
        [3, 1],
        [4, 2],
        {"x": 1, "z": 4},
    ]
    assert recursive_to_dict(inp, exclude=["foo"]) == expect

    # Test recursion
    a = []
    c = C(a)
    a += [c, c]
    # The blocklist of already-seen objects is reentrant: a is converted to string when
    # found inside itself; c must *not* be converted to string the second time it's
    # found, because it's outside of itself.
    assert recursive_to_dict(a, exclude=["foo"]) == [
        ["C:", "[<C>, <C>]"],
        ["C:", "[<C>, <C>]"],
    ]


def test_recursive_to_dict_no_nest():
    class Person:
        def __init__(self, name):
            self.name = name
            self.children = []
            self.pets = []
            ...

        def _to_dict_no_nest(self, exclude=()):
            return recursive_to_dict(self.__dict__, exclude=exclude)

        def __repr__(self):
            return self.name

    class Pet:
        def __init__(self, name):
            self.name = name
            self.owners = []
            ...

        def _to_dict_no_nest(self, exclude=()):
            return recursive_to_dict(self.__dict__, exclude=exclude)

        def __repr__(self):
            return self.name

    alice = Person("Alice")
    bob = Person("Bob")
    charlie = Pet("Charlie")
    alice.children.append(bob)
    alice.pets.append(charlie)
    bob.pets.append(charlie)
    charlie.owners[:] = [alice, bob]
    info = {"people": [alice, bob], "pets": [charlie]}
    expect = {
        "people": [
            {"name": "Alice", "children": ["Bob"], "pets": ["Charlie"]},
            {"name": "Bob", "children": [], "pets": ["Charlie"]},
        ],
        "pets": [
            {"name": "Charlie", "owners": ["Alice", "Bob"]},
        ],
    }
    assert recursive_to_dict(info) == expect


@gen_test()
async def test_log_errors():
    class CustomError(Exception):
        pass

    # Use the logger of the caller module
    with (
        captured_logger("test_utils") as caplog,
        captured_logger("distributed.utils") as caplog2,
    ):
        # Context manager
        with log_errors():
            pass

        with log_errors():
            with log_errors():
                pass

        with log_errors(pdb=True):
            pass

        with pytest.raises(CustomError):
            with log_errors():
                raise CustomError("err1")

        with pytest.raises(CustomError):
            with log_errors():
                with log_errors():
                    raise CustomError("err2")

        # Bare decorator
        @log_errors
        def _():
            return 123

        assert _() == 123
        with pytest.raises(TypeError):
            _(bad=1)

        @log_errors
        def _():
            raise CustomError("err3")

        with pytest.raises(CustomError):
            _()

        @log_errors
        def inner():
            raise CustomError("err4")

        @log_errors
        def outer():
            inner()

        with pytest.raises(CustomError):
            outer()

        # Decorator with parameters
        @log_errors()
        def _():
            return 456

        assert _() == 456

        @log_errors()
        def _():
            with log_errors():
                raise CustomError("err5")

        with pytest.raises(CustomError):
            _()

        @log_errors(pdb=True)
        def _():
            return 789

        assert _() == 789

        # Decorate async function
        @log_errors
        async def _():
            return 123

        assert await _() == 123

        task = asyncio.create_task(_())
        assert await task == 123

        @log_errors
        async def _():
            raise CustomError("err6")

        with pytest.raises(CustomError):
            await _()

        task = asyncio.create_task(_())
        with pytest.raises(CustomError):
            await task

        @log_errors()
        async def _():
            raise CustomError("err7")

        with pytest.raises(CustomError):
            await _()

        task = asyncio.create_task(_())
        with pytest.raises(CustomError):
            await task

        async def _():
            with log_errors():
                raise CustomError("err8")

        task = asyncio.create_task(_())
        with pytest.raises(CustomError):
            await task

    assert [row for row in caplog.getvalue().splitlines() if row.startswith("err")] == [
        "err1",
        "err2",
        "err2",
        "err3",
        "err4",
        "err4",
        "err5",
        "err5",
        "err6",
        "err6",
        "err7",
        "err7",
        "err8",
    ]
    assert "got an unexpected keyword argument 'bad'" in caplog2.getvalue()


@pytest.mark.parametrize("unroll_stack,logger_name", [(0, "test_utils"), (1, "a.b.c")])
def test_log_errors_unroll_stack(unroll_stack, logger_name, tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a").touch("__init__.py")
    (tmp_path / "a" / "b").mkdir()
    (tmp_path / "a" / "b").touch("__init__.py")
    with (tmp_path / "a" / "b" / "c.py").open("w") as fh:
        fh.write("def f():\n    raise ValueError('myerr')\n")

    sys.path.insert(0, str(tmp_path))
    import a.b.c

    del sys.path[0]

    with (
        captured_logger(logger_name) as caplog,
        pytest.raises(ValueError),
        log_errors(unroll_stack=unroll_stack),
    ):
        a.b.c.f()

    assert "myerr" in caplog.getvalue()


def test_load_json_robust_timeout(tmp_path):
    path = tmp_path / "foo.json"
    with pytest.raises(TimeoutError):
        json_load_robust(path, timeout=0.01)

    with ThreadPoolExecutor() as tpe:
        fut = tpe.submit(json_load_robust, path, timeout=30)
        import json

        with open(path, "w") as fd:
            json.dump({"foo": "bar"}, fd)

        assert fut.result() == {"foo": "bar"}

    assert json_load_robust(path) == {"foo": "bar"}


def test_rate_limiter_filter(caplog):
    logger = logging.getLogger("test_rate_limiter_filter")
    logger.addFilter(RateLimiterFilter(r"Hello .*", rate="10ms"))
    logger.warning("Hello Al!")  # Match
    logger.warning("Hello Bianca!")  # Match and <10ms
    logger.warning("Hello %s!", "Charlie")  # Match and <10ms, with args
    logger.warning("Goodbye Al!")  # No match
    sleep(0.02)
    logger.warning("Hello again!")  # Match and >10ms
    RateLimiterFilter.reset_timer("test_rate_limiter_filter")
    logger.warning("Hello once more!")  # Match and <10ms, but after calling clear()
    assert [record.msg for record in caplog.records] == [
        "Hello Al!",
        "Goodbye Al!",
        "Hello again!",
        "Hello once more!",
    ]


@pytest.mark.parametrize(
    "obj1,obj2,expected",
    [
        [(1, 2), (1, 2), False],
        [(1, 2), (1, 3), True],
        [1, 1, False],
        [1, 2, True],
        [None, 0, False],
        [None, (1, 2), True],
    ],
)
def test_tuple_comparable_lt(obj1, obj2, expected):
    assert (TupleComparable(obj1) < TupleComparable(obj2)) == expected


@pytest.mark.parametrize(
    "obj1,obj2,expected",
    [
        [(1, 2), (1, 2), True],
        [(1, 2), (1, 3), False],
        [1, 1, True],
        [1, 2, False],
        [None, 0, True],
        [None, (1, 2), False],
    ],
)
def test_tuple_comparable_eq(obj1, obj2, expected):
    assert (TupleComparable(obj1) == TupleComparable(obj2)) == expected


def test_tuple_comparable_error():
    with pytest.raises(ValueError):
        TupleComparable("string")


@gen_test()
async def test_wait_for():
    # See https://github.com/python/cpython/issues/86296#issuecomment-1281339374
    import asyncio

    from distributed.utils import wait_for

    async def left(tasks, event):
        me = asyncio.current_task()
        assert tasks[0] is me
        await asyncio.sleep(0.1)
        event.set()
        tasks[1].cancel()

    not_cancelled = False

    async def right(tasks, event, use_wait_for):
        nonlocal not_cancelled
        try:
            me = asyncio.current_task()
            assert tasks[1] is me
            if use_wait_for:
                await wait_for(event.wait(), 1)
            else:
                await event.wait()
            not_cancelled = True
        except asyncio.CancelledError:
            raise

    async def left_right(use_wait_for):
        tasks = []
        event = asyncio.Event()
        tasks.append(asyncio.create_task(left(tasks, event)))
        tasks.append(asyncio.create_task(right(tasks, event, use_wait_for)))
        await asyncio.wait(tasks)

    await left_right(False)
    await left_right(True)
    assert not not_cancelled

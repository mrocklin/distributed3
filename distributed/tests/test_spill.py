from __future__ import annotations

import array
import os
import random
import uuid
from pathlib import Path

import pytest

import dask.config
from dask.sizeof import sizeof

from distributed import profile
from distributed.compatibility import WINDOWS
from distributed.metrics import meter
from distributed.spill import SpillBuffer
from distributed.utils import RateLimiterFilter
from distributed.utils_test import captured_logger


def psize(tmp_path: Path, **objs: object) -> tuple[int, int]:
    # zict 2.2: tmp_path/key
    # zict >=3.0: tmp_path/key#0
    fnames = tmp_path.glob("*")
    key_to_fname = {fname.name.split("#")[0]: fname for fname in fnames}
    return (
        sum(sizeof(o) for o in objs.values()),
        sum(os.stat(key_to_fname[k]).st_size for k in objs),
    )


def assert_buf(
    buf: SpillBuffer, tmp_path: Path, expect_fast: dict, expect_slow: dict
) -> None:
    # assertions on fast
    assert dict(buf.fast) == expect_fast
    assert buf.fast.weights == {k: sizeof(v) for k, v in expect_fast.items()}
    assert buf.fast.total_weight == sum(sizeof(v) for v in expect_fast.values())
    for k, v in buf.fast.items():
        assert buf[k] is v

    # assertions on slow
    assert set(buf.slow) == set(expect_slow)
    slow = buf._slow_uncached
    assert slow.weight_by_key == {
        k: psize(tmp_path, **{k: v}) for k, v in expect_slow.items()
    }
    total_weight = psize(tmp_path, **expect_slow)
    assert slow.total_weight == total_weight
    assert buf.spilled_total == total_weight


def test_psize(tmp_path):
    buf = SpillBuffer(str(tmp_path), target=0)
    a = "a" * 100
    assert 100 < sizeof(a) < 200
    buf["a"] = a
    memory_size, disk_size = psize(tmp_path, a=a)
    assert memory_size != disk_size


def test_spillbuffer(tmp_path):
    buf = SpillBuffer(str(tmp_path), target=300)
    # Convenience aliases
    assert buf.memory is buf.fast
    assert buf.disk is buf.slow

    assert_buf(buf, tmp_path, {}, {})

    a, b, c, d = "a" * 100, "b" * 99, "c" * 98, "d" * 97

    # Test assumption made by this test, mostly for non CPython implementations
    assert 100 < sizeof(a) < 200

    buf["a"] = a
    assert_buf(buf, tmp_path, {"a": a}, {})
    assert buf["a"] == a

    buf["b"] = b
    assert_buf(buf, tmp_path, {"a": a, "b": b}, {})

    buf["c"] = c
    assert_buf(buf, tmp_path, {"b": b, "c": c}, {"a": a})

    assert buf["a"] == a
    assert_buf(buf, tmp_path, {"a": a, "c": c}, {"b": b})

    buf["d"] = d
    assert_buf(buf, tmp_path, {"a": a, "d": d}, {"b": b, "c": c})

    # Deleting an in-memory key does not automatically move spilled keys back to memory
    del buf["a"]
    assert_buf(buf, tmp_path, {"d": d}, {"b": b, "c": c})
    with pytest.raises(KeyError):
        buf["a"]

    # Deleting a spilled key updates the metadata
    del buf["b"]
    assert_buf(buf, tmp_path, {"d": d}, {"c": c})
    with pytest.raises(KeyError):
        buf["b"]

    # Updating a spilled key moves it to the top of the LRU and to memory
    c2 = c * 2
    buf["c"] = c2
    assert_buf(buf, tmp_path, {"c": c2}, {"d": d})

    # Single key is larger than target and goes directly into slow
    e = "e" * 500

    buf["e"] = e
    assert_buf(buf, tmp_path, {"c": c2}, {"d": d, "e": e})

    # Updating a spilled key with another larger than target updates slow directly
    d = "d" * 500
    buf["d"] = d
    assert_buf(buf, tmp_path, {"c": c2}, {"d": d, "e": e})


def test_disk_size_calculation(tmp_path):
    buf = SpillBuffer(str(tmp_path), target=0)
    a = "a" * 100
    b = array.array("d", (random.random() for _ in range(100)))
    buf["a"] = a
    buf["b"] = b
    assert_buf(buf, tmp_path, {}, {"a": a, "b": b})


def test_spillbuffer_maxlim(tmp_path_factory):
    buf_dir = tmp_path_factory.mktemp("buf")
    buf = SpillBuffer(str(buf_dir), target=200, max_spill=600)

    a, b, c, d, e = "a" * 200, "b" * 100, "c" * 99, "d" * 199, "e" * 98

    # size of a is bigger than target and is smaller than max_spill;
    # key should be in slow
    buf["a"] = a
    assert_buf(buf, buf_dir, {}, {"a": a})
    assert buf["a"] == a

    # size of b is smaller than target key should be in fast
    buf["b"] = b
    assert_buf(buf, buf_dir, {"b": b}, {"a": a})

    # size of c is smaller than target but b+c > target, c should stay in fast and b
    # move to slow since the max_spill limit has not been reached yet
    buf["c"] = c
    assert_buf(buf, buf_dir, {"c": c}, {"a": a, "b": b})

    # size of e < target but e+c > target, this will trigger movement of c to slow
    # but the max spill limit prevents it. Resulting in e remaining in fast

    with captured_logger("distributed.spill") as logs_e:
        buf["e"] = e

    assert "disk reached capacity" in logs_e.getvalue()
    assert_buf(buf, buf_dir, {"c": c, "e": e}, {"a": a, "b": b})

    # size of d > target, d should go to slow but slow reached the max_spill limit then
    # d will end up on fast with c (which can't be move to slow because it won't fit
    # either)
    RateLimiterFilter.reset_timer("distributed.spill")
    with captured_logger("distributed.spill") as logs_d:
        buf["d"] = d

    assert "disk reached capacity" in logs_d.getvalue()
    assert_buf(buf, buf_dir, {"c": c, "d": d, "e": e}, {"a": a, "b": b})

    # Overwrite a key that was in slow, but the size of the new key is larger than
    # max_spill

    a_large = "a" * 500

    # Assert precondition that a_large is larger than max_spill when written to disk
    unlimited_buf_dir = tmp_path_factory.mktemp("unlimited_buf")
    unlimited_buf = SpillBuffer(unlimited_buf_dir, target=0)
    unlimited_buf["a_large"] = a_large
    assert psize(unlimited_buf_dir, a_large=a_large)[1] > 600

    RateLimiterFilter.reset_timer("distributed.spill")
    with captured_logger("distributed.spill") as logs_alarge:
        buf["a"] = a_large

    assert "disk reached capacity" in logs_alarge.getvalue()
    assert_buf(buf, buf_dir, {"a": a_large, "d": d, "e": e}, {"b": b, "c": c})

    # Overwrite a key that was in fast, but the size of the new key is larger than
    # max_spill

    d_large = "d" * 501
    RateLimiterFilter.reset_timer("distributed.spill")
    with captured_logger("distributed.spill") as logs_dlarge:
        buf["d"] = d_large

    assert "disk reached capacity" in logs_dlarge.getvalue()
    assert_buf(buf, buf_dir, {"a": a_large, "d": d_large, "e": e}, {"b": b, "c": c})


class MyError(Exception):
    pass


class Bad:
    def __init__(self, size):
        self.size = size

    def __getstate__(self):
        raise MyError()

    def __sizeof__(self):
        return self.size


def test_spillbuffer_fail_to_serialize(tmp_path):
    buf = SpillBuffer(str(tmp_path), target=200, max_spill=600)

    # bad data individually larger than spill threshold target 200
    a = Bad(size=201)

    # Exception caught in the worker
    with pytest.raises(TypeError, match="Failed to pickle 'a'") as e:
        with captured_logger("distributed.spill") as logs_bad_key:
            buf["a"] = a
    assert isinstance(e.value.__cause__.__cause__.__cause__, MyError)

    # spill.py must remain silent because we're already logging in worker.py
    assert not logs_bad_key.getvalue()
    assert_buf(buf, tmp_path, {}, {})

    b = Bad(size=100)  # this is small enough to fit in memory/fast

    buf["b"] = b
    assert_buf(buf, tmp_path, {"b": b}, {})

    c = "c" * 100
    with captured_logger("distributed.spill") as logs_bad_key_mem:
        # This will go to fast and try to kick b out,
        # but keep b in fast since it's not pickable
        buf["c"] = c

    # worker.py won't intercept the exception here, so spill.py must dump the traceback
    logs_value = logs_bad_key_mem.getvalue()
    assert "Failed to pickle 'b'" in logs_value  # from distributed.spill
    assert "Traceback" in logs_value  # from distributed.spill
    assert_buf(buf, tmp_path, {"b": b, "c": c}, {})


@pytest.mark.skipif(WINDOWS, reason="Needs chmod")
def test_spillbuffer_oserror(tmp_path):
    buf = SpillBuffer(str(tmp_path), target=200, max_spill=800)

    a, b, c, d = (
        "a" * 200,
        "b" * 100,
        "c" * 201,
        "d" * 101,
    )

    # let's have something in fast and something in slow
    buf["a"] = a
    buf["b"] = b
    assert_buf(buf, tmp_path, {"b": b}, {"a": a})

    # modify permissions of disk to be read only.
    # This causes writes to raise OSError, just like in case of disk full.
    os.chmod(tmp_path, 0o555)

    # Add key > than target
    with captured_logger("distributed.spill") as logs_oserror_slow:
        buf["c"] = c

    assert "Spill to disk failed" in logs_oserror_slow.getvalue()
    assert_buf(buf, tmp_path, {"b": b, "c": c}, {"a": a})

    del buf["c"]
    assert_buf(buf, tmp_path, {"b": b}, {"a": a})

    # add key to fast which is smaller than target but when added it triggers spill,
    # which triggers OSError
    RateLimiterFilter.reset_timer("distributed.spill")
    with captured_logger("distributed.spill") as logs_oserror_evict:
        buf["d"] = d

    assert "Spill to disk failed" in logs_oserror_evict.getvalue()
    assert_buf(buf, tmp_path, {"b": b, "d": d}, {"a": a})


def test_spillbuffer_evict(tmp_path):
    buf = SpillBuffer(str(tmp_path), target=300)

    bad = Bad(size=100)
    a = "a" * 100

    buf["a"] = a
    assert_buf(buf, tmp_path, {"a": a}, {})

    # successful eviction
    weight = buf.evict()
    assert weight == sizeof(a)
    assert_buf(buf, tmp_path, {}, {"a": a})

    buf["bad"] = bad
    assert_buf(buf, tmp_path, {"bad": bad}, {"a": a})

    # unsuccessful eviction
    with captured_logger("distributed.spill") as logs_evict_key:
        weight = buf.evict()
    assert weight == -1

    assert "Failed to pickle" in logs_evict_key.getvalue()
    # bad keys stays in fast
    assert_buf(buf, tmp_path, {"bad": bad}, {"a": a})


def test_no_pop(tmp_path):
    buf = SpillBuffer(str(tmp_path), target=100)
    with pytest.raises(NotImplementedError):
        buf.pop("x", None)


class NoWeakRef:
    """A class which
    1. reports an arbitrary managed memory usage
    2. does not support being targeted by weakref.ref()
    3. has a property `id` which changes every time it is unpickled
    """

    __slots__ = ("size", "id")

    def __init__(self, size):
        self.size = size
        self.id = uuid.uuid4()

    def __sizeof__(self):
        return self.size

    def __reduce__(self):
        return (type(self), (self.size,))


class SupportsWeakRef(NoWeakRef):
    __slots__ = ("__weakref__",)


@pytest.mark.parametrize(
    "cls,expect_cached",
    [(SupportsWeakRef, True), (NoWeakRef, False)],
)
@pytest.mark.parametrize("size", [60, 110])
def test_weakref_cache(tmp_path, cls, expect_cached, size):
    buf = SpillBuffer(str(tmp_path), target=100)

    # Run this test twice:
    # - x is smaller than target and is evicted by y;
    # - x is individually larger than target and it never touches fast
    x = cls(size)
    buf["x"] = x
    if size < 100:
        buf["y"] = cls(60)  # spill x
    assert "x" in buf.slow

    # Test that we update the weakref cache on setitem
    assert (buf["x"] is x) == expect_cached

    # Do not use id_x = id(x), as in CPython id's are C memory addresses and are reused
    # by PyMalloc when you descope objects, so a brand new object might end up having
    # the same id as a deleted one
    id_x = x.id
    del x

    # Ensure that the profiler has stopped and released all references to x so that it can be garbage-collected
    with profile.lock:
        pass

    if size < 100:
        buf["y"]
    assert "x" in buf.slow

    x2 = buf["x"]
    assert x2.id != id_x
    if size < 100:
        buf["y"]
    assert "x" in buf.slow

    # Test that we update the weakref cache on getitem
    assert (buf["x"] is x2) == expect_cached


def test_metrics(tmp_path):
    buf = SpillBuffer(str(tmp_path), target=35_000)
    assert buf.cumulative_metrics == {}

    a = "a" * 20_000  # <target, highly compressible
    b = "b"
    c = random.randbytes(30_000)  # <target, uncompressible
    d = "d" * 40_000  # >target, highly compressible

    with meter() as m:
        buf["a"] = a
        buf["b"] = b
        del buf["b"]  # No effect
        _ = buf["a"]  # Cache hit
        buf["c"] = c  # Auto evict a (c < target; pushes older keys to slow)
        buf.evict()  # Manual evict c
        buf["d"] = d  # Auto evict d (d > target; goes to slow)

        a_size = psize(tmp_path, a=a)
        c_size = psize(tmp_path, c=c)
        d_size = psize(tmp_path, d=d)

        _ = buf["c"]  # Unspill / cache miss (c < target, goes back to fast)
        _ = buf["d"]  # Unspill / cache miss (d > target, stays in slow)

    assert (set(buf.fast), set(buf.slow)) == ({"c"}, {"a", "d"})

    assert {
        (label, unit): v
        for (label, unit), v in buf.cumulative_metrics.items()
        if unit != "seconds"
    } == {
        ("disk-read", "bytes"): c_size[1] + d_size[1],
        ("disk-read", "count"): 2,
        ("disk-write", "bytes"): a_size[1] + c_size[1] + d_size[1],
        ("disk-write", "count"): 3,
        ("memory-read", "bytes"): a_size[0],
        ("memory-read", "count"): 1,
    }

    time_metrics = {
        (label, unit): v
        for (label, unit), v in buf.cumulative_metrics.items()
        if unit == "seconds"
    }
    assert all(v > 0 for v in time_metrics.values())
    assert list(time_metrics) == [
        ("serialize", "seconds"),
        ("compress", "seconds"),
        ("disk-write", "seconds"),
        ("disk-read", "seconds"),
        ("decompress", "seconds"),
        ("deserialize", "seconds"),
    ]
    if not WINDOWS:  # Fiddly rounding; see distributed.metrics._WindowsTime
        assert sum(time_metrics.values()) <= m.delta


@pytest.mark.parametrize(
    "compression,minsize,maxsize",
    [("zlib", 100, 500), (None, 20_000, 20_500)],
)
def test_compression_settings(tmp_path, compression, minsize, maxsize):
    with dask.config.set({"distributed.worker.memory.spill-compression": compression}):
        buf = SpillBuffer(str(tmp_path), target=1)
        x = "x" * 20_000
        buf["x"] = x
        assert minsize <= psize(tmp_path, x=x)[1] <= maxsize


def test_str_collision(tmp_path):
    """keys are converted to strings before landing on disk. In dask, 1 and "1" are two
    different keys; make sure they don't collide.
    """
    buf = SpillBuffer(str(tmp_path), target=100_000)
    buf[1] = 10
    buf["1"] = 20
    assert buf.keys() == {1, "1"}
    assert dict(buf) == {1: 10, "1": 20}
    assert not buf.slow
    buf.evict()
    buf.evict()
    assert not buf.fast
    assert buf.keys() == {1, "1"}
    assert dict(buf) == {1: 10, "1": 20}

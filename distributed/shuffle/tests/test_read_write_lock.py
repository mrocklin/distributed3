from __future__ import annotations

import threading

import pytest

from distributed.shuffle._storage import ReadWriteLock


def read(
    lock: ReadWriteLock, in_event: threading.Event, block_event: threading.Event
) -> None:
    with lock.read():
        in_event.set()
        block_event.wait()


def write(
    lock: ReadWriteLock, in_event: threading.Event, block_event: threading.Event
) -> None:
    with lock.write():
        in_event.set()
        block_event.wait()


def test_basic():
    lock = ReadWriteLock()

    with lock.write():
        pass

    # Write after write
    with lock.write():
        pass

    # Read after write
    with lock.read():
        pass

    # Write after read
    with lock.write():
        pass

    with pytest.raises(RuntimeError, match="unlocked read lock"):
        lock.release_read()

    with pytest.raises(RuntimeError, match="unlocked write lock"):
        lock.release_write()


def test_lock_timeout():
    lock = ReadWriteLock()

    result = lock.acquire_read()
    assert result is True

    result = lock.acquire_write(timeout=0.1)
    assert result is False

    # Timed out write does not prevent read
    with lock.read():
        pass

    lock.release_read()
    result = lock.acquire_write(timeout=0.1)
    assert result is True

    result = lock.acquire_read(timeout=0.1)
    assert result is False

    lock.release_write()
    result = lock.acquire_read(timeout=0.1)
    assert result is True


def test_concurrent_reads_are_allowed():
    in_event = threading.Event()
    block_event = threading.Event()
    lock = ReadWriteLock()

    other = threading.Thread(target=read, args=(lock, in_event, block_event))
    try:
        other.start()
        in_event.wait()
        with lock.read():
            pass
        block_event.set()
    finally:
        block_event.set()
        other.join()


def test_read_blocks_write():
    in_event = threading.Event()
    block_event = threading.Event()
    lock = ReadWriteLock()

    other = threading.Thread(target=read, args=(lock, in_event, block_event))
    try:
        other.start()
        in_event.wait()
        result = lock.acquire_write(timeout=0.1)
        assert result is False

        block_event.set()
        with lock.write():
            pass
    finally:
        block_event.set()
        other.join()


def test_write_blocks_read():
    in_event = threading.Event()
    block_event = threading.Event()
    lock = ReadWriteLock()

    other = threading.Thread(target=write, args=(lock, in_event, block_event))
    try:
        other.start()
        in_event.wait()
        result = lock.acquire_read(timeout=0.1)
        assert result is False

        block_event.set()
        with lock.read():
            pass
    finally:
        block_event.set()
        other.join()


def test_write_blocks_write():
    in_event = threading.Event()
    block_event = threading.Event()
    lock = ReadWriteLock()

    other = threading.Thread(target=write, args=(lock, in_event, block_event))
    try:
        other.start()
        in_event.wait()
        result = lock.acquire_write(timeout=0.1)
        assert result is False

        block_event.set()
        with lock.write():
            pass
    finally:
        block_event.set()
        other.join()


def test_write_preferred_over_read():
    in_read_event = threading.Event()
    block_read_event = threading.Event()
    in_other_read_event = threading.Event()
    block_other_read_event = threading.Event()
    in_write_event = threading.Event()
    block_write_event = threading.Event()

    lock = ReadWriteLock()

    reader = threading.Thread(target=read, args=(lock, in_read_event, block_read_event))
    other_reader = threading.Thread(
        target=read, args=(lock, in_other_read_event, block_other_read_event)
    )
    writer = threading.Thread(
        target=write, args=(lock, in_write_event, block_write_event)
    )

    try:
        reader.start()
        in_read_event.wait()

        writer.start()
        result = in_write_event.wait(0.1)
        # The write is blocked by the read lock
        assert result is False
        assert lock._write_pending is True

        other_reader.start()
        result = in_other_read_event.wait(0.1)
        # The read is blocked by the pending write
        assert result is False

        block_read_event.set()
        reader.join()

        # The write happens next
        in_write_event.wait()
        assert in_other_read_event.is_set() is False

        block_write_event.set()
        writer.join()

        in_other_read_event.wait()
        block_other_read_event.set()
        other_reader.join()
    finally:
        block_read_event.set()
        block_write_event.set()
        block_other_read_event.set()
        reader.join()
        writer.join()
        other_reader.join()

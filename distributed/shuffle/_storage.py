from __future__ import annotations

import asyncio
import contextlib
import pathlib
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable

from distributed.shuffle._buffer import AsyncShardsBuffer
from distributed.shuffle._disk import ReadWriteLock
from distributed.shuffle._limiter import ResourceLimiter
from distributed.utils import log_errors

if TYPE_CHECKING:
    import pyarrow as pa


class StorageBuffer(AsyncShardsBuffer):
    """Accept, buffer, and write many small objects to many files

    This takes in lots of small objects, writes them to a local directory, and
    then reads them back when all writes are complete.  It buffers these
    objects in memory so that it can optimize disk access for larger writes.

    **State**

    -   shards: dict[str, list[bytes]]

        This is our in-memory buffer of data waiting to be written to files.

    -   sizes: dict[str, int]

        The size of each list of shards.  We find the largest and write data from that buffer

    Parameters
    ----------
    directory : str or pathlib.Path
        Where to write and read data.  Ideally points to fast disk.
    memory_limiter : ResourceLimiter
        Limiter for in-memory buffering (at most this much data)
        before writes to disk occur. If the incoming data that has yet
        to be processed exceeds this limit, then the buffer will block
        until below the threshold. See :meth:`.write` for the
        implementation of this scheme.
    """

    def __init__(
        self,
        directory: str | pathlib.Path,
        write: Callable[[Any, pathlib.Path], int],
        read: Callable[[pathlib.Path], tuple[Any, int]],
        executor: ThreadPoolExecutor,
        memory_limiter: ResourceLimiter,
    ):
        super().__init__(
            memory_limiter=memory_limiter,
            # Disk is not able to run concurrently atm
        )
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(exist_ok=True)
        self._write_fn = write
        self._read = read
        self._directory_lock = ReadWriteLock()
        self._executor = executor

    async def _write(self, id: str, shards: list[pa.Table]) -> int:
        """Write one buffer to file

        This function was built to offload the disk IO, but since then we've
        decided to keep this within the event loop (disk bandwidth should be
        prioritized, and writes are typically small enough to not be a big
        deal).

        Most of the logic here is about possibly going back to a separate
        thread, or about diagnostics.  If things don't change much in the
        future then we should consider simplifying this considerably and
        dropping the write into communicate above.
        """

        with log_errors():
            # Consider boosting total_size a bit here to account for duplication
            with self.time("write"):
                # We only need shared (i.e., read) access to the directory to write
                # to a file inside of it.
                with self._directory_lock.read():
                    return await asyncio.get_running_loop().run_in_executor(
                        self._executor,
                        self._write_fn,
                        shards,
                        (self.directory / str(id)).resolve(),
                    )

    def read(self, id: str) -> Any:
        """Read a complete file back into memory"""
        if self._state == "erred":
            assert self._exception
            raise self._exception

        if not self._state == "flushed":
            raise RuntimeError(f"Tried to read from a {self._state} buffer.")

        try:
            with self.time("read"):
                with self._directory_lock.read():
                    if self._state != "flushed":
                        raise RuntimeError("Can't read")
                    data, bytes_read = self._read((self.directory / str(id)).resolve())
                    self.bytes_read += bytes_read
        except FileNotFoundError:
            data = []
        data += self.shards.get(id, [])
        bytes_memory = self.sizes.get(id, 0)
        self.bytes_memory -= bytes_memory
        if data:
            return data
        else:
            raise KeyError(id)

    async def close(self) -> None:
        with self._directory_lock.write():
            await super().close()
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(self.directory)

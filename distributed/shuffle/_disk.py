from __future__ import annotations

import contextlib
import pathlib
import shutil

from distributed.shuffle._buffer import ShardsBuffer
from distributed.shuffle._limiter import ResourceLimiter
from distributed.utils import log_errors


class DiskShardsBuffer(ShardsBuffer):
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
    memory_limiter : ResourceLimiter, optional
        Limiter for in-memory buffering (at most this much data)
        before writes to disk occur. If the incoming data that has yet
        to be processed exceeds this limit, then the buffer will block
        until below the threshold. See :meth:`.write` for the
        implementation of this scheme.
    """

    def __init__(
        self,
        directory: str | pathlib.Path,
        memory_limiter: ResourceLimiter | None = None,
        min_message_size=-1,
    ):
        super().__init__(
            memory_limiter=memory_limiter,
            # Disk is not able to run concurrently atm
            concurrency_limit=1,
            min_message_size=min_message_size,
        )
        self.directory = pathlib.Path(directory)
        self.directory.mkdir(exist_ok=True)

    async def _process(self, id: str, shards: list[bytes]) -> None:
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
                with open(
                    self.directory / str(id), mode="ab", buffering=100_000_000
                ) as f:
                    for shard in shards:
                        f.write(shard)

    def read(self, id: int | str) -> bytes:
        """Read a complete file back into memory"""
        self.raise_on_exception()
        # if not self._inputs_done:
        #     raise RuntimeError("Tried to read from file before done.")
        data = []
        if id in self.shards:
            data = self.shards.pop(id)  # type: ignore
            in_memory_size = self.sizes.pop(id)
            if self.memory_limiter:
                self.memory_limiter.decrease_nonblocking(in_memory_size)
            self.bytes_memory -= in_memory_size
        try:
            with self.time("read"):
                with open(
                    self.directory / str(id), mode="rb", buffering=100_000_000
                ) as f:
                    data_disk = f.read()
                    size = f.tell()
                    data.insert(0, data_disk)
                    self.bytes_read += size
        except FileNotFoundError:
            pass

        if data:
            return b"".join(data)
        else:
            raise KeyError(id)

    async def close(self) -> None:
        await super().close()
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(self.directory)

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from dask.utils import parse_bytes

from distributed.shuffle._disk import ShardsBuffer
from distributed.shuffle._limiter import ResourceLimiter
from distributed.utils import log_errors


class CommShardsBuffer(ShardsBuffer):
    """Accept, buffer, and send many small messages to many workers

    This takes in lots of small messages destined for remote workers, buffers
    those messages in memory, and then sends out batches of them when possible
    to different workers.  This tries to send larger messages when possible,
    while also respecting a memory bound

    **State**

    -   shards: dict[str, list[ShardType]]

        This is our in-memory buffer of data waiting to be sent to other workers.

    -   sizes: dict[str, int]

        The size of each list of shards.  We find the largest and send data from that buffer

    State
    -----
    max_message_size: int
        The maximum size in bytes of a single message that we want to send

    Parameters
    ----------
    send : callable
        How to send a list of shards to a worker
        Expects an address of the target worker (string)
        and a payload of shards (list of bytes) to send to that worker
    memory_limiter : ResourceLimiter, optional
        Limiter for memory usage (in bytes), or None if no limiting
        should be applied. If the incoming data that has yet to be
        processed exceeds this limit, then the buffer will block until
        below the threshold. See :meth:`.write` for the implementation
        of this scheme.
    concurrency_limit : int
        Number of background tasks to run.
    """

    max_message_size = parse_bytes("2 MiB")

    def __init__(
        self,
        send: Callable[[str, list[tuple[Any, bytes]]], Awaitable[None]],
        memory_limiter: ResourceLimiter | None = None,
        concurrency_limit: int = 10,
    ):
        import dask

        super().__init__(
            memory_limiter=memory_limiter,
            concurrency_limit=concurrency_limit,
            max_message_size=parse_bytes(
                dask.config.get("shuffle.comm.max_message_size", default="2 MiB")
            ),
        )
        self.send = send

    async def _process(self, address: str, shards: list[tuple[Any, bytes]]) -> None:
        """Send one message off to a neighboring worker"""
        with log_errors():
            # Consider boosting total_size a bit here to account for duplication
            with self.time("send"):
                await self.send(address, shards)

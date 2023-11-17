from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from distributed.shuffle._buffer import BaseBuffer
from distributed.shuffle._limiter import ResourceLimiter
from distributed.utils import log_errors


class CommShardsBuffer(BaseBuffer):
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
    memory_limiter : ResourceLimiter
        Limiter for memory usage (in bytes). If the incoming data that
        has yet to be processed exceeds this limit, then the buffer will
        block until below the threshold. See :meth:`.write` for the
        implementation of this scheme.
    concurrency_limit : int
        Number of background tasks to run.
    """

    drain = True

    def __init__(
        self,
        send: Callable[[str, list[tuple[Any, Any]]], Awaitable[None]],
        memory_limiter: ResourceLimiter,
        message_bytes_limit: int,
        concurrency_limit: int = 10,
    ):
        super().__init__(
            memory_limiter=memory_limiter,
            concurrency_limit=concurrency_limit,
            max_message_size=message_bytes_limit,
        )
        self.send = send

    @log_errors
    async def _flush(self, id: str, shards: list[Any]) -> int | None:  # type: ignore[return]
        """Send one message off to a neighboring worker"""
        # Consider boosting total_size a bit here to account for duplication
        with self.time("send"):
            await self.send(id, shards)

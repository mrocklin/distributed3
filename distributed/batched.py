import asyncio
import logging
from collections import deque

import dask
from dask.utils import parse_timedelta

from distributed.core import CommClosedError
from distributed.metrics import time

logger = logging.getLogger(__name__)


class BatchedSend:
    """Batch messages in batches on a stream

    This takes an IOStream and an interval (in ms) and ensures that we send no
    more than one message every interval milliseconds.  We send lists of
    messages.

    Batching several messages at once helps performance when sending
    a myriad of tiny messages.

    Examples
    --------
    >>> stream = await connect(address)
    >>> bstream = BatchedSend(interval='10 ms')
    >>> bstream.start(stream)
    >>> bstream.send('Hello,')
    >>> bstream.send('world!')

    On the other side, the recipient will get a message like the following::

        ['Hello,', 'world!']
    """

    def __init__(self, interval, serializers=None, name=None):
        self.interval = parse_timedelta(interval, default="ms")
        self.waker = asyncio.Event()
        self.please_stop = False
        self.buffer = []
        self.comm = None
        self.name = name
        self.message_count = 0
        self.batch_count = 0
        self.byte_count = 0
        self.next_deadline = None
        self.recent_message_log = deque(
            maxlen=dask.config.get("distributed.comm.recent-messages-log-length")
        )
        self.serializers = serializers
        self._background_task = None

    def start(self, comm):
        # A `BatchedSend` instance can be closed and restarted multiple times with new `comm` objects.
        # However, calling `start` on an already-running `BatchedSend` is an error.
        if self._background_task and not self._background_task.done():
            raise RuntimeError(f"Background task still running for {self!r}")
        self.please_stop = False
        self.waker.set()
        self.next_deadline = None
        self.comm = comm

        self._background_task = asyncio.create_task(
            self._background_send(),
            name=f"background-send-{self.name}",
        )

    def closed(self):
        return (self.comm is None or self.comm.closed()) and (
            self._background_task is None or self._background_task.done()
        )

    def __repr__(self):
        if self.closed():
            return f"<BatchedSend {self.name!r}: closed>"
        else:
            return f"<BatchedSend {self.name!r}: {len(self.buffer)} in buffer>"

    async def _background_send(self):
        while not self.please_stop:
            try:
                timeout = None
                if self.next_deadline:
                    timeout = self.next_deadline - time()
                await asyncio.wait_for(self.waker.wait(), timeout=timeout)
                self.waker.clear()
            except asyncio.TimeoutError:
                pass
            if not self.buffer:
                # Nothing to send
                self.next_deadline = None
                continue
            if self.next_deadline is not None and time() < self.next_deadline:
                # Send interval not expired yet
                continue
            payload, self.buffer = self.buffer, []
            self.batch_count += 1
            self.next_deadline = time() + self.interval

            try:
                nbytes = await self.comm.write(
                    payload, serializers=self.serializers, on_error="raise"
                )
                if nbytes < 1e6:
                    self.recent_message_log.append(payload)
                else:
                    self.recent_message_log.append("large-message")
                self.byte_count += nbytes
            except CommClosedError:
                logger.info(
                    f"Batched Comm Closed {self.comm!r} in {self!r}. Lost {len(payload)} messages, ",
                    f"plus {len(self.buffer)} in buffer.",  # <-- due to upcoming `abort()`
                    exc_info=True,
                )
                break
            except Exception:
                # We cannot safely retry self.comm.write, as we have no idea
                # what (if anything) was actually written to the underlying stream.
                # Re-writing messages could result in complete garbage (e.g. if a frame
                # header has been written, but not the frame payload), therefore
                # the only safe thing to do here is to abort the stream without
                # any attempt to re-try `write`.
                logger.exception(
                    f"Error in batched write in {self!r}. Lost {len(payload)} messages, "
                    f"plus {len(self.buffer)} in buffer."
                )
                break
            finally:
                payload = None  # lose ref
        else:
            # nobreak. We've been gracefully closed.
            return

        # If we've reached here, it means `break` was hit above and
        # there was an exception when using `comm`.
        # We can't close gracefully via `.close()` since we can't send messages.
        # So we just abort.
        # This means that any messages in our buffer our lost.
        # The exception will not be propagated.
        self.abort()

    def send(self, *msgs: dict) -> None:
        """Schedule a message for sending to the other side

        This completes quickly and synchronously. (However, note that like all
        `BatchedSend` methods, `send` is not threadsafe.)

        Message delivery is *not* gauranteed. There is no way for callers to know when,
        or whether, a particular message was received by the other side. When the
        underlying comm closes, any currently-buffered messages (as well as data in the
        socket's underlying buffer) will be lost.

        `send` will never raise an error, even if the `BatchedSend` or underlying comm
        is in a closed state.

        While `closed` is True, all calls to `send` will be buffered until the next call
        to `start`. However, calls to `send` made after the underlying comm has closed,
        but before ``await close()`` has returned, may or may not be dropped.

        Because `BatchedSend` will drop messages when the comm closes, users of
        `BatchedSend` are expected to be implementing their own reconnection logic,
        triggered when the comm closes. Reconnection often involves application logic
        reconciling state, then calling `start` again with a new comm object.
        """

        self.message_count += len(msgs)
        self.buffer.extend(msgs)
        # Avoid spurious wakeups if possible

        if self.comm and not self.comm.closed() and self.next_deadline is None:
            self.waker.set()

    async def close(self):
        """Flush existing messages and then close comm"""
        self.please_stop = True
        self.waker.set()

        if self._background_task:
            await self._background_task
            self._background_task = None

        if self.comm and not self.comm.closed():
            try:
                if self.buffer:
                    self.buffer, payload = [], self.buffer
                    await self.comm.write(
                        payload, serializers=self.serializers, on_error="raise"
                    )
            except CommClosedError:
                pass
            await self.comm.close()

    def abort(self):
        self.please_stop = True
        self.buffer = []
        self.waker.set()
        if self.comm and not self.comm.closed():
            self.comm.abort()

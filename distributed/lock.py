from __future__ import annotations

import asyncio
import logging
import uuid
from collections import defaultdict, deque

from dask.utils import parse_timedelta

from distributed.utils import TimeoutError, log_errors, wait_for
from distributed.worker import get_client

logger = logging.getLogger(__name__)


class LockExtension:
    """An extension for the scheduler to manage Locks

    This adds the following routes to the scheduler

    *  lock_acquire
    *  lock_release
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.events = defaultdict(deque)
        self.ids = dict()

        self.scheduler.server.handlers.update(
            {"lock_acquire": self.acquire, "lock_release": self.release}
        )

    @log_errors
    async def acquire(self, name=None, id=None, timeout=None):
        if isinstance(name, list):
            name = tuple(name)
        if name not in self.ids:
            result = True
        else:
            while name in self.ids:
                event = asyncio.Event()
                self.events[name].append(event)
                future = event.wait()
                if timeout is not None:
                    future = wait_for(future, timeout)
                try:
                    await future
                except TimeoutError:
                    result = False
                    break
                else:
                    result = True
                finally:
                    event2 = self.events[name].popleft()
                    assert event is event2
        if result:
            assert name not in self.ids
            self.ids[name] = id
        return result

    @log_errors
    def release(self, name=None, id=None):
        if isinstance(name, list):
            name = tuple(name)
        if self.ids.get(name) != id:
            raise ValueError("This lock has not yet been acquired")
        del self.ids[name]
        if self.events[name]:
            self.scheduler.loop.add_callback(self.events[name][0].set)
        else:
            del self.events[name]


class Lock:
    """Distributed Centralized Lock

    Parameters
    ----------
    name: string (optional)
        Name of the lock to acquire.  Choosing the same name allows two
        disconnected processes to coordinate a lock.  If not given, a random
        name will be generated.
    client: Client (optional)
        Client to use for communication with the scheduler.  If not given, the
        default global client will be used.

    Examples
    --------
    >>> lock = Lock('x')  # doctest: +SKIP
    >>> lock.acquire(timeout=1)  # doctest: +SKIP
    >>> # do things with protected resource
    >>> lock.release()  # doctest: +SKIP
    """

    def __init__(self, name=None, client=None):
        self._client = client
        self.name = name or "lock-" + uuid.uuid4().hex
        self.id = uuid.uuid4().hex
        self._locked = False

    @property
    def client(self):
        if not self._client:
            try:
                self._client = get_client()
            except ValueError:
                pass
        return self._client

    def _verify_running(self):
        if not self.client:
            raise RuntimeError(
                f"{type(self)} object not properly initialized. This can happen"
                " if the object is being deserialized outside of the context of"
                " a Client or Worker."
            )

    def acquire(self, blocking=True, timeout=None):
        """Acquire the lock

        Parameters
        ----------
        blocking : bool, optional
            If false, don't wait on the lock in the scheduler at all.
        timeout : string or number or timedelta, optional
            Seconds to wait on the lock in the scheduler.  This does not
            include local coroutine time, network transfer time, etc..
            It is forbidden to specify a timeout when blocking is false.
            Instead of number of seconds, it is also possible to specify
            a timedelta in string format, e.g. "200ms".

        Examples
        --------
        >>> lock = Lock('x')  # doctest: +SKIP
        >>> lock.acquire(timeout="1s")  # doctest: +SKIP

        Returns
        -------
        True or False whether or not it successfully acquired the lock
        """
        self._verify_running()
        timeout = parse_timedelta(timeout)

        if not blocking:
            if timeout is not None:
                raise ValueError("can't specify a timeout for a non-blocking call")
            timeout = 0

        result = self.client.sync(
            self.client.scheduler.lock_acquire,
            name=self.name,
            id=self.id,
            timeout=timeout,
        )
        self._locked = True
        return result

    def release(self):
        """Release the lock if already acquired"""
        self._verify_running()
        if not self.locked():
            raise ValueError("Lock is not yet acquired")
        result = self.client.sync(
            self.client.scheduler.lock_release, name=self.name, id=self.id
        )
        self._locked = False
        return result

    def locked(self):
        return self._locked

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.release()

    def __reduce__(self):
        return (Lock, (self.name,))

import asyncio
from collections import defaultdict
from contextlib import suppress
import logging
import uuid

from .client import Client
from .utils import log_errors, TimeoutError
from .worker import get_worker

logger = logging.getLogger(__name__)


class EventExtension:
    """ An extension for the scheduler to manage Events

    This adds the following routes to the scheduler

    *  event_wait
    *  event_set
    *  event_clear
    *  event_is_set

    In principle, the implementation logic is quite simple
    as we can reuse the asyncio.Event as much as possible:
    we keep a mapping from name to an asyncio.Event and call
    every function (wait, set, clear, is_set) directly on these
    events.

    However, this would cause a memory leak: created events in the
    dictionary are never removed.
    For this, we also keep a counter for the number of waiters on
    a specific event.
    If an event is set, we need to keep track of this state so
    we can not remove it (the default flag is false).
    If it is unset but there are waiters, we can also not remove
    it, as those waiters would then have dangling futures.
    Therefore the only time we can remove the event from our dict
    is when the number of waiters is 0 and the event flag is cleared.
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler
        # Keep track of all current events, identified by their name
        self._events = dict()
        # Keep track on how many waiters are present, so we know when
        # we can remove the event
        self._waiter_count = defaultdict(int)

        self.scheduler.handlers.update({
            "event_wait": self.event_wait,
            "event_set": self.event_set,
            "event_clear": self.event_clear,
            "event_is_set": self.event_is_set
        })

        self.scheduler.extensions["events"] = self

    async def event_wait(self, stream=None, name=None, timeout=None):
        """ Wait until the event is set to true.
        Returns false, when this did not happen in the given time
        and true otherwise.
        """
        with log_errors():
            name = self._normalize_name(name)

            event = self._get_or_create_event(name)
            future = event.wait()
            if timeout is not None:
                future = asyncio.wait_for(future, timeout)

            self._waiter_count[name] += 1
            try:
                await future
            except TimeoutError:
                return False
            finally:
                self._waiter_count[name] -= 1

            return True

    async def event_set(self, stream=None, name=None):
        """ Set the event with the given name to true.

        All waiters on this event will be notified.
        """
        with log_errors():
            name = self._normalize_name(name)
            # No matter if someone is listening or not,
            # we set the event to true
            event = self._get_or_create_event(name)
            self.scheduler.loop.add_callback(event.set)

    async def event_clear(self, stream=None, name=None):
        """Set the event with the given name to false."""
        with log_errors():
            name = self._normalize_name(name)
            if not self._waiter_count[name]:
                # No one is waiting for this
                # and as the default flag for an event is false
                # we can safely remove it
                with suppress(KeyError):
                    del self._waiter_count[name]
                with suppress(KeyError):
                    del self._events[name]

            else:
                # There are waiters
                # This can happen if an event is "double-cleared"
                # In principle, the event should be unset at this point
                # (because if it is set, all waiters should have been
                # notified). But to prevent race conditions
                # do to unlucky timing, we clear anyways
                assert name in self._events
                event = self._events[name]
                self.scheduler.loop.add_callback(event.clear)

    async def event_is_set(self, stream=None, name=None):
        with log_errors():
            name = self._normalize_name(name)
            # the default flag value is false
            if not name in self._events:
                return False

            return self._events[name].is_set()

    def _normalize_name(self, name):
        """ Helper function to normalize an event name """
        if isinstance(name, list):
            name = tuple(name)

        return name

    def _get_or_create_event(self, name):
        """ Helper function to return or create and return an event """
        if name not in self._events:
            # It seems we are the first one accessing it,
            # so lets create a new event
            self._events[name] = asyncio.Event()

        return self._events[name]


class Event:
    """ Distributed Centralized Event equivalent to asyncio.Event

    An event stores a single flag, which is set to false on start.
    The flag can be set to true (using the set() call) or back to false
    (with the clear() call).
    Every call to wait() blocks until the event flag is set to true.

    Parameters
    ----------
    name: string (optional)
        Name of the event.  Choosing the same name allows two
        disconnected processes to coordinate an event.  If not given, a random
        name will be generated.
    client: Client (optional)
        Client to use for communication with the scheduler.  If not given, the
        default global client will be used.

    Examples
    --------
    >>> event_1 = Event('a')  # doctest: +SKIP
    >>> event_1.wait(timeout=1)  # doctest: +SKIP
    >>> # in another process
    >>> event_2 = Event('a')  # doctest: +SKIP
    >>> event_2.set() # doctest: +SKIP
    >>> # now event_1 will stop waiting
    """
    def __init__(self, name=None, client=None):
        try:
            self.client = client or Client.current()
        except ValueError:
            # Initialise new client
            self.client = get_worker().client
        self.name = name or "event-" + uuid.uuid4().hex

    def wait(self, timeout=None):
        """ Wait until the event is set.

        Parameters
        ----------
        timeout : number, optional
            Seconds to wait on the event in the scheduler.  This does not
            include local coroutine time, network transfer time, etc..

        Examples
        --------
        >>> event = Event('a')  # doctest: +SKIP
        >>> event.wait(timeout=1)  # doctest: +SKIP

        Returns
        -------
        True if the event was set of false, if a timeout happend
        """
        result = self.client.sync(
            self.client.scheduler.event_wait,
            name=self.name,
            timeout=timeout,
        )
        return result

    def clear(self):
        """ Clear the event (set its flag to false).

        All waiters will now block.
        """
        result = self.client.sync(
            self.client.scheduler.event_clear, name=self.name,
        )
        return result

    def set(self):
        """ Set the event (set its flag to false).

        All waiters will now be released.
        """
        result = self.client.sync(
            self.client.scheduler.event_set, name=self.name,
        )
        return result

    def is_set(self):
        """ Check if the event is set """
        result = self.client.sync(
            self.client.scheduler.event_is_set, name=self.name,
        )
        return result

    def __enter__(self):
        self.set()
        return self

    def __exit__(self, *args, **kwargs):
        self.clear()

    async def __aenter__(self):
        await self.set()
        return self

    async def __aexit__(self, *args, **kwargs):
        await self.clear()

    def __reduce__(self):
        return (Event, (self.name,))

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from collections.abc import Iterable
from datetime import timedelta
from typing import TYPE_CHECKING, Literal, cast

import tlz as toolz
from tornado.ioloop import IOLoop

import dask.config
from dask.utils import parse_timedelta

from distributed.compatibility import PeriodicCallback
from distributed.metrics import time
from distributed.utils_comm import retry

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from distributed.scheduler import WorkerState

logger = logging.getLogger(__name__)


AdaptiveStateState: TypeAlias = Literal[
    "starting",
    "running",
    "stopped",
    "inactive",
]


class AdaptiveCore:
    """
    The core logic for adaptive deployments, with none of the cluster details

    This class controls our adaptive scaling behavior.  It is intended to be
    used as a super-class or mixin.  It expects the following state and methods:

    **State**

    plan: set
        A set of workers that we think should exist.
        Here and below worker is just a token, often an address or name string

    requested: set
        A set of workers that the cluster class has successfully requested from
        the resource manager.  We expect that resource manager to work to make
        these exist.

    observed: set
        A set of workers that have successfully checked in with the scheduler

    These sets are not necessarily equivalent.  Often plan and requested will
    be very similar (requesting is usually fast) but there may be a large delay
    between requested and observed (often resource managers don't give us what
    we want).

    **Functions**

    target : -> int
        Returns the target number of workers that should exist.
        This is often obtained by querying the scheduler

    workers_to_close : int -> Set[worker]
        Given a target number of workers,
        returns a set of workers that we should close when we're scaling down

    scale_up : int -> None
        Scales the cluster up to a target number of workers, presumably
        changing at least ``plan`` and hopefully eventually also ``requested``

    scale_down : Set[worker] -> None
        Closes the provided set of workers

    Parameters
    ----------
    minimum: int
        The minimum number of allowed workers
    maximum: int | inf
        The maximum number of allowed workers
    wait_count: int
        The number of scale-down requests we should receive before actually
        scaling down
    interval: str
        The amount of time, like ``"1s"`` between checks
    """

    minimum: int
    maximum: int | float
    wait_count: int
    interval: int | float
    periodic_callback: PeriodicCallback | None
    plan: set[WorkerState]
    requested: set[WorkerState]
    observed: set[WorkerState]
    close_counts: defaultdict[WorkerState, int]
    _adapting: bool
    #: Whether this adaptive strategy is periodically adapting
    _state: AdaptiveStateState
    log: deque[tuple[float, dict]]
    _retry_count: int
    _retry_delay_min: float
    _retry_delay_max: float

    def __init__(
        self,
        minimum: int = 0,
        maximum: int | float = math.inf,
        wait_count: int = 3,
        interval: str | int | float | timedelta = "1s",
    ):
        if not isinstance(maximum, int) and not math.isinf(maximum):
            raise TypeError(f"maximum must be int or inf; got {maximum}")

        self.minimum = minimum
        self.maximum = maximum
        self.wait_count = wait_count
        self.interval = parse_timedelta(interval, "seconds")
        self.periodic_callback = None

        self._retry_count = parse_timedelta(
            dask.config.get("distributed.adaptive.retry.count"), default="s"
        )
        self._retry_delay_min = parse_timedelta(
            dask.config.get("distributed.adaptive.retry.delay.min"), default="s"
        )
        self._retry_delay_max = parse_timedelta(
            dask.config.get("distributed.adaptive.retry.delay.max"), default="s"
        )

        if self.interval:
            import weakref

            self_ref = weakref.ref(self)

            async def _adapt():
                core = self_ref()
                if core:
                    await core.adapt()

            self.periodic_callback = PeriodicCallback(_adapt, self.interval * 1000)
            self._state = "starting"
            self.loop.add_callback(self._start)
        else:
            self._state = "inactive"
        try:
            self.plan = set()
            self.requested = set()
            self.observed = set()
        except Exception:
            pass

        # internal state
        self.close_counts = defaultdict(int)
        self._adapting = False
        self.log = deque(
            maxlen=dask.config.get("distributed.admin.low-level-log-length")
        )

    def _start(self) -> None:
        if self._state != "starting":
            return

        assert self.periodic_callback is not None
        self.periodic_callback.start()
        self._state = "running"
        logger.info(
            "Adaptive scaling started: minimum=%s maximum=%s",
            self.minimum,
            self.maximum,
        )

    def stop(self) -> None:
        if self._state in ("inactive", "stopped"):
            return

        if self._state == "running":
            assert self.periodic_callback is not None
            self.periodic_callback.stop()
            logger.info(
                "Adaptive scaling stopped: minimum=%s maximum=%s",
                self.minimum,
                self.maximum,
            )

        self.periodic_callback = None
        self._state = "stopped"

    async def target(self) -> int:
        """The target number of workers that should exist"""
        raise NotImplementedError()

    async def workers_to_close(self, target: int) -> list:
        """
        Give a list of workers to close that brings us down to target workers
        """
        # TODO, improve me with something that thinks about current load
        return list(self.observed)[target:]

    async def safe_target(self) -> int:
        """Used internally, like target, but respects minimum/maximum"""
        n = await self.target()
        if n > self.maximum:
            n = cast(int, self.maximum)

        if n < self.minimum:
            n = self.minimum

        return n

    async def scale_down(self, n: int) -> None:
        raise NotImplementedError()

    async def scale_up(self, workers: Iterable) -> None:
        raise NotImplementedError()

    async def recommendations(self, target: int) -> dict:
        """
        Make scale up/down recommendations based on current state and target
        """
        plan = self.plan
        requested = self.requested
        observed = self.observed

        if target == len(plan):
            self.close_counts.clear()
            return {"status": "same"}

        if target > len(plan):
            self.close_counts.clear()
            return {"status": "up", "n": target}

        # target < len(plan)
        not_yet_arrived = requested - observed
        to_close = set()
        if not_yet_arrived:
            to_close.update(toolz.take(len(plan) - target, not_yet_arrived))

        if target < len(plan) - len(to_close):
            L = await self.workers_to_close(target=target)
            to_close.update(L)

        firmly_close = set()
        for w in to_close:
            self.close_counts[w] += 1
            if self.close_counts[w] >= self.wait_count:
                firmly_close.add(w)

        for k in list(self.close_counts):  # clear out unseen keys
            if k in firmly_close or k not in to_close:
                del self.close_counts[k]

        if firmly_close:
            return {"status": "down", "workers": list(firmly_close)}
        else:
            return {"status": "same"}

    async def _adapt_callback(self) -> None:
        if self._state != "running":
            return
        try:
            await self.adapt()
        except Exception:
            logger.exception("Adaptive failed to adapt; stopping.")
            self.stop()

    async def adapt(self) -> None:
        """
        Check the current state, make recommendations, call scale

        This is the main event of the system
        """
        if self._adapting:  # Semaphore to avoid overlapping adapt calls
            return
        self._adapting = True
        try:
            await retry(
                self._adapt_once,
                count=self._retry_count,
                delay_min=self._retry_delay_min,
                delay_max=self._retry_delay_max,
            )
        finally:
            self._adapting = False

    async def _adapt_once(self) -> None:
        target = await self.safe_target()
        recommendations = await self.recommendations(target)

        if recommendations["status"] != "same":
            self.log.append((time(), dict(recommendations)))

        status = recommendations.pop("status")
        if status == "same":
            return
        elif status == "up":
            await self.scale_up(**recommendations)
        elif status == "down":
            await self.scale_down(**recommendations)
        else:
            raise RuntimeError(f"Adaptive encountered unexpected status: {status!r}")

    def __del__(self):
        self.stop()

    @property
    def loop(self) -> IOLoop:
        return IOLoop.current()

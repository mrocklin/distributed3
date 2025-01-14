from __future__ import annotations

import logging
from collections.abc import Hashable
from datetime import timedelta
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, Callable, Literal, cast

from tornado.ioloop import IOLoop

import dask.config
from dask.utils import parse_timedelta

from distributed.compatibility import PeriodicCallback
from distributed.core import Status
from distributed.deploy.adaptive_core import AdaptiveCore
from distributed.protocol import pickle
from distributed.utils import log_errors

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    import distributed
    from distributed.deploy.cluster import Cluster

logger = logging.getLogger(__name__)


AdaptiveStateState: TypeAlias = Literal[
    "starting",
    "running",
    "stopped",
    "inactive",
]


class Adaptive(AdaptiveCore):
    '''
    Adaptively allocate workers based on scheduler load.  A superclass.

    Contains logic to dynamically resize a Dask cluster based on current use.
    This class needs to be paired with a system that can create and destroy
    Dask workers using a cluster resource manager.  Typically it is built into
    already existing solutions, rather than used directly by users.
    It is most commonly used from the ``.adapt(...)`` method of various Dask
    cluster classes.

    Parameters
    ----------
    cluster: object
        Must have scale and scale_down methods/coroutines
    interval : timedelta or str, default "1000 ms"
        Milliseconds between checks
    wait_count: int, default 3
        Number of consecutive times that a worker should be suggested for
        removal before we remove it.
    target_duration: timedelta or str, default "5s"
        Amount of time we want a computation to take.
        This affects how aggressively we scale up.
    worker_key: Callable[WorkerState]
        Function to group workers together when scaling down
        See Scheduler.workers_to_close for more information
    minimum: int
        Minimum number of workers to keep around
    maximum: int
        Maximum number of workers to keep around
    **kwargs:
        Extra parameters to pass to Scheduler.workers_to_close

    Examples
    --------

    This is commonly used from existing Dask classes, like KubeCluster

    >>> from dask_kubernetes import KubeCluster
    >>> cluster = KubeCluster()
    >>> cluster.adapt(minimum=10, maximum=100)

    Alternatively you can use it from your own Cluster class by subclassing
    from Dask's Cluster superclass

    >>> from distributed.deploy import Cluster
    >>> class MyCluster(Cluster):
    ...     def scale_up(self, n):
    ...         """ Bring worker count up to n """
    ...     def scale_down(self, workers):
    ...        """ Remove worker addresses from cluster """

    >>> cluster = MyCluster()
    >>> cluster.adapt(minimum=10, maximum=100)

    In some cases, you may want to subclass Adaptive for custom functionality.
    As mentioned, for example, you might need to override :meth:`Adaptive.workers_to_close`
    to determine how workers are removed from the cluster in accordance to your workload's
    memory requirements. Same thing goes for other class methods like :meth:`Adaptive.target`.
    For example, you can modify the actual target time for the work to be done in half of the actual
    target time, which will also increases how rapidly the scheduler asks to scale to get the work
    done.

    >>> from distributed import Adaptive
    >>> from dask.distributed import Client, LocalCluster
    >>> from distributed import Adaptive
    >>> from random import random

    >>> class MyAdaptive(Adaptive):
    ...    def __init__(self, cluster=None, **kwargs):
    ...        self.factor = kwargs.pop('factor', 1) # factor to reduce target time by half
    ...        super().__init__(cluster=cluster, **kwargs)

    ...     async def target(self) -> int:
    ...        """ Determine target number of workers """
    ...        # new duration after modifying target time
    ...        new_duration = self.target_duration * self.factor
    ...        return await self.scheduler.adaptive_target(
    ...             target_duration=new_duration
    ...        )

    >>> def add(one, two):  # simple function to add two numbers
    ...    return one + two

    >>> if __name__ == "__main__":

    ...    cluster = LocalCluster(n_workers=1, memory_limit=0)
    ...    # pass factor as an additional argument to the adapt method
    ...    cluster.adapt(Adaptive = MyAdaptive,minimum=1, maximum=10, factor=0.5)
    ...    client = Client(cluster)

    ...    for i in range(5000):
    ...        client.submit(add, random(), random())


    Notes
    -----
    Subclasses can override :meth:`Adaptive.target` and
    :meth:`Adaptive.workers_to_close` to control when the cluster should be
    resized. The default implementation checks if there are too many tasks
    per worker or too little memory available (see
    :meth:`distributed.Scheduler.adaptive_target`).
    The values for interval, min, max, wait_count and target_duration can be
    specified in the dask config under the distributed.adaptive key.
    '''

    interval: float | None
    periodic_callback: PeriodicCallback | None
    #: Whether this adaptive strategy is periodically adapting
    state: AdaptiveStateState

    def __init__(
        self,
        cluster: Cluster,
        interval: str | float | timedelta | None = None,
        minimum: int | None = None,
        maximum: int | float | None = None,
        wait_count: int | None = None,
        target_duration: str | float | timedelta | None = None,
        worker_key: (
            Callable[[distributed.scheduler.WorkerState], Hashable] | None
        ) = None,
        **kwargs: Any,
    ):
        self.cluster = cluster
        self.worker_key = worker_key
        self._workers_to_close_kwargs = kwargs

        if interval is None:
            interval = dask.config.get("distributed.adaptive.interval")
        if minimum is None:
            minimum = cast(int, dask.config.get("distributed.adaptive.minimum"))
        if maximum is None:
            maximum = cast(float, dask.config.get("distributed.adaptive.maximum"))
        if wait_count is None:
            wait_count = cast(int, dask.config.get("distributed.adaptive.wait-count"))
        if target_duration is None:
            target_duration = cast(
                str, dask.config.get("distributed.adaptive.target-duration")
            )

        super().__init__(minimum=minimum, maximum=maximum, wait_count=wait_count)

        self.interval = parse_timedelta(interval, "seconds")
        self.periodic_callback = None

        if self.interval and self.cluster:
            import weakref

            self_ref = weakref.ref(self)

            async def _adapt():
                adaptive = self_ref()
                if not adaptive or adaptive.state != "running":
                    return
                if adaptive.cluster.status != Status.running:
                    adaptive.stop(reason="cluster-not-running")
                    return
                try:
                    await adaptive.adapt()
                except Exception:
                    logger.warning(
                        "Adaptive encountered an error while adapting", exc_info=True
                    )

            self.periodic_callback = PeriodicCallback(_adapt, self.interval * 1000)
            self.state = "starting"
            self.loop.add_callback(self._start)
        else:
            self.state = "inactive"

        self.target_duration = parse_timedelta(target_duration)

    def _start(self) -> None:
        if self.state != "starting":
            return

        assert self.periodic_callback is not None
        self.periodic_callback.start()
        self.state = "running"
        logger.info(
            "Adaptive scaling started: minimum=%s maximum=%s",
            self.minimum,
            self.maximum,
        )

    def stop(self, reason: str = "unknown") -> None:
        if self.state in ("inactive", "stopped"):
            return

        if self.state == "running":
            assert self.periodic_callback is not None
            self.periodic_callback.stop()
        logger.info(
            "Adaptive scaling stopped: minimum=%s maximum=%s. Reason: %s",
            self.minimum,
            self.maximum,
            reason,
        )

        self.periodic_callback = None
        self.state = "stopped"

    @property
    def scheduler(self):
        return self.cluster.scheduler_comm

    @property
    def plan(self):
        return self.cluster.plan

    @property
    def requested(self):
        return self.cluster.requested

    @property
    def observed(self):
        return self.cluster.observed

    async def target(self):
        """
        Determine target number of workers that should exist.

        Notes
        -----
        ``Adaptive.target`` dispatches to Scheduler.adaptive_target(),
        but may be overridden in subclasses.

        Returns
        -------
        Target number of workers

        See Also
        --------
        Scheduler.adaptive_target
        """
        return await self.scheduler.adaptive_target(
            target_duration=self.target_duration
        )

    async def recommendations(self, target: int) -> dict:
        if len(self.plan) != len(self.requested):
            # Ensure that the number of planned and requested workers
            # are in sync before making recommendations.
            await self.cluster

        return await super().recommendations(target)

    async def workers_to_close(self, target: int) -> list[str]:
        """
        Determine which, if any, workers should potentially be removed from
        the cluster.

        Notes
        -----
        ``Adaptive.workers_to_close`` dispatches to Scheduler.workers_to_close(),
        but may be overridden in subclasses.

        Returns
        -------
        List of worker names to close, if any

        See Also
        --------
        Scheduler.workers_to_close
        """
        return await self.scheduler.workers_to_close(
            target=target,
            key=pickle.dumps(self.worker_key) if self.worker_key else None,
            attribute="name",
            **self._workers_to_close_kwargs,
        )

    @log_errors
    async def scale_down(self, workers):
        if not workers:
            return

        logger.info("Retiring workers %s", workers)
        # Ask scheduler to cleanly retire workers
        await self.scheduler.retire_workers(
            names=workers,
            remove=True,
            close_workers=True,
        )

        # close workers more forcefully
        f = self.cluster.scale_down(workers)
        if isawaitable(f):
            await f

    async def scale_up(self, n):
        f = self.cluster.scale(n)
        if isawaitable(f):
            await f

    @property
    def loop(self) -> IOLoop:
        """Override Adaptive.loop"""
        if self.cluster:
            return self.cluster.loop  # type: ignore[return-value]
        else:
            return IOLoop.current()

    def __del__(self):
        self.stop(reason="adaptive-deleted")

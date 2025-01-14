from __future__ import annotations

r"""
Tasks that are actually stateful services running on all workers, for doing out-of-band
operations in a controlled way.

*Services* provide a structured way to combine task graphs of pure functions (Dask's main
use case) with stateful, distributed subsystems. Some algorithms (DataFrame shuffling,
array rechunking, etc.) are inefficient when represented as task graphs, but can be
implemented simply and performantly as bespoke systems that handle their own low-level
data transfer and computation manually.

The `Service` interface offers a structured way to "hand off" data from the task-graph
paradigm to these out-of-band operations, and to "hand back" the results so downstream
tasks can use them, while maintaining resilience of the overall graph.

In a graph like::

    {
        ("input", 1): 1,
        ("input", 2): 2,
        "foo-service": (FooService, [("input", 1), ("input", 2)]),  # annotated as a service
        ("output", 1): (service_result, "foo-service"),             # never actually runs
        ("output", 2): (service_result, "foo-service"),             # never actually runs
        ("output", 3): (service_result, "foo-service"),             # never actually runs
        ("downstream", 1): (use, ("output", 1)),
        ("downstream", 2): (use, ("output", 2)),
        ("downstream", 3): (use, ("output", 3)),
    }

Or visually::

    d1   d2   d3
    |    |    |
    o1   o2   o3
      \  |   /
    foo-service
        / \
      i1  i2

If the single ``foo-service`` task is annotated as a service, it'll actually be computed
by launching ``FooService`` instances on every worker, passing the inputs into them,
letting them do whatever out-of-band operations they wish (including communicating with
each other or external systems), and waiting for them to produce the output keys. Once
each output key is produced (the `service_result` functions never actually run), any
tasks downstream of it can run.

---------------------------------------------------------------------------------------

Dask handles startup/shutdown, resilience, peer discovery, leader election, RPC
interfaces, and composition for Services:

Startup/shutdown
----------------

* Once the *first* input key to a service task is ready (state ``memory``), service
  instances are started on all workers in the cluster.
* Once the last output key is produced by a service, all instances are told to stop.
* Any instance can error the service task; all other instances will be told to stop.
* Any instance can restart the service task; all other instances will be told to stop,
  then re-launched from scratch.
* If the service task is cancelled (because keys downstream from it are released, etc.),
  all instances are told to stop.

Resilience
----------

* If a worker that's running a service leaves, the entire service task is restarted on
  all workers. TODO change once `peer_joined`/`peer_left` are added.
* If keys downstream of a service need to be recomputed, the service will be rerun, just
  like regular tasks are.
* The service instances are informed at runtime which inputs to expect and which outputs to
  produce, so partial recomputation (some outputs already in memory, some not) can be
  handled correctly.

Peer discovery and RPC
----------------------

* At startup, services are given a list of their peers as RPC handles.
* The RPC handle allows a `Service` to call arbitrary methods on its peers on other
  workers.
* Loopback calls (using the RPC to call yourself) take a fastpath, bypassing
  serialization and the network stack.

Leader election
---------------
* At startup, one instance is designated as the "leader", making it easier to coordinate
  initialization tasks that need to run exactly once.

Composition
-----------

One Service's outputs can be another Service's inputs.

A `Service` can depend on multiple input Services, or both normal tasks and Services, or
have no dependencies at all. Note, though, that input keys are always treated as a flat
structure (`Service.add_key` is called, one key at a time, from whatever worker already
holds the key). Therefore, if particular inputs need to be co-located, the `Service` is
responsible for transferring them around internally. (Doing just that in clever ways is
often the whole point of a `Service`.)

                 o   o   o
                  \  |  /
              combiner-service
          /   /        \   \   \
         o   o          o   o   o
          \ /            \  |  /
     bar-service        baz-service
     /   |   \          /  /  \  \
    o    o    o        i  i    i  i
      \  |   /          \  \  /  /
     foo-service       normal-tasks

Restrictions
~~~~~~~~~~~~

If a task depends on a `Service`, that must be its only dependency. For example, this
will raise an error when the graph is submitted to the scheduler:

     o     o   <-- tasks that depend on a Service cannot have >1 dependency
     | \ / |
     | / \ |
    s-1   s-2
     /\    /\
    i  i  i  i

A `Future` cannot refer to a service task; that is, `TaskState.who_wants` must be
empty if `TaskState.service` is True. In other words, you can't compute a `Service`
directly, only its output keys.
"""

from abc import ABC, abstractmethod
from typing import (
    AbstractSet,
    Any,
    Awaitable,
    Callable,
    Generic,
    Hashable,
    Iterable,
    Mapping,
    NewType,
    NoReturn,
    Protocol,
    Sequence,
    TypeVar,
)

from typing_extensions import ParamSpec

from dask import config
from dask.base import tokenize
from dask.highlevelgraph import Layer


def service_result(service: Service) -> NoReturn:
    """
    Placeholder function used in output tasks for a `Service`.

    The scheduler will not actually run tasks that depend on a `Service`;
    they are just added to the graph to create dependency structure.
    Instead, Services use `Concierge.produce_key` to mark their output keys
    as complete. Then, normal tasks that depend on those output keys can run.
    """
    raise RuntimeError(
        f"`service_result` should not actually be called on {service!r}."
        "Are you sure you're running on a distributed cluster?"
    )


ServiceKey = NewType("ServiceKey", str)
"""
The name of a single key in the graph that's computed using a `Service`.

The term "service task" is used to refer to overall computation of this key,
performed by many `Service` instances running on different workers.
"""

ServiceId = NewType("ServiceId", str)
"Opaque ID of a `Service` instance"

T = TypeVar("T", bound="Service")
# TODO: use pep-673 Self type https://peps.python.org/pep-0673/


class Service(ABC):
    @abstractmethod
    async def start(
        self: T,
        *,
        id: ServiceId,
        key: ServiceKey,
        peers: Mapping[ServiceId, ServiceHandle[T]],
        input_keys: Sequence[str],
        output_keys: Sequence[str],
        concierge: Concierge,
        leader: ServiceId,
        **kwargs,
    ) -> None:
        """
        Called to start the `Service` on a `Worker`.

        Once ``start`` returns, the `Service` instance is considered ready, and the
        `Worker` can call ``add_key``.

        Parameters
        ----------
        id:
            Opaque identifier for this instance.
        key:
            The key in the graph that created the `Service`. All instances created from
            that key (across all workers) receive the same ``key``.

            Multiple instances of the same type of `Service` can exist on a worker at
            once, but all are guaranteed to have a different ``key``.
        peers:
            The IDs of all `Service` (including this one) currently being started to
            handle this service task, and ServiceHandles to communicate with them.

            Note that peer services are not guaranteed to be running yet.

            The order of ``peers`` is undefined, but it will be the same for every
            `Service` in ``peers``.
        input_keys:
            The input keys this service task will receive, *across all workers*, in
            priority order.
        output_keys:
            The output keys this service task must produce, *across all workers*, in
            priority order.
        concierge:
            A `Concierge` instance the `Service` can use to communicate with the
            `Worker`. Used to hand off output keys back to Dask, restart or fail the
            service task, etc.
        leader:
            The ID of the instance chosen to be the "leader" of the service task. The
            scheduler will pick exactly one instance in ``peers`` to be the leader. The
            order in which `Service` instances start up on workers is of course
            undefined; the leader may not be the first instance to actually start.

            The leader has no special treatment, and which instance is selected as the
            leader is irrelevant to Dask. This flag is passed purely as a convenience
            for Services, to make it easier to coordinate initialization tasks that must
            run exactly once.
        **kwargs:
            Forward-compatibility only; any additional arguments can be ignored.
        """
        ...

    # TODO: remove this in favor of `peer_joined` and `peer_left`.
    # Once those methods are supported, and workers joining/leaving doesn't automatically
    # restart the service task, we can't support `all_started` anymore: what if one of the
    # original peers dies before it's started?
    # `peer_joined` and `peer_left` are more powerful, because they allow for Service resizing,
    # but still allow services to restart when workers leave (and wait for all to arrive) if
    # they wish.
    @abstractmethod
    async def all_started(self) -> None:
        """
        Called once all instances in the original ``peers`` list have started.

        Do not make the ``start`` method block until ``all_started`` has been called;
        this will cause a deadlock.
        """
        ...

    @abstractmethod
    async def add_key(self, key: str, data: Any) -> None:
        """
        Called by the `Worker` to "hand off" data to the `Service`.

        Once `add_key` returns, this `Service` instance owns ``data``, and the
        `Worker` will release all references to the data once no other tasks or clients
        are also waiting on ``key``.

        `add_key` can be called as soon as ``start`` has returned. It will only be
        called with keys in ``input_keys``.

        For a given input key, `add_key` will be called exactly once *across all workers*.
        `add_key` is always called from a worker that already holds ``key`` (data is
        not transferred just for ``add_key`). If multiple workers hold replicas of the key,
        the worker on which `add_key` is called is undefined.

        If `add_key` raises an error, it's equivalent to calling `Concierge.error` with that
        exception: the entire service task will be erred, and all instances stopped.
        """
        ...

    @abstractmethod
    async def all_inputs_done(self) -> None:
        """
        Called when, across all workers, `add_key` has returned for every key in ``input_keys``.

        Broadcast by the scheduler to all instances of this service task,
        to help Services coordinate.
        """
        ...

    # Peers joining and leaving won't be supported in the first iteration.
    # For now, new workers will be ignored, and any worker leaving will trigger
    # a restart of the whole service group? (Don't want to have this be released behavior
    # though, since it'll make adding `peer_joined`/`peer_left` backwards-incompatible.)
    # * Dealing with peers joining is somewhat complex. Do current instances get to
    #   veto whether a new peer can join, or does any worker satisfying the restrictions
    #   of the Service task get to join automatically? I assume `peer_joined` doesn't get
    #   called until after `start` has returned on the new instance. What `peers` list gets
    #   passed to the new peer (all current ones, I assume)? What happens when other workers
    #   come up while the new peer's `start` is running—how do we buffer those `peer_joined`
    #   messages to deliver to the peer that hasn't started up yet?
    # * `peer_left` seems more straightforward.
    # In general though, these may not scale well. In a 10k worker cluster, notifying every other
    # worker when one comes and goes is a lot of noise.
    # And mostly, if services would decide that a peer leaving (or joining) means they should
    # restart, then the scheduler will be bombarded by restart messages from (N-1) workers
    # when one worker leaves.

    # async def peer_joined(self: T, id: ServiceId, handle: ServiceHandle[T]) -> None:
    #     ...

    # async def peer_left(self, id: ServiceId) -> None:
    #     ...

    @abstractmethod
    async def stop(self) -> None:
        """
        Called by the `Worker` when this service task should stop.

        Cases in which `stop` is called:
        * Success: all ``output_keys`` have been produced.
        * Failure: `Concierge.error` was called, or `Service.add_key` raised an error.
        * Restart: `Concierge.restart` was called.
        * Restart: a peer worker left (TODO remove this case!).

        The `Service` is not informed what the cause for stopping is.

        The `Service` must clean up all state (files, connections, data, etc.)
        before `stop` returns.

        After `stop` returns, a new `Service` instance with the same service task
        key may be created on the `Worker`.
        """
        ...

    @abstractmethod
    def __sizeof__(self) -> int:
        """
        Amount of memory this `Service` is using for internal state.

        This will be counted as managed memory by dask.
        """
        ...

    @abstractmethod
    def spilled_bytes(self) -> int:
        """
        Number of bytes on disk this `Service` is using for internal state.

        This will be counted as spilled memory by dask.
        """
        ...

    @abstractmethod
    async def please_evict(self, nbytes: int) -> int:
        """
        Called by the `Worker`, requesting that this `Service` reduce its memory usage by at least ``nbytes``.
        """
        ...


P = ParamSpec("P")
R = TypeVar("R")


class Concierge:
    "Interface for `Service` instances to update the progress of a service task"
    key: ServiceKey

    async def produce_key(self, key: str, data: Any) -> None:
        """
        Hand off output data from a `Service` back to Dask.

        Once ``await produce_key(key, data)`` has returned, the data is owned by the
        worker and tracked by the scheduler. The `Service` should release all references
        to the data.

        If ``produce_key`` raises an error, the state of the data is undefined. In most
        cases, the `Service` should call `restart` if this happens.
        """
        ...

    async def restart(self) -> None:
        """
        Request that the service task be restarted.

        `Service.stop` is guaranteed not to be called until after `restart` has
        returned.

        Once `restart` has returned, `Service.stop` will be called on all instances for
        the service task, and workers will release their references those instances. Once
        `stop` has returned on all instances, the task will transition to ``waiting`` on
        the scheduler. Then, it will transition back to ``processing``, and new
        `Service` instances (with the same service task key) will be started on all
        workers.
        """
        ...

    async def error(self, exc: Exception) -> None:
        """
        Fail the service task with an error message.

        `Service.stop` is guaranteed not to be called until after `error` has returned.

        Once `error` has returned, `Service.stop` will be called on all instances for
        the service task, and workers will release their references those instances. Once
        `stop` has returned on all instances, the task will transition to ``erred`` on
        the scheduler. The result of the task will be ``exc``.

        If `error` is called multiple times, only the first ``exc`` to reach the
        scheduler will become the result; all others will be dropped (though all calls
        to `error` are logged on the `Worker`).
        """
        ...

    async def run_in_executor(
        self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs
    ) -> Awaitable[R]:
        """
        Schedule a function to run in the Worker's default executor.

        This will take up a slot in `Worker.executing_count`, reducing the number of
        other tasks (or `run_in_executor` calls) that can run concurrently. This allows
        multiple Services (or Services and normal tasks) to cooperatively run
        resource-intensive synchronous functions concurrently without oversubscribing
        the `Worker`. This way, `Worker.nthreads` is respected even for out-of-band
        functions.

        For event-loop-blocking functions that aren't resource-intensive (network calls,
        etc.), `distributed.compatibility.to_thread` is preferred.
        """
        ...


# TODO: how to type annotate RPCs? https://github.com/python/mypy/issues/5523
class ServiceHandle(Generic[T]):
    "RPC to a Service instance running on another worker"

    def __getattr__(self, attr: str) -> _RPC:
        ...


# https://stackoverflow.com/a/65564936/17100540
class _RPC(Protocol):
    def __call__(self, *args, **kwargs) -> Awaitable:
        ...


# TODO this is certainly incorrect/incomplete, the `Layer` interface is quite hard to follow.
class SimpleServiceLayer(Layer):
    def __init__(
        self,
        name: str,
        service: Service,
        inputs: Sequence[str],
        outputs: Sequence[str],
    ):
        self.name = name
        self.key = f"({name}, {tokenize(outputs)})"
        # ^ NOTE: tokenize the actual key based on which outputs are produced, so the culled
        # version of the Service has a different key from the un-culled version.
        self.service = service
        self.inputs = inputs
        self.outputs = outputs
        super().__init__(
            annotations={**config.get("annotations", {}), self.key: {"service": True}}
        )

    def get_output_keys(self) -> AbstractSet:
        return set(self.outputs)

    def cull(
        self, keys: set, all_hlg_keys: Iterable
    ) -> tuple[Layer, Mapping[Hashable, set]]:
        # FIXME I have no idea what "external key dependencies" means in `Layer.cull`,
        # not sure if an empty dict is correct here or not.
        if keys == set(self.outputs):
            return self, {}
        return type(self)(self.name, self.service, self.inputs, list(keys)), {}

    def _construct_graph(self) -> dict[str, Any]:
        dsk: dict[str, Any] = {o: (service_result, self.key) for o in self.outputs}
        dsk[self.key] = (self.service, self.inputs)
        return dsk

    @property
    def _dict(self):
        """Materialize full dict representation"""
        if hasattr(self, "_cached_dict"):
            return self._cached_dict
        else:
            dsk = self._construct_graph()
            self._cached_dict = dsk
        return self._cached_dict

    def __getitem__(self, key):
        return self._dict[key]

    def __iter__(self):
        yield self.key
        yield from iter(self.outputs)

    def __len__(self):
        return len(self.outputs) + 1

    def is_materialized(self):
        return hasattr(self, "_cached_dict")

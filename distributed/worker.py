from __future__ import annotations

import asyncio
import bisect
import builtins
import errno
import functools
import logging
import os
import pathlib
import random
import sys
import threading
import warnings
import weakref
from collections import defaultdict, deque
from collections.abc import (
    Callable,
    Collection,
    Container,
    Iterable,
    Mapping,
    MutableMapping,
)
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import suppress
from datetime import timedelta
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, cast

from tlz import first, keymap, pluck
from tornado.ioloop import IOLoop, PeriodicCallback

import dask
from dask.core import istask
from dask.system import CPU_COUNT
from dask.utils import (
    apply,
    format_bytes,
    funcname,
    parse_bytes,
    parse_timedelta,
    stringify,
    tmpdir,
    typename,
)

from distributed import preloading, profile, utils
from distributed.batched import BatchedSend
from distributed.collections import LRU
from distributed.comm import Comm, connect, get_address_host, parse_address
from distributed.comm import resolve_address as comm_resolve_address
from distributed.comm.addressing import address_from_user_args
from distributed.comm.utils import OFFLOAD_THRESHOLD
from distributed.compatibility import randbytes, to_thread
from distributed.core import (
    CommClosedError,
    ConnectionPool,
    Status,
    coerce_to_address,
    error_message,
    pingpong,
)
from distributed.core import rpc as RPCType
from distributed.core import send_recv
from distributed.diagnostics import nvml
from distributed.diagnostics.plugin import _get_plugin_name
from distributed.diskutils import WorkDir, WorkSpace
from distributed.http import get_handlers
from distributed.metrics import time
from distributed.node import ServerNode
from distributed.proctitle import setproctitle
from distributed.protocol import pickle, to_serialize
from distributed.pubsub import PubSubWorkerExtension
from distributed.security import Security
from distributed.shuffle import ShuffleWorkerExtension
from distributed.sizeof import safe_sizeof as sizeof
from distributed.utils import (
    TimeoutError,
    _maybe_complex,
    get_ip,
    has_arg,
    import_file,
    in_async_call,
    iscoroutinefunction,
    json_load_robust,
    key_split,
    log_errors,
    offload,
    parse_ports,
    recursive_to_dict,
    silence_logging,
    thread_state,
    warn_on_duration,
)
from distributed.utils_comm import gather_from_workers, pack_data, retry_operation
from distributed.utils_perf import disable_gc_diagnosis, enable_gc_diagnosis
from distributed.versions import get_versions
from distributed.worker_memory import (
    DeprecatedMemoryManagerAttribute,
    DeprecatedMemoryMonitor,
    WorkerMemoryManager,
)
from distributed.worker_state_machine import (
    NO_VALUE,
    AcquireReplicasEvent,
    AlreadyCancelledEvent,
    BaseWorker,
    CancelComputeEvent,
    ComputeTaskEvent,
    DeprecatedWorkerStateAttribute,
    ExecuteFailureEvent,
    ExecuteSuccessEvent,
    FindMissingEvent,
    FreeKeysEvent,
    GatherDepBusyEvent,
    GatherDepFailureEvent,
    GatherDepNetworkFailureEvent,
    GatherDepSuccessEvent,
    PauseEvent,
    RefreshWhoHasEvent,
    RemoveReplicasEvent,
    RescheduleEvent,
    RetryBusyWorkerEvent,
    SecedeEvent,
    StateMachineEvent,
    StealRequestEvent,
    TaskState,
    UnpauseEvent,
    UpdateDataEvent,
    WorkerState,
)
from distributed.worker_state_machine import logger as wsm_logger

if TYPE_CHECKING:
    # FIXME import from typing (needs Python >=3.10)
    from typing_extensions import ParamSpec

    # Circular imports
    from distributed.client import Client
    from distributed.diagnostics.plugin import WorkerPlugin
    from distributed.nanny import Nanny

    P = ParamSpec("P")
    T = TypeVar("T")

logger = logging.getLogger(__name__)

LOG_PDB = dask.config.get("distributed.admin.pdb-on-err")

DEFAULT_EXTENSIONS: dict[str, type] = {
    "pubsub": PubSubWorkerExtension,
    "shuffle": ShuffleWorkerExtension,
}

DEFAULT_METRICS: dict[str, Callable[[Worker], Any]] = {}

DEFAULT_STARTUP_INFORMATION: dict[str, Callable[[Worker], Any]] = {}

WORKER_ANY_RUNNING = {
    Status.running,
    Status.paused,
    Status.closing_gracefully,
}


def fail_hard(method: Callable[P, T]) -> Callable[P, T]:
    """
    Decorator to close the worker if this method encounters an exception.
    """
    if iscoroutinefunction(method):

        @functools.wraps(method)
        async def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> Any:
            try:
                return await method(self, *args, **kwargs)  # type: ignore
            except Exception as e:
                if self.status not in (Status.closed, Status.closing):
                    self.log_event("worker-fail-hard", error_message(e))
                    logger.exception(e)
                await _force_close(self)
                raise

    else:

        @functools.wraps(method)
        def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return method(self, *args, **kwargs)
            except Exception as e:
                if self.status not in (Status.closed, Status.closing):
                    self.log_event("worker-fail-hard", error_message(e))
                    logger.exception(e)
                self.loop.add_callback(_force_close, self)
                raise

    return wrapper  # type: ignore


async def _force_close(self):
    """
    Used with the fail_hard decorator defined above

    1.  Wait for a worker to close
    2.  If it doesn't, log and kill the process
    """
    try:
        await asyncio.wait_for(self.close(nanny=False, executor_wait=False), 30)
    except (KeyboardInterrupt, SystemExit):  # pragma: nocover
        raise
    except (Exception, BaseException):  # pragma: nocover
        # Worker is in a very broken state if closing fails. We need to shut down
        # immediately, to ensure things don't get even worse and this worker potentially
        # deadlocks the cluster.
        if self.state.validate and not self.nanny:
            # We're likely in a unit test. Don't kill the whole test suite!
            raise

        logger.critical(
            "Error trying close worker in response to broken internal state. "
            "Forcibly exiting worker NOW",
            exc_info=True,
        )
        # use `os._exit` instead of `sys.exit` because of uncertainty
        # around propagating `SystemExit` from asyncio callbacks
        os._exit(1)


class Worker(BaseWorker, ServerNode):
    """Worker node in a Dask distributed cluster

    Workers perform two functions:

    1.  **Serve data** from a local dictionary
    2.  **Perform computation** on that data and on data from peers

    Workers keep the scheduler informed of their data and use that scheduler to
    gather data from other workers when necessary to perform a computation.

    You can start a worker with the ``dask-worker`` command line application::

        $ dask-worker scheduler-ip:port

    Use the ``--help`` flag to see more options::

        $ dask-worker --help

    The rest of this docstring is about the internal state that the worker uses
    to manage and track internal computations.

    **State**

    **Informational State**

    These attributes don't change significantly during execution.

    * **nthreads:** ``int``:
        Number of nthreads used by this worker process
    * **executors:** ``dict[str, concurrent.futures.Executor]``:
        Executors used to perform computation. Always contains the default
        executor.
    * **local_directory:** ``path``:
        Path on local machine to store temporary files
    * **scheduler:** ``rpc``:
        Location of scheduler.  See ``.ip/.port`` attributes.
    * **name:** ``string``:
        Alias
    * **services:** ``{str: Server}``:
        Auxiliary web servers running on this worker
    * **service_ports:** ``{str: port}``:
    * **total_in_connections**: ``int``
        The maximum number of concurrent incoming requests for data.
        See also
        :attr:`distributed.worker_state_machine.WorkerState.total_out_connections`.
    * **batched_stream**: ``BatchedSend``
        A batched stream along which we communicate to the scheduler
    * **log**: ``[(message)]``
        A structured and queryable log.  See ``Worker.story``

    **Volatile State**

    These attributes track the progress of tasks that this worker is trying to
    complete. In the descriptions below a ``key`` is the name of a task that
    we want to compute and ``dep`` is the name of a piece of dependent data
    that we want to collect from others.

    * **threads**: ``{key: int}``
        The ID of the thread on which the task ran
    * **active_threads**: ``{int: key}``
        The keys currently running on active threads
    * **state**: ``WorkerState``
        Encapsulated state machine. See
        :class:`~distributed.worker_state_machine.BaseWorker` and
        :class:`~distributed.worker_state_machine.WorkerState`

    Parameters
    ----------
    scheduler_ip: str, optional
    scheduler_port: int, optional
    scheduler_file: str, optional
    host: str, optional
    data: MutableMapping, type, None
        The object to use for storage, builds a disk-backed LRU dict by default
    nthreads: int, optional
    local_directory: str, optional
        Directory where we place local resources
    name: str, optional
    memory_limit: int, float, string
        Number of bytes of memory that this worker should use.
        Set to zero for no limit.  Set to 'auto' to calculate
        as system.MEMORY_LIMIT * min(1, nthreads / total_cores)
        Use strings or numbers like 5GB or 5e9
    memory_target_fraction: float or False
        Fraction of memory to try to stay beneath
        (default: read from config key distributed.worker.memory.target)
    memory_spill_fraction: float or False
        Fraction of memory at which we start spilling to disk
        (default: read from config key distributed.worker.memory.spill)
    memory_pause_fraction: float or False
        Fraction of memory at which we stop running new tasks
        (default: read from config key distributed.worker.memory.pause)
    max_spill: int, string or False
        Limit of number of bytes to be spilled on disk.
        (default: read from config key distributed.worker.memory.max-spill)
    executor: concurrent.futures.Executor, dict[str, concurrent.futures.Executor], "offload"
        The executor(s) to use. Depending on the type, it has the following meanings:
            - Executor instance: The default executor.
            - Dict[str, Executor]: mapping names to Executor instances. If the
              "default" key isn't in the dict, a "default" executor will be created
              using ``ThreadPoolExecutor(nthreads)``.
            - Str: The string "offload", which refer to the same thread pool used for
              offloading communications. This results in the same thread being used
              for deserialization and computation.
    resources: dict
        Resources that this worker has like ``{'GPU': 2}``
    nanny: str
        Address on which to contact nanny, if it exists
    lifetime: str
        Amount of time like "1 hour" after which we gracefully shut down the worker.
        This defaults to None, meaning no explicit shutdown time.
    lifetime_stagger: str
        Amount of time like "5 minutes" to stagger the lifetime value
        The actual lifetime will be selected uniformly at random between
        lifetime +/- lifetime_stagger
    lifetime_restart: bool
        Whether or not to restart a worker after it has reached its lifetime
        Default False
    kwargs: optional
        Additional parameters to ServerNode constructor

    Examples
    --------

    Use the command line to start a worker::

        $ dask-scheduler
        Start scheduler at 127.0.0.1:8786

        $ dask-worker 127.0.0.1:8786
        Start worker at:               127.0.0.1:1234
        Registered with scheduler at:  127.0.0.1:8786

    See Also
    --------
    distributed.scheduler.Scheduler
    distributed.nanny.Nanny
    """

    _instances: ClassVar[weakref.WeakSet[Worker]] = weakref.WeakSet()
    _initialized_clients: ClassVar[weakref.WeakSet[Client]] = weakref.WeakSet()

    nanny: Nanny | None
    _lock: threading.Lock
    total_in_connections: int
    threads: dict[str, int]  # {ts.key: thread ID}
    active_threads_lock: threading.Lock
    active_threads: dict[int, str]  # {thread ID: ts.key}
    active_keys: set[str]
    profile_keys: defaultdict[str, dict[str, Any]]
    profile_keys_history: deque[tuple[float, dict[str, dict[str, Any]]]]
    profile_recent: dict[str, Any]
    profile_history: deque[tuple[float, dict[str, Any]]]
    incoming_transfer_log: deque[dict[str, Any]]
    outgoing_transfer_log: deque[dict[str, Any]]
    incoming_count: int
    outgoing_count: int
    outgoing_current_count: int
    bandwidth: float
    latency: float
    profile_cycle_interval: float
    workspace: WorkSpace
    _workdir: WorkDir
    local_directory: str
    _client: Client | None
    bandwidth_workers: defaultdict[str, tuple[float, int]]
    bandwidth_types: defaultdict[type, tuple[float, int]]
    preloads: list[preloading.Preload]
    contact_address: str | None
    _start_port: int | str | Collection[int] | None = None
    _start_host: str | None
    _interface: str | None
    _protocol: str
    _dashboard_address: str | None
    _dashboard: bool
    _http_prefix: str
    total_resources: dict[str, float]
    death_timeout: float | None
    lifetime: float | None
    lifetime_stagger: float | None
    lifetime_restart: bool
    extensions: dict
    security: Security
    connection_args: dict[str, Any]
    loop: IOLoop
    executors: dict[str, Executor]
    batched_stream: BatchedSend
    name: Any
    scheduler_delay: float
    stream_comms: dict[str, BatchedSend]
    heartbeat_interval: float
    heartbeat_active: bool
    services: dict[str, Any] = {}
    service_specs: dict[str, Any]
    metrics: dict[str, Callable[[Worker], Any]]
    startup_information: dict[str, Callable[[Worker], Any]]
    low_level_profiler: bool
    scheduler: Any
    execution_state: dict[str, Any]
    plugins: dict[str, WorkerPlugin]
    _pending_plugins: tuple[WorkerPlugin, ...]

    def __init__(
        self,
        scheduler_ip: str | None = None,
        scheduler_port: int | None = None,
        *,
        scheduler_file: str | None = None,
        nthreads: int | None = None,
        loop: IOLoop | None = None,  # Deprecated
        local_dir: None = None,  # Deprecated, use local_directory instead
        local_directory: str | None = None,
        services: dict | None = None,
        name: Any | None = None,
        reconnect: bool | None = None,
        executor: Executor | dict[str, Executor] | Literal["offload"] | None = None,
        resources: dict[str, float] | None = None,
        silence_logs: int | None = None,
        death_timeout: Any | None = None,
        preload: list[str] | None = None,
        preload_argv: list[str] | list[list[str]] | None = None,
        security: Security | dict[str, Any] | None = None,
        contact_address: str | None = None,
        heartbeat_interval: Any = "1s",
        extensions: dict[str, type] | None = None,
        metrics: Mapping[str, Callable[[Worker], Any]] = DEFAULT_METRICS,
        startup_information: Mapping[
            str, Callable[[Worker], Any]
        ] = DEFAULT_STARTUP_INFORMATION,
        interface: str | None = None,
        host: str | None = None,
        port: int | str | Collection[int] | None = None,
        protocol: str | None = None,
        dashboard_address: str | None = None,
        dashboard: bool = False,
        http_prefix: str = "/",
        nanny: Nanny | None = None,
        plugins: tuple[WorkerPlugin, ...] = (),
        low_level_profiler: bool | None = None,
        validate: bool | None = None,
        profile_cycle_interval=None,
        lifetime: Any | None = None,
        lifetime_stagger: Any | None = None,
        lifetime_restart: bool | None = None,
        transition_counter_max: int | Literal[False] = False,
        ###################################
        # Parameters to WorkerMemoryManager
        memory_limit: str | float = "auto",
        # Allow overriding the dict-like that stores the task outputs.
        # This is meant for power users only. See WorkerMemoryManager for details.
        data: (
            MutableMapping[str, Any]  # pre-initialised
            | Callable[[], MutableMapping[str, Any]]  # constructor
            | tuple[
                Callable[..., MutableMapping[str, Any]], dict[str, Any]
            ]  # (constructor, kwargs to constructor)
            | None  # create internally
        ) = None,
        # Deprecated parameters; please use dask config instead.
        memory_target_fraction: float | Literal[False] | None = None,
        memory_spill_fraction: float | Literal[False] | None = None,
        memory_pause_fraction: float | Literal[False] | None = None,
        ###################################
        # Parameters to Server
        **kwargs,
    ):
        if reconnect is not None:
            if reconnect:
                raise ValueError(
                    "The `reconnect=True` option for `Worker` has been removed. "
                    "To improve cluster stability, workers now always shut down in the face of network disconnects. "
                    "For details, or if this is an issue for you, see https://github.com/dask/distributed/issues/6350."
                )
            else:
                warnings.warn(
                    "The `reconnect` argument to `Worker` is deprecated, and will be removed in a future release. "
                    "Worker reconnection is now always disabled, so passing `reconnect=False` is unnecessary. "
                    "See https://github.com/dask/distributed/issues/6350 for details.",
                    DeprecationWarning,
                    stacklevel=2,
                )
        if loop is not None:
            warnings.warn(
                "The `loop` argument to `Worker` is ignored, and will be removed in a future release. "
                "The Worker always binds to the current loop",
                DeprecationWarning,
                stacklevel=2,
            )
        self.nanny = nanny
        self._lock = threading.Lock()

        total_out_connections = dask.config.get(
            "distributed.worker.connections.outgoing"
        )
        self.total_in_connections = dask.config.get(
            "distributed.worker.connections.incoming"
        )

        self.threads = {}

        self.active_threads_lock = threading.Lock()
        self.active_threads = {}
        self.active_keys = set()
        self.profile_keys = defaultdict(profile.create)
        self.profile_keys_history = deque(maxlen=3600)
        self.profile_recent = profile.create()
        self.profile_history = deque(maxlen=3600)

        if validate is None:
            validate = dask.config.get("distributed.scheduler.validate")

        self.incoming_transfer_log = deque(maxlen=100000)
        self.incoming_count = 0
        self.outgoing_transfer_log = deque(maxlen=100000)
        self.outgoing_count = 0
        self.outgoing_current_count = 0
        self.bandwidth = parse_bytes(dask.config.get("distributed.scheduler.bandwidth"))
        self.bandwidth_workers = defaultdict(
            lambda: (0, 0)
        )  # bw/count recent transfers
        self.bandwidth_types = defaultdict(lambda: (0, 0))  # bw/count recent transfers
        self.latency = 0.001
        self._client = None

        if profile_cycle_interval is None:
            profile_cycle_interval = dask.config.get("distributed.worker.profile.cycle")
        profile_cycle_interval = parse_timedelta(profile_cycle_interval, default="ms")
        assert profile_cycle_interval

        self._setup_logging(logger, wsm_logger)

        if local_dir is not None:
            warnings.warn(  # type: ignore[unreachable]
                "The local_dir keyword has moved to local_directory"
            )
            local_directory = local_dir

        if not local_directory:
            local_directory = dask.config.get("temporary-directory") or os.getcwd()

        os.makedirs(local_directory, exist_ok=True)
        local_directory = os.path.join(local_directory, "dask-worker-space")

        with warn_on_duration(
            "1s",
            "Creating scratch directories is taking a surprisingly long time. "
            "This is often due to running workers on a network file system. "
            "Consider specifying a local-directory to point workers to write "
            "scratch data to a local disk.",
        ):
            self._workspace = WorkSpace(os.path.abspath(local_directory))
            self._workdir = self._workspace.new_work_dir(prefix="worker-")
            self.local_directory = self._workdir.dir_path

        if not preload:
            preload = dask.config.get("distributed.worker.preload")
        if not preload_argv:
            preload_argv = dask.config.get("distributed.worker.preload-argv")
        assert preload is not None
        assert preload_argv is not None
        self.preloads = preloading.process_preloads(
            self, preload, preload_argv, file_dir=self.local_directory
        )

        if scheduler_file:
            cfg = json_load_robust(scheduler_file)
            scheduler_addr = cfg["address"]
        elif scheduler_ip is None and dask.config.get("scheduler-address", None):
            scheduler_addr = dask.config.get("scheduler-address")
        elif scheduler_port is None:
            scheduler_addr = coerce_to_address(scheduler_ip)
        else:
            scheduler_addr = coerce_to_address((scheduler_ip, scheduler_port))
        self.contact_address = contact_address

        if protocol is None:
            protocol_address = scheduler_addr.split("://")
            if len(protocol_address) == 2:
                protocol = protocol_address[0]
            assert protocol

        self._start_port = port
        self._start_host = host
        if host:
            # Helpful error message if IPv6 specified incorrectly
            _, host_address = parse_address(host)
            if host_address.count(":") > 1 and not host_address.startswith("["):
                raise ValueError(
                    "Host address with IPv6 must be bracketed like '[::1]'; "
                    f"got {host_address}"
                )
        self._interface = interface
        self._protocol = protocol

        nthreads = nthreads or CPU_COUNT
        if resources is None:
            resources = dask.config.get("distributed.worker.resources")
            assert isinstance(resources, dict)
        self.total_resources = resources.copy()

        self.death_timeout = parse_timedelta(death_timeout)

        self.extensions = {}
        if silence_logs:
            silence_logging(level=silence_logs)

        if isinstance(security, dict):
            security = Security(**security)
        self.security = security or Security()
        assert isinstance(self.security, Security)
        self.connection_args = self.security.get_connection_args("worker")

        self.loop = self.io_loop = IOLoop.current()

        # Common executors always available
        self.executors = {
            "offload": utils._offload_executor,
            "actor": ThreadPoolExecutor(1, thread_name_prefix="Dask-Actor-Threads"),
        }
        if nvml.device_get_count() > 0:
            self.executors["gpu"] = ThreadPoolExecutor(
                1, thread_name_prefix="Dask-GPU-Threads"
            )

        # Find the default executor
        if executor == "offload":
            self.executors["default"] = self.executors["offload"]
        elif isinstance(executor, dict):
            self.executors.update(executor)
        elif executor is not None:
            self.executors["default"] = executor
        if "default" not in self.executors:
            self.executors["default"] = ThreadPoolExecutor(
                thread_name_prefix="Dask-Default-Threads"
            )

        self.batched_stream = BatchedSend(interval="2ms", loop=self.loop)
        self.name = name
        self.scheduler_delay = 0
        self.stream_comms = {}
        self.heartbeat_active = False

        if self.local_directory not in sys.path:
            sys.path.insert(0, self.local_directory)

        self.plugins = {}
        self._pending_plugins = plugins

        self.services = {}
        self.service_specs = services or {}

        self._dashboard_address = dashboard_address
        self._dashboard = dashboard
        self._http_prefix = http_prefix

        self.metrics = dict(metrics) if metrics else {}
        self.startup_information = (
            dict(startup_information) if startup_information else {}
        )

        if low_level_profiler is None:
            low_level_profiler = dask.config.get("distributed.worker.profile.low-level")
        self.low_level_profiler = low_level_profiler

        handlers = {
            "gather": self.gather,
            "run": self.run,
            "run_coroutine": self.run_coroutine,
            "get_data": self.get_data,
            "update_data": self.update_data,
            "free_keys": self._handle_remote_stimulus(FreeKeysEvent),
            "terminate": self.close,
            "ping": pingpong,
            "upload_file": self.upload_file,
            "call_stack": self.get_call_stack,
            "profile": self.get_profile,
            "profile_metadata": self.get_profile_metadata,
            "get_logs": self.get_logs,
            "keys": self.keys,
            "versions": self.versions,
            "actor_execute": self.actor_execute,
            "actor_attribute": self.actor_attribute,
            "plugin-add": self.plugin_add,
            "plugin-remove": self.plugin_remove,
            "get_monitor_info": self.get_monitor_info,
            "benchmark_disk": self.benchmark_disk,
            "benchmark_memory": self.benchmark_memory,
            "benchmark_network": self.benchmark_network,
            "get_story": self.get_story,
        }

        stream_handlers = {
            "close": self.close,
            "cancel-compute": self._handle_remote_stimulus(CancelComputeEvent),
            "acquire-replicas": self._handle_remote_stimulus(AcquireReplicasEvent),
            "compute-task": self._handle_remote_stimulus(ComputeTaskEvent),
            "free-keys": self._handle_remote_stimulus(FreeKeysEvent),
            "remove-replicas": self._handle_remote_stimulus(RemoveReplicasEvent),
            "steal-request": self._handle_remote_stimulus(StealRequestEvent),
            "refresh-who-has": self._handle_remote_stimulus(RefreshWhoHasEvent),
            "worker-status-change": self.handle_worker_status_change,
        }

        ServerNode.__init__(
            self,
            handlers=handlers,
            stream_handlers=stream_handlers,
            connection_args=self.connection_args,
            **kwargs,
        )
        self.memory_manager = WorkerMemoryManager(
            self,
            data=data,
            nthreads=nthreads,
            memory_limit=memory_limit,
            memory_target_fraction=memory_target_fraction,
            memory_spill_fraction=memory_spill_fraction,
            memory_pause_fraction=memory_pause_fraction,
        )
        state = WorkerState(
            nthreads=nthreads,
            data=self.memory_manager.data,
            threads=self.threads,
            plugins=self.plugins,
            resources=self.total_resources,
            total_out_connections=total_out_connections,
            validate=validate,
            transition_counter_max=transition_counter_max,
        )
        BaseWorker.__init__(self, state)

        self.scheduler = self.rpc(scheduler_addr)
        self.execution_state = {
            "scheduler": self.scheduler.address,
            "ioloop": self.loop,
            "worker": self,
        }

        self.heartbeat_interval = parse_timedelta(heartbeat_interval, default="ms")
        # FIXME https://github.com/tornadoweb/tornado/issues/3117
        pc = PeriodicCallback(self.heartbeat, self.heartbeat_interval * 1000)  # type: ignore
        self.periodic_callbacks["heartbeat"] = pc

        pc = PeriodicCallback(lambda: self.batched_send({"op": "keep-alive"}), 60000)
        self.periodic_callbacks["keep-alive"] = pc

        pc = PeriodicCallback(self.find_missing, 1000)
        self.periodic_callbacks["find-missing"] = pc

        self._address = contact_address

        if extensions is None:
            extensions = DEFAULT_EXTENSIONS
        self.extensions = {
            name: extension(self) for name, extension in extensions.items()
        }

        setproctitle("dask-worker [not started]")

        if dask.config.get("distributed.worker.profile.enabled"):
            profile_trigger_interval = parse_timedelta(
                dask.config.get("distributed.worker.profile.interval"), default="ms"
            )
            pc = PeriodicCallback(self.trigger_profile, profile_trigger_interval * 1000)
            self.periodic_callbacks["profile"] = pc

            pc = PeriodicCallback(self.cycle_profile, profile_cycle_interval * 1000)
            self.periodic_callbacks["profile-cycle"] = pc

        if lifetime is None:
            lifetime = dask.config.get("distributed.worker.lifetime.duration")
        lifetime = parse_timedelta(lifetime)

        if lifetime_stagger is None:
            lifetime_stagger = dask.config.get("distributed.worker.lifetime.stagger")
        lifetime_stagger = parse_timedelta(lifetime_stagger)

        if lifetime_restart is None:
            lifetime_restart = dask.config.get("distributed.worker.lifetime.restart")
        self.lifetime_restart = lifetime_restart

        if lifetime:
            lifetime += (random.random() * 2 - 1) * lifetime_stagger
            self.io_loop.call_later(lifetime, self.close_gracefully)
        self.lifetime = lifetime

        Worker._instances.add(self)

    ################
    # Memory manager
    ################
    memory_manager: WorkerMemoryManager

    @property
    def data(self) -> MutableMapping[str, Any]:
        """{task key: task payload} of all completed tasks, whether they were computed
        on this Worker or computed somewhere else and then transferred here over the
        network.

        When using the default configuration, this is a zict buffer that automatically
        spills to disk whenever the target threshold is exceeded.
        If spilling is disabled, it is a plain dict instead.
        It could also be a user-defined arbitrary dict-like passed when initialising
        the Worker or the Nanny.
        Worker logic should treat this opaquely and stick to the MutableMapping API.

        .. note::
           This same collection is also available at ``self.state.data`` and
           ``self.memory_manager.data``.
        """
        return self.memory_manager.data

    # Deprecated attributes moved to self.memory_manager.<name>
    memory_limit = DeprecatedMemoryManagerAttribute()
    memory_target_fraction = DeprecatedMemoryManagerAttribute()
    memory_spill_fraction = DeprecatedMemoryManagerAttribute()
    memory_pause_fraction = DeprecatedMemoryManagerAttribute()
    memory_monitor = DeprecatedMemoryMonitor()

    ###########################
    # State machine accessors #
    ###########################

    # Deprecated attributes moved to self.state.<name>
    actors = DeprecatedWorkerStateAttribute()
    available_resources = DeprecatedWorkerStateAttribute()
    busy_workers = DeprecatedWorkerStateAttribute()
    comm_nbytes = DeprecatedWorkerStateAttribute()
    comm_threshold_bytes = DeprecatedWorkerStateAttribute()
    constrained = DeprecatedWorkerStateAttribute()
    data_needed = DeprecatedWorkerStateAttribute()
    data_needed_per_worker = DeprecatedWorkerStateAttribute()
    executed_count = DeprecatedWorkerStateAttribute()
    executing_count = DeprecatedWorkerStateAttribute()
    generation = DeprecatedWorkerStateAttribute()
    has_what = DeprecatedWorkerStateAttribute()
    in_flight_tasks = DeprecatedWorkerStateAttribute(target="in_flight_tasks_count")
    in_flight_workers = DeprecatedWorkerStateAttribute()
    log = DeprecatedWorkerStateAttribute()
    long_running = DeprecatedWorkerStateAttribute()
    nthreads = DeprecatedWorkerStateAttribute()
    stimulus_log = DeprecatedWorkerStateAttribute()
    stimulus_story = DeprecatedWorkerStateAttribute()
    story = DeprecatedWorkerStateAttribute()
    ready = DeprecatedWorkerStateAttribute()
    tasks = DeprecatedWorkerStateAttribute()
    target_message_size = DeprecatedWorkerStateAttribute()
    total_out_connections = DeprecatedWorkerStateAttribute()
    transition_counter = DeprecatedWorkerStateAttribute()
    transition_counter_max = DeprecatedWorkerStateAttribute()
    validate = DeprecatedWorkerStateAttribute()
    validate_task = DeprecatedWorkerStateAttribute()
    waiting_for_data_count = DeprecatedWorkerStateAttribute()

    ##################
    # Administrative #
    ##################

    def __repr__(self):
        name = f", name: {self.name}" if self.name != self.address_safe else ""
        return (
            f"<{self.__class__.__name__} {self.address_safe!r}{name}, "
            f"status: {self.status.name}, "
            f"stored: {len(self.data)}, "
            f"running: {self.state.executing_count}/{self.state.nthreads}, "
            f"ready: {len(self.state.ready)}, "
            f"comm: {self.state.in_flight_tasks_count}, "
            f"waiting: {self.state.waiting_for_data_count}>"
        )

    @property
    def logs(self):
        return self._deque_handler.deque

    def log_event(self, topic: str | Collection[str], msg: Any) -> None:
        full_msg = {
            "op": "log-event",
            "topic": topic,
            "msg": msg,
        }
        if self.thread_id == threading.get_ident():
            self.batched_send(full_msg)
        else:
            self.loop.add_callback(self.batched_send, full_msg)

    @property
    def worker_address(self):
        """For API compatibility with Nanny"""
        return self.address

    @property
    def executor(self):
        return self.executors["default"]

    @ServerNode.status.setter  # type: ignore
    def status(self, value):
        """Override Server.status to notify the Scheduler of status changes.
        Also handles pausing/unpausing.
        """
        prev_status = self.status
        ServerNode.status.__set__(self, value)
        stimulus_id = f"worker-status-change-{time()}"
        self._send_worker_status_change(stimulus_id)
        if prev_status == Status.running:
            self.handle_stimulus(PauseEvent(stimulus_id=stimulus_id))
        elif value == Status.running:
            self.handle_stimulus(UnpauseEvent(stimulus_id=stimulus_id))

    def _send_worker_status_change(self, stimulus_id: str) -> None:
        self.batched_send(
            {
                "op": "worker-status-change",
                "status": self._status.name,
                "stimulus_id": stimulus_id,
            },
        )

    async def get_metrics(self) -> dict:
        try:
            spilled_memory, spilled_disk = self.data.spilled_total  # type: ignore
        except AttributeError:
            # spilling is disabled
            spilled_memory, spilled_disk = 0, 0

        out = dict(
            executing=self.state.executing_count,
            in_memory=len(self.data),
            ready=len(self.state.ready),
            in_flight=self.state.in_flight_tasks_count,
            bandwidth={
                "total": self.bandwidth,
                "workers": dict(self.bandwidth_workers),
                "types": keymap(typename, self.bandwidth_types),
            },
            spilled_nbytes={
                "memory": spilled_memory,
                "disk": spilled_disk,
            },
            event_loop_interval=self._tick_interval_observed,
        )
        out.update(self.monitor.recent())

        for k, metric in self.metrics.items():
            try:
                result = metric(self)
                if isawaitable(result):
                    result = await result
                # In case of collision, prefer core metrics
                out.setdefault(k, result)
            except Exception:  # TODO: log error once
                pass

        return out

    async def get_startup_information(self):
        result = {}
        for k, f in self.startup_information.items():
            try:
                v = f(self)
                if isawaitable(v):
                    v = await v
                result[k] = v
            except Exception:  # TODO: log error once
                pass

        return result

    def identity(self):
        return {
            "type": type(self).__name__,
            "id": self.id,
            "scheduler": self.scheduler.address,
            "nthreads": self.state.nthreads,
            "memory_limit": self.memory_manager.memory_limit,
        }

    def _to_dict(self, *, exclude: Container[str] = ()) -> dict:
        """Dictionary representation for debugging purposes.
        Not type stable and not intended for roundtrips.

        See also
        --------
        Worker.identity
        Client.dump_cluster_state
        distributed.utils.recursive_to_dict
        """
        info = super()._to_dict(exclude=exclude)
        extra = {
            "status": self.status,
            "logs": self.get_logs(),
            "config": dask.config.config,
            "incoming_transfer_log": self.incoming_transfer_log,
            "outgoing_transfer_log": self.outgoing_transfer_log,
        }
        extra = {k: v for k, v in extra.items() if k not in exclude}
        info.update(extra)
        info.update(self.state._to_dict(exclude=exclude))
        info.update(self.memory_manager._to_dict(exclude=exclude))
        return recursive_to_dict(info, exclude=exclude)

    #####################
    # External Services #
    #####################

    def batched_send(self, msg: dict[str, Any]) -> None:
        """Implements BaseWorker abstract method.

        Send a fire-and-forget message to the scheduler through bulk comms.

        If we're not currently connected to the scheduler, the message will be silently
        dropped!

        See also
        --------
        distributed.worker_state_machine.BaseWorker.batched_send
        """
        if (
            self.batched_stream
            and self.batched_stream.comm
            and not self.batched_stream.comm.closed()
        ):
            self.batched_stream.send(msg)

    async def _register_with_scheduler(self) -> None:
        self.periodic_callbacks["keep-alive"].stop()
        self.periodic_callbacks["heartbeat"].stop()
        start = time()
        if self.contact_address is None:
            self.contact_address = self.address
        logger.info("-" * 49)
        while True:
            try:
                _start = time()
                comm = await connect(self.scheduler.address, **self.connection_args)
                comm.name = "Worker->Scheduler"
                comm._server = weakref.ref(self)
                await comm.write(
                    dict(
                        op="register-worker",
                        reply=False,
                        address=self.contact_address,
                        status=self.status.name,
                        keys=list(self.data),
                        nthreads=self.state.nthreads,
                        name=self.name,
                        nbytes={
                            ts.key: ts.get_nbytes()
                            for ts in self.state.tasks.values()
                            # Only if the task is in memory this is a sensible
                            # result since otherwise it simply submits the
                            # default value
                            if ts.state == "memory"
                        },
                        types={k: typename(v) for k, v in self.data.items()},
                        now=time(),
                        resources=self.total_resources,
                        memory_limit=self.memory_manager.memory_limit,
                        local_directory=self.local_directory,
                        services=self.service_ports,
                        nanny=self.nanny,
                        pid=os.getpid(),
                        versions=get_versions(),
                        metrics=await self.get_metrics(),
                        extra=await self.get_startup_information(),
                        stimulus_id=f"worker-connect-{time()}",
                        server_id=self.id,
                    ),
                    serializers=["msgpack"],
                )
                future = comm.read(deserializers=["msgpack"])

                response = await future
                if response.get("warning"):
                    logger.warning(response["warning"])

                _end = time()
                middle = (_start + _end) / 2
                self._update_latency(_end - start)
                self.scheduler_delay = response["time"] - middle
                # self.status = Status.running
                break
            except OSError:
                logger.info("Waiting to connect to: %26s", self.scheduler.address)
                await asyncio.sleep(0.1)
            except TimeoutError:  # pragma: no cover
                logger.info("Timed out when connecting to scheduler")
        if response["status"] != "OK":
            msg = response["message"] if "message" in response else repr(response)
            logger.error(f"Unable to connect to scheduler: {msg}")
            raise ValueError(f"Unexpected response from register: {response!r}")
        else:
            await asyncio.gather(
                *(
                    self.plugin_add(name=name, plugin=plugin)
                    for name, plugin in response["worker-plugins"].items()
                )
            )

            logger.info("        Registered to: %26s", self.scheduler.address)
            logger.info("-" * 49)

        self.batched_stream.start(comm)
        self.periodic_callbacks["keep-alive"].start()
        self.periodic_callbacks["heartbeat"].start()
        self.loop.add_callback(self.handle_scheduler, comm)

    def _update_latency(self, latency: float) -> None:
        self.latency = latency * 0.05 + self.latency * 0.95
        if self.digests is not None:
            self.digests["latency"].add(latency)

    async def heartbeat(self) -> None:
        if self.heartbeat_active:
            logger.debug("Heartbeat skipped: channel busy")
            return
        self.heartbeat_active = True
        logger.debug("Heartbeat: %s", self.address)
        try:
            start = time()
            response = await retry_operation(
                self.scheduler.heartbeat_worker,
                address=self.contact_address,
                now=start,
                metrics=await self.get_metrics(),
                executing={
                    key: start - self.state.tasks[key].start_time
                    for key in self.active_keys
                    if key in self.state.tasks
                },
                extensions={
                    name: extension.heartbeat()
                    for name, extension in self.extensions.items()
                    if hasattr(extension, "heartbeat")
                },
            )
            end = time()
            middle = (start + end) / 2

            self._update_latency(end - start)

            if response["status"] == "missing":
                # Scheduler thought we left. Reconnection is not supported, so just shut down.
                logger.error(
                    f"Scheduler was unaware of this worker {self.address!r}. Shutting down."
                )
                # Something is out of sync; have the nanny restart us if possible.
                await self.close(nanny=False)
                return

            self.scheduler_delay = response["time"] - middle
            self.periodic_callbacks["heartbeat"].callback_time = (
                response["heartbeat-interval"] * 1000
            )
            self.bandwidth_workers.clear()
            self.bandwidth_types.clear()
        except CommClosedError:
            logger.warning("Heartbeat to scheduler failed", exc_info=True)
            await self.close()
        except OSError as e:
            # Scheduler is gone. Respect distributed.comm.timeouts.connect
            if "Timed out trying to connect" in str(e):
                logger.info("Timed out while trying to connect during heartbeat")
                await self.close()
            else:
                logger.exception(e)
                raise e
        finally:
            self.heartbeat_active = False

    @fail_hard
    async def handle_scheduler(self, comm: Comm) -> None:
        await self.handle_stream(comm)
        logger.info(
            "Connection to scheduler broken. Closing without reporting. ID: %s Address %s Status: %s",
            self.id,
            self.address,
            self.status,
        )
        await self.close()

    async def upload_file(
        self, filename: str, data: str | bytes, load: bool = True
    ) -> dict[str, Any]:
        out_filename = os.path.join(self.local_directory, filename)

        def func(data):
            if isinstance(data, str):
                data = data.encode()
            with open(out_filename, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            return data

        if len(data) < 10000:
            data = func(data)
        else:
            data = await offload(func, data)

        if load:
            try:
                import_file(out_filename)
                cache_loads.data.clear()
            except Exception as e:
                logger.exception(e)
                raise e

        return {"status": "OK", "nbytes": len(data)}

    def keys(self) -> list[str]:
        return list(self.data)

    async def gather(self, who_has: dict[str, list[str]]) -> dict[str, Any]:
        who_has = {
            k: [coerce_to_address(addr) for addr in v]
            for k, v in who_has.items()
            if k not in self.data
        }
        result, missing_keys, missing_workers = await gather_from_workers(
            who_has, rpc=self.rpc, who=self.address
        )
        self.update_data(data=result, report=False)
        if missing_keys:
            logger.warning(
                "Could not find data: %s on workers: %s (who_has: %s)",
                missing_keys,
                missing_workers,
                who_has,
            )
            return {"status": "partial-fail", "keys": missing_keys}
        else:
            return {"status": "OK"}

    def get_monitor_info(
        self, recent: bool = False, start: float = 0
    ) -> dict[str, Any]:
        result = dict(
            range_query=(
                self.monitor.recent()
                if recent
                else self.monitor.range_query(start=start)
            ),
            count=self.monitor.count,
            last_time=self.monitor.last_time,
        )
        if nvml.device_get_count() > 0:
            result["gpu_name"] = self.monitor.gpu_name
            result["gpu_memory_total"] = self.monitor.gpu_memory_total
        return result

    #############
    # Lifecycle #
    #############

    async def start_unsafe(self):

        await super().start_unsafe()

        enable_gc_diagnosis()

        ports = parse_ports(self._start_port)
        for port in ports:
            start_address = address_from_user_args(
                host=self._start_host,
                port=port,
                interface=self._interface,
                protocol=self._protocol,
                security=self.security,
            )
            kwargs = self.security.get_listen_args("worker")
            if self._protocol in ("tcp", "tls"):
                kwargs = kwargs.copy()
                kwargs["default_host"] = get_ip(
                    get_address_host(self.scheduler.address)
                )
            try:
                await self.listen(start_address, **kwargs)
            except OSError as e:
                if len(ports) > 1 and e.errno == errno.EADDRINUSE:
                    continue
                else:
                    raise
            else:
                self._start_address = start_address
                break
        else:
            raise ValueError(
                f"Could not start Worker on host {self._start_host} "
                f"with port {self._start_port}"
            )

        # Start HTTP server associated with this Worker node
        routes = get_handlers(
            server=self,
            modules=dask.config.get("distributed.worker.http.routes"),
            prefix=self._http_prefix,
        )
        self.start_http_server(routes, self._dashboard_address)
        if self._dashboard:
            try:
                import distributed.dashboard.worker
            except ImportError:
                logger.debug("To start diagnostics web server please install Bokeh")
            else:
                distributed.dashboard.worker.connect(
                    self.http_application,
                    self.http_server,
                    self,
                    prefix=self._http_prefix,
                )
        self.ip = get_address_host(self.address)

        if self.name is None:
            self.name = self.address

        for preload in self.preloads:
            try:
                await preload.start()
            except Exception:
                logger.exception("Failed to start preload")

        # Services listen on all addresses
        # Note Nanny is not a "real" service, just some metadata
        # passed in service_ports...
        self.start_services(self.ip)

        try:
            listening_address = "%s%s:%d" % (self.listener.prefix, self.ip, self.port)
        except Exception:
            listening_address = f"{self.listener.prefix}{self.ip}"

        logger.info("      Start worker at: %26s", self.address)
        logger.info("         Listening to: %26s", listening_address)
        for k, v in self.service_ports.items():
            logger.info("  {:>16} at: {:>26}".format(k, self.ip + ":" + str(v)))
        logger.info("Waiting to connect to: %26s", self.scheduler.address)
        logger.info("-" * 49)
        logger.info("              Threads: %26d", self.state.nthreads)
        if self.memory_manager.memory_limit:
            logger.info(
                "               Memory: %26s",
                format_bytes(self.memory_manager.memory_limit),
            )
        logger.info("      Local Directory: %26s", self.local_directory)

        setproctitle("dask-worker [%s]" % self.address)

        plugins_msgs = await asyncio.gather(
            *(
                self.plugin_add(plugin=plugin, catch_errors=False)
                for plugin in self._pending_plugins
            ),
            return_exceptions=True,
        )
        plugins_exceptions = [msg for msg in plugins_msgs if isinstance(msg, Exception)]
        if len(plugins_exceptions) >= 1:
            if len(plugins_exceptions) > 1:
                logger.error(
                    "Multiple plugin exceptions raised. All exceptions will be logged, the first is raised."
                )
                for exc in plugins_exceptions:
                    logger.error(repr(exc))
            raise plugins_exceptions[0]

        self._pending_plugins = ()
        self.state.address = self.address
        await self._register_with_scheduler()
        self.start_periodic_callbacks()
        return self

    @log_errors
    async def close(  # type: ignore
        self,
        timeout: float = 30,
        executor_wait: bool = True,
        nanny: bool = True,
    ) -> str | None:
        """Close the worker

        Close asynchronous operations running on the worker, stop all executors and
        comms. If requested, this also closes the nanny.

        Parameters
        ----------
        timeout : float, default 30
            Timeout in seconds for shutting down individual instructions
        executor_wait : bool, default True
            If True, shut down executors synchronously, otherwise asynchronously
        nanny : bool, default True
            If True, close the nanny

        Returns
        -------
        str | None
            None if worker already in closing state or failed, "OK" otherwise
        """
        # FIXME: The worker should not be allowed to close the nanny. Ownership
        # is the other way round. If an external caller wants to close
        # nanny+worker, the nanny must be notified first. ==> Remove kwarg
        # nanny, see also Scheduler.retire_workers
        if self.status in (Status.closed, Status.closing, Status.failed):
            await self.finished()
            return None

        if self.status == Status.init:
            # If the worker is still in startup/init and is started by a nanny,
            # this means the nanny itself is not up, yet. If the Nanny isn't up,
            # yet, it's server will not accept any incoming RPC requests and
            # will block until the startup is finished.
            # Therefore, this worker trying to communicate with the Nanny during
            # startup is not possible and we cannot close it.
            # In this case, the Nanny will automatically close after inspecting
            # the worker status
            nanny = False

        disable_gc_diagnosis()

        try:
            logger.info("Stopping worker at %s", self.address)
        except ValueError:  # address not available if already closed
            logger.info("Stopping worker")
        if self.status not in WORKER_ANY_RUNNING:
            logger.info("Closed worker has not yet started: %s", self.status)
        if not executor_wait:
            logger.info("Not waiting on executor to close")
        self.status = Status.closing

        # Stop callbacks before giving up control in any `await`.
        # We don't want to heartbeat while closing.
        for pc in self.periodic_callbacks.values():
            pc.stop()

        # Cancel async instructions
        await BaseWorker.close(self, timeout=timeout)

        for preload in self.preloads:
            try:
                await preload.teardown()
            except Exception:
                logger.exception("Failed to tear down preload")

        for extension in self.extensions.values():
            if hasattr(extension, "close"):
                result = extension.close()
                if isawaitable(result):
                    result = await result

        if nanny and self.nanny:
            with self.rpc(self.nanny) as r:
                await r.close_gracefully()

        setproctitle("dask-worker [closing]")

        teardowns = [
            plugin.teardown(self)
            for plugin in self.plugins.values()
            if hasattr(plugin, "teardown")
        ]

        await asyncio.gather(*(td for td in teardowns if isawaitable(td)))

        if self._client:
            # If this worker is the last one alive, clean up the worker
            # initialized clients
            if not any(
                w
                for w in Worker._instances
                if w != self and w.status in WORKER_ANY_RUNNING
            ):
                for c in Worker._initialized_clients:
                    # Regardless of what the client was initialized with
                    # we'll require the result as a future. This is
                    # necessary since the heuristics of asynchronous are not
                    # reliable and we might deadlock here
                    c._asynchronous = True
                    if c.asynchronous:
                        await c.close()
                    else:
                        # There is still the chance that even with us
                        # telling the client to be async, itself will decide
                        # otherwise
                        c.close()

        await self.scheduler.close_rpc()
        self._workdir.release()

        self.stop_services()

        # Give some time for a UCX scheduler to complete closing endpoints
        # before closing self.batched_stream, otherwise the local endpoint
        # may be closed too early and errors be raised on the scheduler when
        # trying to send closing message.
        if self._protocol == "ucx":  # pragma: no cover
            await asyncio.sleep(0.2)

        self.batched_send({"op": "close-stream"})

        if self.batched_stream:
            with suppress(TimeoutError):
                await self.batched_stream.close(timedelta(seconds=timeout))

        for executor in self.executors.values():
            if executor is utils._offload_executor:
                continue  # Never shutdown the offload executor

            def _close():
                executor.shutdown(wait=executor_wait)

            # Waiting for the shutdown can block the event loop causing
            # weird deadlocks particularly if the task that is executing in
            # the thread is waiting for a server reply, e.g. when using
            # worker clients, semaphores, etc.
            try:
                await to_thread(_close)
            except RuntimeError:  # Are we shutting down the process?
                _close()  # Just run it directly

        self.stop()
        await self.rpc.close()

        self.status = Status.closed
        await ServerNode.close(self)

        setproctitle("dask-worker [closed]")
        return "OK"

    async def close_gracefully(self, restart=None):
        """Gracefully shut down a worker

        This first informs the scheduler that we're shutting down, and asks it
        to move our data elsewhere. Afterwards, we close as normal
        """
        if self.status in (Status.closing, Status.closing_gracefully):
            await self.finished()

        if self.status == Status.closed:
            return

        if restart is None:
            restart = self.lifetime_restart

        logger.info("Closing worker gracefully: %s", self.address)
        # Wait for all tasks to leave the worker and don't accept any new ones.
        # Scheduler.retire_workers will set the status to closing_gracefully and push it
        # back to this worker.
        await self.scheduler.retire_workers(
            workers=[self.address],
            close_workers=False,
            remove=False,
            stimulus_id=f"worker-close-gracefully-{time()}",
        )
        await self.close(nanny=not restart)

    async def wait_until_closed(self):
        warnings.warn("wait_until_closed has moved to finished()")
        await self.finished()
        assert self.status == Status.closed

    ################
    # Worker Peers #
    ################

    def send_to_worker(self, address, msg):
        if address not in self.stream_comms:
            bcomm = BatchedSend(interval="1ms", loop=self.loop)
            self.stream_comms[address] = bcomm

            async def batched_send_connect():
                comm = await connect(
                    address, **self.connection_args  # TODO, serialization
                )
                comm.name = "Worker->Worker"
                await comm.write({"op": "connection_stream"})

                bcomm.start(comm)

            self._ongoing_background_tasks.call_soon(batched_send_connect)

        self.stream_comms[address].send(msg)

    async def get_data(
        self, comm, keys=None, who=None, serializers=None, max_connections=None
    ) -> dict | Status:
        start = time()

        if max_connections is None:
            max_connections = self.total_in_connections

        # Allow same-host connections more liberally
        if (
            max_connections
            and comm
            and get_address_host(comm.peer_address) == get_address_host(self.address)
        ):
            max_connections = max_connections * 2

        if self.status == Status.paused:
            max_connections = 1
            throttle_msg = " Throttling outgoing connections because worker is paused."
        else:
            throttle_msg = ""

        if (
            max_connections is not False
            and self.outgoing_current_count >= max_connections
        ):
            logger.debug(
                "Worker %s has too many open connections to respond to data request "
                "from %s (%d/%d).%s",
                self.address,
                who,
                self.outgoing_current_count,
                max_connections,
                throttle_msg,
            )
            return {"status": "busy"}

        self.outgoing_current_count += 1
        data = {k: self.data[k] for k in keys if k in self.data}

        if len(data) < len(keys):
            for k in set(keys) - set(data):
                if k in self.state.actors:
                    from distributed.actor import Actor

                    data[k] = Actor(
                        type(self.state.actors[k]), self.address, k, worker=self
                    )

        msg = {"status": "OK", "data": {k: to_serialize(v) for k, v in data.items()}}
        nbytes = {k: self.state.tasks[k].nbytes for k in data if k in self.state.tasks}
        stop = time()
        if self.digests is not None:
            self.digests["get-data-load-duration"].add(stop - start)
        start = time()

        try:
            compressed = await comm.write(msg, serializers=serializers)
            response = await comm.read(deserializers=serializers)
            assert response == "OK", response
        except OSError:
            logger.exception(
                "failed during get data with %s -> %s",
                self.address,
                who,
            )
            comm.abort()
            raise
        finally:
            self.outgoing_current_count -= 1
        stop = time()
        if self.digests is not None:
            self.digests["get-data-send-duration"].add(stop - start)

        total_bytes = sum(filter(None, nbytes.values()))

        self.outgoing_count += 1
        duration = (stop - start) or 0.5  # windows
        self.outgoing_transfer_log.append(
            {
                "start": start + self.scheduler_delay,
                "stop": stop + self.scheduler_delay,
                "middle": (start + stop) / 2,
                "duration": duration,
                "who": who,
                "keys": nbytes,
                "total": total_bytes,
                "compressed": compressed,
                "bandwidth": total_bytes / duration,
            }
        )

        return Status.dont_reply

    ###################
    # Local Execution #
    ###################

    def update_data(
        self,
        data: dict[str, object],
        report: bool = True,
        stimulus_id: str = None,
    ) -> dict[str, Any]:
        self.handle_stimulus(
            UpdateDataEvent(
                data=data,
                report=report,
                stimulus_id=stimulus_id or f"update-data-{time()}",
            )
        )
        return {"nbytes": {k: sizeof(v) for k, v in data.items()}, "status": "OK"}

    async def set_resources(self, **resources) -> None:
        for r, quantity in resources.items():
            if r in self.total_resources:
                self.state.available_resources[r] += quantity - self.total_resources[r]
            else:
                self.state.available_resources[r] = quantity
            self.total_resources[r] = quantity

        await retry_operation(
            self.scheduler.set_resources,
            resources=self.total_resources,
            worker=self.contact_address,
        )

    @log_errors
    async def plugin_add(
        self,
        plugin: WorkerPlugin | bytes,
        name: str | None = None,
        catch_errors: bool = True,
    ) -> dict[str, Any]:
        if isinstance(plugin, bytes):
            # Note: historically we have accepted duck-typed classes that don't
            # inherit from WorkerPlugin. Don't do `assert isinstance`.
            plugin = cast("WorkerPlugin", pickle.loads(plugin))

        if name is None:
            name = _get_plugin_name(plugin)

        assert name

        if name in self.plugins:
            await self.plugin_remove(name=name)

        self.plugins[name] = plugin

        logger.info("Starting Worker plugin %s" % name)
        if hasattr(plugin, "setup"):
            try:
                result = plugin.setup(worker=self)
                if isawaitable(result):
                    result = await result
            except Exception as e:
                if not catch_errors:
                    raise
                msg = error_message(e)
                return cast("dict[str, Any]", msg)

        return {"status": "OK"}

    @log_errors
    async def plugin_remove(self, name: str) -> dict[str, Any]:
        logger.info(f"Removing Worker plugin {name}")
        try:
            plugin = self.plugins.pop(name)
            if hasattr(plugin, "teardown"):
                result = plugin.teardown(worker=self)
                if isawaitable(result):
                    result = await result
        except Exception as e:
            msg = error_message(e)
            return cast("dict[str, Any]", msg)

        return {"status": "OK"}

    def handle_worker_status_change(self, status: str, stimulus_id: str) -> None:
        new_status = Status.lookup[status]  # type: ignore

        if (
            new_status == Status.closing_gracefully
            and self._status not in WORKER_ANY_RUNNING
        ):
            logger.error(
                "Invalid Worker.status transition: %s -> %s", self._status, new_status
            )
            # Reiterate the current status to the scheduler to restore sync
            self._send_worker_status_change(stimulus_id)
        else:
            # Update status and send confirmation to the Scheduler (see status.setter)
            self.status = new_status

    ###################
    # Task Management #
    ###################

    def _handle_remote_stimulus(
        self, cls: type[StateMachineEvent]
    ) -> Callable[..., None]:
        def _(**kwargs):
            event = cls(**kwargs)
            self.handle_stimulus(event)

        _.__name__ = f"_handle_remote_stimulus({cls.__name__})"
        return _

    @fail_hard
    def _handle_stimulus_from_task(
        self, task: asyncio.Task[StateMachineEvent | None]
    ) -> None:
        """Override BaseWorker method for added validation

        See also
        --------
        distributed.worker_state_machine.BaseWorker._handle_stimulus_from_task
        """
        super()._handle_stimulus_from_task(task)

    @fail_hard
    def handle_stimulus(self, *stims: StateMachineEvent) -> None:
        """Override BaseWorker method for added validation

        See also
        --------
        distributed.worker_state_machine.BaseWorker.handle_stimulus
        distributed.worker_state_machine.WorkerState.handle_stimulus
        """
        try:
            super().handle_stimulus(*stims)
        except Exception as e:
            if hasattr(e, "to_event"):
                topic, msg = e.to_event()  # type: ignore
                self.log_event(topic, msg)
            raise

    def stateof(self, key: str) -> dict[str, Any]:
        ts = self.state.tasks[key]
        return {
            "executing": ts.state == "executing",
            "waiting_for_data": bool(ts.waiting_for_data),
            "heap": key in pluck(1, self.state.ready),
            "data": key in self.data,
        }

    async def get_story(self, keys_or_stimuli: Iterable[str]) -> list[tuple]:
        return self.state.story(*keys_or_stimuli)

    ##########################
    # Dependencies gathering #
    ##########################

    def _get_cause(self, keys: Iterable[str]) -> TaskState:
        """For diagnostics, we want to attach a transfer to a single task. This task is
        typically the next to be executed but since we're fetching tasks for potentially
        many dependents, an exact match is not possible. Additionally, if a key was
        fetched through acquire-replicas, dependents may not be known at all.

        Returns
        -------
        The task to attach startstops of this transfer to
        """
        cause = None
        for key in keys:
            ts = self.state.tasks[key]
            if ts.dependents:
                return next(iter(ts.dependents))
            cause = ts
        assert cause  # Always at least one key
        return cause

    def _update_metrics_received_data(
        self,
        start: float,
        stop: float,
        data: dict[str, object],
        cause: TaskState,
        worker: str,
    ) -> None:

        total_bytes = sum(self.state.tasks[key].get_nbytes() for key in data)

        cause.startstops.append(
            {
                "action": "transfer",
                "start": start + self.scheduler_delay,
                "stop": stop + self.scheduler_delay,
                "source": worker,
            }
        )
        duration = (stop - start) or 0.010
        bandwidth = total_bytes / duration
        self.incoming_transfer_log.append(
            {
                "start": start + self.scheduler_delay,
                "stop": stop + self.scheduler_delay,
                "middle": (start + stop) / 2.0 + self.scheduler_delay,
                "duration": duration,
                "keys": {key: self.state.tasks[key].nbytes for key in data},
                "total": total_bytes,
                "bandwidth": bandwidth,
                "who": worker,
            }
        )
        if total_bytes > 1_000_000:
            self.bandwidth = self.bandwidth * 0.95 + bandwidth * 0.05
            bw, cnt = self.bandwidth_workers[worker]
            self.bandwidth_workers[worker] = (bw + bandwidth, cnt + 1)

            types = set(map(type, data.values()))
            if len(types) == 1:
                [typ] = types
                bw, cnt = self.bandwidth_types[typ]
                self.bandwidth_types[typ] = (bw + bandwidth, cnt + 1)

        if self.digests is not None:
            self.digests["transfer-bandwidth"].add(total_bytes / duration)
            self.digests["transfer-duration"].add(duration)
        self.counters["transfer-count"].add(len(data))
        self.incoming_count += 1

    @fail_hard
    async def gather_dep(
        self,
        worker: str,
        to_gather: Collection[str],
        total_nbytes: int,
        *,
        stimulus_id: str,
    ) -> StateMachineEvent | None:
        """Implements BaseWorker abstract method

        See also
        --------
        distributed.worker_state_machine.BaseWorker.gather_dep
        """
        if self.status not in WORKER_ANY_RUNNING:
            return None

        try:
            self.state.log.append(
                ("request-dep", worker, to_gather, stimulus_id, time())
            )
            logger.debug("Request %d keys from %s", len(to_gather), worker)

            start = time()
            response = await get_data_from_worker(
                self.rpc, to_gather, worker, who=self.address
            )
            stop = time()
            if response["status"] == "busy":
                self.state.log.append(
                    ("busy-gather", worker, to_gather, stimulus_id, time())
                )
                return GatherDepBusyEvent(
                    worker=worker,
                    total_nbytes=total_nbytes,
                    stimulus_id=f"gather-dep-busy-{time()}",
                )

            assert response["status"] == "OK"
            cause = self._get_cause(to_gather)
            self._update_metrics_received_data(
                start=start,
                stop=stop,
                data=response["data"],
                cause=cause,
                worker=worker,
            )
            self.state.log.append(
                ("receive-dep", worker, set(response["data"]), stimulus_id, time())
            )
            return GatherDepSuccessEvent(
                worker=worker,
                total_nbytes=total_nbytes,
                data=response["data"],
                stimulus_id=f"gather-dep-success-{time()}",
            )

        except OSError:
            logger.exception("Worker stream died during communication: %s", worker)
            self.state.log.append(
                ("receive-dep-failed", worker, to_gather, stimulus_id, time())
            )
            return GatherDepNetworkFailureEvent(
                worker=worker,
                total_nbytes=total_nbytes,
                stimulus_id=f"gather-dep-network-failure-{time()}",
            )

        except Exception as e:
            # e.g. data failed to deserialize
            logger.exception(e)
            if self.batched_stream and LOG_PDB:
                import pdb

                pdb.set_trace()

            return GatherDepFailureEvent.from_exception(
                e,
                worker=worker,
                total_nbytes=total_nbytes,
                stimulus_id=f"gather-dep-failure-{time()}",
            )

    async def retry_busy_worker_later(self, worker: str) -> StateMachineEvent | None:
        """Wait some time, then take a peer worker out of busy state.
        Implements BaseWorker abstract method.

        See Also
        --------
        distributed.worker_state_machine.BaseWorker.retry_busy_worker_later
        """
        await asyncio.sleep(0.15)
        return RetryBusyWorkerEvent(
            worker=worker, stimulus_id=f"retry-busy-worker-{time()}"
        )

    @log_errors
    def find_missing(self) -> None:
        self.handle_stimulus(FindMissingEvent(stimulus_id=f"find-missing-{time()}"))

        # This is quite arbitrary but the heartbeat has scaling implemented
        self.periodic_callbacks["find-missing"].callback_time = self.periodic_callbacks[
            "heartbeat"
        ].callback_time

    ################
    # Execute Task #
    ################

    def run(self, comm, function, args=(), wait=True, kwargs=None):
        return run(self, comm, function=function, args=args, kwargs=kwargs, wait=wait)

    def run_coroutine(self, comm, function, args=(), kwargs=None, wait=True):
        return run(self, comm, function=function, args=args, kwargs=kwargs, wait=wait)

    async def actor_execute(
        self,
        actor=None,
        function=None,
        args=(),
        kwargs: dict | None = None,
    ) -> dict[str, Any]:
        kwargs = kwargs or {}
        separate_thread = kwargs.pop("separate_thread", True)
        key = actor
        actor = self.state.actors[key]
        func = getattr(actor, function)
        name = key_split(key) + "." + function

        try:
            if iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            elif separate_thread:
                result = await self.loop.run_in_executor(
                    self.executors["actor"],
                    apply_function_actor,
                    func,
                    args,
                    kwargs,
                    self.execution_state,
                    name,
                    self.active_threads,
                    self.active_threads_lock,
                )
            else:
                result = func(*args, **kwargs)
            return {"status": "OK", "result": to_serialize(result)}
        except Exception as ex:
            return {"status": "error", "exception": to_serialize(ex)}

    def actor_attribute(self, actor=None, attribute=None) -> dict[str, Any]:
        try:
            value = getattr(self.state.actors[actor], attribute)
            return {"status": "OK", "result": to_serialize(value)}
        except Exception as ex:
            return {"status": "error", "exception": to_serialize(ex)}

    async def _maybe_deserialize_task(
        self, ts: TaskState
    ) -> tuple[Callable, tuple, dict[str, Any]]:
        assert ts.run_spec is not None
        start = time()
        # Offload deserializing large tasks
        if sizeof(ts.run_spec) > OFFLOAD_THRESHOLD:
            function, args, kwargs = await offload(_deserialize, *ts.run_spec)
        else:
            function, args, kwargs = _deserialize(*ts.run_spec)
        stop = time()

        if stop - start > 0.010:
            ts.startstops.append(
                {"action": "deserialize", "start": start, "stop": stop}
            )
        return function, args, kwargs

    @fail_hard
    async def execute(self, key: str, *, stimulus_id: str) -> StateMachineEvent | None:
        """Execute a task. Implements BaseWorker abstract method.

        See also
        --------
        distributed.worker_state_machine.BaseWorker.execute
        """
        if self.status in {Status.closing, Status.closed, Status.closing_gracefully}:
            return None
        ts = self.state.tasks.get(key)
        if not ts:
            return None
        if ts.state == "cancelled":
            logger.debug(
                "Trying to execute task %s which is not in executing state anymore",
                ts,
            )
            return AlreadyCancelledEvent(key=ts.key, stimulus_id=stimulus_id)

        try:
            function, args, kwargs = await self._maybe_deserialize_task(ts)
        except Exception as exc:
            logger.error("Could not deserialize task %s", key, exc_info=True)
            return ExecuteFailureEvent.from_exception(
                exc,
                key=key,
                stimulus_id=f"run-spec-deserialize-failed-{time()}",
            )

        try:
            if self.state.validate:
                assert not ts.waiting_for_data
                assert ts.state == "executing"
                assert ts.run_spec is not None

            args2, kwargs2 = self._prepare_args_for_execution(ts, args, kwargs)

            assert ts.annotations is not None
            executor = ts.annotations.get("executor", "default")
            try:
                e = self.executors[executor]
            except KeyError:
                raise ValueError(
                    f"Invalid executor {executor!r}; "
                    f"expected one of: {sorted(self.executors)}"
                )

            self.active_keys.add(key)
            try:
                ts.start_time = time()
                if iscoroutinefunction(function):
                    result = await apply_function_async(
                        function,
                        args2,
                        kwargs2,
                        self.scheduler_delay,
                    )
                elif "ThreadPoolExecutor" in str(type(e)):
                    result = await self.loop.run_in_executor(
                        e,
                        apply_function,
                        function,
                        args2,
                        kwargs2,
                        self.execution_state,
                        key,
                        self.active_threads,
                        self.active_threads_lock,
                        self.scheduler_delay,
                    )
                else:
                    result = await self.loop.run_in_executor(
                        e,
                        apply_function_simple,
                        function,
                        args2,
                        kwargs2,
                        self.scheduler_delay,
                    )
            finally:
                self.active_keys.discard(key)

            self.threads[key] = result["thread"]

            if result["op"] == "task-finished":
                if self.digests is not None:
                    self.digests["task-duration"].add(result["stop"] - result["start"])
                return ExecuteSuccessEvent(
                    key=key,
                    value=result["result"],
                    start=result["start"],
                    stop=result["stop"],
                    nbytes=result["nbytes"],
                    type=result["type"],
                    stimulus_id=f"task-finished-{time()}",
                )

            if isinstance(result["actual-exception"], Reschedule):
                return RescheduleEvent(key=ts.key, stimulus_id=stimulus_id)

            logger.warning(
                "Compute Failed\n"
                "Key:       %s\n"
                "Function:  %s\n"
                "args:      %s\n"
                "kwargs:    %s\n"
                "Exception: %r\n",
                key,
                str(funcname(function))[:1000],
                convert_args_to_str(args2, max_len=1000),
                convert_kwargs_to_str(kwargs2, max_len=1000),
                result["exception_text"],
            )
            return ExecuteFailureEvent.from_exception(
                result,
                key=key,
                start=result["start"],
                stop=result["stop"],
                stimulus_id=f"task-erred-{time()}",
            )

        except Exception as exc:
            logger.error("Exception during execution of task %s.", key, exc_info=True)
            return ExecuteFailureEvent.from_exception(
                exc,
                key=key,
                stimulus_id=f"execute-unknown-error-{time()}",
            )

    def _prepare_args_for_execution(
        self, ts: TaskState, args: tuple, kwargs: dict[str, Any]
    ) -> tuple[tuple, dict[str, Any]]:
        start = time()
        data = {}
        for dep in ts.dependencies:
            k = dep.key
            try:
                data[k] = self.data[k]
            except KeyError:
                from distributed.actor import Actor  # TODO: create local actor

                data[k] = Actor(type(self.state.actors[k]), self.address, k, self)
        args2 = pack_data(args, data, key_types=(bytes, str))
        kwargs2 = pack_data(kwargs, data, key_types=(bytes, str))
        stop = time()
        if stop - start > 0.005:
            ts.startstops.append({"action": "disk-read", "start": start, "stop": stop})
            if self.digests is not None:
                self.digests["disk-load-duration"].add(stop - start)
        return args2, kwargs2

    ##################
    # Administrative #
    ##################
    def cycle_profile(self) -> None:
        now = time() + self.scheduler_delay
        prof, self.profile_recent = self.profile_recent, profile.create()
        self.profile_history.append((now, prof))

        self.profile_keys_history.append((now, dict(self.profile_keys)))
        self.profile_keys.clear()

    def trigger_profile(self) -> None:
        """
        Get a frame from all actively computing threads

        Merge these frames into existing profile counts
        """
        if not self.active_threads:  # hope that this is thread-atomic?
            return
        start = time()
        with self.active_threads_lock:
            active_threads = self.active_threads.copy()
        frames = sys._current_frames()
        frames = {ident: frames[ident] for ident in active_threads}
        llframes = {}
        if self.low_level_profiler:
            llframes = {ident: profile.ll_get_stack(ident) for ident in active_threads}
        for ident, frame in frames.items():
            if frame is not None:
                key = key_split(active_threads[ident])
                llframe = llframes.get(ident)

                state = profile.process(
                    frame, True, self.profile_recent, stop="distributed/worker.py"
                )
                profile.llprocess(llframe, None, state)
                profile.process(
                    frame, True, self.profile_keys[key], stop="distributed/worker.py"
                )

        stop = time()
        if self.digests is not None:
            self.digests["profile-duration"].add(stop - start)

    async def get_profile(
        self,
        start=None,
        stop=None,
        key=None,
        server: bool = False,
    ):
        now = time() + self.scheduler_delay
        if server:
            history = self.io_loop.profile
        elif key is None:
            history = self.profile_history
        else:
            history = [(t, d[key]) for t, d in self.profile_keys_history if key in d]

        if start is None:
            istart = 0
        else:
            istart = bisect.bisect_left(history, (start,))

        if stop is None:
            istop = None
        else:
            istop = bisect.bisect_right(history, (stop,)) + 1
            if istop >= len(history):
                istop = None  # include end

        if istart == 0 and istop is None:
            history = list(history)
        else:
            iistop = len(history) if istop is None else istop
            history = [history[i] for i in range(istart, iistop)]

        prof = profile.merge(*pluck(1, history))

        if not history:
            return profile.create()

        if istop is None and (start is None or start < now):
            if key is None:
                recent = self.profile_recent
            else:
                recent = self.profile_keys[key]
            prof = profile.merge(prof, recent)

        return prof

    async def get_profile_metadata(
        self, start: float = 0, stop: float | None = None
    ) -> dict[str, Any]:
        add_recent = stop is None
        now = time() + self.scheduler_delay
        stop = stop or now
        result = {
            "counts": [
                (t, d["count"]) for t, d in self.profile_history if start < t < stop
            ],
            "keys": [
                (t, {k: d["count"] for k, d in v.items()})
                for t, v in self.profile_keys_history
                if start < t < stop
            ],
        }
        if add_recent:
            result["counts"].append((now, self.profile_recent["count"]))
            result["keys"].append(
                (now, {k: v["count"] for k, v in self.profile_keys.items()})
            )
        return result

    def get_call_stack(self, keys: Collection[str] | None = None) -> dict[str, Any]:
        with self.active_threads_lock:
            sys_frames = sys._current_frames()
            frames = {key: sys_frames[tid] for tid, key in self.active_threads.items()}
        if keys is not None:
            frames = {key: frames[key] for key in keys if key in frames}

        return {key: profile.call_stack(frame) for key, frame in frames.items()}

    async def benchmark_disk(self) -> dict[str, float]:
        return await self.loop.run_in_executor(
            self.executor, benchmark_disk, self.local_directory
        )

    async def benchmark_memory(self) -> dict[str, float]:
        return await self.loop.run_in_executor(self.executor, benchmark_memory)

    async def benchmark_network(self, address: str) -> dict[str, float]:
        return await benchmark_network(rpc=self.rpc, address=address)

    #######################################
    # Worker Clients (advanced workloads) #
    #######################################

    @property
    def client(self) -> Client:
        with self._lock:
            if self._client:
                return self._client
            else:
                return self._get_client()

    def _get_client(self, timeout: float | None = None) -> Client:
        """Get local client attached to this worker

        If no such client exists, create one

        See Also
        --------
        get_client
        """

        if timeout is None:
            timeout = dask.config.get("distributed.comm.timeouts.connect")

        timeout = parse_timedelta(timeout, "s")

        try:
            from distributed.client import default_client

            client = default_client()
        except ValueError:  # no clients found, need to make a new one
            pass
        else:
            # must be lazy import otherwise cyclic import
            from distributed.deploy.cluster import Cluster

            if (
                client.scheduler
                and client.scheduler.address == self.scheduler.address
                # The below conditions should only happen in case a second
                # cluster is alive, e.g. if a submitted task spawned its onwn
                # LocalCluster, see gh4565
                or (
                    isinstance(client._start_arg, str)
                    and client._start_arg == self.scheduler.address
                    or isinstance(client._start_arg, Cluster)
                    and client._start_arg.scheduler_address == self.scheduler.address
                )
            ):
                self._client = client

        if not self._client:
            from distributed.client import Client

            asynchronous = in_async_call(self.loop)
            self._client = Client(
                self.scheduler,
                loop=self.loop,
                security=self.security,
                set_as_default=True,
                asynchronous=asynchronous,
                direct_to_workers=True,
                name="worker",
                timeout=timeout,
            )
            Worker._initialized_clients.add(self._client)
            if not asynchronous:
                assert self._client.status == "running"

        return self._client

    def get_current_task(self) -> str:
        """Get the key of the task we are currently running

        This only makes sense to run within a task

        Examples
        --------
        >>> from dask.distributed import get_worker
        >>> def f():
        ...     return get_worker().get_current_task()

        >>> future = client.submit(f)  # doctest: +SKIP
        >>> future.result()  # doctest: +SKIP
        'f-1234'

        See Also
        --------
        get_worker
        """
        return self.active_threads[threading.get_ident()]

    def validate_state(self) -> None:
        try:
            self.state.validate_state()
        except Exception as e:
            logger.error("Validate state failed", exc_info=e)
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()

            if hasattr(e, "to_event"):
                topic, msg = e.to_event()  # type: ignore
                self.log_event(topic, msg)

            raise


def get_worker() -> Worker:
    """Get the worker currently running this task

    Examples
    --------
    >>> def f():
    ...     worker = get_worker()  # The worker on which this task is running
    ...     return worker.address

    >>> future = client.submit(f)  # doctest: +SKIP
    >>> future.result()  # doctest: +SKIP
    'tcp://127.0.0.1:47373'

    See Also
    --------
    get_client
    worker_client
    """
    try:
        return thread_state.execution_state["worker"]
    except AttributeError:
        try:
            return first(w for w in Worker._instances if w.status in WORKER_ANY_RUNNING)
        except StopIteration:
            raise ValueError("No workers found")


def get_client(address=None, timeout=None, resolve_address=True) -> Client:
    """Get a client while within a task.

    This client connects to the same scheduler to which the worker is connected

    Parameters
    ----------
    address : str, optional
        The address of the scheduler to connect to. Defaults to the scheduler
        the worker is connected to.
    timeout : int or str
        Timeout (in seconds) for getting the Client. Defaults to the
        ``distributed.comm.timeouts.connect`` configuration value.
    resolve_address : bool, default True
        Whether to resolve `address` to its canonical form.

    Returns
    -------
    Client

    Examples
    --------
    >>> def f():
    ...     client = get_client(timeout="10s")
    ...     futures = client.map(lambda x: x + 1, range(10))  # spawn many tasks
    ...     results = client.gather(futures)
    ...     return sum(results)

    >>> future = client.submit(f)  # doctest: +SKIP
    >>> future.result()  # doctest: +SKIP
    55

    See Also
    --------
    get_worker
    worker_client
    secede
    """

    if timeout is None:
        timeout = dask.config.get("distributed.comm.timeouts.connect")

    timeout = parse_timedelta(timeout, "s")

    if address and resolve_address:
        address = comm_resolve_address(address)
    try:
        worker = get_worker()
    except ValueError:  # could not find worker
        pass
    else:
        if not address or worker.scheduler.address == address:
            return worker._get_client(timeout=timeout)

    from distributed.client import Client

    try:
        client = Client.current()  # TODO: assumes the same scheduler
    except ValueError:
        client = None
    if client and (not address or client.scheduler.address == address):
        return client
    elif address:
        return Client(address, timeout=timeout)
    else:
        raise ValueError("No global client found and no address provided")


def secede():
    """
    Have this task secede from the worker's thread pool

    This opens up a new scheduling slot and a new thread for a new task. This
    enables the client to schedule tasks on this node, which is
    especially useful while waiting for other jobs to finish (e.g., with
    ``client.gather``).

    Examples
    --------
    >>> def mytask(x):
    ...     # do some work
    ...     client = get_client()
    ...     futures = client.map(...)  # do some remote work
    ...     secede()  # while that work happens, remove ourself from the pool
    ...     return client.gather(futures)  # return gathered results

    See Also
    --------
    get_client
    get_worker
    """
    worker = get_worker()
    duration = time() - thread_state.start_time
    worker.loop.add_callback(
        worker.handle_stimulus,
        SecedeEvent(
            key=thread_state.key,
            compute_duration=duration,
            stimulus_id=f"secede-{time()}",
        ),
    )


class Reschedule(Exception):
    """Reschedule this task

    Raising this exception will stop the current execution of the task and ask
    the scheduler to reschedule this task, possibly on a different machine.

    This does not guarantee that the task will move onto a different machine.
    The scheduler will proceed through its normal heuristics to determine the
    optimal machine to accept this task.  The machine will likely change if the
    load across the cluster has significantly changed since first scheduling
    the task.
    """


async def get_data_from_worker(
    rpc,
    keys,
    worker,
    who=None,
    max_connections=None,
    serializers=None,
    deserializers=None,
):
    """Get keys from worker

    The worker has a two step handshake to acknowledge when data has been fully
    delivered.  This function implements that handshake.

    See Also
    --------
    Worker.get_data
    Worker.gather_dep
    utils_comm.gather_data_from_workers
    """
    if serializers is None:
        serializers = rpc.serializers
    if deserializers is None:
        deserializers = rpc.deserializers

    async def _get_data():
        comm = await rpc.connect(worker)
        comm.name = "Ephemeral Worker->Worker for gather"
        try:
            response = await send_recv(
                comm,
                serializers=serializers,
                deserializers=deserializers,
                op="get_data",
                keys=keys,
                who=who,
                max_connections=max_connections,
            )
            try:
                status = response["status"]
            except KeyError:  # pragma: no cover
                raise ValueError("Unexpected response", response)
            else:
                if status == "OK":
                    await comm.write("OK")
            return response
        finally:
            rpc.reuse(worker, comm)

    return await retry_operation(_get_data, operation="get_data_from_worker")


job_counter = [0]


cache_loads = LRU(maxsize=100)


def loads_function(bytes_object):
    """Load a function from bytes, cache bytes"""
    if len(bytes_object) < 100000:
        try:
            result = cache_loads[bytes_object]
        except KeyError:
            result = pickle.loads(bytes_object)
            cache_loads[bytes_object] = result
        return result
    return pickle.loads(bytes_object)


def _deserialize(function=None, args=None, kwargs=None, task=NO_VALUE):
    """Deserialize task inputs and regularize to func, args, kwargs"""
    if function is not None:
        function = loads_function(function)
    if args and isinstance(args, bytes):
        args = pickle.loads(args)
    if kwargs and isinstance(kwargs, bytes):
        kwargs = pickle.loads(kwargs)

    if task is not NO_VALUE:
        assert not function and not args and not kwargs
        function = execute_task
        args = (task,)

    return function, args or (), kwargs or {}


def execute_task(task):
    """Evaluate a nested task

    >>> inc = lambda x: x + 1
    >>> execute_task((inc, 1))
    2
    >>> execute_task((sum, [1, 2, (inc, 3)]))
    7
    """
    if istask(task):
        func, args = task[0], task[1:]
        return func(*map(execute_task, args))
    elif isinstance(task, list):
        return list(map(execute_task, task))
    else:
        return task


cache_dumps = LRU(maxsize=100)

_cache_lock = threading.Lock()


def dumps_function(func) -> bytes:
    """Dump a function to bytes, cache functions"""
    try:
        with _cache_lock:
            result = cache_dumps[func]
    except KeyError:
        result = pickle.dumps(func, protocol=4)
        if len(result) < 100000:
            with _cache_lock:
                cache_dumps[func] = result
    except TypeError:  # Unhashable function
        result = pickle.dumps(func, protocol=4)
    return result


def dumps_task(task):
    """Serialize a dask task

    Returns a dict of bytestrings that can each be loaded with ``loads``

    Examples
    --------
    Either returns a task as a function, args, kwargs dict

    >>> from operator import add
    >>> dumps_task((add, 1))  # doctest: +SKIP
    {'function': b'\x80\x04\x95\x00\x8c\t_operator\x94\x8c\x03add\x94\x93\x94.'
     'args': b'\x80\x04\x95\x07\x00\x00\x00K\x01K\x02\x86\x94.'}

    Or as a single task blob if it can't easily decompose the result.  This
    happens either if the task is highly nested, or if it isn't a task at all

    >>> dumps_task(1)  # doctest: +SKIP
    {'task': b'\x80\x04\x95\x03\x00\x00\x00\x00\x00\x00\x00K\x01.'}
    """
    if istask(task):
        if task[0] is apply and not any(map(_maybe_complex, task[2:])):
            d = {"function": dumps_function(task[1]), "args": warn_dumps(task[2])}
            if len(task) == 4:
                d["kwargs"] = warn_dumps(task[3])
            return d
        elif not any(map(_maybe_complex, task[1:])):
            return {"function": dumps_function(task[0]), "args": warn_dumps(task[1:])}
    return to_serialize(task)


_warn_dumps_warned = [False]


def warn_dumps(obj, dumps=pickle.dumps, limit=1e6):
    """Dump an object to bytes, warn if those bytes are large"""
    b = dumps(obj, protocol=4)
    if not _warn_dumps_warned[0] and len(b) > limit:
        _warn_dumps_warned[0] = True
        s = str(obj)
        if len(s) > 70:
            s = s[:50] + " ... " + s[-15:]
        warnings.warn(
            "Large object of size %s detected in task graph: \n"
            "  %s\n"
            "Consider scattering large objects ahead of time\n"
            "with client.scatter to reduce scheduler burden and \n"
            "keep data on workers\n\n"
            "    future = client.submit(func, big_data)    # bad\n\n"
            "    big_future = client.scatter(big_data)     # good\n"
            "    future = client.submit(func, big_future)  # good"
            % (format_bytes(len(b)), s)
        )
    return b


def apply_function(
    function,
    args,
    kwargs,
    execution_state,
    key,
    active_threads,
    active_threads_lock,
    time_delay,
):
    """Run a function, collect information

    Returns
    -------
    msg: dictionary with status, result/error, timings, etc..
    """
    ident = threading.get_ident()
    with active_threads_lock:
        active_threads[ident] = key
    thread_state.start_time = time()
    thread_state.execution_state = execution_state
    thread_state.key = key

    msg = apply_function_simple(function, args, kwargs, time_delay)

    with active_threads_lock:
        del active_threads[ident]
    return msg


def apply_function_simple(
    function,
    args,
    kwargs,
    time_delay,
):
    """Run a function, collect information

    Returns
    -------
    msg: dictionary with status, result/error, timings, etc..
    """
    ident = threading.get_ident()
    start = time()
    try:
        result = function(*args, **kwargs)
    except Exception as e:
        msg = error_message(e)
        msg["op"] = "task-erred"
        msg["actual-exception"] = e
    else:
        msg = {
            "op": "task-finished",
            "status": "OK",
            "result": result,
            "nbytes": sizeof(result),
            "type": type(result) if result is not None else None,
        }
    finally:
        end = time()
    msg["start"] = start + time_delay
    msg["stop"] = end + time_delay
    msg["thread"] = ident
    return msg


async def apply_function_async(
    function,
    args,
    kwargs,
    time_delay,
):
    """Run a function, collect information

    Returns
    -------
    msg: dictionary with status, result/error, timings, etc..
    """
    ident = threading.get_ident()
    start = time()
    try:
        result = await function(*args, **kwargs)
    except Exception as e:
        msg = error_message(e)
        msg["op"] = "task-erred"
        msg["actual-exception"] = e
    else:
        msg = {
            "op": "task-finished",
            "status": "OK",
            "result": result,
            "nbytes": sizeof(result),
            "type": type(result) if result is not None else None,
        }
    finally:
        end = time()
    msg["start"] = start + time_delay
    msg["stop"] = end + time_delay
    msg["thread"] = ident
    return msg


def apply_function_actor(
    function, args, kwargs, execution_state, key, active_threads, active_threads_lock
):
    """Run a function, collect information

    Returns
    -------
    msg: dictionary with status, result/error, timings, etc..
    """
    ident = threading.get_ident()

    with active_threads_lock:
        active_threads[ident] = key

    thread_state.execution_state = execution_state
    thread_state.key = key
    thread_state.actor = True

    result = function(*args, **kwargs)

    with active_threads_lock:
        del active_threads[ident]

    return result


def get_msg_safe_str(msg):
    """Make a worker msg, which contains args and kwargs, safe to cast to str:
    allowing for some arguments to raise exceptions during conversion and
    ignoring them.
    """

    class Repr:
        def __init__(self, f, val):
            self._f = f
            self._val = val

        def __repr__(self):
            return self._f(self._val)

    msg = msg.copy()
    if "args" in msg:
        msg["args"] = Repr(convert_args_to_str, msg["args"])
    if "kwargs" in msg:
        msg["kwargs"] = Repr(convert_kwargs_to_str, msg["kwargs"])
    return msg


def convert_args_to_str(args, max_len: int | None = None) -> str:
    """Convert args to a string, allowing for some arguments to raise
    exceptions during conversion and ignoring them.
    """
    length = 0
    strs = ["" for i in range(len(args))]
    for i, arg in enumerate(args):
        try:
            sarg = repr(arg)
        except Exception:
            sarg = "< could not convert arg to str >"
        strs[i] = sarg
        length += len(sarg) + 2
        if max_len is not None and length > max_len:
            return "({}".format(", ".join(strs[: i + 1]))[:max_len]
    else:
        return "({})".format(", ".join(strs))


def convert_kwargs_to_str(kwargs: dict, max_len: int | None = None) -> str:
    """Convert kwargs to a string, allowing for some arguments to raise
    exceptions during conversion and ignoring them.
    """
    length = 0
    strs = ["" for i in range(len(kwargs))]
    for i, (argname, arg) in enumerate(kwargs.items()):
        try:
            sarg = repr(arg)
        except Exception:
            sarg = "< could not convert arg to str >"
        skwarg = repr(argname) + ": " + sarg
        strs[i] = skwarg
        length += len(skwarg) + 2
        if max_len is not None and length > max_len:
            return "{{{}".format(", ".join(strs[: i + 1]))[:max_len]
    else:
        return "{{{}}}".format(", ".join(strs))


async def run(server, comm, function, args=(), kwargs=None, wait=True):
    kwargs = kwargs or {}
    function = pickle.loads(function)
    is_coro = iscoroutinefunction(function)
    assert wait or is_coro, "Combination not supported"
    if args:
        args = pickle.loads(args)
    if kwargs:
        kwargs = pickle.loads(kwargs)
    if has_arg(function, "dask_worker"):
        kwargs["dask_worker"] = server
    if has_arg(function, "dask_scheduler"):
        kwargs["dask_scheduler"] = server
    logger.info("Run out-of-band function %r", funcname(function))
    try:
        if not is_coro:
            result = function(*args, **kwargs)
        else:
            if wait:
                result = await function(*args, **kwargs)
            else:
                server._ongoing_background_tasks.call_soon(function, *args, **kwargs)
                result = None

    except Exception as e:
        logger.warning(
            "Run Failed\nFunction: %s\nargs:     %s\nkwargs:   %s\n",
            str(funcname(function))[:1000],
            convert_args_to_str(args, max_len=1000),
            convert_kwargs_to_str(kwargs, max_len=1000),
            exc_info=True,
        )

        response = error_message(e)
    else:
        response = {"status": "OK", "result": to_serialize(result)}
    return response


_global_workers = Worker._instances

try:
    if nvml.device_get_count() < 1:
        raise RuntimeError
except (Exception, RuntimeError):
    pass
else:

    async def gpu_metric(worker):
        result = await offload(nvml.real_time)
        return result

    DEFAULT_METRICS["gpu"] = gpu_metric

    def gpu_startup(worker):
        return nvml.one_time()

    DEFAULT_STARTUP_INFORMATION["gpu"] = gpu_startup


def print(*args, **kwargs):
    """Dask print function
    This prints both wherever this function is run, and also in the user's
    client session
    """
    try:
        worker = get_worker()
    except ValueError:
        pass
    else:
        msg = {
            "args": tuple(stringify(arg) for arg in args),
            "kwargs": {k: stringify(v) for k, v in kwargs.items()},
        }
        worker.log_event("print", msg)

    builtins.print(*args, **kwargs)


def warn(*args, **kwargs):
    """Dask warn function
    This raises a warning both wherever this function is run, and also
    in the user's client session
    """
    try:
        worker = get_worker()
    except ValueError:  # pragma: no cover
        pass
    else:
        worker.log_event("warn", {"args": args, "kwargs": kwargs})

    warnings.warn(*args, **kwargs)


def benchmark_disk(
    rootdir: str | None = None,
    sizes: Iterable[str] = ("1 kiB", "100 kiB", "1 MiB", "10 MiB", "100 MiB"),
    duration="1 s",
) -> dict[str, float]:
    """
    Benchmark disk bandwidth

    Returns
    -------
    out: dict
        Maps sizes of outputs to measured bandwidths
    """
    duration = parse_timedelta(duration)

    out = {}
    for size_str in sizes:
        with tmpdir(dir=rootdir) as dir:
            dir = pathlib.Path(dir)
            names = list(map(str, range(100)))
            size = parse_bytes(size_str)

            data = randbytes(size)

            start = time()
            total = 0
            while time() < start + duration:
                with open(dir / random.choice(names), mode="ab") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
                total += size

            out[size_str] = total / (time() - start)
    return out


def benchmark_memory(
    sizes: Iterable[str] = ("2 kiB", "10 kiB", "100 kiB", "1 MiB", "10 MiB"),
    duration="200 ms",
) -> dict[str, float]:
    """
    Benchmark memory bandwidth

    Returns
    -------
    out: dict
        Maps sizes of outputs to measured bandwidths
    """
    duration = parse_timedelta(duration)
    out = {}
    for size_str in sizes:
        size = parse_bytes(size_str)
        data = randbytes(size)

        start = time()
        total = 0
        while time() < start + duration:
            _ = data[:-1]
            del _
            total += size

        out[size_str] = total / (time() - start)
    return out


async def benchmark_network(
    address: str,
    rpc: ConnectionPool | Callable[[str], RPCType],
    sizes: Iterable[str] = ("1 kiB", "10 kiB", "100 kiB", "1 MiB", "10 MiB", "50 MiB"),
    duration="1 s",
) -> dict[str, float]:
    """
    Benchmark network communications to another worker

    Returns
    -------
    out: dict
        Maps sizes of outputs to measured bandwidths
    """

    duration = parse_timedelta(duration)
    out = {}
    async with rpc(address) as r:
        for size_str in sizes:
            size = parse_bytes(size_str)
            data = to_serialize(randbytes(size))

            start = time()
            total = 0
            while time() < start + duration:
                await r.echo(data=data)
                total += size * 2

            out[size_str] = total / (time() - start)
    return out

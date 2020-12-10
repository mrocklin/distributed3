import asyncio
import bisect
from collections import defaultdict, deque, namedtuple
from collections.abc import MutableMapping
from contextlib import suppress
from datetime import timedelta
import errno
from functools import partial
import heapq
from inspect import isawaitable
import logging
import os
from pickle import PicklingError
import random
import threading
import sys
import uuid
import warnings
import weakref

import dask
from dask.core import istask
from dask.compatibility import apply
from dask.utils import format_bytes, funcname
from dask.system import CPU_COUNT

from tlz import pluck, merge, first, keymap
from tornado import gen
from tornado.ioloop import IOLoop, PeriodicCallback

from . import profile, comm, system
from .batched import BatchedSend
from .comm import get_address_host, connect
from .comm.addressing import address_from_user_args
from .comm.utils import OFFLOAD_THRESHOLD
from .core import error_message, CommClosedError, send_recv, pingpong, coerce_to_address
from .diskutils import WorkSpace
from .http import get_handlers
from .metrics import time
from .node import ServerNode
from . import preloading
from .proctitle import setproctitle
from .protocol import pickle, to_serialize, deserialize_bytes, serialize_bytelist
from .pubsub import PubSubWorkerExtension
from .security import Security
from .sizeof import safe_sizeof as sizeof
from .threadpoolexecutor import ThreadPoolExecutor, secede as tpe_secede
from .utils import (
    get_ip,
    typename,
    has_arg,
    _maybe_complex,
    log_errors,
    import_file,
    silence_logging,
    thread_state,
    json_load_robust,
    key_split,
    offload,
    parse_bytes,
    parse_timedelta,
    parse_ports,
    iscoroutinefunction,
    warn_on_duration,
    LRU,
    TimeoutError,
)
from .utils_comm import pack_data, gather_from_workers, retry_operation
from .utils_perf import ThrottledGC, enable_gc_diagnosis, disable_gc_diagnosis
from .versions import get_versions

from .core import Status

logger = logging.getLogger(__name__)

LOG_PDB = dask.config.get("distributed.admin.pdb-on-err")

no_value = "--no-value-sentinel--"

IN_PLAY = ("waiting", "ready", "executing", "long-running")
PENDING = ("speculative", "waiting", "ready", "constrained")
PROCESSING = ("waiting", "ready", "constrained", "executing", "long-running")
READY = ("ready", "constrained")


DEFAULT_EXTENSIONS = [PubSubWorkerExtension]

DEFAULT_METRICS = {}

DEFAULT_STARTUP_INFORMATION = {}

DEFAULT_DATA_SIZE = parse_bytes(
    dask.config.get("distributed.scheduler.default-data-size")
)

SerializedTask = namedtuple("SerializedTask", ["function", "args", "kwargs", "task"])


class TaskState:
    """Holds volatile state relating to an individual Dask task


    * **dependencies**: ``set(TaskState instances)``
        The data needed by this key to run
    * **dependents**: ``set(TaskState instances)``
        The keys that use this dependency
    * **duration**: ``float``
        Expected duration the a task
    * **priority**: ``tuple``
        The priority this task given by the scheduler.  Determines run order.
    * **state**: ``str``
        The current state of the task. One of ["waiting", "ready", "executing",
        "memory", "flight", "long-running", "rescheduled", "error"]
    * **who_has**: ``set(worker)``
        Workers that we believe have this data
    * **coming_from**: ``str``
        The worker that current task data is coming from if task is in flight
    * **waiting_for_data**: ``set(keys of dependencies)``
        A dynamic verion of dependencies.  All dependencies that we still don't
        have for a particular key.
    * **resource_restrictions**: ``{str: number}``
        Abstract resources required to run a task
    * **exception**: ``str``
        The exception caused by running a task if it erred
    * **traceback**: ``str``
        The exception caused by running a task if it erred
    * **type**: ``type``
        The type of a particular piece of data
    * **suspicious_count**: ``int``
        The number of times a dependency has not been where we expected it
    * **startstops**: ``[{startstop}]``
        Log of transfer, load, and compute times for a task
    * **start_time**: ``float``
        Time at which task begins running
    * **stop_time**: ``float``
        Time at which task finishes running
    * **metadata**: ``dict``
        Metadata related to task. Stored metadata should be msgpack
        serializable (e.g. int, string, list, dict).
    * **nbytes**: ``int``
        The size of a particular piece of data

    Parameters
    ----------
    key: str
    runspec: SerializedTask
        A named tuple containing the ``function``, ``args``, ``kwargs`` and
        ``task`` associated with this `TaskState` instance. This defaults to
        ``None`` and can remain empty if it is a dependency that this worker
        will receive from another worker.

    """

    def __init__(self, key, runspec=None):
        assert key is not None
        self.key = key
        self.runspec = runspec
        self.dependencies = set()
        self.dependents = set()
        self.duration = None
        self.priority = None
        self.state = None
        self.who_has = set()
        self.coming_from = None
        self.waiting_for_data = set()
        self.resource_restrictions = None
        self.exception = None
        self.traceback = None
        self.type = None
        self.suspicious_count = 0
        self.startstops = list()
        self.start_time = None
        self.stop_time = None
        self.metadata = {}
        self.nbytes = None

    def __repr__(self):
        return "<Task %r %s>" % (self.key, self.state)

    def get_nbytes(self) -> int:
        nbytes = self.nbytes
        return nbytes if nbytes is not None else DEFAULT_DATA_SIZE


class Worker(ServerNode):
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

    The rest of this docstring is about the internal state the the worker uses
    to manage and track internal computations.

    **State**

    **Informational State**

    These attributes don't change significantly during execution.

    * **nthreads:** ``int``:
        Number of nthreads used by this worker process
    * **executor:** ``concurrent.futures.ThreadPoolExecutor``:
        Executor used to perform computation
        This can also be the string "offload" in which case this uses the same
        thread pool used for offloading communications.  This results in the
        same thread being used for deserialization and computation.
    * **local_directory:** ``path``:
        Path on local machine to store temporary files
    * **scheduler:** ``rpc``:
        Location of scheduler.  See ``.ip/.port`` attributes.
    * **name:** ``string``:
        Alias
    * **services:** ``{str: Server}``:
        Auxiliary web servers running on this worker
    * **service_ports:** ``{str: port}``:
    * **total_out_connections**: ``int``
        The maximum number of concurrent outgoing requests for data
    * **total_in_connections**: ``int``
        The maximum number of concurrent incoming requests for data
    * **total_comm_nbytes**: ``int``
    * **batched_stream**: ``BatchedSend``
        A batched stream along which we communicate to the scheduler
    * **log**: ``[(message)]``
        A structured and queryable log.  See ``Worker.story``

    **Volatile State**

    These attributes track the progress of tasks that this worker is trying to
    complete.  In the descriptions below a ``key`` is the name of a task that
    we want to compute and ``dep`` is the name of a piece of dependent data
    that we want to collect from others.

    * **tasks**: ``{key: TaskState}``
        The tasks currently executing on this worker (and any dependencies of those tasks)
    * **data:** ``{key: object}``:
        Prefer using the **host** attribute instead of this, unless
        memory_limit and at least one of memory_target_fraction or
        memory_spill_fraction values are defined, in that case, this attribute
        is a zict.Buffer, from which information on LRU cache can be queried.
    * **data.memory:** ``{key: object}``:
        Dictionary mapping keys to actual values stored in memory. Only
        available if condition for **data** being a zict.Buffer is met.
    * **data.disk:** ``{key: object}``:
        Dictionary mapping keys to actual values stored on disk. Only
        available if condition for **data** being a zict.Buffer is met.
    * **data_needed**: deque(keys)
        The keys whose data we still lack, arranged in a deque
    * **ready**: [keys]
        Keys that are ready to run.  Stored in a LIFO stack
    * **constrained**: [keys]
        Keys for which we have the data to run, but are waiting on abstract
        resources like GPUs.  Stored in a FIFO deque
    * **executing_count**: ``int``
        A count of tasks currently executing on this worker
    * **executed_count**: int
        A number of tasks that this worker has run in its lifetime
    * **long_running**: {keys}
        A set of keys of tasks that are running and have started their own
        long-running clients.
    * **has_what**: ``{worker: {deps}}``
        The data that we care about that we think a worker has
    * **pending_data_per_worker**: ``{worker: [dep]}``
        The data on each worker that we still want, prioritized as a deque
    * **in_flight_tasks**: ``int``
        A count of the number of tasks that are coming to us in current
        peer-to-peer connections
    * **in_flight_workers**: ``{worker: {task}}``
        The workers from which we are currently gathering data and the
        dependencies we expect from those connections
    * **comm_bytes**: ``int``
        The total number of bytes in flight
    * **threads**: ``{key: int}``
        The ID of the thread on which the task ran
    * **active_threads**: ``{int: key}``
        The keys currently running on active threads
    * **waiting_for_data_count**: ``int``
        A count of how many tasks are currently waiting for data


    Parameters
    ----------
    scheduler_ip: str
    scheduler_port: int
    ip: str, optional
    data: MutableMapping, type, None
        The object to use for storage, builds a disk-backed LRU dict by default
    nthreads: int, optional
    loop: tornado.ioloop.IOLoop
    local_directory: str, optional
        Directory where we place local resources
    name: str, optional
    memory_limit: int, float, string
        Number of bytes of memory that this worker should use.
        Set to zero for no limit.  Set to 'auto' to calculate
        as system.MEMORY_LIMIT * min(1, nthreads / total_cores)
        Use strings or numbers like 5GB or 5e9
    memory_target_fraction: float
        Fraction of memory to try to stay beneath
    memory_spill_fraction: float
        Fraction of memory at which we start spilling to disk
    memory_pause_fraction: float
        Fraction of memory at which we stop running new tasks
    executor: concurrent.futures.Executor
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

    _instances = weakref.WeakSet()

    def __init__(
        self,
        scheduler_ip=None,
        scheduler_port=None,
        scheduler_file=None,
        ncores=None,
        nthreads=None,
        loop=None,
        local_dir=None,
        local_directory=None,
        services=None,
        service_ports=None,
        service_kwargs=None,
        name=None,
        reconnect=True,
        memory_limit="auto",
        executor=None,
        resources=None,
        silence_logs=None,
        death_timeout=None,
        preload=None,
        preload_argv=None,
        security=None,
        contact_address=None,
        memory_monitor_interval="200ms",
        extensions=None,
        metrics=DEFAULT_METRICS,
        startup_information=DEFAULT_STARTUP_INFORMATION,
        data=None,
        interface=None,
        host=None,
        port=None,
        protocol=None,
        dashboard_address=None,
        dashboard=False,
        http_prefix="/",
        nanny=None,
        plugins=(),
        low_level_profiler=dask.config.get("distributed.worker.profile.low-level"),
        validate=None,
        profile_cycle_interval=None,
        lifetime=None,
        lifetime_stagger=None,
        lifetime_restart=None,
        **kwargs,
    ):
        self.tasks = dict()
        self.waiting_for_data_count = 0
        self.has_what = defaultdict(set)
        self.pending_data_per_worker = defaultdict(deque)
        self.nanny = nanny
        self._lock = threading.Lock()

        self.data_needed = deque()  # TODO: replace with heap?

        self.in_flight_tasks = 0
        self.in_flight_workers = dict()
        self.total_out_connections = dask.config.get(
            "distributed.worker.connections.outgoing"
        )
        self.total_in_connections = dask.config.get(
            "distributed.worker.connections.incoming"
        )
        self.total_comm_nbytes = 10e6
        self.comm_nbytes = 0
        self._missing_dep_flight = set()

        self.threads = dict()

        self.active_threads_lock = threading.Lock()
        self.active_threads = dict()
        self.profile_keys = defaultdict(profile.create)
        self.profile_keys_history = deque(maxlen=3600)
        self.profile_recent = profile.create()
        self.profile_history = deque(maxlen=3600)

        self.generation = 0

        self.ready = list()
        self.constrained = deque()
        self.executing_count = 0
        self.executed_count = 0
        self.long_running = set()

        self.recent_messages_log = deque(
            maxlen=dask.config.get("distributed.comm.recent-messages-log-length")
        )
        self.target_message_size = 50e6  # 50 MB

        self.log = deque(maxlen=100000)
        if validate is None:
            validate = dask.config.get("distributed.scheduler.validate")
        self.validate = validate

        self._transitions = {
            ("waiting", "ready"): self.transition_waiting_ready,
            ("waiting", "memory"): self.transition_waiting_done,
            ("waiting", "error"): self.transition_waiting_done,
            ("waiting", "flight"): self.transition_waiting_flight,
            ("ready", "executing"): self.transition_ready_executing,
            ("ready", "memory"): self.transition_ready_memory,
            ("constrained", "executing"): self.transition_constrained_executing,
            ("executing", "memory"): self.transition_executing_done,
            ("executing", "error"): self.transition_executing_done,
            ("executing", "rescheduled"): self.transition_executing_done,
            ("executing", "long-running"): self.transition_executing_long_running,
            ("long-running", "error"): self.transition_executing_done,
            ("long-running", "memory"): self.transition_executing_done,
            ("long-running", "rescheduled"): self.transition_executing_done,
            ("flight", "memory"): self.transition_flight_memory,
            ("flight", "ready"): self.transition_flight_memory,
            ("flight", "waiting"): self.transition_flight_waiting,
            ("speculative", "ready"): self.transition_speculative_ready,
        }

        self.incoming_transfer_log = deque(maxlen=100000)
        self.incoming_count = 0
        self.outgoing_transfer_log = deque(maxlen=100000)
        self.outgoing_count = 0
        self.outgoing_current_count = 0
        self.repetitively_busy = 0
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

        self._setup_logging(logger)

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

        # Target interface on which we contact the scheduler by default
        # TODO: it is unfortunate that we special-case inproc here
        if not host and not interface and not scheduler_addr.startswith("inproc://"):
            host = get_ip(get_address_host(scheduler_addr.split("://")[-1]))

        self._start_port = port
        self._start_host = host
        self._interface = interface
        self._protocol = protocol

        if ncores is not None:
            warnings.warn("the ncores= parameter has moved to nthreads=")
            nthreads = ncores

        self.nthreads = nthreads or CPU_COUNT
        self.total_resources = resources or {}
        self.available_resources = (resources or {}).copy()
        self.death_timeout = parse_timedelta(death_timeout)

        self.extensions = dict()
        if silence_logs:
            silence_logging(level=silence_logs)

        if local_dir is not None:
            warnings.warn("The local_dir keyword has moved to local_directory")
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

        if preload is None:
            preload = dask.config.get("distributed.worker.preload")
        if preload_argv is None:
            preload_argv = dask.config.get("distributed.worker.preload-argv")
        self.preloads = preloading.process_preloads(
            self, preload, preload_argv, file_dir=self.local_directory
        )

        if isinstance(security, dict):
            security = Security(**security)
        self.security = security or Security()
        assert isinstance(self.security, Security)
        self.connection_args = self.security.get_connection_args("worker")

        self.memory_limit = parse_memory_limit(memory_limit, self.nthreads)

        self.paused = False

        if "memory_target_fraction" in kwargs:
            self.memory_target_fraction = kwargs.pop("memory_target_fraction")
        else:
            self.memory_target_fraction = dask.config.get(
                "distributed.worker.memory.target"
            )
        if "memory_spill_fraction" in kwargs:
            self.memory_spill_fraction = kwargs.pop("memory_spill_fraction")
        else:
            self.memory_spill_fraction = dask.config.get(
                "distributed.worker.memory.spill"
            )
        if "memory_pause_fraction" in kwargs:
            self.memory_pause_fraction = kwargs.pop("memory_pause_fraction")
        else:
            self.memory_pause_fraction = dask.config.get(
                "distributed.worker.memory.pause"
            )

        if isinstance(data, MutableMapping):
            self.data = data
        elif callable(data):
            self.data = data()
        elif isinstance(data, tuple):
            self.data = data[0](**data[1])
        elif self.memory_limit and (
            self.memory_target_fraction or self.memory_spill_fraction
        ):
            try:
                from zict import Buffer, File, Func
            except ImportError:
                raise ImportError(
                    "Please `python -m pip install zict` for spill-to-disk workers"
                )
            path = os.path.join(self.local_directory, "storage")
            storage = Func(
                partial(serialize_bytelist, on_error="raise"),
                deserialize_bytes,
                File(path),
            )
            target = (
                int(float(self.memory_limit) * self.memory_target_fraction)
                or sys.maxsize
            )
            self.data = Buffer({}, storage, target, weight)
            self.data.memory = self.data.fast
            self.data.disk = self.data.slow
        else:
            self.data = dict()

        self.actors = {}
        self.loop = loop or IOLoop.current()
        self.reconnect = reconnect
        if executor == "offload":
            from distributed.utils import _offload_executor as executor
        self.executor = executor or ThreadPoolExecutor(
            self.nthreads, thread_name_prefix="Dask-Worker-Threads'"
        )
        self.actor_executor = ThreadPoolExecutor(
            1, thread_name_prefix="Dask-Actor-Threads"
        )
        self.batched_stream = BatchedSend(interval="2ms", loop=self.loop)
        self.name = name
        self.scheduler_delay = 0
        self.stream_comms = dict()
        self.heartbeat_active = False
        self._ipython_kernel = None

        if self.local_directory not in sys.path:
            sys.path.insert(0, self.local_directory)

        self.services = {}
        self.service_specs = services or {}

        self._dashboard_address = dashboard_address
        self._dashboard = dashboard
        self._http_prefix = http_prefix

        self.metrics = dict(metrics) if metrics else {}
        self.startup_information = (
            dict(startup_information) if startup_information else {}
        )

        self.low_level_profiler = low_level_profiler

        handlers = {
            "gather": self.gather,
            "run": self.run,
            "run_coroutine": self.run_coroutine,
            "get_data": self.get_data,
            "update_data": self.update_data,
            "delete_data": self.delete_data,
            "terminate": self.close,
            "ping": pingpong,
            "upload_file": self.upload_file,
            "start_ipython": self.start_ipython,
            "call_stack": self.get_call_stack,
            "profile": self.get_profile,
            "profile_metadata": self.get_profile_metadata,
            "get_logs": self.get_logs,
            "keys": self.keys,
            "versions": self.versions,
            "actor_execute": self.actor_execute,
            "actor_attribute": self.actor_attribute,
            "plugin-add": self.plugin_add,
        }

        stream_handlers = {
            "close": self.close,
            "compute-task": self.add_task,
            "release-task": partial(self.release_key, report=False),
            "delete-data": self.delete_data,
            "steal-request": self.steal_request,
        }

        super().__init__(
            handlers=handlers,
            stream_handlers=stream_handlers,
            io_loop=self.loop,
            connection_args=self.connection_args,
            **kwargs,
        )

        self.scheduler = self.rpc(scheduler_addr)
        self.execution_state = {
            "scheduler": self.scheduler.address,
            "ioloop": self.loop,
            "worker": self,
        }

        pc = PeriodicCallback(self.heartbeat, 1000)
        self.periodic_callbacks["heartbeat"] = pc
        pc = PeriodicCallback(
            lambda: self.batched_stream.send({"op": "keep-alive"}), 60000
        )
        self.periodic_callbacks["keep-alive"] = pc

        self._address = contact_address

        self.memory_monitor_interval = parse_timedelta(
            memory_monitor_interval, default="ms"
        )
        if self.memory_limit:
            self._memory_monitoring = False
            pc = PeriodicCallback(
                self.memory_monitor, self.memory_monitor_interval * 1000
            )
            self.periodic_callbacks["memory"] = pc

        if extensions is None:
            extensions = DEFAULT_EXTENSIONS
        for ext in extensions:
            ext(self)

        self._throttled_gc = ThrottledGC(logger=logger)

        setproctitle("dask-worker [not started]")

        profile_trigger_interval = parse_timedelta(
            dask.config.get("distributed.worker.profile.interval"), default="ms"
        )
        pc = PeriodicCallback(self.trigger_profile, profile_trigger_interval * 1000)
        self.periodic_callbacks["profile"] = pc

        pc = PeriodicCallback(self.cycle_profile, profile_cycle_interval * 1000)
        self.periodic_callbacks["profile-cycle"] = pc

        self.plugins = {}
        self._pending_plugins = plugins

        self.lifetime = lifetime or dask.config.get(
            "distributed.worker.lifetime.duration"
        )
        lifetime_stagger = lifetime_stagger or dask.config.get(
            "distributed.worker.lifetime.stagger"
        )
        self.lifetime_restart = lifetime_restart or dask.config.get(
            "distributed.worker.lifetime.restart"
        )
        if isinstance(self.lifetime, str):
            self.lifetime = parse_timedelta(self.lifetime)
        if isinstance(lifetime_stagger, str):
            lifetime_stagger = parse_timedelta(lifetime_stagger)
        if self.lifetime:
            self.lifetime += (random.random() * 2 - 1) * lifetime_stagger
            self.io_loop.call_later(self.lifetime, self.close_gracefully)

        Worker._instances.add(self)

    ##################
    # Administrative #
    ##################

    def __repr__(self):
        return "<%s: %r, %s, %s, stored: %d, running: %d/%d, ready: %d, comm: %d, waiting: %d>" % (
            self.__class__.__name__,
            self.address,
            self.name,
            self.status,
            len(self.data),
            self.executing_count,
            self.nthreads,
            len(self.ready),
            self.in_flight_tasks,
            self.waiting_for_data_count,
        )

    @property
    def logs(self):
        return self._deque_handler.deque

    def log_event(self, topic, msg):
        self.batched_stream.send(
            {
                "op": "log-event",
                "topic": topic,
                "msg": msg,
            }
        )

    @property
    def worker_address(self):
        """ For API compatibility with Nanny """
        return self.address

    @property
    def local_dir(self):
        """ For API compatibility with Nanny """
        warnings.warn(
            "The local_dir attribute has moved to local_directory", stacklevel=2
        )
        return self.local_directory

    async def get_metrics(self):
        now = time()
        core = dict(
            in_memory=len(self.data),
            ready=len(self.ready),
            in_flight=self.in_flight_tasks,
            bandwidth={
                "total": self.bandwidth,
                "workers": dict(self.bandwidth_workers),
                "types": keymap(typename, self.bandwidth_types),
            },
            executing={
                key: now - self.tasks[key].start_time
                for key in self.active_threads.values()
                if key in self.tasks
            },
        )
        custom = {}
        for k, metric in self.metrics.items():
            try:
                result = metric(self)
                if isawaitable(result):
                    result = await result
                custom[k] = result
            except Exception:  # TODO: log error once
                pass

        return merge(custom, self.monitor.recent(), core)

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

    def identity(self, comm=None):
        return {
            "type": type(self).__name__,
            "id": self.id,
            "scheduler": self.scheduler.address,
            "nthreads": self.nthreads,
            "ncores": self.nthreads,  # backwards compatibility
            "memory_limit": self.memory_limit,
        }

    #####################
    # External Services #
    #####################

    async def _register_with_scheduler(self):
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
                        keys=list(self.data),
                        nthreads=self.nthreads,
                        name=self.name,
                        nbytes={ts.key: ts.get_nbytes() for ts in self.tasks.values()},
                        types={k: typename(v) for k, v in self.data.items()},
                        now=time(),
                        resources=self.total_resources,
                        memory_limit=self.memory_limit,
                        local_directory=self.local_directory,
                        services=self.service_ports,
                        nanny=self.nanny,
                        pid=os.getpid(),
                        versions=get_versions(),
                        metrics=await self.get_metrics(),
                        extra=await self.get_startup_information(),
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
                self.status = Status.running
                break
            except EnvironmentError:
                logger.info("Waiting to connect to: %26s", self.scheduler.address)
                await asyncio.sleep(0.1)
            except TimeoutError:
                logger.info("Timed out when connecting to scheduler")
        if response["status"] != "OK":
            raise ValueError("Unexpected response from register: %r" % (response,))
        else:
            await asyncio.gather(
                *[
                    self.plugin_add(**plugin_kwargs)
                    for plugin_kwargs in response["worker-plugins"]
                ]
            )

            logger.info("        Registered to: %26s", self.scheduler.address)
            logger.info("-" * 49)

        self.batched_stream.start(comm)
        self.periodic_callbacks["keep-alive"].start()
        self.periodic_callbacks["heartbeat"].start()
        self.loop.add_callback(self.handle_scheduler, comm)

    def _update_latency(self, latency):
        self.latency = latency * 0.05 + self.latency * 0.95
        if self.digests is not None:
            self.digests["latency"].add(latency)

    async def heartbeat(self):
        if not self.heartbeat_active:
            self.heartbeat_active = True
            logger.debug("Heartbeat: %s" % self.address)
            try:
                start = time()
                response = await retry_operation(
                    self.scheduler.heartbeat_worker,
                    address=self.contact_address,
                    now=time(),
                    metrics=await self.get_metrics(),
                )
                end = time()
                middle = (start + end) / 2

                self._update_latency(end - start)

                if response["status"] == "missing":
                    for i in range(10):
                        if self.status != Status.running:
                            break
                        else:
                            await asyncio.sleep(0.05)
                    else:
                        await self._register_with_scheduler()
                    return
                self.scheduler_delay = response["time"] - middle
                self.periodic_callbacks["heartbeat"].callback_time = (
                    response["heartbeat-interval"] * 1000
                )
                self.bandwidth_workers.clear()
                self.bandwidth_types.clear()
            except CommClosedError:
                logger.warning("Heartbeat to scheduler failed")
                if not self.reconnect:
                    await self.close(report=False)
            except IOError as e:
                # Scheduler is gone. Respect distributed.comm.timeouts.connect
                if "Timed out trying to connect" in str(e):
                    await self.close(report=False)
                else:
                    raise e
            finally:
                self.heartbeat_active = False
        else:
            logger.debug("Heartbeat skipped: channel busy")

    async def handle_scheduler(self, comm):
        try:
            await self.handle_stream(
                comm, every_cycle=[self.ensure_communicating, self.ensure_computing]
            )
        except Exception as e:
            logger.exception(e)
            raise
        finally:
            if self.reconnect and self.status == Status.running:
                logger.info("Connection to scheduler broken.  Reconnecting...")
                self.loop.add_callback(self.heartbeat)
            else:
                await self.close(report=False)

    def start_ipython(self, comm):
        """Start an IPython kernel

        Returns Jupyter connection info dictionary.
        """
        from ._ipython_utils import start_ipython

        if self._ipython_kernel is None:
            self._ipython_kernel = start_ipython(
                ip=self.ip, ns={"worker": self}, log=logger
            )
        return self._ipython_kernel.get_connection_info()

    async def upload_file(self, comm, filename=None, data=None, load=True):
        out_filename = os.path.join(self.local_directory, filename)

        def func(data):
            if isinstance(data, str):
                data = data.encode()
            with open(out_filename, "wb") as f:
                f.write(data)
                f.flush()
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

    def keys(self, comm=None):
        return list(self.data)

    async def gather(self, comm=None, who_has=None):
        who_has = {
            k: [coerce_to_address(addr) for addr in v]
            for k, v in who_has.items()
            if k not in self.data
        }
        result, missing_keys, missing_workers = await gather_from_workers(
            who_has, rpc=self.rpc, who=self.address
        )
        if missing_keys:
            logger.warning(
                "Could not find data: %s on workers: %s (who_has: %s)",
                missing_keys,
                missing_workers,
                who_has,
            )
            return {"status": "missing-data", "keys": missing_keys}
        else:
            self.update_data(data=result, report=False)
            return {"status": "OK"}

    #############
    # Lifecycle #
    #############

    async def start(self):
        if self.status and self.status in (
            Status.closed,
            Status.closing,
            Status.closing_gracefully,
        ):
            return
        assert self.status is Status.undefined, self.status

        await super().start()

        enable_gc_diagnosis()
        thread_state.on_event_loop_thread = True

        ports = parse_ports(self._start_port)
        for port in ports:
            start_address = address_from_user_args(
                host=self._start_host,
                port=port,
                interface=self._interface,
                protocol=self._protocol,
                security=self.security,
            )
            try:
                await self.listen(
                    start_address, **self.security.get_listen_args("worker")
                )
            except OSError as e:
                if len(ports) > 1 and e.errno == errno.EADDRINUSE:
                    continue
                else:
                    raise e
            else:
                self._start_address = start_address
                break
        else:
            raise ValueError(
                f"Could not start Worker on host {self._start_host}"
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
            await preload.start()

        # Services listen on all addresses
        # Note Nanny is not a "real" service, just some metadata
        # passed in service_ports...
        self.start_services(self.ip)

        try:
            listening_address = "%s%s:%d" % (self.listener.prefix, self.ip, self.port)
        except Exception:
            listening_address = "%s%s" % (self.listener.prefix, self.ip)

        logger.info("      Start worker at: %26s", self.address)
        logger.info("         Listening to: %26s", listening_address)
        for k, v in self.service_ports.items():
            logger.info("  %16s at: %26s" % (k, self.ip + ":" + str(v)))
        logger.info("Waiting to connect to: %26s", self.scheduler.address)
        logger.info("-" * 49)
        logger.info("              Threads: %26d", self.nthreads)
        if self.memory_limit:
            logger.info("               Memory: %26s", format_bytes(self.memory_limit))
        logger.info("      Local Directory: %26s", self.local_directory)

        setproctitle("dask-worker [%s]" % self.address)

        await asyncio.gather(
            *[self.plugin_add(plugin=plugin) for plugin in self._pending_plugins]
        )
        self._pending_plugins = ()

        await self._register_with_scheduler()

        self.start_periodic_callbacks()
        return self

    def _close(self, *args, **kwargs):
        warnings.warn("Worker._close has moved to Worker.close", stacklevel=2)
        return self.close(*args, **kwargs)

    async def close(
        self, report=True, timeout=10, nanny=True, executor_wait=True, safe=False
    ):
        with log_errors():
            if self.status in (Status.closed, Status.closing):
                await self.finished()
                return

            self.reconnect = False
            disable_gc_diagnosis()

            try:
                logger.info("Stopping worker at %s", self.address)
            except ValueError:  # address not available if already closed
                logger.info("Stopping worker")
            if self.status not in (Status.running, Status.closing_gracefully):
                logger.info("Closed worker has not yet started: %s", self.status)
            self.status = Status.closing

            for preload in self.preloads:
                await preload.teardown()

            if nanny and self.nanny:
                with self.rpc(self.nanny) as r:
                    await r.close_gracefully()

            setproctitle("dask-worker [closing]")

            teardowns = [
                plugin.teardown(self)
                for plugin in self.plugins.values()
                if hasattr(plugin, "teardown")
            ]

            await asyncio.gather(*[td for td in teardowns if isawaitable(td)])

            for pc in self.periodic_callbacks.values():
                pc.stop()
            with suppress(EnvironmentError, TimeoutError):
                if report and self.contact_address is not None:
                    await asyncio.wait_for(
                        self.scheduler.unregister(
                            address=self.contact_address, safe=safe
                        ),
                        timeout,
                    )
            await self.scheduler.close_rpc()
            self._workdir.release()

            self.stop_services()

            if (
                self.batched_stream
                and self.batched_stream.comm
                and not self.batched_stream.comm.closed()
            ):
                self.batched_stream.send({"op": "close-stream"})

            if self.batched_stream:
                with suppress(TimeoutError):
                    await self.batched_stream.close(timedelta(seconds=timeout))

            self.actor_executor._work_queue.queue.clear()
            if isinstance(self.executor, ThreadPoolExecutor):
                self.executor._work_queue.queue.clear()
                self.executor.shutdown(wait=executor_wait, timeout=timeout)
            else:
                self.executor.shutdown(wait=False)
            self.actor_executor.shutdown(wait=executor_wait, timeout=timeout)

            self.stop()
            await self.rpc.close()

            self.status = Status.closed
            await ServerNode.close(self)

            setproctitle("dask-worker [closed]")
        return "OK"

    async def close_gracefully(self, restart=None):
        """Gracefully shut down a worker

        This first informs the scheduler that we're shutting down, and asks it
        to move our data elsewhere.  Afterwards, we close as normal
        """
        if self.status in (Status.closing, Status.closing_gracefully):
            await self.finished()

        if self.status == Status.closed:
            return

        if restart is None:
            restart = self.lifetime_restart

        logger.info("Closing worker gracefully: %s", self.address)
        self.status = Status.closing_gracefully
        await self.scheduler.retire_workers(workers=[self.address], remove=False)
        await self.close(safe=True, nanny=not restart)

    async def terminate(self, comm=None, report=True, **kwargs):
        await self.close(report=report, **kwargs)
        return "OK"

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

            self.loop.add_callback(batched_send_connect)

        self.stream_comms[address].send(msg)

    async def get_data(
        self, comm, keys=None, who=None, serializers=None, max_connections=None
    ):
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

        if self.paused:
            max_connections = 1
            throttle_msg = " Throttling outgoing connections because worker is paused."
        else:
            throttle_msg = ""

        if (
            max_connections is not False
            and self.outgoing_current_count >= max_connections
        ):
            logger.debug(
                "Worker %s has too many open connections to respond to data request from %s (%d/%d).%s",
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
                if k in self.actors:
                    from .actor import Actor

                    data[k] = Actor(type(self.actors[k]), self.address, k)

        msg = {"status": "OK", "data": {k: to_serialize(v) for k, v in data.items()}}
        nbytes = {k: self.tasks[k].nbytes for k in data if k in self.tasks}
        stop = time()
        if self.digests is not None:
            self.digests["get-data-load-duration"].add(stop - start)
        start = time()

        try:
            compressed = await comm.write(msg, serializers=serializers)
            response = await comm.read(deserializers=serializers)
            assert response == "OK", response
        except EnvironmentError:
            logger.exception(
                "failed during get data with %s -> %s", self.address, who, exc_info=True
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

    def update_data(self, comm=None, data=None, report=True, serializers=None):
        for key, value in data.items():
            ts = self.tasks.get(key)
            if getattr(ts, "state", None) is not None:
                self.transition(ts, "memory", value=value)
            else:
                self.tasks[key] = ts = TaskState(key)
                self.put_key_in_memory(ts, value)
                ts.priority = None
                ts.duration = None

            self.log.append((key, "receive-from-scatter"))

        if report:
            self.batched_stream.send({"op": "add-keys", "keys": list(data)})
        info = {"nbytes": {k: sizeof(v) for k, v in data.items()}, "status": "OK"}
        return info

    def delete_data(self, comm=None, keys=None, report=True):
        if keys:
            for key in list(keys):
                self.log.append((key, "delete"))
                self.release_key(key)

            logger.debug("Worker %s -- Deleted %d keys", self.name, len(keys))
        return "OK"

    async def set_resources(self, **resources):
        for r, quantity in resources.items():
            if r in self.total_resources:
                self.available_resources[r] += quantity - self.total_resources[r]
            else:
                self.available_resources[r] = quantity
            self.total_resources[r] = quantity

        await retry_operation(
            self.scheduler.set_resources,
            resources=self.total_resources,
            worker=self.contact_address,
        )

    ###################
    # Task Management #
    ###################

    def add_task(
        self,
        key,
        function=None,
        args=None,
        kwargs=None,
        task=no_value,
        who_has=None,
        nbytes=None,
        priority=None,
        duration=None,
        resource_restrictions=None,
        actor=False,
        speculative=False,
        **kwargs2,
    ):
        try:
            if key in self.tasks:
                ts = self.tasks[key]
                if ts.state == "memory":
                    assert key in self.data or key in self.actors
                    logger.debug(
                        "Asked to compute pre-existing result: %s: %s", key, ts.state
                    )
                    self.send_task_state_to_scheduler(ts)
                    return
                if ts.state in IN_PLAY:
                    return
                if ts.state == "erred":
                    ts.exception = None
                    ts.traceback = None
                else:
                    ts.state = "waiting"
            else:
                self.log.append((key, "new"))
                self.tasks[key] = ts = TaskState(
                    key=key, runspec=SerializedTask(function, args, kwargs, task)
                )
                ts.state = "waiting" if not speculative else "speculative"

            if priority is not None:
                priority = tuple(priority) + (self.generation,)
                self.generation -= 1

            if actor:
                self.actors[ts.key] = None

            ts.priority = priority
            ts.duration = duration
            if resource_restrictions:
                ts.resource_restrictions = resource_restrictions

            who_has = who_has or {}

            for dependency_key, workers in who_has.items():
                assert workers
                if dependency_key not in self.tasks:
                    self.tasks[dependency_key] = dep_ts = TaskState(key=dependency_key)
                    dep_ts.state = (
                        "waiting" if dependency_key not in self.data else "memory"
                    )

                dep_ts = self.tasks[dependency_key]
                self.log.append((dependency_key, "new-dep", dep_ts.state))

                if dep_ts.state != "memory":
                    ts.waiting_for_data.add(dep_ts.key)
                    self.waiting_for_data_count += 1

                dep_ts.who_has.update(workers)

                ts.dependencies.add(dep_ts)
                dep_ts.dependents.add(ts)

                for worker in workers:
                    self.has_what[worker].add(dep_ts.key)
                    if dep_ts.state != "memory":
                        self.pending_data_per_worker[worker].append(dep_ts.key)

            if nbytes is not None:
                for key, value in nbytes.items():
                    # TODO: sometimes `value` is `None`, which is probably due to being mis-set somewhere else
                    self.tasks[key].nbytes = value or DEFAULT_DATA_SIZE

            if ts.waiting_for_data:
                self.data_needed.append(ts.key)
            else:
                self.transition(ts, "ready")
            if self.validate:
                if who_has:
                    assert all(self.tasks[dep] in ts.dependencies for dep in who_has)
                    assert all(self.tasks[dep.key] for dep in ts.dependencies)
                    for dependency in ts.dependencies:
                        self.validate_task(dependency)
                    self.validate_task(ts)
        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    def transition(self, ts, finish, **kwargs):
        if ts is None:
            return
        start = ts.state
        if start == finish:
            return
        func = self._transitions[start, finish]
        state = func(ts, **kwargs)
        self.log.append((ts.key, start, state or finish))
        ts.state = state or finish
        if self.validate:
            self.validate_task(ts)
        self._notify_plugins("transition", ts.key, start, state or finish, **kwargs)

    def transition_speculative_ready(self, ts):
        try:
            if self.validate:
                assert ts.state == "speculative"
                assert all(dep.key in self.data for dep in ts.dependencies)
                assert not ts.waiting_for_data

            # TODO add constrained condition
            heapq.heappush(self.ready, (ts.priority, ts.key))

        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    def transition_waiting_flight(self, ts, worker=None):
        try:
            if self.validate:
                assert ts.state != "flight"
                assert ts.dependents

            ts.coming_from = worker
            self.in_flight_tasks += 1
        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    def transition_flight_waiting(self, ts, worker=None, remove=True):
        try:
            if self.validate:
                assert ts.state == "flight"

            self.in_flight_tasks -= 1
            ts.coming_from = None
            if remove:
                try:
                    ts.who_has.remove(worker)
                    self.has_what[worker].remove(ts.key)
                except KeyError:
                    pass

            if not ts.who_has:
                if ts.key not in self._missing_dep_flight:
                    self._missing_dep_flight.add(ts.key)
                    self.loop.add_callback(self.handle_missing_dep, ts)
            for dependent in ts.dependents:
                if dependent.state == "waiting":
                    if remove:  # try a new worker immediately
                        self.data_needed.appendleft(dependent.key)
                    else:  # worker was probably busy, wait a while
                        self.data_needed.append(dependent.key)

            if not ts.dependents:
                self.release_key(ts.key)
        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    def transition_flight_memory(self, ts, value=None):
        try:
            if self.validate:
                assert ts.state == "flight"

            self.in_flight_tasks -= 1
            ts.coming_from = None
            if ts.dependents:
                self.put_key_in_memory(ts, value)
                for dependent in ts.dependents:
                    try:
                        dependent.waiting_for_data.remove(ts.key)
                        self.waiting_for_data_count -= 1
                    except KeyError:
                        pass

                self.batched_stream.send({"op": "add-keys", "keys": [ts.key]})
            else:
                self.release_key(ts.key)

        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    def transition_waiting_ready(self, ts):
        try:
            if self.validate:
                assert ts.state == "waiting"
                assert not ts.waiting_for_data
                assert all(
                    dep.key in self.data or dep.key in self.actors
                    for dep in ts.dependencies
                )
                assert all(dep.state == "memory" for dep in ts.dependencies)
                assert ts.key not in self.ready

            ts.waiting_for_data.clear()

            if ts.resource_restrictions is not None:
                self.constrained.append(ts.key)
                return "constrained"
            else:
                heapq.heappush(self.ready, (ts.priority, ts.key))
        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    def transition_waiting_done(self, ts, value=None):
        try:
            if self.validate:
                assert ts.state == "waiting"
                assert ts.key not in self.ready

            self.waiting_for_data_count -= len(ts.waiting_for_data)
            ts.waiting_for_data.clear()
            if value is not None:
                self.put_key_in_memory(ts, value)
            self.send_task_state_to_scheduler(ts)
        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    def transition_ready_executing(self, ts):
        try:
            if self.validate:
                assert not ts.waiting_for_data
                assert ts.key not in self.data
                assert ts.state in READY
                assert ts.key not in self.ready
                assert all(
                    dep.key in self.data or dep.key in self.actors
                    for dep in ts.dependencies
                )

            self.executing_count += 1
            self.loop.add_callback(self.execute, ts.key)
        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    def transition_ready_memory(self, ts, value=None):
        self.send_task_state_to_scheduler(ts)

    def transition_constrained_executing(self, ts):
        self.transition_ready_executing(ts)
        for resource, quantity in ts.resource_restrictions.items():
            self.available_resources[resource] -= quantity

        if self.validate:
            assert all(v >= 0 for v in self.available_resources.values())

    def transition_executing_done(self, ts, value=no_value, report=True):
        try:
            if self.validate:
                assert ts.state == "executing" or ts.key in self.long_running
                assert not ts.waiting_for_data
                assert ts.key not in self.ready

            out = None
            if ts.resource_restrictions is not None:
                for resource, quantity in ts.resource_restrictions.items():
                    self.available_resources[resource] += quantity

            if ts.state == "executing":
                self.executing_count -= 1
                self.executed_count += 1
            elif ts.state == "long-running":
                self.long_running.remove(ts.key)

            if value is not no_value:
                try:
                    self.put_key_in_memory(ts, value, transition=False)
                except Exception as e:
                    logger.info("Failed to put key in memory", exc_info=True)
                    msg = error_message(e)
                    ts.exception = msg["exception"]
                    ts.traceback = msg["traceback"]
                    ts.state = "error"
                    out = "error"

                # Don't release the dependency keys, but do remove them from `dependents`
                for dependency in ts.dependencies:
                    dependency.dependents.discard(ts)
                ts.dependencies.clear()

            if report and self.batched_stream and self.status == Status.running:
                self.send_task_state_to_scheduler(ts)
            else:
                raise CommClosedError

            return out

        except EnvironmentError:
            logger.info("Comm closed")
        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    def transition_executing_long_running(self, ts, compute_duration=None):
        try:
            if self.validate:
                assert ts.state == "executing"

            self.executing_count -= 1
            self.long_running.add(ts.key)
            self.batched_stream.send(
                {
                    "op": "long-running",
                    "key": ts.key,
                    "compute_duration": compute_duration,
                }
            )

            self.io_loop.add_callback(self.ensure_computing)
        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    def maybe_transition_long_running(self, ts, compute_duration=None):
        if ts.state == "executing":
            self.transition(ts, "long-running", compute_duration=compute_duration)

    def stateof(self, key):
        ts = self.tasks[key]
        return {
            "executing": ts.state == "executing",
            "waiting_for_data": bool(ts.waiting_for_data),
            "heap": key in pluck(1, self.ready),
            "data": key in self.data,
        }

    def story(self, *keys):
        return [
            msg
            for msg in self.log
            if any(key in msg for key in keys)
            or any(
                key in c
                for key in keys
                for c in msg
                if isinstance(c, (tuple, list, set))
            )
        ]

    def ensure_communicating(self):
        changed = True
        try:
            while (
                changed
                and self.data_needed
                and len(self.in_flight_workers) < self.total_out_connections
            ):
                changed = False
                logger.debug(
                    "Ensure communicating.  Pending: %d.  Connections: %d/%d",
                    len(self.data_needed),
                    len(self.in_flight_workers),
                    self.total_out_connections,
                )

                key = self.data_needed[0]

                if key not in self.tasks:
                    self.data_needed.popleft()
                    changed = True
                    continue

                ts = self.tasks[key]
                if ts.state != "waiting":
                    self.log.append((key, "communication pass"))
                    self.data_needed.popleft()
                    changed = True
                    continue

                deps = ts.dependencies
                if self.validate:
                    assert all(dep.key in self.tasks for dep in deps)

                deps = [dep for dep in deps if dep.state == "waiting"]

                missing_deps = {dep for dep in deps if not dep.who_has}
                if missing_deps:
                    logger.info("Can't find dependencies for key %s", key)
                    missing_deps2 = {
                        dep
                        for dep in missing_deps
                        if dep.key not in self._missing_dep_flight
                    }
                    for dep in missing_deps2:
                        self._missing_dep_flight.add(dep.key)
                    self.loop.add_callback(self.handle_missing_dep, *missing_deps2)

                    deps = [dep for dep in deps if dep not in missing_deps]

                self.log.append(("gather-dependencies", key, deps))

                in_flight = False

                while deps and (
                    len(self.in_flight_workers) < self.total_out_connections
                    or self.comm_nbytes < self.total_comm_nbytes
                ):
                    dep = deps.pop()
                    if dep.state != "waiting":
                        continue
                    if not dep.who_has:
                        continue
                    workers = [
                        w for w in dep.who_has if w not in self.in_flight_workers
                    ]
                    if not workers:
                        in_flight = True
                        continue
                    host = get_address_host(self.address)
                    local = [w for w in workers if get_address_host(w) == host]
                    if local:
                        worker = random.choice(local)
                    else:
                        worker = random.choice(list(workers))
                    to_gather, total_nbytes = self.select_keys_for_gather(
                        worker, dep.key
                    )
                    self.comm_nbytes += total_nbytes
                    self.in_flight_workers[worker] = to_gather
                    for d in to_gather:
                        self.transition(self.tasks[d], "flight", worker=worker)
                    self.loop.add_callback(
                        self.gather_dep, worker, dep, to_gather, total_nbytes, cause=key
                    )
                    changed = True

                if not deps and not in_flight:
                    self.data_needed.popleft()

        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    def send_task_state_to_scheduler(self, ts):
        if ts.key in self.data or self.actors.get(ts.key):
            typ = ts.type
            if ts.nbytes is None or typ is None:
                try:
                    value = self.data[ts.key]
                except KeyError:
                    value = self.actors[ts.key]
                nbytes = ts.nbytes = sizeof(value)
                typ = ts.type = type(value)
                del value
            try:
                typ_serialized = dumps_function(typ)
            except PicklingError:
                # Some types fail pickling (example: _thread.lock objects),
                # send their name as a best effort.
                typ_serialized = pickle.dumps(typ.__name__, protocol=4)
            d = {
                "op": "task-finished",
                "status": "OK",
                "key": ts.key,
                "nbytes": ts.nbytes,
                "thread": self.threads.get(ts.key),
                "type": typ_serialized,
                "typename": typename(typ),
                "metadata": ts.metadata,
            }
        elif ts.exception is not None:
            d = {
                "op": "task-erred",
                "status": "error",
                "key": ts.key,
                "thread": self.threads.get(ts.key),
                "exception": ts.exception,
                "traceback": ts.traceback,
            }
        else:
            logger.error("Key not ready to send to worker, %s: %s", ts.key, ts.state)
            return

        if ts.startstops:
            d["startstops"] = ts.startstops
        self.batched_stream.send(d)

    def put_key_in_memory(self, ts, value, transition=True):
        if ts.key in self.data:
            ts.state = "memory"
            return

        if ts.key in self.actors:
            self.actors[ts.key] = value

        else:
            start = time()
            self.data[ts.key] = value
            ts.state = "memory"
            stop = time()
            if stop - start > 0.020:
                ts.startstops.append(
                    {"action": "disk-write", "start": start, "stop": stop}
                )

        if ts.nbytes is None:
            ts.nbytes = sizeof(value)

        ts.type = type(value)

        for dep in ts.dependents:
            try:
                dep.waiting_for_data.remove(ts.key)
                self.waiting_for_data_count -= 1
            except KeyError:
                pass
            if not dep.waiting_for_data:
                self.transition(dep, "ready")

        self.log.append((ts.key, "put-in-memory"))

    def select_keys_for_gather(self, worker, dep):
        assert isinstance(dep, str)
        deps = {dep}

        total_bytes = self.tasks[dep].get_nbytes()
        L = self.pending_data_per_worker[worker]

        while L:
            d = L.popleft()
            ts = self.tasks.get(d)
            if ts is None or ts.state != "waiting":
                continue
            if total_bytes + ts.get_nbytes() > self.target_message_size:
                break
            deps.add(d)
            total_bytes += ts.get_nbytes()

        return deps, total_bytes

    async def gather_dep(self, worker, dep, deps, total_nbytes, cause=None):
        """Gather dependencies for a task from a worker who has them

        Parameters
        ----------
        worker : str
            address of worker to gather dependency from
        dep : TaskState
            task we want to gather dependencies for
        deps : list
            keys of dependencies to gather from worker -- this is not
            necessarily equivalent to the full list of dependencies of ``dep``
            as some dependencies may already be present on this worker.
        """
        if self.status != Status.running:
            return
        with log_errors():
            response = {}
            try:
                if self.validate:
                    self.validate_state()

                # dep states may have changed before gather_dep runs
                # if a dep is no longer in-flight then don't fetch it
                deps_ts = [self.tasks.get(key, None) or TaskState(key) for key in deps]
                deps_ts = tuple(ts for ts in deps_ts if ts.state == "flight")
                deps = [d.key for d in deps_ts]

                self.log.append(("request-dep", dep.key, worker, deps))
                logger.debug("Request %d keys", len(deps))

                start = time()
                response = await get_data_from_worker(
                    self.rpc, deps, worker, who=self.address
                )
                stop = time()

                if response["status"] == "busy":
                    self.log.append(("busy-gather", worker, deps))
                    for ts in deps_ts:
                        if ts.state == "flight":
                            self.transition(ts, "waiting")
                    return

                if cause:
                    cause_ts = self.tasks.get(cause, TaskState(key=cause))
                    cause_ts.startstops.append(
                        {
                            "action": "transfer",
                            "start": start + self.scheduler_delay,
                            "stop": stop + self.scheduler_delay,
                            "source": worker,
                        }
                    )

                total_bytes = sum(
                    self.tasks[key].get_nbytes()
                    for key in response["data"]
                    if key in self.tasks
                )
                duration = (stop - start) or 0.010
                bandwidth = total_bytes / duration
                self.incoming_transfer_log.append(
                    {
                        "start": start + self.scheduler_delay,
                        "stop": stop + self.scheduler_delay,
                        "middle": (start + stop) / 2.0 + self.scheduler_delay,
                        "duration": duration,
                        "keys": {
                            key: self.tasks[key].nbytes
                            for key in response["data"]
                            if key in self.tasks
                        },
                        "total": total_bytes,
                        "bandwidth": bandwidth,
                        "who": worker,
                    }
                )
                if total_bytes > 1000000:
                    self.bandwidth = self.bandwidth * 0.95 + bandwidth * 0.05
                    bw, cnt = self.bandwidth_workers[worker]
                    self.bandwidth_workers[worker] = (bw + bandwidth, cnt + 1)

                    types = set(map(type, response["data"].values()))
                    if len(types) == 1:
                        [typ] = types
                        bw, cnt = self.bandwidth_types[typ]
                        self.bandwidth_types[typ] = (bw + bandwidth, cnt + 1)

                if self.digests is not None:
                    self.digests["transfer-bandwidth"].add(total_bytes / duration)
                    self.digests["transfer-duration"].add(duration)
                self.counters["transfer-count"].add(len(response["data"]))
                self.incoming_count += 1

                self.log.append(("receive-dep", worker, list(response["data"])))
            except EnvironmentError as e:
                logger.exception("Worker stream died during communication: %s", worker)
                self.log.append(("receive-dep-failed", worker))
                for d in self.has_what.pop(worker):
                    self.tasks[d].who_has.remove(worker)

            except Exception as e:
                logger.exception(e)
                if self.batched_stream and LOG_PDB:
                    import pdb

                    pdb.set_trace()
                raise
            finally:
                self.comm_nbytes -= total_nbytes
                busy = response.get("status", "") == "busy"
                data = response.get("data", {})

                for d in self.in_flight_workers.pop(worker):

                    ts = self.tasks.get(d)

                    if not busy and d in data:
                        self.transition(ts, "memory", value=data[d])
                    elif ts is None or ts.state == "executing":
                        self.release_key(d)
                        continue
                    elif ts.state not in ("ready", "memory"):
                        self.transition(ts, "waiting", worker=worker, remove=not busy)

                    if not busy and d not in data and ts.dependents:
                        self.log.append(("missing-dep", d))
                        self.batched_stream.send(
                            {"op": "missing-data", "errant_worker": worker, "key": d}
                        )

                if self.validate:
                    self.validate_state()

                await self.ensure_computing()

                if not busy:
                    self.repetitively_busy = 0
                    self.ensure_communicating()
                else:
                    # Exponential backoff to avoid hammering scheduler/worker
                    self.repetitively_busy += 1
                    await asyncio.sleep(0.100 * 1.5 ** self.repetitively_busy)

                    # See if anyone new has the data
                    await self.query_who_has(dep.key)
                    self.ensure_communicating()

    def bad_dep(self, dep):
        exc = ValueError(
            "Could not find dependent %s.  Check worker logs" % str(dep.key)
        )
        for ts in dep.dependents:
            msg = error_message(exc)
            ts.exception = msg["exception"]
            ts.traceback = msg["traceback"]
            self.transition(ts, "error")
        self.release_key(dep.key)

    async def handle_missing_dep(self, *deps, **kwargs):
        self.log.append(("handle-missing", deps))
        try:
            deps = {dep for dep in deps if dep.dependents}
            if not deps:
                return

            for dep in deps:
                if dep.suspicious_count > 5:
                    deps.remove(dep)
                    self.bad_dep(dep)
            if not deps:
                return

            for dep in deps:
                logger.info(
                    "Dependent not found: %s %s .  Asking scheduler",
                    dep.key,
                    dep.suspicious_count,
                )

            who_has = await retry_operation(
                self.scheduler.who_has, keys=list(dep.key for dep in deps)
            )
            who_has = {k: v for k, v in who_has.items() if v}
            self.update_who_has(who_has)
            for dep in deps:
                dep.suspicious_count += 1

                if not who_has.get(dep.key):
                    self.log.append((dep.key, "no workers found", dep.dependents))
                    self.release_key(dep.key)
                else:
                    self.log.append((dep.key, "new workers found"))
                    for dependent in dep.dependents:
                        if dependent.key in dep.waiting_for_data:
                            self.data_needed.append(dependent.key)

        except Exception:
            logger.error("Handle missing dep failed, retrying", exc_info=True)
            retries = kwargs.get("retries", 5)
            self.log.append(("handle-missing-failed", retries, deps))
            if retries > 0:
                await self.handle_missing_dep(*deps, retries=retries - 1)
            else:
                raise
        finally:
            try:
                for dep in deps:
                    self._missing_dep_flight.remove(dep.key)
            except KeyError:
                pass

            self.ensure_communicating()

    async def query_who_has(self, *deps):
        with log_errors():
            response = await retry_operation(self.scheduler.who_has, keys=deps)
            self.update_who_has(response)
            return response

    def update_who_has(self, who_has):
        try:
            for dep, workers in who_has.items():
                if not workers:
                    continue

                self.tasks[dep].who_has.update(workers)

                for worker in workers:
                    self.has_what[worker].add(dep)
        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    def steal_request(self, key):
        # There may be a race condition between stealing and releasing a task.
        # In this case the self.tasks is already cleared. The `None` will be
        # registered as `already-computing` on the other end
        if key in self.tasks:
            state = self.tasks[key].state
        else:
            state = None

        response = {"op": "steal-response", "key": key, "state": state}
        self.batched_stream.send(response)

        if state in ("ready", "waiting", "constrained"):
            self.release_key(key)

    def release_key(self, key, cause=None, reason=None, report=True):
        try:
            ts = self.tasks.get(key, TaskState(key=key))
            if cause:
                self.log.append((key, "release-key", {"cause": cause}))
            else:
                self.log.append((key, "release-key"))
            if key in self.data and not ts.dependents:
                try:
                    del self.data[key]
                except FileNotFoundError:
                    logger.error("Tried to delete %s but no file found", exc_info=True)
            if key in self.actors and not ts.dependents:
                del self.actors[key]

            # for any dependencies of key we are releasing remove task as dependent
            for dependency in ts.dependencies:
                dependency.dependents.discard(ts)
                if not dependency.dependents and dependency.state in (
                    "waiting",
                    "flight",
                ):
                    self.release_key(dependency.key)

            for worker in ts.who_has:
                self.has_what[worker].discard(ts.key)

            if key in self.threads:
                del self.threads[key]

            if ts.state == "executing":
                self.executing_count -= 1

            if ts.resource_restrictions is not None:
                if ts.state == "executing":
                    for resource, quantity in ts.resource_restrictions.items():
                        self.available_resources[resource] += quantity

            if report and ts.state in PROCESSING:  # not finished
                self.batched_stream.send({"op": "release", "key": key, "cause": cause})

            self._notify_plugins("release_key", key, ts.state, cause, reason, report)
            if key in self.tasks:
                self.tasks.pop(key)
            del ts
        except CommClosedError:
            pass
        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    def rescind_key(self, key):
        try:
            if self.tasks[key].state not in PENDING:
                return

            ts = self.tasks.pop(key)

            # Task has been rescinded
            # For every task that it required
            for dependency in ts.dependencies:
                # Remove it as a dependent
                dependency.dependents.remove(key)
                # If the dependent is now without purpose (no dependencies), remove it
                if not dependency.dependents:
                    self.release_key(
                        dependency.key, reason="All dependent keys rescinded"
                    )

        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    ################
    # Execute Task #
    ################

    # FIXME: this breaks if changed to async def...
    # xref: https://github.com/dask/distributed/issues/3938
    @gen.coroutine
    def executor_submit(self, key, function, args=(), kwargs=None, executor=None):
        """Safely run function in thread pool executor

        We've run into issues running concurrent.future futures within
        tornado.  Apparently it's advantageous to use timeouts and periodic
        callbacks to ensure things run smoothly.  This can get tricky, so we
        pull it off into an separate method.
        """
        executor = executor or self.executor
        job_counter[0] += 1
        # logger.info("%s:%d Starts job %d, %s", self.ip, self.port, i, key)
        kwargs = kwargs or {}
        future = executor.submit(function, *args, **kwargs)
        pc = PeriodicCallback(
            lambda: logger.debug("future state: %s - %s", key, future._state), 1000
        )
        ts = self.tasks.get(key)
        if ts is not None:
            ts.start_time = time()
        pc.start()
        try:
            yield future
        finally:
            pc.stop()
            if ts is not None:
                ts.stop_time = time()

        result = future.result()

        # logger.info("Finish job %d, %s", i, key)
        raise gen.Return(result)

    def run(self, comm, function, args=(), wait=True, kwargs=None):
        return run(self, comm, function=function, args=args, kwargs=kwargs, wait=wait)

    def run_coroutine(self, comm, function, args=(), kwargs=None, wait=True):
        return run(self, comm, function=function, args=args, kwargs=kwargs, wait=wait)

    async def plugin_add(self, comm=None, plugin=None, name=None):
        with log_errors(pdb=False):
            if isinstance(plugin, bytes):
                plugin = pickle.loads(plugin)
            if not name:
                if hasattr(plugin, "name"):
                    name = plugin.name
                else:
                    name = funcname(plugin) + "-" + str(uuid.uuid4())

            assert name

            if name in self.plugins:
                return {"status": "repeat"}
            else:
                self.plugins[name] = plugin

                logger.info("Starting Worker plugin %s" % name)
                if hasattr(plugin, "setup"):
                    try:
                        result = plugin.setup(worker=self)
                        if isawaitable(result):
                            result = await result
                    except Exception as e:
                        msg = error_message(e)
                        return msg

                return {"status": "OK"}

    async def actor_execute(
        self, comm=None, actor=None, function=None, args=(), kwargs={}
    ):
        separate_thread = kwargs.pop("separate_thread", True)
        key = actor
        actor = self.actors[key]
        func = getattr(actor, function)
        name = key_split(key) + "." + function

        if iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        elif separate_thread:
            result = await self.executor_submit(
                name,
                apply_function_actor,
                args=(
                    func,
                    args,
                    kwargs,
                    self.execution_state,
                    name,
                    self.active_threads,
                    self.active_threads_lock,
                ),
                executor=self.actor_executor,
            )
        else:
            result = func(*args, **kwargs)
        return {"status": "OK", "result": to_serialize(result)}

    def actor_attribute(self, comm=None, actor=None, attribute=None):
        value = getattr(self.actors[actor], attribute)
        return {"status": "OK", "result": to_serialize(value)}

    def meets_resource_constraints(self, key):
        ts = self.tasks[key]
        if not ts.resource_restrictions:
            return True
        for resource, needed in ts.resource_restrictions.items():
            if self.available_resources[resource] < needed:
                return False

        return True

    async def _maybe_deserialize_task(self, ts):
        if not isinstance(ts.runspec, SerializedTask):
            return ts.runspec
        try:
            start = time()
            # Offload deserializing large tasks
            if sizeof(ts.runspec) > OFFLOAD_THRESHOLD:
                function, args, kwargs = await offload(_deserialize, *ts.runspec)
            else:
                function, args, kwargs = _deserialize(*ts.runspec)
            stop = time()

            if stop - start > 0.010:
                ts.startstops.append(
                    {"action": "deserialize", "start": start, "stop": stop}
                )
            return function, args, kwargs
        except Exception as e:
            logger.warning("Could not deserialize task", exc_info=True)
            emsg = error_message(e)
            emsg["key"] = ts.key
            emsg["op"] = "task-erred"
            self.batched_stream.send(emsg)
            self.log.append((ts.key, "deserialize-error"))
            raise

    async def ensure_computing(self):
        if self.paused:
            return
        try:
            while self.constrained and self.executing_count < self.nthreads:
                key = self.constrained[0]
                ts = self.tasks.get(key, None)
                if ts is None or ts.state != "constrained":
                    self.constrained.popleft()
                    continue
                if self.meets_resource_constraints(key):
                    self.constrained.popleft()
                    try:
                        # Ensure task is deserialized prior to execution
                        ts.runspec = await self._maybe_deserialize_task(ts)
                    except Exception:
                        continue
                    self.transition(ts, "executing")
                else:
                    break
            while self.ready and self.executing_count < self.nthreads:
                priority, key = heapq.heappop(self.ready)
                ts = self.tasks.get(key)
                if ts is None:
                    # It is possible for tasks to be released while still remaining on `ready`
                    # The scheduler might have re-routed to a new worker and told this worker
                    # to release.  If the task has "disappeared" just continue through the heap
                    continue
                elif ts.key in self.data:
                    self.transition(ts, "memory")
                elif ts.state in READY:
                    try:
                        # Ensure task is deserialized prior to execution
                        ts.runspec = await self._maybe_deserialize_task(ts)
                    except Exception:
                        continue
                    self.transition(ts, "executing")
        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    async def execute(self, key, report=False):
        executor_error = None
        if self.status in (Status.closing, Status.closed, Status.closing_gracefully):
            return
        try:
            if key not in self.tasks:
                return
            ts = self.tasks[key]
            if ts.state != "executing" or ts.runspec is None:
                return
            if self.validate:
                assert not ts.waiting_for_data
                assert ts.state == "executing"

            function, args, kwargs = ts.runspec

            start = time()
            data = {}
            for dep in ts.dependencies:
                k = dep.key
                try:
                    data[k] = self.data[k]
                except KeyError:
                    from .actor import Actor  # TODO: create local actor

                    data[k] = Actor(type(self.actors[k]), self.address, k, self)
            args2 = pack_data(args, data, key_types=(bytes, str))
            kwargs2 = pack_data(kwargs, data, key_types=(bytes, str))
            stop = time()
            if stop - start > 0.005:
                ts.startstops.append(
                    {"action": "disk-read", "start": start, "stop": stop}
                )
                if self.digests is not None:
                    self.digests["disk-load-duration"].add(stop - start)

            logger.debug(
                "Execute key: %s worker: %s", ts.key, self.address
            )  # TODO: comment out?
            assert key == ts.key
            try:
                result = await self.executor_submit(
                    ts.key,
                    apply_function,
                    args=(
                        function,
                        args2,
                        kwargs2,
                        self.execution_state,
                        ts.key,
                        self.active_threads,
                        self.active_threads_lock,
                        self.scheduler_delay,
                    ),
                )
            except RuntimeError as e:
                executor_error = e
                raise

            if ts.state not in ("executing", "long-running"):
                return

            result["key"] = ts.key
            value = result.pop("result", None)
            ts.startstops.append(
                {"action": "compute", "start": result["start"], "stop": result["stop"]}
            )
            self.threads[ts.key] = result["thread"]

            if result["op"] == "task-finished":
                ts.nbytes = result["nbytes"]
                ts.type = result["type"]
                self.transition(ts, "memory", value=value)
                if self.digests is not None:
                    self.digests["task-duration"].add(result["stop"] - result["start"])
            else:
                if isinstance(result.pop("actual-exception"), Reschedule):
                    self.batched_stream.send({"op": "reschedule", "key": ts.key})
                    self.transition(ts, "rescheduled", report=False)
                    self.release_key(ts.key, report=False)
                else:
                    ts.exception = result["exception"]
                    ts.traceback = result["traceback"]
                    logger.warning(
                        " Compute Failed\n"
                        "Function:  %s\n"
                        "args:      %s\n"
                        "kwargs:    %s\n"
                        "Exception: %s\n",
                        str(funcname(function))[:1000],
                        convert_args_to_str(args2, max_len=1000),
                        convert_kwargs_to_str(kwargs2, max_len=1000),
                        repr(result["exception"].data),
                    )
                    self.transition(ts, "error")

            logger.debug("Send compute response to scheduler: %s, %s", ts.key, result)

            if self.validate:
                assert ts.state != "executing"
                assert not ts.waiting_for_data

            await self.ensure_computing()
            self.ensure_communicating()
        except Exception as e:
            if executor_error is e:
                logger.error("Thread Pool Executor error: %s", e)
            else:
                logger.exception(e)
                if LOG_PDB:
                    import pdb

                    pdb.set_trace()
                raise

    ##################
    # Administrative #
    ##################

    async def memory_monitor(self):
        """Track this process's memory usage and act accordingly

        If we rise above 70% memory use, start dumping data to disk.

        If we rise above 80% memory use, stop execution of new tasks
        """
        if self._memory_monitoring:
            return
        self._memory_monitoring = True
        total = 0

        proc = self.monitor.proc
        memory = proc.memory_info().rss
        frac = memory / self.memory_limit

        async def check_pause(memory):
            frac = memory / self.memory_limit
            # Pause worker threads if above 80% memory use
            if self.memory_pause_fraction and frac > self.memory_pause_fraction:
                # Try to free some memory while in paused state
                self._throttled_gc.collect()
                if not self.paused:
                    logger.warning(
                        "Worker is at %d%% memory usage. Pausing worker.  "
                        "Process memory: %s -- Worker memory limit: %s",
                        int(frac * 100),
                        format_bytes(memory),
                        format_bytes(self.memory_limit)
                        if self.memory_limit is not None
                        else "None",
                    )
                    self.paused = True
            elif self.paused:
                logger.warning(
                    "Worker is at %d%% memory usage. Resuming worker. "
                    "Process memory: %s -- Worker memory limit: %s",
                    int(frac * 100),
                    format_bytes(memory),
                    format_bytes(self.memory_limit)
                    if self.memory_limit is not None
                    else "None",
                )
                self.paused = False
                await self.ensure_computing()

        await check_pause(memory)
        # Dump data to disk if above 70%
        if self.memory_spill_fraction and frac > self.memory_spill_fraction:
            logger.debug(
                "Worker is at %d%% memory usage. Start spilling data to disk.",
                int(frac * 100),
            )
            start = time()
            target = self.memory_limit * self.memory_target_fraction
            count = 0
            need = memory - target
            while memory > target:
                if not self.data.fast:
                    logger.warning(
                        "Memory use is high but worker has no data "
                        "to store to disk.  Perhaps some other process "
                        "is leaking memory?  Process memory: %s -- "
                        "Worker memory limit: %s",
                        format_bytes(memory),
                        format_bytes(self.memory_limit)
                        if self.memory_limit is not None
                        else "None",
                    )
                    break
                k, v, weight = self.data.fast.evict()
                del k, v
                total += weight
                count += 1
                # If the current buffer is filled with a lot of small values,
                # evicting one at a time is very slow and the worker might
                # generate new data faster than it is able to evict. Therefore,
                # only pass on control if we spent at least 0.5s evicting
                if time() - start > 0.5:
                    await asyncio.sleep(0)
                    start = time()
                memory = proc.memory_info().rss
                if total > need and memory > target:
                    # Issue a GC to ensure that the evicted data is actually
                    # freed from memory and taken into account by the monitor
                    # before trying to evict even more data.
                    self._throttled_gc.collect()
                    memory = proc.memory_info().rss
            await check_pause(memory)
            if count:
                logger.debug(
                    "Moved %d pieces of data data and %s to disk",
                    count,
                    format_bytes(total),
                )

        self._memory_monitoring = False
        return total

    def cycle_profile(self):
        now = time() + self.scheduler_delay
        prof, self.profile_recent = self.profile_recent, profile.create()
        self.profile_history.append((now, prof))

        self.profile_keys_history.append((now, dict(self.profile_keys)))
        self.profile_keys.clear()

    def trigger_profile(self):
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
        self, comm=None, start=None, stop=None, key=None, server=False
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

    async def get_profile_metadata(self, comm=None, start=0, stop=None):
        if stop is None:
            add_recent = True
        now = time() + self.scheduler_delay
        stop = stop or now
        start = start or 0
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

    def get_call_stack(self, comm=None, keys=None):
        with self.active_threads_lock:
            frames = sys._current_frames()
            active_threads = self.active_threads.copy()
            frames = {k: frames[ident] for ident, k in active_threads.items()}
        if keys is not None:
            frames = {k: frame for k, frame in frames.items() if k in keys}

        result = {k: profile.call_stack(frame) for k, frame in frames.items()}
        return result

    def _notify_plugins(self, method_name, *args, **kwargs):
        for name, plugin in self.plugins.items():
            if hasattr(plugin, method_name):
                try:
                    getattr(plugin, method_name)(*args, **kwargs)
                except Exception:
                    logger.info(
                        "Plugin '%s' failed with exception" % name, exc_info=True
                    )

    ##############
    # Validation #
    ##############

    def validate_task_memory(self, ts):
        assert ts.key in self.data or ts.key in self.actors
        assert ts.nbytes is not None
        assert not ts.waiting_for_data
        assert ts.key not in self.ready
        assert ts.state == "memory"

    def validate_task_executing(self, ts):
        assert ts.state == "executing"
        assert ts.key not in self.data
        assert not ts.waiting_for_data
        assert all(
            dep.key in self.data or dep.key in self.actors for dep in ts.dependencies
        )

    def validate_task_ready(self, ts):
        assert ts.key in pluck(1, self.ready)
        assert ts.key not in self.data
        assert ts.state != "executing"
        assert not ts.waiting_for_data
        assert all(
            dep.key in self.data or dep.key in self.actors for dep in ts.dependencies
        )

    def validate_task_waiting(self, ts):
        assert ts.key not in self.data
        assert ts.state == "waiting"
        if ts.dependencies:
            assert not all(dep.key in self.data for dep in ts.dependencies)

    def validate_task_flight(self, ts):
        assert ts.key not in self.data
        assert not any(dep.key in self.ready for dep in ts.dependents)
        assert ts.key in self.in_flight_workers[ts.coming_from]

    def validate_task(self, ts):
        try:
            if ts.state == "memory":
                self.validate_task_memory(ts)
            elif ts.state == "waiting":
                self.validate_task_waiting(ts)
            elif ts.state == "ready":
                self.validate_task_ready(ts)
            elif ts.state == "executing":
                self.validate_task_executing(ts)
            elif ts.state == "flight":
                self.validate_task_flight(ts)
        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    def validate_state(self):
        if self.status != Status.running:
            return
        try:
            for ts in self.tasks.values():
                assert ts.state is not None
                # check that worker has task
                for worker in ts.who_has:
                    assert ts.key in self.has_what[worker]
                # check that deps have a set state and that dependency<->dependent links are there
                for dep in ts.dependencies:
                    # self.tasks was just a dict of tasks
                    # and this check was originally that the key was in `task_state`
                    # so we may have popped the key out of `self.tasks` but the
                    # dependency can still be in `memory` before GC grabs it...?
                    # Might need better bookkeeping
                    assert dep.state is not None
                    assert ts in dep.dependents
                for key in ts.waiting_for_data:
                    # these are tasks that the current task (ts) needs in memory
                    # before it can proceed. With speculative tasks, the ts_wait
                    # can reasonably include tasks that are "ready" -- they will
                    # be executed here (soon) and then the data will be
                    # available.
                    ts_wait = self.tasks[key]
                    assert (
                        ts_wait.state
                        in ("flight", "waiting", "ready", "executing", "speculative")
                        or ts_wait.key in self._missing_dep_flight
                        or ts_wait.who_has.issubset(self.in_flight_workers)
                    )
                if ts.state == "memory":
                    assert isinstance(ts.nbytes, int)
                    assert not ts.waiting_for_data
                    assert ts.key in self.data or ts.key in self.actors

            for worker, keys in self.has_what.items():
                for k in keys:
                    assert worker in self.tasks[k].who_has

            for ts in self.tasks.values():
                self.validate_task(ts)

        except Exception as e:
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    #######################################
    # Worker Clients (advanced workloads) #
    #######################################

    @property
    def client(self):
        with self._lock:
            if self._client:
                return self._client
            else:
                return self._get_client()

    def _get_client(self, timeout=None):
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
            from .client import default_client

            client = default_client()
        except ValueError:  # no clients found, need to make a new one
            pass
        else:
            if (
                client.scheduler
                and client.scheduler.address == self.scheduler.address
                or client._start_arg == self.scheduler.address
            ):
                self._client = client

        if not self._client:
            from .client import Client

            asynchronous = self.loop is IOLoop.current()
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
            if not asynchronous:
                assert self._client.status == "running"

        return self._client

    def get_current_task(self):
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


def get_worker():
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
            return first(w for w in Worker._instances if w.status == Status.running)
        except StopIteration:
            raise ValueError("No workers found")


def get_client(address=None, timeout=None, resolve_address=True):
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
        address = comm.resolve_address(address)
    try:
        worker = get_worker()
    except ValueError:  # could not find worker
        pass
    else:
        if not address or worker.scheduler.address == address:
            return worker._get_client(timeout=timeout)

    from .client import Client

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
    tpe_secede()  # have this thread secede from the thread pool
    duration = time() - thread_state.start_time
    worker.loop.add_callback(
        worker.maybe_transition_long_running,
        worker.tasks[thread_state.key],
        compute_duration=duration,
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

    pass


def parse_memory_limit(memory_limit, nthreads, total_cores=CPU_COUNT):
    if memory_limit is None:
        return None

    if memory_limit == "auto":
        memory_limit = int(system.MEMORY_LIMIT * min(1, nthreads / total_cores))
    with suppress(ValueError, TypeError):
        memory_limit = float(memory_limit)
        if isinstance(memory_limit, float) and memory_limit <= 1:
            memory_limit = int(memory_limit * system.MEMORY_LIMIT)

    if isinstance(memory_limit, str):
        memory_limit = parse_bytes(memory_limit)
    else:
        memory_limit = int(memory_limit)

    return min(memory_limit, system.MEMORY_LIMIT)


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
    Worker.gather_deps
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
            except KeyError:
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
    """ Load a function from bytes, cache bytes """
    if len(bytes_object) < 100000:
        try:
            result = cache_loads[bytes_object]
        except KeyError:
            result = pickle.loads(bytes_object)
            cache_loads[bytes_object] = result
        return result
    return pickle.loads(bytes_object)


def _deserialize(function=None, args=None, kwargs=None, task=no_value):
    """ Deserialize task inputs and regularize to func, args, kwargs """
    if function is not None:
        function = loads_function(function)
    if args:
        args = pickle.loads(args)
    if kwargs:
        kwargs = pickle.loads(kwargs)

    if task is not no_value:
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


def dumps_function(func):
    """ Dump a function to bytes, cache functions """
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
    """ Dump an object to bytes, warn if those bytes are large """
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
    with active_threads_lock:
        del active_threads[ident]
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


def convert_args_to_str(args, max_len=None):
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


def convert_kwargs_to_str(kwargs, max_len=None):
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


def weight(k, v):
    return sizeof(v)


async def run(server, comm, function, args=(), kwargs=None, is_coro=None, wait=True):
    kwargs = kwargs or {}
    function = pickle.loads(function)
    if is_coro is None:
        is_coro = iscoroutinefunction(function)
    else:
        warnings.warn(
            "The is_coro= parameter is deprecated. "
            "We now automatically detect coroutines/async functions"
        )
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
                server.loop.add_callback(function, *args, **kwargs)
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
    from .diagnostics import nvml
except Exception:
    pass
else:

    @gen.coroutine
    def gpu_metric(worker):
        result = yield offload(nvml.real_time)
        return result

    DEFAULT_METRICS["gpu"] = gpu_metric

    def gpu_startup(worker):
        return nvml.one_time()

    DEFAULT_STARTUP_INFORMATION["gpu"] = gpu_startup

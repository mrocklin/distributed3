from __future__ import annotations

import asyncio
import logging
import os
import ssl
import sys
import tempfile
import threading
import warnings
import weakref
from collections import defaultdict, deque
from collections.abc import Container, Coroutine, Hashable
from contextlib import suppress
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar, final

import tlz
from tornado.httpserver import HTTPServer
from tornado.ioloop import IOLoop

import dask
from dask.utils import parse_timedelta

from distributed import profile
from distributed._async_taskgroup import AsyncTaskGroup
from distributed.comm import get_address_host, get_tcp_server_addresses
from distributed.compatibility import PeriodicCallback
from distributed.core import Server, Status
from distributed.counter import Counter
from distributed.diskutils import WorkDir, WorkSpace
from distributed.http.routing import RoutingApplication
from distributed.metrics import context_meter, time
from distributed.system_monitor import SystemMonitor
from distributed.utils import (
    DequeHandler,
    clean_dashboard_address,
    import_file,
    offload,
    recursive_to_dict,
    wait_for,
    warn_on_duration,
)
from distributed.versions import get_versions

if TYPE_CHECKING:
    from typing_extensions import ParamSpec

    from distributed.counter import Digest

    P = ParamSpec("P")
    R = TypeVar("R")
    T = TypeVar("T")
    Coro = Coroutine[Any, Any, T]


logger = logging.getLogger(__name__)
tick_maximum_delay = parse_timedelta(
    dask.config.get("distributed.admin.tick.limit"), default="ms"
)


class Node:
    _startup_lock: asyncio.Lock
    __startup_exc: Exception | None
    local_directory: str
    monitor: SystemMonitor

    periodic_callbacks: dict[str, PeriodicCallback]
    digests: defaultdict[Hashable, Digest] | None
    digests_total: defaultdict[Hashable, float]
    digests_total_since_heartbeat: defaultdict[Hashable, float]
    digests_max: defaultdict[Hashable, float]

    _last_tick: float
    _tick_counter: int
    _last_tick_counter: int
    _tick_interval: float
    _tick_interval_observed: float

    _original_local_dir: str
    _updated_sys_path: bool
    _workspace: WorkSpace
    _workdir: None | WorkDir
    _ongoing_background_tasks: AsyncTaskGroup

    _event_finished: asyncio.Event

    _is_finalizing: staticmethod[[], bool] = staticmethod(sys.is_finalizing)

    def __init__(
        self,
        local_directory=None,
        needs_workdir=True,
    ):
        if local_directory is None:
            local_directory = (
                dask.config.get("temporary-directory") or tempfile.gettempdir()
            )

        if "dask-scratch-space" not in str(local_directory):
            local_directory = os.path.join(local_directory, "dask-scratch-space")
        self.monitor = SystemMonitor()

        self._original_local_dir = local_directory
        self._ongoing_background_tasks = AsyncTaskGroup()
        with warn_on_duration(
            "1s",
            "Creating scratch directories is taking a surprisingly long time. ({duration:.2f}s) "
            "This is often due to running workers on a network file system. "
            "Consider specifying a local-directory to point workers to write "
            "scratch data to a local disk.",
        ):
            self._workspace = WorkSpace(local_directory)

            if not needs_workdir:  # eg. Nanny will not need a WorkDir
                self._workdir = None
                self.local_directory = self._workspace.base_dir
            else:
                name = type(self).__name__.lower()
                self._workdir = self._workspace.new_work_dir(prefix=f"{name}-")
                self.local_directory = self._workdir.dir_path

        self._updated_sys_path = False
        if self.local_directory not in sys.path:
            sys.path.insert(0, self.local_directory)
            self._updated_sys_path = True

        self.io_loop = self.loop = IOLoop.current()

        if not hasattr(self.io_loop, "profile"):
            if dask.config.get("distributed.worker.profile.enabled"):
                ref = weakref.ref(self.io_loop)

                def stop() -> bool:
                    loop = ref()
                    return loop is None or loop.asyncio_loop.is_closed()

                self.io_loop.profile = profile.watch(
                    omit=("profile.py", "selectors.py"),
                    interval=dask.config.get("distributed.worker.profile.interval"),
                    cycle=dask.config.get("distributed.worker.profile.cycle"),
                    stop=stop,
                )
            else:
                self.io_loop.profile = deque()

        self.periodic_callbacks = {}

        # Statistics counters for various events
        try:
            from distributed.counter import Digest

            self.digests = defaultdict(Digest)
        except ImportError:
            self.digests = None

        # Also log cumulative totals (reset at server restart)
        # and local maximums (reset by prometheus poll)
        # Don't cast int metrics to float
        self.digests_total = defaultdict(int)
        self.digests_total_since_heartbeat = defaultdict(int)
        self.digests_max = defaultdict(int)

        self.counters = defaultdict(Counter)
        pc = PeriodicCallback(self._shift_counters, 5000)
        self.periodic_callbacks["shift_counters"] = pc

        pc = PeriodicCallback(
            self.monitor.update,
            parse_timedelta(
                dask.config.get("distributed.admin.system-monitor.interval")
            )
            * 1000,
        )
        self.periodic_callbacks["monitor"] = pc

        self.__startup_exc = None
        self._startup_lock = asyncio.Lock()

        self._last_tick = time()
        self._tick_counter = 0
        self._last_tick_counter = 0
        self._last_tick_cycle = time()
        self._tick_interval = parse_timedelta(
            dask.config.get("distributed.admin.tick.interval"), default="ms"
        )
        self._tick_interval_observed = self._tick_interval
        self.periodic_callbacks["tick"] = PeriodicCallback(
            self._measure_tick, self._tick_interval * 1000
        )
        self.periodic_callbacks["ticks"] = PeriodicCallback(
            self._cycle_ticks,
            parse_timedelta(dask.config.get("distributed.admin.tick.cycle")) * 1000,
        )

        self.thread_id = 0

        def set_thread_ident():
            self.thread_id = threading.get_ident()

        self.io_loop.add_callback(set_thread_ident)
        self.status = Status.init

    def __await__(self):
        async def _():
            await self.start()
            return self

        return _().__await__()

    async def __aenter__(self):
        await self
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.close()

    async def close(self, reason: str | None = None) -> None:
        for pc in self.periodic_callbacks.values():
            pc.stop()

        self.monitor.close()
        await self._ongoing_background_tasks.stop()
        if self._workdir is not None:
            self._workdir.release()

        # Remove scratch directory from global sys.path
        if self._updated_sys_path and sys.path[0] == self.local_directory:
            sys.path.remove(self.local_directory)

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
            except Exception as e:
                logger.exception(e)
                raise e

        return {"status": "OK", "nbytes": len(data)}

    def _shift_counters(self):
        for counter in self.counters.values():
            counter.shift()
        if self.digests is not None:
            for digest in self.digests.values():
                digest.shift()

    def start_periodic_callbacks(self):
        """Start Periodic Callbacks consistently

        This starts all PeriodicCallbacks stored in self.periodic_callbacks if
        they are not yet running. It does this safely by checking that it is using the
        correct event loop.
        """
        if self.io_loop.asyncio_loop is not asyncio.get_running_loop():
            raise RuntimeError(f"{self!r} is bound to a different event loop")

        self._last_tick = time()
        for pc in self.periodic_callbacks.values():
            if not pc.is_running():
                pc.start()

    def versions(self, packages=None):
        return get_versions(packages=packages)

    def start_services(self, default_listen_ip):
        if default_listen_ip == "0.0.0.0":
            default_listen_ip = ""  # for IPV6

        for k, v in self.service_specs.items():
            listen_ip = None
            if isinstance(k, tuple):
                k, port = k
            else:
                port = 0

            if isinstance(port, str):
                port = port.split(":")

            if isinstance(port, (tuple, list)):
                if len(port) == 2:
                    listen_ip, port = (port[0], int(port[1]))
                elif len(port) == 1:
                    [listen_ip], port = port, 0
                else:
                    raise ValueError(port)

            if isinstance(v, tuple):
                v, kwargs = v
            else:
                kwargs = {}

            try:
                service = v(self, io_loop=self.loop, **kwargs)
                service.listen(
                    (listen_ip if listen_ip is not None else default_listen_ip, port)
                )
                self.services[k] = service
            except Exception as e:
                warnings.warn(
                    f"\nCould not launch service '{k}' on port {port}. "
                    + "Got the following message:\n\n"
                    + str(e),
                    stacklevel=3,
                )

    def stop_services(self):
        if hasattr(self, "http_application"):
            for application in self.http_application.applications:
                if hasattr(application, "stop") and callable(application.stop):
                    application.stop()
        for service in self.services.values():
            service.stop()

    @property
    def service_ports(self):
        return {k: v.port for k, v in self.services.items()}

    def _setup_logging(self, logger: logging.Logger) -> None:
        self._deque_handler = DequeHandler(
            n=dask.config.get("distributed.admin.log-length")
        )
        self._deque_handler.setFormatter(
            logging.Formatter(dask.config.get("distributed.admin.log-format"))
        )
        logger.addHandler(self._deque_handler)
        weakref.finalize(self, logger.removeHandler, self._deque_handler)

    def get_logs(self, start=0, n=None, timestamps=False):
        """
        Fetch log entries for this node

        Parameters
        ----------
        start : float, optional
            A time (in seconds) to begin filtering log entries from
        n : int, optional
            Maximum number of log entries to return from filtered results
        timestamps : bool, default False
            Do we want log entries to include the time they were generated?

        Returns
        -------
        List of tuples containing the log level, message, and (optional) timestamp for each filtered entry, newest first
        """
        deque_handler = self._deque_handler

        L = []
        for count, msg in enumerate(reversed(deque_handler.deque)):
            if n and count >= n or msg.created < start:
                break
            if timestamps:
                L.append((msg.created, msg.levelname, deque_handler.format(msg)))
            else:
                L.append((msg.levelname, deque_handler.format(msg)))
        return L

    def start_http_server(
        self, routes, dashboard_address, default_port=0, ssl_options=None
    ):
        """This creates an HTTP Server running on this node"""

        self.http_application = RoutingApplication(routes)

        # TLS configuration
        tls_key = dask.config.get("distributed.scheduler.dashboard.tls.key")
        tls_cert = dask.config.get("distributed.scheduler.dashboard.tls.cert")
        tls_ca_file = dask.config.get("distributed.scheduler.dashboard.tls.ca-file")
        if tls_cert:
            ssl_options = ssl.create_default_context(
                cafile=tls_ca_file, purpose=ssl.Purpose.CLIENT_AUTH
            )
            ssl_options.load_cert_chain(tls_cert, keyfile=tls_key)

        self.http_server = HTTPServer(self.http_application, ssl_options=ssl_options)

        http_addresses = clean_dashboard_address(dashboard_address or default_port)

        for http_address in http_addresses:
            if http_address["address"] is None:
                address = self._start_address
                if isinstance(address, (list, tuple)):
                    address = address[0]
                if address:
                    with suppress(ValueError):
                        http_address["address"] = get_address_host(address)

            change_port = False
            retries_left = 3
            while True:
                try:
                    if not change_port:
                        self.http_server.listen(**http_address)
                    else:
                        self.http_server.listen(**tlz.merge(http_address, {"port": 0}))
                    break
                except Exception:
                    change_port = True
                    retries_left = retries_left - 1
                    if retries_left < 1:
                        raise

        bound_addresses = get_tcp_server_addresses(self.http_server)

        # If more than one address is configured we just use the first here
        self.http_server.address, self.http_server.port = bound_addresses[0]
        self.services["dashboard"] = self.http_server

        # Warn on port changes
        for expected, actual in zip(
            [a["port"] for a in http_addresses], [b[1] for b in bound_addresses]
        ):
            if expected != actual and expected > 0:
                warnings.warn(
                    f"Port {expected} is already in use.\n"
                    "Perhaps you already have a cluster running?\n"
                    f"Hosting the HTTP server on port {actual} instead"
                )

    def _measure_tick(self):
        now = time()
        tick_duration = now - self._last_tick
        self._last_tick = now
        self._tick_counter += 1
        # This metric is exposed in Prometheus and is reset there during
        # collection
        if tick_duration > tick_maximum_delay:
            logger.info(
                "Event loop was unresponsive in %s for %.2fs.  "
                "This is often caused by long-running GIL-holding "
                "functions or moving large chunks of data. "
                "This can cause timeouts and instability.",
                type(self).__name__,
                tick_duration,
            )
        self.digest_metric("tick-duration", tick_duration)

    def _cycle_ticks(self):
        if not self._tick_counter:
            return
        now = time()
        last_tick_cycle, self._last_tick_cycle = self._last_tick_cycle, now
        count = self._tick_counter - self._last_tick_counter
        self._last_tick_counter = self._tick_counter
        self._tick_interval_observed = (now - last_tick_cycle) / (count or 1)

    def digest_metric(self, name: Hashable, value: float) -> None:
        # Granular data (requires crick)
        if self.digests is not None:
            self.digests[name].add(value)
        # Cumulative data (reset by server restart)
        self.digests_total[name] += value
        # Cumulative data sent to scheduler and reset on heartbeat
        self.digests_total_since_heartbeat[name] += value
        # Local maximums (reset by Prometheus poll)
        self.digests_max[name] = max(self.digests_max[name], value)

    @property
    def status(self) -> Status:
        try:
            return self._status
        except AttributeError:
            return Status.undefined

    @status.setter
    def status(self, value: Status) -> None:
        if not isinstance(value, Status):
            raise TypeError(f"Expected Status; got {value!r}")
        self._status = value

    async def start_unsafe(self):
        """Attempt to start the server. This is not idempotent and not protected against concurrent startup attempts.

        This is intended to be overwritten or called by subclasses. For a safe
        startup, please use ``Node.start`` instead.

        If ``death_timeout`` is configured, we will require this coroutine to
        finish before this timeout is reached. If the timeout is reached we will
        close the instance and raise an ``asyncio.TimeoutError``
        """
        return self

    @final
    async def start(self):
        async with self._startup_lock:
            if self.status == Status.failed:
                assert self.__startup_exc is not None
                raise self.__startup_exc
            elif self.status != Status.init:
                return self
            timeout = getattr(self, "death_timeout", None)

            async def _close_on_failure(exc: Exception) -> None:
                await self.close(reason=f"failure-to-start-{str(type(exc))}")
                self.status = Status.failed
                self.__startup_exc = exc

            try:
                await wait_for(self.start_unsafe(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                await _close_on_failure(exc)
                raise asyncio.TimeoutError(
                    f"{type(self).__name__} start timed out after {timeout}s."
                ) from exc
            except Exception as exc:
                await _close_on_failure(exc)
                raise RuntimeError(f"{type(self).__name__} failed to start.") from exc
            if self.status == Status.init:
                self.status = Status.running
        return self


class ServerNode(Node):
    def __init__(
        self,
        handlers,
        blocked_handlers=None,
        stream_handlers=None,
        connection_limit=512,
        deserialize=True,
        serializers=None,
        deserializers=None,
        connection_args=None,
        timeout=None,
        local_directory=None,
        needs_workdir=True,
    ):
        self._event_finished = asyncio.Event()

        _handlers = {
            "dump_state": self._to_dict,
            "identity": self.identity,
        }
        if handlers:
            _handlers.update(handlers)
        import uuid

        self.id = type(self).__name__ + "-" + str(uuid.uuid4())

        if blocked_handlers is None:
            blocked_handlers = dask.config.get(
                "distributed.%s.blocked-handlers" % type(self).__name__.lower(), []
            )
        self.server = Server(
            handlers=_handlers,
            blocked_handlers=blocked_handlers,
            stream_handlers=stream_handlers,
            connection_limit=connection_limit,
            deserialize=deserialize,
            serializers=serializers,
            deserializers=deserializers,
            connection_args=connection_args,
            timeout=timeout,
        )
        super().__init__(
            local_directory=local_directory,
            needs_workdir=needs_workdir,
        )

    def identity(self) -> dict[str, str]:
        return {"type": type(self).__name__, "id": self.id}

    @property
    def port(self):
        return self.server.port

    @property
    def listen_address(self):
        return self.server.address

    @property
    def address(self):
        return self.server.address

    @property
    def address_safe(self):
        return self.server.address_safe

    async def start_unsafe(self):
        await self.server
        await super().start_unsafe()
        return self

    async def finished(self) -> None:
        """Wait until the server has finished"""
        await self._event_finished.wait()

    async def close(self, reason: str | None = None) -> None:
        try:
            # Close network connections and background tasks
            await self.server.close()
            await Node.close(self, reason=reason)
            self.status = Status.closed
        finally:
            self._event_finished.set()

    def _to_dict(self, *, exclude: Container[str] = ()) -> dict[str, Any]:
        """Dictionary representation for debugging purposes.
        Not type stable and not intended for roundtrips.

        See also
        --------
        Server.identity
        Client.dump_cluster_state
        distributed.utils.recursive_to_dict
        """
        info: dict[str, Any] = self.identity()
        extra = {
            "address": self.server.address,
            "status": self.status.name,
            "thread_id": self.thread_id,
        }
        info.update(extra)
        info = {k: v for k, v in info.items() if k not in exclude}
        return recursive_to_dict(info, exclude=exclude)


def context_meter_to_node_digest(digest_tag: str) -> Callable:
    """Decorator for an async method of a Node subclass that calls
    ``distributed.metrics.context_meter.meter`` and/or ``digest_metric``.
    It routes the calls from ``context_meter.digest_metric(label, value, unit)`` to
    ``Node.digest_metric((digest_tag, label, unit), value)``.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self: Node, *args: Any, **kwargs: Any) -> Any:
            def metrics_callback(label: Hashable, value: float, unit: str) -> None:
                if not isinstance(label, tuple):
                    label = (label,)
                name = (digest_tag, *label, unit)
                self.digest_metric(name, value)

            with context_meter.add_callback(metrics_callback, allow_offload=True):
                return await func(self, *args, **kwargs)

        return wrapper

    return decorator

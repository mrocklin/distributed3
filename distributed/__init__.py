from distributed import config  # isort:skip; load distributed configuration first
from distributed import widgets  # isort:skip; load distributed widgets second

import atexit

import dask
from dask.config import config  # type: ignore

from distributed._version import get_versions
from distributed.actor import Actor, ActorFuture, BaseActorFuture
from distributed.client import (
    Client,
    CompatibleExecutor,
    Future,
    as_completed,
    default_client,
    fire_and_forget,
    futures_of,
    get_task_metadata,
    get_task_stream,
    performance_report,
    wait,
)
from distributed.core import Status, connect, rpc
from distributed.deploy import Adaptive, LocalCluster, SpecCluster, SSHCluster
from distributed.diagnostics.plugin import (
    Environ,
    NannyPlugin,
    PipInstall,
    SchedulerPlugin,
    UploadDirectory,
    UploadFile,
    WorkerPlugin,
)
from distributed.diagnostics.progressbar import progress
from distributed.event import Event
from distributed.lock import Lock
from distributed.multi_lock import MultiLock
from distributed.nanny import Nanny
from distributed.pubsub import Pub, Sub
from distributed.queues import Queue
from distributed.scheduler import KilledWorker, Scheduler
from distributed.security import Security
from distributed.semaphore import Semaphore
from distributed.utils import CancelledError, TimeoutError, sync
from distributed.variable import Variable
from distributed.worker import (
    Reschedule,
    Worker,
    get_client,
    get_worker,
    print,
    secede,
    warn,
)
from distributed.worker_client import local_client, worker_client


def __getattr__(name):
    global __version__, __git_revision__

    if name == "__version__":
        from importlib.metadata import version

        __version__ = version("distributed")
        return __version__

    if name == "__git_revision__":
        from distributed._version import get_versions

        __git_revision__ = get_versions()["full-revisionid"]
        return __git_revision__

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


_python_shutting_down = False


@atexit.register
def _():
    """Set a global when Python shuts down.

    Note
    ----
    This function must be registered with atexit *after* any class that invokes
    ``dstributed.utils.is_python_shutting_down`` has been defined. This way it
    will be called before the ``__del__`` method of those classes.

    See Also
    --------
    distributed.utils.is_python_shutting_down
    """
    global _python_shutting_down
    _python_shutting_down = True

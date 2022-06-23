import warnings
from contextlib import contextmanager

import dask

from distributed.metrics import time
from distributed.worker import get_client, get_worker, thread_state
from distributed.worker_state_machine import SecedeEvent


@contextmanager
def worker_client(timeout=None):
    """Get client for this thread

    This context manager is intended to be called within functions that we run
    on workers.  When run as a context manager it delivers a client
    ``Client`` object that can submit other tasks directly from that worker.

    Parameters
    ----------
    timeout : Number or String
        Timeout after which to error out. Defaults to the
        ``distributed.comm.timeouts.connect`` configuration value.


    Examples
    --------
    >>> def func(x):
    ...     with worker_client(timeout="10s") as c:  # connect from worker back to scheduler
    ...         a = c.submit(inc, x)     # this task can submit more tasks
    ...         b = c.submit(dec, x)
    ...         result = c.gather([a, b])  # and gather results
    ...     return result

    >>> future = client.submit(func, 1)  # submit func(1) on cluster

    See Also
    --------
    get_worker
    get_client
    secede
    """

    if timeout is None:
        timeout = dask.config.get("distributed.comm.timeouts.connect")

    timeout = dask.utils.parse_timedelta(timeout, "s")

    client = get_client(timeout=timeout)
    with secede():
        yield client


@contextmanager
def secede():
    worker = get_worker()
    duration = time() - thread_state.start_time
    worker.loop.add_callback(
        worker.handle_stimulus,
        SecedeEvent(
            key=thread_state.key,
            compute_duration=duration,
            stimulus_id=f"worker-client-secede-{time()}",
        ),
    )

    yield

    # FIXME: handle_stimulus_rejoin see https://github.com/dask/distributed/issues/5882


def local_client(*args, **kwargs):
    warnings.warn("local_client has moved to worker_client")
    return worker_client(*args, **kwargs)

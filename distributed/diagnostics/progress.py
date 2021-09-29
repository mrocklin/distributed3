import asyncio
import logging
import time
from collections import defaultdict
from timeit import default_timer
from typing import Dict, List, Tuple

from tlz import groupby, valmap
from tornado.ioloop import PeriodicCallback

from dask.base import tokenize
from dask.utils import stringify

from ..utils import key_split, key_split_group, log_errors
from .plugin import SchedulerPlugin

logger = logging.getLogger(__name__)


def dependent_keys(tasks, complete=False):
    """
    All keys that need to compute for these keys to finish.

    If *complete* is false, omit tasks that are busy processing or
    have finished executing.
    """
    out = set()
    errors = set()
    stack = list(tasks)
    while stack:
        ts = stack.pop()
        key = ts.key
        if key in out:
            continue
        if not complete and ts.who_has:
            continue
        if ts.exception is not None:
            errors.add(key)
            if not complete:
                continue

        out.add(key)
        stack.extend(ts.dependencies)
    return out, errors


class Progress(SchedulerPlugin):
    """Tracks progress of a set of keys or futures

    On creation we provide a set of keys or futures that interest us as well as
    a scheduler.  We traverse through the scheduler's dependencies to find all
    relevant keys on which our keys depend.  We then plug into the scheduler to
    learn when our keys become available in memory at which point we record
    their completion.

    State
    -----
    keys: set
        Set of keys that are not yet computed
    all_keys: set
        Set of all keys that we track

    This class performs no visualization.  However it is used by other classes,
    notably TextProgressBar and ProgressWidget, which do perform visualization.
    """

    def __init__(self, keys, scheduler, minimum=0, dt=0.1, complete=False, name=None):
        self.name = name or f"progress-{tokenize(keys, minimum, dt, complete)}"
        self.keys = {k.key if hasattr(k, "key") else k for k in keys}
        self.keys = {stringify(k) for k in self.keys}
        self.scheduler = scheduler
        self.complete = complete
        self._minimum = minimum
        self._dt = dt
        self.last_duration = 0
        self._start_time = default_timer()
        self._running = False
        self.status = None
        self.extra = {}

    async def setup(self):
        keys = self.keys

        while not keys.issubset(self.scheduler.tasks):
            await asyncio.sleep(0.05)

        tasks = [self.scheduler.tasks[k] for k in keys]

        self.keys = None

        self.scheduler.add_plugin(self)  # subtle race condition here
        self.all_keys, errors = dependent_keys(tasks, complete=self.complete)
        if not self.complete:
            self.keys = self.all_keys.copy()
        else:
            self.keys, _ = dependent_keys(tasks, complete=False)
        self.all_keys.update(keys)
        self.keys |= errors & self.all_keys

        if not self.keys:
            self.stop(exception=None, key=None)

        logger.debug("Set up Progress keys")

        for k in errors:
            self.transition(k, None, "erred", exception=True)

    def transition(self, key, start, finish, *args, **kwargs):
        if key in self.keys and start == "processing" and finish == "memory":
            logger.debug("Progress sees key %s", key)
            self.keys.remove(key)

            if not self.keys:
                self.stop()

        if key in self.all_keys and finish == "erred":
            logger.debug("Progress sees task erred")
            self.stop(exception=kwargs["exception"], key=key)

        if key in self.keys and finish == "forgotten":
            logger.debug("A task was cancelled (%s), stopping progress", key)
            self.stop(exception=True, key=key)

    def restart(self, scheduler):
        self.stop()

    def stop(self, exception=None, key=None):
        if self.name in self.scheduler.plugins:
            self.scheduler.remove_plugin(name=self.name)
        if exception:
            self.status = "error"
            self.extra.update(
                {"exception": self.scheduler.tasks[key].exception, "key": key}
            )
        else:
            self.status = "finished"
        logger.debug("Remove Progress plugin")


class MultiProgress(Progress):
    """Progress variant that keeps track of different groups of keys

    See Progress for most details.  This only adds a function ``func=``
    that splits keys.  This defaults to ``key_split`` which aligns with naming
    conventions chosen in the dask project (tuples, hyphens, etc..)

    State
    -----
    keys: dict
        Maps group name to set of not-yet-complete keys for that group
    all_keys: dict
        Maps group name to set of all keys for that group

    Examples
    --------
    >>> split = lambda s: s.split('-')[0]
    >>> p = MultiProgress(['y-2'], func=split)  # doctest: +SKIP
    >>> p.keys   # doctest: +SKIP
    {'x': {'x-1', 'x-2', 'x-3'},
     'y': {'y-1', 'y-2'}}
    """

    def __init__(
        self, keys, scheduler=None, func=key_split, minimum=0, dt=0.1, complete=False
    ):
        self.func = func
        name = f"multi-progress-{tokenize(keys, func, minimum, dt, complete)}"
        super().__init__(
            keys, scheduler, minimum=minimum, dt=dt, complete=complete, name=name
        )

    async def setup(self):
        keys = self.keys

        while not keys.issubset(self.scheduler.tasks):
            await asyncio.sleep(0.05)

        tasks = [self.scheduler.tasks[k] for k in keys]

        self.keys = None

        self.scheduler.add_plugin(self)  # subtle race condition here
        self.all_keys, errors = dependent_keys(tasks, complete=self.complete)
        if not self.complete:
            self.keys = self.all_keys.copy()
        else:
            self.keys, _ = dependent_keys(tasks, complete=False)
        self.all_keys.update(keys)
        self.keys |= errors & self.all_keys

        if not self.keys:
            self.stop(exception=None, key=None)

        # Group keys by func name
        self.keys = valmap(set, groupby(self.func, self.keys))
        self.all_keys = valmap(set, groupby(self.func, self.all_keys))
        for k in self.all_keys:
            if k not in self.keys:
                self.keys[k] = set()

        for k in errors:
            self.transition(k, None, "erred", exception=True)
        logger.debug("Set up Progress keys")

    def transition(self, key, start, finish, *args, **kwargs):
        if start == "processing" and finish == "memory":
            s = self.keys.get(self.func(key), None)
            if s and key in s:
                s.remove(key)

            if not self.keys or not any(self.keys.values()):
                self.stop()

        if finish == "erred":
            logger.debug("Progress sees task erred")
            k = self.func(key)
            if k in self.all_keys and key in self.all_keys[k]:
                self.stop(exception=kwargs.get("exception"), key=key)

        if finish == "forgotten":
            k = self.func(key)
            if k in self.all_keys and key in self.all_keys[k]:
                logger.debug("A task was cancelled (%s), stopping progress", key)
                self.stop(exception=True)


def format_time(t):
    """Format seconds into a human readable form.

    >>> format_time(10.4)
    '10.4s'
    >>> format_time(1000.4)
    '16min 40.4s'
    >>> format_time(100000.4)
    '27hr 46min 40.4s'
    """
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:2.0f}hr {m:2.0f}min {s:4.1f}s"
    elif m:
        return f"{m:2.0f}min {s:4.1f}s"
    else:
        return f"{s:4.1f}s"


class AllProgress(SchedulerPlugin):
    """Keep track of all keys, grouped by key_split"""

    name = "all-progress"

    def __init__(self, scheduler):
        self.all = defaultdict(set)
        self.nbytes = defaultdict(lambda: 0)
        self.state = defaultdict(lambda: defaultdict(set))
        self.scheduler = scheduler

        for ts in self.scheduler.tasks.values():
            key = ts.key
            prefix = ts.prefix.name
            self.all[prefix].add(key)
            self.state[ts.state][prefix].add(key)
            if ts.nbytes >= 0:
                self.nbytes[prefix] += ts.nbytes

        scheduler.add_plugin(self)

    def transition(self, key, start, finish, *args, **kwargs):
        ts = self.scheduler.tasks[key]
        prefix = ts.prefix.name
        self.all[prefix].add(key)
        try:
            self.state[start][prefix].remove(key)
        except KeyError:  # TODO: remove me once we have a new or clean state
            pass

        if start == "memory" and ts.nbytes >= 0:
            # XXX why not respect DEFAULT_DATA_SIZE?
            self.nbytes[prefix] -= ts.nbytes
        if finish == "memory" and ts.nbytes >= 0:
            self.nbytes[prefix] += ts.nbytes

        if finish != "forgotten":
            self.state[finish][prefix].add(key)
        else:
            s = self.all[prefix]
            s.remove(key)
            if not s:
                del self.all[prefix]
                self.nbytes.pop(prefix, None)
                for v in self.state.values():
                    v.pop(prefix, None)

    def restart(self, scheduler):
        self.all.clear()
        self.state.clear()


class GroupTiming(SchedulerPlugin):
    """Keep track of high-level timing information for task group progress"""

    name = "group-timing"

    def __init__(self, scheduler):
        self.scheduler = scheduler
        # Time series of task states
        self.states: Dict[str, List[Tuple[float, Dict[str, int]]]] = dict()
        # Time series of task durations
        self.all_durations: Dict[str, List[Tuple[float, Dict[str, float]]]] = dict()
        # Time series of bytes stored per task group
        self.nbytes: Dict[str, List[Tuple[float, int]]] = dict()
        # Time series of the number of threads on the scheduler aligned to group checkpoints
        self.nthreads_group: Dict[str, List[Tuple[float, int]]] = dict()
        # Overall time series of the number of threads on the scheduler
        self.nthreads: List[Tuple[float, int]] = list()
        # We snapshot the task state after every `delta` number of
        # tasks are completed.
        self._deltas: Dict[str, int] = dict()

        t = time.time()
        for name, group in self.scheduler.task_groups.items():
            self.create(name, group)
            self.insert(t, name, group)

        scheduler.add_plugin(self)

        nthreads_pc = PeriodicCallback(self.track_threads, 1.0 * 1000.0)
        self.scheduler.periodic_callbacks["nthreads_pc"] = nthreads_pc
        nthreads_pc.start()

    def create(self, name, group):
        """
        Set up timeseries data for a new task group
        """
        with log_errors():
            self.states[name] = []
            self.nbytes[name] = []
            self.nthreads_group[name] = []
            self.all_durations[name] = []
            # Snapshot states roughly after every 1% of tasks are completed.
            # This could conceivably change if new tasks are added after delta
            # is computed, so it's meant to be approximate.
            n_tasks = sum(group.states.values())
            delta = max(int(n_tasks / 100), 1)
            self._deltas[name] = delta

    def insert(self, timestamp, name, group):
        """
        Append a new entry to our tasks timeseries.
        """
        with log_errors():
            if not len(self.states[name]):
                # If the timeseries is empty, just insert the current value.
                self.states[name].append((timestamp, group.states.copy()))
                self.all_durations[name].append((timestamp, group.all_durations.copy()))
                self.nbytes[name].append((timestamp, group.nbytes_total))
                self.nthreads_group[name].append(
                    (timestamp, self.scheduler.total_nthreads)
                )
            else:
                # If the timeseries exists, we check the most recent entry,
                # and determine whether `delta` tasks have been completed since then.
                prev = self.states[name][-1]
                pstates = prev[1]
                pcount = pstates["erred"] + pstates["memory"] + pstates["released"]

                states = group.states
                count = states["erred"] + states["memory"] + states["released"]

                if count - pcount >= self._deltas[name]:
                    self.states[name].append((timestamp, group.states.copy()))
                    self.all_durations[name].append(
                        (timestamp, group.all_durations.copy())
                    )
                    self.nbytes[name].append((timestamp, group.nbytes_total))

    def track_threads(self):
        self.nthreads.append((time.time(), self.scheduler.total_nthreads))

    def transition(self, key, start, finish, *args, **kwargs):
        # We mostly are interested in when tasks move from processing to memory or err,
        # so we only check if the transition starts from that state.
        if start == "processing":
            with log_errors():
                name = key_split_group(key)
                group = self.scheduler.task_groups[name]
                t = time.time()
                if name not in self.states:
                    self.create(name, group)
                self.insert(t, name, group)

    def restart(self, scheduler):
        self.states.clear()
        self.nbytes.clear()
        self.all_durations.clear()
        self.nthreads.clear()
        self.nthreads_group.clear()
        self._deltas.clear()

    async def close(self):
        nthreads_pc = self.scheduler.periodic_callbacks.pop("progress_nthreads", None)
        if nthreads_pc:
            nthreads_pc.stop()

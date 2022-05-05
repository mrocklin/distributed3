from __future__ import annotations

import asyncio
import ctypes
import random
import sys
import time
from typing import Literal

from dask.utils import parse_timedelta

from distributed.diagnostics.plugin import WorkerPlugin
from distributed.utils_test import hold_gil


class BlockGIL(WorkerPlugin):
    def __init__(
        self,
        interval: str | int | float = "100 s",
        duration: str | int | float = "1 s",
    ):
        self.interval = parse_timedelta(interval)
        self.duration = parse_timedelta(duration)
        self.stop = False

    async def setup(self, worker):
        self.worker = worker
        loop = self.worker.loop.asyncio_loop

        def _hold_gil():

            time.sleep(random.expovariate(1 / self.interval))
            hold_gil(self.duration)
            if self.stop:
                return
            loop.call_soon(_hold_gil)

        _hold_gil()

    def teardown(self):
        self.stop = True


class KillWorker(WorkerPlugin):
    """Kill Workers Randomly

    This kills workers in a cluster randomly.  It is intended to be used in
    stress testing.

    Parameters
    ----------
    delay: str
        The expected amount of time for a worker to live.
        The actual time will vary, treating worker death as a poisson process.
    mode: str
        or "graceful" which calls worker.close(...)
        Either "sys.exit" which calls sys.exit(0)
        or "segfault" which triggers a segfault
    """

    def __init__(
        self,
        delay: str | int | float = "100 s",
        mode: Literal["sys.exit", "graceful", "segfault"] = "sys.exit",
    ):
        self.delay = parse_timedelta(delay)
        if mode not in ("sys.exit", "graceful", "segfault"):
            raise ValueError(
                f"Three modes supported, 'sys.exit', 'graceful', and 'segfault'. "
                f"got {mode!r}"
            )
        self.mode = mode

    async def setup(self, worker):
        self.worker = worker
        if self.mode == "graceful":
            f = self.graceful
        elif self.mode == "sys.exit":
            f = self.sys_exit
        elif self.mode == "segfault":
            f = self.segfault

        self.worker.loop.asyncio_loop.call_later(
            delay=random.expovariate(1 / self.delay),
            callback=f,
        )

    def graceful(self):
        asyncio.create_task(
            self.worker.close(report=False, nanny=False, executor_wait=False)
        )

    def sys_exit(self):
        sys.exit(0)

    def segfault(self):
        """
        Magic, from https://gist.github.com/coolreader18/6dbe0be2ae2192e90e1a809f1624c694?permalink_comment_id=3874116#gistcomment-3874116
        """
        ctypes.string_at(0)

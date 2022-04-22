from __future__ import annotations

import asyncio
import logging
import random
import uuid
from collections import defaultdict, deque
from collections.abc import Container
from functools import partial
from math import log2
from time import time

from tlz import topk
from tornado.ioloop import PeriodicCallback

import dask
from dask.utils import parse_timedelta

from distributed.comm.addressing import get_address_host
from distributed.core import CommClosedError, Status
from distributed.diagnostics.plugin import SchedulerPlugin
from distributed.utils import log_errors, recursive_to_dict

# Stealing requires multiple network bounces and if successful also task
# submission which may include code serialization. Therefore, be very
# conservative in the latency estimation to suppress too aggressive stealing
# of small tasks
LATENCY = 0.1

logger = logging.getLogger(__name__)


LOG_PDB = dask.config.get("distributed.admin.pdb-on-err")

_WORKER_STATE_CONFIRM = {
    "ready",
    "constrained",
    "waiting",
}

_WORKER_STATE_REJECT = {
    "memory",
    "executing",
    "long-running",
    "cancelled",
    "resumed",
}
_WORKER_STATE_UNDEFINED = {
    "released",
    None,
}


class WorkStealing(SchedulerPlugin):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        # { level: { task states } }
        self.stealable_all = [set() for i in range(15)]
        # { worker: { level: { task states } } }
        self.stealable = dict()
        # { task state: (worker, level) }
        self.key_stealable = dict()

        self.cost_multipliers = [1 + 2 ** (i - 6) for i in range(15)]
        self.cost_multipliers[0] = 1

        for worker in scheduler.workers:
            self.add_worker(worker=worker)

        self._callback_time = parse_timedelta(
            dask.config.get("distributed.scheduler.work-stealing-interval"),
            default="ms",
        )
        # `callback_time` is in milliseconds
        self.scheduler.add_plugin(self)
        self.scheduler.events["stealing"] = deque(maxlen=100000)
        self.count = 0
        # { task state: <stealing info dict> }
        self.in_flight = dict()
        # { worker state: occupancy }
        self.in_flight_occupancy = defaultdict(lambda: 0)
        self._in_flight_event = asyncio.Event()

        self.scheduler.stream_handlers["steal-response"] = self.move_task_confirm

    async def start(self, scheduler=None):
        """Start the background coroutine to balance the tasks on the cluster.
        Idempotent.
        The scheduler argument is ignored. It is merely required to satisify the
        plugin interface. Since this class is simultaneouly an extension, the
        scheudler instance is already registered during initialization
        """
        if "stealing" in self.scheduler.periodic_callbacks:
            return
        pc = PeriodicCallback(
            callback=self.balance, callback_time=self._callback_time * 1000
        )
        pc.start()
        self.scheduler.periodic_callbacks["stealing"] = pc
        self._in_flight_event.set()

    async def stop(self):
        """Stop the background task balancing tasks on the cluster.
        This will block until all currently running stealing requests are
        finished. Idempotent
        """
        pc = self.scheduler.periodic_callbacks.pop("stealing", None)
        if pc:
            pc.stop()
        await self._in_flight_event.wait()

    def _to_dict_no_nest(self, *, exclude: Container[str] = ()) -> dict:
        """Dictionary representation for debugging purposes.
        Not type stable and not intended for roundtrips.

        See also
        --------
        Client.dump_cluster_state
        distributed.utils.recursive_to_dict
        """
        return recursive_to_dict(self, exclude=exclude, members=True)

    def log(self, msg):
        return self.scheduler.log_event("stealing", msg)

    def add_worker(self, scheduler=None, worker=None):
        self.stealable[worker] = [set() for i in range(15)]

    def remove_worker(self, scheduler=None, worker=None):
        del self.stealable[worker]

    def teardown(self):
        pcs = self.scheduler.periodic_callbacks
        if "stealing" in pcs:
            pcs["stealing"].stop()
            del pcs["stealing"]

    def transition(
        self, key, start, finish, compute_start=None, compute_stop=None, *args, **kwargs
    ):
        if finish == "processing":
            ts = self.scheduler.tasks[key]
            self.put_key_in_stealable(ts)
        elif start == "processing":
            ts = self.scheduler.tasks[key]
            self.remove_key_from_stealable(ts)
            d = self.in_flight.pop(ts, None)
            if d:
                thief = d["thief"]
                victim = d["victim"]
                self.in_flight_occupancy[thief] -= d["thief_duration"]
                self.in_flight_occupancy[victim] += d["victim_duration"]
                if not self.in_flight:
                    self.in_flight_occupancy.clear()
                    self._in_flight_event.set()

    def recalculate_cost(self, ts):
        if ts not in self.in_flight:
            self.remove_key_from_stealable(ts)
            self.put_key_in_stealable(ts)

    def put_key_in_stealable(self, ts):
        cost_multiplier, level = self.steal_time_ratio(ts)
        if cost_multiplier is not None:
            ws = ts.processing_on
            worker = ws.address
            self.stealable_all[level].add(ts)
            self.stealable[worker][level].add(ts)
            self.key_stealable[ts] = (worker, level)

    def remove_key_from_stealable(self, ts):
        result = self.key_stealable.pop(ts, None)
        if result is None:
            return

        worker, level = result
        try:
            self.stealable[worker][level].remove(ts)
        except KeyError:
            pass
        try:
            self.stealable_all[level].remove(ts)
        except KeyError:
            pass

    def steal_time_ratio(self, ts):
        """The compute to communication time ratio of a key

        Returns
        -------
        cost_multiplier: The increased cost from moving this task as a factor.
        For example a result of zero implies a task without dependencies.
        level: The location within a stealable list to place this value
        """
        if not ts.dependencies:  # no dependencies fast path
            return 0, 0

        split = ts.prefix.name
        if split in fast_tasks:
            return None, None

        ws = ts.processing_on
        compute_time = ws.processing[ts]

        nbytes = ts.get_nbytes_deps()
        transfer_time = nbytes / self.scheduler.bandwidth + LATENCY
        cost_multiplier = transfer_time / compute_time

        level = int(round(log2(cost_multiplier) + 6))
        if level < 1:
            level = 1
        level = min(len(self.cost_multipliers) - 1, level)

        return cost_multiplier, level

    def move_task_request(self, ts, victim, thief) -> str:
        try:
            if ts in self.in_flight:
                return "in-flight"
            # Stimulus IDs are used to verify the response, see
            # `move_task_confirm`. Therefore, this must be truly unique.
            stimulus_id = f"steal-{uuid.uuid4().hex}"

            key = ts.key
            self.remove_key_from_stealable(ts)
            logger.debug(
                "Request move %s, %s: %2f -> %s: %2f",
                key,
                victim,
                victim.occupancy,
                thief,
                thief.occupancy,
            )

            victim_duration = victim.processing[ts]

            thief_duration = self.scheduler.get_task_duration(
                ts
            ) + self.scheduler.get_comm_cost(ts, thief)

            self.scheduler.stream_comms[victim.address].send(
                {"op": "steal-request", "key": key, "stimulus_id": stimulus_id}
            )
            self.in_flight[ts] = {
                "victim": victim,  # guaranteed to be processing_on
                "thief": thief,
                "victim_duration": victim_duration,
                "thief_duration": thief_duration,
                "stimulus_id": stimulus_id,
            }
            self._in_flight_event.clear()

            self.in_flight_occupancy[victim] -= victim_duration
            self.in_flight_occupancy[thief] += thief_duration
            return stimulus_id
        except CommClosedError:
            logger.info("Worker comm %r closed while stealing: %r", victim, ts)
            return "comm-closed"
        except Exception as e:  # pragma: no cover
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise

    async def move_task_confirm(self, *, key, state, stimulus_id, worker=None):
        try:
            ts = self.scheduler.tasks[key]
        except KeyError:
            logger.debug("Key released between request and confirm: %s", key)
            return
        try:
            d = self.in_flight.pop(ts)
            if d["stimulus_id"] != stimulus_id:
                self.log(("stale-response", key, state, worker, stimulus_id))
                self.in_flight[ts] = d
                return
        except KeyError:
            self.log(("already-aborted", key, state, worker, stimulus_id))
            return

        thief = d["thief"]
        victim = d["victim"]

        logger.debug("Confirm move %s, %s -> %s.  State: %s", key, victim, thief, state)

        self.in_flight_occupancy[thief] -= d["thief_duration"]
        self.in_flight_occupancy[victim] += d["victim_duration"]

        if not self.in_flight:
            self.in_flight_occupancy.clear()
            self._in_flight_event.set()

        if self.scheduler.validate:
            assert ts.processing_on == victim

        try:
            _log_msg = [key, state, victim.address, thief.address, stimulus_id]

            if ts.state != "processing":
                self.scheduler._reevaluate_occupancy_worker(thief)
                self.scheduler._reevaluate_occupancy_worker(victim)
            elif (
                state in _WORKER_STATE_UNDEFINED
                or state in _WORKER_STATE_CONFIRM
                and thief.address not in self.scheduler.workers
            ):
                self.log(
                    (
                        "reschedule",
                        thief.address not in self.scheduler.workers,
                        *_log_msg,
                    )
                )
                self.scheduler.reschedule(key)
            # Victim had already started execution
            elif state in _WORKER_STATE_REJECT:
                self.log(("already-computing", *_log_msg))
            # Victim was waiting, has given up task, enact steal
            elif state in _WORKER_STATE_CONFIRM:
                self.remove_key_from_stealable(ts)
                ts.processing_on = thief
                duration = victim.processing.pop(ts)
                victim.occupancy -= duration
                self.scheduler.total_occupancy -= duration
                if not victim.processing:
                    self.scheduler.total_occupancy -= victim.occupancy
                    victim.occupancy = 0
                thief.processing[ts] = d["thief_duration"]
                thief.occupancy += d["thief_duration"]
                self.scheduler.total_occupancy += d["thief_duration"]
                self.put_key_in_stealable(ts)

                self.scheduler.send_task_to_worker(thief.address, ts)
                self.log(("confirm", *_log_msg))
            else:
                raise ValueError(f"Unexpected task state: {state}")
        except Exception as e:  # pragma: no cover
            logger.exception(e)
            if LOG_PDB:
                import pdb

                pdb.set_trace()
            raise
        finally:
            self.scheduler.check_idle_saturated(thief)
            self.scheduler.check_idle_saturated(victim)

    def balance(self):
        s = self.scheduler

        def combined_occupancy(ws):
            return ws.occupancy + self.in_flight_occupancy[ws]

        def maybe_move_task(level, ts, sat, idl, duration, cost_multiplier):
            occ_idl = combined_occupancy(idl)
            occ_sat = combined_occupancy(sat)

            if occ_idl + cost_multiplier * duration <= occ_sat - duration / 2:
                self.move_task_request(ts, sat, idl)
                log.append(
                    # The format of this message is tightly coupled to the dashboard
                    (
                        start,
                        level,
                        ts.key,
                        duration,
                        sat.address,
                        occ_sat,
                        idl.address,
                        occ_idl,
                    )
                )
                s.check_idle_saturated(sat, occ=occ_sat)
                s.check_idle_saturated(idl, occ=occ_idl)

        with log_errors():
            i = 0
            # Paused and closing workers must never become thieves
            idle = [ws for ws in s.idle.values() if ws.status == Status.running]
            if not idle or len(idle) == len(s.workers):
                return

            log = []
            start = time()

            saturated = s.saturated
            if not saturated:
                saturated = topk(10, s.workers.values(), key=combined_occupancy)
                saturated = [
                    ws
                    for ws in saturated
                    if combined_occupancy(ws) > 0.2 and len(ws.processing) > ws.nthreads
                ]
            elif len(saturated) < 20:
                saturated = sorted(saturated, key=combined_occupancy, reverse=True)
            if len(idle) < 20:
                idle = sorted(idle, key=combined_occupancy)

            for level, cost_multiplier in enumerate(self.cost_multipliers):
                if not idle:
                    break
                for sat in list(saturated):
                    stealable = self.stealable[sat.address][level]
                    if not stealable or not idle:
                        continue

                    for ts in list(stealable):
                        if ts not in self.key_stealable or ts.processing_on is not sat:
                            stealable.discard(ts)
                            continue
                        i += 1
                        if not idle:
                            break

                        thieves = _potential_thieves_for(ts, idle)
                        if not thieves:
                            break

                        thief = self._maybe_pick_thief(ts, thieves)
                        if not thief:
                            thief = thieves[i % len(thieves)]

                        duration = sat.processing.get(ts)
                        if duration is None:
                            stealable.discard(ts)
                            continue

                        maybe_move_task(
                            level, ts, sat, thief, duration, cost_multiplier
                        )

                if self.cost_multipliers[level] < 20:  # don't steal from public at cost
                    stealable = self.stealable_all[level]
                    for ts in list(stealable):
                        if not idle:
                            break
                        if ts not in self.key_stealable:
                            stealable.discard(ts)
                            continue

                        sat = ts.processing_on
                        if sat is None:
                            stealable.discard(ts)
                            continue
                        if combined_occupancy(sat) < 0.2:
                            continue
                        if len(sat.processing) <= sat.nthreads:
                            continue

                        i += 1
                        thieves = _potential_thieves_for(ts, idle)
                        if not thieves:
                            continue
                        thief = self._maybe_pick_thief(ts, thieves)
                        if not thief:
                            thief = thieves[i % len(thieves)]

                        duration = sat.processing[ts]

                        maybe_move_task(
                            level, ts, sat, thief, duration, cost_multiplier
                        )

            if log:
                self.log(log)
                self.count += 1
            stop = time()
            if s.digests:
                s.digests["steal-duration"].add(stop - start)

    def _maybe_pick_thief(self, ts, thieves):
        """Try to be smart about picking a thief given some options. We're
        trying to pick a thief which has dependencies for a given task, if
        possible and will pick the one which works best for us given the
        Scheduler.worker_objective

        If no idle worker with dependencies is found, this returns None.
        """
        if ts._dependencies:
            who_has = set()
            for dep in ts._dependencies:
                who_has.update(dep.who_has)

            thieves_with_data = list(who_has & set(thieves))

            # If there are potential thieves with dependencies we
            # should prefer them and pick the one which works best.
            # Otherwise just random/round robin
            if thieves_with_data:
                if len(thieves_with_data) > 10:
                    thieves_with_data = random.choices(thieves_with_data, k=10)
                return min(
                    thieves_with_data,
                    key=partial(self.scheduler.worker_objective, ts),
                )

    def restart(self, scheduler):
        for stealable in self.stealable.values():
            for s in stealable:
                s.clear()

        for s in self.stealable_all:
            s.clear()
        self.key_stealable.clear()

    def story(self, *keys):
        keys = {key.key if not isinstance(key, str) else key for key in keys}
        out = []
        for _, L in self.scheduler.get_events(topic="stealing"):
            if not isinstance(L, list):
                L = [L]
            for t in L:
                if any(x in keys for x in t):
                    out.append(t)
        return out


def _potential_thieves_for(ts, idle):
    """Return the list of workers from ``idle`` that could steal ``ts``."""
    if _has_restrictions(ts):
        return [ws for ws in idle if _can_steal(ws, ts)]
    else:
        return idle


def _can_steal(thief, ts):
    """Determine whether worker ``thief`` can steal task ``ts``.

    Assumes that `ts` has some restrictions.
    """
    if (
        ts.host_restrictions
        and get_address_host(thief.address) not in ts.host_restrictions
    ):
        return False
    elif ts.worker_restrictions and thief.address not in ts.worker_restrictions:
        return False

    if not ts.resource_restrictions:
        return True

    for resource, value in ts.resource_restrictions.items():
        try:
            supplied = thief.resources[resource]
        except KeyError:
            return False
        else:
            if supplied < value:
                return False
    return True


def _has_restrictions(ts):
    """Determine whether the given task has restrictions and whether these
    restrictions are strict.
    """
    return not ts.loose_restrictions and (
        ts.host_restrictions or ts.worker_restrictions or ts.resource_restrictions
    )


fast_tasks = {"split-shuffle"}

from __future__ import annotations

import abc
import asyncio
import contextlib
import logging
import os
import pickle
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

import pyarrow as pa
import toolz

import dask
from dask.context import thread_state
from dask.utils import parse_bytes

from distributed.core import PooledRPCCall
from distributed.exceptions import Reschedule
from distributed.protocol.serialize import ToPickle
from distributed.shuffle._arrow import convert_partition
from distributed.shuffle._comms import CommShardsBuffer
from distributed.shuffle._disk import DiskShardsBuffer
from distributed.shuffle._limiter import ResourceLimiter
from distributed.shuffle._rechunk import ChunkedAxes, NDIndex
from distributed.shuffle._rechunk import ShardID as ArrayRechunkShardID
from distributed.shuffle._rechunk import split_axes
from distributed.shuffle._shuffle import ShuffleId, ShuffleType
from distributed.sizeof import sizeof
from distributed.utils import log_errors, sync

if TYPE_CHECKING:
    # TODO import from typing (requires Python >=3.10)
    import numpy as np
    import pandas as pd
    import pyarrow as pa

    from distributed.worker import Worker

T_transfer_shard_id = TypeVar("T_transfer_shard_id")
T_partition_id = TypeVar("T_partition_id")
T_partition_type = TypeVar("T_partition_type")
T = TypeVar("T")

logger = logging.getLogger(__name__)


class ShuffleClosedError(RuntimeError):
    pass


class ShuffleRun(Generic[T_transfer_shard_id, T_partition_id, T_partition_type]):
    def __init__(
        self,
        id: ShuffleId,
        run_id: int,
        output_workers: set[str],
        local_address: str,
        directory: str,
        executor: ThreadPoolExecutor,
        rpc: Callable[[str], PooledRPCCall],
        scheduler: PooledRPCCall,
        memory_limiter_disk: ResourceLimiter,
        memory_limiter_comms: ResourceLimiter,
    ):
        self.id = id
        self.run_id = run_id
        self.output_workers = output_workers
        self.local_address = local_address
        self.executor = executor
        self.rpc = rpc
        self.scheduler = scheduler
        self.closed = False

        self._disk_buffer = DiskShardsBuffer(
            write=self.write_to_disk,
            directory=directory,
            memory_limiter=memory_limiter_disk,
        )

        self._comm_buffer = CommShardsBuffer(
            send=self.send, memory_limiter=memory_limiter_comms
        )
        # TODO: reduce number of connections to number of workers
        # MultiComm.max_connections = min(10, n_workers)

        self.diagnostics: dict[str, float] = defaultdict(float)
        self.transferred = False
        self.received: set[T_transfer_shard_id] = set()
        self.total_recvd = 0
        self.start_time = time.time()
        self._exception: Exception | None = None
        self._closed_event = asyncio.Event()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: id={self.id!r}, run_id={self.run_id!r}, local_address={self.local_address!r}, closed={self.closed!r}, transferred={self.transferred!r}>"

    def __str__(self) -> str:
        return f"{self.__class__.__name__}<{self.id}[{self.run_id}]> on {self.local_address}"

    def __hash__(self) -> int:
        return self.run_id

    @contextlib.contextmanager
    def time(self, name: str) -> Iterator[None]:
        start = time.time()
        yield
        stop = time.time()
        self.diagnostics[name] += stop - start

    async def barrier(self) -> None:
        self.raise_if_closed()
        # TODO: Consider broadcast pinging once when the shuffle starts to warm
        # up the comm pool on scheduler side
        await self.scheduler.shuffle_barrier(id=self.id, run_id=self.run_id)

    # FIXME: The level of indirection is way too high
    async def write_to_disk(self, path, shards):
        def _():
            with log_errors():
                tab = pa.concat_tables(shards)
                with self.time("write"):
                    # TODO: compression is trivial with this assuming the reader
                    # also uses pyarrow to open the file
                    with pa.output_stream(path) as fd:
                        with pa.ipc.new_stream(fd, schema=tab.schema) as stream:
                            stream.write_table(tab)

        # TODO: CPU instrumentation is off with this
        return await self.offload(_)

    async def send(
        self, address: str, shards: list[tuple[T_transfer_shard_id, bytes]]
    ) -> None:
        class mylist(list):
            ...
        self.raise_if_closed()
        return await self.rpc(address).shuffle_receive(
            data=ToPickle(mylist(shards)),
            shuffle_id=self.id,
            run_id=self.run_id,
        )

    async def offload(self, func: Callable[..., T], *args: Any) -> T:
        self.raise_if_closed()
        with self.time("cpu"):
            return await asyncio.get_running_loop().run_in_executor(
                self.executor,
                func,
                *args,
            )

    def heartbeat(self) -> dict[str, Any]:
        comm_heartbeat = self._comm_buffer.heartbeat()
        comm_heartbeat["read"] = self.total_recvd
        return {
            "disk": self._disk_buffer.heartbeat(),
            "comm": comm_heartbeat,
            "diagnostics": self.diagnostics,
            "start": self.start_time,
        }

    async def _write_to_comm(
        self, data: dict[str, list[tuple[T_transfer_shard_id, bytes]]]
    ) -> None:
        self.raise_if_closed()
        await self._comm_buffer.write(data)

    async def _write_to_disk(self, data: dict[NDIndex, list[bytes]]) -> None:
        self.raise_if_closed()
        await self._disk_buffer.write(
            {"_".join(str(i) for i in k): v for k, v in data.items()}
        )

    def raise_if_closed(self) -> None:
        if self.closed:
            if self._exception:
                raise self._exception
            raise ShuffleClosedError(f"{self} has already been closed")

    async def inputs_done(self) -> None:
        self.raise_if_closed()
        self.transferred = True
        await self._flush_comm()
        try:
            self._comm_buffer.raise_on_exception()
        except Exception as e:
            self._exception = e
            raise

    async def _flush_comm(self) -> None:
        self.raise_if_closed()
        await self._comm_buffer.flush()

    async def flush_receive(self) -> None:
        self.raise_if_closed()
        await self._disk_buffer.flush()

    async def close(self) -> None:
        if self.closed:  # pragma: no cover
            await self._closed_event.wait()
            return

        self.closed = True
        await self._comm_buffer.close()
        await self._disk_buffer.close()
        self._closed_event.set()

    def fail(self, exception: Exception) -> None:
        if not self.closed:
            self._exception = exception

    def _read_from_disk(self, id: NDIndex) -> bytes:
        self.raise_if_closed()
        data: bytes = self._disk_buffer.read("_".join(str(i) for i in id))
        return data

    async def receive(self, data: list[tuple[T_transfer_shard_id, bytes]]) -> None:
        await self._receive(data)

    async def _ensure_output_worker(self, i: T_partition_id, key: str) -> None:
        assigned_worker = self._get_assigned_worker(i)

        if assigned_worker != self.local_address:
            result = await self.scheduler.shuffle_restrict_task(
                id=self.id, run_id=self.run_id, key=key, worker=assigned_worker
            )
            if result["status"] == "error":
                raise RuntimeError(result["message"])
            assert result["status"] == "OK"
            raise Reschedule()

    @abc.abstractmethod
    def _get_assigned_worker(self, i: T_partition_id) -> str:
        """Get the address of the worker assigned to the output partition"""

    @abc.abstractmethod
    async def _receive(self, data: list[tuple[T_transfer_shard_id, bytes]]) -> None:
        """Receive shards belonging to output partitions of this shuffle run"""

    @abc.abstractmethod
    async def add_partition(
        self, data: T_partition_type, partition_id: T_partition_id
    ) -> int:
        """Add an input partition to the shuffle run"""

    @abc.abstractmethod
    async def get_output_partition(
        self, partition_id: T_partition_id, key: str, meta: pd.DataFrame | None = None
    ) -> T_partition_type:
        """Get an output partition to the shuffle run"""


class ArrayRechunkRun(ShuffleRun[ArrayRechunkShardID, NDIndex, "np.ndarray"]):
    """State for a single active rechunk execution

    This object is responsible for splitting, sending, receiving and combining
    data shards.

    It is entirely agnostic to the distributed system and can perform a shuffle
    with other `Shuffle` instances using `rpc` and `broadcast`.

    The user of this needs to guarantee that only `Shuffle`s of the same unique
    `ShuffleID` interact.

    Parameters
    ----------
    worker_for:
        A mapping partition_id -> worker_address.
    output_workers:
        A set of all participating worker (addresses).
    old:
        Existing chunking of the array per dimension.
    new:
        Desired chunking of the array per dimension.
    id:
        A unique `ShuffleID` this belongs to.
    run_id:
        A unique identifier of the specific execution of the shuffle this belongs to.
    local_address:
        The local address this Shuffle can be contacted by using `rpc`.
    directory:
        The scratch directory to buffer data in.
    executor:
        Thread pool to use for offloading compute.
    loop:
        The event loop.
    rpc:
        A callable returning a PooledRPCCall to contact other Shuffle instances.
        Typically a ConnectionPool.
    scheduler:
        A PooledRPCCall to to contact the scheduler.
    memory_limiter_disk:
    memory_limiter_comm:
        A ``ResourceLimiter`` limiting the total amount of memory used in either
        buffer.
    """

    def __init__(
        self,
        worker_for: dict[NDIndex, str],
        output_workers: set,
        old: ChunkedAxes,
        new: ChunkedAxes,
        id: ShuffleId,
        run_id: int,
        local_address: str,
        directory: str,
        executor: ThreadPoolExecutor,
        rpc: Callable[[str], PooledRPCCall],
        scheduler: PooledRPCCall,
        memory_limiter_disk: ResourceLimiter,
        memory_limiter_comms: ResourceLimiter,
    ):
        super().__init__(
            id=id,
            run_id=run_id,
            output_workers=output_workers,
            local_address=local_address,
            directory=directory,
            executor=executor,
            rpc=rpc,
            scheduler=scheduler,
            memory_limiter_comms=memory_limiter_comms,
            memory_limiter_disk=memory_limiter_disk,
        )
        self.old = old
        self.new = new
        partitions_of = defaultdict(list)
        for part, addr in worker_for.items():
            partitions_of[addr].append(part)
        self.partitions_of = dict(partitions_of)
        self.worker_for = worker_for
        self.split_axes = split_axes(old, new)

    async def _receive(self, data: list[tuple[ArrayRechunkShardID, bytes]]) -> None:
        self.raise_if_closed()

        buffers = defaultdict(list)
        for d in data:
            id, payload = d
            if id in self.received:
                continue
            self.received.add(id)
            self.total_recvd += sizeof(d)

            buffers[id.chunk_index].append(payload)

        del data
        if not buffers:
            return
        try:
            await self._write_to_disk(buffers)
        except Exception as e:
            self._exception = e
            raise

    async def add_partition(self, data: np.ndarray, partition_id: NDIndex) -> int:
        self.raise_if_closed()
        if self.transferred:
            raise RuntimeError(f"Cannot add more partitions to {self}")

        def _() -> dict[str, list[tuple[ArrayRechunkShardID, bytes]]]:
            """Return a mapping of worker addresses to a list of tuples of shard IDs
            and shard data.

            As shard data, we serialize the payload together with the sub-index of the
            slice within the new chunk. To assemble the new chunk from its shards, it
            needs the sub-index to know where each shard belongs within the chunk.
            Adding the sub-index into the serialized payload on the sender allows us to
            write the serialized payload directly to disk on the receiver.
            """
            out: dict[str, list[tuple[ArrayRechunkShardID, bytes]]] = defaultdict(list)
            from itertools import product

            ndsplits = product(
                *(axis[i] for axis, i in zip(self.split_axes, partition_id))
            )

            for ndsplit in ndsplits:
                chunk_index, shard_index, ndslice = zip(*ndsplit)
                id = ArrayRechunkShardID(chunk_index, shard_index)
                out[self.worker_for[chunk_index]].append(
                    (id, pickle.dumps((shard_index, data[ndslice])))
                )
            return out

        out = await self.offload(_)
        await self._write_to_comm(out)
        return self.run_id

    async def get_output_partition(
        self, partition_id: NDIndex, key: str, meta: pd.DataFrame | None = None
    ) -> np.ndarray:
        self.raise_if_closed()
        assert meta is None
        assert self.transferred, "`get_output_partition` called before barrier task"

        await self._ensure_output_worker(partition_id, key)

        await self.flush_receive()

        data = self._read_from_disk(partition_id)

        def _() -> np.ndarray:
            return convert_chunk(data)

        return await self.offload(_)

    def _get_assigned_worker(self, id: NDIndex) -> str:
        return self.worker_for[id]


class DataFrameShuffleRun(ShuffleRun[int, int, "pd.DataFrame"]):
    """State for a single active shuffle execution

    This object is responsible for splitting, sending, receiving and combining
    data shards.

    It is entirely agnostic to the distributed system and can perform a shuffle
    with other `Shuffle` instances using `rpc` and `broadcast`.

    The user of this needs to guarantee that only `Shuffle`s of the same unique
    `ShuffleID` interact.

    Parameters
    ----------
    worker_for:
        A mapping partition_id -> worker_address.
    output_workers:
        A set of all participating worker (addresses).
    column:
        The data column we split the input partition by.
    id:
        A unique `ShuffleID` this belongs to.
    run_id:
        A unique identifier of the specific execution of the shuffle this belongs to.
    local_address:
        The local address this Shuffle can be contacted by using `rpc`.
    directory:
        The scratch directory to buffer data in.
    executor:
        Thread pool to use for offloading compute.
    loop:
        The event loop.
    rpc:
        A callable returning a PooledRPCCall to contact other Shuffle instances.
        Typically a ConnectionPool.
    scheduler:
        A PooledRPCCall to to contact the scheduler.
    memory_limiter_disk:
    memory_limiter_comm:
        A ``ResourceLimiter`` limiting the total amount of memory used in either
        buffer.
    """

    def __init__(
        self,
        worker_for: dict[int, str],
        output_workers: set,
        column: str,
        id: ShuffleId,
        run_id: int,
        local_address: str,
        directory: str,
        executor: ThreadPoolExecutor,
        rpc: Callable[[str], PooledRPCCall],
        scheduler: PooledRPCCall,
        memory_limiter_disk: ResourceLimiter,
        memory_limiter_comms: ResourceLimiter,
    ):
        import pandas as pd

        super().__init__(
            id=id,
            run_id=run_id,
            output_workers=output_workers,
            local_address=local_address,
            directory=directory,
            executor=executor,
            rpc=rpc,
            scheduler=scheduler,
            memory_limiter_comms=memory_limiter_comms,
            memory_limiter_disk=memory_limiter_disk,
        )
        self.column = column
        partitions_of = defaultdict(list)
        for part, addr in worker_for.items():
            partitions_of[addr].append(part)
        self.partitions_of = dict(partitions_of)
        self.worker_for = pd.Series(worker_for, name="_workers").astype("category")

    async def receive(self, data: list[tuple[int, bytes]]) -> None:
        await self._receive(data)

    async def _receive(self, data: list[tuple[int, bytes]]) -> None:
        self.raise_if_closed()

        filtered = []
        for d in data:
            if d[0] not in self.received:
                filtered.append(d[1])
                self.received.add(d[0])
                self.total_recvd += sizeof(d)
        del data
        if not filtered:
            return
        try:
            groups = await self.offload(self._repartition_buffers, filtered)
            del filtered
            await self._write_to_disk(groups)
        except Exception as e:
            self._exception = e
            raise

    def _repartition_buffers(self, data: list[bytes]) -> dict[NDIndex, list[bytes]]:
        table = pa.concat_tables(data)
        groups = split_by_partition(table, self.column)
        assert len(table) == sum(map(len, groups.values()))
        del data
        res = {(k,): [v] for k, v in groups.items()}
        class mydict(dict):
            pass
        return mydict(res)

    async def add_partition(self, data: pd.DataFrame, partition_id: int) -> int:
        self.raise_if_closed()
        if self.transferred:
            raise RuntimeError(f"Cannot add more partitions to {self}")

        def _() -> dict[str, list[tuple[int, bytes]]]:
            out = split_by_worker(
                data,
                self.column,
                self.worker_for,
            )
            out = {k: [(partition_id, t)] for k, t in out.items()}
            return out

        out = await self.offload(_)
        await self._write_to_comm(out)
        return self.run_id

    async def get_output_partition(
        self, partition_id: int, key: str, meta: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        self.raise_if_closed()
        assert meta is not None
        assert self.transferred, "`get_output_partition` called before barrier task"

        await self._ensure_output_worker(partition_id, key)

        await self.flush_receive()
        try:
            data = self._read_from_disk((partition_id,))

            def _() -> pd.DataFrame:
                return convert_partition(data, meta)  # type: ignore

            out = await self.offload(_)
        except KeyError:
            out = meta.copy()
        return out

    def _get_assigned_worker(self, id: int) -> str:
        return self.worker_for[id]


class ShuffleWorkerExtension:
    """Interface between a Worker and a Shuffle.

    This extension is responsible for

    - Lifecycle of Shuffle instances
    - ensuring connectivity between remote shuffle instances
    - ensuring connectivity and integration with the scheduler
    - routing concurrent calls to the appropriate `Shuffle` based on its `ShuffleID`
    - collecting instrumentation of ongoing shuffles and route to scheduler/worker
    """

    worker: Worker
    shuffles: dict[ShuffleId, ShuffleRun]
    _runs: set[ShuffleRun]
    memory_limiter_comms: ResourceLimiter
    memory_limiter_disk: ResourceLimiter
    closed: bool

    def __init__(self, worker: Worker) -> None:
        # Attach to worker
        worker.handlers["shuffle_receive"] = self.shuffle_receive
        worker.handlers["shuffle_inputs_done"] = self.shuffle_inputs_done
        worker.stream_handlers["shuffle-fail"] = self.shuffle_fail
        worker.extensions["shuffle"] = self

        # Initialize
        self.worker = worker
        self.shuffles = {}
        self._runs = set()
        self.memory_limiter_comms = ResourceLimiter(parse_bytes("100 MiB"))
        self.memory_limiter_disk = ResourceLimiter(parse_bytes("1 GiB"))
        self.closed = False
        self._executor = ThreadPoolExecutor(self.worker.state.nthreads * 2)

    def __str__(self) -> str:
        return f"ShuffleWorkerExtension on {self.worker.address}"

    def __repr__(self) -> str:
        return f"<ShuffleWorkerExtension, worker={self.worker.address_safe!r}, closed={self.closed}>"

    # Handlers
    ##########
    # NOTE: handlers are not threadsafe, but they're called from async comms, so that's okay

    def heartbeat(self) -> dict:
        return {id: shuffle.heartbeat() for id, shuffle in self.shuffles.items()}

    async def shuffle_receive(
        self,
        shuffle_id: ShuffleId,
        run_id: int,
        data: list[tuple[int, bytes]],
    ) -> None:
        """
        Handler: Receive an incoming shard of data from a peer worker.
        Using an unknown ``shuffle_id`` is an error.
        """
        shuffle = await self._get_shuffle_run(shuffle_id, run_id)
        await shuffle.receive(data)

    async def shuffle_inputs_done(self, shuffle_id: ShuffleId, run_id: int) -> None:
        """
        Handler: Inform the extension that all input partitions have been handed off to extensions.
        Using an unknown ``shuffle_id`` is an error.
        """
        with log_errors():
            shuffle = await self._get_shuffle_run(shuffle_id, run_id)
            await shuffle.inputs_done()

    def shuffle_fail(self, shuffle_id: ShuffleId, run_id: int, message: str) -> None:
        """Fails the shuffle run with the message as exception and triggers cleanup.

        .. warning::
            To guarantee the correct order of operations, shuffle_fail must be
            synchronous. See
            https://github.com/dask/distributed/pull/7486#discussion_r1088857185
            for more details.
        """
        shuffle = self.shuffles.get(shuffle_id, None)
        if shuffle is None or shuffle.run_id != run_id:
            return
        self.shuffles.pop(shuffle_id)
        exception = RuntimeError(message)
        shuffle.fail(exception)

        async def _(extension: ShuffleWorkerExtension, shuffle: ShuffleRun) -> None:
            await shuffle.close()
            extension._runs.remove(shuffle)

        self.worker._ongoing_background_tasks.call_soon(_, self, shuffle)

    def add_partition(
        self,
        data: Any,
        partition_id: int | tuple[int, ...],
        shuffle_id: ShuffleId,
        type: ShuffleType,
        **kwargs: Any,
    ) -> int:
        shuffle = self.get_or_create_shuffle(shuffle_id, type=type, **kwargs)
        return sync(
            self.worker.loop,
            shuffle.add_partition,
            data=data,
            partition_id=partition_id,
        )

    async def _barrier(self, shuffle_id: ShuffleId, run_ids: list[int]) -> int:
        """
        Task: Note that the barrier task has been reached (`add_partition` called for all input partitions)

        Using an unknown ``shuffle_id`` is an error. Calling this before all partitions have been
        added is undefined.
        """
        run_id = run_ids[0]
        # Assert that all input data has been shuffled using the same run_id
        assert all(run_id == id for id in run_ids)
        # Tell all peers that we've reached the barrier
        # Note that this will call `shuffle_inputs_done` on our own worker as well
        shuffle = await self._get_shuffle_run(shuffle_id, run_id)
        await shuffle.barrier()
        return run_id

    async def _get_shuffle_run(
        self,
        shuffle_id: ShuffleId,
        run_id: int,
    ) -> ShuffleRun:
        """Get or create the shuffle matching the ID and run ID.

        Parameters
        ----------
        shuffle_id
            Unique identifier of the shuffle
        run_id
            Unique identifier of the shuffle run

        Raises
        ------
        KeyError
            If the shuffle does not exist
        RuntimeError
            If the run_id is stale
        """
        shuffle = self.shuffles.get(shuffle_id, None)
        if shuffle is None or shuffle.run_id < run_id:
            shuffle = await self._refresh_shuffle(
                shuffle_id=shuffle_id,
            )

        if shuffle.run_id > run_id:
            raise RuntimeError(f"{run_id=} stale, got {shuffle}")
        elif shuffle.run_id < run_id:
            raise RuntimeError(f"{run_id=} invalid, got {shuffle}")

        if shuffle._exception:
            raise shuffle._exception
        return shuffle

    async def _get_or_create_shuffle(
        self,
        shuffle_id: ShuffleId,
        type: ShuffleType,
        **kwargs: Any,
    ) -> ShuffleRun:
        """Get or create a shuffle matching the ID and data spec.

        Parameters
        ----------
        shuffle_id
            Unique identifier of the shuffle
        type:
            Type of the shuffle operation
        """
        shuffle = self.shuffles.get(shuffle_id, None)
        if shuffle is None:
            shuffle = await self._refresh_shuffle(
                shuffle_id=shuffle_id,
                type=type,
                kwargs=kwargs,
            )

        if self.closed:
            raise ShuffleClosedError(f"{self} has already been closed")
        if shuffle._exception:
            raise shuffle._exception
        return shuffle

    @overload
    async def _refresh_shuffle(
        self,
        shuffle_id: ShuffleId,
    ) -> ShuffleRun:
        ...

    @overload
    async def _refresh_shuffle(
        self,
        shuffle_id: ShuffleId,
        type: ShuffleType,
        kwargs: dict,
    ) -> ShuffleRun:
        ...

    async def _refresh_shuffle(
        self,
        shuffle_id: ShuffleId,
        type: ShuffleType | None = None,
        kwargs: dict | None = None,
    ) -> ShuffleRun:
        if type is None:
            result = await self.worker.scheduler.shuffle_get(
                id=shuffle_id,
                worker=self.worker.address,
            )
        elif type == ShuffleType.DATAFRAME:
            assert kwargs is not None
            result = await self.worker.scheduler.shuffle_get_or_create(
                id=shuffle_id,
                type=type,
                spec={
                    "npartitions": kwargs["npartitions"],
                    "column": kwargs["column"],
                    "parts_out": kwargs["parts_out"],
                },
                worker=self.worker.address,
            )
        elif type == ShuffleType.ARRAY_RECHUNK:
            assert kwargs is not None
            result = await self.worker.scheduler.shuffle_get_or_create(
                id=shuffle_id,
                type=type,
                spec=kwargs,
                worker=self.worker.address,
            )
        else:  # pragma: no cover
            raise TypeError(type)
        if result["status"] == "error":
            raise RuntimeError(result["message"])
        assert result["status"] == "OK"

        if self.closed:
            raise ShuffleClosedError(f"{self} has already been closed")
        if shuffle_id in self.shuffles:
            existing = self.shuffles[shuffle_id]
            if existing.run_id >= result["run_id"]:
                return existing
            else:
                self.shuffles.pop(shuffle_id)
                existing.fail(RuntimeError("Stale Shuffle"))

                async def _(
                    extension: ShuffleWorkerExtension, shuffle: ShuffleRun
                ) -> None:
                    await shuffle.close()
                    extension._runs.remove(shuffle)

                self.worker._ongoing_background_tasks.call_soon(_, self, existing)
        shuffle: ShuffleRun
        if result["type"] == ShuffleType.DATAFRAME:
            shuffle = DataFrameShuffleRun(
                column=result["column"],
                worker_for=result["worker_for"],
                output_workers=result["output_workers"],
                id=shuffle_id,
                run_id=result["run_id"],
                directory=os.path.join(
                    self.worker.local_directory,
                    f"shuffle-{shuffle_id}-{result['run_id']}",
                ),
                executor=self._executor,
                local_address=self.worker.address,
                rpc=self.worker.rpc,
                scheduler=self.worker.scheduler,
                memory_limiter_disk=self.memory_limiter_disk,
                memory_limiter_comms=self.memory_limiter_comms,
            )
        elif result["type"] == ShuffleType.ARRAY_RECHUNK:
            shuffle = ArrayRechunkRun(
                worker_for=result["worker_for"],
                output_workers=result["output_workers"],
                old=result["old"],
                new=result["new"],
                id=shuffle_id,
                run_id=result["run_id"],
                directory=os.path.join(
                    self.worker.local_directory,
                    f"shuffle-{shuffle_id}-{result['run_id']}",
                ),
                executor=self._executor,
                local_address=self.worker.address,
                rpc=self.worker.rpc,
                scheduler=self.worker.scheduler,
                memory_limiter_disk=self.memory_limiter_disk,
                memory_limiter_comms=self.memory_limiter_comms,
            )
        else:  # pragma: no cover
            raise TypeError(result["type"])
        self.shuffles[shuffle_id] = shuffle
        self._runs.add(shuffle)
        return shuffle

    async def close(self) -> None:
        assert not self.closed

        self.closed = True
        while self.shuffles:
            _, shuffle = self.shuffles.popitem()
            await shuffle.close()
            self._runs.remove(shuffle)
        try:
            self._executor.shutdown(cancel_futures=True)
        except Exception:  # pragma: no cover
            self._executor.shutdown()

    #############################
    # Methods for worker thread #
    #############################

    def barrier(self, shuffle_id: ShuffleId, run_ids: list[int]) -> int:
        result = sync(self.worker.loop, self._barrier, shuffle_id, run_ids)
        return result

    def get_shuffle_run(
        self,
        shuffle_id: ShuffleId,
        run_id: int,
    ) -> ShuffleRun:
        return sync(
            self.worker.loop,
            self._get_shuffle_run,
            shuffle_id,
            run_id,
        )

    def get_or_create_shuffle(
        self,
        shuffle_id: ShuffleId,
        type: ShuffleType,
        **kwargs: Any,
    ) -> ShuffleRun:
        return sync(
            self.worker.loop,
            self._get_or_create_shuffle,
            shuffle_id,
            type,
            **kwargs,
        )

    def get_output_partition(
        self,
        shuffle_id: ShuffleId,
        run_id: int,
        partition_id: int | NDIndex,
        meta: pd.DataFrame | None = None,
    ) -> Any:
        """
        Task: Retrieve a shuffled output partition from the ShuffleExtension.

        Calling this for a ``shuffle_id`` which is unknown or incomplete is an error.
        """
        shuffle = self.get_shuffle_run(shuffle_id, run_id)
        key = thread_state.key
        return sync(
            self.worker.loop,
            shuffle.get_output_partition,
            partition_id=partition_id,
            key=key,
            meta=meta,
        )


def split_by_worker(
    df: pd.DataFrame,
    column: str,
    worker_for: pd.Series,
) -> dict[Any, pa.Table]:
    """
    Split data into many arrow batches, partitioned by destination worker
    """
    import numpy as np
    import pyarrow as pa

    df = df.merge(
        right=worker_for.cat.codes.rename("_worker"),
        left_on=column,
        right_index=True,
        how="inner",
    )
    nrows = len(df)
    if not nrows:
        return {}
    # assert len(df) == nrows  # Not true if some outputs aren't wanted
    # FIXME: If we do not preserve the index something is corrupting the
    # bytestream such that it cannot be deserialized anymore
    t = pa.Table.from_pandas(df, preserve_index=True)
    t = t.sort_by("_worker")
    codes = np.asarray(t.select(["_worker"]))[0]
    t = t.drop(["_worker"])
    del df

    splits = np.where(codes[1:] != codes[:-1])[0] + 1
    splits = np.concatenate([[0], splits])

    shards = [
        t.slice(offset=a, length=b - a) for a, b in toolz.sliding_window(2, splits)
    ]
    shards.append(t.slice(offset=splits[-1], length=None))

    unique_codes = codes[splits]
    out = {
        # FIXME https://github.com/pandas-dev/pandas-stubs/issues/43
        worker_for.cat.categories[code]: shard
        for code, shard in zip(unique_codes, shards)
    }
    assert sum(map(len, out.values())) == nrows
    return out


def split_by_partition(t: pa.Table, column: str) -> dict[Any, pa.Table]:
    """
    Split data into many arrow batches, partitioned by final partition
    """
    import numpy as np

    partitions = t.select([column]).to_pandas()[column].unique()
    partitions.sort()
    t = t.sort_by(column)

    partition = np.asarray(t.select([column]))[0]
    splits = np.where(partition[1:] != partition[:-1])[0] + 1
    splits = np.concatenate([[0], splits])

    shards = [
        t.slice(offset=a, length=b - a) for a, b in toolz.sliding_window(2, splits)
    ]
    shards.append(t.slice(offset=splits[-1], length=None))
    assert len(t) == sum(map(len, shards))
    assert len(partitions) == len(shards)
    return dict(zip(partitions, shards))


def convert_chunk(data: bytes) -> np.ndarray:
    import numpy as np

    from dask.array.core import concatenate3

    file = BytesIO(data)
    shards: dict[NDIndex, np.ndarray] = {}

    while file.tell() < len(data):
        index, shard = pickle.load(file)
        shards[index] = shard

    subshape = [max(dim) + 1 for dim in zip(*shards.keys())]
    assert len(shards) == np.prod(subshape)

    rec_cat_arg = np.empty(subshape, dtype="O")
    for index, shard in shards.items():
        rec_cat_arg[tuple(index)] = shard
    del data
    del file
    arrs = rec_cat_arg.tolist()
    return concatenate3(arrs)

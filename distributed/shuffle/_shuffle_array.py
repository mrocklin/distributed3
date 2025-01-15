from __future__ import annotations

import functools
import mmap
import os
from collections import defaultdict
from collections.abc import Generator, Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tornado.ioloop import IOLoop

from dask.sizeof import sizeof
from dask.typing import Key

from distributed.core import PooledRPCCall
from distributed.metrics import context_meter
from distributed.shuffle import ShuffleWorkerPlugin
from distributed.shuffle._core import (
    NDIndex,
    ShuffleId,
    ShuffleRun,
    ShuffleSpec,
    barrier_key,
    get_worker_plugin,
    handle_transfer_errors,
)
from distributed.shuffle._limiter import ResourceLimiter
from distributed.shuffle._pickle import unpickle_bytestream
from distributed.shuffle._rechunk import rechunk_unpack
from distributed.shuffle._shuffle import shuffle_barrier


def shuffle_name(token: str) -> str:
    return f"shuffle-p2p-{token}"


def _p2p_shuffle(  # type: ignore[no-untyped-def]
    chunks, new_chunks, axis, in_name: str, out_name: str, disk: bool = True
) -> dict[Key, Any]:
    from dask.array._shuffle import convert_key

    arrays = []
    for i, new_chunk in enumerate(new_chunks):
        arrays.append(np.array([new_chunk, [i] * len(new_chunk)]))

    result = np.concatenate(arrays, axis=1)
    sorter = np.argsort(result[0, :])
    sorted_indexer = result[:, sorter]
    chunk_boundaries = np.cumsum((0,) + chunks[axis])

    dsk: dict[Key, Any] = {}

    # Use `token` to generate a canonical group for the entire rechunk
    token = out_name.split("-")[-1]
    transfer_group = f"shuffle-transfer-{token}"
    unpack_group = out_name
    _barrier_key = barrier_key(ShuffleId(token))

    # Get existing chunk tuple locations
    chunk_tuples = list(
        product(*(range(len(c)) for i, c in enumerate(chunks) if i != axis))
    )
    chunk_lengths = [len(c) for c in chunks]
    chunk_lengths[axis] = len(np.unique(result[1, :]))

    transfer_keys = []

    for i, (start, stop) in enumerate(zip(chunk_boundaries[:-1], chunk_boundaries[1:])):
        start = np.searchsorted(sorted_indexer[0, :], start)
        stop = np.searchsorted(sorted_indexer[0, :], stop)

        chunk_indexer = sorted_indexer[:, start:stop].copy()
        if chunk_indexer.shape[1] == 0:
            # skip output chunks that don't get any data
            continue

        chunk_indexer[0, :] -= chunk_boundaries[i]

        for chunk_tuple in chunk_tuples:
            key = (transfer_group,) + convert_key(chunk_tuple, i, axis)
            transfer_keys.append(key)
            dsk[key] = (
                shuffle_transfer,
                (in_name,) + convert_key(chunk_tuple, i, axis),
                token,
                chunk_indexer,
                chunk_tuple,
                chunk_lengths,
                axis,
                convert_key(chunk_tuple, i, axis),
                disk,
            )

    dsk[_barrier_key] = (shuffle_barrier, token, transfer_keys)

    for axis_chunk in np.unique(result[1, :]):
        sorter = np.argsort(result[0, result[1, :] == axis_chunk])

        for chunk_tuple in chunk_tuples:
            chunk_key = convert_key(chunk_tuple, int(axis_chunk), axis)

            dsk[(unpack_group,) + chunk_key] = (
                _shuffle_unpack,
                token,
                chunk_key,
                _barrier_key,
                sorter,
                axis,
            )
    return dsk


def _shuffle_unpack(
    id: ShuffleId,
    output_chunk: NDIndex,
    barrier_run_id: int,
    sorter: np.ndarray,
    axis: int,
) -> np.ndarray:
    result = rechunk_unpack(id, output_chunk, barrier_run_id)
    slicer = [
        slice(None),
    ] * len(result.shape)
    slicer[axis] = np.argsort(sorter)  # type: ignore[call-overload]
    return result[*slicer]


def shuffle_transfer(
    input: np.ndarray,
    id: ShuffleId,
    chunk_indexer: tuple[np.ndarray, np.ndarray],
    chunk_tuple: tuple[int, ...],
    chunk_lengths: tuple[int, ...],
    axis: int,
    input_chunk: Any,
    disk: bool,
) -> int:
    with handle_transfer_errors(id):
        return get_worker_plugin().add_partition(
            input,
            partition_id=input_chunk,
            spec=ArrayShuffleSpec(
                id=id, chunk_lengths=chunk_lengths, axis=axis, disk=disk
            ),
            chunk_indexer=chunk_indexer,
            chunk_tuple=chunk_tuple,
        )


@dataclass(frozen=True)
class ArrayShuffleSpec(ShuffleSpec[NDIndex]):
    chunk_lengths: tuple[int, ...]
    axis: int

    @property
    def output_partitions(self) -> Generator[NDIndex, None, None]:
        yield from product(*(range(c) for c in self.chunk_lengths))

    @functools.cached_property
    def positions(self) -> list[int]:
        return [1] + np.cumprod(self.chunk_lengths).tolist()

    def pick_worker(self, partition: NDIndex, workers: Sequence[str]) -> str:
        npartitions = 1
        for c in self.chunk_lengths:
            npartitions *= c
        ix = 0
        for dim, pos in enumerate(partition):
            ix += self.positions[dim] * pos
        i = len(workers) * ix // npartitions
        return workers[i]

    def create_run_on_worker(
        self,
        run_id: int,
        span_id: str | None,
        worker_for: dict[NDIndex, str],
        plugin: ShuffleWorkerPlugin,
    ) -> ShuffleRun:
        return ArrayShuffleRun(
            worker_for=worker_for,
            axis=self.axis,
            id=self.id,
            run_id=run_id,
            span_id=span_id,
            directory=os.path.join(
                plugin.worker.local_directory,
                f"shuffle-{self.id}-{run_id}",
            ),
            executor=plugin._executor,
            local_address=plugin.worker.address,
            rpc=plugin.worker.rpc,
            digest_metric=plugin.worker.digest_metric,
            scheduler=plugin.worker.scheduler,
            memory_limiter_disk=plugin.memory_limiter_disk,
            memory_limiter_comms=plugin.memory_limiter_comms,
            disk=self.disk,
            loop=plugin.worker.loop,
        )


class ArrayShuffleRun(ShuffleRun[NDIndex, "np.ndarray"]):
    """State for a single active rechunk execution

    This object is responsible for splitting, sending, receiving and combining
    data shards.

    It is entirely agnostic to the distributed system and can perform a rechunk
    with other run instances using `rpc``.

    The user of this needs to guarantee that only `ArrayRechunkRun`s of the same unique
    `ShuffleID` and `run_id` interact.

    Parameters
    ----------
    worker_for:
        A mapping partition_id -> worker_address.
    axis: int
        Axis to shuffle along.
    id:
        A unique `ShuffleID` this belongs to.
    run_id:
        A unique identifier of the specific execution of the shuffle this belongs to.
    span_id:
        Span identifier; see :doc:`spans`
    local_address:
        The local address this Shuffle can be contacted by using `rpc`.
    directory:
        The scratch directory to buffer data in.
    executor:
        Thread pool to use for offloading compute.
    rpc:
        A callable returning a PooledRPCCall to contact other Shuffle instances.
        Typically a ConnectionPool.
    digest_metric:
        A callable to ingest a performance metric.
        Typically Server.digest_metric.
    scheduler:
        A PooledRPCCall to contact the scheduler.
    memory_limiter_disk:
    memory_limiter_comm:
        A ``ResourceLimiter`` limiting the total amount of memory used in either
        buffer.
    """

    def __init__(
        self,
        worker_for: dict[NDIndex, str],
        axis: int,
        id: ShuffleId,
        run_id: int,
        span_id: str | None,
        local_address: str,
        directory: str,
        executor: ThreadPoolExecutor,
        rpc: Callable[[str], PooledRPCCall],
        digest_metric: Callable[[Hashable, float], None],
        scheduler: PooledRPCCall,
        memory_limiter_disk: ResourceLimiter,
        memory_limiter_comms: ResourceLimiter,
        disk: bool,
        loop: IOLoop,
    ):
        super().__init__(
            id=id,
            run_id=run_id,
            span_id=span_id,
            local_address=local_address,
            directory=directory,
            executor=executor,
            rpc=rpc,
            digest_metric=digest_metric,
            scheduler=scheduler,
            memory_limiter_comms=memory_limiter_comms,
            memory_limiter_disk=memory_limiter_disk,
            disk=disk,
            loop=loop,
        )
        self.axis = axis
        partitions_of = defaultdict(list)
        for part, addr in worker_for.items():
            partitions_of[addr].append(part)
        self.partitions_of = dict(partitions_of)
        self.worker_for = worker_for

    def _shard_partition(  # type: ignore[override]
        self,
        data: np.ndarray,
        partition_id: NDIndex,
        chunk_indexer: tuple[np.ndarray, np.ndarray],
        chunk_tuple: tuple[int, ...],
    ) -> dict[str, tuple[NDIndex, list[tuple[NDIndex, tuple[NDIndex, np.ndarray]]]]]:
        from dask.array._shuffle import convert_key

        out: dict[str, list[tuple[NDIndex, tuple[NDIndex, np.ndarray]]]] = defaultdict(
            list
        )
        shards_size = 0
        shards_count = 0

        target_chunk_nrs, taker_boundary = np.unique(
            chunk_indexer[1], return_index=True
        )

        for target_chunk in target_chunk_nrs:
            ndslice = [
                slice(None),
            ] * len(data.shape)
            ndslice[self.axis] = chunk_indexer[0][chunk_indexer[1] == target_chunk]
            shard = data[*ndslice]
            # Don't wait until all shards have been transferred over the network
            # before data can be released
            if shard.base is not None:
                shard = shard.copy()

            shards_size += shard.nbytes
            shards_count += 1
            chunk_index = convert_key(chunk_tuple, target_chunk, self.axis)

            out[self.worker_for[chunk_index]].append(
                (chunk_index, (partition_id, shard))
            )

        context_meter.digest_metric("p2p-shards", shards_size, "bytes")
        context_meter.digest_metric("p2p-shards", shards_count, "count")
        return {k: (partition_id, v) for k, v in out.items()}

    async def _receive(
        self,
        data: list[tuple[NDIndex, list[tuple[NDIndex, tuple[NDIndex, np.ndarray]]]]],
    ) -> None:
        self.raise_if_closed()

        # Repartition shards and filter out already received ones
        shards = defaultdict(list)
        for d in data:
            id1, payload = d
            if id1 in self.received:
                continue
            self.received.add(id1)
            for id2, shard in payload:
                shards[id2].append(shard)
            self.total_recvd += sizeof(d)
        del data
        if not shards:
            return

        try:
            await self._write_to_disk(shards)
        except Exception as e:
            self._exception = e
            raise

    def _get_output_partition(
        self, partition_id: NDIndex, key: Key, **kwargs: Any
    ) -> np.ndarray:
        # Quickly read metadata from disk.
        # This is a bunch of seek()'s interleaved with short reads.
        data = self._read_from_disk(partition_id)
        # Copy the memory-mapped buffers from disk into memory.
        # This is where we'll spend most time.
        return _convert_chunk(data, self.axis)

    def deserialize(self, buffer: Any) -> Any:
        return buffer

    def read(self, path: Path) -> tuple[list[list[tuple[NDIndex, np.ndarray]]], int]:
        """Open a memory-mapped file descriptor to disk, read all metadata, and unpickle
        all arrays. This is a fast sequence of short reads interleaved with seeks.
        Do not read in memory the actual data; the arrays' buffers will point to the
        memory-mapped area.

        The file descriptor will be automatically closed by the kernel when all the
        returned arrays are dereferenced, which will happen after the call to
        concatenate3.
        """
        with path.open(mode="r+b") as fh:
            buffer = memoryview(mmap.mmap(fh.fileno(), 0))

        # The file descriptor has *not* been closed!
        shards = list(unpickle_bytestream(buffer))
        return shards, buffer.nbytes

    def _get_assigned_worker(self, id: NDIndex) -> str:
        return self.worker_for[id]


def _convert_chunk(
    shards: list[list[tuple[NDIndex, np.ndarray]]], axis: int
) -> np.ndarray:
    import numpy as np

    indexed: dict[NDIndex, np.ndarray] = {}
    for sublist in shards:
        for index, shard in sublist:
            indexed[index] = shard

    arrs = [indexed[i] for i in sorted(indexed.keys())]
    # This may block for several seconds, as it physically reads the memory-mapped
    # buffers from disk
    return np.concatenate(arrs, axis=axis)

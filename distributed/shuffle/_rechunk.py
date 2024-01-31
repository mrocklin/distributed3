"""
Utilities for rechunking arrays through p2p shuffles
====================================================

Tensors (or n-D arrays) in dask are split up across the workers as
regular n-D "chunks" or bricks. These bricks are stacked up to form
the global array.

A key algorithm for these tensors is to "rechunk" them. That is to
reassemble the same global representation using differently shaped n-D
bricks.

For example, to take an FFT of an n-D array, one uses a sequence of 1D
FFTs along each axis. The implementation in dask (and indeed almost
all distributed array frameworks) requires that 1D
axis along which the FFT is taken is local to a single brick. So to
perform the global FFT we need to arrange that each axis in turn is
local to bricks.

This can be achieved through all-to-all communication between the
workers to exchange sub-pieces of their individual bricks, given a
"rechunking" scheme.

To perform the redistribution, each input brick is cut up into some
number of smaller pieces, each of which contributes to one of the
output bricks. The mapping from input brick to output bricks
decomposes into the Cartesian product of axis-by-axis mappings. To
see this, consider first a 1D example.

Suppose our array is split up into three equally sized bricks::

    |----0----|----1----|----2----|

And the requested output chunks are::

    |--A--|--B--|----C----|---D---|

So brick 0 contributes to output bricks A and B; brick 1 contributes
to B and C; and brick 2 contributes to C and D.

Now consider a 2D example of the same problem::

    +----0----+----1----+----2----+
    |         |         |         |
    α         |         |         |
    |         |         |         |
    +---------+---------+---------+
    |         |         |         |
    β         |         |         |
    |         |         |         |
    +---------+---------+---------+
    |         |         |         |
    γ         |         |         |
    |         |         |         |
    +---------+---------+---------+

Each brick can be described as the ordered pair of row and column
1D bricks, (0, α), (0, β), ..., (2, γ). Since the rechunking does
not also reshape the array, axes do not "interfere" with one another
when determining output bricks::

    +--A--+--B--+----C----+---D---+
    |     |     |         |       |
    Σ     |     |         |       |
    |     |     |         |       |
    +-----+ ----+---------+-------+
    |     |     |         |       |
    |     |     |         |       |
    |     |     |         |       |
    Π     |     |         |       |
    |     |     |         |       |
    |     |     |         |       |
    |     |     |         |       |
    +-----+-----+---------+-------+

Consider the output (B, Σ) brick. This is contributed to by the
input (0, α) and (1, α) bricks. Determination of the subslices is
just done by slicing the the axes separately and combining them.

The key thing to note here is that we never need to create, and
store, the dense 2D mapping, we can instead construct it on the fly
for each output brick in turn as necessary.

The implementation here uses :func:`split_axes` to construct these
1D rechunkings. The output partitioning in
:meth:`~.ArrayRechunkRun.add_partition` then lazily constructs the
subsection of the Cartesian product it needs to determine the slices
of the current input brick.

This approach relies on the generic p2p buffering machinery to
ensure that there are not too many small messages exchanged, since
no special effort is made to minimise messages between workers when
a worker might have two adjacent input bricks that are sliced into
the same output brick.
"""

from __future__ import annotations

import mmap
import os
from collections import defaultdict
from collections.abc import Callable, Generator, Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import toolz
from tornado.ioloop import IOLoop

import dask
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph, MaterializedLayer
from dask.typing import Key

from distributed.core import PooledRPCCall
from distributed.metrics import context_meter
from distributed.shuffle._core import (
    NDIndex,
    ShuffleId,
    ShuffleRun,
    ShuffleSpec,
    get_worker_plugin,
    handle_transfer_errors,
    handle_unpack_errors,
)
from distributed.shuffle._limiter import ResourceLimiter
from distributed.shuffle._pickle import unpickle_bytestream
from distributed.shuffle._shuffle import barrier_key, shuffle_barrier
from distributed.shuffle._worker_plugin import ShuffleWorkerPlugin
from distributed.sizeof import sizeof

if TYPE_CHECKING:
    import numpy as np
    from typing_extensions import TypeAlias

    import dask.array as da

ChunkedAxis: TypeAlias = tuple[float, ...]  # chunks must either be an int or NaN
ChunkedAxes: TypeAlias = tuple[ChunkedAxis, ...]
NDSlice: TypeAlias = tuple[slice, ...]


def rechunk_transfer(
    input: np.ndarray,
    id: ShuffleId,
    input_chunk: NDIndex,
    new: ChunkedAxes,
    old: ChunkedAxes,
    disk: bool,
) -> int:
    with handle_transfer_errors(id):
        return get_worker_plugin().add_partition(
            input,
            partition_id=input_chunk,
            spec=ArrayRechunkSpec(id=id, new=new, old=old, disk=disk),
        )


def rechunk_unpack(
    id: ShuffleId, output_chunk: NDIndex, barrier_run_id: int
) -> np.ndarray:
    with handle_unpack_errors(id):
        return get_worker_plugin().get_output_partition(
            id, barrier_run_id, output_chunk
        )


class _Partial(NamedTuple):
    """Information used to perform a partial rechunk along a single axis"""

    #: Slice of the old chunks along this axis that belong to the partial
    old: slice
    #: Slice of the new chunks along this axis that belong to the partial
    new: slice
    #: Index of the first value of the left-most old chunk along this axis
    #: to include in this partial. Everything left to this index belongs to
    #: the previous partial.
    left_start: int
    #: Index of the first value of the right-most old chunk along this axis
    #: to exclude from this partial.
    #: This corresponds to `left_start` of the subsequent partial.
    right_stop: int


class _NDPartial(NamedTuple):
    """Information used to perform a partial rechunk along all axes"""

    #: n-dimensional slice of the old chunks along each axis that belong to the partial
    old: NDSlice
    #: n-dimensional slice of the new chunks along each axis that belong to the partial
    new: NDSlice
    #: Indices of the first value of the left-most old chunk along each axis
    #: to include in this partial. Everything left to this index belongs to
    #: the previous partial.
    left_starts: NDIndex
    #: Indices of the first value of the right-most old chunk along each axis
    #: to exclude from this partial.
    #: This corresponds to `left_start` of the subsequent partial.
    right_stops: NDIndex


def rechunk_p2p(x: da.Array, chunks: ChunkedAxes) -> da.Array:
    import dask.array as da

    if x.size == 0:
        # Special case for empty array, as the algorithm below does not behave correctly
        return da.empty(x.shape, chunks=chunks, dtype=x.dtype)

    dsk = {}
    token = tokenize(x, chunks)
    name = f"rechunk-p2p-{token}"
    for ndpartial in _split_partials(x, chunks):
        if all(slc.stop == slc.start + 1 for slc in ndpartial.new):
            # Single output chunk
            dsk.update(partial_concatenate(x, chunks, ndpartial, name))
        else:
            dsk.update(partial_rechunk(x, chunks, ndpartial, name))
    layer = MaterializedLayer(dsk)
    graph = HighLevelGraph.from_collections(name, layer, dependencies=[x])
    arr = da.Array(graph, name, chunks, meta=x)
    return arr


def _split_partials(
    x: da.Array, chunks: ChunkedAxes
) -> Generator[_NDPartial, None, None]:
    """Split the rechunking into partials that can be performed separately"""
    partials_per_axis = _split_partials_per_axis(x, chunks)
    for partial_per_axis in product(*partials_per_axis):
        old, new, left_starts, right_stops = zip(*partial_per_axis)
        yield _NDPartial(old, new, left_starts, right_stops)


def _split_partials_per_axis(
    x: da.Array, chunks: ChunkedAxes
) -> tuple[tuple[_Partial, ...], ...]:
    """Split the rechunking into partials that can be performed separately
    on each axis"""
    from dask.array.rechunk import old_to_new

    chunked_shape = tuple(len(axis) for axis in chunks)
    _old_to_new = old_to_new(x.chunks, chunks)

    sliced_axes = _partial_slices(_old_to_new, chunked_shape)

    partial_axes = []
    for axis_index, slices in enumerate(sliced_axes):
        partials = []
        for slice_ in slices:
            last_old_chunk: int
            first_old_chunk, first_old_slice = _old_to_new[axis_index][slice_.start][0]
            last_old_chunk, last_old_slice = _old_to_new[axis_index][slice_.stop - 1][
                -1
            ]
            partials.append(
                _Partial(
                    old=slice(first_old_chunk, last_old_chunk + 1),
                    new=slice_,
                    left_start=first_old_slice.start,
                    right_stop=last_old_slice.stop,
                )
            )
        partial_axes.append(tuple(partials))
    return tuple(partial_axes)


def _partial_slices(
    old_to_new: list[list[list[tuple[int, slice]]]], chunked_shape: NDIndex
) -> tuple[tuple[slice, ...], ...]:
    """Compute the slices of the new chunks that can be computed separately"""
    sliced_axes = []

    for axis_index, old_to_new_axis in enumerate(old_to_new):
        # Two consecutive output chunks A and B belong to the same partial rechunk
        # if B is fully included in the right-most input chunk of A, i.e.,
        # separating A and B would not allow us to cull more input tasks.

        # Index of the last input chunk of this partial rechunk
        last_old_chunk: int | None = None
        partial_splits = [0]
        recipe: list[tuple[int, slice]]
        for new_chunk_index, recipe in enumerate(old_to_new_axis):
            if len(recipe) == 0:
                continue
            current_last_old_chunk, old_slice = recipe[-1]
            if last_old_chunk is None:
                last_old_chunk = current_last_old_chunk
            elif last_old_chunk != current_last_old_chunk:
                partial_splits.append(new_chunk_index)
                last_old_chunk = current_last_old_chunk
        partial_splits.append(chunked_shape[axis_index])
        sliced_axes.append(
            tuple(slice(a, b) for a, b in toolz.sliding_window(2, partial_splits))
        )
    return tuple(sliced_axes)


def _partial_ndindex(ndslice: NDSlice) -> np.ndindex:
    import numpy as np

    return np.ndindex(tuple(slice.stop - slice.start for slice in ndslice))


def _global_index(partial_index: NDIndex, partial_offset: NDIndex) -> NDIndex:
    return tuple(index + offset for index, offset in zip(partial_index, partial_offset))


def partial_concatenate(
    x: da.Array,
    chunks: ChunkedAxes,
    ndpartial: _NDPartial,
    name: str,
) -> dict[Key, Any]:
    import numpy as np

    from dask.array.chunk import getitem
    from dask.array.core import concatenate3

    dsk: dict[Key, Any] = {}

    partial_token = tokenize(x, chunks, ndpartial.new)
    slice_name = f"rechunk-slice-{partial_token}"
    old_offset = tuple(slice_.start for slice_ in ndpartial.old)

    shape = tuple(slice_.stop - slice_.start for slice_ in ndpartial.old)
    rec_cat_arg = np.empty(shape, dtype="O")

    partial_old = _compute_partial_old_chunks(ndpartial, x.chunks)

    for old_partial_index in _partial_ndindex(ndpartial.old):
        old_global_index = _global_index(old_partial_index, old_offset)
        # TODO: Precompute slicing to avoid duplicate work
        ndslice = ndslice_for(
            old_partial_index, partial_old, ndpartial.left_starts, ndpartial.right_stops
        )

        original_shape = tuple(
            axis[index] for index, axis in zip(old_global_index, x.chunks)
        )
        if _slicing_is_necessary(ndslice, original_shape):
            rec_cat_arg[old_partial_index] = (slice_name,) + old_global_index
            dsk[(slice_name,) + old_global_index] = (
                getitem,
                (x.name,) + old_global_index,
                ndslice,
            )
        else:
            rec_cat_arg[old_partial_index] = (x.name,) + old_global_index
    global_index = tuple(int(slice_.start) for slice_ in ndpartial.new)
    dsk[(name,) + global_index] = (
        concatenate3,
        rec_cat_arg.tolist(),
    )
    return dsk


def _compute_partial_old_chunks(
    partial: _NDPartial, chunks: ChunkedAxes
) -> ChunkedAxes:
    _partial_old = []
    for axis_index in range(len(partial.old)):
        c = list(chunks[axis_index][partial.old[axis_index]])
        c[0] = c[0] - partial.left_starts[axis_index]
        if (stop := partial.right_stops[axis_index]) is not None:
            c[-1] = stop
        _partial_old.append(tuple(c))
    return tuple(_partial_old)


def _slicing_is_necessary(slice: NDSlice, shape: tuple[int | None, ...]) -> bool:
    """Return True if applying the slice alters the shape, False otherwise."""
    return not all(
        slc.start == 0 and (size is None and slc.stop is None or slc.stop == size)
        for slc, size in zip(slice, shape)
    )


def partial_rechunk(
    x: da.Array,
    chunks: ChunkedAxes,
    ndpartial: _NDPartial,
    name: str,
) -> dict[Key, Any]:
    from dask.array.chunk import getitem

    dsk: dict[Key, Any] = {}

    old_partial_offset = tuple(slice_.start for slice_ in ndpartial.old)

    partial_token = tokenize(x, chunks, ndpartial.new)
    _barrier_key = barrier_key(ShuffleId(partial_token))
    slice_name = f"rechunk-slice-{partial_token}"
    transfer_name = f"rechunk-transfer-{partial_token}"
    disk: bool = dask.config.get("distributed.p2p.disk")

    ndim = len(x.shape)

    partial_old = _compute_partial_old_chunks(ndpartial, x.chunks)
    partial_new: ChunkedAxes = tuple(
        chunks[axis_index][ndpartial.new[axis_index]] for axis_index in range(ndim)
    )

    transfer_keys = []
    for partial_index in _partial_ndindex(ndpartial.old):
        ndslice = ndslice_for(
            partial_index, partial_old, ndpartial.left_starts, ndpartial.right_stops
        )

        global_index = _global_index(partial_index, old_partial_offset)

        original_shape = tuple(
            axis[index] for index, axis in zip(global_index, x.chunks)
        )
        if _slicing_is_necessary(ndslice, original_shape):
            input_task = (slice_name,) + global_index
            dsk[(slice_name,) + global_index] = (
                getitem,
                (x.name,) + global_index,
                ndslice,
            )
        else:
            input_task = (x.name,) + global_index

        transfer_keys.append((transfer_name,) + global_index)
        dsk[(transfer_name,) + global_index] = (
            rechunk_transfer,
            input_task,
            partial_token,
            partial_index,
            partial_new,
            partial_old,
            disk,
        )

    dsk[_barrier_key] = (shuffle_barrier, partial_token, transfer_keys)

    new_partial_offset = tuple(axis.start for axis in ndpartial.new)
    for partial_index in _partial_ndindex(ndpartial.new):
        global_index = _global_index(partial_index, new_partial_offset)
        dsk[(name,) + global_index] = (
            rechunk_unpack,
            partial_token,
            partial_index,
            _barrier_key,
        )
    return dsk


def ndslice_for(
    partial_index: NDIndex,
    chunks: ChunkedAxes,
    left_starts: NDIndex,
    right_stops: NDIndex,
) -> NDSlice:
    slices = []
    shape = tuple(len(axis) for axis in chunks)
    for axis_index, chunked_axis in enumerate(chunks):
        chunk_index = partial_index[axis_index]
        start = left_starts[axis_index] if chunk_index == 0 else 0
        stop = (
            right_stops[axis_index]
            if chunk_index == shape[axis_index] - 1
            else chunked_axis[chunk_index] + start
        )
        slices.append(slice(start, stop))
    return tuple(slices)


class Split(NamedTuple):
    """Slice of a chunk that is concatenated with other splits to create a new chunk

    Splits define how to slice an input chunk on a single axis into small pieces
    that can be concatenated together with splits from other input chunks to create
    output chunks of a rechunk operation.
    """

    #: Index of the new output chunk to which this split belongs.
    chunk_index: int

    #: Index of the split within the list of splits that are concatenated
    #: to create the new chunk.
    split_index: int

    #: Slice of the input chunk.
    slice: slice


SplitChunk: TypeAlias = list[Split]
SplitAxis: TypeAlias = list[SplitChunk]
SplitAxes: TypeAlias = list[SplitAxis]


def split_axes(old: ChunkedAxes, new: ChunkedAxes) -> SplitAxes:
    """Calculate how to split the old chunks on each axis to create the new chunks

    Parameters
    ----------
    old : ChunkedAxes
        Chunks along each axis of the old array
    new : ChunkedAxes
        Chunks along each axis of the new array

    Returns
    -------
    SplitAxes
        Splits along each axis that determine how to slice the input chunks to create
        the new chunks by concatenating the resulting shards.
    """
    from dask.array.rechunk import old_to_new

    _old_to_new = old_to_new(old, new)

    axes = []
    for axis_index, new_axis in enumerate(_old_to_new):
        old_axis: SplitAxis = [[] for _ in old[axis_index]]
        for new_chunk_index, new_chunk in enumerate(new_axis):
            for split_index, (old_chunk_index, slice) in enumerate(new_chunk):
                old_axis[old_chunk_index].append(
                    Split(new_chunk_index, split_index, slice)
                )
        for old_chunk in old_axis:
            old_chunk.sort(key=lambda split: split.slice.start)
        axes.append(old_axis)
    return axes


def convert_chunk(shards: list[list[tuple[NDIndex, np.ndarray]]]) -> np.ndarray:
    import numpy as np

    from dask.array.core import concatenate3

    indexed: dict[NDIndex, np.ndarray] = {}
    for sublist in shards:
        for index, shard in sublist:
            indexed[index] = shard

    subshape = [max(dim) + 1 for dim in zip(*indexed.keys())]
    assert len(indexed) == np.prod(subshape)

    rec_cat_arg = np.empty(subshape, dtype="O")
    for index, shard in indexed.items():
        rec_cat_arg[tuple(index)] = shard
    arrs = rec_cat_arg.tolist()

    # This may block for several seconds, as it physically reads the memory-mapped
    # buffers from disk
    return concatenate3(arrs)


class ArrayRechunkRun(ShuffleRun[NDIndex, "np.ndarray"]):
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
    old:
        Existing chunking of the array per dimension.
    new:
        Desired chunking of the array per dimension.
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
        old: ChunkedAxes,
        new: ChunkedAxes,
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
        self.old = old
        self.new = new
        partitions_of = defaultdict(list)
        for part, addr in worker_for.items():
            partitions_of[addr].append(part)
        self.partitions_of = dict(partitions_of)
        self.worker_for = worker_for
        self.split_axes = split_axes(old, new)

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

    def _shard_partition(
        self, data: np.ndarray, partition_id: NDIndex
    ) -> dict[str, tuple[NDIndex, list[tuple[NDIndex, tuple[NDIndex, np.ndarray]]]]]:
        out: dict[str, list[tuple[NDIndex, tuple[NDIndex, np.ndarray]]]] = defaultdict(
            list
        )
        shards_size = 0
        shards_count = 0

        ndsplits = product(*(axis[i] for axis, i in zip(self.split_axes, partition_id)))

        for ndsplit in ndsplits:
            chunk_index, shard_index, ndslice = zip(*ndsplit)

            shard = data[ndslice]
            # Don't wait until all shards have been transferred over the network
            # before data can be released
            if shard.base is not None:
                shard = shard.copy()

            shards_size += shard.nbytes
            shards_count += 1

            out[self.worker_for[chunk_index]].append(
                (chunk_index, (shard_index, shard))
            )

        context_meter.digest_metric("p2p-shards", shards_size, "bytes")
        context_meter.digest_metric("p2p-shards", shards_count, "count")
        return {k: (partition_id, v) for k, v in out.items()}

    def _get_output_partition(
        self, partition_id: NDIndex, key: Key, **kwargs: Any
    ) -> np.ndarray:
        # Quickly read metadata from disk.
        # This is a bunch of seek()'s interleaved with short reads.
        data = self._read_from_disk(partition_id)
        # Copy the memory-mapped buffers from disk into memory.
        # This is where we'll spend most time.
        return convert_chunk(data)

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


@dataclass(frozen=True)
class ArrayRechunkSpec(ShuffleSpec[NDIndex]):
    new: ChunkedAxes
    old: ChunkedAxes

    @property
    def output_partitions(self) -> Generator[NDIndex, None, None]:
        yield from product(*(range(len(c)) for c in self.new))

    def pick_worker(self, partition: NDIndex, workers: Sequence[str]) -> str:
        npartitions = 1
        for c in self.new:
            npartitions *= len(c)
        ix = 0
        for dim, pos in enumerate(partition):
            if dim > 0:
                ix += len(self.new[dim - 1]) * pos
            else:
                ix += pos
        i = len(workers) * ix // npartitions
        return workers[i]

    def create_run_on_worker(
        self,
        run_id: int,
        span_id: str | None,
        worker_for: dict[NDIndex, str],
        plugin: ShuffleWorkerPlugin,
    ) -> ShuffleRun:
        return ArrayRechunkRun(
            worker_for=worker_for,
            old=self.old,
            new=self.new,
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
            scheduler=plugin.worker.scheduler_ordered,  # type: ignore
            memory_limiter_disk=plugin.memory_limiter_disk,
            memory_limiter_comms=plugin.memory_limiter_comms,
            disk=self.disk,
            loop=plugin.worker.loop,
        )

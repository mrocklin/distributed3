from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

import toolz
from tornado.ioloop import IOLoop

import dask
from dask.base import tokenize
from dask.highlevelgraph import HighLevelGraph
from dask.layers import Layer
from dask.typing import Key

from distributed.core import PooledRPCCall
from distributed.exceptions import Reschedule
from distributed.shuffle._arrow import (
    check_dtype_support,
    check_minimal_arrow_version,
    concat_tables,
    convert_shards,
    copy_table,
    read_from_disk,
    write_to_disk,
)
from distributed.shuffle._core import (
    NDIndex,
    ShuffleId,
    ShuffleRun,
    ShuffleSpec,
    barrier_key,
    get_worker_plugin,
    handle_transfer_errors,
    handle_unpack_errors,
)
from distributed.shuffle._limiter import ResourceLimiter
from distributed.shuffle._scheduler_plugin import ShuffleSchedulerPlugin
from distributed.shuffle._worker_plugin import ShuffleWorkerPlugin
from distributed.sizeof import sizeof

logger = logging.getLogger("distributed.shuffle")
if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

    # TODO import from typing (requires Python >=3.10)
    from typing_extensions import TypeAlias

    from dask.dataframe import DataFrame


def shuffle_transfer(
    input: pd.DataFrame,
    id: ShuffleId,
    input_partition: int,
    npartitions: int,
    column: str,
    meta: pd.DataFrame,
    parts_out: set[int],
    disk: bool,
) -> int:
    with handle_transfer_errors(id):
        return get_worker_plugin().add_partition(
            input,
            input_partition,
            spec=DataFrameShuffleSpec(
                id=id,
                npartitions=npartitions,
                column=column,
                meta=meta,
                parts_out=parts_out,
                disk=disk,
            ),
        )


def shuffle_unpack(
    id: ShuffleId, output_partition: int, barrier_run_id: int
) -> pd.DataFrame:
    with handle_unpack_errors(id):
        return get_worker_plugin().get_output_partition(
            id, barrier_run_id, output_partition
        )


def shuffle_barrier(id: ShuffleId, run_ids: list[int]) -> int:
    try:
        return get_worker_plugin().barrier(id, run_ids)
    except Reschedule as e:
        raise e
    except Exception as e:
        raise RuntimeError(f"shuffle_barrier failed during shuffle {id}") from e


def rearrange_by_column_p2p(
    df: DataFrame,
    column: str,
    npartitions: int | None = None,
) -> DataFrame:
    import pandas as pd

    from dask.dataframe.core import new_dd_object

    meta = df._meta
    if not pd.api.types.is_integer_dtype(meta[column].dtype):
        raise TypeError(
            f"Expected meta {column=} to be an integer column, is {meta[column].dtype}."
        )
    check_dtype_support(meta)
    npartitions = npartitions or df.npartitions
    token = tokenize(df, column, npartitions)

    if any(not isinstance(c, str) for c in meta.columns):
        unsupported = {c: type(c) for c in meta.columns if not isinstance(c, str)}
        raise TypeError(
            f"p2p requires all column names to be str, found: {unsupported}",
        )

    name = f"shuffle_p2p-{token}"
    disk: bool = dask.config.get("distributed.p2p.disk")

    layer = P2PShuffleLayer(
        name,
        column,
        npartitions,
        npartitions_input=df.npartitions,
        name_input=df._name,
        meta_input=meta,
        disk=disk,
    )
    return new_dd_object(
        HighLevelGraph.from_collections(name, layer, [df]),
        name,
        meta,
        [None] * (npartitions + 1),
    )


_T_LowLevelGraph: TypeAlias = dict[Key, tuple]


class P2PShuffleLayer(Layer):
    name: str
    column: str
    npartitions: int
    npartitions_input: int
    name_input: str
    meta_input: pd.DataFrame
    disk: bool
    parts_out: set[int]

    def __init__(
        self,
        name: str,
        column: str,
        npartitions: int,
        npartitions_input: int,
        name_input: str,
        meta_input: pd.DataFrame,
        disk: bool,
        parts_out: Iterable[int] | None = None,
        annotations: dict | None = None,
    ):
        check_minimal_arrow_version()
        self.name = name
        self.column = column
        self.npartitions = npartitions
        self.name_input = name_input
        self.meta_input = meta_input
        self.disk = disk
        if parts_out:
            self.parts_out = set(parts_out)
        else:
            self.parts_out = set(range(self.npartitions))
        self.npartitions_input = npartitions_input
        super().__init__(annotations=annotations)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}<name='{self.name}', npartitions={self.npartitions}>"
        )

    def get_output_keys(self) -> set[Key]:
        return {(self.name, part) for part in self.parts_out}

    def is_materialized(self) -> bool:
        return hasattr(self, "_cached_dict")

    @property
    def _dict(self) -> _T_LowLevelGraph:
        """Materialize full dict representation"""
        self._cached_dict: _T_LowLevelGraph
        dsk: _T_LowLevelGraph
        if hasattr(self, "_cached_dict"):
            return self._cached_dict
        else:
            dsk = self._construct_graph()
            self._cached_dict = dsk
        return self._cached_dict

    def __getitem__(self, key: Key) -> tuple:
        return self._dict[key]

    def __iter__(self) -> Iterator[Key]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def _cull(self, parts_out: Iterable[int]) -> P2PShuffleLayer:
        return P2PShuffleLayer(
            self.name,
            self.column,
            self.npartitions,
            self.npartitions_input,
            self.name_input,
            self.meta_input,
            self.disk,
            parts_out=parts_out,
        )

    def _keys_to_parts(self, keys: Iterable[Key]) -> set[int]:
        """Simple utility to convert keys to partition indices."""
        parts = set()
        for key in keys:
            if isinstance(key, tuple) and len(key) == 2:
                name, part = key
                if name == self.name:
                    assert isinstance(part, int)
                    parts.add(part)
        return parts

    def cull(
        self, keys: set[Key], all_keys: Collection[Key]
    ) -> tuple[P2PShuffleLayer, dict]:
        """Cull a P2PShuffleLayer HighLevelGraph layer.

        The underlying graph will only include the necessary
        tasks to produce the keys (indices) included in `parts_out`.
        Therefore, "culling" the layer only requires us to reset this
        parameter.
        """
        parts_out = self._keys_to_parts(keys)
        # Protect against mutations later on with frozenset
        input_parts = frozenset(
            {(self.name_input, i) for i in range(self.npartitions_input)}
        )
        culled_deps = {(self.name, part): input_parts for part in parts_out}

        if parts_out != set(self.parts_out):
            culled_layer = self._cull(parts_out)
            return culled_layer, culled_deps
        else:
            return self, culled_deps

    def _construct_graph(self) -> _T_LowLevelGraph:
        token = tokenize(self.name_input, self.column, self.npartitions, self.parts_out)
        dsk: _T_LowLevelGraph = {}
        _barrier_key = barrier_key(ShuffleId(token))
        name = "shuffle-transfer-" + token
        transfer_keys = list()
        for i in range(self.npartitions_input):
            transfer_keys.append((name, i))
            dsk[(name, i)] = (
                shuffle_transfer,
                (self.name_input, i),
                token,
                i,
                self.npartitions,
                self.column,
                self.meta_input,
                self.parts_out,
                self.disk,
            )

        dsk[_barrier_key] = (shuffle_barrier, token, transfer_keys)

        name = self.name
        for part_out in self.parts_out:
            dsk[(name, part_out)] = (
                shuffle_unpack,
                token,
                part_out,
                _barrier_key,
            )
        return dsk


def split_by_worker(
    df: pd.DataFrame, column: str, meta: pd.DataFrame, worker_for: pd.Series
) -> dict[Any, pa.Table]:
    """
    Split data into many arrow batches, partitioned by destination worker
    """
    import numpy as np

    from dask.dataframe.dispatch import to_pyarrow_table_dispatch

    # (cudf support) Avoid pd.Series
    constructor = df._constructor_sliced
    assert isinstance(constructor, type)
    worker_for = constructor(worker_for)

    df = (
        worker_for.cat.codes.rename("_worker")
        .sort_values()
        .to_frame()
        .merge(
            right=df,
            right_on=column,
            left_index=True,
            how="inner",
        )
    )
    assert df["_worker"].is_monotonic_increasing
    nrows = len(df)
    if not nrows:
        return {}
    t = to_pyarrow_table_dispatch(df)
    codes = np.asarray(t["_worker"])
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


def split_by_partition(t: pa.Table, column: str) -> dict[int, pa.Table]:
    """
    Split data into many arrow batches, partitioned by final partition
    """
    import numpy as np

    partitions = t.select([column]).to_pandas()[column].unique()
    partitions.sort()

    partition = np.asarray(t[column])
    splits = np.where(partition[1:] != partition[:-1])[0] + 1
    splits = np.concatenate([[0], splits])

    shards = [
        t.slice(offset=a, length=b - a) for a, b in toolz.sliding_window(2, splits)
    ]
    shards.append(t.slice(offset=splits[-1], length=None))
    shards = [copy_table(shard) for shard in shards]
    assert len(t) == sum(map(len, shards))
    assert len(partitions) == len(shards)
    return dict(zip(partitions, shards))


class DataFrameShuffleRun(ShuffleRun[int, "pd.DataFrame"]):
    """State for a single active shuffle execution

    This object is responsible for splitting, sending, receiving and combining
    data shards.

    It is entirely agnostic to the distributed system and can perform a shuffle
    with other run instances using `rpc`.

    The user of this needs to guarantee that only `DataFrameShuffleRun`s of the
    same unique `ShuffleID` and `run_id` interact.

    Parameters
    ----------
    worker_for:
        A mapping partition_id -> worker_address.
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

    column: str
    meta: pd.DataFrame
    worker_for: pd.Series

    def __init__(
        self,
        worker_for: dict[int, str],
        column: str,
        meta: pd.DataFrame,
        id: ShuffleId,
        run_id: int,
        local_address: str,
        directory: str,
        executor: ThreadPoolExecutor,
        io_executor: ThreadPoolExecutor,
        rpc: Callable[[str], PooledRPCCall],
        scheduler: PooledRPCCall,
        memory_limiter_disk: ResourceLimiter,
        memory_limiter_comms: ResourceLimiter,
        disk: bool,
        loop: IOLoop,
    ):
        import pandas as pd

        super().__init__(
            id=id,
            run_id=run_id,
            local_address=local_address,
            directory=directory,
            executor=executor,
            io_executor=io_executor,
            rpc=rpc,
            scheduler=scheduler,
            memory_limiter_comms=memory_limiter_comms,
            memory_limiter_disk=memory_limiter_disk,
            disk=disk,
            loop=loop,
        )
        self.column = column
        self.meta = meta
        partitions_of = defaultdict(list)
        for part, addr in worker_for.items():
            partitions_of[addr].append(part)
        self.partitions_of = dict(partitions_of)
        self.worker_for = pd.Series(worker_for, name="_workers").astype("category")

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

    def _repartition_buffers(
        self, data: list[pa.Table]
    ) -> dict[NDIndex, tuple[pa.Table, int]]:
        nrows = sum(map(len, data))
        shards: defaultdict[int, list[Any]] = defaultdict(lambda: [[], 0])
        while data:
            table = data.pop()
            metadata_size = sizeof(table.schema.metadata)
            groups = split_by_partition(table, self.column)
            for key, shard in groups.items():
                shards[key][0].append(shard)
                shards[key][1] += metadata_size + shard.nbytes
        actual = sum(map(len, chain(*toolz.pluck(0, shards.values()))))
        assert actual == nrows
        return {
            (k,): (concat_tables(shards), size) for k, (shards, size) in shards.items()
        }

    def _shard_partition(
        self,
        data: pd.DataFrame,
        partition_id: int,
        **kwargs: Any,
    ) -> dict[str, tuple[int, bytes]]:
        out = split_by_worker(
            data,
            self.column,
            self.meta,
            self.worker_for,
        )
        del data
        out = {k: (partition_id, t) for k, t in out.items()}
        return out

    def _get_output_partition(
        self,
        partition_id: int,
        key: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        try:
            data = self._read_from_disk((partition_id,))
            return convert_shards(data, self.meta)
        except KeyError:
            return self.meta.copy()

    def _get_assigned_worker(self, id: int) -> str:
        return self.worker_for[id]

    def write(self, data: list[pd.DataFrame], path: Path) -> int:
        return write_to_disk(data, path)

    def read(self, path: Path) -> tuple[pa.Table, int]:
        return read_from_disk(path)


@dataclass(frozen=True)
class DataFrameShuffleSpec(ShuffleSpec[int]):
    npartitions: int
    column: str
    meta: pd.DataFrame
    parts_out: set[int]

    def _pin_output_workers(self, plugin: ShuffleSchedulerPlugin) -> dict[int, str]:
        pick_worker = partial(_get_worker_for_range_sharding, self.npartitions)
        return plugin._pin_output_workers(self.id, self.parts_out, pick_worker)

    def create_run_on_worker(
        self, run_id: int, worker_for: dict[int, str], plugin: ShuffleWorkerPlugin
    ) -> ShuffleRun:
        return DataFrameShuffleRun(
            column=self.column,
            meta=self.meta,
            worker_for=worker_for,
            id=self.id,
            run_id=run_id,
            directory=os.path.join(
                plugin.worker.local_directory,
                f"shuffle-{self.id}-{run_id}",
            ),
            executor=plugin._executor,
            io_executor=plugin._io_executor,
            local_address=plugin.worker.address,
            rpc=plugin.worker.rpc,
            scheduler=plugin.worker.scheduler,
            memory_limiter_disk=plugin.memory_limiter_disk,
            memory_limiter_comms=plugin.memory_limiter_comms,
            disk=self.disk,
            loop=plugin.worker.loop,
        )


def _get_worker_for_range_sharding(
    npartitions: int, output_partition: int, workers: Sequence[str]
) -> str:
    """Get address of target worker for this output partition using range sharding"""
    i = len(workers) * output_partition // npartitions
    return workers[i]

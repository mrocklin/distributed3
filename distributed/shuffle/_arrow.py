from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from packaging.version import parse

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa


def check_dtype_support(meta_input: pd.DataFrame) -> None:
    import pandas as pd

    for name in meta_input:
        column = meta_input[name]
        # FIXME: PyArrow does not support complex numbers: https://issues.apache.org/jira/browse/ARROW-638
        if pd.api.types.is_complex_dtype(column):
            raise TypeError(
                f"p2p does not support data of type '{column.dtype}' found in column '{name}'."
            )
        # FIXME: PyArrow does not support sparse data: https://issues.apache.org/jira/browse/ARROW-8679
        if isinstance(column.dtype, pd.SparseDtype):
            raise TypeError("p2p does not support sparse data found in column '{name}'")


def check_minimal_arrow_version() -> None:
    """Verify that the the correct version of pyarrow is installed to support
    the P2P extension.

    Raises a ModuleNotFoundError if pyarrow is not installed or an
    ImportError if the installed version is not recent enough.
    """
    minversion = "7.0.0"
    try:
        import pyarrow as pa
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"P2P shuffling requires pyarrow>={minversion}")
    if parse(pa.__version__) < parse(minversion):
        raise ImportError(
            f"P2P shuffling requires pyarrow>={minversion} but only found {pa.__version__}"
        )


def combine_tables(tables: Iterable[pa.Table], deep_copy: bool = True) -> pa.Table:
    table = concat_tables(tables)
    if deep_copy:
        return copy_table(table)
    return table.combine_chunks()


def copy_table(table: pa.Table) -> pa.Table:
    """Creates a deep-copy of the table"""
    import pyarrow as pa

    # concat_arrays forced a deep-copy even if the input arrays only have a single chunk.
    return pa.table(
        [concat_arrays(column.chunks) for column in table.columns], schema=table.schema
    )


def concat_tables(tables: Iterable[pa.Table]) -> pa.Table:
    import pyarrow as pa

    if parse(pa.__version__) >= parse("14.0.0"):
        return pa.concat_tables(tables, promote_options="permissive")
    try:
        return pa.concat_tables(tables, promote=True)
    except pa.ArrowNotImplementedError as e:
        if parse(pa.__version__) >= parse("12.0.0"):
            raise e
        raise


def convert_shards(shards: list[pa.Table], meta: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    from pandas.core.dtypes.cast import find_common_type  # type: ignore[attr-defined]

    from dask.dataframe.dispatch import from_pyarrow_table_dispatch

    table = concat_tables(shards)

    df = from_pyarrow_table_dispatch(meta, table, self_destruct=True)
    reconciled_dtypes = {}
    for column, dtype in meta.dtypes.items():
        actual = df[column].dtype
        if actual == dtype:
            continue
        # Use the specific string dtype from meta (e.g., string[pyarrow])
        if isinstance(actual, pd.StringDtype) and isinstance(dtype, pd.StringDtype):
            reconciled_dtypes[column] = dtype
            continue
        # meta might not be aware of the actual categories so the two dtype objects are not equal
        # Also, the categories_dtype does not properly roundtrip through Arrow
        if isinstance(actual, pd.CategoricalDtype) and isinstance(
            dtype, pd.CategoricalDtype
        ):
            continue
        reconciled_dtypes[column] = find_common_type([actual, dtype])
    return df.astype(reconciled_dtypes, copy=False)


def list_of_buffers_to_table(data: list[bytes]) -> pa.Table:
    """Convert a list of arrow buffers and a schema to an Arrow Table"""

    tables = (buffer for buffer in data)
    return concat_tables(tables)


def serialize_table(table: pa.Table) -> bytes:
    import pyarrow as pa

    stream = pa.BufferOutputStream()
    with pa.ipc.new_stream(stream, table.schema) as writer:
        writer.write_table(table)
    return stream.getvalue()


def deserialize_table(buffer: bytes) -> pa.Table:
    import pyarrow as pa

    with pa.ipc.open_stream(pa.py_buffer(buffer)) as reader:
        return reader.read_all()


def write_to_disk(data: list[pa.Table], path: Path) -> int:
    import pyarrow as pa

    table = combine_tables(data)
    del data
    with path.open(mode="ab") as f:
        start = f.tell()
        with pa.ipc.new_stream(f, table.schema) as writer:
            writer.write_table(table)
        return f.tell() - start


def read_from_disk(path: Path) -> tuple[list[pa.Table], int]:
    import pyarrow as pa

    shards = []

    with pa.OSFile(str(path), mode="rb") as f:
        size = f.seek(0, whence=2)
        f.seek(0)
        while f.tell() < size:
            sr = pa.RecordBatchStreamReader(f)
            shard = sr.read_all()
            shards.append(shard)

    if shards:
        shards = [combine_tables(shards)]
    return shards, size


def concat_arrays(arrays: Iterable[pa.Array]) -> pa.Array:
    import pyarrow as pa

    try:
        return pa.concat_arrays(arrays)
    except pa.ArrowNotImplementedError as e:
        if parse(pa.__version__) >= parse("12.0.0"):
            raise
        if e.args[0].startswith("concatenation of extension"):
            raise RuntimeError(
                "P2P shuffling requires pyarrow>=12.0.0 to support extension types."
            ) from e
        raise

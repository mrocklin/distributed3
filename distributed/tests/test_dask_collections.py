from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from packaging.version import parse as parse_version
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

import dask
import dask.bag as db
import dask.dataframe as dd
from dask.dataframe._compat import PANDAS_GE_220

from distributed.client import wait
from distributed.nanny import Nanny
from distributed.utils_test import gen_cluster

dfs = [
    pd.DataFrame({"x": [1, 2, 3]}, index=[0, 10, 20]),
    pd.DataFrame({"x": [4, 5, 6]}, index=[30, 40, 50]),
    pd.DataFrame({"x": [7, 8, 9]}, index=[60, 70, 80]),
]


ignore_single_machine_warning = pytest.mark.filterwarnings(
    "ignore:Running on a single-machine scheduler:UserWarning"
)


def assert_equal(a, b):
    assert type(a) == type(b)
    if isinstance(a, pd.DataFrame):
        assert_frame_equal(a, b)
    elif isinstance(a, pd.Series):
        assert_series_equal(a, b)
    elif isinstance(a, pd.Index):
        assert_index_equal(a, b)
    else:
        assert a == b


@ignore_single_machine_warning
@gen_cluster(client=True)
async def test_dataframes(c, s, a, b):
    df = pd.DataFrame(
        {"x": np.random.random(1000), "y": np.random.random(1000)},
        index=np.arange(1000),
    )
    ldf = dd.from_pandas(df, npartitions=10)

    # rdf = await c.persist(ldf)
    # assert rdf.divisions == ldf.divisions

    # remote = c.compute(rdf)
    # result = await remote

    # assert_frame_equal(result, ldf.compute(scheduler="sync"))

    exprs = [
        lambda df: df.x.mean(),
        lambda df: df.y.std(),
        lambda df: df.assign(z=df.x + df.y).drop_duplicates(),
        lambda df: df.index,
        lambda df: df.x,
        lambda df: df.x.cumsum(),
        lambda df: df.groupby(["x", "y"]).count(),
        lambda df: df.loc[50:75],
    ]
    for f in exprs:
        local = await c.gather(c.compute(f(ldf)))
        # remote = c.compute(f(rdf))
        # remote = await remote
        # assert_equal(local, remote)


@ignore_single_machine_warning
@gen_cluster(
    client=True,
    # FIXME https://github.com/dask/distributed/issues/7566
    config={"distributed.scheduler.validate": False},
)
async def test_dask_array_collections(c, s, a, b):
    import dask.array as da

    x_dsk = {("x", i, j): np.random.random((3, 3)) for i in range(3) for j in range(2)}
    y_dsk = {("y", i, j): np.random.random((3, 3)) for i in range(2) for j in range(3)}
    x_futures = await c.scatter(x_dsk)
    y_futures = await c.scatter(y_dsk)

    dt = np.random.random(0).dtype
    x_local = da.Array(x_dsk, "x", ((3, 3, 3), (3, 3)), dt)
    y_local = da.Array(y_dsk, "y", ((3, 3), (3, 3, 3)), dt)

    x_remote = da.Array(x_futures, "x", ((3, 3, 3), (3, 3)), dt)
    y_remote = da.Array(y_futures, "y", ((3, 3), (3, 3, 3)), dt)

    exprs = [
        lambda x, y: x.T + y,
        lambda x, y: x.mean() + y.mean(),
        lambda x, y: x.dot(y).std(axis=0),
        lambda x, y: x - x.mean(axis=1)[:, None],
    ]

    for expr in exprs:
        z_local = expr(x_local, y_local)
        o_local = z_local.compute(scheduler="sync")

        z_remote = expr(x_remote, y_remote)
        o_remote = await c.compute(z_remote)
        np.testing.assert_equal(o_local, o_remote)


@pytest.mark.skipif(
    (
        parse_version(dask.__version__) < parse_version("2023.2.2")
        and parse_version(dask.__version__) >= parse_version("2023.2.1")
    ),
    reason="https://github.com/dask/dask/pull/10005",
)
@gen_cluster(client=True)
async def test_bag_groupby_tasks_default(c, s, a, b):
    b = db.range(100, npartitions=10)
    b2 = b.groupby(lambda x: x % 13)
    assert not any("partd" in k[0] for k in b2.dask)


@gen_cluster(client=True, Worker=Nanny)
async def test_bag_groupby_key_hashing(c, s, a, b):
    # https://github.com/dask/distributed/issues/4141
    dsk = {("x", 0): (range, 5), ("x", 1): (range, 5), ("x", 2): (range, 5)}
    grouped = db.Bag(dsk, "x", 3).groupby(lambda x: "even" if x % 2 == 0 else "odd")
    remote = c.compute(grouped)
    result = await remote
    assert len(result) == 2
    assert ("odd", [1, 3] * 3) in result
    assert ("even", [0, 2, 4] * 3) in result


@pytest.mark.parametrize("wait", [wait, lambda x: None])
def test_dataframe_set_index_sync(wait, client):
    partition_freq_unit = "ME" if PANDAS_GE_220 else "M"
    df = dask.datasets.timeseries(
        start="2000",
        end="2001",
        dtypes={"value": float, "name": str, "id": int},
        freq="2h",
        partition_freq=f"1{partition_freq_unit}",
        seed=1,
    )
    df = df.persist()
    wait(df)

    with dask.config.set({"dataframe.shuffle.method": "tasks"}):
        df2 = df.set_index("name")
    df2 = df2.persist()

    assert len(df2)


def make_time_dataframe():
    return pd.DataFrame(
        np.random.randn(30, 4),
        columns=list("ABCD"),
        index=pd.date_range("2000", periods=30, freq="B"),
    )


def test_loc_sync(client):
    df = make_time_dataframe()
    ddf = dd.from_pandas(df, npartitions=10)
    ddf.loc["2000-01-17":"2000-01-24"].compute()


def test_rolling_sync(client):
    df = make_time_dataframe()
    ddf = dd.from_pandas(df, npartitions=10)
    ddf.A.rolling(2).mean().compute()


@gen_cluster(client=True)
async def test_loc(c, s, a, b):
    df = make_time_dataframe()
    ddf = dd.from_pandas(df, npartitions=10)
    future = c.compute(ddf.loc["2000-01-17":"2000-01-24"])
    await future


@ignore_single_machine_warning
def test_dataframe_groupby_tasks(client):
    INCLUDE_GROUPS = {"include_groups": False} if PANDAS_GE_220 else {}
    df = make_time_dataframe()

    df["A"] = df.A // 0.1
    df["B"] = df.B // 0.1
    ddf = dd.from_pandas(df, npartitions=10)

    for ind in [lambda x: "A", lambda x: x.A]:
        a = df.groupby(ind(df)).apply(len, **INCLUDE_GROUPS)
        b = ddf.groupby(ind(ddf)).apply(len, meta=(None, int), **INCLUDE_GROUPS)
        assert_equal(a, b.compute().sort_index())
        assert not any("partd" in k[0] for k in b.dask)

        a = df.groupby(ind(df)).B.apply(len, **INCLUDE_GROUPS)
        b = ddf.groupby(ind(ddf)).B.apply(len, meta=("B", int), **INCLUDE_GROUPS)
        assert_equal(a, b.compute().sort_index())
        assert not any("partd" in k[0] for k in b.dask)

    with pytest.raises((NotImplementedError, ValueError)):
        ddf.groupby(ddf[["A", "B"]]).apply(len, meta=int, **INCLUDE_GROUPS)

    a = df.groupby(["A", "B"]).apply(len, **INCLUDE_GROUPS)
    b = ddf.groupby(["A", "B"]).apply(len, meta=(None, int), **INCLUDE_GROUPS)

    assert_equal(a, b.compute().sort_index())


@gen_cluster(client=True)
async def test_sparse_arrays(c, s, a, b):
    sparse = pytest.importorskip("sparse")
    da = pytest.importorskip("dask.array")

    x = da.random.random((100, 10), chunks=(10, 10))
    x[x < 0.95] = 0
    s = x.map_blocks(sparse.COO)
    future = c.compute(s.sum(axis=0)[:10])

    await future


@gen_cluster(client=True, nthreads=[("127.0.0.1", 1)])
async def test_delayed_none(c, s, w):
    x = dask.delayed(None)
    y = dask.delayed(123)
    [xx, yy] = c.compute([x, y])
    assert await xx is None
    assert await yy == 123


@pytest.mark.parametrize("typ", [tuple, list])
def test_tuple_futures_arg(client, typ):
    x = client.submit(
        make_time_dataframe,
    )
    df2 = client.submit(
        pd.concat,
        typ(
            [
                x,
            ]
        ),
    )
    dd.assert_eq(df2.result().iloc[:0], make_time_dataframe().iloc[:0])

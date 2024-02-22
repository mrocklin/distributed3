from __future__ import annotations

import numpy as np
import pytest
from distributed.client import Client

from distributed.utils_test import gen_cluster

dd = pytest.importorskip("dask_expr")

import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

from dask.dataframe.utils import assert_eq

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


@gen_cluster(client=True)
async def test_dask_expr(c, s, a, b):
    df = dd.datasets.timeseries()
    expr = df["id"].mean()
    expected = expr.compute(scheduler="sync")
    actual = await c.compute(expr)
    assert_eq(actual, expected)


@gen_cluster(client=True)
async def test_dask_expr_multiple_inputs(c, s, a, b):
    df = dd.datasets.timeseries()
    expr1 = df["id"].mean()
    expr2 = df["id"].sum()
    expected1 = expr1.compute(scheduler="sync")
    expected2 = expr2.compute(scheduler="sync")
    actual1, actual2 = await c.gather(c.compute((expr1, expr2)))
    assert_eq(actual1, expected1)
    assert_eq(actual2, expected2)

    actual2, actual1 = await c.gather(c.compute((expr2, expr1)))
    assert_eq(actual1, expected1)
    assert_eq(actual2, expected2)


@ignore_single_machine_warning
@gen_cluster()
async def test_dataframes(s, a, b):
    async with Client(s.address, asynchronous=True, set_as_default=False) as c:
        df = pd.DataFrame(
            {"x": np.random.random(1000), "y": np.random.random(1000)},
            index=np.arange(1000),
        )
        ldf = dd.from_pandas(df, npartitions=10)
        with c.as_current():
            rdf = await c.persist(ldf)
        assert rdf.divisions == ldf.divisions

        remote = c.compute(rdf)
        result = await remote

        assert_frame_equal(result, ldf.compute(scheduler="sync"))

        exprs = [
            lambda df: df.x.mean(),
            lambda df: df.y.std(),
            lambda df: df.index,
            lambda df: df.x,
            lambda df: df.x.cumsum(),
            # FIXME: This stuff is broken
            # lambda df: df.assign(z=df.x + df.y).drop_duplicates(),
            # lambda df: df.groupby(["x", "y"]).count(),
            # lambda df: df.loc[50:75],
        ]
        for f in exprs:
            print(f)
            # FIXME: Default shuffle method detection breaks here with a
            # defaultclient
            local = f(ldf).compute(scheduler="sync")
            remote = c.compute(f(rdf))
            remote = await remote
            assert_equal(local, remote)

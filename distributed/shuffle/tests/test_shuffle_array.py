from __future__ import annotations

import numpy as np
import pytest

import dask.array as da
from dask.core import flatten

from distributed.utils_test import gen_cluster


@pytest.fixture()
def arr():
    return np.arange(0, 24).reshape(8, 3).T.copy()


@pytest.fixture()
def darr(arr):
    return da.from_array(arr, chunks=((2, 1), (4, 4)))


@pytest.mark.parametrize(
    "indexer, chunks",
    [
        ([[1, 5, 6], [0, 2, 3, 4, 7]], (3, 5)),
        ([[1, 5, 6], [0, 3], [4, 2, 7]], (5, 3)),
        ([[1], [0, 6, 5, 3, 2, 4], [7]], (1, 6, 1)),
        ([[1, 5, 1, 5, 1, 5], [1, 6, 4, 2, 7]], (6, 5)),
    ],
)
@gen_cluster(client=True)
async def test_shuffle(c, s, *ws, arr, darr, indexer, chunks):
    result = darr.shuffle(indexer, axis=1)
    expected = arr[:, list(flatten(indexer))]
    x = await c.compute(result)
    np.testing.assert_array_equal(x, expected)
    assert result.chunks[0] == darr.chunks[0]
    assert result.chunks[1] == chunks


@gen_cluster(client=True)
async def test_shuffle_larger_array(c, s, *ws):
    arr = da.random.random((15, 15, 15), chunks=(5, 5, 5))
    indexer = np.arange(0, 15)
    np.random.shuffle(indexer)
    indexer = [indexer[0:6], indexer[6:8], indexer[8:9], indexer[9:]]
    indexer = list(map(list, indexer))
    take_indexer = list(flatten(indexer))

    x = await c.compute(arr.shuffle(indexer, axis=1))
    y = await c.compute(arr[..., take_indexer, :])
    np.testing.assert_array_equal(x, y)
    z = await c.compute(arr)
    z = z[..., take_indexer, :]
    np.testing.assert_array_equal(x, z)

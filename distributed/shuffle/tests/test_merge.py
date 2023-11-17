from __future__ import annotations

import contextlib
from unittest import mock

import pytest

from distributed.shuffle._core import id_from_key
from distributed.shuffle._merge import hash_join
from distributed.utils_test import gen_cluster

dd = pytest.importorskip("dask.dataframe")
import pandas as pd

import dask
from dask.dataframe._compat import PANDAS_GE_200, tm
from dask.dataframe.utils import assert_eq
from dask.utils_test import hlg_layer_topological

try:
    import pyarrow as pa
except ImportError:
    pa = None

pytestmark = pytest.mark.ci1


def list_eq(aa, bb):
    if isinstance(aa, dd.DataFrame):
        a = aa.compute(scheduler="sync")
    else:
        a = aa
    if isinstance(bb, dd.DataFrame):
        b = bb.compute(scheduler="sync")
    else:
        b = bb
    tm.assert_index_equal(a.columns, b.columns)

    if isinstance(a, pd.DataFrame):
        av = a.sort_values(list(a.columns)).values
        bv = b.sort_values(list(b.columns)).values
    else:
        av = a.sort_values().values
        bv = b.sort_values().values

    dd._compat.assert_numpy_array_equal(av, bv)


@gen_cluster(client=True)
async def test_minimal_version(c, s, a, b):
    no_pyarrow_ctx = (
        mock.patch.dict("sys.modules", {"pyarrow": None})
        if pa is not None
        else contextlib.nullcontext()
    )
    with no_pyarrow_ctx:
        A = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [1, 1, 2, 2, 3, 4]})
        a = dd.repartition(A, [0, 4, 5])

        B = pd.DataFrame({"y": [1, 3, 4, 4, 5, 6], "z": [6, 5, 4, 3, 2, 1]})
        b = dd.repartition(B, [0, 2, 5])

        with pytest.raises(ModuleNotFoundError, match="requires pyarrow"):
            await c.compute(dd.merge(a, b, left_on="x", right_on="z", shuffle="p2p"))


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
@gen_cluster(client=True)
async def test_basic_merge(c, s, a, b, how):
    A = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [1, 1, 2, 2, 3, 4]})
    a = dd.repartition(A, [0, 4, 5])

    B = pd.DataFrame({"y": [1, 3, 4, 4, 5, 6], "z": [6, 5, 4, 3, 2, 1]})
    b = dd.repartition(B, [0, 2, 5])

    joined = hash_join(a, "y", b, "y", how)

    assert not hlg_layer_topological(joined.dask, -1).is_materialized()
    result = await c.compute(joined)
    expected = pd.merge(A, B, how, "y")
    list_eq(result, expected)

    # Different columns and npartitions
    joined = hash_join(a, "x", b, "z", "outer", npartitions=3)
    assert not hlg_layer_topological(joined.dask, -1).is_materialized()
    assert joined.npartitions == 3

    result = await c.compute(joined)
    expected = pd.merge(A, B, "outer", None, "x", "z")

    list_eq(result, expected)

    assert (
        hash_join(a, "y", b, "y", "inner")._name
        == hash_join(a, "y", b, "y", "inner")._name
    )
    assert (
        hash_join(a, "y", b, "y", "inner")._name
        != hash_join(a, "y", b, "y", "outer")._name
    )


@gen_cluster(client=True)
async def test_merge_p2p_shuffle_reused_dataframe_with_different_parameters(c, s, a, b):
    pdf1 = pd.DataFrame({"a": range(100), "b": range(0, 200, 2)})
    pdf2 = pd.DataFrame({"x": range(200), "y": [1, 2, 3, 4] * 50})
    ddf1 = dd.from_pandas(pdf1, npartitions=5)
    ddf2 = dd.from_pandas(pdf2, npartitions=10)

    out = (
        ddf1.merge(ddf2, left_on="a", right_on="x", shuffle="p2p")
        # Vary the number of output partitions for the shuffles of dd2
        .repartition(20).merge(ddf2, left_on="b", right_on="x", shuffle="p2p")
    )
    # Generate unique shuffle IDs if the input frame is the same but parameters differ
    assert sum(id_from_key(k) is not None for k in out.dask) == 4
    result = await c.compute(out)
    expected = pdf1.merge(pdf2, left_on="a", right_on="x").merge(
        pdf2, left_on="b", right_on="x"
    )
    dd.assert_eq(result, expected, check_index=False)


@gen_cluster(client=True)
async def test_merge_p2p_shuffle_reused_dataframe_with_same_parameters(c, s, a, b):
    pdf1 = pd.DataFrame({"a": range(100), "b": range(0, 200, 2)})
    pdf2 = pd.DataFrame({"x": range(200), "y": [1, 2, 3, 4] * 50})
    ddf1 = dd.from_pandas(pdf1, npartitions=5)
    ddf2 = dd.from_pandas(pdf2, npartitions=10)

    # This performs two shuffles:
    #   * ddf1 is shuffled on `a`
    #   * ddf2 is shuffled on `x`
    ddf3 = ddf1.merge(
        ddf2,
        left_on="a",
        right_on="x",
        shuffle="p2p",
    )

    # This performs one shuffle:
    #   * ddf3 is shuffled on `b`
    # We can reuse the shuffle of dd2 on `x` from the previous merge.
    out = ddf2.merge(
        ddf3,
        left_on="x",
        right_on="b",
        shuffle="p2p",
    )
    # Generate the same shuffle IDs if the input frame is the same and all its parameters match
    assert sum(id_from_key(k) is not None for k in out.dask) == 3
    result = await c.compute(out)
    expected = pdf2.merge(
        pdf1.merge(pdf2, left_on="a", right_on="x"), left_on="x", right_on="b"
    )
    dd.assert_eq(result, expected, check_index=False)


@pytest.mark.parametrize("how", ["inner", "outer", "left", "right"])
@pytest.mark.parametrize("disk", [True, False])
@gen_cluster(client=True)
async def test_merge(c, s, a, b, how, disk):
    A = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [1, 1, 2, 2, 3, 4]})
    a = dd.repartition(A, [0, 4, 5])

    B = pd.DataFrame({"y": [1, 3, 4, 4, 5, 6], "z": [6, 5, 4, 3, 2, 1]})
    b = dd.repartition(B, [0, 2, 5])

    with dask.config.set({"distributed.p2p.storage.disk": disk}):
        joined = dd.merge(
            a, b, left_index=True, right_index=True, how=how, shuffle="p2p"
        )
    res = await c.compute(joined)
    assert_eq(
        res,
        pd.merge(A, B, left_index=True, right_index=True, how=how),
    )
    joined = dd.merge(a, b, on="y", how=how)
    result = await c.compute(joined)
    list_eq(result, pd.merge(A, B, on="y", how=how))
    assert all(d is None for d in joined.divisions)

    list_eq(
        await c.compute(
            dd.merge(a, b, left_on="x", right_on="z", how=how, shuffle="p2p")
        ),
        pd.merge(A, B, left_on="x", right_on="z", how=how),
    )
    list_eq(
        await c.compute(
            dd.merge(
                a,
                b,
                left_on="x",
                right_on="z",
                how=how,
                suffixes=("1", "2"),
                shuffle="p2p",
            )
        ),
        pd.merge(A, B, left_on="x", right_on="z", how=how, suffixes=("1", "2")),
    )

    list_eq(
        await c.compute(dd.merge(a, b, how=how, shuffle="p2p")),
        pd.merge(A, B, how=how),
    )
    list_eq(
        await c.compute(dd.merge(a, B, how=how, shuffle="p2p")),
        pd.merge(A, B, how=how),
    )
    list_eq(
        await c.compute(dd.merge(A, b, how=how, shuffle="p2p")),
        pd.merge(A, B, how=how),
    )
    # Note: No await since A and B are both pandas dataframes and this doesn't
    # actually submit anything
    list_eq(
        c.compute(dd.merge(A, B, how=how, shuffle="p2p")),
        pd.merge(A, B, how=how),
    )

    list_eq(
        await c.compute(
            dd.merge(a, b, left_index=True, right_index=True, how=how, shuffle="p2p")
        ),
        pd.merge(A, B, left_index=True, right_index=True, how=how),
    )
    list_eq(
        await c.compute(
            dd.merge(
                a,
                b,
                left_index=True,
                right_index=True,
                how=how,
                suffixes=("1", "2"),
                shuffle="p2p",
            )
        ),
        pd.merge(A, B, left_index=True, right_index=True, how=how, suffixes=("1", "2")),
    )

    list_eq(
        await c.compute(
            dd.merge(a, b, left_on="x", right_index=True, how=how, shuffle="p2p")
        ),
        pd.merge(A, B, left_on="x", right_index=True, how=how),
    )
    list_eq(
        await c.compute(
            dd.merge(
                a,
                b,
                left_on="x",
                right_index=True,
                how=how,
                suffixes=("1", "2"),
                shuffle="p2p",
            )
        ),
        pd.merge(A, B, left_on="x", right_index=True, how=how, suffixes=("1", "2")),
    )


@pytest.mark.slow
@gen_cluster(client=True, timeout=120)
@pytest.mark.parametrize("how", ["inner", "outer", "left", "right"])
async def test_merge_by_multiple_columns(c, s, a, b, how):
    # warnings here from pandas
    pdf1l = pd.DataFrame(
        {
            "a": list("abcdefghij"),
            "b": list("abcdefghij"),
            "c": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
        index=list("abcdefghij"),
    )
    pdf1r = pd.DataFrame(
        {
            "d": list("abcdefghij"),
            "e": list("abcdefghij"),
            "f": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        },
        index=list("abcdefghij"),
    )

    pdf2l = pd.DataFrame(
        {
            "a": list("abcdeabcde"),
            "b": list("abcabcabca"),
            "c": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
        index=list("abcdefghij"),
    )
    pdf2r = pd.DataFrame(
        {
            "d": list("edcbaedcba"),
            "e": list("aaabbbcccd"),
            "f": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        },
        index=list("fghijklmno"),
    )

    pdf3l = pd.DataFrame(
        {
            "a": list("aaaaaaaaaa"),
            "b": list("aaaaaaaaaa"),
            "c": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        },
        index=list("abcdefghij"),
    )
    pdf3r = pd.DataFrame(
        {
            "d": list("aaabbbccaa"),
            "e": list("abbbbbbbbb"),
            "f": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        },
        index=list("ABCDEFGHIJ"),
    )

    for pdl, pdr in [(pdf1l, pdf1r), (pdf2l, pdf2r), (pdf3l, pdf3r)]:
        for lpart, rpart in [(2, 2), (3, 2), (2, 3)]:
            ddl = dd.from_pandas(pdl, lpart)
            ddr = dd.from_pandas(pdr, rpart)

            expected = pdl.join(pdr, how=how)
            assert_eq(
                await c.compute(ddl.join(ddr, how=how, shuffle="p2p")),
                expected,
                # FIXME: There's an discrepancy with an empty index for
                # pandas=2.0 (xref https://github.com/dask/dask/issues/9957).
                # Temporarily avoid index check until the discrepancy is fixed.
                check_index=not (PANDAS_GE_200 and expected.index.empty),
            )

            expected = pdr.join(pdl, how=how)
            assert_eq(
                await c.compute(ddr.join(ddl, how=how, shuffle="p2p")),
                expected,
                # FIXME: There's an discrepancy with an empty index for
                # pandas=2.0 (xref https://github.com/dask/dask/issues/9957).
                # Temporarily avoid index check until the discrepancy is fixed.
                check_index=not (PANDAS_GE_200 and expected.index.empty),
            )

            expected = pd.merge(pdl, pdr, how=how, left_index=True, right_index=True)
            assert_eq(
                await c.compute(
                    dd.merge(
                        ddl,
                        ddr,
                        how=how,
                        left_index=True,
                        right_index=True,
                        shuffle="p2p",
                    )
                ),
                expected,
                # FIXME: There's an discrepancy with an empty index for
                # pandas=2.0 (xref https://github.com/dask/dask/issues/9957).
                # Temporarily avoid index check until the discrepancy is fixed.
                check_index=not (PANDAS_GE_200 and expected.index.empty),
            )

            expected = pd.merge(pdr, pdl, how=how, left_index=True, right_index=True)
            assert_eq(
                await c.compute(
                    dd.merge(
                        ddr,
                        ddl,
                        how=how,
                        left_index=True,
                        right_index=True,
                        shuffle="p2p",
                    )
                ),
                expected,
                # FIXME: There's an discrepancy with an empty index for
                # pandas=2.0 (xref https://github.com/dask/dask/issues/9957).
                # Temporarily avoid index check until the discrepancy is fixed.
                check_index=not (PANDAS_GE_200 and expected.index.empty),
            )

            # hash join
            list_eq(
                await c.compute(
                    dd.merge(
                        ddl,
                        ddr,
                        how=how,
                        left_on="a",
                        right_on="d",
                        shuffle="p2p",
                    )
                ),
                pd.merge(pdl, pdr, how=how, left_on="a", right_on="d"),
            )
            list_eq(
                await c.compute(
                    dd.merge(
                        ddl,
                        ddr,
                        how=how,
                        left_on="b",
                        right_on="e",
                        shuffle="p2p",
                    )
                ),
                pd.merge(pdl, pdr, how=how, left_on="b", right_on="e"),
            )

            list_eq(
                await c.compute(
                    dd.merge(
                        ddr,
                        ddl,
                        how=how,
                        left_on="d",
                        right_on="a",
                        shuffle="p2p",
                    )
                ),
                pd.merge(pdr, pdl, how=how, left_on="d", right_on="a"),
            )
            list_eq(
                await c.compute(
                    dd.merge(
                        ddr,
                        ddl,
                        how=how,
                        left_on="e",
                        right_on="b",
                        shuffle="p2p",
                    )
                ),
                pd.merge(pdr, pdl, how=how, left_on="e", right_on="b"),
            )

            list_eq(
                await c.compute(
                    dd.merge(
                        ddl,
                        ddr,
                        how=how,
                        left_on=["a", "b"],
                        right_on=["d", "e"],
                        shuffle="p2p",
                    )
                ),
                pd.merge(pdl, pdr, how=how, left_on=["a", "b"], right_on=["d", "e"]),
            )


@pytest.mark.parametrize("how", ["inner", "left", "right", "outer"])
@gen_cluster(client=True)
async def test_index_merge_p2p(c, s, a, b, how):
    pdf_left = pd.DataFrame({"a": [4, 2, 3] * 10, "b": 1}).set_index("a")
    pdf_right = pd.DataFrame({"a": [4, 2, 3] * 10, "c": 1})

    left = dd.from_pandas(pdf_left, npartitions=5, sort=False)
    right = dd.from_pandas(pdf_right, npartitions=6)

    assert_eq(
        await c.compute(
            left.merge(right, how=how, left_index=True, right_on="a", shuffle="p2p")
        ),
        pdf_left.merge(pdf_right, how=how, left_index=True, right_on="a"),
    )

    assert_eq(
        await c.compute(
            right.merge(left, how=how, right_index=True, left_on="a", shuffle="p2p")
        ),
        pdf_right.merge(pdf_left, how=how, right_index=True, left_on="a"),
    )

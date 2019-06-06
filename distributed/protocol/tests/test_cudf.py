import pytest

cudf = pytest.importorskip("cudf")

from distributed.protocol import serialize, deserialize
import dask.dataframe as dd


@pytest.mark.parametrize(
    "df",
    [
        cudf.Series([1, 2, 3]),
        cudf.Series([1, 2, None]),
        cudf.DataFrame({"x": [1, 2, 3], "y": [1.0, 2.0, 3.0]}),
        cudf.DataFrame({"x": [1, 2, 3], "s": ["a", "bb", "ccc"]}),
        cudf.DataFrame(
            {"x": [1, 2, None], "y": [1.0, 2.0, None], "s": ["a", "bb", None]}
        ),
    ],
)
def test_basic(df):
    header, frames = serialize(
        df, serializers=("cuda", "dask", "pickle"), on_error="raise"
    )
    assert header["serializer"] == "cuda"
    assert not any(isinstance(frame, (bytes, memoryview)) for frame in frames)

    df2 = deserialize(header, frames, deserializers=("cuda", "dask", "pickle"))
    dd.assert_eq(df, df2)

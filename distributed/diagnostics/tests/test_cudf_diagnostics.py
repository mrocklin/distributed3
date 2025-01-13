from __future__ import annotations

import sys

import pytest

import dask
from dask.distributed import worker

from distributed.utils_test import async_poll_for, gen_cluster

cudf = pytest.importorskip("cudf")


@pytest.fixture
def cudf_spill():
    """
    Configures cuDF options to enable spilling.

    Returns the settings to their original values after the test.
    """
    spill = cudf.get_option("spill")
    spill_stats = cudf.get_option("spill_stats")

    cudf.set_option("spill", True)
    cudf.set_option("spill_stats", 1)

    yield

    cudf.set_option("spill", spill)
    cudf.set_option("spill_stats", spill_stats)


def force_spill():
    from cudf.core.buffer.spill_manager import get_global_manager

    manager = get_global_manager()

    # Allocate a new dataframe and trigger spilling by setting a 1 byte limit
    df = cudf.DataFrame({"a": [1, 2, 3]})
    manager.spill_to_device_limit(1)

    # Get bytes spilled from GPU to CPU
    spill_totals, _ = get_global_manager().statistics.spill_totals[("gpu", "cpu")]
    return spill_totals


@gen_cluster(
    client=True,
    nthreads=[("127.0.0.1", 1)],
    # whether worker.cudf_metric is in DEFAULT_METRICS depends on the value
    # of distributed.diagnostics.cudf when distributed.worker is imported.
    worker_kwargs={
        "metrics": {**worker.DEFAULT_METRICS, "cudf": worker.cudf_metric},
    },
)
@pytest.mark.usefixtures("cudf_spill")
async def test_cudf_metrics(c, s, *workers):
    w = list(s.workers.values())[0]
    assert "cudf" in w.metrics
    assert w.metrics["cudf"]["cudf-spilled"] == 0

    spill_totals = (await c.run(force_spill, workers=[w.address]))[w.address]
    assert spill_totals > 0
    await async_poll_for(lambda: w.metrics["cudf"]["cudf-spilled"] > 0, timeout=2)
    assert w.metrics["cudf"]["cudf-spilled"] == spill_totals


def test_cudf_default_metrics(monkeypatch):
    with dask.config.set(**{"distributed.diagnostics.cudf": 1}):
        del sys.modules["distributed.worker"]
        import distributed.worker

    assert "cudf" in distributed.worker.DEFAULT_METRICS

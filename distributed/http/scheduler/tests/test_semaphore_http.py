from __future__ import annotations

import pytest

from distributed import Semaphore
from distributed.utils_test import fetch_metrics, gen_cluster


@gen_cluster(client=True, clean_kwargs={"threads": False})
async def test_prometheus(c, s, a, b):
    pytest.importorskip("prometheus_client")

    active_metrics = await fetch_metrics(s.http_server.port, "dask_semaphore_")

    expected_metrics = {
        "dask_semaphore_max_leases",
        "dask_semaphore_active_leases",
        "dask_semaphore_pending_leases",
        "dask_semaphore_acquire",
        "dask_semaphore_release",
        "dask_semaphore_average_pending_lease_time_seconds",
    }

    assert active_metrics.keys() == expected_metrics
    for v in active_metrics.values():  # Not yet any semaphore created
        assert v.samples == []

    sem = await Semaphore(name="test", max_leases=2)

    active_metrics = await fetch_metrics(s.http_server.port, "dask_semaphore_")
    assert active_metrics.keys() == expected_metrics
    # Assert values are set upon initialization
    for name, v in active_metrics.items():
        samples = v.samples
        assert len(samples) == 1
        sample = samples.pop()
        assert sample.labels["name"] == "test"
        if name == "dask_semaphore_max_leases":
            assert sample.value == 2
        else:
            assert sample.value == 0

    assert await sem.acquire()
    active_metrics = await fetch_metrics(s.http_server.port, "dask_semaphore_")
    assert active_metrics["dask_semaphore_max_leases"].samples[0].value == 2
    assert active_metrics["dask_semaphore_active_leases"].samples[0].value == 1
    assert (
        active_metrics["dask_semaphore_average_pending_lease_time_seconds"]
        .samples[0]
        .value
        > 0
    )
    assert active_metrics["dask_semaphore_acquire"].samples[0].value == 1
    assert active_metrics["dask_semaphore_release"].samples[0].value == 0
    assert active_metrics["dask_semaphore_pending_leases"].samples[0].value == 0

    assert await sem.release() is True
    active_metrics = await fetch_metrics(s.http_server.port, "dask_semaphore_")
    assert active_metrics["dask_semaphore_max_leases"].samples[0].value == 2
    assert active_metrics["dask_semaphore_active_leases"].samples[0].value == 0
    assert (
        active_metrics["dask_semaphore_average_pending_lease_time_seconds"]
        .samples[0]
        .value
        > 0
    )
    assert active_metrics["dask_semaphore_acquire"].samples[0].value == 1
    assert active_metrics["dask_semaphore_release"].samples[0].value == 1
    assert active_metrics["dask_semaphore_pending_leases"].samples[0].value == 0

    await sem.close()
    active_metrics = await fetch_metrics(s.http_server.port, "dask_semaphore_")
    assert active_metrics.keys() == expected_metrics
    for v in active_metrics.values():
        assert v.samples == []


@gen_cluster(
    client=True, clean_kwargs={"threads": False}, scheduler_kwargs={"extensions": {}}
)
async def test_prometheus_without_semaphore_extension(c, s, a, b):
    pytest.importorskip("prometheus_client")

    active_metrics = await fetch_metrics(s.http_server.port, "dask_semaphore_")
    assert not active_metrics

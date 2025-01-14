from __future__ import annotations

import pathlib
from unittest import mock

import pytest
from tornado.httpclient import AsyncHTTPClient

from distributed import Semaphore
from distributed.utils_test import fetch_metrics_sample_names, gen_cluster, inc


@gen_cluster(client=True)
async def test_scheduler(c, s, a, b):
    client = AsyncHTTPClient()
    response = await client.fetch(f"http://localhost:{s.http_server.port}/health")
    assert response.code == 200


@mock.patch("warnings.warn", return_value=None)
@gen_cluster(
    client=True,
    nthreads=[("", 1)],
    config={"distributed.admin.system-monitor.gil.enabled": True},
)
async def test_prometheus_api_doc(c, s, a, _):
    """Test that the Sphinx documentation of Prometheus endpoints matches the
    implementation.
    """
    pytest.importorskip("prometheus_client")

    root_dir = pathlib.Path(__file__).parent.parent.parent.parent
    if not (root_dir / ".git").exists():
        pytest.skip("Not running in a git checkout")

    documented = set()
    with open(root_dir / "docs" / "source" / "prometheus.rst") as fh:
        for row in fh:
            row = row.strip()
            if row.startswith("dask_"):
                documented.add(row)

    # Some metrics only appear if there are tasks on the cluster
    fut = c.submit(inc, 1)
    await fut

    a.data.evict()

    # Semaphore metrics only appear after semaphores are used
    sem = await Semaphore()
    await sem.acquire()
    await sem.release()

    # Note: built-in Prometheus metrics are undocumented
    scheduler_metrics = await fetch_metrics_sample_names(
        s.http_server.port, prefix="dask_"
    )
    worker_metrics = await fetch_metrics_sample_names(
        a.http_server.port, prefix="dask_"
    )

    try:
        import crick  # noqa: F401

        crick_metrics = set()  # Already in worker_metrics
    except ImportError:
        crick_metrics = {
            "dask_worker_tick_duration_median_seconds",
            "dask_worker_task_duration_median_seconds",
            "dask_worker_transfer_bandwidth_median_bytes",
        }

    try:
        import gilknocker  # noqa: F401

        gil_metrics = set()  # Already in worker_metrics
    except ImportError:
        gil_metrics = {
            "dask_scheduler_gil_contention_seconds_total",
            "dask_worker_gil_contention_seconds_total",
        }

    implemented = scheduler_metrics | worker_metrics | crick_metrics | gil_metrics
    assert documented == implemented

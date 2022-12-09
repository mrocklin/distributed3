from __future__ import annotations

import asyncio
import json
import re
from unittest import mock

import aiohttp
import pytest

pytest.importorskip("bokeh")

from tornado.escape import url_escape
from tornado.httpclient import AsyncHTTPClient, HTTPClientError

import dask.config
from dask.sizeof import sizeof

from distributed import Lock, Scheduler
from distributed.client import wait
from distributed.utils import is_valid_xml
from distributed.utils_test import (
    div,
    fetch_metrics,
    fetch_metrics_body,
    gen_cluster,
    gen_test,
    inc,
    lock_inc,
    slowinc,
)

DEFAULT_ROUTES = dask.config.get("distributed.scheduler.http.routes")


@gen_cluster(client=True)
async def test_connect(c, s, a, b):
    lock = Lock()
    async with lock:
        future = c.submit(lambda x: x + 1, 1)
        x = c.submit(lock_inc, 1, lock=lock, retries=5)
        await future
        http_client = AsyncHTTPClient()
        for suffix in [
            "info/main/workers.html",
            "info/worker/" + url_escape(a.address) + ".html",
            "info/task/" + url_escape(future.key) + ".html",
            "info/main/logs.html",
            "info/logs/" + url_escape(a.address) + ".html",
            "info/call-stack/" + url_escape(x.key) + ".html",
            "info/call-stacks/" + url_escape(a.address) + ".html",
            "json/counts.json",
            "json/identity.json",
            "json/index.html",
            "individual-plots.json",
            "sitemap.json",
        ]:
            response = await http_client.fetch(
                "http://localhost:%d/%s" % (s.http_server.port, suffix)
            )
            assert response.code == 200
            body = response.body.decode()
            if suffix.endswith(".json"):
                json.loads(body)
            else:
                assert is_valid_xml(body)
                assert not re.search("href=./", body)  # no absolute links


@gen_cluster(client=True, nthreads=[])
async def test_worker_404(c, s):
    http_client = AsyncHTTPClient()
    with pytest.raises(HTTPClientError) as err:
        await http_client.fetch(
            "http://localhost:%d/info/worker/unknown" % s.http_server.port
        )
    assert err.value.code == 404
    with pytest.raises(HTTPClientError) as err:
        await http_client.fetch(
            "http://localhost:%d/info/task/unknown" % s.http_server.port
        )
    assert err.value.code == 404


@gen_cluster(client=True, scheduler_kwargs={"http_prefix": "/foo", "dashboard": True})
async def test_prefix(c, s, a, b):
    http_client = AsyncHTTPClient()
    for suffix in ["foo/info/main/workers.html", "foo/json/index.html", "foo/system"]:
        response = await http_client.fetch(
            "http://localhost:%d/%s" % (s.http_server.port, suffix)
        )
        assert response.code == 200
        body = response.body.decode()
        if suffix.endswith(".json"):
            json.loads(body)
        else:
            assert is_valid_xml(body)


@gen_cluster(client=True, clean_kwargs={"threads": False})
async def test_prometheus(c, s, a, b):
    pytest.importorskip("prometheus_client")

    active_metrics = await fetch_metrics(s.http_server.port, "dask_scheduler_")

    expected_metrics = {
        "dask_scheduler_clients",
        "dask_scheduler_desired_workers",
        "dask_scheduler_workers",
        "dask_scheduler_tasks",
        "dask_scheduler_tasks_suspicious",
        "dask_scheduler_tasks_forgotten",
        "dask_scheduler_prefix_state",
        "dask_scheduler_tick_count",
        "dask_scheduler_tick_duration_maximum_seconds",
    }

    assert set(active_metrics.keys()) == expected_metrics
    assert active_metrics["dask_scheduler_clients"].samples[0].value == 1.0

    # request data twice since there once was a case where metrics got registered multiple times resulting in
    # prometheus_client errors
    await fetch_metrics(s.http_server.port, "dask_scheduler_")


@pytest.fixture
def prometheus_not_available():
    import sys

    with mock.patch.dict("sys.modules", {"prometheus_client": None}):
        sys.modules.pop("distributed.http.scheduler.prometheus", None)
        yield


@gen_test()
async def test_metrics_when_prometheus_client_not_installed(prometheus_not_available):
    async with Scheduler(dashboard_address=":0") as s:
        body = await fetch_metrics_body(s.http_server.port)
        assert "Prometheus metrics are not available" in body


@gen_cluster(client=True, clean_kwargs={"threads": False})
async def test_prometheus_collect_task_states(c, s, a, b):
    pytest.importorskip("prometheus_client")

    async def fetch_state_metrics():
        families = await fetch_metrics(s.http_server.port, prefix="dask_scheduler_")

        active_metrics = {
            sample.labels["state"]: sample.value
            for sample in families["dask_scheduler_tasks"].samples
        }
        forgotten_tasks = [
            sample.value
            for sample in families["dask_scheduler_tasks_forgotten"].samples
        ]
        return active_metrics, forgotten_tasks

    expected = {
        "memory",
        "released",
        "queued",
        "processing",
        "waiting",
        "no-worker",
        "erred",
    }

    # Ensure that we get full zero metrics for all states even though the
    # scheduler did nothing, yet
    assert not s.tasks
    active_metrics, forgotten_tasks = await fetch_state_metrics()
    assert active_metrics.keys() == expected
    assert sum(active_metrics.values()) == 0.0
    assert sum(forgotten_tasks) == 0.0

    # submit a task which should show up in the prometheus scraping
    future = c.submit(slowinc, 1, delay=0.5)
    while not any(future.key in w.state.tasks for w in [a, b]):
        await asyncio.sleep(0.001)

    active_metrics, forgotten_tasks = await fetch_state_metrics()
    assert active_metrics.keys() == expected
    assert sum(active_metrics.values()) == 1.0
    assert sum(forgotten_tasks) == 0.0

    res = await c.gather(future)
    assert res == 2

    future.release()

    while any(future.key in w.state.tasks for w in [a, b]):
        await asyncio.sleep(0.001)

    active_metrics, forgotten_tasks = await fetch_state_metrics()
    assert active_metrics.keys() == expected
    assert sum(active_metrics.values()) == 0.0
    assert sum(forgotten_tasks) == 0.0


@gen_cluster(client=True, clean_kwargs={"threads": False})
async def test_prometheus_collect_task_prefix_counts(c, s, a, b):
    pytest.importorskip("prometheus_client")
    from prometheus_client.parser import text_string_to_metric_families

    http_client = AsyncHTTPClient()

    async def fetch_metrics():
        port = s.http_server.port
        response = await http_client.fetch(f"http://localhost:{port}/metrics")
        txt = response.body.decode("utf8")
        families = {
            family.name: family for family in text_string_to_metric_families(txt)
        }

        prefix_state_counts = {
            (sample.labels["task_prefix_name"], sample.labels["state"]): sample.value
            for sample in families["dask_scheduler_prefix_state"].samples
        }

        return prefix_state_counts

    # do some compute and check the counts for each prefix and state
    futures = c.map(inc, range(10))
    await c.gather(futures)

    prefix_state_counts = await fetch_metrics()
    assert prefix_state_counts.get(("inc", "memory")) == 10
    assert prefix_state_counts.get(("inc", "erred"), 0) == 0

    f = c.submit(div, 1, 0)
    await wait(f)

    prefix_state_counts = await fetch_metrics()
    assert prefix_state_counts.get(("div", "erred")) == 1


@gen_cluster(client=True, clean_kwargs={"threads": False})
async def test_health(c, s, a, b):
    http_client = AsyncHTTPClient()

    response = await http_client.fetch(
        "http://localhost:%d/health" % s.http_server.port
    )
    assert response.code == 200
    assert response.headers["Content-Type"] == "text/plain"

    txt = response.body.decode("utf8")
    assert txt == "ok"


@gen_cluster()
async def test_sitemap(s, a, b):
    http_client = AsyncHTTPClient()

    response = await http_client.fetch(
        "http://localhost:%d/sitemap.json" % s.http_server.port
    )
    out = json.loads(response.body.decode())
    assert "paths" in out
    assert "/sitemap.json" in out["paths"]
    assert "/health" in out["paths"]
    assert "/statics/css/base.css" in out["paths"]


@gen_cluster(client=True)
async def test_task_page(c, s, a, b):
    future = c.submit(lambda x: x + 1, 1, workers=a.address)
    x = c.submit(inc, 1)
    await future
    http_client = AsyncHTTPClient()

    "info/task/" + url_escape(future.key) + ".html",
    response = await http_client.fetch(
        "http://localhost:%d/info/task/" % s.http_server.port
        + url_escape(future.key)
        + ".html"
    )
    assert response.code == 200
    body = response.body.decode()

    assert str(sizeof(1)) in body
    assert "int" in body
    assert a.address in body
    assert "memory" in body


@gen_cluster(
    client=True,
    scheduler_kwargs={"dashboard": True},
    config={
        "distributed.scheduler.dashboard.bokeh-application.allow_websocket_origin": [
            "good.invalid"
        ]
    },
)
async def test_allow_websocket_origin(c, s, a, b):
    from tornado.httpclient import HTTPRequest
    from tornado.websocket import websocket_connect

    url = (
        "ws://localhost:%d/status/ws?bokeh-protocol-version=1.0&bokeh-session-id=1"
        % s.http_server.port
    )
    with pytest.raises(HTTPClientError) as err:
        await websocket_connect(
            HTTPRequest(url, headers={"Origin": "http://evil.invalid"})
        )
    assert err.value.code == 403


@gen_cluster(client=True)
async def test_eventstream(c, s, a, b):
    from tornado.websocket import websocket_connect

    ws_client = await websocket_connect(
        "ws://localhost:%d/%s" % (s.http_server.port, "eventstream")
    )
    assert "websocket" in str(s.plugins).lower()
    ws_client.close()


def test_api_disabled_by_default():
    assert "distributed.http.scheduler.api" not in dask.config.get(
        "distributed.scheduler.http.routes"
    )


@gen_cluster(
    client=True,
    clean_kwargs={"threads": False},
    config={
        "distributed.scheduler.http.routes": DEFAULT_ROUTES
        + ["distributed.http.scheduler.api"]
    },
)
async def test_api(c, s, a, b):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "http://localhost:%d/api/v1" % s.http_server.port
        ) as resp:
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "text/plain"
            assert (await resp.text()) == "API V1"


@gen_cluster(
    client=True,
    clean_kwargs={"threads": False},
    config={
        "distributed.scheduler.http.routes": DEFAULT_ROUTES
        + ["distributed.http.scheduler.api"]
    },
)
async def test_retire_workers(c, s, a, b):
    async with aiohttp.ClientSession() as session:
        params = {"workers": [a.address, b.address]}
        async with session.post(
            "http://localhost:%d/api/v1/retire_workers" % s.http_server.port,
            json=params,
        ) as resp:
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "application/json"
            retired_workers_info = json.loads(await resp.text())
            assert len(retired_workers_info) == 2


@gen_cluster(
    client=True,
    clean_kwargs={"threads": False},
    config={
        "distributed.scheduler.http.routes": DEFAULT_ROUTES
        + ["distributed.http.scheduler.api"]
    },
)
async def test_get_workers(c, s, a, b):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "http://localhost:%d/api/v1/get_workers" % s.http_server.port
        ) as resp:
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "application/json"
            workers_info = json.loads(await resp.text())["workers"]
            workers_address = [worker.get("address") for worker in workers_info]
            assert set(workers_address) == {a.address, b.address}


@gen_cluster(
    client=True,
    clean_kwargs={"threads": False},
    config={
        "distributed.scheduler.http.routes": DEFAULT_ROUTES
        + ["distributed.http.scheduler.api"]
    },
)
async def test_adaptive_target(c, s, a, b):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "http://localhost:%d/api/v1/adaptive_target" % s.http_server.port
        ) as resp:
            assert resp.status == 200
            assert resp.headers["Content-Type"] == "application/json"
            num_workers = json.loads(await resp.text())["workers"]
            assert num_workers == 0

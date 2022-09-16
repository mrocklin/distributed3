from __future__ import annotations

import re
import sys

import pytest
import tornado

from distributed import Client, Worker
from distributed.utils_test import gen_cluster
from distributed.versions import error_message, get_versions

# if one of the nodes reports this version, there's a mismatch
mismatched_version = get_versions()
mismatched_version["packages"]["distributed"] = "0.0.0.dev0"

# if no key is available for one package, we assume it's MISSING
missing_version = get_versions()
del missing_version["packages"]["distributed"]

# if a node doesn't report any version info, we treat them as UNKNOWN
# the happens if the node is pre-32cb96e, i.e. <=2.9.1
unknown_version = None


@pytest.fixture
def kwargs_matching():
    return dict(
        scheduler=get_versions(),
        workers={f"worker-{i}": get_versions() for i in range(3)},
        source=get_versions(),
    )


def test_versions_match(kwargs_matching):
    assert error_message(**kwargs_matching)["warning"] == ""


@pytest.fixture(params=["source", "scheduler", "worker-1"])
def node(request):
    """Node affected by version mismatch."""
    return request.param


@pytest.fixture(params=["MISMATCHED", "MISSING", "KEY_ERROR", "NONE"])
def effect(request):
    """Specify type of mismatch."""
    return request.param


@pytest.fixture
def kwargs_not_matching(kwargs_matching, node, effect):
    affected_version = {
        "MISMATCHED": mismatched_version,
        "MISSING": missing_version,
        "KEY_ERROR": {},
        "NONE": unknown_version,
    }[effect]
    kwargs = kwargs_matching
    if node in kwargs["workers"]:
        kwargs["workers"][node] = affected_version
    else:
        kwargs[node] = affected_version
    return kwargs


@pytest.fixture
def pattern(effect):
    """String to match in the right column."""
    return {
        "MISMATCHED": "0.0.0.dev0",
        "MISSING": "MISSING",
        "KEY_ERROR": "UNKNOWN",
        "NONE": "UNKNOWN",
    }[effect]


def test_version_mismatch(node, effect, kwargs_not_matching, pattern):
    column_matching = {"source": 1, "scheduler": 2, "workers": 3}
    msg = error_message(**kwargs_not_matching)
    i = column_matching.get(node, 3)
    assert "Mismatched versions found" in msg["warning"]
    assert "distributed" in msg["warning"]
    assert (
        pattern
        in re.search(
            r"distributed\s+(?:(?:\|[^|\r\n]*)+\|(?:\r?\n|\r)?)+", msg["warning"]
        )
        .group(0)
        .split("|")[i]
        .strip()
    )


def test_scheduler_mismatched_irrelevant_package(kwargs_matching):
    """An irrelevant package on the scheduler can have any version."""
    kwargs_matching["scheduler"]["packages"]["numpy"] = "0.0.0"
    assert "numpy" in kwargs_matching["source"]["packages"]

    assert error_message(**kwargs_matching)["warning"] == ""


def test_scheduler_additional_irrelevant_package(kwargs_matching):
    """An irrelevant package on the scheduler does not need to be present elsewhere."""
    kwargs_matching["scheduler"]["packages"]["pyspark"] = "0.0.0"

    assert error_message(**kwargs_matching)["warning"] == ""


def test_python_mismatch(kwargs_matching):
    kwargs_matching["source"]["packages"]["python"] = "0.0.0"
    msg = error_message(**kwargs_matching)
    assert "Mismatched versions found" in msg["error"]
    assert "python" in msg["error"]
    assert (
        "0.0.0"
        in re.search(r"python\s+(?:(?:\|[^|\r\n]*)+\|(?:\r?\n|\r)?)+", msg["error"])
        .group(0)
        .split("|")[1]
        .strip()
    )


@gen_cluster()
async def test_version_warning_in_cluster(s, a, b):
    s.workers[a.address].versions["packages"]["dask"] = "0.0.0"

    with pytest.warns(Warning) as record:
        async with Client(s.address, asynchronous=True):
            pass

    assert record
    assert any("dask" in str(r.message) for r in record)
    assert any("0.0.0" in str(r.message) for r in record)

    async with Worker(s.address) as w:
        assert any(w.id in line.message for line in w.logs)
        assert any("Workers" in line.message for line in w.logs)
        assert any("dask" in line.message for line in w.logs)
        assert any("0.0.0" in line.message in line.message for line in w.logs)


def test_python_version():
    required = get_versions()["packages"]
    assert required["python"] == ".".join(map(str, sys.version_info))


def test_version_custom_pkgs():
    out = get_versions(
        [
            # Use custom function
            ("distributed", lambda mod: "123"),
            # Use version_of_package
            "notexist",
            ("pytest", None),  # has __version__
            "tornado",  # has version
            "math",  # has nothing
        ]
    )["packages"]
    assert out["distributed"] == "123"
    assert out["notexist"] is None
    assert out["pytest"] == pytest.__version__
    assert out["tornado"] == tornado.version
    assert out["math"] is None

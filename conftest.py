from __future__ import annotations

import pytest

# Uncomment to enable more logging and checks
# (https://docs.python.org/3/library/asyncio-dev.html)
# Note this makes things slower and might consume much memory.
# os.environ["PYTHONASYNCIODEBUG"] = "1"

try:
    import faulthandler
except ImportError:
    pass
else:
    try:
        faulthandler.enable()
    except Exception:
        pass

# Make all fixtures available
from distributed.utils_test import *  # noqa


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")


def pytest_collection_modifyitems(config, items):
    # https://pytest.org/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
    if skip_slow := not config.getoption("--runslow"):
        # --runslow not given in cli: skip slow tests
        skip_slow_marker = pytest.mark.skip(reason="need --runslow option to run")

    for item in items:
        if skip_slow and "slow" in item.keywords:
            item.add_marker(skip_slow_marker)

        if "ws" in item.fixturenames:
            item.add_marker(pytest.mark.workerstate)


pytest_plugins = ["distributed.pytest_resourceleaks"]

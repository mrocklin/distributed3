# https://pytest.org/latest/example/simple.html#control-skipping-of-tests-according-to-command-line-option
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

from distributed.config import ensure_logging_configured

ensure_logging_configured()

# Make all fixtures available
from distributed.utils_test import *  # noqa


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

        if "ws" in item.fixturenames:
            item.add_marker(pytest.mark.workerstate)


pytest_plugins = ["distributed.pytest_resourceleaks"]

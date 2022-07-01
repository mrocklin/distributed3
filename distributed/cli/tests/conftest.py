import pytest

from distributed.utils import open_port


@pytest.fixture(scope="module")
def default_dashboard_port():
    return open_port()

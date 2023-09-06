from __future__ import annotations

import pytest

from distributed import Nanny
from distributed.utils_test import gen_cluster


@gen_cluster(client=True, nthreads=[("", 1)], Worker=Nanny)
async def test_duck_typed_plugin_is_not_allowed(c, s, a):
    class DuckPlugin:
        def setup(self, nanny):
            pass

        def teardown(self, nanny):
            pass

    n_existing_plugins = len(a.plugins)
    with pytest.raises(TypeError, match="subclass `NannyPlugin`"):
        await c.register_worker_plugin(DuckPlugin(), nanny=True)
    assert len(a.plugins) == n_existing_plugins

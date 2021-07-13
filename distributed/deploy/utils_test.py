import pytest

from ..client import Client


class ClusterTest:
    Cluster = None
    kwargs = {}

    def setUp(self):
        self.cluster = self.Cluster(2, scheduler_port=0, **self.kwargs)
        self.client = Client(self.cluster.scheduler_address)

    def tearDown(self):
        self.client.close()
        self.cluster.close()

    @pytest.mark.xfail()
    def test_cores(self):
        self.client.scheduler_info()
        assert len(self.client.nthreads()) == 2

    def test_submit(self):
        future = self.client.submit(lambda x: x + 1, 1)
        assert future.result() == 2

    def test_context_manager(self):
        kwargs = self.kwargs.copy()
        kwargs.pop("dashboard_address")
        with self.Cluster(dashboard_address=":54321", **kwargs) as c:
            with Client(c) as e:
                assert e.nthreads()

    def test_no_workers(self):
        kwargs = self.kwargs.copy()
        kwargs.pop("dashboard_address")
        with self.Cluster(0, dashboard_address=":54321", scheduler_port=0, **kwargs):
            pass

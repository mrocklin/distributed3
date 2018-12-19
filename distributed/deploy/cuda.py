import os

from tornado import gen

from .local import LocalCluster


def get_n_gpus():
    return len(os.popen("nvidia-smi -L").read().strip().split("\n"))


class LocalCUDACluster(LocalCluster):
    def __init__(
        self, n_workers=get_n_gpus(), threads_per_worker=1, processes=True, **kwargs
    ):
        if not processes:
            raise NotImplementedError("Need processes to segment GPUs")
        if n_workers > get_n_gpus():
            raise ValueError("Can not specify more processes than GPUs")
        LocalCluster.__init__(
            self, n_workers=n_workers, threads_per_worker=threads_per_worker, **kwargs
        )

    @gen.coroutine
    def _start(self, ip=None, n_workers=0):
        """
        Start all cluster services.
        """
        if self.status == "running":
            return
        if (ip is None) and (not self.scheduler_port) and (not self.processes):
            # Use inproc transport for optimization
            scheduler_address = "inproc://"
        elif ip is not None and ip.startswith("tls://"):
            scheduler_address = "%s:%d" % (ip, self.scheduler_port)
        else:
            if ip is None:
                ip = "127.0.0.1"
            scheduler_address = (ip, self.scheduler_port)
        self.scheduler.start(scheduler_address)

        yield [
            self._start_worker(
                **self.worker_kwargs, env={"CUDA_VISIBLE_DEVICES": str(i)}
            )
            for i in range(n_workers)
        ]

        self.status = "running"

        raise gen.Return(self)

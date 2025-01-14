from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
from multiprocessing import cpu_count
from time import sleep

import pytest
from click.testing import CliRunner

from dask.utils import tmpfile

from distributed import Client
from distributed.cli.dask_worker import _apportion_ports, main
from distributed.compatibility import LINUX, WINDOWS
from distributed.deploy.utils import nprocesses_nthreads
from distributed.metrics import time
from distributed.utils import get_ip, open_port
from distributed.utils_test import (
    gen_cluster,
    inc,
    popen,
    requires_ipv6,
    wait_for_log_line,
)


@pytest.mark.parametrize(
    # args: (worker_port, nanny_port, n_workers, nanny)
    # Passing *args tuple instead of single args is to improve readability with black
    "args,expect",
    [
        # Single worker
        (
            (None, None, 1, False),
            [{"port": None}],
        ),
        (
            (None, None, 1, True),
            [{"port": None, "worker_port": None}],
        ),
        (("123", None, 1, False), [{"port": 123}]),
        (
            ("123", None, 1, True),
            [{"port": None, "worker_port": 123}],
        ),
        (
            (None, "456", 1, True),
            [{"port": 456, "worker_port": None}],
        ),
        (
            ("123", "456", 1, True),
            [{"port": 456, "worker_port": 123}],
        ),
        # port=None or 0 and multiple workers
        (
            (None, None, 2, False),
            [
                {"port": None},
                {"port": None},
            ],
        ),
        (
            (None, None, 2, True),
            [
                {"port": None, "worker_port": None},
                {"port": None, "worker_port": None},
            ],
        ),
        (
            (0, "0", 2, True),
            [
                {"port": 0, "worker_port": 0},
                {"port": 0, "worker_port": 0},
            ],
        ),
        (
            ("0", None, 2, True),
            [
                {"port": None, "worker_port": 0},
                {"port": None, "worker_port": 0},
            ],
        ),
        # port ranges
        (
            ("100:103", None, 1, False),
            [{"port": [100, 101, 102, 103]}],
        ),
        (
            ("100:103", None, 2, False),
            [
                {"port": [100, 102]},  # Round robin apportion
                {"port": [101, 103]},
            ],
        ),
        # port range is not an exact multiple of n_workers
        (
            ("100:107", None, 3, False),
            [
                {"port": [100, 103, 106]},
                {"port": [101, 104, 107]},
                {"port": [102, 105]},
            ],
        ),
        (
            ("100:103", None, 2, True),
            [
                {"port": None, "worker_port": [100, 102]},
                {"port": None, "worker_port": [101, 103]},
            ],
        ),
        (
            (None, "110:113", 2, True),
            [
                {"port": [110, 112], "worker_port": None},
                {"port": [111, 113], "worker_port": None},
            ],
        ),
        # port ranges have different length between nannies and workers
        (
            ("100:103", "110:114", 2, True),
            [
                {"port": [110, 112, 114], "worker_port": [100, 102]},
                {"port": [111, 113], "worker_port": [101, 103]},
            ],
        ),
        # identical port ranges
        (
            ("100:103", "100:103", 2, True),
            [
                {"port": 101, "worker_port": 100},
                {"port": 103, "worker_port": 102},
            ],
        ),
        # overlapping port ranges
        (
            ("100:105", "104:106", 2, True),
            [
                {"port": [104, 106], "worker_port": [100, 102]},
                {"port": 105, "worker_port": [101, 103]},
            ],
        ),
    ],
)
def test_apportion_ports(args, expect):
    assert _apportion_ports(*args) == expect


def test_apportion_ports_bad():
    with pytest.raises(ValueError, match="Not enough ports in range"):
        _apportion_ports("100:102", None, 4, False)
    with pytest.raises(ValueError, match="Not enough ports in range"):
        _apportion_ports(None, "100:102", 4, False)
    with pytest.raises(ValueError, match="Not enough ports in range"):
        _apportion_ports("100:102", "100:102", 3, True)
    with pytest.raises(ValueError, match="Not enough ports in range"):
        _apportion_ports("100:102", "102:104", 3, True)
    with pytest.raises(ValueError, match="port_stop must be greater than port_start"):
        _apportion_ports("102:100", None, 4, False)
    with pytest.raises(ValueError, match="invalid literal for int"):
        _apportion_ports("foo", None, 1, False)
    with pytest.raises(ValueError, match="too many values to unpack"):
        _apportion_ports("100:101:102", None, 1, False)


@pytest.mark.slow
@gen_cluster(client=True, nthreads=[])
async def test_nanny_worker_ports(c, s):
    worker_port = open_port()
    nanny_port = open_port()
    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "worker",
            s.address,
            "--host",
            "127.0.0.1",
            f"--worker-port={worker_port}",
            f"--nanny-port={nanny_port}",
            "--no-dashboard",
        ]
    ):
        await c.wait_for_workers(1)
        d = await c.scheduler.identity()
        assert (
            d["workers"][f"tcp://127.0.0.1:{worker_port}"]["nanny"]
            == f"tcp://127.0.0.1:{nanny_port}"
        )


@pytest.mark.skip(reason="flaky, probably due to popen")
@pytest.mark.slow
@pytest.mark.flaky(
    LINUX and sys.version_info[:2] == (3, 9),
    reason="Race condition in Nanny.process.start? "
    "See https://github.com/dask/distributed/issues/6045",
)
@gen_cluster(client=True, nthreads=[])
async def test_nanny_worker_port_range(c, s):
    async def assert_ports(min_: int, max_: int, nanny: bool) -> None:
        port_ranges = await c.run(
            lambda dask_worker: dask_worker._start_port, nanny=nanny
        )

        for a in port_ranges.values():
            assert isinstance(a, list)
            assert len(a) in (333, 334)
            assert all(min_ <= i <= max_ for i in a)

            # Test no overlap
            for b in port_ranges.values():
                assert a is b or not set(a) & set(b)

        ports = await c.run(lambda dask_worker: dask_worker.port, nanny=nanny)
        assert all(min_ <= p <= max_ for p in ports.values())
        for addr, range in port_ranges.items():
            assert ports[addr] in range

    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "worker",
            s.address,
            "--nworkers",
            "3",
            "--host",
            "127.0.0.1",
            "--worker-port",
            "10000:11000",
            "--nanny-port",
            "11000:12000",
            "--no-dashboard",
        ]
    ):
        await c.wait_for_workers(3)
        await assert_ports(10000, 10999, nanny=False)
        await assert_ports(11000, 12000, nanny=True)


@gen_cluster(nthreads=[])
async def test_nanny_worker_port_range_too_many_workers_raises(s):
    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "worker",
            s.address,
            "--nworkers",
            "3",
            "--host",
            "127.0.0.1",
            "--worker-port",
            "9684:9685",
            "--nanny-port",
            "9686:9687",
            "--no-dashboard",
        ],
        capture_output=True,
    ) as worker:
        wait_for_log_line(b"Not enough ports in range", worker.stdout, max_lines=100)


@pytest.mark.slow
@gen_cluster(client=True, nthreads=[])
async def test_memory_limit(c, s):
    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "worker",
            s.address,
            "--memory-limit",
            "2e3MB",
            "--no-dashboard",
        ]
    ):
        await c.wait_for_workers(1)
        info = c.scheduler_info()
        (d,) = info["workers"].values()
        assert isinstance(d["memory_limit"], int)
        assert d["memory_limit"] == 2e9


@gen_cluster(client=True, nthreads=[])
async def test_no_nanny(c, s):
    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "worker",
            s.address,
            "--no-nanny",
            "--no-dashboard",
        ]
    ):
        await c.wait_for_workers(1)


@pytest.mark.slow
@gen_cluster(client=True, nthreads=[])
async def test_resources(c, s):
    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "worker",
            s.address,
            "--no-dashboard",
            "--resources",
            "A=1 B=2,C=3",
        ]
    ):
        await c.wait_for_workers(1)
        info = c.scheduler_info()
        (d,) = info["workers"].values()
        assert d["resources"] == {"A": 1, "B": 2, "C": 3}


@pytest.mark.slow
@pytest.mark.parametrize("nanny", ["--nanny", "--no-nanny"])
@gen_cluster(client=True, nthreads=[])
async def test_local_directory(c, s, nanny, tmp_path):
    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "worker",
            s.address,
            nanny,
            "--no-dashboard",
            "--local-directory",
            str(tmp_path),
        ]
    ):
        await c.wait_for_workers(1)
        info = c.scheduler_info()
        (d,) = info["workers"].values()
        assert d["local_directory"].startswith(str(tmp_path))


@pytest.mark.slow
@pytest.mark.parametrize("nanny", ["--nanny", "--no-nanny"])
def test_scheduler_file(loop, nanny):
    with (
        tmpfile() as fn,
        popen(
            [
                sys.executable,
                "-m",
                "dask",
                "scheduler",
                "--no-dashboard",
                "--scheduler-file",
                fn,
            ]
        ),
        popen(
            [
                sys.executable,
                "-m",
                "dask",
                "worker",
                "--scheduler-file",
                fn,
                nanny,
                "--no-dashboard",
            ]
        ),
        Client(scheduler_file=fn, loop=loop) as c,
    ):
        start = time()
        while not c.scheduler_info()["workers"]:
            sleep(0.1)
            assert time() < start + 10


@pytest.mark.slow
def test_scheduler_address_env(loop, monkeypatch):
    port = open_port()
    monkeypatch.setenv("DASK_SCHEDULER_ADDRESS", f"tcp://127.0.0.1:{port}")
    # The env var is only picked up by the dask worker command
    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "scheduler",
            "--no-dashboard",
            "--port",
            str(port),
        ]
    ):
        with popen([sys.executable, "-m", "dask", "worker", "--no-dashboard"]):
            with Client(os.environ["DASK_SCHEDULER_ADDRESS"], loop=loop) as c:
                start = time()
                while not c.scheduler_info()["workers"]:
                    sleep(0.1)
                    assert time() < start + 10


@gen_cluster(nthreads=[])
async def test_nworkers_requires_nanny(s):
    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "worker",
            s.address,
            "--nworkers=2",
            "--no-nanny",
        ],
        capture_output=True,
    ) as worker:
        wait_for_log_line(b"Failed to launch worker", worker.stdout, max_lines=15)


@pytest.mark.slow
@gen_cluster(client=True, nthreads=[])
async def test_nworkers_negative(c, s):
    with popen([sys.executable, "-m", "dask", "worker", s.address, "--nworkers=-1"]):
        await c.wait_for_workers(cpu_count())


@pytest.mark.slow
@gen_cluster(client=True, nthreads=[])
async def test_nworkers_auto(c, s):
    with popen([sys.executable, "-m", "dask", "worker", s.address, "--nworkers=auto"]):
        procs, _ = nprocesses_nthreads()
        await c.wait_for_workers(procs)


@pytest.mark.slow
@gen_cluster(client=True, nthreads=[])
async def test_nworkers_expands_name(c, s):
    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "worker",
            s.address,
            "--nworkers",
            "2",
            "--name",
            "0",
        ]
    ):
        with popen(
            [sys.executable, "-m", "dask", "worker", s.address, "--nworkers", "2"]
        ):
            await c.wait_for_workers(4)
            info = c.scheduler_info()

    names = [d["name"] for d in info["workers"].values()]
    assert len(names) == len(set(names)) == 4
    zeros = [n for n in names if n.startswith("0-")]
    assert len(zeros) == 2


@pytest.mark.slow
@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
@pytest.mark.parametrize("nanny", ["--nanny", "--no-nanny"])
@pytest.mark.parametrize("listen_address", ["tcp://0.0.0.0:", "tcp://127.0.0.2:"])
@gen_cluster(client=True, nthreads=[])
async def test_contact_listen_address(c, s, nanny, listen_address):
    port = open_port()
    listen_address += str(port)
    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "worker",
            s.address,
            nanny,
            "--no-dashboard",
            "--contact-address",
            f"tcp://127.0.0.2:{port}",
            "--listen-address",
            listen_address,
        ]
    ):
        await c.wait_for_workers(1)
        info = c.scheduler_info()
        assert info["workers"].keys() == {f"tcp://127.0.0.2:{port}"}

        # roundtrip works
        assert await c.submit(lambda x: x + 1, 10) == 11

        def func(dask_worker):
            return dask_worker.listener.listen_address

        assert await c.run(func) == {f"tcp://127.0.0.2:{port}": listen_address}


@pytest.mark.slow
@requires_ipv6
@pytest.mark.parametrize("nanny", ["--nanny", "--no-nanny"])
@pytest.mark.parametrize("listen_address", ["tcp://:", "tcp://[::1]:"])
@gen_cluster(client=True, nthreads=[])
async def test_listen_address_ipv6(c, s, nanny, listen_address):
    port = open_port()
    listen_address += str(port)
    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "worker",
            s.address,
            nanny,
            "--no-dashboard",
            "--listen-address",
            listen_address,
        ]
    ):
        # IPv4 used by default for name of global listener; IPv6 used by default when
        # listening only on IPv6.
        bind_all = "[::1]" not in listen_address
        expected_ip = "127.0.0.1" if bind_all else "[::1]"
        expected_name = f"tcp://{expected_ip}:{port}"
        expected_listen = f"tcp://0.0.0.0:{port}" if bind_all else listen_address

        await c.wait_for_workers(1)
        info = c.scheduler_info()
        assert info["workers"].keys() == {expected_name}
        assert await c.submit(lambda x: x + 1, 10) == 11

        def func(dask_worker):
            return dask_worker.listener.listen_address

        assert await c.run(func) == {expected_name: expected_listen}


@pytest.mark.slow
@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
@pytest.mark.parametrize("nanny", ["--nanny", "--no-nanny"])
@pytest.mark.parametrize("host", ["127.0.0.2", "0.0.0.0"])
@gen_cluster(client=True, nthreads=[])
async def test_respect_host_listen_address(c, s, nanny, host):
    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "worker",
            s.address,
            nanny,
            "--no-dashboard",
            "--host",
            host,
        ]
    ):
        await c.wait_for_workers(1)

        # roundtrip works
        assert await c.submit(lambda x: x + 1, 10) == 11

        def func(dask_worker):
            return dask_worker.listener.listen_address

        listen_addresses = await c.run(func)
        assert all(host in v for v in listen_addresses.values())


@pytest.mark.slow
def test_dashboard_non_standard_ports():
    pytest.importorskip("bokeh")
    requests = pytest.importorskip("requests")

    s_host = "127.0.0.1"
    # use internal ip instead of localhost ip to verify GlobalProxyHandler will update
    # to allow internal host ip of a worker.
    w_host = get_ip()
    s_port = open_port(s_host)
    s_dashboard_port = open_port(s_host)
    w_dashboard_port = open_port(w_host)
    s_cmd = f"dask scheduler --host {s_host} --port {s_port} --dashboard-address :{s_dashboard_port}"
    w_cmd = f"dask worker {s_host}:{s_port} --dashboard-address :{w_dashboard_port} --host {w_host}"

    with popen(s_cmd.split()), popen(w_cmd.split()):
        with Client(f"{s_host}:{s_port}") as c:
            c.wait_for_workers(1)

        response = requests.get(f"http://{s_host}:{w_dashboard_port}/status")
        response.raise_for_status()

        # TEST PROXYING WORKS
        response = requests.get(
            f"http://{s_host}:{s_dashboard_port}/proxy/{w_dashboard_port}/{w_host}/status"
        )
        response.raise_for_status()

    with pytest.raises(requests.ConnectionError):
        requests.get("http://localhost:4833/status/")


def test_version_option():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0


@pytest.mark.slow
@pytest.mark.parametrize("no_nanny", [True, False])
def test_worker_timeout(no_nanny):
    runner = CliRunner()
    args = ["192.168.1.100:7777", "--death-timeout=1"]
    if no_nanny:
        args.append("--no-nanny")
    result = runner.invoke(main, args)
    assert result.exit_code == 1


@pytest.mark.slow
@gen_cluster(nthreads=[])
async def test_integer_names(s):
    with popen([sys.executable, "-m", "dask", "worker", s.address, "--name", "123"]):
        while not s.workers:
            await asyncio.sleep(0.01)
        [ws] = s.workers.values()
        assert ws.name == 123


@pytest.mark.slow
@pytest.mark.parametrize("nanny", ["--nanny", "--no-nanny"])
@gen_cluster(client=True, nthreads=[])
async def test_worker_class(c, s, tmp_path, nanny):
    # Create module with custom worker class
    WORKER_CLASS_TEXT = """
from distributed.worker import Worker

class MyWorker(Worker):
    pass
"""
    tmpdir = str(tmp_path)
    tmpfile = str(tmp_path / "myworker.py")
    with open(tmpfile, "w") as f:
        f.write(WORKER_CLASS_TEXT)

    # Put module on PYTHONPATH
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = tmpdir + ":" + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = tmpdir

    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "worker",
            s.address,
            nanny,
            "--worker-class",
            "myworker.MyWorker",
        ],
        env=env,
    ):
        await c.wait_for_workers(1)

        def worker_type(dask_worker):
            return type(dask_worker).__name__

        worker_types = await c.run(worker_type)
        assert all(name == "MyWorker" for name in worker_types.values())


@pytest.mark.slow
@gen_cluster(nthreads=[], client=True)
async def test_preload_config(c, s):
    # Ensure dask worker pulls the preload from the Dask config if
    # not specified via a command line option
    preload_text = """
def dask_setup(worker):
    worker.foo = 'setup'
"""
    env = os.environ.copy()
    env["DASK_DISTRIBUTED__WORKER__PRELOAD"] = preload_text
    with popen([sys.executable, "-m", "dask", "worker", s.address], env=env):
        await c.wait_for_workers(1)
        [foo] = (await c.run(lambda dask_worker: dask_worker.foo)).values()
        assert foo == "setup"


@pytest.mark.slow
@gen_cluster(nthreads=[], client=True)
async def test_set_lifetime_stagger_via_env_var(c, s):
    # Ensure that lifetime stagger can be set via an environment variable
    env = os.environ.copy()
    env["DASK_DISTRIBUTED__WORKER__LIFETIME__DURATION"] = "10 seconds"
    env["DASK_DISTRIBUTED__WORKER__LIFETIME__STAGGER"] = "2 seconds"
    with (
        popen([sys.executable, "-m", "dask", "worker", s.address], env=env),
        popen([sys.executable, "-m", "dask", "worker", s.address], env=env),
    ):
        await c.wait_for_workers(2)
        [lifetime1, lifetime2] = (
            await c.run(lambda dask_worker: dask_worker.lifetime)
        ).values()
        assert lifetime1 != lifetime2
        assert 8 <= lifetime1 <= 12
        assert 8 <= lifetime2 <= 12


@pytest.mark.slow
@gen_cluster(nthreads=[], client=True)
async def test_set_lifetime_restart_via_env_var(c, s):
    # Ensure that lifetime restart can be set via an environment variable
    env = os.environ.copy()
    env["DASK_DISTRIBUTED__WORKER__LIFETIME__RESTART"] = "True"
    with popen([sys.executable, "-m", "dask", "worker", s.address], env=env):
        await c.wait_for_workers(1)
        [lifetime_restart] = (
            await c.run(lambda dask_worker: dask_worker.lifetime_restart)
        ).values()
        assert lifetime_restart


@pytest.mark.slow
@pytest.mark.parametrize("nanny", ["--nanny", "--no-nanny"])
def test_timeout(nanny):
    worker = subprocess.run(
        [
            sys.executable,
            "-m",
            "distributed.cli.dask_worker",
            "192.168.1.100:7777",
            nanny,
            "--death-timeout=1",
        ],
        text=True,
        encoding="utf8",
        capture_output=True,
    )

    assert "timed out starting worker" in worker.stderr.lower()
    assert "end worker" in worker.stderr.lower()
    assert worker.returncode == 1


@pytest.mark.skipif(WINDOWS, reason="POSIX only")
@pytest.mark.parametrize("nanny", ["--nanny", "--no-nanny"])
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
@gen_cluster(client=True, nthreads=[])
async def test_signal_handling(c, s, nanny, sig):
    with subprocess.Popen(
        [sys.executable, "-m", "distributed.cli.dask_worker", s.address, nanny],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as worker:
        await c.wait_for_workers(1)

        worker.send_signal(sig)
        stdout, stderr = worker.communicate()
        logs = stdout.decode().lower()
        assert stderr is None
        assert sig.name.lower() in logs
        assert worker.returncode == 0
        if nanny == "--nanny":
            assert "closing nanny" in logs
            assert "stopping worker" in logs
        else:
            assert "nanny" not in logs
        assert "end worker" in logs
        assert "timed out" not in logs
        assert "error" not in logs
        assert "exception" not in logs


@pytest.mark.parametrize("nanny", ["--nanny", "--no-nanny"])
def test_error_during_startup(monkeypatch, nanny, loop):
    # see https://github.com/dask/distributed/issues/6320
    scheduler_port = open_port()
    scheduler_addr = f"tcp://127.0.0.1:{scheduler_port}"

    monkeypatch.setenv("DASK_SCHEDULER_ADDRESS", scheduler_addr)
    with popen(
        [
            sys.executable,
            "-m",
            "dask",
            "scheduler",
            f"--port={scheduler_port}",
            "--dashboard-address=:0",
        ],
    ):
        with Client(scheduler_addr, loop=loop) as c:
            with popen(
                [
                    sys.executable,
                    "-m",
                    "dask",
                    "worker",
                    scheduler_addr,
                    nanny,
                    # This should clash due to a port conflict
                    f"--worker-port={scheduler_port}",
                ],
            ) as worker:
                assert worker.wait(10) == 1


def test_single_executable_deprecated():
    assert (
        b"FutureWarning: dask-worker is deprecated"
        in subprocess.run(["dask-worker"], capture_output=True).stderr
    )


@pytest.mark.slow
@gen_cluster(nthreads=[], client=True)
async def test_single_executable_works(c, s):
    with popen(["dask-worker", s.address]):
        # make sure the worker still works
        await c.wait_for_workers(1)
        results = await c.submit(inc, 1).result()
        assert results == 2

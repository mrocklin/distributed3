from __future__ import annotations

import re

import psutil
import pytest

pytest.importorskip("requests")
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from time import sleep

import requests
from click.testing import CliRunner

import dask
from dask.utils import tmpfile

import distributed
import distributed.cli.dask_scheduler
from distributed import Client, Scheduler
from distributed.compatibility import LINUX, WINDOWS
from distributed.metrics import time
from distributed.utils import get_ip, get_ip_interface, open_port
from distributed.utils_test import (
    assert_can_connect_from_everywhere_4_6,
    assert_can_connect_locally_4,
    popen,
)


def _get_dashboard_port(client: Client) -> int:
    match = re.search(r":(\d+)\/status", client.dashboard_link)
    assert match
    return int(match.group(1))


def test_defaults(loop, requires_default_ports):
    with popen(["dask-scheduler"]):

        async def f():
            # Default behaviour is to listen on all addresses
            await assert_can_connect_from_everywhere_4_6(
                Scheduler.default_port, timeout=5.0
            )

        with Client(f"127.0.0.1:{Scheduler.default_port}", loop=loop) as c:
            c.sync(f)
            assert _get_dashboard_port(c) == 8787


def test_hostport(loop):
    port = open_port()
    with popen(["dask-scheduler", "--no-dashboard", "--host", f"127.0.0.1:{port}"]):

        async def f():
            # The scheduler's main port can't be contacted from the outside
            await assert_can_connect_locally_4(int(port), timeout=5.0)

        with Client(f"127.0.0.1:{port}", loop=loop) as c:
            assert len(c.nthreads()) == 0
            c.sync(f)


def test_no_dashboard(loop, requires_default_ports):
    dashboard_default_port = dask.config.get(
        "distributed.scheduler.dashboard.default-port"
    )
    with popen(["dask-scheduler", "--no-dashboard"]):
        with Client(f"127.0.0.1:{Scheduler.default_port}", loop=loop):
            response = requests.get(
                f"http://127.0.0.1:{dashboard_default_port}/status/"
            )
            assert response.status_code == 404


def test_dashboard(loop):
    pytest.importorskip("bokeh")
    port = open_port()

    with popen(
        ["dask-scheduler", "--host", f"127.0.0.1:{port}"],
    ):

        with Client(f"127.0.0.1:{port}", loop=loop) as c:
            dashboard_port = _get_dashboard_port(c)

        names = ["localhost", "127.0.0.1", get_ip()]
        start = time()
        while True:
            try:
                # All addresses should respond
                for name in names:
                    uri = f"http://{name}:{dashboard_port}/status/"
                    response = requests.get(uri)
                    response.raise_for_status()
                break
            except Exception as e:
                print(f"Got error on {uri!r}: {e.__class__.__name__}: {e}")
                elapsed = time() - start
                if elapsed > 10:
                    print(f"Timed out after {elapsed:.2f} seconds")
                    raise
                sleep(0.1)

    with pytest.raises(Exception):
        requests.get(f"http://127.0.0.1:{dashboard_port}/status/")


def test_dashboard_non_standard_ports(loop):
    pytest.importorskip("bokeh")
    port1 = open_port()
    port2 = open_port()
    with popen(
        [
            "dask-scheduler",
            f"--port={port1}",
            f"--dashboard-address=:{port2}",
        ]
    ) as proc:
        with Client(f"127.0.0.1:{port1}", loop=loop) as c:
            pass

        start = time()
        while True:
            try:
                response = requests.get(f"http://localhost:{port2}/status/")
                assert response.ok
                break
            except Exception:
                sleep(0.1)
                assert time() < start + 20
    with pytest.raises(Exception):
        requests.get(f"http://localhost:{port2}/status/")


@pytest.mark.skipif(not LINUX, reason="Need 127.0.0.2 to mean localhost")
def test_dashboard_allowlist(loop, dashboard_default_port):
    pytest.importorskip("bokeh")
    with pytest.raises(Exception):
        requests.get(f"http://localhost:{dashboard_default_port}/status/").ok

    port = open_port()
    with popen(
        [
            "dask-scheduler",
            f"--port={port}",
        ]
    ) as proc:
        with Client(f"127.0.0.1:{port}", loop=loop) as c:
            pass

        start = time()
        while True:
            try:
                for name in ["127.0.0.2", "127.0.0.3"]:
                    response = requests.get(
                        f"http://{name}:{dashboard_default_port}/status/"
                    )
                    assert response.ok
                break
            except Exception as f:
                print(f)
                sleep(0.1)
                assert time() < start + 20


def test_interface(loop):
    if_names = sorted(psutil.net_if_addrs())
    for if_name in if_names:
        try:
            ipv4_addr = get_ip_interface(if_name)
        except ValueError:
            pass
        else:
            if ipv4_addr == "127.0.0.1":
                break
    else:
        pytest.skip(
            "Could not find loopback interface. "
            "Available interfaces are: %s." % (if_names,)
        )

    port = open_port()
    with popen(
        [
            "dask-scheduler",
            f"--port={port}",
            "--no-dashboard",
            "--interface",
            if_name,
        ]
    ) as s:
        with popen(
            [
                "dask-worker",
                f"127.0.0.1:{port}",
                "--no-dashboard",
                "--interface",
                if_name,
            ]
        ) as a:
            with Client(f"tcp://127.0.0.1:{port}", loop=loop) as c:
                start = time()
                while not len(c.nthreads()):
                    sleep(0.1)
                    assert time() - start < 30
                info = c.scheduler_info()
                assert "tcp://127.0.0.1" in info["address"]
                assert all("127.0.0.1" == d["host"] for d in info["workers"].values())


@pytest.mark.flaky(reruns=10, reruns_delay=5)
def test_pid_file(loop):
    port = open_port()

    def check_pidfile(proc, pidfile):
        start = time()
        while not os.path.exists(pidfile):
            sleep(0.01)
            assert time() < start + 30

        text = False
        start = time()
        while not text:
            sleep(0.01)
            assert time() < start + 30
            with open(pidfile) as f:
                text = f.read()
        pid = int(text)
        if sys.platform.startswith("win"):
            # On Windows, `dask-XXX` invokes the dask-XXX.exe
            # shim, but the PID is written out by the child Python process
            assert pid
        else:
            assert proc.pid == pid

    with tmpfile() as s:
        with popen(["dask-scheduler", "--pid-file", s, "--no-dashboard"]) as sched:
            check_pidfile(sched, s)

        with tmpfile() as w:
            with popen(
                [
                    "dask-worker",
                    f"127.0.0.1:{port}",
                    "--pid-file",
                    w,
                    "--no-dashboard",
                ]
            ) as worker:
                check_pidfile(worker, w)


def test_scheduler_port_zero(loop):
    with tmpfile() as fn:
        with popen(
            ["dask-scheduler", "--no-dashboard", "--scheduler-file", fn, "--port", "0"]
        ):
            with Client(scheduler_file=fn, loop=loop) as c:
                assert c.scheduler.port
                assert c.scheduler.port != 8786


def test_dashboard_port_zero(loop):
    pytest.importorskip("bokeh")
    port = open_port()
    with popen(
        [
            "dask-scheduler",
            "--host",
            f"127.0.0.1:{port}",
            "--dashboard-address",
            ":0",
        ],
    ):
        with Client(f"tcp://127.0.0.1:{port}", loop=loop) as c:
            port = _get_dashboard_port(c)
            assert port > 0


PRELOAD_TEXT = """
_scheduler_info = {}

def dask_setup(scheduler):
    _scheduler_info['address'] = scheduler.address
    scheduler.foo = "bar"

def get_scheduler_address():
    return _scheduler_info['address']
"""


def test_preload_file(loop, tmp_path):
    def check_scheduler():
        import scheduler_info

        return scheduler_info.get_scheduler_address()

    path = tmp_path / "scheduler_info.py"
    with open(path, "w") as f:
        f.write(PRELOAD_TEXT)
    with tmpfile() as fn:
        with popen(
            [
                "dask-scheduler",
                "--scheduler-file",
                fn,
                "--preload",
                path,
                f"--port={open_port()}",
            ]
        ):
            with Client(scheduler_file=fn, loop=loop) as c:
                assert c.run_on_scheduler(check_scheduler) == c.scheduler.address


def test_preload_module(loop, tmp_path):
    def check_scheduler():
        import scheduler_info

        return scheduler_info.get_scheduler_address()

    path = tmp_path / "scheduler_info.py"
    with open(path, "w") as f:
        f.write(PRELOAD_TEXT)
    env = os.environ.copy()
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = str(tmp_path) + ":" + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = str(tmp_path)
    with tmpfile() as fn:
        with popen(
            [
                "dask-scheduler",
                "--scheduler-file",
                fn,
                "--preload",
                "scheduler_info",
                f"--port={open_port()}",
            ],
            env=env,
        ):
            with Client(scheduler_file=fn, loop=loop) as c:
                assert c.run_on_scheduler(check_scheduler) == c.scheduler.address


def test_preload_remote_module(loop, tmp_path):
    with open(tmp_path / "scheduler_info.py", "w") as f:
        f.write(PRELOAD_TEXT)
    http_server_port = open_port()
    with popen(
        [sys.executable, "-m", "http.server", str(http_server_port)], cwd=tmp_path
    ):
        with popen(
            [
                "dask-scheduler",
                "--scheduler-file",
                str(tmp_path / "scheduler-file.json"),
                "--preload",
                f"http://localhost:{http_server_port}/scheduler_info.py",
                f"--port={open_port()}",
            ]
        ) as proc:
            with Client(
                scheduler_file=tmp_path / "scheduler-file.json", loop=loop
            ) as c:
                assert (
                    c.run_on_scheduler(
                        lambda dask_scheduler: getattr(dask_scheduler, "foo", None)
                    )
                    == "bar"
                )


def test_preload_config(loop):
    # Ensure dask-scheduler pulls the preload from the Dask config if
    # not specified via a command line option
    with tmpfile() as fn:
        env = os.environ.copy()
        env["DASK_DISTRIBUTED__SCHEDULER__PRELOAD"] = PRELOAD_TEXT
        with popen(["dask-scheduler", "--scheduler-file", fn], env=env):
            with Client(scheduler_file=fn, loop=loop) as c:
                assert (
                    c.run_on_scheduler(lambda dask_scheduler: dask_scheduler.foo)
                    == "bar"
                )


PRELOAD_COMMAND_TEXT = """
import click
_config = {}

@click.command()
@click.option("--passthrough", type=str, default="default")
def dask_setup(scheduler, passthrough):
    _config["passthrough"] = passthrough

def get_passthrough():
    return _config["passthrough"]
"""


def test_preload_command(loop):
    def check_passthrough():
        import passthrough_info

        return passthrough_info.get_passthrough()

    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "passthrough_info.py")
        with open(path, "w") as f:
            f.write(PRELOAD_COMMAND_TEXT)

        with tmpfile() as fn:
            print(fn)
            with popen(
                [
                    "dask-scheduler",
                    "--scheduler-file",
                    fn,
                    "--preload",
                    path,
                    "--passthrough",
                    "foobar",
                ]
            ):
                with Client(scheduler_file=fn, loop=loop) as c:
                    assert c.run_on_scheduler(check_passthrough) == "foobar"
    finally:
        shutil.rmtree(tmpdir)


def test_preload_command_default(loop):
    def check_passthrough():
        import passthrough_info

        return passthrough_info.get_passthrough()

    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "passthrough_info.py")
        with open(path, "w") as f:
            f.write(PRELOAD_COMMAND_TEXT)

        with tmpfile() as fn2:
            print(fn2)
            with popen(
                ["dask-scheduler", "--scheduler-file", fn2, "--preload", path],
                stdout=sys.stdout,
                stderr=sys.stderr,
            ):
                with Client(scheduler_file=fn2, loop=loop) as c:
                    assert c.run_on_scheduler(check_passthrough) == "default"

    finally:
        shutil.rmtree(tmpdir)


def test_version_option():
    runner = CliRunner()
    result = runner.invoke(distributed.cli.dask_scheduler.main, ["--version"])
    assert result.exit_code == 0


@pytest.mark.slow
def test_idle_timeout():
    start = time()
    runner = CliRunner()
    result = runner.invoke(
        distributed.cli.dask_scheduler.main, ["--idle-timeout", "1s"]
    )
    stop = time()
    assert 1 < stop - start < 10
    assert result.exit_code == 0


@pytest.mark.slow
def test_restores_signal_handler():
    # another test could have altered the signal handler, so use a new function
    # that both has sensible sigint behaviour *and* can be used as a sentinel
    def raise_ki():
        raise KeyboardInterrupt

    original_handler = signal.signal(signal.SIGINT, raise_ki)
    try:
        CliRunner().invoke(
            distributed.cli.dask_scheduler.main, ["--idle-timeout", "1s"]
        )
        assert signal.getsignal(signal.SIGINT) is raise_ki
    finally:
        signal.signal(signal.SIGINT, original_handler)


def test_multiple_workers_2(loop):
    text = """
def dask_setup(worker):
    worker.foo = 'setup'
"""
    port = open_port()
    with popen(
        ["dask-scheduler", "--no-dashboard", "--host", f"127.0.0.1:{port}"]
    ) as s:
        with popen(
            [
                "dask-worker",
                f"localhost:{port}",
                "--no-dashboard",
                "--preload",
                text,
                "--preload-nanny",
                text,
            ]
        ) as a:
            with Client(f"127.0.0.1:{port}", loop=loop) as c:
                c.wait_for_workers(1)
                [foo] = c.run(lambda dask_worker: dask_worker.foo).values()
                assert foo == "setup"
                [foo] = c.run(lambda dask_worker: dask_worker.foo, nanny=True).values()
                assert foo == "setup"


def test_multiple_workers(loop):
    scheduler_address = f"127.0.0.1:{open_port()}"
    with popen(["dask-scheduler", "--no-dashboard", "--host", scheduler_address]) as s:
        with popen(["dask-worker", scheduler_address, "--no-dashboard"]) as a:
            with popen(["dask-worker", scheduler_address, "--no-dashboard"]) as b:
                with Client(scheduler_address, loop=loop) as c:
                    start = time()
                    while len(c.nthreads()) < 2:
                        sleep(0.1)
                        assert time() < start + 10


@pytest.mark.slow
@pytest.mark.skipif(WINDOWS, reason="POSIX only")
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_signal_handling(loop, sig):
    port = open_port()
    with subprocess.Popen(
        [
            "python",
            "-m",
            "distributed.cli.dask_scheduler",
            f"--port={port}",
            "--dashboard-address=:0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as scheduler:
        # Wait for scheduler to start
        with Client(f"127.0.0.1:{port}", loop=loop) as c:
            pass
        scheduler.send_signal(sig)
        stdout, stderr = scheduler.communicate()
        logs = stdout.decode().lower()
        assert stderr is None
        assert sig.name.lower() in logs
        assert scheduler.returncode == 0
        assert "scheduler closing" in logs
        assert "end scheduler" in logs

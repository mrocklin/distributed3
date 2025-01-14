from __future__ import annotations

import os
import ssl
from contextlib import contextmanager

import pytest

import dask

from distributed.comm import connect, listen
from distributed.security import Security
from distributed.utils_test import gen_test, get_cert, xfail_ssl_issue5601

ca_file = get_cert("tls-ca-cert.pem")

cert1 = get_cert("tls-cert.pem")
key1 = get_cert("tls-key.pem")
keycert1 = get_cert("tls-key-cert.pem")

# Note this cipher uses RSA auth as this matches our test certs
FORCED_CIPHER = "ECDHE-RSA-AES128-GCM-SHA256"

TLS_13_CIPHERS = [
    "TLS_AES_128_GCM_SHA256",
    "TLS_AES_256_GCM_SHA384",
    "TLS_CHACHA20_POLY1305_SHA256",
    "TLS_AES_128_CCM_SHA256",
    "TLS_AES_128_CCM_8_SHA256",
]


def test_defaults():
    sec = Security()
    assert sec.require_encryption in (None, False)
    assert sec.tls_ca_file is None
    assert sec.tls_ciphers is None
    assert sec.tls_client_key is None
    assert sec.tls_client_cert is None
    assert sec.tls_scheduler_key is None
    assert sec.tls_scheduler_cert is None
    assert sec.tls_worker_key is None
    assert sec.tls_worker_cert is None
    assert sec.tls_min_version is ssl.TLSVersion.TLSv1_2
    assert sec.tls_max_version is None
    assert sec.extra_conn_args == {}


def test_constructor_errors():
    with pytest.raises(TypeError) as exc:
        Security(unknown_keyword="bar")

    assert "unknown_keyword" in str(exc.value)


def test_attribute_error():
    sec = Security()
    assert hasattr(sec, "tls_ca_file")
    with pytest.raises(AttributeError):
        sec.tls_foobar
    with pytest.raises(AttributeError):
        sec.tls_foobar = ""


def test_from_config():
    c = {
        "distributed.comm.tls.ca-file": "ca.pem",
        "distributed.comm.tls.scheduler.key": "skey.pem",
        "distributed.comm.tls.scheduler.cert": "scert.pem",
        "distributed.comm.tls.worker.cert": "wcert.pem",
        "distributed.comm.tls.ciphers": FORCED_CIPHER,
        "distributed.comm.require-encryption": True,
    }

    with dask.config.set(c):
        sec = Security()

    assert sec.require_encryption is True
    assert sec.tls_ca_file == "ca.pem"
    assert sec.tls_ciphers == FORCED_CIPHER
    assert sec.tls_client_key is None
    assert sec.tls_client_cert is None
    assert sec.tls_scheduler_key == "skey.pem"
    assert sec.tls_scheduler_cert == "scert.pem"
    assert sec.tls_worker_key is None
    assert sec.tls_worker_cert == "wcert.pem"


def test_expand_vars():
    from pathlib import Path

    cert_path = str(Path("{HOME}") / "cert.pem")
    s = Security(tls_ca_file=cert_path)
    assert s.tls_ca_file == str(Path(os.environ["HOME"]) / "cert.pem")


@pytest.mark.parametrize("min_ver", [None, 1.2, 1.3])
@pytest.mark.parametrize("max_ver", [None, 1.2, 1.3])
def test_min_max_version_from_config(min_ver, max_ver):
    versions = {1.2: ssl.TLSVersion.TLSv1_2, 1.3: ssl.TLSVersion.TLSv1_3}
    min_ver_val = versions.get(min_ver, ssl.TLSVersion.TLSv1_2)
    max_ver_val = versions.get(max_ver)
    c = {
        "distributed.comm.tls.min-version": min_ver,
        "distributed.comm.tls.max-version": max_ver,
    }
    with dask.config.set(c):
        sec = Security()
    assert sec.tls_min_version == min_ver_val
    assert sec.tls_max_version == max_ver_val


@pytest.mark.parametrize("field", ["min-version", "max-version"])
def test_min_max_version_config_errors(field):
    with dask.config.set({f"distributed.comm.tls.{field}": "bad"}):
        with pytest.raises(ValueError, match="'bad' is not supported, expected one of"):
            sec = Security()


def test_invalid_min_version_from_config_errors():
    with dask.config.set({"distributed.comm.tls.min-version": None}):
        with pytest.raises(
            ValueError, match=r"tls_min_version=.* is not supported, expected one of .*"
        ):
            Security(tls_min_version=ssl.TLSVersion.MINIMUM_SUPPORTED)


def test_kwargs():
    c = {
        "distributed.comm.tls.ca-file": "ca.pem",
        "distributed.comm.tls.scheduler.key": "skey.pem",
        "distributed.comm.tls.scheduler.cert": "scert.pem",
    }
    with dask.config.set(c):
        sec = Security(
            tls_scheduler_cert="newcert.pem",
            require_encryption=True,
            tls_ca_file=None,
            tls_min_version=None,
            tls_max_version=ssl.TLSVersion.TLSv1_3,
            extra_conn_args={"headers": {"Auth": "Token abc"}},
        )
    assert sec.require_encryption is True
    assert sec.tls_ca_file is None
    assert sec.tls_ciphers is None
    assert sec.tls_min_version is ssl.TLSVersion.TLSv1_2
    assert sec.tls_max_version is ssl.TLSVersion.TLSv1_3
    assert sec.tls_client_key is None
    assert sec.tls_client_cert is None
    assert sec.tls_scheduler_key == "skey.pem"
    assert sec.tls_scheduler_cert == "newcert.pem"
    assert sec.tls_worker_key is None
    assert sec.tls_worker_cert is None
    assert sec.extra_conn_args == {"headers": {"Auth": "Token abc"}}


@pytest.mark.parametrize("key", ["tls_min_version", "tls_max_version"])
def test_min_max_version_kwarg_errors(key):
    with pytest.raises(ValueError, match="'bad' is not supported, expected one of"):
        sec = Security(**{key: "bad"})


def test_repr_temp_keys():
    xfail_ssl_issue5601()
    pytest.importorskip("cryptography")
    sec = Security.temporary()
    representation = repr(sec)
    assert "Temporary (In-memory)" in representation


def test_repr_local_keys():
    sec = Security(tls_ca_file="ca.pem", tls_scheduler_cert="scert.pem")
    representation = repr(sec)
    assert "ca.pem" in representation
    assert "scert.pem" in representation


def test_tls_config_for_role():
    c = {
        "distributed.comm.tls.ca-file": "ca.pem",
        "distributed.comm.tls.scheduler.key": "skey.pem",
        "distributed.comm.tls.scheduler.cert": "scert.pem",
        "distributed.comm.tls.worker.cert": "wcert.pem",
        "distributed.comm.tls.ciphers": FORCED_CIPHER,
    }
    with dask.config.set(c):
        sec = Security()
    t = sec.get_tls_config_for_role("scheduler")
    assert t == {
        "ca_file": "ca.pem",
        "key": "skey.pem",
        "cert": "scert.pem",
        "ciphers": FORCED_CIPHER,
    }
    t = sec.get_tls_config_for_role("worker")
    assert t == {
        "ca_file": "ca.pem",
        "key": None,
        "cert": "wcert.pem",
        "ciphers": FORCED_CIPHER,
    }
    t = sec.get_tls_config_for_role("client")
    assert t == {
        "ca_file": "ca.pem",
        "key": None,
        "cert": None,
        "ciphers": FORCED_CIPHER,
    }
    with pytest.raises(ValueError):
        sec.get_tls_config_for_role("supervisor")


def assert_many_ciphers(ctx):
    assert len(ctx.get_ciphers()) > 2  # Most likely


def test_connection_args():
    def basic_checks(ctx):
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.check_hostname is False
        assert ctx.minimum_version is ssl.TLSVersion.TLSv1_2
        assert ctx.maximum_version is ssl.TLSVersion.TLSv1_3

    c = {
        "distributed.comm.tls.ca-file": ca_file,
        "distributed.comm.tls.scheduler.key": key1,
        "distributed.comm.tls.scheduler.cert": cert1,
        "distributed.comm.tls.worker.cert": keycert1,
        "distributed.comm.tls.min-version": None,
        "distributed.comm.tls.max-version": 1.3,
    }
    with dask.config.set(c):
        sec = Security()

    d = sec.get_connection_args("scheduler")
    assert not d["require_encryption"]
    ctx = d["ssl_context"]
    basic_checks(ctx)
    assert_many_ciphers(ctx)

    d = sec.get_connection_args("worker")
    ctx = d["ssl_context"]
    basic_checks(ctx)
    assert_many_ciphers(ctx)

    # No cert defined => no TLS
    d = sec.get_connection_args("client")
    assert d.get("ssl_context") is None

    # With more settings
    c["distributed.comm.tls.ciphers"] = FORCED_CIPHER
    c["distributed.comm.require-encryption"] = True

    with dask.config.set(c):
        sec = Security()

    d = sec.get_listen_args("scheduler")
    assert d["require_encryption"]
    ctx = d["ssl_context"]
    basic_checks(ctx)

    supported_ciphers = ctx.get_ciphers()
    tls_12_ciphers = [c for c in supported_ciphers if "TLSv1.2" in c["description"]]
    assert len(tls_12_ciphers) == 1
    tls_13_ciphers = [c for c in supported_ciphers if "TLSv1.3" in c["description"]]
    assert len(tls_13_ciphers) == 0 or len(tls_13_ciphers) >= 3


def test_extra_conn_args_connection_args():
    c = {
        "distributed.comm.tls.ca-file": ca_file,
        "distributed.comm.tls.scheduler.key": key1,
        "distributed.comm.tls.scheduler.cert": cert1,
        "distributed.comm.tls.worker.cert": keycert1,
    }
    with dask.config.set(c):
        sec = Security(extra_conn_args={"headers": {"Authorization": "Token abcd"}})

    d = sec.get_connection_args("scheduler")
    assert not d["require_encryption"]
    assert d["extra_conn_args"]["headers"] == {"Authorization": "Token abcd"}
    ctx = d["ssl_context"]

    d = sec.get_connection_args("worker")
    assert d["extra_conn_args"]["headers"] == {"Authorization": "Token abcd"}

    # No cert defined => no TLS
    d = sec.get_connection_args("client")
    assert d.get("ssl_context") is None
    assert d["extra_conn_args"]["headers"] == {"Authorization": "Token abcd"}


def test_listen_args():
    def basic_checks(ctx):
        assert ctx.verify_mode == ssl.CERT_REQUIRED
        assert ctx.check_hostname is False
        assert ctx.minimum_version is ssl.TLSVersion.TLSv1_2
        assert ctx.maximum_version is ssl.TLSVersion.TLSv1_3

    c = {
        "distributed.comm.tls.ca-file": ca_file,
        "distributed.comm.tls.scheduler.key": key1,
        "distributed.comm.tls.scheduler.cert": cert1,
        "distributed.comm.tls.worker.cert": keycert1,
        "distributed.comm.tls.min-version": None,
        "distributed.comm.tls.max-version": 1.3,
    }
    with dask.config.set(c):
        sec = Security()

    d = sec.get_listen_args("scheduler")
    assert not d["require_encryption"]
    ctx = d["ssl_context"]
    basic_checks(ctx)
    assert_many_ciphers(ctx)

    d = sec.get_listen_args("worker")
    ctx = d["ssl_context"]
    basic_checks(ctx)
    assert_many_ciphers(ctx)

    # No cert defined => no TLS
    d = sec.get_listen_args("client")
    assert d.get("ssl_context") is None

    # With more settings
    c["distributed.comm.tls.ciphers"] = FORCED_CIPHER
    c["distributed.comm.require-encryption"] = True

    with dask.config.set(c):
        sec = Security()

    d = sec.get_listen_args("scheduler")
    assert d["require_encryption"]
    ctx = d["ssl_context"]
    basic_checks(ctx)

    supported_ciphers = ctx.get_ciphers()
    tls_12_ciphers = [c for c in supported_ciphers if "TLSv1.2" in c["description"]]
    assert len(tls_12_ciphers) == 1
    tls_13_ciphers = [c for c in supported_ciphers if "TLSv1.3" in c["description"]]
    assert len(tls_13_ciphers) == 0 or len(tls_13_ciphers) >= 3


@gen_test()
async def test_tls_listen_connect():
    """
    Functional test for TLS connection args.
    """

    async def handle_comm(comm):
        peer_addr = comm.peer_address
        assert peer_addr.startswith("tls://")
        await comm.write("hello")
        await comm.close()

    c = {
        "distributed.comm.tls.ca-file": ca_file,
        "distributed.comm.tls.scheduler.key": key1,
        "distributed.comm.tls.scheduler.cert": cert1,
        "distributed.comm.tls.worker.cert": keycert1,
    }
    with dask.config.set(c):
        sec = Security()

    c["distributed.comm.tls.ciphers"] = FORCED_CIPHER
    with dask.config.set(c):
        forced_cipher_sec = Security()

    async with listen(
        "tls://", handle_comm, **sec.get_listen_args("scheduler")
    ) as listener:
        comm = await connect(
            listener.contact_address, **sec.get_connection_args("worker")
        )
        msg = await comm.read()
        assert msg == "hello"
        comm.abort()

        # No SSL context for client
        with pytest.raises(TypeError):
            await connect(listener.contact_address, **sec.get_connection_args("client"))

        # Check forced cipher
        comm = await connect(
            listener.contact_address, **forced_cipher_sec.get_connection_args("worker")
        )
        cipher, _, _ = comm.extra_info["cipher"]
        assert cipher in [FORCED_CIPHER] + TLS_13_CIPHERS
        comm.abort()


@gen_test()
async def test_require_encryption():
    """
    Functional test for "require_encryption" setting.
    """

    async def handle_comm(comm):
        comm.abort()

    c = {
        "distributed.comm.tls.ca-file": ca_file,
        "distributed.comm.tls.scheduler.key": key1,
        "distributed.comm.tls.scheduler.cert": cert1,
        "distributed.comm.tls.worker.cert": keycert1,
    }
    with dask.config.set(c):
        sec = Security()

    c["distributed.comm.require-encryption"] = True
    with dask.config.set(c):
        sec2 = Security()

    for listen_addr in ["inproc://", "tls://"]:
        async with listen(
            listen_addr, handle_comm, **sec.get_listen_args("scheduler")
        ) as listener:
            comm = await connect(
                listener.contact_address, **sec2.get_connection_args("worker")
            )
            comm.abort()

        async with listen(
            listen_addr, handle_comm, **sec2.get_listen_args("scheduler")
        ) as listener:
            comm = await connect(
                listener.contact_address, **sec2.get_connection_args("worker")
            )
            comm.abort()

    @contextmanager
    def check_encryption_error():
        with pytest.raises(RuntimeError) as excinfo:
            yield
        assert "encryption required" in str(excinfo.value)

    for listen_addr in ["tcp://"]:
        async with listen(
            listen_addr, handle_comm, **sec.get_listen_args("scheduler")
        ) as listener:
            comm = await connect(
                listener.contact_address, **sec.get_connection_args("worker")
            )
            comm.abort()

            with pytest.raises(RuntimeError):
                await connect(
                    listener.contact_address, **sec2.get_connection_args("worker")
                )

        with pytest.raises(RuntimeError):
            listen(listen_addr, handle_comm, **sec2.get_listen_args("scheduler"))


def test_temporary_credentials():
    xfail_ssl_issue5601()
    pytest.importorskip("cryptography")

    sec = Security.temporary()
    sec_repr = repr(sec)
    fields = ["tls_ca_file"]
    fields.extend(
        f"tls_{role}_{kind}"
        for role in ["client", "scheduler", "worker"]
        for kind in ["key", "cert"]
    )
    for f in fields:
        val = getattr(sec, f)
        assert "\n" in val
        assert val not in sec_repr


def test_extra_conn_args_in_temporary_credentials():
    xfail_ssl_issue5601()
    pytest.importorskip("cryptography")

    sec = Security.temporary(extra_conn_args={"headers": {"X-Request-ID": "abcd"}})
    assert sec.extra_conn_args == {"headers": {"X-Request-ID": "abcd"}}


@gen_test()
async def test_tls_temporary_credentials_functional():
    xfail_ssl_issue5601()
    pytest.importorskip("cryptography")

    async def handle_comm(comm):
        peer_addr = comm.peer_address
        assert peer_addr.startswith("tls://")
        await comm.write("hello")
        await comm.close()

    sec = Security.temporary()

    async with listen(
        "tls://", handle_comm, **sec.get_listen_args("scheduler")
    ) as listener:
        comm = await connect(
            listener.contact_address, **sec.get_connection_args("worker")
        )
        msg = await comm.read()
        assert msg == "hello"
        comm.abort()

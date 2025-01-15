from __future__ import annotations

import contextlib
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap

import pytest
import yaml

import dask.config

from distributed.config import initialize_logging
from distributed.utils_test import captured_handler, new_config, new_config_file


@pytest.mark.parametrize(
    "config",
    [
        {},
        None,
    ],
)
def test_logging_default(caplog, config):
    """
    Test default logging configuration.
    """
    if config is None:
        ctx = contextlib.nullcontext()
    else:
        ctx = new_config(config)
    with ctx:
        d = logging.getLogger("distributed")
        assert isinstance(d.handlers[0], logging.StreamHandler)

        dfb = logging.getLogger("distributed.foo.bar")
        dfc = logging.getLogger("distributed.client")
        f = logging.getLogger("foo")
        fb = logging.getLogger("foo.bar")

        with captured_handler(d.handlers[0]) as distributed_log:
            d.debug("1: debug")
            d.info("2: info")
            dfb.info("3: info")
            fb.info("4: info")
            fb.error("5: error")
            f.info("6: info")
            f.error("7: error")
            dfc.info("8: ignore me")
            dfc.warning("9: important")

        # default logging sets propagate=False so caplog does not capture
        # distributed log records
        assert caplog.record_tuples == [
            # Info logs of foreign libraries are not logged because default is
            # WARNING
            ("foo.bar", logging.ERROR, "5: error"),
            ("foo", logging.ERROR, "7: error"),
        ]
        assert re.match(
            r"\A\d+-\d+-\d+ \d+:\d+:\d+,\d+ - distributed - INFO - 2: info\n"
            r"\d+-\d+-\d+ \d+:\d+:\d+,\d+ - distributed.foo.bar - INFO - 3: info\n"
            r"\d+-\d+-\d+ \d+:\d+:\d+,\d+ - distributed.client - WARNING - 9: important\n\Z",
            distributed_log.getvalue(),
        )


def test_logging_simple_under_distributed():
    """
    Test simple ("old-style") logging configuration under the distributed key.
    """
    c = {
        "distributed": {
            "logging": {"distributed.foo": "info", "distributed.foo.bar": "error"}
        }
    }
    # Must test using a subprocess to avoid wrecking pre-existing configuration
    with new_config_file(c):
        code = """if 1:
            import logging
            import dask

            from distributed.config import ensure_logging_configured
            ensure_logging_configured()
            from distributed.utils_test import captured_handler

            d = logging.getLogger('distributed')
            assert len(d.handlers) == 1
            assert isinstance(d.handlers[0], logging.StreamHandler)
            df = logging.getLogger('distributed.foo')
            dfb = logging.getLogger('distributed.foo.bar')

            with captured_handler(d.handlers[0]) as distributed_log:
                df.info("1: info")
                dfb.warning("2: warning")
                dfb.error("3: error")

            distributed_log = distributed_log.getvalue().splitlines()

            assert len(distributed_log) == 2, (dask.config.config, distributed_log)
            """

        subprocess.check_call([sys.executable, "-c", code])


def test_logging_simple():
    """
    Test simple ("old-style") logging configuration.
    """
    c = {"logging": {"distributed.foo": "info", "distributed.foo.bar": "error"}}
    # Must test using a subprocess to avoid wrecking pre-existing configuration
    with new_config_file(c):
        code = """if 1:
            import logging
            import dask

            from distributed.config import ensure_logging_configured
            ensure_logging_configured()
            from distributed.utils_test import captured_handler

            d = logging.getLogger('distributed')
            assert len(d.handlers) == 1
            assert isinstance(d.handlers[0], logging.StreamHandler)
            df = logging.getLogger('distributed.foo')
            dfb = logging.getLogger('distributed.foo.bar')

            with captured_handler(d.handlers[0]) as distributed_log:
                df.info("1: info")
                dfb.warning("2: warning")
                dfb.error("3: error")

            distributed_log = distributed_log.getvalue().splitlines()

            assert len(distributed_log) == 2, (dask.config.config, distributed_log)
            """

        subprocess.check_call([sys.executable, "-c", code])


def test_logging_extended():
    """
    Test extended ("new-style") logging configuration.
    """
    c = {
        "logging": {
            "version": "1",
            "formatters": {
                "simple": {"format": "%(levelname)s: %(name)s: %(message)s"}
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                    "formatter": "simple",
                }
            },
            "loggers": {
                "distributed.foo": {
                    "level": "INFO",
                    # 'handlers': ['console'],
                },
                "distributed.foo.bar": {
                    "level": "ERROR",
                    # 'handlers': ['console'],
                },
            },
            "root": {"level": "WARNING", "handlers": ["console"]},
        }
    }
    # Must test using a subprocess to avoid wrecking pre-existing configuration
    with new_config_file(c):
        code = """if 1:
            import logging

            from distributed.config import ensure_logging_configured
            ensure_logging_configured()
            from distributed.utils_test import captured_handler

            root = logging.getLogger()
            d = logging.getLogger('distributed')
            df = logging.getLogger('distributed.foo')
            dfb = logging.getLogger('distributed.foo.bar')

            with captured_handler(root.handlers[0]) as root_log:
                df.info("1: info")
                dfb.warning("2: warning")
                dfb.error("3: error")
                d.info("4: info")
                d.warning("5: warning")

            root_log = root_log.getvalue().splitlines()
            print(root_log)

            assert root_log == [
                "INFO: distributed.foo: 1: info",
                "ERROR: distributed.foo.bar: 3: error",
                "WARNING: distributed: 5: warning",
                ]
            """

        subprocess.check_call([sys.executable, "-c", code])


def test_default_logging_does_not_override_basic_config():
    code = textwrap.dedent(
        """\
        import logging
        logging.basicConfig()
        import distributed
        from distributed.config import ensure_logging_configured
        ensure_logging_configured()
        logging.getLogger("distributed").warning("hello")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, encoding="utf8"
    )
    assert proc.stdout == ""
    assert proc.stderr == "WARNING:distributed:hello\n"


def test_basic_config_does_not_override_default_logging():
    code = textwrap.dedent(
        """\
        import logging
        import distributed
        from distributed.config import ensure_logging_configured
        ensure_logging_configured()
        logging.basicConfig()
        logging.getLogger("distributed").warning("hello")
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, encoding="utf8"
    )
    assert proc.stdout == ""
    assert re.match(
        r"\A\d+-\d+-\d+ \d+:\d+:\d+,\d+ - distributed - WARNING - hello\n\Z",
        proc.stderr,
    )


def test_logging_mutual_exclusive():
    """
    Ensure that 'logging-file-config' and 'logging' have to be mutual exclusive.
    """
    config = {"logging": {"dask": "warning"}, "logging-file-config": "/path/to/config"}
    with pytest.raises(RuntimeError):
        initialize_logging(config)


def test_logging_file_config():
    """
    Test `logging-file-config` logging configuration
    """
    logging_config_contents = """
[handlers]
keys=console

[formatters]
keys=simple

[loggers]
keys=root, foo, foo_bar

[handler_console]
class=StreamHandler
level=INFO
formatter=simple
args=(sys.stdout,)

[formatter_simple]
format=%(levelname)s: %(name)s: %(message)s

[logger_root]
level=WARNING
handlers=console

[logger_foo]
level=INFO
handlers=console
qualname=foo

[logger_foo_bar]
level=ERROR
handlers=console
qualname=foo.bar
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as logging_config:
        logging_config.write(logging_config_contents)
    dask_config = {"logging-file-config": logging_config.name}
    with new_config_file(dask_config):
        code = """if 1:
            import logging
            from distributed.config import ensure_logging_configured
            ensure_logging_configured()
            foo = logging.getLogger('foo')
            bar = logging.getLogger('foo.bar')
            assert logging.INFO == foo.getEffectiveLevel()
            assert logging.ERROR == bar.getEffectiveLevel()
            """
        subprocess.check_call([sys.executable, "-c", code])
    os.remove(logging_config.name)


def test_schema():
    jsonschema = pytest.importorskip("jsonschema")
    config_fn = os.path.join(os.path.dirname(__file__), "..", "distributed.yaml")
    schema_fn = os.path.join(os.path.dirname(__file__), "..", "distributed-schema.yaml")

    with open(config_fn) as f:
        config = yaml.safe_load(f)

    with open(schema_fn) as f:
        schema = yaml.safe_load(f)

    jsonschema.validate(config, schema)


def test_schema_is_complete():
    config_fn = os.path.join(os.path.dirname(__file__), "..", "distributed.yaml")
    schema_fn = os.path.join(os.path.dirname(__file__), "..", "distributed-schema.yaml")

    with open(config_fn) as f:
        config = yaml.safe_load(f)

    with open(schema_fn) as f:
        schema = yaml.safe_load(f)

    skip = {
        "distributed.scheduler.default-task-durations",
        "distributed.scheduler.dashboard.bokeh-application",
        "distributed.nanny.environ",
        "distributed.nanny.pre-spawn-environ",
        "distributed.comm.ucx.environment",
    }

    def test_matches(c, s, root):
        if set(c) != set(s["properties"]):
            raise ValueError(
                "\nThe distributed.yaml and distributed-schema.yaml files are not in sync.\n"
                "This usually happens when we add a new configuration value,\n"
                "but don't add the schema of that value to the distributed-schema.yaml file\n"
                "Please modify these files to include the missing values: \n\n"
                "    distributed.yaml:        {}\n"
                "    distributed-schema.yaml: {}\n\n"
                "Examples in these files should be a good start, \n"
                "even if you are not familiar with the jsonschema spec".format(
                    sorted(c), sorted(s["properties"])
                )
            )
        for k, v in c.items():
            key = f"{root}.{k}" if root else k
            if isinstance(v, dict) and key not in skip:
                test_matches(c[k], s["properties"][k], key)

    test_matches(config, schema, "")


def test_uvloop_event_loop():
    """Check that configuring distributed to use uvloop actually sets the event loop policy"""
    pytest.importorskip("uvloop")
    script = (
        "import distributed, asyncio, uvloop\n"
        "assert isinstance(asyncio.get_event_loop_policy(), uvloop.EventLoopPolicy)"
    )
    subprocess.check_call(
        [sys.executable, "-c", script],
        env={"DASK_DISTRIBUTED__ADMIN__EVENT_LOOP": "uvloop"},
    )


@pytest.mark.parametrize(
    "args,kwargs",
    [
        ((), {"allowed_failures": 123}),
        (({"allowed_failures": 123},), {}),
        (({"allowed-failures": 123},), {}),
    ],
)
def test_deprecations_on_set(args, kwargs):
    with pytest.warns(FutureWarning) as info:
        with dask.config.set(*args, **kwargs):
            assert dask.config.get("distributed.scheduler.allowed-failures") == 123

    assert "distributed.scheduler.allowed-failures" in str(info[0].message)


def test_deprecations_on_env_variables(monkeypatch):
    d = {}
    monkeypatch.setenv("DASK_ALLOWED_FAILURES", "123")
    with pytest.warns(FutureWarning) as info:
        dask.config.refresh(config=d)
    assert "distributed.scheduler.allowed-failures" in str(info[0].message)
    assert dask.config.get("distributed.scheduler.allowed-failures", config=d) == 123


@pytest.mark.parametrize("key", ["allowed-failures", "allowed_failures"])
def test_deprecations_on_yaml(tmp_path, key):
    d = {}
    with open(tmp_path / "dask.yaml", "w") as fh:
        yaml.dump({key: 123}, fh)

    with pytest.warns(FutureWarning) as info:
        dask.config.refresh(config=d, paths=[tmp_path])
    assert "distributed.scheduler.allowed-failures" in str(info[0].message)
    assert dask.config.get("distributed.scheduler.allowed-failures", config=d) == 123

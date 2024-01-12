from __future__ import annotations

import inspect
import io
import logging
import pickle
from copyreg import dispatch_table
from types import FunctionType

import cloudpickle
from packaging.version import parse as parse_version

from distributed.protocol.serialize import dask_deserialize, dask_serialize

CLOUDPICKLE_GE_20 = parse_version(cloudpickle.__version__) >= parse_version("2.0.0")

HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL

logger = logging.getLogger(__name__)


class _DaskPickler(pickle.Pickler):
    def reducer_override(self, obj):
        mod = inspect.getmodule(type(obj))

        # If a thing is local scoped, use cloudpickle
        # This check is not guaranteed and evaluates false positively for
        # dynamically created types, e.g. numpy scalars
        if getattr(mod, type(obj).__name__, None) is not None:
            return pickle.loads, (cloudpickle.dumps(obj),)
        if isinstance(obj, FunctionType):
            module_name = pickle.whichmodule(obj, None)
            if (
                module_name == "__main__"
                or CLOUDPICKLE_GE_20
                and module_name in cloudpickle.list_registry_pickle_by_value()
            ):
                return pickle.loads, (cloudpickle.dumps(obj),)
        elif type(obj) is memoryview:
            return memoryview, (pickle.PickleBuffer(obj),)
        elif type(obj) not in dispatch_table:
            try:
                serialize = dask_serialize.dispatch(type(obj))
                deserialize = dask_deserialize.dispatch(type(obj))
                rv = deserialize, serialize(obj)
                return rv
            except Exception:
                return NotImplemented
        return NotImplemented


def dumps(x, *, buffer_callback=None, protocol=HIGHEST_PROTOCOL):
    """Manage between cloudpickle and pickle

    1.  Try pickle
    2.  If it is short then check if it contains __main__
    3.  If it is long, then first check type, then check __main__
    """
    buffers = []
    dump_kwargs = {"protocol": protocol or HIGHEST_PROTOCOL}

    if dump_kwargs["protocol"] >= 5 and buffer_callback is not None:
        dump_kwargs["buffer_callback"] = buffers.append

    try:
        f = io.BytesIO()
        pickler = _DaskPickler(f, **dump_kwargs)
        pickler.dump(x)
        result = f.getvalue()
    except Exception as exc:
        import traceback

        traceback.print_tb(exc.__traceback__)
        try:
            buffers.clear()
            result = cloudpickle.dumps(x, **dump_kwargs)
        except Exception:
            logger.exception("Failed to serialize %s.", x)
            raise
    if buffer_callback is not None:
        for b in buffers:
            buffer_callback(b)
    return result


def loads(x, *, buffers=()):
    try:
        if buffers:
            return pickle.loads(x, buffers=buffers)
        else:
            return pickle.loads(x)
    except Exception:
        logger.info("Failed to deserialize %s", x[:10000], exc_info=True)
        raise

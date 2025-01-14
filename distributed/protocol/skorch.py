import skorch

from . import pickle
from .serialize import dask_deserialize, dask_serialize


@dask_serialize.register(skorch.NeuralNet)
def serialize_skorch(x, context=None):
    protocol = (context or {}).get("pickle-protocol", None)
    headers = {}
    has_module = hasattr(x, "module_")
    if has_module:
        module = x.__dict__.pop("module_")
        # module's is an interactively defined class on client so its namespace is often `__main__` .
        # Pickle has problems pickling when interactively defined classes  when they are
        # set as an attributes of another object.
        # By pickling it on its own we are able to serialize successfully
        frames = [None]
        buffer_callback = lambda f: frames.append(memoryview(f))
        frames[0] = pickle.dumps(x, buffer_callback=buffer_callback, protocol=protocol)
        headers["subframe-split"] = i = len(frames)
        frames.append(None)
        frames[i] = pickle.dumps(
            module, buffer_callback=buffer_callback, protocol=protocol
        )
        x.__dict__["module_"] = module
    else:
        frames = [None]
        buffer_callback = lambda f: frames.append(memoryview(f))
        frames[0] = pickle.dumps(x, buffer_callback=buffer_callback, protocol=protocol)

    return headers, frames


@dask_deserialize.register(skorch.NeuralNet)
def deserialize_skorch(header, frames):
    i = header.get("subframe-split")
    model = pickle.loads(frames[0], buffers=frames[1:i])
    if i is not None:
        module = pickle.loads(frames[i], buffers=frames[i + 1 :])
        model.module_ = module
    return model

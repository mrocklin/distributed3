"Utilities for generating and analyzing cluster dumps"

from __future__ import annotations

from typing import IO, Any, Awaitable, Callable, Literal

import fsspec
import msgpack

from distributed.compatibility import to_thread


def _tuple_to_list(node):
    if isinstance(node, (list, tuple)):
        return [_tuple_to_list(el) for el in node]
    elif isinstance(node, dict):
        return {k: _tuple_to_list(v) for k, v in node.items()}
    else:
        return node


async def write_state(
    get_state: Callable[[], Awaitable[Any]],
    url: str,
    format: Literal["msgpack", "yaml"],
    **storage_options: dict[str, Any],
) -> None:
    "Await a cluster dump, then serialize and write it to a path"
    if format == "msgpack":
        mode = "wb"
        suffix = ".msgpack.gz"
        if not url.endswith(suffix):
            url += suffix
        writer = msgpack.pack
    elif format == "yaml":
        import yaml

        mode = "w"
        suffix = ".yaml"
        if not url.endswith(suffix):
            url += suffix

        def writer(state: dict, f: IO):
            # YAML adds unnecessary `!!python/tuple` tags; convert tuples to lists to avoid them.
            # Unnecessary for msgpack, since tuples and lists are encoded the same.
            yaml.dump(_tuple_to_list(state), f)

    else:
        raise ValueError(
            f"Unsupported format {format!r}. Possible values are 'msgpack' or 'yaml'."
        )

    # Eagerly open the file to catch any errors before doing the full dump
    # NOTE: `compression="infer"` will automatically use gzip via the `.gz` suffix
    with fsspec.open(url, mode, compression="infer", **storage_options) as f:
        state = await get_state()
        # Write from a thread so we don't block the event loop quite as badly
        # (the writer will still hold the GIL a lot though).
        await to_thread(writer, state, f)


def get_tasks_in_state(
    url: str,
    state: str,
    worker: bool = False,
) -> dict:
    if url.endswith(".msgpack.gz"):
        mode = "rb"
        reader = msgpack.unpack
    elif url.endswith(".yaml"):
        import yaml

        mode = "r"
        reader = yaml.safe_load
    else:
        raise ValueError(f"url ({url}) must have a .msgpack.gz or .yaml suffix")

    with fsspec.open(url, mode, compression="infer") as f:
        dump = reader(f)

    context_str = "workers" if worker else "scheduler"

    try:
        context = dump[context_str]
    except KeyError:
        raise ValueError(
            f"The '{context_str}' context was not present in the dumped state"
        )

    try:
        tasks = context["tasks"]
    except KeyError:
        raise ValueError(
            f"'tasks' was not present within the '{context_str}' "
            f"context of the dumped state"
        )

    if state:
        return {k: v for k, v in tasks.items() if v["state"] == state}

    return tasks

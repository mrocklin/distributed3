from __future__ import annotations

import asyncio
import logging
import random
from collections import defaultdict
from functools import partial
from itertools import cycle
from typing import Any, Callable

from tlz import concat, drop, groupby, merge

import dask.config
from dask.optimization import SubgraphCallable
from dask.utils import is_namedtuple_instance, parse_timedelta, stringify

from distributed.utils import All

logger = logging.getLogger(__name__)


async def gather_from_workers(who_has, rpc, close=True, serializers=None, who=None):
    """Gather data directly from peers

    Parameters
    ----------
    who_has: dict
        Dict mapping keys to sets of workers that may have that key
    rpc: callable

    Returns dict mapping key to value

    See Also
    --------
    gather
    _gather
    """
    from distributed.worker import get_data_from_worker

    bad_addresses = set()
    missing_workers = set()
    original_who_has = who_has
    who_has = {k: set(v) for k, v in who_has.items()}
    results = dict()
    all_bad_keys = set()

    while len(results) + len(all_bad_keys) < len(who_has):
        d = defaultdict(list)
        rev = dict()
        bad_keys = set()
        for key, addresses in who_has.items():
            if key in results:
                continue
            try:
                addr = random.choice(list(addresses - bad_addresses))
                d[addr].append(key)
                rev[key] = addr
            except IndexError:
                bad_keys.add(key)
        if bad_keys:
            all_bad_keys |= bad_keys
        coroutines = {
            address: asyncio.create_task(
                get_data_from_worker(
                    rpc,
                    keys,
                    address,
                    who=who,
                    serializers=serializers,
                    max_connections=False,
                ),
                name=f"get-data-from-{address}",
            )
            for address, keys in d.items()
        }
        response = {}
        for worker, c in coroutines.items():
            try:
                r = await c
            except OSError:
                missing_workers.add(worker)
            except ValueError as e:
                logger.info(
                    "Got an unexpected error while collecting from workers: %s", e
                )
                missing_workers.add(worker)
            else:
                response.update(r["data"])

        bad_addresses |= {v for k, v in rev.items() if k not in response}
        results.update(response)

    bad_keys = {k: list(original_who_has[k]) for k in all_bad_keys}
    return (results, bad_keys, list(missing_workers))


class WrappedKey:
    """Interface for a key in a dask graph.

    Subclasses must have .key attribute that refers to a key in a dask graph.

    Sometimes we want to associate metadata to keys in a dask graph.  For
    example we might know that that key lives on a particular machine or can
    only be accessed in a certain way.  Schedulers may have particular needs
    that can only be addressed by additional metadata.
    """

    def __init__(self, key):
        self.key = key

    def __repr__(self):
        return f"{type(self).__name__}('{self.key}')"


_round_robin_counter = [0]


async def scatter_to_workers(nthreads, data, rpc, report=True):
    """Scatter data directly to workers

    This distributes data in a round-robin fashion to a set of workers based on
    how many cores they have.  nthreads should be a dictionary mapping worker
    identities to numbers of cores.

    See scatter for parameter docstring
    """
    assert isinstance(nthreads, dict)
    assert isinstance(data, dict)

    workers = list(concat([w] * nc for w, nc in nthreads.items()))
    names, data = list(zip(*data.items()))

    worker_iter = drop(_round_robin_counter[0] % len(workers), cycle(workers))
    _round_robin_counter[0] += len(data)

    L = list(zip(worker_iter, names, data))
    d = groupby(0, L)
    d = {worker: {key: value for _, key, value in v} for worker, v in d.items()}

    rpcs = {addr: rpc(addr) for addr in d}
    out = await All(
        [
            rpcs[address].update_data(
                data=v,
                report=report,
            )
            for address, v in d.items()
        ]
    )
    nbytes = merge(o["nbytes"] for o in out)

    who_has = {k: [w for w, _, _ in v] for k, v in groupby(1, L).items()}

    return (names, who_has, nbytes)


collection_types = (tuple, list, set, frozenset)


def _namedtuple_packing(o: Any, handler: Callable[..., Any]) -> Any:
    """Special pack/unpack dispatcher for namedtuples respecting their potential constructors."""
    assert is_namedtuple_instance(o)
    typ = type(o)
    if hasattr(o, "__getnewargs_ex__"):
        args, kwargs = o.__getnewargs_ex__()
        handled_args = [handler(item) for item in args]
        handled_kwargs = {k: handler(v) for k, v in kwargs.items()}
        return typ(*handled_args, **handled_kwargs)
    args = o.__getnewargs__() if hasattr(typ, "__getnewargs__") else o
    handled_args = [handler(item) for item in args]
    return typ(*handled_args)


def _unpack_remotedata_inner(
    o: Any, byte_keys: bool, found_keys: set[WrappedKey]
) -> Any:
    """Inner implementation of `unpack_remotedata` that adds found wrapped keys to `found_keys`"""

    typ = type(o)
    if typ is tuple:
        if not o:
            return o
        if type(o[0]) is SubgraphCallable:

            # Unpack futures within the arguments of the subgraph callable
            futures: set[WrappedKey] = set()
            args = tuple(_unpack_remotedata_inner(i, byte_keys, futures) for i in o[1:])
            found_keys.update(futures)

            # Unpack futures within the subgraph callable itself
            sc: SubgraphCallable = o[0]
            futures = set()
            dsk = {
                k: _unpack_remotedata_inner(v, byte_keys, futures)
                for k, v in sc.dsk.items()
            }
            future_keys: tuple = ()
            if futures:  # If no futures is in the subgraph, we just use `sc` as-is
                found_keys.update(futures)
                future_keys = (
                    tuple(stringify(f.key) for f in futures)
                    if byte_keys
                    else tuple(f.key for f in futures)
                )
                inkeys = tuple(sc.inkeys) + future_keys
                sc = SubgraphCallable(dsk, sc.outkey, inkeys, sc.name)
            return (sc,) + args + future_keys
        else:
            return tuple(
                _unpack_remotedata_inner(item, byte_keys, found_keys) for item in o
            )
    elif is_namedtuple_instance(o):
        return _namedtuple_packing(
            o,
            partial(
                _unpack_remotedata_inner, byte_keys=byte_keys, found_keys=found_keys
            ),
        )

    if typ in collection_types:
        if not o:
            return o
        outs = [_unpack_remotedata_inner(item, byte_keys, found_keys) for item in o]
        return typ(outs)
    elif typ is dict:
        if o:
            return {
                k: _unpack_remotedata_inner(v, byte_keys, found_keys)
                for k, v in o.items()
            }
        else:
            return o
    elif issubclass(typ, WrappedKey):  # TODO use type is Future
        k = o.key
        if byte_keys:
            k = stringify(k)
        found_keys.add(o)
        return k
    else:
        return o


def unpack_remotedata(o: Any, byte_keys: bool = False) -> tuple[Any, set]:
    """Unpack WrappedKey objects from collection

    Returns original collection and set of all found WrappedKey objects

    Examples
    --------
    >>> rd = WrappedKey('mykey')
    >>> unpack_remotedata(1)
    (1, set())
    >>> unpack_remotedata(())
    ((), set())
    >>> unpack_remotedata(rd)
    ('mykey', {WrappedKey('mykey')})
    >>> unpack_remotedata([1, rd])
    ([1, 'mykey'], {WrappedKey('mykey')})
    >>> unpack_remotedata({1: rd})
    ({1: 'mykey'}, {WrappedKey('mykey')})
    >>> unpack_remotedata({1: [rd]})
    ({1: ['mykey']}, {WrappedKey('mykey')})

    Use the ``byte_keys=True`` keyword to force string keys

    >>> rd = WrappedKey(('x', 1))
    >>> unpack_remotedata(rd, byte_keys=True)
    ("('x', 1)", {WrappedKey('('x', 1)')})
    """
    found_keys: set[Any] = set()
    return _unpack_remotedata_inner(o, byte_keys, found_keys), found_keys


def pack_data(o, d, key_types=object):
    """Merge known data into tuple or dict

    Parameters
    ----------
    o
        core data structures containing literals and keys
    d : dict
        mapping of keys to data

    Examples
    --------
    >>> data = {'x': 1}
    >>> pack_data(('x', 'y'), data)
    (1, 'y')
    >>> pack_data({'a': 'x', 'b': 'y'}, data)  # doctest: +SKIP
    {'a': 1, 'b': 'y'}
    >>> pack_data({'a': ['x'], 'b': 'y'}, data)  # doctest: +SKIP
    {'a': [1], 'b': 'y'}
    """
    typ = type(o)
    try:
        if isinstance(o, key_types) and o in d:
            return d[o]
    except TypeError:
        pass

    if typ in collection_types:
        return typ([pack_data(x, d, key_types=key_types) for x in o])
    elif typ is dict:
        return {k: pack_data(v, d, key_types=key_types) for k, v in o.items()}
    elif is_namedtuple_instance(o):
        return _namedtuple_packing(o, partial(pack_data, d=d, key_types=key_types))
    else:
        return o


def subs_multiple(o, d):
    """Perform substitutions on a tasks

    Parameters
    ----------
    o
        Core data structures containing literals and keys
    d : dict
        Mapping of keys to values

    Examples
    --------
    >>> dsk = {"a": (sum, ["x", 2])}
    >>> data = {"x": 1}
    >>> subs_multiple(dsk, data)  # doctest: +SKIP
    {'a': (sum, [1, 2])}

    """
    typ = type(o)
    if typ is tuple and o and callable(o[0]):  # istask(o)
        return (o[0],) + tuple(subs_multiple(i, d) for i in o[1:])
    elif typ is list:
        return [subs_multiple(i, d) for i in o]
    elif typ is dict:
        return {k: subs_multiple(v, d) for (k, v) in o.items()}
    else:
        try:
            return d.get(o, o)
        except TypeError:
            return o


async def retry(
    coro,
    count,
    delay_min,
    delay_max,
    jitter_fraction=0.1,
    retry_on_exceptions=(EnvironmentError, IOError),
    operation=None,
):
    """
    Return the result of ``await coro()``, re-trying in case of exceptions

    The delay between attempts is ``delay_min * (2 ** i - 1)`` where ``i`` enumerates the attempt that just failed
    (starting at 0), but never larger than ``delay_max``.
    This yields no delay between the first and second attempt, then ``delay_min``, ``3 * delay_min``, etc.
    (The reason to re-try with no delay is that in most cases this is sufficient and will thus recover faster
    from a communication failure).

    Parameters
    ----------
    coro
        The coroutine function to call and await
    count
        The maximum number of re-tries before giving up. 0 means no re-try; must be >= 0.
    delay_min
        The base factor for the delay (in seconds); this is the first non-zero delay between re-tries.
    delay_max
        The maximum delay (in seconds) between consecutive re-tries (without jitter)
    jitter_fraction
        The maximum jitter to add to the delay, as fraction of the total delay. No jitter is added if this
        value is <= 0.
        Using a non-zero value here avoids "herd effects" of many operations re-tried at the same time
    retry_on_exceptions
        A tuple of exception classes to retry. Other exceptions are not caught and re-tried, but propagate immediately.
    operation
        A human-readable description of the operation attempted; used only for logging failures

    Returns
    -------
    Any
        Whatever `await coro()` returned
    """
    # this loop is a no-op in case max_retries<=0
    for i_try in range(count):
        try:
            return await coro()
        except retry_on_exceptions as ex:
            operation = operation or str(coro)
            logger.info(
                f"Retrying {operation} after exception in attempt {i_try}/{count}: {ex}"
            )
            delay = min(delay_min * (2**i_try - 1), delay_max)
            if jitter_fraction > 0:
                delay *= 1 + random.random() * jitter_fraction
            await asyncio.sleep(delay)
    return await coro()


async def retry_operation(coro, *args, operation=None, **kwargs):
    """
    Retry an operation using the configuration values for the retry parameters
    """

    retry_count = dask.config.get("distributed.comm.retry.count")
    retry_delay_min = parse_timedelta(
        dask.config.get("distributed.comm.retry.delay.min"), default="s"
    )
    retry_delay_max = parse_timedelta(
        dask.config.get("distributed.comm.retry.delay.max"), default="s"
    )
    return await retry(
        partial(coro, *args, **kwargs),
        count=retry_count,
        delay_min=retry_delay_min,
        delay_max=retry_delay_max,
        operation=operation,
    )

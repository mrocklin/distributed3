""" This file is experimental and may disappear without warning """
from __future__ import print_function, division, absolute_import

import logging
from math import log
import os
import io

from dask.imperative import Value
from tornado import gen
from toolz import merge

from .executor import default_executor
from .utils import ignoring


logger = logging.getLogger(__name__)


def get_block_locations(hdfs, filename):
    """ Get block locations from a filename or globstring """
    return [merge({'filename': fn}, block)
            for fn in hdfs.glob(filename)
            for block in hdfs.get_block_locations(fn)]


def read_bytes(fn, executor=None, hdfs=None, lazy=False, delimiter=None, **hdfs_auth):
    """ Convert location in HDFS to a list of distributed futures

    Parameters
    ----------
    fn: string
        location in HDFS
    executor: Executor (optional)
        defaults to most recently created executor
    hdfs: HDFileSystem
    **hdfs_auth: keyword arguments
        Extra keywords to send to ``hdfs3.HDFileSystem``

    Returns
    -------
    List of ``distributed.Future`` objects
    """
    from hdfs3 import HDFileSystem
    hdfs = hdfs or HDFileSystem(**hdfs_auth)
    executor = default_executor(executor)
    blocks = get_block_locations(hdfs, fn)
    filenames = [d['filename'] for d in blocks]
    offsets = [d['offset'] for d in blocks]
    lengths = [d['length'] for d in blocks]
    workers = [d['hosts'] for d in blocks]
    names = ['read-binary-%s-%d-%d' % (fn, offset, length)
            for fn, offset, length in zip(filenames, offsets, lengths)]

    logger.debug("Read %d blocks of binary bytes from %s", len(blocks), fn)
    if lazy:
        restrictions = dict(zip(names, workers))
        executor._send_to_scheduler({'op': 'update-graph',
                                    'dsk': {},
                                    'keys': [],
                                    'restrictions': restrictions,
                                    'loose_restrictions': set(names)})
        values = [Value(name, [{name: (hdfs.read_block, fn, offset, length, delimiter)}])
                  for name, fn, offset, length in zip(names, filenames, offsets, lengths)]
        return values
    else:
        return executor.map(hdfs.read_block, filenames, offsets, lengths,
                            delimiter=delimiter, workers=workers, allow_other_workers=True)


def buffer_to_csv(b, **kwargs):
    from io import BytesIO
    import pandas as pd
    bio = BytesIO(b)
    return pd.read_csv(bio, **kwargs)


@gen.coroutine
def _read_csv(fn, executor=None, hdfs=None, lazy=False, lineterminator='\n',
        header=True, names=None, **kwargs):
    from hdfs3 import HDFileSystem
    from dask import do
    import pandas as pd
    hdfs = hdfs or HDFileSystem()
    executor = default_executor(executor)
    kwargs['lineterminator'] = lineterminator
    filenames = hdfs.glob(fn)
    blockss = [read_bytes(fn, executor, hdfs, lazy=True, delimiter=lineterminator)
               for fn in filenames]
    if names is None and header:
        with hdfs.open(filenames[0]) as f:
            head = pd.read_csv(f, nrows=5, **kwargs)
            names = head.columns

    dfs1 = [[do(buffer_to_csv)(blocks[0], names=names, skiprows=1, **kwargs)] +
            [do(buffer_to_csv)(b, names=names, **kwargs) for b in blocks[1:]]
            for blocks in blockss]
    dfs2 = sum(dfs1, [])
    if lazy:
        from dask.dataframe import from_imperative
        raise gen.Return(from_imperative(dfs2, columns=names))
    else:
        futures = executor.compute(*dfs2)
        from distributed.collections import _futures_to_dask_dataframe
        df = yield _futures_to_dask_dataframe(futures)
        raise gen.Return(df)

def avro_header(f):
    import json
    import fastavro._reader as fa

    f.seek(0)
    header = fa.read_data(f, fa.HEADER_SCHEMA)
    metadata = dict((k, fa.btou(v)) for k, v in fa.iteritems(header['meta']))
    schema =  json.loads(metadata['avro.schema'])
    codec = metadata.get('avro.codec', 'null')
    
    fa.acquaint_schema(schema, fa.READERS)
    sync_marker = header['sync']
    
    read_block = fa.BLOCK_READERS.get(codec)
    if not read_block:
        raise ValueError('Unrecognized codec: %r' % codec)
    return schema, sync_marker, f.tell(), codec

def avro_body(data, schema, sync_marker, codec):
    import fastavro._reader as fa

    if not data.endswith(sync_marker):
        ## Read delimited should keep end-of-block delimiter
        data = data + sync_marker
    stream = io.BytesIO(data)
    block_count = 0
    read_block = fa.BLOCK_READERS.get(codec)
    while True:
        block_count = fa.read_long(stream, None)
        block_fo = read_block(stream)    
        for i in fa.xrange(block_count):
            yield fa.read_data(block_fo, schema, None)    
        fa.skip_sync(stream, sync_marker)

@gen.coroutine
def _read_avro(fn, executor=None, hdfs=None, lazy=False, **kwargs):
    from hdfs3 import HDFileSystem
    from dask import do
    import pandas as pd
    hdfs = hdfs or HDFileSystem()
    executor = default_executor(executor)
    filenames = hdfs.glob(fn)
    #blockss = [read_bytes(fn, executor, hdfs, lazy=True, delimiter=lineterminator)
    #           for fn in filenames]
    blockss = []
    for fn in filenames:
        f = hdfs.open(fn)
        schema, marker, offset, codec = avro_header(f)
        blocks = hdfs.get_block_locations(fn)
        offsets = [d['offset'] for d in blocks]
        offsets[0] = offset - len(marker)
        lengths = [d['length'] for d in blocks]
        workers = [d['hosts'] for d in blocks]
        names = ['read-binary-%s-%d-%d' % (fn, offset, length)
            for fn, offset, length in zip(filenames, offsets, lengths)]

        if lazy:
            restrictions = dict(zip(names, workers))
            executor._send_to_scheduler({'op': 'update-graph',
                                        'dsk': {},
                                        'keys': [],
                                        'restrictions': restrictions,
                                        'loose_restrictions': set(names)})
            blockss.extend([Value(name, [{name: (hdfs.read_block, fn, offset, length, marker)}])
                      for name, fn, offset, length in zip(names, filenames, offsets, lengths)])
        else:
            blockss.extend(executor.map(hdfs.read_block, filenames, offsets, lengths,
                                delimiter=marker, workers=workers, allow_other_workers=True))
    logger.debug("Read %d blocks of binary bytes from %s", len(blocks), fn)

    dfs1 = [do(avro_body)(b, schema, marker, codec) for b in blockss]
    return dfs1, blockss
    if lazy:
        from dask.dataframe import from_imperative
        raise gen.Return(from_imperative(dfs1, columns=names))
    else:
        futures = executor.compute(*dfs1)
        from distributed.collections import _futures_to_dask_dataframe
        df = yield _futures_to_dask_dataframe(futures)
        raise gen.Return(df)
    


def write_block_to_hdfs(fn, data, hdfs=None):
    """ Write bytes to HDFS """
    if not isinstance(data, bytes):
        raise TypeError("Data to write to HDFS must be of type bytes, got %s" %
                        type(data).__name__)
    with hdfs.open(fn, 'w') as f:
        f.write(data)
    return len(data)


def write_bytes(path, futures, executor=None, hdfs=None, **hdfs_auth):
    """ Write bytestring futures to HDFS

    Parameters
    ----------
    path: string
        Path on HDFS to write data.  Either globstring like ``/data/file.*.dat``
        or a directory name like ``/data`` (directory will be created)
    futures: list
        List of futures.  Each future should refer to a block of bytes.
    executor: Executor
    hdfs: HDFileSystem

    Returns
    -------
    Futures that wait until writing is complete.  Returns the number of bytes
    written.

    Examples
    --------

    >>> write_bytes('/data/file.*.dat', futures, hdfs=hdfs)  # doctest: +SKIP
    >>> write_bytes('/data/', futures, hdfs=hdfs)  # doctest: +SKIP
    """
    from hdfs3 import HDFileSystem
    hdfs = hdfs or HDFileSystem(**hdfs_auth)
    executor = default_executor(executor)

    n = len(futures)
    n_digits = int(log(n) / log(10))
    template = '%0' + str(n_digits) + 'd'

    if '*' in path:
        dirname = os.path.split(path)[0]
        hdfs.mkdir(dirname)
        filenames = [path.replace('*', template % i) for i in range(n)]
    else:
        hdfs.mkdir(path)
        filenames = [os.path.join(path, template % i) for i in range(n)]

    return executor.map(write_block_to_hdfs, filenames, futures, hdfs=hdfs)

from __future__ import print_function, division, absolute_import, unicode_literals

import logging

from tornado import gen

from .plugin import SchedulerPlugin

from ..compatibility import unicode as str
from ..core import connect, coerce_to_address
from ..worker import dumps_function


logger = logging.getLogger(__name__)


class EventStream(SchedulerPlugin):
    """ Maintain a copy of worker events """
    def __init__(self, scheduler=None):
        self.buffer = []
        if scheduler:
            scheduler.add_plugin(self)

    def transition(self, key, start, finish, *args, **kwargs):
        if start == 'processing':
            kwargs['key'] = key
            if finish == 'memory' or finish == 'erred':
                self.buffer.append(kwargs)


def swap_buffer(scheduler, es):
    es.buffer, buffer = [], es.buffer
    return buffer


def teardown(scheduler, es):
    scheduler.remove_plugin(es)


@gen.coroutine
def eventstream(address, interval):
    """ Open a TCP connection to scheduler, receive batched task messages

    The messages coming back are lists of dicts.  Each dict is of the following
    form::

        {'key': 'mykey', 'worker': 'host:port', 'status': status,
         'compute_start': time(), 'compute_stop': time(),
         'transfer_start': time(), 'transfer_stop': time(),
         'disk_load_start': time(), 'disk_load_stop': time(),
         'other': 'junk'}

    Where ``status`` is either 'OK', or 'error'

    Parameters
    ----------
    address: address of scheduler
    interval: time between batches, in seconds

    Examples
    --------
    >>> stream = yield eventstream('127.0.0.1:8786', 0.100)  # doctest: +SKIP
    >>> print(yield read(stream))  # doctest: +SKIP
    [{'key': 'x', 'status': 'OK', 'worker': '192.168.0.1:54684', ...},
     {'key': 'y', 'status': 'error', 'worker': '192.168.0.1:54684', ...}]
    """
    address = coerce_to_address(address)
    comm = yield connect(address)
    yield comm.write({'op': 'feed',
                      'setup': dumps_function(EventStream),
                      'function': dumps_function(swap_buffer),
                      'interval': interval,
                      'teardown': dumps_function(teardown)})
    raise gen.Return(comm)

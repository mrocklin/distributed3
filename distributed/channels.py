from __future__ import print_function, division, absolute_import

from collections import deque, defaultdict
from functools import partial
import logging
from time import sleep
import threading

from tornado.iostream import StreamClosedError

from .client import Future
from .utils import tokey, log_errors

logger = logging.getLogger(__name__)


class ChannelScheduler(object):
    """ A plugin for the scheduler to manage channels

    This adds the following routes to the scheduler

    *  channel-subscribe
    *  channel-unsubsribe
    *  channel-append
    """
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.deques = dict()
        self.counts = dict()
        self.clients = dict()

        handlers = {'channel-subscribe': self.subscribe,
                    'channel-unsubscribe': self.unsubscribe,
                    'channel-append': self.append}

        self.scheduler.compute_handlers.update(handlers)

    def subscribe(self, channel=None, client=None, maxlen=None):
        logger.info("Add new client to channel, %s, %s", client, channel)
        if channel not in self.deques:
            logger.info("Add new channel %s", channel)
            self.deques[channel] = deque(maxlen=maxlen)
            self.counts[channel] = 0
            self.clients[channel] = set()
        self.clients[channel].add(client)

        stream = self.scheduler.streams[client]
        for key in self.deques[channel]:
            stream.send({'op': 'channel-append',
                         'key': key,
                         'channel': channel})

    def unsubscribe(self, channel=None, client=None):
        logger.info("Remove client from channel, %s, %s", client, channel)
        self.clients[channel].remove(client)
        if not self.clients[channel]:
            del self.deques[channel]
            del self.counts[channel]
            del self.clients[channel]

    def append(self, channel=None, key=None):
        if len(self.deques[channel]) == self.deques[channel].maxlen:
            self.scheduler.client_releases_keys(keys=[self.deques[channel][0]],
                                                client='streaming-%s' % channel)

        self.deques[channel].append(key)
        self.counts[channel] += 1
        self.report(channel, key)

        client='streaming-%s' % channel
        self.scheduler.client_desires_keys(keys=[key], client=client)

    def report(self, channel, key):
        for client in list(self.clients[channel]):
            try:
                stream = self.scheduler.streams[client]
                stream.send({'op': 'channel-append',
                             'key': key,
                             'channel': channel})
            except (KeyError, StreamClosedError):
                self.unsubscribe(channel, client)


class ChannelClient(object):
    def __init__(self, client):
        self.client = client
        self.channels = dict()
        self.client._channel_handler = self

        handlers = {'channel-append': self.receive_key}

        self.client._handlers.update(handlers)

        self.client.channel = self._create_channel  # monkey patch

    def _create_channel(self, channel, maxlen=None):
        if channel not in self.channels:
            c = Channel(self.client, channel, maxlen=maxlen)
            self.channels[channel] = c
            return c
        else:
            return self.channels[channel]

    def receive_key(self, channel=None, key=None):
        self.channels[channel]._receive_update(key)


class Channel(object):
    def __init__(self, client, name, maxlen=None):
        self.client = client
        self.name = name
        self.futures = deque(maxlen=maxlen)
        self.count = 0
        self._pending = dict()
        self._thread_condition = threading.Condition()

        self.client._send_to_scheduler({'op': 'channel-subscribe',
                                        'channel': name,
                                        'maxlen': maxlen,
                                        'client': self.client.id})

    def append(self, future):
        self.client._send_to_scheduler({'op': 'channel-append',
                                        'channel': self.name,
                                        'key': tokey(future.key)})
        self._pending[future.key] = future  # hold on to reference until ack

    def _receive_update(self, key=None):
        self.count += 1
        self.futures.append(Future(key, self.client))
        self.client._send_to_scheduler({'op': 'client-desires-keys',
                                        'keys': [key],
                                        'client': self.client.id})
        if key in self._pending:
            del self._pending[key]

        with self._thread_condition:
            self._thread_condition.notify_all()

    def flush(self):
        while self._pending:
            sleep(0.01)

    def __del__(self):
        if not self.client.scheduler_stream.stream:
            self.client._send_to_scheduler({'op': 'channel-unsubscribe',
                                            'channel': self.name,
                                            'client': self.client.id})

    def __iter__(self):
        with log_errors():
            last = self.count
            L = list(self.futures)
            for future in L:
                yield future

            while True:
                if self.count == last:
                    self._thread_condition.acquire()
                    self._thread_condition.wait()
                    self._thread_condition.release()

                n = min(self.count - last, len(self.futures))
                L = [self.futures[i] for i in range(-n, 0)]
                last = self.count
                for f in L:
                    yield f


    def __len__(self):
        return len(self.futures)

    def __str__(self):
        return "<Channel: %s - %d elements>" % (self.name, len(self.futures))

    __repr__ = __str__

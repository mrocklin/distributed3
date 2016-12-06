from collections import deque, defaultdict

from .client import Future
from .utils import tokey


class BufferScheduler(object):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.deques = dict()
        self.counts = dict()
        self.clients = dict()

        handlers = {'topic-subscribe': self.subscribe,
                    'topic-unsubscribe': self.unsubscribe,
                    'topic-append': self.append}

        self.scheduler.compute_handlers.update(handlers)

    def subscribe(self, topic=None, client=None, maxlen=None):
        if topic not in self.deques:
            self.deques[topic] = deque(maxlen=maxlen)
            self.counts[topic] = 0
            self.clients[topic] = set()
        self.clients[topic].add(client)

        stream = self.scheduler.streams[client]
        for key in self.deques[topic]:
            stream.send({'op': 'topic-append',
                         'key': key,
                         'topic': topic})

    def unsubscribe(self, topic=None, client=None):
        self.clients[topic].remove(client)
        if self.clients[topic]:
            del self.deques[topic]
            del self.counts[topic]
            del self.clients[topic]

    def append(self, topic=None, key=None):
        self.deques[topic].append(topic)
        self.counts[topic] += 1
        self.report(topic, key)

    def report(self, topic, key):
        for c in list(self.clients[topic]):
            stream = self.scheduler.streams[c]
            try:
                stream.send({'op': 'topic-append',
                             'key': key,
                             'topic': topic})
            except StreamClosedError:
                self.unsubscribe(topic, client)


class BufferClient(object):
    def __init__(self, client):
        self.client = client
        self.buffers = dict()
        self.client.buffers = self

        handlers = {'topic-append': self.receive_key}

        self.client._handlers.update(handlers)

    def receive_key(self, topic=None, key=None):
        for buff in self.buffers[topic]:
            buff._receive_update(key)

    def add_buffer(self, buffer):
        if buffer.topic not in self.buffers:
            self.buffers[buffer.topic] = {buffer}
        else:
            self.buffers[buffer.topic].add(buffer)


class Buffer(object):
    def __init__(self, client, topic, maxlen=None):
        self.client = client
        self.topic = topic
        self.futures = deque(maxlen=maxlen)
        self.client.buffers.add_buffer(self)  # circular reference

        self.client._send_to_scheduler({'op': 'topic-subscribe',
                                        'topic': topic,
                                        'maxlen': maxlen,
                                        'client': self.client.id})

    def append(self, future):
        self.client._send_to_scheduler({'op': 'topic-append',
                                        'topic': self.topic,
                                        'key': tokey(future.key)})

    def _receive_update(self, key=None):
        self.futures.append(Future(key, self.client))

    def __del__(self):
        self.client._send_to_scheduler({'op': 'topic-unsubscribe',
                                        'topic': self.topic,
                                        'client': self.client.id})

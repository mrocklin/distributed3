from __future__ import print_function, division, absolute_import

from collections import deque
import psutil
from time import time

from .compatibility import WINDOWS


class SystemMonitor(object):
    def __init__(self, n=1000):
        self.proc = psutil.Process()

        self.time = deque(maxlen=n)
        self.cpu = deque(maxlen=n)
        self.memory = deque(maxlen=n)
        self.read_bytes = deque(maxlen=n)
        self.write_bytes = deque(maxlen=n)

        self.last_time = time()
        self.count = 0

        self._last_io_counters = self.proc.io_counters()
        self.quantities = {'cpu': self.cpu,
                           'memory': self.memory,
                           'time': self.time,
                           'read_bytes': self.read_bytes,
                           'write_bytes': self.write_bytes}
        if not WINDOWS:
            self.num_fds = deque(maxlen=n)
            self.quantities['num_fds'] = self.num_fds

    def update(self):
        cpu = self.proc.cpu_percent()
        memory = self.proc.memory_info().rss

        now = time()
        ioc = self.proc.io_counters()
        last = self._last_io_counters
        read_bytes = (ioc.read_bytes - last.read_bytes) / (now - self.last_time)
        write_bytes = (ioc.write_bytes - last.write_bytes) / (now - self.last_time)
        self._last_io_counters = ioc
        self.last_time = now

        self.cpu.append(cpu)
        self.memory.append(memory)
        self.time.append(now)

        self.read_bytes.append(read_bytes)
        self.write_bytes.append(write_bytes)

        self.count += 1

        result = {'cpu': cpu, 'memory': memory, 'time': now,
                  'count': self.count, 'read_bytes': read_bytes,
                  'write_bytes': write_bytes}

        if not WINDOWS:
            num_fds = self.proc.num_fds()
            self.num_fds.append(num_fds)
            result['num_fds'] = num_fds

        return result

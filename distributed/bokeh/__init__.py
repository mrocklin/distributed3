from collections import deque

messages = {}

n = 60
messages['workers'] = {'interval': 1000,
                       'deque': deque(maxlen=n),
                       'times': deque(maxlen=n),
                       'index': deque(maxlen=n),
                       'plot-data': {'time': deque(maxlen=n),
                                     'cpu': deque(maxlen=n),
                                     'memory_percent': deque(maxlen=n),
                                     'network-send': deque(maxlen=n),
                                     'network-recv': deque(maxlen=n)}}

messages['tasks'] = {'interval': 150,
                     'deque': deque(maxlen=100),
                     'times': deque(maxlen=100)}

messages['progress'] = {}

messages['processing'] = {'stacks': {}, 'processing': {},
                          'memory': 0, 'waiting': 0, 'ready': 0}


Logging
=======

.. currentmodule:: distributed

There are several ways in which state and other activities are logged throughout
a Dask cluster.


Logs
----

The scheduler, workers, and client all log various administrative events using Python's standard
logging module. Both the logging level and logging handlers are customizable.
See the `Debugging docs <https://docs.dask.org/en/latest/debugging.html#logs>`_ for more information.


Task transition logs
--------------------

The scheduler keeps track of all :ref:`state transitions <scheduler-task-state>` for each task.
This gives insight into how tasks progressed through their computation and can be particularly
valuable when debugging.
To retrieve the transition logs for a given task, pass the task's key to the :meth:`Scheduler.story` method.

.. code-block:: Python

    >>> f = client.submit(inc, 123)
    >>> f
    <Future: finished, type: builtins.int, key: inc-aad7bbea25dc61c8e53d929c7ec50bed>
    >>> s.story(f.key)
    [('inc-aad7bbea25dc61c8e53d929c7ec50bed', 'released', 'waiting', {'inc-aad7bbea25dc61c8e53d929c7ec50bed': 'processing'}, 1605143345.7283862),
     ('inc-aad7bbea25dc61c8e53d929c7ec50bed', 'waiting', 'processing', {}, 1605143345.7284858),
     ('inc-aad7bbea25dc61c8e53d929c7ec50bed', 'processing', 'memory', {}, 1605143345.731495)]


Structured logs
---------------

The scheduler, workers, and client all support logging structured events to a centralized ledger,
which is indexed by topic. By default, Dask will log a few administrative events to this system
(e.g. when workers enter and leave the cluster) but custom events can be logged using the
:meth:`Scheduler.log_event`, :meth:`Worker.log_event`, or :meth:`Client.log_event` methods.

For example, below we log start and stop times to the ``"runtimes"`` topic using the worker's
``log_event`` method:

.. code-block:: python

    >>> def myfunc(x):
    ...     start = time()
    ...     ...
    ...     stop = time()
    ...     dask.distributed.get_worker().log_event("runtimes", {"start": start, "stop": stop})
    >>> futures = client.map(myfunc, range(10))
    >>> client.get_events("runtimes")
    ((1605207481.77175, {'start': 1605207481.769397, 'stop': 1605207481.769397}),
     (1605207481.772021, {'start': 1605207481.770036, 'stop': 1605207481.770037}),
     ...
    )

Events for a given topic can be retrieved using the :meth:`Client.get_events` method.
In the above example, we retrieved the logged start and stop times with
``client.get_events("runtimes")``. Note that ``Client.get_events`` returns a tuple for
each logged event which contains the logged message along with a timestamp for when
the event was logged.

When combined with scheduler and worker plugins, the structured events system can produce
rich logging / diagnostic systems.

It is possible to feed stdlib logging records into this system as well using the options in ``distributed.log-events``. The logs are collected in a dedicated topic per log level, e.g. ``ERROR`` and a global ``logs`` topic.

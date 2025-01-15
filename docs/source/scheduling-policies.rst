Scheduling Policies
===================

This document describes the policies used to select the preference of tasks and
to select the preference of workers used by Dask's distributed scheduler.  For
more information on how this these policies are enacted efficiently see
:doc:`Scheduling State<scheduling-state>`.


.. _decide-worker:

Choosing Workers
----------------

When a task transitions from waiting to a processing state, we decide a suitable
worker for that task. If the task has significant data dependencies or if the
workers are under heavy load, then this choice of worker can strongly impact
global performance. Similarly, the placement of root tasks affects performance
of downstream computations, since it can determine how much data will need to be
transferred between workers in the future. Different heuristics are used for these
different scenarios:

Initial Task Placement - queuing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When :ref:`queuing <queuing>` is enabled (the default), each initial task is simply
scheduled on the least-busy worker at the moment. If an initial task wants to run, but
all worker threads are occupied, then the task instead goes onto (or stays on) the
queue and is not sent to any worker.

.. _co-assignment:

Initial Task Placement - no queuing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Currently, this scheduling is only used when :ref:`queuing <queuing>` is disabled
(the ``distributed.scheduler.worker-saturation`` config value is set to ``inf``).

We want neighboring root tasks to run on the same worker, since there's a
good chance those neighbors will be combined in a downstream operation::

      i       j
     / \     / \
    e   f   g   h
    |   |   |   |
    a   b   c   d
    \   \  /   /
         X

In the above case, we want ``a`` and ``b`` to run on the same worker,
and ``c`` and ``d`` to run on the same worker, reducing future
data transfer. We can also ignore the location of ``X``, because assuming
we split the ``a b c d`` group across all workers to maximize parallelism,
then ``X`` will eventually get transferred everywhere.
(Note that wanting to co-locate ``a b`` and ``c d`` would still apply even if
``X`` didn't exist.)

Calculating these cousin tasks directly by traversing the graph would be expensive.
Instead, we use the task's TaskGroup, which is the collection of all tasks with the
same key prefix. (``(random-a1b2c3, 0)``, ``(random-a1b2c3, 1)``, ``(random-a1b2c3, 2)``
would all belong to the TaskGroup ``random-a1b2c3``.)

To identify the root(ish) tasks, we check whether the task has been explicitly marked as
(non-)rootish. If this is not the case, we use this heuristic:

1.  The TaskGroup has 2x more tasks than there are threads in the cluster
2.  The TaskGroup has fewer than 5 unique dependencies across *all* tasks in the group.

    We don't just say "The task has no dependencies", because real-world cases like
    :obj:`dask.array.from_zarr` and :obj:`dask.array.from_array` produce graphs like the one
    above, where the data-creation tasks (``a b c d``) all share one dependency
    (``X``)---the Zarr dataset, for example. Though ``a b c d`` are not technically
    root tasks, we want to treat them as such, hence allowing a small number of trivial
    dependencies shard by all tasks.

Then, we use the same priority described in :ref:`priority-break-ties` to
determine which tasks are related. This depth-first-with-child-weights metric
can usually be used to properly segment the leaves of a graph into decently
well-separated sub-graphs with relatively low inter-sub-graph connectedness.

Iterating through tasks in this priority order, we assign a batch of subsequent tasks
to a worker, then select a new worker (the least-busy one) and repeat.

Though this does not provide perfect initial task assignment (a handful of sibling
tasks may be split across workers), it does well in most cases, while adding
minimal scheduling overhead.

Initial task placement is a forward-looking decision. By colocating related root tasks,
we ensure that their downstream tasks are set up for success.

Downstream Task Placement
~~~~~~~~~~~~~~~~~~~~~~~~~

When initial tasks are well-placed, placing subsequent tasks is backwards-looking:
where can the task run the soonest, considering both data transfer and worker busyness?

Tasks that don't meet the root-ish criteria described above are selected as follows:

First, we identify the pool of viable workers:

1.  If the task has no dependencies and no restrictions, then we find the
    least-occupied worker.
2.  Otherwise, if a task has user-provided restrictions (for example it must
    run on a machine with a GPU) then we restrict the available pool of workers
    to just that set. Otherwise, we consider all workers.
3.  We restrict the above set to just workers that hold at least one dependency
    of the task.

From among this pool of workers, we then determine the worker where we think the task will
start running the soonest, using :meth:`Scheduler.worker_objective`. For each worker:

1.  We consider the estimated runtime of other tasks already queued on that worker.
    Then, we add how long it will take to transfer any dependencies to that worker that
    it doesn't already have, based on their size, in bytes, and the measured network
    bandwidth between workers. Note that this does *not* consider (de)serialization
    time, time to retrieve the data from disk if it was spilled, or potential differences
    between size in memory and serialized size. In practice, the
    queue-wait-time (known as *occupancy*) usually dominates, so data will usually be
    transferred to a different worker if it means the task can start any sooner.
2.  It's possible for ties to occur with the "start soonest" metric, though uncommon
    when all workers are busy. We break ties by choosing the worker that has the
    fewest number of bytes of Dask data stored (including spilled data). Note that
    this is the same as :ref:`managed <memtypes>` plus :ref:`spilled <memtypes>`
    memory, not the :ref:`process <memtypes>` memory.

This process is easy to change (and indeed this document may be outdated).  We
encourage readers to inspect the ``decide_worker`` and ``worker_objective``
functions in ``scheduler.py``.

.. currentmodule:: distributed.scheduler

.. autosummary:: decide_worker
.. autosummary:: Scheduler.worker_objective


Choosing Tasks
--------------

We often have a choice between running many valid tasks.  There are a few
competing interests that might motivate our choice:

1.  Run tasks on a first-come-first-served basis for fairness between
    multiple clients
2.  Run tasks that are part of the critical path in an effort to
    reduce total running time and minimize straggler workloads
3.  Run tasks that allow us to release many dependencies in an effort to keep
    the memory footprint small
4.  Run tasks that are related so that large chunks of work can be completely
    eliminated before running new chunks of work
5.  Run tasks that use existing work before starting tasks that create new work

Accomplishing all of these objectives simultaneously is impossible.  Optimizing
for any of these objectives perfectly can result in costly overhead.  The
heuristics with the scheduler do a decent but imperfect job of optimizing for
all of these (they all come up in important workloads) quickly.

Last in, first out
~~~~~~~~~~~~~~~~~~

When a worker finishes a task, the immediate dependencies of that task get top
priority.  This encourages a behavior of finishing ongoing work immediately
before starting new work (depth-first graph traversal). This often conflicts with
the first-come-first-served objective, but often results in significantly reduced
memory footprints and, due to avoiding data spillage to disk, better overall runtimes.

.. _priority-break-ties:

Break ties with children and depth
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Often a task has multiple dependencies and we need to break ties between them
with some other objective.  Breaking these ties has a surprisingly strong
impact on performance and memory footprint.

When a client submits a graph we perform a few linear scans over the graph to
determine something like the number of descendants of each node (not quite,
because it's a DAG rather than a tree, but this is a close proxy).  This number
can be used to break ties and helps us to prioritize nodes with longer critical
paths and nodes with many children.  The actual algorithms used are somewhat
more complex and are described in detail in `dask/order.py`_

.. _`dask/order.py`: https://github.com/dask/dask/blob/main/dask/order.py

First-Come-First-Served, Coarsely
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The last-in-first-out behavior used by the workers to minimize memory footprint
can distort the task order provided by the clients.  Tasks submitted recently
may run sooner than tasks submitted long ago because they happen to be more
convenient given the current data in memory.  This behavior can be *unfair* but
improves global runtimes and system efficiency, sometimes quite significantly.

However, workers inevitably run out of tasks that were related to tasks they
were just working on and the last-in-first-out policy eventually exhausts
itself.  In these cases workers often pull tasks from the common task pool.
The tasks in this pool *are* ordered in a first-come-first-served basis and so
workers do behave in a scheduling manner that's fair to multiple submissions
at a *coarse* level, if not a fine-grained one.

Dask's scheduling policies are short-term-efficient and long-term-fair
to multiple clients.

.. _queuing:

Avoid over-saturating workers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When there are many initial tasks to run, workers don't need to know about all of them
up front::

     o   o   o   o   o   o   o   o   o   o
    / \ / \ / \ / \ / \ / \ / \ / \ / \ / \
    o o o o o o o o o o o o o o o o o o o o
    | | | | | | | | | | | | | | | | | | | |
    * * * * * * * * * * * * * * * * * * * *  <-- initial tasks

The scheduler only submits initial tasks (``*`` tasks in the figure above) to workers
until all worker threads are filled up [#]_. The remaining initial tasks are put in a queue
on the scheduler, ordered by priority.

Tasks are popped off this queue and scheduled whenever a thread opens up on a worker
*and* there are no other higher-priority tasks (``o`` tasks in this diagram) that could
run instead.

This ensures we finish existing streams of work before starting on new work. This keeps
memory use as low as possible and generally gives much more stable execution compared to
submitting all initial tasks at once.

There are two downsides to this queueing:

1. Initial tasks are not :ref:`co-assigned <co-assignment>`. This means that workers may
   have to make data transfers which could have been avoided. This can cause a moderate
   slow-down on some workloads compared to disabling queuing. However, in many of those
   cases, disabling queuing might cause workers to run out of memory, so the slow-down
   is usually a better tradeoff.
2. For embarrassingly-parallel workloads like a ``client.map``, there can be a minor
   increase in overhead per task, because each time a task finishes, a
   scheduler<->worker roundtrip message is required before the next task starts. In most
   cases, this overhead is not even measurable and not something to worry about.

   This will only matter if you have very fast tasks, or a very slow network—that is, if
   your task runtime is the same order of magnitude as your network latency. For
   example, if each task only takes 1ms, and a scheduler<->worker roundtrip message
   takes 10ms, all those roundtrip messages will dominate the runtime.

   This means you should make your tasks bigger (via larger chunksizes, or batching more
   work into single Dask tasks). In general, task runtime should be significantly larger
   than network latency for Dask to perform well.

.. [#] By default, it will actually submit slightly more tasks than threads per worker
       (for example, 1 extra task for workers with <= 10 threads). This slight buffering
       maintains better performance when tasks are very fast. See next section for
       details.

.. _adjust-queuing:

Adjusting or disabling queuing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*It's rare to need to adjust queuing. The default value works well for almost all cases.
*Only advanced users wanting to tune performance in unusual cases might consider
*adjusting this parameter.*

Queuing behavior is controlled by the ``distributed.scheduler.worker-saturation`` config
value. This is set via the :doc:`Dask configuration system <configuration>`. The config
value must be set on the scheduler, before the scheduler starts.

The value controls how many initial chunks of data workers will have in memory at once.
This is basically the "breadth" of execution through the graph. Specifically, up to
``ceil(worker-saturation * nthreads)`` initial tasks are sent to a worker at a time.

By default, ``worker-saturation`` is ``1.1``. This value was chosen to keep worker
memory relatively low (workers with <= 10 threads will only get 1 extra initial chunk in
memory each), while mitigating the effects of the extra latency for users running on
very slow networks.

* If workers are running out of memory, consider setting ``worker-saturation`` to
  ``1.0`` instead of ``1.1``.
* If your network is very slow, or your tasks are extremely fast, and you want to
  decrease runtime, consider increasing ``worker-saturation``. This *may* speed things
  up slightly, at the cost of increased memory use. Values above ``2.0`` usually have
  little benefit.
* If your graph would benefit from :ref:`co-assignment <co-assignment>`, and you have
  plenty of memory on the cluster, consider disabling queueing by setting
  ``worker-saturation`` to ``inf`` to speed up runtime.

Where these decisions are made
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The objectives above are mostly followed by small decisions made by the client,
scheduler, and workers at various points in the computation.

1.  As we submit a graph from the client to the scheduler we assign a numeric
    priority to each task of that graph.  This priority focuses on
    computing deeply before broadly, preferring critical paths, preferring
    nodes with many dependencies, etc..  This is the same logic used by the
    single-machine scheduler and lives in `dask/order.py
    <https://github.com/dask/dask/blob/main/dask/order.py>`_.
2.  When the graph reaches the scheduler the scheduler changes each of these
    numeric priorities into a tuple of two numbers, the first of which is an
    increasing counter, the second of which is the client-generated priority
    described above.  This per-graph counter encourages a first-in-first-out
    policy between computations.  All tasks from a previous call to compute
    have a higher priority than all tasks from a subsequent call to compute (or
    submit, persist, map, or any operation that generates futures).
3.  Whenever a task is ready to run (its dependencies, if any, are complete),
    the scheduler assigns it to a worker. When multiple tasks are ready at once,
    they are submitted to workers, in priority order. If scheduler-side queuing
    is active, they are submitted until all workers are full, then any leftover
    runnable tasks are put in the scheduler queue. If queuing is disabled, then
    all runnable tasks are submitted at once.
4.  However, when the worker receives these tasks, it considers their priorities
    when determining which tasks to prioritize for fetching data or for
    computation.  The worker maintains a heap of all ready-to-run tasks ordered
    by this priority.
5.  If scheduler-side queuing is active: when any task completes on a worker,
    if there are no other higher-priority tasks to run, the scheduler pops off
    the next queued task and runs it on that worker.

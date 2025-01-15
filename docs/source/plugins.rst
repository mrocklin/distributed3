Plugins
~~~~~~~

Dask's plugin system enables you to run custom Python code for certain events. You can use plugins
that are specific to schedulers, workers, or nannies. A worker plugin, for example,
allows you to run custom Python code on all your workers at certain event in the worker's lifecycle (e.g. when the worker process is started).
In each section below, you'll see how to create your own plugin or use a Dask-provided built-in
plugin.

Scheduler Plugins
=================

.. autoclass:: distributed.diagnostics.plugin.SchedulerPlugin
   :members:


RabbitMQ Example
----------------

RabbitMQ is a distributed messaging queue that we can use to post updates about
task transitions. By posting transitions to RabbitMQ, we allow other machines
to do the processing of transitions and keep scheduler processing to a minimum.
See the
`RabbitMQ tutorial <https://www.rabbitmq.com/tutorials/tutorial-two-python.html>`_
for more information on RabbitMQ and how to consume the messages.

.. code-block:: python

   import json
   from distributed.diagnostics.plugin import SchedulerPlugin
   import pika

   class RabbitMQPlugin(SchedulerPlugin):
       def __init__(self):
           # Update host to be your RabbitMQ host
           self.connection = pika.BlockingConnection(
               pika.ConnectionParameters(host='localhost'))
           self.channel = self.connection.channel()
           self.channel.queue_declare(queue='dask_task_status', durable=True)

       def transition(self, key, start, finish, *args, **kwargs):
           message = dict(
               key=key,
               start=start,
               finish=finish,
           )
           self.channel.basic_publish(
               exchange='',
               routing_key='dask_task_status',
               body=json.dumps(message),
               properties=pika.BasicProperties(
                   delivery_mode=2,  # make message persistent
               ))

   @click.command()
   def dask_setup(scheduler):
       plugin = RabbitMQPlugin()
       scheduler.add_plugin(plugin)

Run with: ``dask scheduler --preload <filename.py>``

Accessing Full Task State
-------------------------

If you would like to access the full :class:`distributed.scheduler.TaskState`
stored in the scheduler you can do this by passing and storing a reference to
the scheduler. This can be done in one of two ways.

Firstly, you can use the ``dask_setup`` function and run the scheduler with
``dask scheduler --preload <filename.py>``:

.. code-block:: python

   import click
   from distributed.diagnostics.plugin import SchedulerPlugin

   class MyPlugin(SchedulerPlugin):
       def __init__(self, scheduler):
            self.scheduler = scheduler

       def transition(self, key, start, finish, *args, **kwargs):
            # Get full TaskState
            ts = self.scheduler.tasks[key]

   @click.command()
   def dask_setup(scheduler):
       plugin = MyPlugin(scheduler)
       scheduler.add_plugin(plugin)

You can also use the :meth:`distributed.diagnostics.plugin.SchedulerPlugin.start` method, which will receive a reference to the scheduler:

.. code-block:: python

    from distributed.diagnostics.plugin import SchedulerPlugin

    class MyPlugin(SchedulerPlugin):
        def __init__(self):
            self.scheduler = None

        def start(self, scheduler):
            self.scheduler = scheduler

        def transition(self, key, start, finish, *args, **kwargs):
            # Get full TaskState
            ts = self.scheduler.tasks[key]

    if __name__ == '__main__':
        # Create and register the plugin in your main code
        from distributed import Client, LocalCluster

        cluster = LocalCluster()
        client = Client(cluster)

        plugin = MyPlugin()
        client.register_plugin(plugin)


Built-In Scheduler Plugins
--------------------------

.. autoclass:: distributed.diagnostics.plugin.PipInstall
.. autoclass:: distributed.diagnostics.plugin.CondaInstall
.. autoclass:: distributed.diagnostics.plugin.InstallPlugin
.. autoclass:: distributed.diagnostics.plugin.SchedulerUploadFile

Worker Plugins
==============

:class:`distributed.diagnostics.plugin.WorkerPlugin` provides a base class
for creating your own worker plugins. In addition, Dask provides some
:ref:`built-in plugins <plugins.builtin>`.

Watch the video below for an example using a ``WorkerPlugin`` to add a
:py:class:`concurrent.futures.ProcessPoolExecutor`:

.. raw:: html

    <iframe width="560"
            height="315" src="https://www.youtube.com/embed/vF2VItVU5zg?start=468"
            style="margin: 0 auto 20px auto; display: block;" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

.. autoclass:: distributed.diagnostics.plugin.WorkerPlugin
   :members:

.. _plugins.builtin:

Built-In Worker Plugins
-----------------------

.. autoclass:: distributed.diagnostics.plugin.UploadFile


Nanny Plugins
=============

.. autoclass:: distributed.diagnostics.plugin.NannyPlugin
   :members:


Built-In Nanny Plugins
----------------------

.. autoclass:: distributed.diagnostics.plugin.Environ
.. autoclass:: distributed.diagnostics.plugin.UploadDirectory

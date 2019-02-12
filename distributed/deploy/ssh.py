from __future__ import print_function, division, absolute_import

import logging
import socket
import os
import sys
import time
import traceback

try:
    from queue import Queue
except ImportError:  # Python 2.7 fix
    from Queue import Queue

from threading import Thread

from toolz import merge

from tornado import gen


logger = logging.getLogger(__name__)


# These are handy for creating colorful terminal output to enhance readability
# of the output generated by dask-ssh.
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def async_ssh(cmd_dict):
    import paramiko
    from paramiko.buffered_pipe import PipeTimeout
    from paramiko.ssh_exception import (SSHException, PasswordRequiredException)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    retries = 0
    while True:  # Be robust to transient SSH failures.
        try:
            # Set paramiko logging to WARN or higher to squelch INFO messages.
            logging.getLogger('paramiko').setLevel(logging.WARN)

            ssh.connect(hostname=cmd_dict['address'],
                        username=cmd_dict['ssh_username'],
                        port=cmd_dict['ssh_port'],
                        key_filename=cmd_dict['ssh_private_key'],
                        compress=True,
                        timeout=20,
                        banner_timeout=20)  # Helps prevent timeouts when many concurrent ssh connections are opened.
            # Connection successful, break out of while loop
            break

        except (SSHException,
                PasswordRequiredException) as e:

            print('[ dask-ssh ] : ' + bcolors.FAIL +
                  'SSH connection error when connecting to {addr}:{port}'
                  'to run \'{cmd}\''.format(addr=cmd_dict['address'],
                                            port=cmd_dict['ssh_port'],
                                            cmd=cmd_dict['cmd']) + bcolors.ENDC)

            print(bcolors.FAIL + '               SSH reported this exception: ' + str(e) + bcolors.ENDC)

            # Print an exception traceback
            traceback.print_exc()

            # Transient SSH errors can occur when many SSH connections are
            # simultaneously opened to the same server. This makes a few
            # attempts to retry.
            retries += 1
            if retries >= 3:
                print('[ dask-ssh ] : '
                      + bcolors.FAIL
                      + 'SSH connection failed after 3 retries. Exiting.'
                      + bcolors.ENDC)

                # Connection failed after multiple attempts.  Terminate this thread.
                os._exit(1)

            # Wait a moment before retrying
            print('               ' + bcolors.FAIL +
                  'Retrying... (attempt {n}/{total})'.format(n=retries, total=3) +
                  bcolors.ENDC)

            time.sleep(1)

    # Execute the command, and grab file handles for stdout and stderr. Note
    # that we run the command using the user's default shell, but force it to
    # run in an interactive login shell, which hopefully ensures that all of the
    # user's normal environment variables (via the dot files) have been loaded
    # before the command is run. This should help to ensure that important
    # aspects of the environment like PATH and PYTHONPATH are configured.

    print('[ {label} ] : {cmd}'.format(label=cmd_dict['label'],
                                       cmd=cmd_dict['cmd']))
    stdin, stdout, stderr = ssh.exec_command('$SHELL -i -c \'' + cmd_dict['cmd'] + '\'', get_pty=True)

    # Set up channel timeout (which we rely on below to make readline() non-blocking)
    channel = stdout.channel
    channel.settimeout(0.1)

    def read_from_stdout():
        """
        Read stdout stream, time out if necessary.
        """
        try:
            line = stdout.readline()
            while len(line) > 0:    # Loops until a timeout exception occurs
                line = line.rstrip()
                logger.debug('stdout from ssh channel: %s', line)
                cmd_dict['output_queue'].put('[ {label} ] : {output}'.format(label=cmd_dict['label'],
                                                                             output=line))
                line = stdout.readline()
        except (PipeTimeout, socket.timeout):
            pass

    def read_from_stderr():
        """
        Read stderr stream, time out if necessary.
        """
        try:
            line = stderr.readline()
            while len(line) > 0:
                line = line.rstrip()
                logger.debug('stderr from ssh channel: %s', line)
                cmd_dict['output_queue'].put('[ {label} ] : '.format(label=cmd_dict['label']) +
                                             bcolors.FAIL + '{output}'.format(output=line) + bcolors.ENDC)
                line = stderr.readline()
        except (PipeTimeout, socket.timeout):
            pass

    def communicate():
        """
        Communicate a little bit, without blocking too long.
        Return True if the command ended.
        """
        read_from_stdout()
        read_from_stderr()

        # Check to see if the process has exited. If it has, we let this thread
        # terminate.
        if channel.exit_status_ready():
            exit_status = channel.recv_exit_status()
            cmd_dict['output_queue'].put('[ {label} ] : '.format(label=cmd_dict['label']) +
                                         bcolors.FAIL +
                                         "remote process exited with exit status " +
                                         str(exit_status) + bcolors.ENDC)
            return True

    # Wait for a message on the input_queue. Any message received signals this
    # thread to shut itself down.
    while cmd_dict['input_queue'].empty():
        # Kill some time so that this thread does not hog the CPU.
        time.sleep(1.0)
        if communicate():
            break

    # Ctrl-C the executing command and wait a bit for command to end cleanly
    start = time.time()
    while time.time() < start + 5.0:
        channel.send(b'\x03')  # Ctrl-C
        if communicate():
            break
        time.sleep(1.0)

    # Shutdown the channel, and close the SSH connection
    channel.close()
    ssh.close()


def start_scheduler(logdir, addr, port, ssh_username, ssh_port, ssh_private_key, remote_python=None):
    cmd = '{python} -m distributed.cli.dask_scheduler --port {port}'.format(
        python=remote_python or sys.executable, port=port, logdir=logdir)

    # Optionally re-direct stdout and stderr to a logfile
    if logdir is not None:
        cmd = 'mkdir -p {logdir} && '.format(logdir=logdir) + cmd
        cmd += '&> {logdir}/dask_scheduler_{addr}:{port}.log'.format(addr=addr,
                                                                     port=port, logdir=logdir)

    # Format output labels we can prepend to each line of output, and create
    # a 'status' key to keep track of jobs that terminate prematurely.
    label = (bcolors.BOLD +
             'scheduler {addr}:{port}'.format(addr=addr, port=port) +
             bcolors.ENDC)

    # Create a command dictionary, which contains everything we need to run and
    # interact with this command.
    input_queue = Queue()
    output_queue = Queue()
    cmd_dict = {'cmd': cmd, 'label': label, 'address': addr, 'port': port,
                'input_queue': input_queue, 'output_queue': output_queue,
                'ssh_username': ssh_username, 'ssh_port': ssh_port,
                'ssh_private_key': ssh_private_key}

    # Start the thread
    thread = Thread(target=async_ssh, args=[cmd_dict])
    thread.daemon = True
    thread.start()

    return merge(cmd_dict, {'thread': thread})


def start_worker(logdir, scheduler_addr, scheduler_port, worker_addr, nthreads, nprocs,
                 ssh_username, ssh_port, ssh_private_key, nohost,
                 memory_limit,
                 worker_port,
                 nanny_port,
                 remote_python=None,
                 remote_dask_worker=None):

    cmd = ('{python} -m {remote_dask_worker} '
           '{scheduler_addr}:{scheduler_port} '
           '--nthreads {nthreads} --nprocs {nprocs} ')

    if not nohost:
        cmd += ' --host {worker_addr} '

    if memory_limit:
        cmd += '--memory-limit {memory_limit} '

    if worker_port:
        cmd += '--worker-port {worker_port} '

    if nanny_port:
        cmd += '--nanny-port {nanny_port} '

    cmd = cmd.format(
        python=remote_python or sys.executable,
        remote_dask_worker=remote_dask_worker or 'distributed.cli.dask_worker',
        scheduler_addr=scheduler_addr,
        scheduler_port=scheduler_port,
        worker_addr=worker_addr,
        nthreads=nthreads,
        nprocs=nprocs,
        memory_limit=memory_limit,
        worker_port=worker_port,
        nanny_port=nanny_port)

    # Optionally redirect stdout and stderr to a logfile
    if logdir is not None:
        cmd = 'mkdir -p {logdir} && '.format(logdir=logdir) + cmd
        cmd += '&> {logdir}/dask_scheduler_{addr}.log'.format(
            addr=worker_addr, logdir=logdir)

    label = 'worker {addr}'.format(addr=worker_addr)

    # Create a command dictionary, which contains everything we need to run and
    # interact with this command.
    input_queue = Queue()
    output_queue = Queue()
    cmd_dict = {'cmd': cmd, 'label': label, 'address': worker_addr,
                'input_queue': input_queue, 'output_queue': output_queue,
                'ssh_username': ssh_username, 'ssh_port': ssh_port,
                'ssh_private_key': ssh_private_key}

    # Start the thread
    thread = Thread(target=async_ssh, args=[cmd_dict])
    thread.daemon = True
    thread.start()

    return merge(cmd_dict, {'thread': thread})


class SSHCluster(object):

    def __init__(self, scheduler_addr, scheduler_port, worker_addrs, nthreads=0, nprocs=1,
                 ssh_username=None, ssh_port=22, ssh_private_key=None,
                 nohost=False, logdir=None, remote_python=None,
                 memory_limit=None, worker_port=None, nanny_port=None,
                 remote_dask_worker=None):

        self.scheduler_addr = scheduler_addr
        self.scheduler_port = scheduler_port
        self.nthreads = nthreads
        self.nprocs = nprocs

        self.ssh_username = ssh_username
        self.ssh_port = ssh_port
        self.ssh_private_key = ssh_private_key

        self.nohost = nohost

        self.remote_python = remote_python

        self.memory_limit = memory_limit
        self.worker_port = worker_port
        self.nanny_port = nanny_port
        self.remote_dask_worker = remote_dask_worker

        # Generate a universal timestamp to use for log files
        import datetime
        if logdir is not None:
            logdir = os.path.join(logdir, "dask-ssh_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))
            print(bcolors.WARNING + 'Output will be redirected to logfiles '
                  'stored locally on individual worker nodes under "{logdir}".'.format(logdir=logdir)
                  + bcolors.ENDC)
        self.logdir = logdir

        # Keep track of all running threads
        self.threads = []

        # Start the scheduler node
        self.scheduler = start_scheduler(logdir, scheduler_addr,
                                         scheduler_port, ssh_username, ssh_port,
                                         ssh_private_key, remote_python)

        # Start worker nodes
        self.workers = []
        for i, addr in enumerate(worker_addrs):
            self.add_worker(addr)

    @gen.coroutine
    def _start(self):
        pass

    @property
    def scheduler_address(self):
        return '%s:%d' % (self.scheduler_addr, self.scheduler_port)

    def monitor_remote_processes(self):

        # Form a list containing all processes, since we treat them equally from here on out.
        all_processes = [self.scheduler] + self.workers

        try:
            while True:
                for process in all_processes:
                    while not process['output_queue'].empty():
                        print(process['output_queue'].get())

                # Kill some time and free up CPU before starting the next sweep
                # through the processes.
                time.sleep(0.1)

            # end while true

        except KeyboardInterrupt:
            pass   # Return execution to the calling process

    def add_worker(self, address):
        self.workers.append(start_worker(self.logdir, self.scheduler_addr,
                                         self.scheduler_port, address,
                                         self.nthreads, self.nprocs,
                                         self.ssh_username, self.ssh_port,
                                         self.ssh_private_key, self.nohost,
                                         self.memory_limit,
                                         self.worker_port,
                                         self.nanny_port,
                                         self.remote_python,
                                         self.remote_dask_worker))

    def shutdown(self):
        all_processes = [self.scheduler] + self.workers

        for process in all_processes:
            process['input_queue'].put('shutdown')
            process['thread'].join()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()

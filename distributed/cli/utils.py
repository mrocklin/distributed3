from __future__ import print_function, division, absolute_import

py3_err_msg = """
Your terminal does not properly support unicode text required by command line
utilities running Python 3.  This is commonly solved by specifying encoding
environment variables, though exact solutions may depend on your system:

    $ export LC_ALL=C.UTF-8
    $ export LANG=C.UTF-8

For more information see: http://click.pocoo.org/5/python3/
""".strip()


from distributed.comm import (parse_address, unparse_address,
                              parse_host_port, unparse_host_port)


def check_python_3():
    """Ensures that the environment is good for unicode on Python 3."""
    try:
        from click import _unicodefun
        _unicodefun._verify_python3_env()
    except (TypeError, RuntimeError) as e:
        import sys
        import click
        click.echo(py3_err_msg, err=True)
        sys.exit(1)


def install_signal_handlers(loop, cleanup=None):
    """Install global signal handlers to halt the Tornado IOLoop in case of
    a SIGINT or SIGTERM."""
    import signal

    old_handlers = {}

    def handle_signal(sig, frame):
        def cleanup_and_stop():
            try:
                if cleanup is not None:
                    cleanup(sig)
            finally:
                loop.stop()

        loop.add_callback_from_signal(cleanup_and_stop)
        signal.signal(sig, old_handlers[sig])

    for sig in [signal.SIGINT, signal.SIGTERM]:
        old_handlers[sig] = signal.signal(sig, handle_signal)


def uri_from_host_port(host_arg, port_arg, default_port):
    """
    Process the *host* and *port* CLI options.
    Return a URI.
    """
    # Much of distributed depends on a well-known IP being assigned to
    # each entity (Worker, Scheduler, etc.), so avoid "universal" addresses
    # like '' which would listen on all registered IPs and interfaces.
    scheme, loc = parse_address(host_arg or '')

    host, port = parse_host_port(loc, port_arg if port_arg is not None else default_port)

    if port is None and port_arg is None:
        port_arg = default_port

    if port and port_arg and port != port_arg:
        raise ValueError("port number given twice in options: "
                         "host %r and port %r" % (host_arg, port_arg))
    if port is None and port_arg is not None:
        port = port_arg
    # Note `port = 0` means "choose a random port"
    if port is None:
        port = default_port
    loc = unparse_host_port(host, port)
    addr = unparse_address(scheme, loc)

    return addr

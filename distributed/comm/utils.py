import logging
import socket

from tornado import gen

from .. import protocol
from ..utils import get_ip, get_ipv6, nbytes, offload, offload_to_current_event_loop


logger = logging.getLogger(__name__)


# Offload (de)serializing large frames to improve event loop responsiveness.
# We use at most 4 threads to allow for parallel processing of large messages.

FRAME_OFFLOAD_THRESHOLD = 10 * 1024 ** 2  # 10 MB


@gen.coroutine
def to_frames(msg, serializers=None, on_error="message", context=None):
    """
    Serialize a message into a list of Distributed protocol frames.
    """

    def _to_frames():
        try:
            return list(
                protocol.dumps(
                    msg, serializers=serializers, on_error=on_error, context=context
                )
            )
        except Exception as e:
            logger.info("Unserializable Message: %s", msg)
            logger.exception(e)
            raise

    res = yield offload_to_current_event_loop(_to_frames)

    raise gen.Return(res)


@gen.coroutine
def from_frames(frames, deserialize=True, deserializers=None):
    """
    Unserialize a list of Distributed protocol frames.
    """
    size = sum(map(nbytes, frames))

    def _from_frames():
        try:
            return protocol.loads(
                frames, deserialize=deserialize, deserializers=deserializers
            )
        except EOFError:
            if size > 1000:
                datastr = "[too large to display]"
            else:
                datastr = frames
            # Aid diagnosing
            logger.error("truncated data stream (%d bytes): %s", size, datastr)
            raise

    if deserialize and size > FRAME_OFFLOAD_THRESHOLD:
        res = yield offload_to_current_event_loop(_from_frames)
    else:
        res = _from_frames()

    raise gen.Return(res)


def get_tcp_server_address(tcp_server):
    """
    Get the bound address of a started Tornado TCPServer.
    """
    sockets = list(tcp_server._sockets.values())
    if not sockets:
        raise RuntimeError("TCP Server %r not started yet?" % (tcp_server,))

    def _look_for_family(fam):
        for sock in sockets:
            if sock.family == fam:
                return sock
        return None

    # If listening on both IPv4 and IPv6, prefer IPv4 as defective IPv6
    # is common (e.g. Travis-CI).
    sock = _look_for_family(socket.AF_INET)
    if sock is None:
        sock = _look_for_family(socket.AF_INET6)
    if sock is None:
        raise RuntimeError("No Internet socket found on TCPServer??")

    return sock.getsockname()


def ensure_concrete_host(host):
    """
    Ensure the given host string (or IP) denotes a concrete host, not a
    wildcard listening address.
    """
    if host in ("0.0.0.0", ""):
        return get_ip()
    elif host == "::":
        return get_ipv6()
    else:
        return host

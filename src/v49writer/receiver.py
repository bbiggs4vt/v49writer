"""UDP and TCP receivers for VRT streams."""

from __future__ import annotations

import ipaddress
import logging
import socket
import struct
import time
from typing import Callable, Optional

from . import vita49
from .capture import CaptureManager, StreamFramer

log = logging.getLogger(__name__)

_SOCK_TIMEOUT = 0.5
_RECV_BUFSIZE = 65536


def _deadline(duration: Optional[float]) -> Optional[float]:
    return time.monotonic() + duration if duration else None


def _expired(deadline: Optional[float]) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _is_multicast(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_multicast
    except ValueError:
        return False


def receive_udp(manager: CaptureManager, host: str, port: int,
                duration: Optional[float] = None,
                stop: Optional[Callable[[], bool]] = None,
                on_ready: Optional[Callable[[], None]] = None) -> None:
    """Receive VRT packets over UDP until every captured stream is done,
    the duration elapses, or ``stop()`` returns True. If ``host`` is a
    multicast group address, the group is joined."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    except OSError:
        pass
    if _is_multicast(host):
        sock.bind(('', port))
        mreq = struct.pack('=4s4s', socket.inet_aton(host),
                           socket.inet_aton('0.0.0.0'))
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
        log.info('joined multicast group %s, listening on port %d', host, port)
    else:
        sock.bind((host, port))
        log.info('listening for UDP on %s:%d', host, port)
    sock.settimeout(_SOCK_TIMEOUT)
    if on_ready:
        on_ready()
    deadline = _deadline(duration)
    try:
        while not manager.done and not _expired(deadline):
            if stop and stop():
                break
            try:
                datagram, _addr = sock.recvfrom(_RECV_BUFSIZE)
            except socket.timeout:
                continue
            try:
                for pkt in vita49.iter_packets(datagram):
                    manager.handle_packet(pkt)
            except vita49.VrtParseError as exc:
                log.warning('bad datagram dropped: %s', exc)
    finally:
        sock.close()


def receive_tcp(manager: CaptureManager, host: str, port: int,
                connect: bool = False,
                duration: Optional[float] = None,
                stop: Optional[Callable[[], bool]] = None,
                on_ready: Optional[Callable[[], None]] = None) -> None:
    """Receive VRT packets over a TCP byte stream.

    By default listens on host:port and accepts a single connection;
    with ``connect=True``, connects out to host:port instead. The stream
    must consist of back-to-back VRT packets (framed by the packet size
    field in each header). Streams are demultiplexed by stream ID like
    UDP.
    """
    deadline = _deadline(duration)
    conn = None
    listener = None
    try:
        if connect:
            conn = socket.create_connection((host, port), timeout=10)
            log.info('connected to %s:%d', host, port)
        else:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen(1)
            listener.settimeout(_SOCK_TIMEOUT)
            log.info('listening for TCP on %s:%d', host, port)
            if on_ready:
                on_ready()
            while conn is None:
                if (stop and stop()) or _expired(deadline):
                    return
                try:
                    conn, addr = listener.accept()
                except socket.timeout:
                    continue
            log.info('accepted connection from %s:%d', *addr[:2])
        conn.settimeout(_SOCK_TIMEOUT)
        if connect and on_ready:
            on_ready()
        framer = StreamFramer()
        while not manager.done and not _expired(deadline):
            if stop and stop():
                break
            try:
                data = conn.recv(_RECV_BUFSIZE)
            except socket.timeout:
                continue
            if not data:
                log.info('connection closed by peer')
                break
            for raw in framer.feed(data):
                pkt, _ = vita49.parse_packet(raw)
                if pkt is not None:
                    manager.handle_packet(pkt)
    finally:
        if conn is not None:
            conn.close()
        if listener is not None:
            listener.close()

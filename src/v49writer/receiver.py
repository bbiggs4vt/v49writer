"""UDP receiver for VRT streams."""

from __future__ import annotations

import ipaddress
import logging
import socket
import struct
import time
from typing import Callable, Optional

from . import vita49
from .capture import CaptureManager

log = logging.getLogger(__name__)

_SOCK_TIMEOUT = 0.5
_UDP_BUFSIZE = 65536


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
    deadline = time.monotonic() + duration if duration else None
    try:
        while not manager.done:
            if deadline is not None and time.monotonic() >= deadline:
                break
            if stop and stop():
                break
            try:
                datagram, _addr = sock.recvfrom(_UDP_BUFSIZE)
            except socket.timeout:
                continue
            try:
                for pkt in vita49.iter_packets(datagram):
                    manager.handle_packet(pkt)
            except vita49.VrtParseError as exc:
                log.warning('bad datagram dropped: %s', exc)
    finally:
        sock.close()

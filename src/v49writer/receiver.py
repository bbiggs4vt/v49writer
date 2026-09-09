"""UDP and TCP receivers for VRT streams."""

from __future__ import annotations

import ipaddress
import logging
import selectors
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
    """Receive VRT packets over TCP byte streams.

    By default listens on host:port and accepts any number of concurrent
    connections, each carrying back-to-back VRT packets (framed by the
    packet size field in each header); with ``connect=True``, connects
    out to host:port as a single client instead. All connections feed
    the same CaptureManager, so streams are demultiplexed by stream ID
    exactly like UDP regardless of which connection they arrive on. In
    listen mode the capture keeps running when a sender disconnects,
    until the duration elapses, ``stop()`` returns True, or every stream
    is done.
    """
    if connect:
        _tcp_connect_loop(manager, host, port, duration, stop, on_ready)
    else:
        _tcp_listen_loop(manager, host, port, duration, stop, on_ready)


def _handle_frames(manager, framer, data):
    """Feed received bytes through a framer into the manager. Returns
    False if the byte stream desynced (the connection should be dropped);
    individual malformed packets are skipped with a warning."""
    frames = framer.feed(data)
    while True:
        try:
            raw = next(frames)
        except StopIteration:
            return True
        except vita49.VrtParseError as exc:
            log.warning('TCP stream desynced: %s', exc)
            return False
        try:
            pkt, _ = vita49.parse_packet(raw)
        except vita49.VrtParseError as exc:
            log.warning('bad packet dropped: %s', exc)
            continue
        if pkt is not None:
            manager.handle_packet(pkt)


def _tcp_connect_loop(manager, host, port, duration, stop, on_ready):
    deadline = _deadline(duration)
    conn = socket.create_connection((host, port), timeout=10)
    log.info('connected to %s:%d', host, port)
    conn.settimeout(_SOCK_TIMEOUT)
    if on_ready:
        on_ready()
    framer = StreamFramer()
    try:
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
            if not _handle_frames(manager, framer, data):
                break
    finally:
        conn.close()


def _tcp_listen_loop(manager, host, port, duration, stop, on_ready):
    deadline = _deadline(duration)
    sel = selectors.DefaultSelector()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(16)
    listener.setblocking(False)
    sel.register(listener, selectors.EVENT_READ, None)
    log.info('listening for TCP on %s:%d', host, port)
    if on_ready:
        on_ready()

    def n_active():
        return len(sel.get_map()) - 1  # minus the listener

    def drop(sock):
        sel.unregister(sock)
        sock.close()

    try:
        while not manager.done and not _expired(deadline):
            if stop and stop():
                break
            for key, _mask in sel.select(timeout=_SOCK_TIMEOUT):
                sock = key.fileobj
                if sock is listener:
                    conn, addr = listener.accept()
                    conn.setblocking(False)
                    sel.register(conn, selectors.EVENT_READ,
                                 (StreamFramer(), addr))
                    log.info('accepted connection from %s:%d '
                             '(%d active)', addr[0], addr[1], n_active())
                    continue
                framer, addr = key.data
                try:
                    data = sock.recv(_RECV_BUFSIZE)
                except (BlockingIOError, InterruptedError):
                    continue
                except (ConnectionResetError, OSError):
                    data = b''
                if not data:
                    drop(sock)
                    log.info('connection from %s:%d closed (%d active)',
                             addr[0], addr[1], n_active())
                    continue
                if not _handle_frames(manager, framer, data):
                    # A desynced byte stream cannot be re-framed safely;
                    # drop this connection, keep the others.
                    drop(sock)
                    log.warning('closing desynced connection from %s:%d '
                                '(%d active)', addr[0], addr[1], n_active())
    finally:
        for key in list(sel.get_map().values()):
            key.fileobj.close()
        sel.close()

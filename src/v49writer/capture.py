"""Glue between the VITA 49 parser and the BLUE file writer.

A CaptureSession consumes parsed VRT packets, tracks stream state
(sample rate, RF frequency, timestamps, dropped packets), converts IQ
payloads to little-endian, and streams them into a BlueWriter.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from . import bluefile, vita49

log = logging.getLogger(__name__)


class CaptureSession:
    """Writes one VRT signal-data stream to one BLUE file.

    ``fmt`` is a BLUE format digraph ('CI', 'SI', 'CF', ...) or 'auto' to
    take the format from a VRT context packet's Signal Data Payload Format
    field (falling back to ``fallback_fmt`` if no context arrives before
    the first data packet).
    """

    def __init__(self, path: str, fmt: str = 'auto',
                 fallback_fmt: str = 'CI',
                 stream_id: Optional[int] = None,
                 sample_rate: Optional[float] = None,
                 payload_endian: str = 'big',
                 max_samples: Optional[int] = None,
                 extra_keywords=None):
        self.path = path
        self._fmt_arg = fmt.upper()
        self._fallback_fmt = fallback_fmt.upper()
        self._stream_filter = stream_id
        self._sample_rate_override = sample_rate
        self._payload_big_endian = (payload_endian == 'big')
        self._max_samples = max_samples
        self._extra_keywords = list(extra_keywords or [])

        self._writer: Optional[bluefile.BlueWriter] = None
        self._stream_id: Optional[int] = None
        self._last_count: Optional[int] = None
        self._first_timestamp: Optional[float] = None

        self.sample_rate: Optional[float] = sample_rate
        self.rf_freq: Optional[float] = None
        self.if_freq: Optional[float] = None
        self.bandwidth: Optional[float] = None
        self.gain_db: Optional[float] = None
        self.ref_level_dbm: Optional[float] = None
        self._context_fmt: Optional[str] = None

        self.data_packets = 0
        self.context_packets = 0
        self.dropped_packets = 0
        self.samples_written = 0
        self.done = False

    # ------------------------------------------------------------------ #

    def handle_packet(self, pkt) -> None:
        if self.done:
            return
        if isinstance(pkt, vita49.ContextPacket):
            self._handle_context(pkt)
        elif isinstance(pkt, vita49.DataPacket):
            self._handle_data(pkt)

    def _accepts_stream(self, stream_id: Optional[int]) -> bool:
        if self._stream_filter is not None:
            return stream_id == self._stream_filter
        if self._stream_id is None:
            self._stream_id = stream_id
            if stream_id is not None:
                log.info('locked to stream ID 0x%08X', stream_id)
        return stream_id == self._stream_id

    def _handle_context(self, pkt: vita49.ContextPacket) -> None:
        if (self._stream_filter is not None
                and pkt.stream_id != self._stream_filter):
            return
        self.context_packets += 1
        if pkt.sample_rate_hz is not None and self._sample_rate_override is None:
            if self.sample_rate != pkt.sample_rate_hz:
                log.info('context: sample rate %.6g Hz', pkt.sample_rate_hz)
            self.sample_rate = pkt.sample_rate_hz
        if pkt.rf_ref_freq_hz is not None:
            self.rf_freq = pkt.rf_ref_freq_hz
        if pkt.if_ref_freq_hz is not None:
            self.if_freq = pkt.if_ref_freq_hz
        if pkt.bandwidth_hz is not None:
            self.bandwidth = pkt.bandwidth_hz
        if pkt.gain_db is not None:
            self.gain_db = pkt.gain_db
        if pkt.reference_level_dbm is not None:
            self.ref_level_dbm = pkt.reference_level_dbm
        if pkt.payload_format is not None:
            fmt = pkt.payload_format.blue_format()
            if fmt is None:
                log.warning('context payload format not representable in '
                            'BLUE (item_format=%d, size=%d bits); ignoring',
                            pkt.payload_format.item_format,
                            pkt.payload_format.data_item_size)
            else:
                if self._context_fmt != fmt:
                    log.info('context: payload format %s', fmt)
                self._context_fmt = fmt

    def _resolve_format(self) -> str:
        if self._fmt_arg != 'AUTO':
            return self._fmt_arg
        if self._context_fmt is not None:
            return self._context_fmt
        log.warning('no context packet seen before first data packet; '
                    'assuming %s (16-bit complex)', self._fallback_fmt)
        return self._fallback_fmt

    def _handle_data(self, pkt: vita49.DataPacket) -> None:
        if not self._accepts_stream(pkt.stream_id):
            return
        if self._last_count is not None:
            expected = (self._last_count + 1) & 0xF
            if pkt.count != expected:
                missed = (pkt.count - expected) & 0xF
                # A wrapped-but-equal count means at least 16 lost packets;
                # report the minimum we can prove.
                missed = missed if missed else 16
                self.dropped_packets += missed
                log.warning('packet count gap: expected %d got %d '
                            '(>= %d packet(s) lost)',
                            expected, pkt.count, missed)
        self._last_count = pkt.count
        self.data_packets += 1

        if self._writer is None:
            fmt = self._resolve_format()
            xdelta = 1.0 / self.sample_rate if self.sample_rate else 1.0
            self._writer = bluefile.BlueWriter(self.path, fmt=fmt,
                                               xdelta=xdelta)
            log.info('writing %s (format %s)', self.path, fmt)

        if self._first_timestamp is None:
            utc = pkt.timestamp.to_utc_seconds()
            if utc is not None:
                self._first_timestamp = utc
                self._writer.timecode_unix = utc

        data = self._convert_payload(pkt.payload)
        nelem, _dtype, bpa = bluefile.format_info(self._writer.fmt)
        if self._max_samples is not None:
            remaining = self._max_samples - self.samples_written
            keep = remaining * bpa
            if len(data) >= keep:
                data = data[:keep]
                self.done = True
        self._writer.write(data)
        self.samples_written += len(data) // bpa

    def _convert_payload(self, payload: bytes) -> bytes:
        """Byte-swap a big-endian VRT payload to little-endian for BLUE."""
        assert self._writer is not None
        _nelem, dtype, _bpa = bluefile.format_info(self._writer.fmt)
        esize = dtype.itemsize
        usable = (len(payload) // esize) * esize
        if usable != len(payload):
            log.warning('payload length %d not a multiple of element size '
                        '%d; trailing bytes dropped', len(payload), esize)
            payload = payload[:usable]
        if esize == 1 or not self._payload_big_endian:
            return payload
        arr = np.frombuffer(payload, dtype=dtype.newbyteorder('>'))
        return arr.astype(dtype).tobytes()

    # ------------------------------------------------------------------ #

    def keywords(self):
        kw = []
        if self.sample_rate is not None:
            kw.append(('SAMPLE_RATE', float(self.sample_rate)))
        if self.rf_freq is not None:
            kw.append(('RF_FREQ', float(self.rf_freq)))
        if self.if_freq is not None:
            kw.append(('IF_FREQ', float(self.if_freq)))
        if self.bandwidth is not None:
            kw.append(('BANDWIDTH', float(self.bandwidth)))
        if self.gain_db is not None:
            kw.append(('GAIN', float(self.gain_db)))
        if self.ref_level_dbm is not None:
            kw.append(('REF_LEVEL', float(self.ref_level_dbm)))
        if self._stream_id is not None:
            kw.append(('VRT_STREAM_ID', int(self._stream_id)))
        kw.append(('VRT_DATA_PACKETS', int(self.data_packets)))
        kw.append(('VRT_DROPPED_PACKETS', int(self.dropped_packets)))
        kw.extend(self._extra_keywords)
        return kw

    def close(self) -> None:
        if self._writer is None:
            log.warning('no data packets received; no file written')
            return
        if self.sample_rate:
            self._writer.xdelta = 1.0 / self.sample_rate
        self._writer.set_keywords(self.keywords())
        self._writer.close()
        log.info('wrote %d samples (%d bytes) from %d data packets '
                 '(%d dropped) to %s',
                 self.samples_written, self._writer.data_bytes,
                 self.data_packets, self.dropped_packets, self.path)


class StreamFramer:
    """Splits a TCP byte stream into whole VRT packets using the packet
    size field in each VRT header."""

    def __init__(self):
        self._buf = bytearray()

    def feed(self, data: bytes):
        """Yield complete VRT packet byte strings."""
        self._buf += data
        while True:
            size = vita49.peek_packet_size(self._buf)
            if size is None or len(self._buf) < size:
                return
            if size == 0:
                raise vita49.VrtParseError(
                    'VRT packet with size 0 in TCP stream; stream is not '
                    'aligned to packet boundaries')
            packet = bytes(self._buf[:size])
            del self._buf[:size]
            yield packet

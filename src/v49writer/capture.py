"""Glue between the VITA 49 parser and the BLUE file writer.

A CaptureManager consumes parsed VRT packets and demultiplexes them by
stream ID: each stream ID gets its own StreamCapture, which tracks stream
state (sample rate, RF frequency, timestamps, dropped packets), converts
IQ payloads to little-endian, and streams them into its own BLUE file.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

import numpy as np

from . import bluefile, vita49

log = logging.getLogger(__name__)


def stream_path(template: str, stream_id: Optional[int]) -> str:
    """Build the output path for one stream.

    If ``template`` contains a ``{sid}`` token it is replaced with the
    stream ID as 8 hex digits ('nosid' for data packets without a stream
    ID); otherwise the ID is appended to the file stem.
    """
    label = 'nosid' if stream_id is None else '%08X' % (stream_id & 0xFFFFFFFF)
    if '{sid}' in template:
        return template.replace('{sid}', label)
    root, ext = os.path.splitext(template)
    return '%s_%s%s' % (root, label, ext)


class StreamCapture:
    """Writes one VRT signal-data stream to one BLUE file.

    ``fmt`` is a BLUE format digraph ('CI', 'SI', 'CF', ...) or 'AUTO' to
    take the format from this stream's VRT context packets (falling back
    to ``fallback_fmt`` if no context arrives before the first data
    packet).
    """

    def __init__(self, path: str, stream_id: Optional[int],
                 fmt: str = 'AUTO', fallback_fmt: str = 'CI',
                 sample_rate: Optional[float] = None,
                 payload_endian: str = 'big',
                 max_samples: Optional[int] = None,
                 extra_keywords=None):
        self.path = path
        self.stream_id = stream_id
        self._fmt_arg = fmt.upper()
        self._fallback_fmt = fallback_fmt.upper()
        self._sample_rate_override = sample_rate
        self._payload_big_endian = (payload_endian == 'big')
        self._max_samples = max_samples
        self._extra_keywords = list(extra_keywords or [])

        self._writer: Optional[bluefile.BlueWriter] = None
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

    def handle_context(self, pkt: vita49.ContextPacket) -> None:
        self.context_packets += 1
        if pkt.sample_rate_hz is not None and self._sample_rate_override is None:
            if self.sample_rate != pkt.sample_rate_hz:
                log.info('[%s] context: sample rate %.6g Hz',
                         self._label(), pkt.sample_rate_hz)
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
                log.warning('[%s] context payload format not representable '
                            'in BLUE (item_format=%d, size=%d bits); ignoring',
                            self._label(), pkt.payload_format.item_format,
                            pkt.payload_format.data_item_size)
            else:
                if self._context_fmt != fmt:
                    log.info('[%s] context: payload format %s',
                             self._label(), fmt)
                self._context_fmt = fmt

    def _label(self) -> str:
        return 'nosid' if self.stream_id is None else '0x%08X' % self.stream_id

    def _resolve_format(self) -> str:
        if self._fmt_arg != 'AUTO':
            return self._fmt_arg
        if self._context_fmt is not None:
            return self._context_fmt
        log.warning('[%s] no context packet seen before first data packet; '
                    'assuming %s (16-bit complex)',
                    self._label(), self._fallback_fmt)
        return self._fallback_fmt

    def handle_data(self, pkt: vita49.DataPacket) -> None:
        if self.done:
            return
        if self._last_count is not None:
            expected = (self._last_count + 1) & 0xF
            if pkt.count != expected:
                missed = (pkt.count - expected) & 0xF
                # A wrapped-but-equal count means at least 16 lost packets;
                # report the minimum we can prove.
                missed = missed if missed else 16
                self.dropped_packets += missed
                log.warning('[%s] packet count gap: expected %d got %d '
                            '(>= %d packet(s) lost)',
                            self._label(), expected, pkt.count, missed)
        self._last_count = pkt.count
        self.data_packets += 1

        if self._writer is None:
            fmt = self._resolve_format()
            xdelta = 1.0 / self.sample_rate if self.sample_rate else 1.0
            self._writer = bluefile.BlueWriter(self.path, fmt=fmt,
                                               xdelta=xdelta)
            log.info('[%s] writing %s (format %s)',
                     self._label(), self.path, fmt)

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
            log.warning('[%s] payload length %d not a multiple of element '
                        'size %d; trailing bytes dropped',
                        self._label(), len(payload), esize)
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
        if self.stream_id is not None:
            kw.append(('VRT_STREAM_ID', int(self.stream_id)))
        kw.append(('VRT_DATA_PACKETS', int(self.data_packets)))
        kw.append(('VRT_DROPPED_PACKETS', int(self.dropped_packets)))
        kw.extend(self._extra_keywords)
        return kw

    def close(self) -> None:
        if self._writer is None:
            log.warning('[%s] no data packets received; no file written',
                        self._label())
            return
        if self.sample_rate:
            self._writer.xdelta = 1.0 / self.sample_rate
        self._writer.set_keywords(self.keywords())
        self._writer.close()
        log.info('[%s] wrote %d samples (%d bytes) from %d data packets '
                 '(%d dropped) to %s',
                 self._label(), self.samples_written, self._writer.data_bytes,
                 self.data_packets, self.dropped_packets, self.path)


class CaptureManager:
    """Demultiplexes VRT packets by stream ID into one BLUE file each.

    ``output`` is a path template (see :func:`stream_path`). Context and
    data packets are routed to the StreamCapture for their stream ID,
    creating it on first sight. If ``stream_id`` is given, all other
    streams are ignored. ``max_samples`` applies per stream; the manager
    reports ``done`` once every stream seen so far has hit its limit.
    """

    def __init__(self, output: str, fmt: str = 'auto',
                 stream_id: Optional[int] = None,
                 sample_rate: Optional[float] = None,
                 payload_endian: str = 'big',
                 max_samples: Optional[int] = None,
                 extra_keywords=None):
        self.output = output
        self._fmt = fmt
        self._stream_filter = stream_id
        self._sample_rate = sample_rate
        self._payload_endian = payload_endian
        self._max_samples = max_samples
        self._extra_keywords = list(extra_keywords or [])
        self._streams: Dict[Optional[int], StreamCapture] = {}

    def _stream_for(self, stream_id: Optional[int]) -> Optional[StreamCapture]:
        if (self._stream_filter is not None
                and stream_id != self._stream_filter):
            return None
        stream = self._streams.get(stream_id)
        if stream is None:
            stream = StreamCapture(
                path=stream_path(self.output, stream_id),
                stream_id=stream_id,
                fmt=self._fmt,
                sample_rate=self._sample_rate,
                payload_endian=self._payload_endian,
                max_samples=self._max_samples,
                extra_keywords=self._extra_keywords)
            self._streams[stream_id] = stream
            if stream_id is not None:
                log.info('new stream ID 0x%08X', stream_id)
        return stream

    def handle_packet(self, pkt) -> None:
        if isinstance(pkt, vita49.ContextPacket):
            stream = self._stream_for(pkt.stream_id)
            if stream is not None:
                stream.handle_context(pkt)
        elif isinstance(pkt, vita49.DataPacket):
            stream = self._stream_for(pkt.stream_id)
            if stream is not None:
                stream.handle_data(pkt)

    @property
    def done(self) -> bool:
        if self._max_samples is None:
            return False
        with_data = [s for s in self._streams.values() if s.data_packets]
        return bool(with_data) and all(s.done for s in with_data)

    @property
    def streams(self) -> Dict[Optional[int], StreamCapture]:
        return self._streams

    @property
    def data_packets(self) -> int:
        return sum(s.data_packets for s in self._streams.values())

    @property
    def dropped_packets(self) -> int:
        return sum(s.dropped_packets for s in self._streams.values())

    @property
    def samples_written(self) -> int:
        return sum(s.samples_written for s in self._streams.values())

    def close(self) -> None:
        if not self._streams:
            log.warning('no VRT packets received; no files written')
        for stream in self._streams.values():
            stream.close()

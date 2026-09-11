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


def stream_path(template: str, stream_id: Optional[int],
                rf_freq: Optional[float] = None) -> str:
    """Build the output path for one stream.

    A ``{sid}`` token is replaced with the stream ID as 8 hex digits
    ('nosid' for data packets without a stream ID); if the template has
    no ``{sid}`` token, the ID is appended to the file stem. A ``{freq}``
    token is replaced with the stream's RF reference frequency in MHz
    with kHz resolution (e.g. '915.000MHz') as parsed from its VRT
    context packets ('nofreq' if none has announced one).
    """
    sid_label = ('nosid' if stream_id is None
                 else '%08X' % (stream_id & 0xFFFFFFFF))
    if rf_freq is None:
        freq_label = 'nofreq'
    else:
        mhz, khz = divmod(round(rf_freq / 1e3), 1000)
        freq_label = '%d.%03dMHz' % (mhz, khz)
    template = template.replace('{freq}', freq_label)
    if '{sid}' in template:
        return template.replace('{sid}', sid_label)
    root, ext = os.path.splitext(template)
    return '%s_%s%s' % (root, sid_label, ext)


class StreamCapture:
    """Writes one VRT signal-data stream to one BLUE file.

    ``fmt`` is a BLUE format digraph ('CI', 'SI', 'CF', ...) or 'AUTO' to
    take the format from this stream's VRT context packets (falling back
    to ``fallback_fmt`` if no context arrives before the first data
    packet).
    """

    def __init__(self, template: str, stream_id: Optional[int],
                 fmt: str = 'AUTO', fallback_fmt: str = 'CI',
                 sample_rate: Optional[float] = None,
                 payload_endian: str = 'big',
                 max_samples: Optional[int] = None,
                 extra_keywords=None):
        self.template = template
        self.path: Optional[str] = None  # resolved when the file is opened
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
        self._warned = set()
        self._context_fmt_unusable = False

    def _warn_once(self, key: str, msg: str, *args) -> None:
        """Log a warning the first time ``key`` occurs for this stream;
        repeats go to debug so a malformed stream can't flood the log."""
        if key in self._warned:
            log.debug('[%s] (repeat) ' + msg, self._label(), *args)
        else:
            self._warned.add(key)
            log.warning('[%s] ' + msg + ' (repeats logged at debug level)',
                        self._label(), *args)

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
                self._context_fmt_unusable = True
                self._warn_once(
                    'bad-payload-format',
                    'context payload format not representable in BLUE: raw '
                    'field words 0x%08X 0x%08X decode as real/complex=%d, '
                    'item_format=0x%02X, item size=%d bits, packing size='
                    '%d bits; ignoring. For complex float32 the field '
                    'should be 0x2E0007DF 0x00000000',
                    pkt.payload_format.raw_word1,
                    pkt.payload_format.raw_word2,
                    pkt.payload_format.real_complex,
                    pkt.payload_format.item_format,
                    pkt.payload_format.data_item_size,
                    pkt.payload_format.item_packing_size)
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
        if self._context_fmt_unusable:
            log.warning('[%s] the context payload format field could not '
                        'be used (see earlier warning); assuming %s '
                        '(16-bit complex). Fix the field at the source or '
                        'force the format with -f',
                        self._label(), self._fallback_fmt)
        elif self.context_packets:
            log.warning('[%s] %d context packet(s) seen, but none carried '
                        'a Signal Data Payload Format field (CIF0 bit 15); '
                        'assuming %s (16-bit complex). If the stream is not '
                        '16-bit complex fixed point, set the field at the '
                        'source or force the format with -f',
                        self._label(), self.context_packets,
                        self._fallback_fmt)
        else:
            log.warning('[%s] no context packet seen before first data '
                        'packet; assuming %s (16-bit complex). If the '
                        'stream is not 16-bit complex fixed point, force '
                        'the format with -f', self._label(),
                        self._fallback_fmt)
        return self._fallback_fmt

    def handle_data(self, pkt: vita49.DataPacket) -> None:
        if self.done:
            return
        if self.data_packets == 0:
            log.debug('[%s] first data packet: type %d, %d bytes payload, '
                      'tsi=%s, tsf=%s, class_id=%s, trailer=%s',
                      self._label(), pkt.packet_type,
                      len(pkt.payload), pkt.timestamp.tsi.name,
                      pkt.timestamp.tsf.name, pkt.class_id,
                      'present' if pkt.trailer is not None else 'absent')
            if pkt.timestamp.tsi == vita49.Tsi.NONE:
                log.info('[%s] data packets carry no integer timestamps; '
                         'BLUE timecode will be 0', self._label())
            elif pkt.timestamp.tsi != vita49.Tsi.UTC:
                self._warn_once(
                    'non-utc-timestamp',
                    'data packet timestamps are %s, not UTC; BLUE timecode '
                    'will be 0', pkt.timestamp.tsi.name)
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
            self.path = stream_path(self.template, self.stream_id,
                                    self.rf_freq)
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
            self._warn_once(
                'odd-payload-length',
                'payload length %d is not a multiple of the %d-byte element '
                'size; trailing bytes dropped. The stream may not be %s '
                'formatted', len(payload), esize, self._writer.fmt)
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
        self._rename_if_freq_learned()
        log.info('[%s] wrote %d samples (%d bytes) from %d data packets '
                 '(%d dropped) to %s',
                 self._label(), self.samples_written, self._writer.data_bytes,
                 self.data_packets, self.dropped_packets, self.path)

    def _rename_if_freq_learned(self) -> None:
        """If the template names the file by frequency but the RF frequency
        only became known after the file was opened (context arrived after
        the first data packet, or changed mid-capture), rename the finished
        file to match."""
        if '{freq}' not in self.template:
            return
        desired = stream_path(self.template, self.stream_id, self.rf_freq)
        if desired == self.path:
            return
        if os.path.exists(desired):
            log.warning('[%s] not renaming %s to %s: target already exists',
                        self._label(), self.path, desired)
            return
        os.rename(self.path, desired)
        log.info('[%s] renamed %s to %s (RF frequency learned after the '
                 'file was opened)', self._label(), self.path, desired)
        self.path = desired


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
        self.skipped_packets: Dict[str, int] = {}

    def _stream_for(self, stream_id: Optional[int]) -> Optional[StreamCapture]:
        if (self._stream_filter is not None
                and stream_id != self._stream_filter):
            return None
        stream = self._streams.get(stream_id)
        if stream is None:
            stream = StreamCapture(
                template=self.output,
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
        elif isinstance(pkt, vita49.SkippedPacket):
            desc = pkt.describe()
            first = desc not in self.skipped_packets
            self.skipped_packets[desc] = self.skipped_packets.get(desc, 0) + 1
            if first:
                log.warning('received unsupported VRT %s%s; ignoring it '
                            '(repeats logged at debug level, totals '
                            'reported at close)', desc,
                            '' if pkt.stream_id is None
                            else ' on stream 0x%08X' % pkt.stream_id)
            else:
                log.debug('unsupported VRT %s (%d so far)', desc,
                          self.skipped_packets[desc])

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
        if not self._streams and not self.skipped_packets:
            log.warning('no VRT packets received; no files written')
        elif not self._streams:
            log.warning('no signal data or context packets received; '
                        'no files written')
        for stream in self._streams.values():
            stream.close()
        for desc, count in sorted(self.skipped_packets.items()):
            log.warning('ignored %d unsupported VRT %s', count, desc)


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

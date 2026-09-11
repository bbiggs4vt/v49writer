"""VITA 49 (VRT) packet parsing.

Supports the packet types needed to capture an IQ stream:

* Signal data packets (types 0 and 1), including class ID, timestamps and
  trailer fields.
* Context packets (type 4), parsing the CIF0 fields commonly needed to
  describe an IQ stream (bandwidth, IF/RF reference frequency, reference
  level, gain, sample rate and the signal data payload format).

All VRT fields are big-endian (network order) per the standard.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

log = logging.getLogger(__name__)

VRT_WORD = 4  # bytes per 32-bit VRT word

# GPS epoch (1980-01-06T00:00:00 UTC) in UNIX time.
GPS_EPOCH_UNIX = 315964800

# Leap seconds between GPS time and UTC. GPS time is not adjusted for leap
# seconds, so UTC = GPS - offset. 18 has been correct since 2017-01-01;
# override if another leap second is ever introduced.
DEFAULT_GPS_LEAP_SECONDS = 18


class PacketType(IntEnum):
    SIGNAL_DATA = 0x0
    SIGNAL_DATA_STREAM_ID = 0x1
    EXT_DATA = 0x2
    EXT_DATA_STREAM_ID = 0x3
    CONTEXT = 0x4
    EXT_CONTEXT = 0x5
    COMMAND = 0x6
    EXT_COMMAND = 0x7


class Tsi(IntEnum):
    NONE = 0
    UTC = 1
    GPS = 2
    OTHER = 3


class Tsf(IntEnum):
    NONE = 0
    SAMPLE_COUNT = 1
    REAL_TIME_PS = 2
    FREE_RUNNING = 3


@dataclass
class ClassId:
    oui: int
    info_class: int
    packet_class: int


@dataclass
class Timestamp:
    tsi: Tsi
    tsf: Tsf
    integer: Optional[int] = None
    fractional: Optional[int] = None

    def to_utc_seconds(
            self,
            gps_leap_seconds: Optional[int] = DEFAULT_GPS_LEAP_SECONDS,
    ) -> Optional[float]:
        """Return seconds since the UNIX epoch, or None if unavailable.

        UTC timestamps are returned directly. GPS timestamps (seconds
        since 1980-01-06, unadjusted for leap seconds) are converted
        using ``gps_leap_seconds``; pass None to disable GPS conversion.
        """
        if self.integer is None:
            return None
        if self.tsi == Tsi.UTC:
            seconds = float(self.integer)
        elif self.tsi == Tsi.GPS and gps_leap_seconds is not None:
            seconds = float(GPS_EPOCH_UNIX + self.integer - gps_leap_seconds)
        else:
            return None
        if self.tsf == Tsf.REAL_TIME_PS and self.fractional is not None:
            seconds += self.fractional * 1e-12
        return seconds


@dataclass
class DataPacket:
    packet_type: PacketType
    count: int
    size_words: int
    stream_id: Optional[int]
    class_id: Optional[ClassId]
    timestamp: Timestamp
    payload: bytes
    trailer: Optional[int]


@dataclass
class PayloadFormat:
    """Decoded VRT Signal Data Packet Payload Format field (CIF0 bit 15)."""
    packing_method: int
    real_complex: int      # 0 = real, 1 = complex cartesian, 2 = complex polar
    item_format: int       # 0 = signed fixed point, 14 = IEEE float32, ...
    item_packing_size: int  # bits
    data_item_size: int     # bits
    raw_word1: int = 0      # the two field words as received, for diagnostics
    raw_word2: int = 0

    ITEM_FMT_SIGNED_FIXED = 0x00
    ITEM_FMT_UNSIGNED_FIXED = 0x10
    ITEM_FMT_FLOAT16 = 0x0D
    ITEM_FMT_FLOAT32 = 0x0E
    ITEM_FMT_FLOAT64 = 0x0F

    def blue_format(self) -> Optional[str]:
        """Best-effort mapping to a Midas BLUE 2-character format code."""
        if self.real_complex == 0:
            mode = 'S'
        elif self.real_complex == 1:
            mode = 'C'
        else:
            return None
        if self.item_format == self.ITEM_FMT_SIGNED_FIXED:
            ftype = {8: 'B', 16: 'I', 32: 'L', 64: 'X'}.get(self.data_item_size)
        elif self.item_format == self.ITEM_FMT_FLOAT32:
            ftype = 'F'
        elif self.item_format == self.ITEM_FMT_FLOAT64:
            ftype = 'D'
        else:
            ftype = None
        if ftype is None:
            return None
        return mode + ftype


@dataclass
class ContextPacket:
    count: int
    size_words: int
    stream_id: Optional[int]
    class_id: Optional[ClassId]
    timestamp: Timestamp
    cif0: int
    change_indicator: bool = False
    bandwidth_hz: Optional[float] = None
    if_ref_freq_hz: Optional[float] = None
    rf_ref_freq_hz: Optional[float] = None
    rf_ref_freq_offset_hz: Optional[float] = None
    if_band_offset_hz: Optional[float] = None
    reference_level_dbm: Optional[float] = None
    gain_db: Optional[float] = None
    sample_rate_hz: Optional[float] = None
    payload_format: Optional[PayloadFormat] = None


@dataclass
class SkippedPacket:
    """A syntactically valid VRT packet of a type this application does
    not process (extension data/context, command, or a reserved type).
    Returned so callers can count and report them instead of silently
    dropping traffic from a misconfigured source."""
    packet_type_bits: int
    size_words: int
    stream_id: Optional[int] = None

    _NAMES = {
        0x2: 'extension data',
        0x3: 'extension data (with stream ID)',
        0x5: 'extension context',
        0x6: 'command',
        0x7: 'extension command',
    }

    def describe(self) -> str:
        name = self._NAMES.get(self.packet_type_bits,
                               'reserved/unknown type %d'
                               % self.packet_type_bits)
        return '%s packet (type 0x%X)' % (name, self.packet_type_bits)


class VrtParseError(ValueError):
    pass


def _u64_fixed_q20(hi: int, lo: int) -> float:
    """Decode a 64-bit two's-complement fixed-point value with a 20-bit
    fractional part (used for frequencies, bandwidth and sample rate)."""
    raw = (hi << 32) | lo
    if raw & (1 << 63):
        raw -= 1 << 64
    return raw / float(1 << 20)


def _s16_fixed_q7(raw: int) -> float:
    """Decode a 16-bit two's-complement fixed point value with a 7-bit
    fractional part (used for reference level and gain)."""
    if raw & 0x8000:
        raw -= 0x10000
    return raw / 128.0


def peek_packet_size(buf: bytes, offset: int = 0) -> Optional[int]:
    """Return the size in bytes of the VRT packet starting at ``offset``,
    or None if fewer than 4 bytes are available."""
    if len(buf) - offset < VRT_WORD:
        return None
    (word0,) = struct.unpack_from('>I', buf, offset)
    return (word0 & 0xFFFF) * VRT_WORD


def parse_packet(buf: bytes, offset: int = 0):
    """Parse one VRT packet starting at ``offset``.

    Returns a (packet, next_offset) tuple. ``packet`` is a DataPacket,
    ContextPacket, or a SkippedPacket for packet types we do not handle
    (the offset still advances past them).
    """
    if len(buf) - offset < VRT_WORD:
        raise VrtParseError('short buffer: no room for VRT header')
    (word0,) = struct.unpack_from('>I', buf, offset)
    ptype_bits = (word0 >> 28) & 0xF
    has_class_id = bool(word0 & (1 << 27))
    tsi = Tsi((word0 >> 22) & 0x3)
    tsf = Tsf((word0 >> 20) & 0x3)
    count = (word0 >> 16) & 0xF
    size_words = word0 & 0xFFFF

    if size_words == 0:
        raise VrtParseError('VRT packet with size 0')
    end = offset + size_words * VRT_WORD
    if end > len(buf):
        raise VrtParseError(
            'truncated VRT packet: need %d bytes, have %d'
            % (size_words * VRT_WORD, len(buf) - offset))

    try:
        ptype = PacketType(ptype_bits)
    except ValueError:
        log.debug('skipping VRT packet with reserved type bits 0x%X '
                  '(%d words)', ptype_bits, size_words)
        return SkippedPacket(ptype_bits, size_words), end

    pos = offset + VRT_WORD

    stream_id = None
    if ptype in (PacketType.SIGNAL_DATA_STREAM_ID, PacketType.EXT_DATA_STREAM_ID,
                 PacketType.CONTEXT, PacketType.EXT_CONTEXT,
                 PacketType.COMMAND, PacketType.EXT_COMMAND):
        (stream_id,) = struct.unpack_from('>I', buf, pos)
        pos += VRT_WORD

    class_id = None
    if has_class_id:
        w1, w2 = struct.unpack_from('>II', buf, pos)
        class_id = ClassId(oui=w1 & 0xFFFFFF,
                           info_class=(w2 >> 16) & 0xFFFF,
                           packet_class=w2 & 0xFFFF)
        pos += 2 * VRT_WORD

    ts_int = None
    if tsi != Tsi.NONE:
        (ts_int,) = struct.unpack_from('>I', buf, pos)
        pos += VRT_WORD

    ts_frac = None
    if tsf != Tsf.NONE:
        hi, lo = struct.unpack_from('>II', buf, pos)
        ts_frac = (hi << 32) | lo
        pos += 2 * VRT_WORD

    timestamp = Timestamp(tsi=tsi, tsf=tsf, integer=ts_int, fractional=ts_frac)

    if ptype in (PacketType.SIGNAL_DATA, PacketType.SIGNAL_DATA_STREAM_ID):
        has_trailer = bool(word0 & (1 << 26))
        payload_end = end - (VRT_WORD if has_trailer else 0)
        if payload_end < pos:
            raise VrtParseError('VRT data packet smaller than its own headers')
        trailer = None
        if has_trailer:
            (trailer,) = struct.unpack_from('>I', buf, payload_end)
        pkt = DataPacket(packet_type=ptype, count=count, size_words=size_words,
                         stream_id=stream_id, class_id=class_id,
                         timestamp=timestamp,
                         payload=bytes(buf[pos:payload_end]),
                         trailer=trailer)
        return pkt, end

    if ptype == PacketType.CONTEXT:
        pkt = _parse_context(buf, pos, end, count, size_words, stream_id,
                             class_id, timestamp)
        return pkt, end

    # Extension data/context and command packets: skip, but report.
    log.debug('skipping unsupported VRT %s (%d words, stream %s)',
              SkippedPacket(ptype_bits, size_words).describe(), size_words,
              '0x%08X' % stream_id if stream_id is not None else 'n/a')
    return SkippedPacket(ptype_bits, size_words, stream_id), end


def _parse_context(buf, pos, end, count, size_words, stream_id, class_id,
                   timestamp) -> ContextPacket:
    if end - pos < VRT_WORD:
        raise VrtParseError('context packet has no CIF0 word')
    (cif0,) = struct.unpack_from('>I', buf, pos)
    pos += VRT_WORD

    pkt = ContextPacket(count=count, size_words=size_words,
                        stream_id=stream_id, class_id=class_id,
                        timestamp=timestamp, cif0=cif0,
                        change_indicator=bool(cif0 & (1 << 31)))

    def take(nwords):
        nonlocal pos
        if pos + nwords * VRT_WORD > end:
            raise VrtParseError('context packet truncated mid-field')
        vals = struct.unpack_from('>%dI' % nwords, buf, pos)
        pos += nwords * VRT_WORD
        return vals

    # VITA 49.2: enabled CIF1/CIF2/CIF3/CIF7 indicator words directly
    # follow CIF0, before any field data. Consume them so the CIF0 fields
    # below parse from the right offset; their own fields trail the CIF0
    # fields and are simply left unparsed.
    extra_cifs = sum(1 for bit in (1, 2, 3, 7) if cif0 & (1 << bit))
    if extra_cifs:
        take(extra_cifs)
        log.debug('context packet enables %d additional CIF word(s) '
                  '(CIF0=0x%08X); their fields are not parsed',
                  extra_cifs, cif0)

    # CIF0 fields appear in descending bit order. Sizes (in words) come from
    # ANSI/VITA 49.2 table 9.1-1; VITA 49.0 uses the same layout.
    if cif0 & (1 << 30):   # Reference point identifier
        take(1)
    if cif0 & (1 << 29):   # Bandwidth
        hi, lo = take(2)
        pkt.bandwidth_hz = _u64_fixed_q20(hi, lo)
    if cif0 & (1 << 28):   # IF reference frequency
        hi, lo = take(2)
        pkt.if_ref_freq_hz = _u64_fixed_q20(hi, lo)
    if cif0 & (1 << 27):   # RF reference frequency
        hi, lo = take(2)
        pkt.rf_ref_freq_hz = _u64_fixed_q20(hi, lo)
    if cif0 & (1 << 26):   # RF reference frequency offset
        hi, lo = take(2)
        pkt.rf_ref_freq_offset_hz = _u64_fixed_q20(hi, lo)
    if cif0 & (1 << 25):   # IF band offset
        hi, lo = take(2)
        pkt.if_band_offset_hz = _u64_fixed_q20(hi, lo)
    if cif0 & (1 << 24):   # Reference level
        (w,) = take(1)
        pkt.reference_level_dbm = _s16_fixed_q7(w & 0xFFFF)
    if cif0 & (1 << 23):   # Gain (stage 2 in upper 16 bits, stage 1 in lower)
        (w,) = take(1)
        pkt.gain_db = _s16_fixed_q7(w & 0xFFFF) + _s16_fixed_q7((w >> 16) & 0xFFFF)
    if cif0 & (1 << 22):   # Over-range count
        take(1)
    if cif0 & (1 << 21):   # Sample rate
        hi, lo = take(2)
        pkt.sample_rate_hz = _u64_fixed_q20(hi, lo)
    if cif0 & (1 << 20):   # Timestamp adjustment
        take(2)
    if cif0 & (1 << 19):   # Timestamp calibration time
        take(1)
    if cif0 & (1 << 18):   # Temperature
        take(1)
    if cif0 & (1 << 17):   # Device identifier
        take(2)
    if cif0 & (1 << 16):   # State and event indicators
        take(1)
    if cif0 & (1 << 15):   # Signal data packet payload format
        w1, w2 = take(2)
        pkt.payload_format = PayloadFormat(
            packing_method=(w1 >> 31) & 0x1,
            real_complex=(w1 >> 29) & 0x3,
            item_format=(w1 >> 24) & 0x1F,
            item_packing_size=((w1 >> 6) & 0x3F) + 1,
            data_item_size=(w1 & 0x3F) + 1,
            raw_word1=w1, raw_word2=w2,
        )
    # Remaining CIF0 fields (GPS/INS/ephemeris/ASCII/association lists) are
    # variable-length or positional data we do not need; stop here.
    return pkt


def iter_packets(buf: bytes):
    """Yield parsed packets from a buffer containing one or more
    back-to-back VRT packets (e.g. a UDP datagram)."""
    offset = 0
    while offset + VRT_WORD <= len(buf):
        pkt, offset = parse_packet(buf, offset)
        if pkt is not None:
            yield pkt

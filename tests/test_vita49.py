import struct

import pytest

from v49writer import vita49
from v49writer.generator import build_context_packet, build_data_packet


def test_parse_data_packet_roundtrip():
    payload = bytes(range(16)) * 4
    raw = build_data_packet(payload, stream_id=0xDEADBEEF, count=7,
                            utc_seconds=1_700_000_000,
                            frac_ps=250_000_000_000)
    pkt, end = vita49.parse_packet(raw)
    assert end == len(raw)
    assert isinstance(pkt, vita49.DataPacket)
    assert pkt.stream_id == 0xDEADBEEF
    assert pkt.count == 7
    assert pkt.payload == payload
    assert pkt.timestamp.tsi == vita49.Tsi.UTC
    assert pkt.timestamp.integer == 1_700_000_000
    assert pkt.timestamp.fractional == 250_000_000_000
    assert pkt.timestamp.to_utc_seconds() == pytest.approx(1_700_000_000.25)


def test_parse_data_packet_no_timestamps():
    raw = build_data_packet(b'\x01\x02\x03\x04', stream_id=1, count=0)
    pkt, _ = vita49.parse_packet(raw)
    assert pkt.timestamp.to_utc_seconds() is None
    assert pkt.payload == b'\x01\x02\x03\x04'


def test_parse_data_packet_with_trailer():
    # Type 0 (no stream ID) with trailer bit set.
    payload = b'\xAA\xBB\xCC\xDD' * 2
    words = 1 + len(payload) // 4 + 1
    word0 = (0x0 << 28) | (1 << 26) | (5 << 16) | words
    raw = struct.pack('>I', word0) + payload + struct.pack('>I', 0x12345678)
    pkt, _ = vita49.parse_packet(raw)
    assert pkt.stream_id is None
    assert pkt.payload == payload
    assert pkt.trailer == 0x12345678


def test_parse_context_packet():
    raw = build_context_packet(0x42, 3, sample_rate=10e6, rf_freq=2.4e9,
                               bandwidth=8e6, item_size_bits=16,
                               utc_seconds=1_700_000_123)
    pkt, end = vita49.parse_packet(raw)
    assert end == len(raw)
    assert isinstance(pkt, vita49.ContextPacket)
    assert pkt.stream_id == 0x42
    assert pkt.sample_rate_hz == pytest.approx(10e6)
    assert pkt.rf_ref_freq_hz == pytest.approx(2.4e9)
    assert pkt.bandwidth_hz == pytest.approx(8e6)
    assert pkt.payload_format is not None
    assert pkt.payload_format.data_item_size == 16
    assert pkt.payload_format.real_complex == 1
    assert pkt.payload_format.blue_format() == 'CI'


def test_payload_format_mapping():
    fmt = vita49.PayloadFormat(0, 1, vita49.PayloadFormat.ITEM_FMT_FLOAT32,
                               32, 32)
    assert fmt.blue_format() == 'CF'
    fmt = vita49.PayloadFormat(0, 0, 0, 8, 8)
    assert fmt.blue_format() == 'SB'
    # complex polar has no BLUE representation
    fmt = vita49.PayloadFormat(0, 2, 0, 16, 16)
    assert fmt.blue_format() is None


def test_iter_packets_multiple():
    raw = (build_context_packet(1, 0, sample_rate=1e6)
           + build_data_packet(b'\0' * 8, 1, 0)
           + build_data_packet(b'\0' * 8, 1, 1))
    pkts = list(vita49.iter_packets(raw))
    assert len(pkts) == 3
    assert isinstance(pkts[0], vita49.ContextPacket)
    assert isinstance(pkts[1], vita49.DataPacket)
    assert pkts[2].count == 1


def test_truncated_packet_raises():
    raw = build_data_packet(b'\0' * 64, 1, 0)
    with pytest.raises(vita49.VrtParseError):
        vita49.parse_packet(raw[:20])


def test_fixed_point_negative():
    # -1.5 Hz in Q20: two's complement
    raw = -int(1.5 * (1 << 20)) & 0xFFFFFFFFFFFFFFFF
    assert vita49._u64_fixed_q20(raw >> 32, raw & 0xFFFFFFFF) == -1.5
    assert vita49._s16_fixed_q7(0xFF80) == -1.0

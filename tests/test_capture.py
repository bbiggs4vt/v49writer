import numpy as np
import pytest

from v49writer import bluefile, vita49
from v49writer.capture import CaptureSession, StreamFramer
from v49writer.generator import (build_context_packet, build_data_packet,
                                 make_tone)


def _feed(session, raw):
    for pkt in vita49.iter_packets(raw):
        session.handle_packet(pkt)


def test_capture_tone_with_context(tmp_path):
    path = str(tmp_path / 'cap.tmp')
    fs, ftone = 1e6, 100e3
    session = CaptureSession(path, fmt='auto')
    raw = build_context_packet(0x99, 0, sample_rate=fs, rf_freq=2.4e9,
                               bandwidth=0.8 * fs, item_size_bits=16)
    total = 0
    for i in range(5):
        payload = make_tone(500, fs, ftone, start_sample=total)
        raw += build_data_packet(payload, 0x99, i,
                                 utc_seconds=1_700_000_000 + i)
        total += 500
    _feed(session, raw)
    session.close()

    hdr = bluefile.read_header(path)
    assert hdr['format'] == 'CI'
    assert hdr['xdelta'] == pytest.approx(1.0 / fs)
    assert hdr['timecode'] == pytest.approx(
        1_700_000_000 + bluefile.J1950_TO_UNIX)
    kw = dict(hdr['ext_header'])
    assert kw['SAMPLE_RATE'] == pytest.approx(fs)
    assert kw['RF_FREQ'] == pytest.approx(2.4e9)
    assert kw['VRT_STREAM_ID'] == 0x99
    assert kw['VRT_DATA_PACKETS'] == 5
    assert kw['VRT_DROPPED_PACKETS'] == 0

    data = bluefile.read_data(path)
    assert len(data) == total
    expected = make_tone(total, fs, ftone)
    exp = np.frombuffer(expected, dtype='>i2').astype(np.float64)
    np.testing.assert_array_equal(data.real, exp[0::2])
    np.testing.assert_array_equal(data.imag, exp[1::2])


def test_capture_detects_drops(tmp_path):
    session = CaptureSession(str(tmp_path / 'cap.tmp'), fmt='ci')
    payload = b'\0' * 8
    _feed(session, build_data_packet(payload, 1, 0))
    _feed(session, build_data_packet(payload, 1, 1))
    _feed(session, build_data_packet(payload, 1, 5))  # 3 lost
    session.close()
    assert session.dropped_packets == 3


def test_capture_stream_filter(tmp_path):
    path = str(tmp_path / 'cap.tmp')
    session = CaptureSession(path, fmt='ci', stream_id=0x2)
    _feed(session, build_data_packet(b'\x00\x01\x00\x02', 0x1, 0))
    _feed(session, build_data_packet(b'\x00\x03\x00\x04', 0x2, 0))
    session.close()
    assert session.data_packets == 1
    data = bluefile.read_data(path)
    np.testing.assert_array_equal(data, [3 + 4j])


def test_capture_locks_first_stream(tmp_path):
    session = CaptureSession(str(tmp_path / 'cap.tmp'), fmt='ci')
    _feed(session, build_data_packet(b'\0' * 4, 0xA, 0))
    _feed(session, build_data_packet(b'\0' * 4, 0xB, 0))  # ignored
    _feed(session, build_data_packet(b'\0' * 4, 0xA, 1))
    session.close()
    assert session.data_packets == 2
    assert session.dropped_packets == 0


def test_capture_max_samples(tmp_path):
    path = str(tmp_path / 'cap.tmp')
    session = CaptureSession(path, fmt='ci', max_samples=3)
    _feed(session, build_data_packet(b'\0' * 8, 1, 0))  # 2 samples
    _feed(session, build_data_packet(b'\0' * 8, 1, 1))  # 1 kept
    assert session.done
    _feed(session, build_data_packet(b'\0' * 8, 1, 2))  # ignored
    session.close()
    assert session.samples_written == 3
    assert bluefile.read_header(path)['data_size'] == 12.0


def test_capture_sample_rate_override(tmp_path):
    path = str(tmp_path / 'cap.tmp')
    session = CaptureSession(path, fmt='ci', sample_rate=5e6)
    _feed(session, build_context_packet(1, 0, sample_rate=1e6))
    _feed(session, build_data_packet(b'\0' * 8, 1, 0))
    session.close()
    assert bluefile.read_header(path)['xdelta'] == pytest.approx(1 / 5e6)


def test_capture_float_format_from_context(tmp_path):
    path = str(tmp_path / 'cap.tmp')
    session = CaptureSession(path, fmt='auto')
    _feed(session, build_context_packet(
        1, 0, sample_rate=1e3, item_size_bits=32,
        item_format=vita49.PayloadFormat.ITEM_FMT_FLOAT32))
    payload = np.array([1.5, -2.5], dtype='>f4').tobytes()
    _feed(session, build_data_packet(payload, 1, 0))
    session.close()
    hdr = bluefile.read_header(path)
    assert hdr['format'] == 'CF'
    np.testing.assert_array_equal(bluefile.read_data(path), [1.5 - 2.5j])


def test_stream_framer_reassembly():
    raw = (build_data_packet(b'\x01' * 8, 1, 0)
           + build_data_packet(b'\x02' * 12, 1, 1)
           + build_data_packet(b'\x03' * 4, 1, 2))
    framer = StreamFramer()
    packets = []
    # Feed one byte at a time to exercise partial-frame handling.
    for i in range(len(raw)):
        packets.extend(framer.feed(raw[i:i + 1]))
    assert len(packets) == 3
    counts = [vita49.parse_packet(p)[0].count for p in packets]
    assert counts == [0, 1, 2]

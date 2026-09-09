import numpy as np
import pytest

from v49writer import bluefile, vita49
from v49writer.capture import CaptureManager, stream_path
from v49writer.generator import (build_context_packet, build_data_packet,
                                 make_tone)


def _feed(manager, raw):
    for pkt in vita49.iter_packets(raw):
        manager.handle_packet(pkt)


def test_stream_path_naming():
    assert stream_path('/a/out.tmp', 0x1234) == '/a/out_00001234.tmp'
    assert stream_path('out', 0xDEADBEEF) == 'out_DEADBEEF'
    assert stream_path('cap_{sid}.tmp', 0x42) == 'cap_00000042.tmp'
    assert stream_path('out.tmp', None) == 'out_nosid.tmp'


def test_capture_tone_with_context(tmp_path):
    template = str(tmp_path / 'cap.tmp')
    fs, ftone = 1e6, 100e3
    manager = CaptureManager(template, fmt='auto')
    raw = build_context_packet(0x99, 0, sample_rate=fs, rf_freq=2.4e9,
                               bandwidth=0.8 * fs, item_size_bits=16)
    total = 0
    for i in range(5):
        payload = make_tone(500, fs, ftone, start_sample=total)
        raw += build_data_packet(payload, 0x99, i,
                                 utc_seconds=1_700_000_000 + i)
        total += 500
    _feed(manager, raw)
    manager.close()

    path = str(tmp_path / 'cap_00000099.tmp')
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


def test_one_file_per_stream(tmp_path):
    template = str(tmp_path / 'cap.tmp')
    manager = CaptureManager(template, fmt='ci')
    _feed(manager, build_context_packet(0xA, 0, sample_rate=1e6))
    _feed(manager, build_context_packet(0xB, 0, sample_rate=2e6))
    _feed(manager, build_data_packet(b'\x00\x01\x00\x02', 0xA, 0))
    _feed(manager, build_data_packet(b'\x00\x03\x00\x04', 0xB, 0))
    _feed(manager, build_data_packet(b'\x00\x05\x00\x06', 0xA, 1))
    manager.close()

    hdr_a = bluefile.read_header(str(tmp_path / 'cap_0000000A.tmp'))
    hdr_b = bluefile.read_header(str(tmp_path / 'cap_0000000B.tmp'))
    # Context is tracked per stream.
    assert hdr_a['xdelta'] == pytest.approx(1e-6)
    assert hdr_b['xdelta'] == pytest.approx(5e-7)
    assert dict(hdr_a['ext_header'])['VRT_STREAM_ID'] == 0xA
    assert dict(hdr_b['ext_header'])['VRT_STREAM_ID'] == 0xB
    np.testing.assert_array_equal(
        bluefile.read_data(str(tmp_path / 'cap_0000000A.tmp')),
        [1 + 2j, 5 + 6j])
    np.testing.assert_array_equal(
        bluefile.read_data(str(tmp_path / 'cap_0000000B.tmp')),
        [3 + 4j])


def test_drops_counted_per_stream(tmp_path):
    manager = CaptureManager(str(tmp_path / 'cap.tmp'), fmt='ci')
    payload = b'\0' * 8
    # Stream 1 drops 3 packets; interleaved stream 2 stays contiguous.
    _feed(manager, build_data_packet(payload, 1, 0))
    _feed(manager, build_data_packet(payload, 2, 0))
    _feed(manager, build_data_packet(payload, 1, 1))
    _feed(manager, build_data_packet(payload, 2, 1))
    _feed(manager, build_data_packet(payload, 1, 5))  # 3 lost
    manager.close()
    assert manager.streams[1].dropped_packets == 3
    assert manager.streams[2].dropped_packets == 0
    assert manager.dropped_packets == 3


def test_capture_stream_filter(tmp_path):
    template = str(tmp_path / 'cap.tmp')
    manager = CaptureManager(template, fmt='ci', stream_id=0x2)
    _feed(manager, build_data_packet(b'\x00\x01\x00\x02', 0x1, 0))
    _feed(manager, build_data_packet(b'\x00\x03\x00\x04', 0x2, 0))
    manager.close()
    assert manager.data_packets == 1
    assert list(manager.streams) == [0x2]
    data = bluefile.read_data(str(tmp_path / 'cap_00000002.tmp'))
    np.testing.assert_array_equal(data, [3 + 4j])
    assert not (tmp_path / 'cap_00000001.tmp').exists()


def test_capture_max_samples_per_stream(tmp_path):
    template = str(tmp_path / 'cap.tmp')
    manager = CaptureManager(template, fmt='ci', max_samples=3)
    _feed(manager, build_data_packet(b'\0' * 8, 1, 0))  # 2 samples
    assert not manager.done
    _feed(manager, build_data_packet(b'\0' * 8, 2, 0))  # stream 2: 2 samples
    _feed(manager, build_data_packet(b'\0' * 8, 1, 1))  # stream 1 done at 3
    assert not manager.done
    _feed(manager, build_data_packet(b'\0' * 8, 2, 1))  # stream 2 done at 3
    assert manager.done
    _feed(manager, build_data_packet(b'\0' * 8, 1, 2))  # ignored
    manager.close()
    assert manager.streams[1].samples_written == 3
    assert manager.streams[2].samples_written == 3
    hdr = bluefile.read_header(str(tmp_path / 'cap_00000001.tmp'))
    assert hdr['data_size'] == 12.0


def test_capture_sample_rate_override(tmp_path):
    template = str(tmp_path / 'cap.tmp')
    manager = CaptureManager(template, fmt='ci', sample_rate=5e6)
    _feed(manager, build_context_packet(1, 0, sample_rate=1e6))
    _feed(manager, build_data_packet(b'\0' * 8, 1, 0))
    manager.close()
    hdr = bluefile.read_header(str(tmp_path / 'cap_00000001.tmp'))
    assert hdr['xdelta'] == pytest.approx(1 / 5e6)


def test_capture_float_format_from_context(tmp_path):
    template = str(tmp_path / 'cap.tmp')
    manager = CaptureManager(template, fmt='auto')
    _feed(manager, build_context_packet(
        1, 0, sample_rate=1e3, item_size_bits=32,
        item_format=vita49.PayloadFormat.ITEM_FMT_FLOAT32))
    payload = np.array([1.5, -2.5], dtype='>f4').tobytes()
    _feed(manager, build_data_packet(payload, 1, 0))
    manager.close()
    path = str(tmp_path / 'cap_00000001.tmp')
    assert bluefile.read_header(path)['format'] == 'CF'
    np.testing.assert_array_equal(bluefile.read_data(path), [1.5 - 2.5j])


def test_no_max_samples_never_done(tmp_path):
    manager = CaptureManager(str(tmp_path / 'cap.tmp'), fmt='ci')
    _feed(manager, build_data_packet(b'\0' * 8, 1, 0))
    assert not manager.done
    manager.close()

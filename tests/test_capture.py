import numpy as np
import pytest

from v49writer import bluefile, vita49
from v49writer.capture import CaptureManager, StreamFramer, stream_path
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
    # {freq} token: RF frequency in MHz at kHz resolution.
    assert (stream_path('cap_{sid}_{freq}.tmp', 0x42, 915e6)
            == 'cap_00000042_915.000MHz.tmp')
    assert (stream_path('cap_{sid}_{freq}.tmp', 0x42, 2.4e9)
            == 'cap_00000042_2400.000MHz.tmp')
    assert (stream_path('cap_{sid}_{freq}.tmp', 0x42, 99.6127e6)
            == 'cap_00000042_99.613MHz.tmp')  # rounded to nearest kHz
    assert stream_path('cap_{sid}_{freq}.tmp', 0x42) == 'cap_00000042_nofreq.tmp'
    # Without {sid}, the stream ID is still appended (collision safety).
    assert (stream_path('cap_{freq}.tmp', None, 99.6e6)
            == 'cap_99.600MHz_nosid.tmp')


def test_freq_in_filename_from_context(tmp_path):
    # Context (with RF freq) before data: the file opens under its final name.
    template = str(tmp_path / 'cap_{sid}_{freq}.tmp')
    manager = CaptureManager(template, fmt='ci')
    _feed(manager, build_context_packet(0x7, 0, sample_rate=1e6,
                                        rf_freq=100e6))
    _feed(manager, build_data_packet(b'\0' * 8, 0x7, 0))
    manager.close()
    path = tmp_path / 'cap_00000007_100.000MHz.tmp'
    assert path.exists()
    assert dict(bluefile.read_header(str(path))['ext_header'])['RF_FREQ'] \
        == pytest.approx(100e6)


def test_freq_learned_after_open_renames(tmp_path):
    # Data arrives first, context later: the file is renamed on close.
    template = str(tmp_path / 'cap_{sid}_{freq}.tmp')
    manager = CaptureManager(template, fmt='ci')
    _feed(manager, build_data_packet(b'\0' * 8, 0x7, 0))
    _feed(manager, build_context_packet(0x7, 0, rf_freq=2.4e9))
    _feed(manager, build_data_packet(b'\0' * 8, 0x7, 1))
    manager.close()
    assert not (tmp_path / 'cap_00000007_nofreq.tmp').exists()
    path = tmp_path / 'cap_00000007_2400.000MHz.tmp'
    assert path.exists()
    assert bluefile.read_header(str(path))['data_size'] == 16.0
    assert manager.streams[0x7].path == str(path)


def test_freq_never_known_keeps_nofreq(tmp_path):
    template = str(tmp_path / 'cap_{sid}_{freq}.tmp')
    manager = CaptureManager(template, fmt='ci')
    _feed(manager, build_data_packet(b'\0' * 8, 0x7, 0))
    manager.close()
    assert (tmp_path / 'cap_00000007_nofreq.tmp').exists()


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


def test_context_without_payload_format_warns_specifically(tmp_path, caplog):
    # A context packet that carries sample rate but no payload format
    # field must produce the CIF0-bit-15 diagnostic, not the generic
    # "no context packet seen" message.
    manager = CaptureManager(str(tmp_path / 'cap.tmp'), fmt='auto')
    with caplog.at_level('WARNING'):
        _feed(manager, build_context_packet(0x1, 0, sample_rate=1e6,
                                            item_size_bits=None))
        _feed(manager, build_data_packet(b'\0' * 8, 0x1, 0))
    manager.close()
    msgs = [r.message for r in caplog.records]
    assert any('Signal Data Payload Format field' in m for m in msgs)
    assert not any('no context packet seen' in m for m in msgs)


def test_little_endian_payload(tmp_path):
    # Samples sent little-endian: --payload-endian little must interpret
    # them without byte-swapping.
    path_tpl = str(tmp_path / 'cap.tmp')
    manager = CaptureManager(path_tpl, fmt='ci', payload_endian='little')
    payload = np.array([100, -200, 300, -400], dtype='<i2').tobytes()
    _feed(manager, build_data_packet(payload, 0x9, 0))
    manager.close()
    data = bluefile.read_data(str(tmp_path / 'cap_00000009.tmp'))
    np.testing.assert_array_equal(data, [100 - 200j, 300 - 400j])

    # The same bytes read as big-endian (the default) decode differently,
    # proving the flag changes interpretation.
    manager2 = CaptureManager(str(tmp_path / 'cap2.tmp'), fmt='ci')
    _feed(manager2, build_data_packet(payload, 0x9, 0))
    manager2.close()
    data2 = bluefile.read_data(str(tmp_path / 'cap2_00000009.tmp'))
    assert not np.array_equal(data, data2)


def test_output_directories_created(tmp_path):
    template = str(tmp_path / 'captures' / 'run1' / 'cap_{sid}.tmp')
    manager = CaptureManager(template, fmt='ci')
    _feed(manager, build_data_packet(b'\x00\x01\x00\x02', 0x3, 0))
    manager.close()
    path = tmp_path / 'captures' / 'run1' / 'cap_00000003.tmp'
    assert path.exists()
    np.testing.assert_array_equal(bluefile.read_data(str(path)), [1 + 2j])


def test_skipped_packets_counted_and_capture_unaffected(tmp_path, caplog):
    import struct

    manager = CaptureManager(str(tmp_path / 'cap.tmp'), fmt='ci')
    # An extension-data packet (type 3) interleaved with real data.
    ext = struct.pack('>III', (0x3 << 28) | 3, 0x77, 0)
    with caplog.at_level('WARNING'):
        _feed(manager, build_data_packet(b'\x00\x01\x00\x02', 0x1, 0))
        _feed(manager, ext)
        _feed(manager, ext)
        _feed(manager, build_data_packet(b'\x00\x03\x00\x04', 0x1, 1))
    manager.close()

    assert manager.skipped_packets == {
        'extension data (with stream ID) packet (type 0x3)': 2}
    # Warned once, not per packet.
    warnings = [r for r in caplog.records
                if 'unsupported VRT extension data' in r.message
                and 'ignoring' in r.message]
    assert len(warnings) == 1
    # The real stream captured normally.
    data = bluefile.read_data(str(tmp_path / 'cap_00000001.tmp'))
    np.testing.assert_array_equal(data, [1 + 2j, 3 + 4j])


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

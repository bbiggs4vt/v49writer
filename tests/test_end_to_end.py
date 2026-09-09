"""End-to-end tests: generator -> loopback socket -> receiver -> BLUE files."""

import socket
import threading

import numpy as np
import pytest

from v49writer import bluefile, generator, receiver
from v49writer.capture import CaptureManager


def _free_port(kind):
    s = socket.socket(socket.AF_INET, kind)
    s.bind(('127.0.0.1', 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_tcp_multiple_connections(tmp_path):
    """Two senders connect concurrently, each carrying its own stream ID;
    both streams land in their own files."""
    from v49writer.generator import build_context_packet, build_data_packet

    port = _free_port(socket.SOCK_STREAM)
    template = str(tmp_path / 'multi_{sid}.tmp')
    manager = CaptureManager(template, fmt='ci')
    ready = threading.Event()

    def both_streams_complete():
        # max_samples can't stop this capture deterministically: the first
        # stream could finish before the second connection is even seen.
        return all(sid in manager.streams
                   and manager.streams[sid].samples_written >= 200
                   for sid in (0xA1, 0xB2))

    rx = threading.Thread(
        target=receiver.receive_tcp,
        args=(manager, '127.0.0.1', port),
        kwargs={'duration': 15, 'on_ready': ready.set,
                'stop': both_streams_complete},
        daemon=True)
    rx.start()
    assert ready.wait(timeout=10)

    def sender(sid, value):
        conn = socket.create_connection(('127.0.0.1', port))
        try:
            conn.sendall(build_context_packet(sid, 0, sample_rate=1e6))
            payload = np.full(200, value, dtype='>i2').tobytes()  # 100 cx
            for i in range(2):
                conn.sendall(build_data_packet(payload, sid, i))
        finally:
            conn.close()

    threads = [threading.Thread(target=sender, args=(0xA1, 7)),
               threading.Thread(target=sender, args=(0xB2, 9))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    rx.join(timeout=20)
    assert not rx.is_alive()
    manager.close()

    assert set(manager.streams) == {0xA1, 0xB2}
    data_a = bluefile.read_data(str(tmp_path / 'multi_000000A1.tmp'))
    data_b = bluefile.read_data(str(tmp_path / 'multi_000000B2.tmp'))
    assert len(data_a) == 200 and len(data_b) == 200
    np.testing.assert_array_equal(data_a, np.full(200, 7 + 7j))
    np.testing.assert_array_equal(data_b, np.full(200, 9 + 9j))


def test_tcp_desynced_connection_dropped_others_survive(tmp_path):
    """A connection sending garbage is closed; a good connection keeps
    capturing."""
    from v49writer.generator import build_data_packet

    port = _free_port(socket.SOCK_STREAM)
    manager = CaptureManager(str(tmp_path / 'cap.tmp'), fmt='ci',
                             max_samples=4)
    ready = threading.Event()
    rx = threading.Thread(
        target=receiver.receive_tcp,
        args=(manager, '127.0.0.1', port),
        kwargs={'duration': 15, 'on_ready': ready.set},
        daemon=True)
    rx.start()
    assert ready.wait(timeout=10)

    bad = socket.create_connection(('127.0.0.1', port))
    good = socket.create_connection(('127.0.0.1', port))
    try:
        # A zero first word decodes as packet size 0: unframeable garbage.
        bad.sendall(b'\x00' * 8)
        good.sendall(build_data_packet(b'\x00\x01\x00\x02' * 2, 0x5, 0))
        good.sendall(build_data_packet(b'\x00\x01\x00\x02' * 2, 0x5, 1))
    finally:
        bad.close()
        good.close()
    rx.join(timeout=20)
    assert not rx.is_alive()
    manager.close()

    assert list(manager.streams) == [0x5]
    assert manager.streams[0x5].samples_written == 4


@pytest.mark.parametrize('transport', ['udp', 'tcp'])
def test_end_to_end_multistream(tmp_path, transport):
    kind = socket.SOCK_DGRAM if transport == 'udp' else socket.SOCK_STREAM
    port = _free_port(kind)
    template = str(tmp_path / 'e2e_{sid}_{freq}.tmp')
    fs, ftone, tone_step, nsamp = 1e6, 125e3, 50e3, 20_000
    rf_freq = 915e6
    stream_ids = [0x10, 0x20]

    manager = CaptureManager(template, fmt='auto', max_samples=nsamp)
    ready = threading.Event()

    def run_receiver():
        if transport == 'udp':
            receiver.receive_udp(manager, '127.0.0.1', port, duration=15,
                                 on_ready=ready.set)
        else:
            receiver.receive_tcp(manager, '127.0.0.1', port, duration=15,
                                 on_ready=ready.set)

    rx = threading.Thread(target=run_receiver, daemon=True)
    rx.start()
    assert ready.wait(timeout=10)

    generator.main(['-t', transport, '-H', '127.0.0.1', '-p', str(port),
                    '-r', str(fs), '--tone-freq', str(ftone),
                    '--tone-step', str(tone_step),
                    '--rf-freq', str(rf_freq),
                    '-s', '0x10', '-s', '0x20',
                    '-n', str(nsamp + 5000),
                    '--samples-per-packet', '500',
                    '--context-interval', '10'])
    rx.join(timeout=20)
    assert not rx.is_alive()
    manager.close()

    assert set(manager.streams) == set(stream_ids)
    for idx, sid in enumerate(stream_ids):
        stream = manager.streams[sid]
        assert stream.samples_written == nsamp
        if transport == 'tcp':
            assert stream.dropped_packets == 0

        path = str(tmp_path / ('e2e_%08X_915.000MHz.tmp' % sid))
        hdr = bluefile.read_header(path)
        assert hdr['type'] == 1000
        assert hdr['format'] == 'CI'
        assert hdr['xdelta'] == pytest.approx(1.0 / fs)
        assert hdr['timecode'] > bluefile.J1950_TO_UNIX
        kw = dict(hdr['ext_header'])
        assert kw['SAMPLE_RATE'] == pytest.approx(fs)
        assert kw['RF_FREQ'] == pytest.approx(rf_freq)
        assert kw['VRT_STREAM_ID'] == sid

        data = bluefile.read_data(path)
        assert len(data) == nsamp
        expected_tone = ftone + idx * tone_step
        if transport == 'tcp':
            # Lossless transport: verify the tone content exactly.
            expected = np.frombuffer(
                generator.make_tone(nsamp, fs, expected_tone), dtype='>i2'
            ).astype(np.float64)
            np.testing.assert_array_equal(data.real, expected[0::2])
            np.testing.assert_array_equal(data.imag, expected[1::2])
        # Each stream carries its own tone; find the spectral peak.
        # (UDP over loopback rarely drops, but keep this loss-tolerant.)
        spec = np.abs(np.fft.fft(data * np.hanning(len(data))))
        freqs = np.fft.fftfreq(len(data), d=1.0 / fs)
        assert freqs[int(np.argmax(spec))] == pytest.approx(
            expected_tone, rel=1e-3)

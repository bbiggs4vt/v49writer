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

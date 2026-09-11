"""VRT packet building and a test-signal generator CLI (v49gen).

Builds standard-conformant VITA 49 signal data and context packets and
sends complex tones over UDP or TCP, for testing v49writer end to end.
Give --stream-id more than once to interleave multiple streams.
"""

from __future__ import annotations

import argparse
import logging
import socket
import struct
import time
from typing import Optional

import numpy as np

log = logging.getLogger('v49gen')


def build_data_packet(payload: bytes, stream_id: int, count: int,
                      utc_seconds: Optional[int] = None,
                      frac_ps: Optional[int] = None) -> bytes:
    """Build a signal data packet (type 1) with optional UTC/psec
    timestamps. ``payload`` must be a multiple of 4 bytes."""
    if len(payload) % 4:
        raise ValueError('payload must be a whole number of 32-bit words')
    tsi = 1 if utc_seconds is not None else 0
    tsf = 2 if frac_ps is not None else 0
    words = 1 + 1 + (1 if tsi else 0) + (2 if tsf else 0) + len(payload) // 4
    word0 = (0x1 << 28) | (tsi << 22) | (tsf << 20) | ((count & 0xF) << 16) | words
    out = struct.pack('>II', word0, stream_id)
    if tsi:
        out += struct.pack('>I', utc_seconds)
    if tsf:
        out += struct.pack('>Q', frac_ps)
    return out + payload


def _q20(hz: float) -> int:
    return int(round(hz * (1 << 20))) & 0xFFFFFFFFFFFFFFFF


def build_context_packet(stream_id: int, count: int,
                         sample_rate: Optional[float] = None,
                         rf_freq: Optional[float] = None,
                         bandwidth: Optional[float] = None,
                         item_size_bits: Optional[int] = 16,
                         item_format: int = 0,
                         real_complex: int = 1,
                         utc_seconds: Optional[int] = None) -> bytes:
    """Build a context packet (type 4) carrying the given CIF0 fields."""
    cif0 = 1 << 31  # context field change indicator
    body = b''
    if bandwidth is not None:
        cif0 |= 1 << 29
        body += struct.pack('>Q', _q20(bandwidth))
    if rf_freq is not None:
        cif0 |= 1 << 27
        body += struct.pack('>Q', _q20(rf_freq))
    if sample_rate is not None:
        cif0 |= 1 << 21
        body += struct.pack('>Q', _q20(sample_rate))
    if item_size_bits is not None:
        cif0 |= 1 << 15
        w1 = ((real_complex & 0x3) << 29) | ((item_format & 0x1F) << 24)
        w1 |= ((item_size_bits - 1) & 0x3F) << 6
        w1 |= (item_size_bits - 1) & 0x3F
        body += struct.pack('>II', w1, 0)
    tsi = 1 if utc_seconds is not None else 0
    words = 1 + 1 + (1 if tsi else 0) + 1 + len(body) // 4
    word0 = (0x4 << 28) | (tsi << 22) | ((count & 0xF) << 16) | words
    out = struct.pack('>II', word0, stream_id)
    if tsi:
        out += struct.pack('>I', utc_seconds)
    return out + struct.pack('>I', cif0) + body


def make_tone(num_samples: int, sample_rate: float, tone_freq: float,
              amplitude: float = 0.5, start_sample: int = 0,
              item_size_bits: int = 16, endian: str = 'big') -> bytes:
    """Generate interleaved complex fixed-point tone samples, big-endian
    per the VITA 49 standard by default."""
    n = np.arange(start_sample, start_sample + num_samples)
    phase = 2 * np.pi * tone_freq * n / sample_rate
    iq = np.empty(2 * num_samples, dtype=np.float64)
    iq[0::2] = np.cos(phase)
    iq[1::2] = np.sin(phase)
    scale = amplitude * (2 ** (item_size_bits - 1) - 1)
    order = '>' if endian == 'big' else '<'
    dtype = {8: 'i1', 16: 'i2', 32: 'i4'}[item_size_bits]
    return (iq * scale).round().astype(order + dtype).tobytes()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog='v49gen',
        description='Send VITA 49 test IQ streams (complex tones) over '
                    'UDP or TCP.')
    p.add_argument('-t', '--transport', choices=['udp', 'tcp'],
                   default='udp')
    p.add_argument('-H', '--host', default='127.0.0.1')
    p.add_argument('-p', '--port', type=int, required=True)
    p.add_argument('-r', '--sample-rate', type=float, default=1e6)
    p.add_argument('--tone-freq', type=float, default=100e3,
                   help='tone frequency of the first stream; each further '
                        'stream is offset by --tone-step')
    p.add_argument('--tone-step', type=float, default=50e3)
    p.add_argument('--rf-freq', type=float, default=100e6)
    p.add_argument('-s', '--stream-id', action='append',
                   type=lambda s: int(s, 0), metavar='ID',
                   help='VRT stream ID; repeat to interleave multiple '
                        'streams (default: 0x1234)')
    p.add_argument('-n', '--num-samples', type=int, default=1_000_000,
                   help='samples to send per stream (default 1e6)')
    p.add_argument('--samples-per-packet', type=int, default=1000)
    p.add_argument('--bits', type=int, choices=[8, 16, 32], default=16,
                   help='bits per I/Q component (default 16)')
    p.add_argument('--context-interval', type=int, default=100,
                   help='send a context packet every N data packets')
    p.add_argument('--payload-endian', choices=['big', 'little'],
                   default='big',
                   help='byte order of the IQ samples in the payload '
                        '(default: big, per the VITA 49 standard; little '
                        'simulates a nonconformant source)')
    p.add_argument('--throttle', action='store_true',
                   help='pace transmission at the sample rate')
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    stream_ids = args.stream_id or [0x1234]
    if args.transport == 'udp':
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((args.host, args.port))
    else:
        sock = socket.create_connection((args.host, args.port))
    log.info('sending %d samples/stream at %.6g Hz to %s:%d over %s '
             '(stream IDs: %s)',
             args.num_samples, args.sample_rate, args.host, args.port,
             args.transport, ', '.join('0x%08X' % s for s in stream_ids))

    data_count = 0
    ctx_count = 0
    sent = 0
    start = time.monotonic()
    try:
        while sent < args.num_samples:
            n = min(args.samples_per_packet, args.num_samples - sent)
            for idx, stream_id in enumerate(stream_ids):
                if data_count % args.context_interval == 0:
                    sock.sendall(build_context_packet(
                        stream_id, ctx_count & 0xF,
                        sample_rate=args.sample_rate, rf_freq=args.rf_freq,
                        bandwidth=args.sample_rate * 0.8,
                        item_size_bits=args.bits,
                        utc_seconds=int(time.time())))
                payload = make_tone(
                    n, args.sample_rate,
                    args.tone_freq + idx * args.tone_step,
                    start_sample=sent, item_size_bits=args.bits,
                    endian=args.payload_endian)
                now = time.time()
                sock.sendall(build_data_packet(
                    payload, stream_id, data_count & 0xF,
                    utc_seconds=int(now),
                    frac_ps=int((now % 1) * 1e12)))
            if data_count % args.context_interval == 0:
                ctx_count += 1
            data_count += 1
            sent += n
            if args.throttle:
                target = start + sent / args.sample_rate
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
    except (ConnectionResetError, ConnectionRefusedError, BrokenPipeError) as exc:
        log.info('receiver went away (%s); stopping', exc)
    finally:
        sock.close()
    log.info('sent %d samples per stream in %d data packets per stream',
             sent, data_count)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

"""Command-line interface: v49writer."""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from .capture import CaptureManager
from . import receiver

log = logging.getLogger('v49writer')

FORMATS = ['auto', 'cb', 'ci', 'cl', 'cx', 'cf', 'cd',
           'sb', 'si', 'sl', 'sx', 'sf', 'sd']


def _parse_keyword(text: str):
    if '=' not in text:
        raise argparse.ArgumentTypeError('keyword must be KEY=VALUE')
    key, value = text.split('=', 1)
    for conv in (int, float):
        try:
            return key, conv(value)
        except ValueError:
            pass
    return key, value


def _parse_stream_id(text: str) -> int:
    return int(text, 0)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='v49writer',
        description='Receive VITA 49 (VRT) IQ streams over UDP or TCP and '
                    'write each stream ID to its own Midas BLUE (type 1000) '
                    'file.')
    p.add_argument('output',
                   help='output BLUE file path template; each stream ID '
                        'gets its own file. A {sid} token is replaced with '
                        'the stream ID as 8 hex digits, otherwise the ID is '
                        'appended to the file stem (out.tmp -> '
                        'out_00001234.tmp). A {freq} token is replaced with '
                        "the stream's RF frequency in MHz at kHz resolution "
                        'as parsed from VRT context packets (or "nofreq"), '
                        'e.g. cap_{sid}_{freq}.tmp -> '
                        'cap_00001234_915.000MHz.tmp')
    p.add_argument('-t', '--transport', choices=['udp', 'tcp'], default='udp',
                   help='transport to receive on (default: udp)')
    p.add_argument('-H', '--host', default='0.0.0.0',
                   help='address to bind/listen on, a UDP multicast group '
                        'to join, or the remote host with --connect '
                        '(default: 0.0.0.0)')
    p.add_argument('-p', '--port', type=int, required=True,
                   help='UDP/TCP port')
    p.add_argument('--connect', action='store_true',
                   help='TCP only: connect to HOST:PORT instead of listening')
    p.add_argument('-f', '--format', choices=FORMATS, default='auto',
                   help="BLUE data format: 'c'=complex/'s'=scalar + "
                        "'b'=int8,'i'=int16,'l'=int32,'x'=int64,"
                        "'f'=float32,'d'=float64. 'auto' uses the VRT "
                        'context payload format, falling back to ci '
                        '(default: auto)')
    p.add_argument('-r', '--sample-rate', type=float,
                   help='sample rate in Hz (overrides VRT context; applies '
                        'to all streams)')
    p.add_argument('-s', '--stream-id', type=_parse_stream_id,
                   help='only capture this VRT stream ID (accepts 0x hex); '
                        'default: capture every stream seen')
    p.add_argument('-n', '--max-samples', type=int,
                   help='stop each stream after this many samples (complex '
                        'pairs count as one sample); capture ends when all '
                        'streams seen have finished')
    p.add_argument('-d', '--duration', type=float,
                   help='stop after this many seconds')
    p.add_argument('--payload-endian', choices=['big', 'little'],
                   default='big',
                   help='byte order of IQ samples in the VRT payload '
                        '(default: big, per the VITA 49 standard)')
    p.add_argument('-k', '--keyword', type=_parse_keyword, action='append',
                   default=[], metavar='KEY=VALUE',
                   help='extra extended-header keyword, added to every '
                        'file (repeatable)')
    p.add_argument('-v', '--verbose', action='store_true',
                   help='debug logging')
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s')

    if args.connect and args.transport != 'tcp':
        log.error('--connect only applies to --transport tcp')
        return 2

    manager = CaptureManager(
        output=args.output,
        fmt=args.format,
        stream_id=args.stream_id,
        sample_rate=args.sample_rate,
        payload_endian=args.payload_endian,
        max_samples=args.max_samples,
        extra_keywords=args.keyword,
    )

    stopping = []

    def _sigint(_sig, _frm):
        if stopping:
            raise KeyboardInterrupt
        log.info('interrupt received; finishing up '
                 '(interrupt again to force quit)')
        stopping.append(True)

    signal.signal(signal.SIGINT, _sigint)

    try:
        if args.transport == 'udp':
            receiver.receive_udp(manager, args.host, args.port,
                                 duration=args.duration,
                                 stop=lambda: bool(stopping))
        else:
            receiver.receive_tcp(manager, args.host, args.port,
                                 connect=args.connect,
                                 duration=args.duration,
                                 stop=lambda: bool(stopping))
    finally:
        manager.close()
    return 0 if manager.data_packets else 1


if __name__ == '__main__':
    sys.exit(main())

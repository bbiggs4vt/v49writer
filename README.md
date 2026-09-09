# v49writer

Receive VITA 49 (VRT) IQ streams over UDP or TCP and write them to
Midas BLUE files (X-Midas / REDHAWK compatible, type 1000).

## Features

- **UDP** (unicast or multicast — pass a multicast group as the host and it
  is joined automatically) and **TCP** (listen or connect) transports.
- Parses VRT **signal data packets** (types 0 and 1) with class IDs,
  timestamps, and trailers, and **context packets** (type 4) for sample
  rate, RF/IF frequency, bandwidth, gain, reference level, and the signal
  data payload format.
- Writes **type 1000 BLUE files** with little-endian (`EEEI`) header and
  data: `xdelta` from the stream's sample rate, `timecode` (J1950 epoch)
  from the first packet's UTC timestamp, and stream metadata as
  extended-header keywords (`SAMPLE_RATE`, `RF_FREQ`, `BANDWIDTH`,
  `VRT_STREAM_ID`, ...).
- **Automatic data format** selection from the VRT context payload format
  field (8/16/32/64-bit fixed point, float32/64, real or complex), or force
  one with `--format`.
- Locks to the first data stream seen, or filter with `--stream-id`.
- Detects dropped packets via the VRT packet count and records the tally.
- Stop with Ctrl-C, `--duration`, or `--max-samples`; the header is
  finalized on close either way.

## Install

```sh
pip install .          # runtime (numpy only)
pip install -e '.[dev]'  # with test dependencies
```

## Usage

Capture a UDP VRT stream on port 5000 to `capture.tmp`:

```sh
v49writer -p 5000 capture.tmp
```

More examples:

```sh
# Join a multicast group, stop after 10 seconds
v49writer -H 239.1.2.3 -p 5000 -d 10 capture.tmp

# TCP: listen for a sender
v49writer -t tcp -p 5000 capture.tmp

# TCP: connect out to a source, force 32-bit float complex payloads
v49writer -t tcp --connect -H 10.0.0.5 -p 5000 -f cf capture.tmp

# Capture exactly 1M samples of a specific stream, override sample rate,
# and add custom keywords
v49writer -p 5000 -s 0x1234 -n 1000000 -r 10e6 -k MISSION=test1 capture.tmp
```

Data format codes (`--format`) are BLUE digraphs: first letter `c` (complex)
or `s` (scalar/real); second letter `b`=int8, `i`=int16, `l`=int32,
`x`=int64, `f`=float32, `d`=float64. The default `auto` uses the format
announced in VRT context packets, falling back to `ci` (16-bit complex,
the common VITA 49 IQ format).

### Test signal generator

`v49gen` sends a standard-conformant VRT stream (complex tone plus context
packets) for testing:

```sh
v49gen -p 5000 -r 1e6 --tone-freq 100e3 -n 1000000        # UDP
v49gen -t tcp -H 127.0.0.1 -p 5000 --throttle             # TCP, real-time paced
```

## Notes

- VRT payloads are big-endian per the VITA 49 standard; samples are
  byte-swapped to little-endian on write. For nonconformant sources use
  `--payload-endian little`.
- The BLUE `timecode` is seconds since 1950-01-01 (J1950), taken from the
  first data packet carrying a UTC (TSI=1) timestamp; the fractional part
  uses real-time picosecond (TSF=2) timestamps when present.
- One capture session writes one stream to one file. Run multiple instances
  with `--stream-id` filters to split a multi-stream aggregate.

## Development

```sh
python -m pytest
```

The test suite covers the VRT parser, the BLUE writer (including raw header
byte-offset checks against the BLUE ICD), the capture logic, and end-to-end
UDP/TCP loopback captures.

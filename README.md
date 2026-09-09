# v49writer

Receive VITA 49 (VRT) IQ streams over UDP or TCP and write them to Midas
BLUE files (X-Midas / REDHAWK compatible, type 1000) — one file per VRT
stream ID.

## Features

- **UDP** unicast or multicast (pass a multicast group as the host and it
  is joined automatically) and **TCP** (listen or connect). Both
  transports demultiplex by stream ID identically; TCP listen mode
  accepts any number of concurrent sender connections, all feeding the
  same capture.
- **One BLUE file per stream ID**: packets are demultiplexed by the VRT
  stream ID, and each stream gets its own file, sample-rate/format state,
  drop counter, and metadata keywords. Filter to a single stream with
  `--stream-id`.
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
- Detects dropped packets per stream via the VRT packet count and records
  the tally.
- **Stream-health diagnostics**: unsupported VRT packet types (extension
  data/context, command, reserved), non-UTC timestamps, payload formats
  BLUE can't represent, and payload sizes that don't match the data
  format are each warned about once (repeats at debug level, totals at
  close), so a misformatted stream is caught without flooding the log.
  `-v` adds per-stream debug detail (first-packet header fields, CIF
  parsing notes).
- Output directories are created automatically if they don't exist.
- Stop with Ctrl-C, `--duration`, or `--max-samples`; headers are
  finalized on close either way.

## Install

```sh
pip install .            # runtime (numpy only)
pip install -e '.[dev]'  # with test dependencies
```

## Usage

Capture VRT streams from UDP port 5000:

```sh
v49writer -p 5000 capture.tmp
```

The output argument is a path template: a stream with ID `0x1234` is
written to `capture_00001234.tmp` (a data stream with no stream ID goes to
`capture_nosid.tmp`). Two tokens are available:

- `{sid}` — the stream ID as 8 hex digits. Without this token the ID is
  appended to the file stem.
- `{freq}` — the stream's RF reference frequency in MHz at kHz resolution
  (e.g. `915.000MHz`), as parsed from its VRT context packets (`nofreq` if
  no context announces one). If the frequency only becomes known after the
  file is opened — context arriving after the first data packet, or a
  mid-capture retune — the finished file is renamed to match on close.

For example `v49writer -p 5000 'cap_{sid}_{freq}.tmp'` writes files like
`cap_00001234_915.000MHz.tmp`.

More examples:

```sh
# Join a multicast group, stop after 10 seconds
v49writer -H 239.1.2.3 -p 5000 -d 10 capture.tmp

# TCP: listen; multiple senders may connect concurrently
v49writer -t tcp -p 5000 capture.tmp

# TCP: connect out to a source
v49writer -t tcp --connect -H 10.0.0.5 -p 5000 capture.tmp

# Capture only stream 0x1234, exactly 1M samples, override the sample
# rate, and add custom keywords
v49writer -p 5000 -s 0x1234 -n 1000000 -r 10e6 -k MISSION=test1 capture.tmp

# Force 32-bit float complex payloads
v49writer -p 5000 -f cf capture.tmp
```

Data format codes (`--format`) are BLUE digraphs: first letter `c` (complex)
or `s` (scalar/real); second letter `b`=int8, `i`=int16, `l`=int32,
`x`=int64, `f`=float32, `d`=float64. The default `auto` uses the format
announced in each stream's VRT context packets, falling back to `ci`
(16-bit complex, the common VITA 49 IQ format).

### Test signal generator

`v49gen` sends standard-conformant VRT streams (complex tones plus context
packets) over UDP for testing. Repeat `--stream-id` to interleave multiple
streams (each stream's tone is offset by `--tone-step`):

```sh
v49gen -p 5000 -r 1e6 --tone-freq 100e3 -n 1000000
v49gen -p 5000 -s 0x11 -s 0x22 --throttle       # two streams, real-time paced
v49gen -t tcp -H 127.0.0.1 -p 5000              # over TCP
```

## Notes

- VRT payloads are big-endian per the VITA 49 standard; samples are
  byte-swapped to little-endian on write. For nonconformant sources use
  `--payload-endian little`.
- The BLUE `timecode` is seconds since 1950-01-01 (J1950), taken from each
  stream's first data packet carrying a UTC (TSI=1) timestamp; the
  fractional part uses real-time picosecond (TSF=2) timestamps when
  present.
- `--max-samples` applies per stream; the capture ends once every stream
  seen so far has reached the limit (or on Ctrl-C / `--duration`).

## Development

```sh
python -m pytest
```

The test suite covers the VRT parser, the BLUE writer (including raw header
byte-offset checks against the BLUE ICD), the per-stream capture logic, and
end-to-end two-stream UDP and TCP loopback captures.

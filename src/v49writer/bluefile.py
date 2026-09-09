"""Minimal Midas BLUE (X-Midas / BLUE 1.1) file writer.

Writes type 1000 (one-dimensional series) files with a little-endian
('EEEI') header and data representation, and an extended header holding
(key, value) keywords in standard X-Midas packed format.

The header control block layout and the extended-header keyword record
layout follow the BLUE file ICD as implemented by X-Midas and REDHAWK:

  keyword record: lkey(int32) lext(int16) ltag(int8) type(char)
                  <data bytes> <tag bytes> <pad to 8-byte multiple>

The J1950 epoch (1950-01-01T00:00:00 UTC) is used for the timecode field.
"""

from __future__ import annotations

import os
import struct
from typing import List, Optional, Tuple, Union

import numpy as np

HEADER_SIZE = 512
BLOCK = 512

# Seconds from the J1950 epoch (1950-01-01) to the UNIX epoch (1970-01-01):
# 7305 days.
J1950_TO_UNIX = 631152000.0

# BLUE format digraph -> numpy dtype (little-endian) for one scalar element.
# Complex formats store interleaved (real, imag) pairs of the element type.
_TYPE_DTYPE = {
    'B': np.dtype('<i1'),
    'I': np.dtype('<i2'),
    'L': np.dtype('<i4'),
    'X': np.dtype('<i8'),
    'F': np.dtype('<f4'),
    'D': np.dtype('<f8'),
}
_MODE_SIZE = {'S': 1, 'C': 2}


def format_info(fmt: str) -> Tuple[int, np.dtype, int]:
    """Return (elements_per_atom, element_dtype, bytes_per_atom) for a BLUE
    format digraph such as 'CI' or 'SF'."""
    fmt = fmt.upper()
    if len(fmt) != 2 or fmt[0] not in _MODE_SIZE or fmt[1] not in _TYPE_DTYPE:
        raise ValueError('unsupported BLUE format %r' % fmt)
    nelem = _MODE_SIZE[fmt[0]]
    dtype = _TYPE_DTYPE[fmt[1]]
    return nelem, dtype, nelem * dtype.itemsize


KeywordValue = Union[int, float, str, bytes]


def pack_keywords(keywords: List[Tuple[str, KeywordValue]]) -> bytes:
    """Pack (tag, value) pairs into X-Midas extended-header format
    (little-endian). Ints pack as 'L' (or 'X' when they exceed 32 bits),
    floats as 'D', str/bytes as 'A'."""
    out = bytearray()
    for tag, value in keywords:
        tag_b = tag.encode('ascii')
        if isinstance(value, bool):
            value = int(value)
        if isinstance(value, int):
            if -2**31 <= value < 2**31:
                data = struct.pack('<i', value)
                ktype = b'L'
            else:
                data = struct.pack('<q', value)
                ktype = b'X'
        elif isinstance(value, float):
            data = struct.pack('<d', value)
            ktype = b'D'
        elif isinstance(value, bytes):
            data = value
            ktype = b'A'
        else:
            data = str(value).encode('ascii')
            ktype = b'A'
        ldata = len(data)
        ltag = len(tag_b)
        lkey = ((ldata + ltag + 15) // 8) * 8   # 8-byte hdr, padded to 8
        lext = lkey - ldata
        pad = lkey - 8 - ldata - ltag
        out += struct.pack('<ihb', lkey, lext, ltag) + ktype
        out += data + tag_b + b'\0' * pad
    return bytes(out)


def unpack_keywords(buf: bytes) -> List[Tuple[str, KeywordValue]]:
    """Unpack little-endian X-Midas extended-header keywords."""
    out = []
    ii = 0
    while ii + 8 <= len(buf):
        lkey, lext, ltag = struct.unpack_from('<ihb', buf, ii)
        ktype = buf[ii + 7:ii + 8]
        ldata = lkey - lext
        if ldata < 0 or lkey <= 0 or ii + lkey > len(buf):
            raise ValueError('corrupt extended header at offset %d' % ii)
        data = buf[ii + 8:ii + 8 + ldata]
        tag = buf[ii + 8 + ldata:ii + 8 + ldata + ltag].decode('ascii')
        if ktype == b'L':
            value: KeywordValue = struct.unpack('<i', data)[0]
        elif ktype == b'X':
            value = struct.unpack('<q', data)[0]
        elif ktype == b'D':
            value = struct.unpack('<d', data)[0]
        elif ktype == b'F':
            value = struct.unpack('<f', data)[0]
        else:
            value = data.decode('ascii', 'replace')
        out.append((tag, value))
        ii += lkey
    return out


class BlueWriter:
    """Streaming writer for a type 1000 BLUE file.

    Usage:
        w = BlueWriter(path, fmt='CI')
        w.xdelta = 1.0 / sample_rate      # any time before close()
        w.timecode_unix = first_packet_utc_seconds
        w.write(raw_little_endian_bytes)
        ...
        w.set_keywords([('SAMPLE_RATE', 10e6), ...])
        w.close()

    A placeholder header is written on open; the real header and the
    extended header are written on close(), so an interrupted capture
    still leaves the raw data recoverable.
    """

    def __init__(self, path: str, fmt: str = 'CI',
                 xstart: float = 0.0, xdelta: float = 1.0,
                 xunits: int = 1):
        self.path = path
        self.fmt = fmt.upper()
        format_info(self.fmt)  # validate
        self.xstart = xstart
        self.xdelta = xdelta
        self.xunits = xunits   # 1 = time in seconds
        self.timecode_unix: Optional[float] = None
        self._keywords: List[Tuple[str, KeywordValue]] = []
        self._data_bytes = 0
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._fh = open(path, 'wb')
        self._fh.write(b'\0' * HEADER_SIZE)
        self._closed = False

    @property
    def data_bytes(self) -> int:
        return self._data_bytes

    def write(self, data: bytes) -> None:
        self._fh.write(data)
        self._data_bytes += len(data)

    def set_keywords(self, keywords: List[Tuple[str, KeywordValue]]) -> None:
        self._keywords = list(keywords)

    def _build_header(self, ext_start_blocks: int, ext_size: int) -> bytes:
        timecode = 0.0
        if self.timecode_unix is not None:
            timecode = self.timecode_unix + J1950_TO_UNIX
        keywords_main = b'\0'
        adjunct = struct.pack('<ddiiddii',
                              self.xstart, self.xdelta, self.xunits,
                              0, 0.0, 1.0, 0, 0)
        hdr = struct.pack(
            '<4s4s4siiiiiddi2shdhhiiidd8di92s256s',
            b'BLUE',            # version
            b'EEEI',            # head_rep
            b'EEEI',            # data_rep
            0,                  # detached
            0,                  # protected
            0,                  # pipe
            ext_start_blocks,   # ext_start (512-byte blocks)
            ext_size,           # ext_size (bytes)
            float(HEADER_SIZE),  # data_start
            float(self._data_bytes),  # data_size
            1000,               # type
            self.fmt.encode('ascii'),  # format
            0,                  # flagmask
            timecode,           # timecode (s since J1950)
            0, 0,               # inlet, outlets
            0,                  # outmask
            0, 0,               # pipeloc, pipesize
            0.0, 0.0,           # in_byte, out_byte
            *([0.0] * 8),       # outbytes
            len(keywords_main),  # keylength
            keywords_main.ljust(92, b'\0'),
            adjunct.ljust(256, b'\0'),
        )
        assert len(hdr) == HEADER_SIZE
        return hdr

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        ext = pack_keywords(self._keywords)
        ext_start_blocks = 0
        ext_size = len(ext)
        if ext_size:
            data_end = HEADER_SIZE + self._data_bytes
            ext_offset = ((data_end + BLOCK - 1) // BLOCK) * BLOCK
            self._fh.seek(0, 2)
            self._fh.write(b'\0' * (ext_offset - data_end))
            padded = ((ext_size + BLOCK - 1) // BLOCK) * BLOCK
            self._fh.write(ext.ljust(padded, b'\0'))
            ext_start_blocks = ext_offset // BLOCK
        self._fh.seek(0)
        self._fh.write(self._build_header(ext_start_blocks, ext_size))
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_header(path: str) -> dict:
    """Read a BLUE header (and extended-header keywords) for verification.
    Handles both EEEI and IEEE header representations."""
    with open(path, 'rb') as fh:
        raw = fh.read(HEADER_SIZE)
        if len(raw) < HEADER_SIZE:
            raise ValueError('file shorter than a BLUE header')
        head_rep = raw[4:8]
        endian = '<' if head_rep == b'EEEI' else '>'
        fields = struct.unpack(endian + '4s4s4siiiiiddi2shdhhiiidd8di92s',
                               raw[:256])
        hdr = {
            'version': fields[0].decode('ascii', 'replace'),
            'head_rep': fields[1].decode('ascii', 'replace'),
            'data_rep': fields[2].decode('ascii', 'replace'),
            'ext_start': fields[6],
            'ext_size': fields[7],
            'data_start': fields[8],
            'data_size': fields[9],
            'type': fields[10],
            'format': fields[11].decode('ascii', 'replace'),
            'timecode': fields[13],
        }
        adjunct = raw[256:]
        xstart, xdelta, xunits = struct.unpack(endian + 'ddi', adjunct[:20])
        hdr['xstart'] = xstart
        hdr['xdelta'] = xdelta
        hdr['xunits'] = xunits
        hdr['ext_header'] = []
        if hdr['ext_size'] > 0:
            fh.seek(hdr['ext_start'] * BLOCK)
            ext = fh.read(hdr['ext_size'])
            if endian == '<':
                hdr['ext_header'] = unpack_keywords(ext)
        return hdr


def read_data(path: str) -> np.ndarray:
    """Read the data portion of a type 1000 BLUE file written by this
    module. Complex formats are returned as numpy complex arrays."""
    hdr = read_header(path)
    nelem, dtype, _bpa = format_info(hdr['format'])
    if hdr['data_rep'] == 'IEEE':
        dtype = dtype.newbyteorder('>')
    with open(path, 'rb') as fh:
        fh.seek(int(hdr['data_start']))
        raw = fh.read(int(hdr['data_size']))
    arr = np.frombuffer(raw, dtype=dtype)
    if nelem == 2:
        arr = arr.astype(np.float64 if dtype.kind != 'f' else arr.dtype)
        arr = arr[0::2] + 1j * arr[1::2]
    return arr

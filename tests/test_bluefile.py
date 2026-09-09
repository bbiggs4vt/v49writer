import struct

import numpy as np
import pytest

from v49writer import bluefile


def test_header_layout(tmp_path):
    path = str(tmp_path / 'out.tmp')
    w = bluefile.BlueWriter(path, fmt='CI', xdelta=1e-6)
    w.timecode_unix = 1_700_000_000.5
    samples = np.arange(-8, 8, dtype='<i2').tobytes()
    w.write(samples)
    w.set_keywords([('SAMPLE_RATE', 1e6), ('VRT_STREAM_ID', 0x1234),
                    ('NOTE', 'hello')])
    w.close()

    with open(path, 'rb') as fh:
        raw = fh.read()
    # Spot-check raw field offsets against the BLUE ICD.
    assert raw[0:4] == b'BLUE'
    assert raw[4:8] == b'EEEI'
    assert raw[8:12] == b'EEEI'
    assert struct.unpack_from('<d', raw, 32)[0] == 512.0     # data_start
    assert struct.unpack_from('<d', raw, 40)[0] == 32.0      # data_size
    assert struct.unpack_from('<i', raw, 48)[0] == 1000      # type
    assert raw[52:54] == b'CI'                               # format
    timecode = struct.unpack_from('<d', raw, 56)[0]
    assert timecode == pytest.approx(1_700_000_000.5 + bluefile.J1950_TO_UNIX)
    xstart, xdelta, xunits = struct.unpack_from('<ddi', raw, 256)
    assert xstart == 0.0
    assert xdelta == 1e-6
    assert xunits == 1
    # data follows the header
    assert raw[512:544] == samples


def test_read_header_and_keywords(tmp_path):
    path = str(tmp_path / 'out.tmp')
    with bluefile.BlueWriter(path, fmt='CF', xdelta=0.5) as w:
        w.write(np.array([1.0, -1.0, 2.0, -2.0], dtype='<f4').tobytes())
        w.set_keywords([('SAMPLE_RATE', 2.0), ('COUNT', 3),
                        ('BIG', 2**40), ('NAME', 'tone')])
    hdr = bluefile.read_header(path)
    assert hdr['version'] == 'BLUE'
    assert hdr['type'] == 1000
    assert hdr['format'] == 'CF'
    assert hdr['xdelta'] == 0.5
    assert hdr['data_size'] == 16.0
    kw = dict(hdr['ext_header'])
    assert kw['SAMPLE_RATE'] == 2.0
    assert kw['COUNT'] == 3
    assert kw['BIG'] == 2**40
    assert kw['NAME'] == 'tone'
    # ext header starts on a 512-byte block boundary
    assert hdr['ext_start'] * 512 >= 512 + hdr['data_size']


def test_keyword_record_structure():
    packed = bluefile.pack_keywords([('FS', 1.5)])
    # 8-byte descriptor + 8-byte double + 2-byte tag -> padded to 24
    lkey, lext, ltag = struct.unpack_from('<ihb', packed, 0)
    assert lkey == 24
    assert lkey % 8 == 0
    assert lext == lkey - 8   # ldata = 8
    assert ltag == 2
    assert packed[7:8] == b'D'
    assert struct.unpack_from('<d', packed, 8)[0] == 1.5
    assert packed[16:18] == b'FS'
    assert len(packed) == lkey
    assert bluefile.unpack_keywords(packed) == [('FS', 1.5)]


def test_keywords_roundtrip_many():
    kws = [('A', 1), ('LONG_TAG_NAME_HERE', -2.5), ('S', 'x' * 37),
           ('NEG', -1), ('HUGE', -2**40)]
    assert bluefile.unpack_keywords(bluefile.pack_keywords(kws)) == kws


def test_read_data_complex(tmp_path):
    path = str(tmp_path / 'out.tmp')
    with bluefile.BlueWriter(path, fmt='CI') as w:
        w.write(np.array([100, -200, 300, -400], dtype='<i2').tobytes())
    data = bluefile.read_data(path)
    np.testing.assert_array_equal(data, [100 - 200j, 300 - 400j])


def test_ext_header_padding(tmp_path):
    # Data size not a multiple of 512: ext header must be block-aligned.
    path = str(tmp_path / 'out.tmp')
    with bluefile.BlueWriter(path, fmt='SB') as w:
        w.write(b'\x01' * 100)
        w.set_keywords([('K', 1)])
    hdr = bluefile.read_header(path)
    assert hdr['data_size'] == 100.0
    assert hdr['ext_start'] == 2   # 512 (header) + 100 -> next block = 1024
    assert dict(hdr['ext_header'])['K'] == 1


def test_bad_format_rejected(tmp_path):
    with pytest.raises(ValueError):
        bluefile.BlueWriter(str(tmp_path / 'x.tmp'), fmt='ZZ')

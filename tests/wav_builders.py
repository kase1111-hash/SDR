"""WAV files packed by hand from the specifications, for the reader's tests.

Microsoft's WAVEFORMATEX / WAVEFORMATEXTENSIBLE (integer PCM: 8-bit
unsigned, wider samples signed and little-endian, the bits left-justified in
their container; IEEE floating point: format 3), in a classic RIFF file, in
RF64 (EBU Tech 3306) and in Sony Wave64. Nothing here uses the code under
test, so a reader and this writer can't agree on a mistake they share.
"""

import struct

import numpy as np

#: KSDATAFORMAT_SUBTYPE_PCM and _IEEE_FLOAT as a file stores them.
SUBTYPE_PCM = bytes.fromhex("0100000000001000800000aa00389b71")
SUBTYPE_FLOAT = bytes.fromhex("0300000000001000800000aa00389b71")
_W64_TAIL = bytes.fromhex("f3acd3118cd100c04f8edb8a")


def full_scale(width):
    """The largest value of a signed ``width``-byte sample."""
    return 2 ** (8 * width - 1) - 1


def encode(values, kind="int", width=2):
    """Interleaved ``values`` (floats, full scale 1.0) as a WAV file stores
    them: integers rounded to the nearest step (8-bit unsigned, 128 is
    zero), floating point as it is."""
    values = np.asarray(values, dtype=np.float64)
    if kind == "float":
        return values.astype("<f4" if width == 4 else "<f8").tobytes()
    if width == 1:
        return np.clip(np.rint(values * 128 + 128), 0, 255).astype(np.uint8).tobytes()
    top = full_scale(width)
    ints = np.clip(np.rint(values * top), -top - 1, top).astype(np.int64)
    return integers(ints, width)


def integers(ints, width):
    """Signed integers as ``width``-byte little-endian samples."""
    ints = np.asarray(ints, dtype=np.int64)
    if width == 2:
        return ints.astype("<i2").tobytes()
    if width == 4:
        return ints.astype("<i4").tobytes()
    if width == 8:
        return ints.astype("<i8").tobytes()
    low = (ints & 0xFFFFFF).astype("<u4").view(np.uint8).reshape(-1, 4)
    return low[:, :3].tobytes()


def interleave(x):
    """Complex ``x`` as I, Q, I, Q... (real ``x`` as it is)."""
    x = np.asarray(x)
    if not np.iscomplexobj(x):
        return x.astype(np.float64)
    out = np.empty(2 * len(x), np.float64)
    out[0::2], out[1::2] = x.real, x.imag
    return out


def fmt_body(
    channels, rate, width, kind="int", extensible=False, bits=None, valid=None
):
    """A ``fmt `` chunk's body: WAVEFORMATEX, or WAVEFORMATEXTENSIBLE with
    the PCM or float subformat (``valid``: its valid bits per sample)."""
    bits = 8 * width if bits is None else bits
    tag = 0xFFFE if extensible else (3 if kind == "float" else 1)
    body = struct.pack(
        "<HHIIHH", tag, channels, rate, rate * channels * width, channels * width, bits
    )
    if extensible:
        sub = SUBTYPE_FLOAT if kind == "float" else SUBTYPE_PCM
        mask = 3 if channels == 2 else 4
        body += struct.pack("<HHI", 22, bits if valid is None else valid, mask) + sub
    elif kind == "float":
        body += struct.pack("<H", 0)  # (cbSize, as format 3 has)
    return body


def chunk(ckid, body):
    pad = b"\0" if len(body) % 2 else b""
    return ckid + struct.pack("<I", len(body)) + body + pad


def write_wav(path, fmt, data, container="RIFF"):
    """Write a WAV file holding the ``fmt `` body and the samples ``data``
    (bytes) as a classic RIFF file, RF64 (its 32-bit sizes -1, the real
    ones in ds64) or Sony Wave64 ("W64"). Returns where the samples start."""
    if container == "W64":
        fmt_ck = b"fmt " + _W64_TAIL + struct.pack("<Q", 24 + len(fmt)) + fmt
        fmt_ck += b"\0" * (-len(fmt_ck) % 8)
        data_ck = b"data" + _W64_TAIL + struct.pack("<Q", 24 + len(data))
        riff = b"riff" + bytes.fromhex("2e91cf11a5d628db04c10000")
        wave_id = b"wave" + _W64_TAIL
        size = 40 + len(fmt_ck) + len(data_ck) + len(data)
        head = riff + struct.pack("<Q", size) + wave_id + fmt_ck + data_ck
    elif container == "RF64":
        rest = chunk(b"fmt ", fmt) + b"data" + struct.pack("<I", 0xFFFFFFFF)
        offset = 12 + 36 + len(rest)
        ds64 = b"ds64" + struct.pack(
            "<IQQQI", 28, offset + len(data) - 8, len(data), 0, 0
        )
        head = b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + ds64 + rest
    else:
        chunks = chunk(b"fmt ", fmt) + b"data" + struct.pack("<I", len(data))
        size = 4 + len(chunks) + len(data) + (len(data) & 1)
        head = b"RIFF" + struct.pack("<I", size) + b"WAVE" + chunks
    with open(path, "wb") as handle:
        handle.write(
            head + data + (b"\0" if container == "RIFF" and len(data) & 1 else b"")
        )
    return len(head)


def iq_wav(
    path, x, kind="int", width=2, rate=48000, container="RIFF", extensible=False
):
    """``x`` (complex: two channels, I and Q; real: one channel) as a WAV
    file of ``width``-byte ``kind`` samples. Returns where they start."""
    channels = 2 if np.iscomplexobj(x) else 1
    fmt = fmt_body(channels, rate, width, kind, extensible)
    return write_wav(path, fmt, encode(interleave(x), kind, width), container)

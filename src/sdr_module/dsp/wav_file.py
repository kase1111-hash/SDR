"""
WAV files of any size: classic RIFF up to 4 GiB, RF64 beyond.

A classic WAV file (RIFF) counts its sizes in 32 bits, so it holds 4 GiB at
most: about 7 minutes of 16-bit I/Q at 2.4 MS/s (Python's ``wave`` module
refuses more). RF64 (EBU Tech 3306; ITU-R BS.2088 calls it BW64) is the same
file with ``RF64`` in place of ``RIFF``, its 32-bit sizes set to 0xFFFFFFFF
and the real ones, in 64 bits, in a ``ds64`` chunk ahead of the others.

:class:`WavWriter` writes integer PCM samples: a classic file while they
fit, RF64 once they don't. A file whose size is known to fit gets the
classic 44-byte header (the ``wave`` module's, byte for byte), readable
everywhere; any other begins with a ``JUNK`` chunk where ``ds64`` goes (as
EBU Tech 3306 recommends), which WAV readers skip, and becomes RF64 in
place once it passes 4 GiB. The header is brought up to date after every
write, so a file cut short (a crash) still reads to its last write.

:func:`read_wav_header` reads classic, RF64 and BW64 files, and Sony Wave64
(``.w64``): how the samples are stored and where they lie. A size larger
than the file (a recording cut short), an unknown one (0xFFFFFFFF, written
while streaming; in RF64 or Wave64 also a size never filled in) and a
classic 32-bit size that went round past 4 GiB (a writer that knows no
RF64) are all read as what the file holds.
"""

from __future__ import annotations

import os
import struct
import wave
from dataclasses import dataclass
from typing import IO, Any, Dict, Optional, Tuple, Union

#: The largest number a 32-bit size field holds; RF64 writes it in place of
#: the sizes its ds64 chunk gives.
SIZE_32_MAX = 0xFFFFFFFF

#: The largest RIFF size (the file's size less 8 bytes) a classic WAV file
#: declares: :class:`WavWriter` writes RF64 past it. (Tests lower it, to
#: make a short file "over 4 GiB".)
RIFF_SIZE_LIMIT = SIZE_32_MAX

#: The bytes ahead of the samples in a classic file (its RIFF, fmt and data
#: headers), and in one with room for ds64 (a 36-byte JUNK chunk more).
CLASSIC_HEADER_BYTES = 44
RF64_HEADER_BYTES = 80

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

#: What every KSDATAFORMAT_SUBTYPE GUID ends with (as stored); the two
#: bytes before it are the subformat's format tag (1: PCM, 3: float).
_SUBTYPE_TAIL = bytes.fromhex("000000001000800000aa00389b71")

# Sony Wave64's ids (GUIDs, as stored): its RIFF and WAVE, fmt and data.
_W64_TAIL = bytes.fromhex("f3acd3118cd100c04f8edb8a")
W64_RIFF = b"riff" + bytes.fromhex("2e91cf11a5d628db04c10000")
W64_WAVE = b"wave" + _W64_TAIL
W64_FMT = b"fmt " + _W64_TAIL
W64_DATA = b"data" + _W64_TAIL

_RIFF_IDS = (b"RIFF", b"RF64", b"BW64")
_DS64_BODY = 28  # RIFF size, data size, sample count (64-bit), table length
_MAX_CHUNKS = 4096  # (looked at before the data: a damaged file ends)
_MAX_READ = 1 << 16  # (bytes of a fmt or ds64 chunk read)

#: A file to read or write: its path, or a binary file open on it.
Source = Union[str, "os.PathLike[str]", IO[bytes]]


class WavError(wave.Error, ValueError):
    """Not a WAV file this module reads, or samples it can't write."""


@dataclass(frozen=True)
class WavHeader:
    """How a WAV file's samples are stored and where they lie."""

    #: "RIFF" (a classic WAV file), "RF64", "BW64" or "W64" (Sony Wave64).
    container: str
    channels: int
    #: Bytes of one channel's sample (1: 8-bit, 2: 16-bit).
    sample_width: int
    #: Frames (a sample of every channel) a second.
    frame_rate: int
    #: Where the first sample is, from the start of the file.
    data_offset: int
    #: Bytes of samples the file holds.
    data_bytes: int

    @property
    def frame_bytes(self) -> int:
        return self.channels * self.sample_width

    @property
    def frames(self) -> int:
        """Whole frames the file holds."""
        return self.data_bytes // self.frame_bytes


@dataclass
class _Found:
    """What a look through a WAV file's chunks found."""

    container: str
    #: (format tag, channels, frame rate, bits per sample).
    fmt: Tuple[int, int, int, int]
    data_offset: int
    #: The data's size as the file says it (None: unknown, to the end).
    declared: Optional[int]
    file_size: int


def looks_like_wav(head: bytes) -> bool:
    """Whether a file whose first bytes are ``head`` (16 or more) starts
    like a WAV file: classic, RF64, BW64 or Wave64. (A RIFF file of another
    kind, an AVI, starts the same: :func:`read_wav_header` tells.)"""
    return head[:4] in _RIFF_IDS or head[:16] == W64_RIFF


def is_wav_file(path: Union[str, "os.PathLike[str]"]) -> bool:
    """Whether ``path`` starts like a WAV file (see :func:`looks_like_wav`),
    whatever its name; False when it can't be read."""
    try:
        with open(path, "rb") as handle:
            return looks_like_wav(handle.read(16))
    except OSError:
        return False


def fits_classic(data_bytes: int) -> bool:
    """Whether ``data_bytes`` of samples fit in a classic WAV file (with its
    44-byte header): 4 GiB less 36 bytes."""
    data = int(data_bytes)
    return CLASSIC_HEADER_BYTES - 8 + data + (data & 1) <= RIFF_SIZE_LIMIT


def read_wav_header(source: Source) -> WavHeader:
    """How ``source``'s samples are stored and where they lie (a path, or a
    binary file open on a WAV file, read from its start).

    Raises WavError (a ValueError) for a file that isn't WAV or whose
    samples aren't integer PCM, and OSError when it can't be read.
    """
    handle, own = _opened(source)
    try:
        found = _scan(handle)
    finally:
        if own:
            handle.close()
    tag, channels, rate, bits = found.fmt
    if tag == WAVE_FORMAT_IEEE_FLOAT:
        raise WavError(
            "its samples are floating point (WAV format 3), and only integer "
            "(PCM) WAV files are read"
        )
    if tag != WAVE_FORMAT_PCM:
        kind = f"WAV format {tag}" if tag >= 0 else "an extensible WAV format"
        raise WavError(
            f"its samples are in {kind}, not integer PCM, and only PCM WAV files "
            "are read"
        )
    width = (bits + 7) // 8
    if channels < 1 or width < 1:
        raise WavError("its format (fmt) chunk is damaged: no channels or samples")
    return WavHeader(
        container=found.container,
        channels=channels,
        sample_width=width,
        frame_rate=rate,
        data_offset=found.data_offset,
        data_bytes=_data_bytes(found),
    )


def data_offset(source: Source) -> int:
    """Where ``source``'s samples start (classic, RF64 or Wave64, whatever
    format they are in), or 0 when it isn't a WAV file or can't be read."""
    try:
        handle, own = _opened(source)
    except OSError:
        return 0
    try:
        return _scan(handle).data_offset
    except (OSError, ValueError, struct.error):
        return 0
    finally:
        if own:
            handle.close()


def _opened(source: Source) -> Tuple[IO[bytes], bool]:
    """``source`` as a binary file, and whether it was opened here."""
    if isinstance(source, (str, os.PathLike)):
        return open(source, "rb"), True
    return source, False


def _scan(handle: IO[bytes]) -> _Found:
    """Find the format and the samples of the WAV file in ``handle``."""
    size = handle.seek(0, os.SEEK_END)
    handle.seek(0)
    head = handle.read(40)
    if head[:4] in _RIFF_IDS:
        if head[8:12] != b"WAVE":
            raise WavError("it is a RIFF file of another kind, not WAV")
        return _scan_riff(handle, head[:4].decode("ascii"), size)
    if head[:16] == W64_RIFF and head[24:40] == W64_WAVE:
        return _scan_w64(handle, size)
    raise WavError("it doesn't start like a WAV file (RIFF, RF64 or Wave64)")


def _scan_riff(handle: IO[bytes], container: str, size: int) -> _Found:
    """A classic, RF64 or BW64 file: its chunks after ``WAVE``. In RF64 a
    size of 0xFFFFFFFF is in ds64: the data's, or another chunk's in its
    table."""
    big = container != "RIFF"
    ds64_data: Optional[int] = None
    table: Dict[bytes, int] = {}
    fmt: Optional[Tuple[int, int, int, int]] = None
    pos = 12
    for _ in range(_MAX_CHUNKS):
        handle.seek(pos)
        header = handle.read(8)
        if len(header) < 8:
            break
        ckid = header[:4]
        cksize = struct.unpack("<I", header[4:])[0]
        if ckid == b"ds64" and big:
            raw = handle.read(min(cksize, _MAX_READ))
            if len(raw) >= 24:
                ds64_data = struct.unpack_from("<Q", raw, 8)[0]
            if len(raw) >= _DS64_BODY:
                entries = struct.unpack_from("<I", raw, 24)[0]
                for i in range(min(entries, (len(raw) - _DS64_BODY) // 12)):
                    name, length = struct.unpack_from("<4sQ", raw, _DS64_BODY + 12 * i)
                    table[name] = length
        elif ckid == b"fmt ":
            fmt = _parse_fmt(handle.read(min(cksize, _MAX_READ)))
        elif ckid == b"data":
            if fmt is None:
                raise WavError("its samples come before their format (fmt) chunk")
            declared: Optional[int] = cksize
            if cksize == SIZE_32_MAX:
                # (In RF64, ds64's; 0 there is a size never filled in, as by
                # a writer that stopped before it finished: to the end too.)
                declared = (ds64_data or None) if big else None
            elif big and ds64_data is not None and ds64_data & SIZE_32_MAX == cksize:
                declared = ds64_data  # (a writer that gave the low 32 bits)
            return _Found(container, fmt, pos + 8, declared, size)
        if big and cksize == SIZE_32_MAX:
            if ckid not in table:
                raise WavError(f"the size of its {ckid!r} chunk is missing (RF64)")
            cksize = table[ckid]
        pos += 8 + cksize + (cksize & 1)
    if fmt is None:
        raise WavError("it has no format (fmt) chunk")
    raise WavError("it holds no samples (no data chunk)")


def _scan_w64(handle: IO[bytes], size: int) -> _Found:
    """A Sony Wave64 file: GUIDs for chunk ids, 64-bit sizes counting the
    chunk's own 24-byte header, chunks on 8-byte boundaries."""
    fmt: Optional[Tuple[int, int, int, int]] = None
    pos = 40
    for _ in range(_MAX_CHUNKS):
        handle.seek(pos)
        header = handle.read(24)
        if len(header) < 24:
            break
        guid = header[:16]
        cksize = struct.unpack("<Q", header[16:])[0]
        if guid == W64_DATA:
            if fmt is None:
                raise WavError("its samples come before their format (fmt) chunk")
            # (A size short of its own header was never filled in: to the end.)
            declared = cksize - 24 if cksize >= 24 else None
            return _Found("W64", fmt, pos + 24, declared, size)
        if cksize < 24:
            raise WavError("a chunk of this Wave64 file is damaged")
        if guid == W64_FMT:
            fmt = _parse_fmt(handle.read(min(cksize - 24, _MAX_READ)))
        pos += (cksize + 7) & ~7
    if fmt is None:
        raise WavError("it has no format (fmt) chunk")
    raise WavError("it holds no samples (no data chunk)")


def _parse_fmt(raw: bytes) -> Tuple[int, int, int, int]:
    """(format tag, channels, frame rate, bits per sample) of a ``fmt ``
    chunk; an extensible format's tag is its subformat's (-1: not one of
    the standard subformats)."""
    if len(raw) < 16:
        raise WavError("its format (fmt) chunk is cut short")
    tag, channels, rate, _byte_rate, _align, bits = struct.unpack_from("<HHIIHH", raw)
    if tag == WAVE_FORMAT_EXTENSIBLE:
        if len(raw) < 40:
            raise WavError("its format (fmt) chunk is cut short")
        sub = raw[24:40]
        tag = struct.unpack_from("<H", sub)[0] if sub[2:] == _SUBTYPE_TAIL else -1
    return tag, channels, rate, bits


def _data_bytes(found: _Found) -> int:
    """The bytes of samples the file holds: as it says, except a size it
    doesn't hold (cut short), an unknown one (to the end) and a classic
    32-bit size that went round past 4 GiB (the rest of the file)."""
    available = max(0, found.file_size - found.data_offset)
    declared = found.declared
    if declared is None or declared > available:
        return available
    if (
        found.container == "RIFF"
        and available > SIZE_32_MAX
        and (available - declared) % (SIZE_32_MAX + 1) == 0
    ):
        return available
    return declared


class WavWriter:
    """Writes integer PCM samples to a WAV file: a classic one while they
    fit, RF64 past 4 GiB (see the module docstring).

    ``target`` is a path, or a binary file open for writing where the WAV
    file is to begin (it is left open). ``expected_bytes`` is how many bytes
    of samples will be written, when known: a file that fits a classic WAV
    file then gets its 44-byte header (and must stay within it); without it,
    or past the limit, room is kept for RF64's ds64 chunk.

    Raises ValueError for a format a WAV header can't hold, OSError when the
    file can't be written (a file created here is removed again).
    """

    def __init__(
        self,
        target: Source,
        channels: int,
        sample_width: int,
        frame_rate: int,
        expected_bytes: Optional[int] = None,
    ):
        channels, width, rate = int(channels), int(sample_width), int(frame_rate)
        if channels < 1 or not 1 <= width <= 4:
            raise ValueError(
                f"a WAV file can't hold {channels} channels of {width}-byte samples"
            )
        byte_rate = rate * channels * width
        if rate <= 0 or byte_rate > SIZE_32_MAX:
            raise ValueError(f"a WAV file can't hold a sample rate of {rate} Hz")
        self._frame_bytes = channels * width
        self._reserved = expected_bytes is None or not fits_classic(expected_bytes)
        fmt = struct.pack(
            "<4sIHHIIHH",
            b"fmt ",
            16,
            WAVE_FORMAT_PCM,
            channels,
            rate,
            byte_rate,
            self._frame_bytes,
            8 * width,
        )
        if self._reserved:
            riff = b"RIFF" + struct.pack("<I", RF64_HEADER_BYTES - 8) + b"WAVE"
            riff += b"JUNK" + struct.pack("<I", _DS64_BODY) + bytes(_DS64_BODY)
        else:
            riff = b"RIFF" + struct.pack("<I", CLASSIC_HEADER_BYTES - 8) + b"WAVE"
        header = riff + fmt + b"data" + struct.pack("<I", 0)
        path: Optional[Union[str, "os.PathLike[str]"]] = None
        if isinstance(target, (str, os.PathLike)):
            path = target
            handle: IO[bytes] = open(path, "wb")
        else:
            handle = target
        self._own = path is not None
        try:
            self._base = handle.tell()
            handle.write(header)
        except BaseException:
            if path is not None:
                handle.close()
                try:
                    os.remove(path)
                except OSError:
                    pass
            raise
        self._file: Optional[IO[bytes]] = handle
        self._data_start = self._base + len(header)
        self._data_bytes = 0
        self._rf64 = False

    @property
    def container(self) -> str:
        """ "RIFF" (a classic file), or "RF64" once past 4 GiB."""
        return "RF64" if self._rf64 else "RIFF"

    @property
    def data_bytes(self) -> int:
        """Bytes of samples written."""
        return self._data_bytes

    @property
    def frames_written(self) -> int:
        return self._data_bytes // self._frame_bytes

    def writeframes(self, data: Any) -> None:
        """Add ``data``: samples as the file stores them (little-endian
        integer PCM, whole frames), as bytes or any buffer (a NumPy array).
        The header is up to date afterwards."""
        handle = self._file
        if handle is None:
            raise ValueError("the WAV file is closed")
        view = memoryview(data).cast("B")
        if not self._reserved and not fits_classic(self._data_bytes + view.nbytes):
            raise WavError(
                "more samples than expected: a classic WAV file can't hold them"
            )
        handle.write(view)
        # (Counted from where the file is: what is in it.)
        self._data_bytes = handle.tell() - self._data_start
        self._update(handle)

    def close(self) -> None:
        """Finish the file: a pad byte after an odd number of bytes (RIFF
        chunks take whole 16-bit words) and the header's final sizes. Closes
        the file if it was opened here; a second call does nothing."""
        handle, self._file = self._file, None
        if handle is None:
            return
        try:
            pad = self._data_bytes & 1
            if pad:
                handle.seek(self._data_start + self._data_bytes)
                handle.write(b"\0")
            self._update(handle, pad)
            handle.flush()
        finally:
            if self._own:
                handle.close()

    def __enter__(self) -> "WavWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _update(self, handle: IO[bytes], pad: int = 0) -> None:
        """Bring the header up to date with the samples written (and the
        pad byte after them), making the file RF64 once past the limit: the
        64-bit sizes first, then the ids, so the file reads at every step."""
        data = self._data_bytes
        base = self._base
        riff = self._data_start - base - 8 + data + pad
        promote = not self._rf64 and riff > RIFF_SIZE_LIMIT
        if self._rf64 or promote:
            handle.seek(base + 20)  # (the ds64 chunk's body, or the JUNK's)
            handle.write(struct.pack("<QQQ", riff, data, data // self._frame_bytes))
        if promote:
            # (Only a file with room for ds64 gets here: writeframes keeps a
            # classic one within its limit.)
            handle.seek(base + 12)
            handle.write(b"ds64")
            handle.seek(self._data_start - 4)
            handle.write(struct.pack("<I", SIZE_32_MAX))
            handle.seek(base)
            handle.write(b"RF64" + struct.pack("<I", SIZE_32_MAX))
            self._rf64 = True
        elif not self._rf64:
            handle.seek(base + 4)
            handle.write(struct.pack("<I", riff))
            handle.seek(self._data_start - 4)
            handle.write(struct.pack("<I", data))
        handle.seek(self._data_start + data + pad)

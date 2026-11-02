"""K-R1: WAV recordings of any size (RF64 past 4 GiB, EBU Tech 3306).

Before, a WAV file held 4 GiB at most (about 7 minutes of 16-bit I/Q at
2.4 MS/s): its RIFF header counts sizes in 32 bits and Python's wave module
refuses more, so a longer take couldn't be saved as WAV, and an RF64 file,
what other programs write past 4 GiB, couldn't even be opened ("file does
not start with RIFF id"). Now a WAV file past 4 GiB is written as RF64, and
RF64, BW64 and Sony Wave64 files are read; a smaller WAV file of known size
is still a classic one, byte for byte what the wave module writes.

Headers are checked against layouts packed here by hand from the
specifications, not by the code under test. A file over 4 GiB is checked
twice: in a sparse file kept in memory (every platform), and on the disk as
a sparse file where the file system makes one (only its first and last
samples then take room).
"""

import io
import os
import re
import struct
import wave
from pathlib import Path

import numpy as np
import pytest

from sdr_module.dsp import wav_file
from sdr_module.dsp.recording import (
    AudioPlayer,
    AudioRecorder,
    AudioSampleFormat,
    FileFormat,
    IQPlayer,
    IQRecorder,
    SampleFormat,
    check_iq_samples,
    detect_format,
    get_file_info,
    load_audio_file,
    load_iq_file,
    save_audio_file,
    save_iq_file,
    sidecar_path,
    wav_data_offset,
    write_sidecar,
)

RATE = 2_400_000
GiB = 1 << 30
#: A cs16 frame (I and Q) in bytes.
FRAME = 4
#: One step of a 16-bit sample: what rounding to cs16 can change.
STEP = 1.0 / 32767


def _tone(n, cycles=0.01, amplitude=0.5):
    t = np.arange(n)
    return (amplitude * np.exp(2j * np.pi * cycles * t)).astype(np.complex64)


def _cs16(x):
    """Interleaved little-endian 16-bit I/Q as a WAV file stores it
    (rounded in float64, as IQRecorder rounds)."""
    v = np.empty(2 * len(x), "<i2")
    v[0::2] = np.clip(np.rint(x.real.astype(np.float64) * 32767), -32768, 32767)
    v[1::2] = np.clip(np.rint(x.imag.astype(np.float64) * 32767), -32768, 32767)
    return v.tobytes()


def _fmt(channels=2, width=2, rate=RATE, tag=1):
    """A PCM ``fmt `` chunk's body (WAVEFORMAT + bits per sample)."""
    return struct.pack(
        "<HHIIHH",
        tag,
        channels,
        rate,
        rate * channels * width,
        channels * width,
        8 * width,
    )


def _chunk(ckid, body):
    pad = b"\0" if len(body) % 2 else b""
    return ckid + struct.pack("<I", len(body)) + body + pad


def _rf64(path, frames, head=b"", tail=b"", magic=b"RF64", extra=b"", rate=RATE):
    """An RF64 file as EBU Tech 3306 lays it out: ``ds64`` first, then
    ``fmt ``, the ``extra`` chunks and ``data`` with its 32-bit size -1,
    holding ``frames`` cs16 frames: ``head`` at the start, ``tail`` at the
    end, the rest a hole (sparse where the file system makes holes).
    Returns where the samples start."""
    data_bytes = frames * FRAME
    rest = _chunk(b"fmt ", _fmt(rate=rate)) + extra
    rest += b"data" + struct.pack("<I", 0xFFFFFFFF)
    offset = 12 + 36 + len(rest)
    ds64 = b"ds64" + struct.pack(
        "<IQQQI", 28, offset + data_bytes - 8, data_bytes, frames, 0
    )
    with open(path, "wb") as handle:
        handle.write(magic + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + ds64 + rest)
        handle.write(head)
        if tail:
            handle.seek(offset + data_bytes - len(tail))
            handle.write(tail)
        else:
            handle.truncate(offset + data_bytes)
    return offset


def _check_rf64(raw, x, rate=RATE):
    """``raw`` is an RF64 file as EBU Tech 3306 lays it out (as this app
    writes it: ds64 where the classic file's JUNK was), holding ``x``."""
    n = len(x)
    assert raw[:4] == b"RF64"
    assert struct.unpack("<I", raw[4:8])[0] == 0xFFFFFFFF
    assert raw[8:12] == b"WAVE"
    assert raw[12:16] == b"ds64"
    size, riff, data, frames, table = struct.unpack("<IQQQI", raw[16:48])
    assert (size, riff, data, frames, table) == (28, len(raw) - 8, FRAME * n, n, 0)
    assert raw[48:56] == b"fmt " + struct.pack("<I", 16)
    assert raw[56:72] == _fmt(rate=rate)
    assert raw[72:76] == b"data"
    assert struct.unpack("<I", raw[76:80])[0] == 0xFFFFFFFF
    assert raw[80:] == _cs16(x)


@pytest.fixture
def small_limit(monkeypatch):
    """Classic WAV files hold 4000 bytes of samples meanwhile (4 GiB
    really), so a few thousand samples are "over 4 GiB"."""
    monkeypatch.setattr(wav_file, "RIFF_SIZE_LIMIT", 36 + 4000)
    return 4000


class _SparseFile:
    """A file of any size in memory: only the pages written are kept, the
    rest (its holes) reads as zeros, as a sparse file on disk does."""

    PAGE = 1 << 16

    def __init__(self):
        self.pages = {}
        self.pos = 0
        self.size = 0

    def write(self, data):
        data = memoryview(data).cast("B")
        done = 0
        while done < len(data):
            page, at = divmod(self.pos, self.PAGE)
            room = min(self.PAGE - at, len(data) - done)
            buf = self.pages.setdefault(page, bytearray(self.PAGE))
            buf[at : at + room] = data[done : done + room]
            done += room
            self.pos += room
        self.size = max(self.size, self.pos)
        return done

    def read(self, n=-1):
        end = self.size if n is None or n < 0 else min(self.size, self.pos + n)
        out = bytearray()
        while self.pos < end:
            page, at = divmod(self.pos, self.PAGE)
            room = min(self.PAGE - at, end - self.pos)
            buf = self.pages.get(page)
            out += buf[at : at + room] if buf is not None else bytes(room)
            self.pos += room
        return bytes(out)

    def seek(self, offset, whence=os.SEEK_SET):
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self.pos, os.SEEK_END: self.size}
        self.pos = base[whence] + offset
        return self.pos

    def tell(self):
        return self.pos

    def flush(self):
        pass


def _sparse_files_here(folder):
    """Whether files in ``folder`` can have holes (no room taken on the
    disk): ext4, XFS, Btrfs, APFS do; NTFS doesn't unless asked to."""
    probe = Path(folder) / "sparse.probe"
    try:
        with open(probe, "wb") as handle:
            handle.seek(64 << 20)
            handle.write(b"x")
        blocks = getattr(os.stat(probe), "st_blocks", None)
        return blocks is not None and blocks * 512 < (1 << 20)
    finally:
        probe.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


class TestSmallFilesStayClassic:
    """A WAV file that fits is a classic one: read everywhere."""

    @pytest.mark.parametrize(
        "fmt, width", [(SampleFormat.INT16, 2), (SampleFormat.UINT8, 1)]
    )
    def test_known_size_is_the_wave_module_s_file_byte_for_byte(
        self, tmp_path, fmt, width
    ):
        x = _tone(1000)
        ours = tmp_path / "ours.wav"
        save_iq_file(ours, x, RATE, 0.0, fmt, FileFormat.WAV)
        if width == 2:
            frames = _cs16(x)
        else:
            v = np.empty(2 * len(x), np.uint8)
            for part, values in ((0, x.real), (1, x.imag)):
                scaled = values.astype(np.float64) * 127.5 + 127.5
                v[part::2] = np.clip(np.rint(scaled), 0, 255)
            frames = v.tobytes()
        reference = tmp_path / "reference.wav"
        with wave.open(str(reference), "wb") as handle:
            handle.setnchannels(2)
            handle.setsampwidth(width)
            handle.setframerate(RATE)
            handle.writeframes(frames)
        assert ours.read_bytes() == reference.read_bytes()

    def test_a_recording_of_unknown_length_keeps_room_for_rf64(self, tmp_path):
        """Written as it comes (RecordingSession), its size unknown: a JUNK
        chunk first, where ds64 would go (EBU Tech 3306), which WAV
        readers skip."""
        path = tmp_path / "stream.wav"
        recorder = IQRecorder(RATE, 100e6, SampleFormat.INT16, FileFormat.WAV)
        recorder.start(path)
        x = _tone(3000)
        for part in np.array_split(x, 3):
            recorder.write(part)
        recorder.stop()
        raw = path.read_bytes()
        assert raw[:4] == b"RIFF" and raw[8:12] == b"WAVE"
        assert raw[12:16] == b"JUNK"
        assert struct.unpack("<I", raw[16:20])[0] == 28
        assert struct.unpack("<I", raw[4:8])[0] == len(raw) - 8
        assert raw[48:56] == b"fmt " + struct.pack("<I", 16)
        assert raw[72:80] == b"data" + struct.pack("<I", FRAME * len(x))
        with wave.open(str(path), "rb") as handle:  # (skips the JUNK)
            assert handle.getnchannels() == 2
            assert handle.getsampwidth() == 2
            assert handle.getframerate() == RATE
            assert handle.readframes(len(x)) == _cs16(x)


class TestPastTheLimitIsRf64:
    def test_a_known_size_past_the_limit(self, tmp_path, small_limit):
        x = _tone(1500)  # (6000 bytes: past the 4000 a classic file holds)
        path = tmp_path / "long.wav"
        save_iq_file(path, x, RATE, 0.0, SampleFormat.INT16, FileFormat.WAV)
        _check_rf64(path.read_bytes(), x)

    def test_a_recording_becomes_rf64_as_it_grows_past_the_limit(
        self, tmp_path, small_limit
    ):
        path = tmp_path / "grows.wav"
        recorder = IQRecorder(RATE, 0.0, SampleFormat.INT16, FileFormat.WAV)
        recorder.start(path)
        x = _tone(1500)
        recorder.write(x[:900])  # (3600 bytes: still classic)
        raw = path.read_bytes()
        assert raw[:4] == b"RIFF" and raw[12:16] == b"JUNK"
        recorder.write(x[900:])
        recorder.stop()
        _check_rf64(path.read_bytes(), x)

    def test_the_header_is_up_to_date_after_every_write(self, tmp_path, small_limit):
        """A recording cut short (a crash: no stop()) reads up to its last
        write, before and after it became RF64."""
        path = tmp_path / "cut.wav"
        recorder = IQRecorder(RATE, 0.0, SampleFormat.INT16, FileFormat.WAV)
        recorder.start(path)
        x = _tone(1500)
        recorder.write(x[:500])
        with IQPlayer(path) as player:
            assert player.total_samples == 500
        recorder.write(x[500:])
        with IQPlayer(path) as player:
            assert player.total_samples == 1500
            np.testing.assert_allclose(player.read(1500), x, atol=STEP)
        recorder.stop()

    def test_a_classic_file_never_grows_past_its_limit(self, tmp_path, small_limit):
        """Promised to fit (its size known), a file given more than fits is
        refused before it is written: its header couldn't say so."""
        path = tmp_path / "promised.wav"
        writer = wav_file.WavWriter(path, 2, 2, RATE, expected_bytes=2000)
        writer.writeframes(_cs16(_tone(500)))
        with pytest.raises(ValueError):
            writer.writeframes(_cs16(_tone(1000)))  # (6000 bytes in all)
        writer.close()
        with IQPlayer(path) as player:
            assert player.total_samples == 500

    def test_an_odd_size_is_padded_to_a_whole_word(self, tmp_path, small_limit):
        """RIFF chunks take whole 16-bit words: one 8-bit mono frame too
        many is followed by a pad byte, counted in RF64's RIFF size."""
        path = tmp_path / "odd.wav"
        writer = wav_file.WavWriter(path, 1, 1, 48000)
        writer.writeframes(bytes(range(256)) * 20 + b"\x80")  # (5121 bytes)
        writer.close()
        raw = path.read_bytes()
        assert raw[:4] == b"RF64" and len(raw) == 80 + 5121 + 1
        _size, riff, data, frames, _table = struct.unpack("<IQQQI", raw[16:48])
        assert (riff, data, frames) == (len(raw) - 8, 5121, 5121)
        header = wav_file.read_wav_header(path)
        assert (header.data_bytes, header.frames) == (5121, 5121)

    def test_any_buffer_is_written_even_a_strided_one(self, tmp_path):
        """writeframes takes any buffer: a NumPy slice with a step (not
        contiguous) is written as its values, not refused."""
        values = np.arange(40, dtype="<i2")
        path = tmp_path / "strided.wav"
        with wav_file.WavWriter(path, 2, 2, RATE, expected_bytes=40) as writer:
            writer.writeframes(values[::2])
        raw = path.read_bytes()
        assert raw[44:] == values[::2].tobytes()
        assert wav_file.read_wav_header(path).frames == 10  # (40 bytes)

    def test_a_riff_file_cut_short_says_so(self, tmp_path):
        path = tmp_path / "short.wav"
        path.write_bytes(b"RIFF\x10\x00")
        with pytest.raises(ValueError, match="cut short"):
            wav_file.read_wav_header(path)

    def test_a_sample_rate_a_header_cant_hold_is_refused(self, tmp_path):
        for rate in (0, 2**32):
            with pytest.raises(ValueError):
                wav_file.WavWriter(tmp_path / "rate.wav", 2, 2, rate)
        assert not (tmp_path / "rate.wav").exists()


class TestOver4GiB:
    """Real sizes past 4 GiB: 64-bit numbers in the header."""

    HEAD = _tone(4096)
    TAIL = _tone(4096, cycles=-0.02, amplitude=0.25)
    FRAMES = 5 * GiB // FRAME  # (a 5 GiB file: 9 minutes at 2.4 MS/s)

    def _write(self, target):
        """Write the 5 GiB file to ``target`` (a path or a file object):
        its first and last samples, the rest skipped over (a hole)."""
        writer = wav_file.WavWriter(target, 2, 2, RATE)
        writer.writeframes(_cs16(self.HEAD))
        gap = (self.FRAMES - len(self.HEAD) - len(self.TAIL)) * FRAME
        writer._file.seek(gap, os.SEEK_CUR)  # (the middle: a hole)
        writer.writeframes(_cs16(self.TAIL))
        assert writer.container == "RF64"
        writer.close()

    def _check_header(self, raw, size):
        assert raw[:4] == b"RF64" and raw[12:16] == b"ds64"
        _size, riff, data, frames, _table = struct.unpack("<IQQQI", raw[16:48])
        assert (riff, data, frames) == (size - 8, self.FRAMES * FRAME, self.FRAMES)
        assert struct.unpack("<I", raw[76:80])[0] == 0xFFFFFFFF

    def test_in_a_sparse_file_in_memory(self):
        handle = _SparseFile()
        self._write(handle)
        size = 80 + self.FRAMES * FRAME
        assert handle.size == size
        handle.seek(0)
        self._check_header(handle.read(80), size)
        header = wav_file.read_wav_header(handle)
        assert header.container == "RF64"
        assert (header.data_offset, header.frames) == (80, self.FRAMES)
        assert header.frame_rate == RATE
        handle.seek(80 + (self.FRAMES - len(self.TAIL)) * FRAME)
        assert handle.read(len(self.TAIL) * FRAME) == _cs16(self.TAIL)
        assert wav_file.data_offset(handle) == 80

    def test_on_the_disk(self, tmp_path):
        if not _sparse_files_here(tmp_path):
            pytest.skip(
                "this file system makes no sparse files: 5 GiB would be written"
            )
        path = tmp_path / "huge.wav"
        self._write(path)
        stat = os.stat(path)
        assert stat.st_size == 80 + self.FRAMES * FRAME
        assert stat.st_blocks * 512 < 16 << 20  # (sparse: a few blocks)
        with open(path, "rb") as handle:
            self._check_header(handle.read(80), stat.st_size)
        with IQPlayer(path) as player:
            assert (player.total_samples, player.sample_rate) == (self.FRAMES, RATE)
            np.testing.assert_allclose(player.read(4096), self.HEAD, atol=STEP)
            player.seek(self.FRAMES - len(self.TAIL))
            np.testing.assert_allclose(player.read(8192), self.TAIL, atol=STEP)
            assert len(player.read(16)) == 0 and player.eof
        info = detect_format(path)
        assert info.is_valid and info.file_format is FileFormat.WAV
        assert info.num_samples == self.FRAMES
        assert info.duration_seconds == pytest.approx(self.FRAMES / RATE)
        assert info.metadata["wav_container"] == "RF64"
        assert wav_data_offset(path) == 80


# ---------------------------------------------------------------------------
# Reading other programs' files
# ---------------------------------------------------------------------------


class TestReadingOtherProgramsFiles:
    def test_rf64_with_a_list_chunk_before_its_data(self, tmp_path):
        """As ffmpeg writes it (-rf64 always): a LIST chunk before data."""
        x = _tone(5000)
        info_chunk = _chunk(b"LIST", b"INFOISFT" + struct.pack("<I", 5) + b"test\0")
        path = tmp_path / "other.wav"
        offset = _rf64(path, len(x), head=_cs16(x), extra=info_chunk)
        header = wav_file.read_wav_header(path)
        assert header.container == "RF64"
        assert (header.channels, header.sample_width) == (2, 2)
        assert (header.data_offset, header.frames) == (offset, len(x))
        samples, meta = load_iq_file(path)
        np.testing.assert_allclose(samples, x, atol=STEP)
        assert meta.sample_rate == RATE
        assert wav_data_offset(path) == offset

    def test_bw64_is_rf64_by_another_name(self, tmp_path):
        x = _tone(2000)
        path = tmp_path / "broadcast.wav"
        _rf64(path, len(x), head=_cs16(x), magic=b"BW64")
        assert wav_file.read_wav_header(path).container == "BW64"
        np.testing.assert_allclose(load_iq_file(path)[0], x, atol=STEP)

    def test_a_sparse_rf64_file_over_4_gib(self, tmp_path):
        if not _sparse_files_here(tmp_path):
            pytest.skip(
                "this file system makes no sparse files: 5 GiB would be written"
            )
        frames = 5 * GiB // FRAME
        head, tail = _tone(1000), _tone(1000, cycles=0.2)
        path = tmp_path / "big.wav"
        _rf64(path, frames, head=_cs16(head), tail=_cs16(tail))
        with IQPlayer(path) as player:
            assert player.total_samples == frames
            player.seek(frames - 1000)
            np.testing.assert_allclose(player.read(1000), tail, atol=STEP)
        assert detect_format(path).num_samples == frames

    def test_wave64(self, tmp_path):
        """Sony Wave64 (.w64): GUIDs for chunk ids, 64-bit sizes that count
        their own 24-byte header, chunks on 8-byte boundaries."""
        tail = bytes.fromhex("f3acd3118cd100c04f8edb8a")
        x = _tone(3000)
        body = _fmt()
        fmt = b"fmt " + tail + struct.pack("<Q", 24 + len(body)) + body
        data = b"data" + tail + struct.pack("<Q", 24 + len(x) * FRAME) + _cs16(x)
        riff = b"riff" + bytes.fromhex("2e91cf11a5d628db04c10000")
        wave_id = b"wave" + tail
        size = 40 + len(fmt) + len(data)
        path = tmp_path / "take.w64"
        path.write_bytes(riff + struct.pack("<Q", size) + wave_id + fmt + data)
        header = wav_file.read_wav_header(path)
        assert header.container == "W64"
        assert (header.data_offset, header.frames) == (40 + len(fmt) + 24, len(x))
        samples, meta = load_iq_file(path)
        np.testing.assert_allclose(samples, x, atol=STEP)
        assert meta.sample_rate == RATE
        info = detect_format(path)
        assert info.file_format is FileFormat.WAV and info.num_samples == len(x)

    def test_an_extensible_pcm_file(self, tmp_path):
        """WAVE_FORMAT_EXTENSIBLE with the PCM subformat, as many
        programs write a 2-channel file."""
        x = _tone(1000)
        pcm = bytes.fromhex("0100000000001000800000aa00389b71")
        body = _fmt(tag=0xFFFE) + struct.pack("<HHI", 22, 16, 3) + pcm
        path = tmp_path / "extensible.wav"
        chunks = _chunk(b"fmt ", body) + _chunk(b"data", _cs16(x))
        path.write_bytes(
            b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks
        )
        np.testing.assert_allclose(load_iq_file(path)[0], x, atol=STEP)

    def test_floating_point_samples_are_refused_plainly(self, tmp_path):
        x = np.zeros(200, np.float32)
        chunks = _chunk(b"fmt ", _fmt(tag=3, width=4)) + _chunk(b"data", x.tobytes())
        path = tmp_path / "float.wav"
        path.write_bytes(
            b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks
        )
        info = detect_format(path)
        assert not info.is_valid
        assert "floating" in info.error_message

    def test_a_file_that_isnt_wav(self, tmp_path):
        path = tmp_path / "noise.wav"
        path.write_bytes(os.urandom(1000))
        with pytest.raises(ValueError):
            wav_file.read_wav_header(path)
        assert not detect_format(path).is_valid
        assert wav_data_offset(path) == 0


class TestDamagedAndUnfinishedFiles:
    def _classic(self, path, x, declared):
        body = _cs16(x)
        fmt = _chunk(b"fmt ", _fmt())
        size = 4 + len(fmt) + 8 + len(body)
        path.write_bytes(
            b"RIFF"
            + struct.pack("<I", size)
            + b"WAVE"
            + fmt
            + b"data"
            + struct.pack("<I", declared)
            + body
        )

    def test_a_recording_cut_short_counts_what_is_there(self, tmp_path):
        """Its header says more than the file holds (the writer stopped
        before the end): what is there is read, and counted right."""
        x = _tone(1000)
        path = tmp_path / "cut.wav"
        self._classic(path, x, declared=FRAME * 5000)
        with IQPlayer(path) as player:
            assert player.total_samples == 1000
            np.testing.assert_allclose(player.read(5000), x, atol=STEP)
        assert detect_format(path).num_samples == 1000

    def test_an_unknown_length_reads_to_the_end(self, tmp_path):
        """A data size of 0xFFFFFFFF: written by a program streaming it,
        whose length was unknown."""
        x = _tone(1000)
        path = tmp_path / "streamed.wav"
        self._classic(path, x, declared=0xFFFFFFFF)
        assert wav_file.read_wav_header(path).frames == 1000
        # (Open Recording, Recording Info and the playback strip counted
        # 1,073,741,823 samples.)
        with IQPlayer(path) as player:
            assert player.total_samples == 1000
            np.testing.assert_allclose(player.read(5000), x, atol=STEP)
        assert detect_format(path).num_samples == 1000

    def test_an_unfinished_rf64_file_reads_to_the_end(self, tmp_path):
        """RF64 written by a program that stopped before it filled in its
        sizes (a crash): ds64's data size still 0, the chunk's own -1."""
        x = _tone(1000)
        path = tmp_path / "unfinished.wav"
        offset = _rf64(path, len(x), head=_cs16(x))
        with open(path, "r+b") as handle:
            handle.seek(20)
            handle.write(struct.pack("<QQQ", 0, 0, 0))
        header = wav_file.read_wav_header(path)
        assert (header.data_offset, header.frames) == (offset, len(x))

    def test_an_unfinished_wave64_file_reads_to_the_end(self, tmp_path):
        """Wave64 whose data size was never written (0)."""
        tail = bytes.fromhex("f3acd3118cd100c04f8edb8a")
        x = _tone(500)
        body = _fmt()
        fmt = b"fmt " + tail + struct.pack("<Q", 24 + len(body)) + body
        data = b"data" + tail + struct.pack("<Q", 0) + _cs16(x)
        riff = b"riff" + bytes.fromhex("2e91cf11a5d628db04c10000")
        path = tmp_path / "unfinished.w64"
        path.write_bytes(riff + struct.pack("<Q", 0) + b"wave" + tail + fmt + data)
        assert wav_file.read_wav_header(path).frames == len(x)

    def test_a_classic_file_over_4_gib_whose_size_wrapped(self, tmp_path):
        """A writer that knows no RF64 wrote past 4 GiB: its 32-bit data
        size went round (it holds the size less 4 GiB)."""
        if not _sparse_files_here(tmp_path):
            pytest.skip(
                "this file system makes no sparse files: 5 GiB would be written"
            )
        frames = 5 * GiB // FRAME
        path = tmp_path / "wrapped.wav"
        tail = _cs16(_tone(100, cycles=0.1))
        fmt = _chunk(b"fmt ", _fmt())
        with open(path, "wb") as handle:
            data = frames * FRAME
            handle.write(b"RIFF" + struct.pack("<I", (36 + data) % (1 << 32)) + b"WAVE")
            handle.write(fmt + b"data" + struct.pack("<I", data % (1 << 32)))
            handle.seek(44 + data - len(tail))
            handle.write(tail)
        assert wav_file.read_wav_header(path).frames == frames

    @pytest.mark.parametrize(
        "kind, said",
        [
            ("w64 fmt", "no data chunk"),
            ("w64 other", r"no format \(fmt\) chunk"),
            ("rf64 table", "no data chunk"),
        ],
    )
    def test_a_chunk_size_past_the_end_says_so_plainly(self, tmp_path, kind, said):
        """A damaged 64-bit chunk size (RF64's ds64 table, Wave64) that
        runs past the end of the file, and past what a file position can
        hold: plain WavError, not "cannot fit 'int' into an offset-sized
        integer" (a path) or OverflowError (an open file)."""
        tail = bytes.fromhex("f3acd3118cd100c04f8edb8a")
        samples = bytes(400)
        if kind.startswith("w64"):
            fmt = b"fmt " + tail + struct.pack("<Q", 24 + 16) + _fmt()
            if kind == "w64 fmt":
                fmt = fmt[:16] + struct.pack("<Q", 2**64 - 8) + fmt[24:]
            else:  # (an unknown chunk before fmt)
                fmt = b"junk" + tail + struct.pack("<Q", 2**63 + 24) + fmt
            data = b"data" + tail + struct.pack("<Q", 24 + len(samples)) + samples
            riff = b"riff" + bytes.fromhex("2e91cf11a5d628db04c10000")
            raw = riff + struct.pack("<Q", 0) + b"wave" + tail + fmt + data
        else:
            table = struct.pack("<4sQ", b"LIST", 2**63)
            ds64 = b"ds64" + struct.pack("<IQQQI", 40, 0, 400, 100, 1) + table
            rest = _chunk(b"fmt ", _fmt()) + b"LIST" + struct.pack("<I", 0xFFFFFFFF)
            rest += _chunk(b"data", samples)
            raw = b"RF64" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE" + ds64 + rest
        path = tmp_path / "damaged.wav"
        path.write_bytes(raw)
        with pytest.raises(wav_file.WavError, match=said):
            wav_file.read_wav_header(path)
        with pytest.raises(wav_file.WavError, match=said):
            wav_file.read_wav_header(io.BytesIO(raw))
        assert wav_file.data_offset(io.BytesIO(raw)) == 0
        assert wav_data_offset(path) == 0
        with pytest.raises(ValueError, match=said):
            IQPlayer(path)
        info = detect_format(path)
        assert not info.is_valid and re.search(said, info.error_message)


# ---------------------------------------------------------------------------
# The rest of the recording code with RF64
# ---------------------------------------------------------------------------


class TestRf64Recordings:
    def test_saved_and_loaded_with_a_sidecar(self, tmp_path, small_limit):
        """A take saved past the limit keeps its center frequency in its
        sidecar, whose header_bytes lead a SigMF reader to the samples."""
        x = _tone(2000)
        path = tmp_path / "take.wav"
        save_iq_file(path, x, RATE, 145.8e6, SampleFormat.INT16, FileFormat.WAV)
        assert path.read_bytes()[:4] == b"RF64"
        write_sidecar(
            path,
            SampleFormat.INT16,
            RATE,
            145.8e6,
            header_bytes=wav_data_offset(path),
        )
        samples, meta = load_iq_file(path)
        np.testing.assert_allclose(samples, x, atol=STEP)
        assert (meta.sample_rate, meta.center_frequency) == (RATE, 145.8e6)
        # The metadata file itself reads the samples after the header.
        samples, meta = load_iq_file(sidecar_path(path))
        np.testing.assert_allclose(samples, x, atol=STEP)
        info = get_file_info(path)
        assert info["num_samples"] == len(x)
        assert info["center_frequency_hz"] == 145.8e6

    def test_the_sample_check_knows_the_file_says_its_format(self, tmp_path):
        """A WAV file states its format: clipped samples are the signal's,
        not a sign of reading it in the wrong format."""
        x = np.full(3000, 1.0 + 1.0j, np.complex64)  # (at full scale)
        path = tmp_path / "loud.wav"
        _rf64(path, len(x), head=_cs16(x))
        check = check_iq_samples(path)
        assert check.ok, check.problem
        assert "clipped when it was recorded" in check.note

    def test_a_file_named_otherwise_is_still_read_as_wav(self, tmp_path):
        """detect_format takes a file starting like WAV for WAV whatever
        its name; reading it agrees (it read the header as cf32 samples)."""
        x = _tone(1000)
        path = tmp_path / "capture.rf64"
        _rf64(path, len(x), head=_cs16(x))
        assert detect_format(path).file_format is FileFormat.WAV
        with IQPlayer(path) as player:
            assert player.sample_format is SampleFormat.INT16
            np.testing.assert_allclose(player.read(1000), x, atol=STEP)


# ---------------------------------------------------------------------------
# Record Audio: a WAV recording of unknown length, hours long
# ---------------------------------------------------------------------------


class TestRecordAudioPastTheLimit:
    """Record Audio (48 kHz 16-bit WAV of what is heard) reaches 4 GiB after
    about 12 hours of mono or 6 of stereo. It wrote through the wave module,
    which failed there ("'L' format requires 0 <= number <= 4294967295") with
    its file position left in the header, where the take's closing fade-out
    then landed: the hours recorded no longer opened as WAV. Now it becomes
    RF64 and goes on, as an I/Q recording does."""

    AUDIO_RATE = 48000

    def _blocks(self, n=6, size=1000):
        rng = np.random.default_rng(3)
        return [
            np.clip(0.4 * rng.standard_normal(size), -1, 1).astype(np.float32)
            for _ in range(n)
        ]

    @staticmethod
    def _int16(blocks):
        """The blocks as AudioRecorder stores them."""
        return b"".join(
            (np.clip(b, -1.0, 1.0) * 32767).astype("<i2").tobytes() for b in blocks
        )

    def test_a_file_of_known_length_is_the_wave_module_s(self, tmp_path):
        x = (0.5 * np.sin(0.05 * np.arange(4800))).astype(np.float32)
        ours = tmp_path / "ours.wav"
        save_audio_file(ours, x, self.AUDIO_RATE)
        reference = tmp_path / "reference.wav"
        with wave.open(str(reference), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(self.AUDIO_RATE)
            handle.writeframes(self._int16([x]))
        assert ours.read_bytes() == reference.read_bytes()

    def test_it_becomes_rf64_as_it_grows_past_the_limit(self, tmp_path, small_limit):
        path = tmp_path / "heard.wav"
        recorder = AudioRecorder(self.AUDIO_RATE, channels=1, normalize=False)
        recorder.start(path)
        blocks = self._blocks()  # (6 x 2000 bytes: past the 4000 a file holds)
        recorder.write(blocks[0])
        raw = path.read_bytes()
        assert raw[:4] == b"RIFF" and raw[12:16] == b"JUNK"
        for done, block in enumerate(blocks[1:], start=2):
            recorder.write(block)
            # (Up to date after every write: a crash keeps what was heard.)
            assert wav_file.read_wav_header(path).frames == 1000 * done
        recorder.stop()
        raw = path.read_bytes()
        assert raw[:4] == b"RF64" and raw[12:16] == b"ds64"
        header = wav_file.read_wav_header(path)
        assert header.container == "RF64"
        assert (header.channels, header.sample_width) == (1, 2)
        assert (header.frame_rate, header.frames) == (self.AUDIO_RATE, 6000)
        assert raw[header.data_offset :] == self._int16(blocks)

    def test_a_stereo_take_too(self, tmp_path, small_limit):
        """Two channels (Record Audio of broadcast FM in stereo)."""
        path = tmp_path / "stereo.wav"
        recorder = AudioRecorder(self.AUDIO_RATE, channels=2, normalize=False)
        recorder.start(path)
        blocks = [np.stack([b, -b], axis=1) for b in self._blocks(n=3)]
        for block in blocks:
            recorder.write(block)
        recorder.stop()
        header = wav_file.read_wav_header(path)
        assert header.container == "RF64"
        assert (header.channels, header.frames) == (2, 3000)
        raw = path.read_bytes()
        assert raw[header.data_offset :] == self._int16(blocks)

    def test_a_stereo_take_counts_frames_not_values(self, tmp_path):
        """samples_written, duration_seconds and the metadata count each
        moment once (samples of each channel, as start's expected_samples
        does), not both channels' values: a stereo take read twice its
        length."""
        path = tmp_path / "stereo_count.wav"
        recorder = AudioRecorder(self.AUDIO_RATE, channels=2, normalize=False)
        recorder.start(path)
        block = np.stack([np.zeros(1000), np.zeros(1000)], axis=1)
        assert recorder.write(block) == 1000
        recorder.write(block)
        assert recorder.samples_written == 2000
        assert recorder.duration_seconds == 2000 / self.AUDIO_RATE
        meta = recorder.stop()
        assert meta.num_samples == 2000
        assert wav_file.read_wav_header(path).frames == 2000


class TestPlayingAudioBack:
    """AudioPlayer and load_audio_file read what Record Audio writes, of any
    length. They read through the wave module, which refuses RF64 ("file
    does not start with RIFF id"), so a take past 4 GiB (about 12 hours of
    mono, 6 of stereo) couldn't be read back, and a take cut short (or
    streamed, its size unknown) counted what its header said and failed to
    read. Now they read classic WAV, RF64, BW64 and Wave64 files as the I/Q
    reader does, counting the samples a file holds."""

    AUDIO_RATE = 48000

    def _take(self, path, blocks, channels, sample_format=None):
        kwargs = {} if sample_format is None else {"sample_format": sample_format}
        recorder = AudioRecorder(
            self.AUDIO_RATE, channels=channels, normalize=False, **kwargs
        )
        recorder.start(path)
        for block in blocks:
            recorder.write(block)
        recorder.stop()

    @staticmethod
    def _stored(values):
        """``values`` as a 16-bit take stores them, read back as floats."""
        ints = (np.clip(values, -1.0, 1.0) * 32767).astype(np.int16)
        return ints.astype(np.float32) / 32767.0

    def test_a_stereo_take_past_the_limit(self, tmp_path, small_limit):
        rng = np.random.default_rng(5)
        blocks = [
            np.clip(0.4 * rng.standard_normal((700, 2)), -1.0, 1.0) for _ in range(6)
        ]
        path = tmp_path / "long.wav"
        self._take(path, blocks, channels=2)
        assert wav_file.read_wav_header(path).container == "RF64"
        expected = self._stored(np.concatenate(blocks))
        with AudioPlayer(path) as player:
            assert (player.channels, player.sample_rate) == (2, self.AUDIO_RATE)
            assert player.total_samples == 4200
            assert player.duration_seconds == 4200 / self.AUDIO_RATE
            first = player.read(1000)
            assert first.shape == (1000, 2) and first.dtype == np.float32
            np.testing.assert_array_equal(first, expected[:1000])
            assert player.position == 1000
            player.seek(4000)
            np.testing.assert_array_equal(player.read(500), expected[4000:])
            assert player.position == 4200 and not player.eof
            assert player.read(10).size == 0 and player.eof
            np.testing.assert_array_equal(player.read_all(), expected)
        samples, meta = load_audio_file(path)
        np.testing.assert_array_equal(samples, expected)
        assert (meta.channels, meta.num_samples) == (2, 4200)
        assert meta.sample_rate == self.AUDIO_RATE

    def test_a_take_cut_short_or_of_unknown_length(self, tmp_path):
        """A take whose writing stopped part way (a crash, a full disk) says
        more samples than the file holds, and a streamed one says none it
        can count (0xFFFFFFFF): what the file holds is read."""
        rng = np.random.default_rng(6)
        x = np.clip(0.3 * rng.standard_normal(3000), -1.0, 1.0)
        path = tmp_path / "cut.wav"
        self._take(path, [x], channels=1)
        raw = path.read_bytes()
        path.write_bytes(raw[: len(raw) - 2 * 1000 - 1])  # 1000.5 samples short
        with AudioPlayer(path) as player:
            assert player.total_samples == 1999
            samples = player.read_all()
        np.testing.assert_array_equal(samples, self._stored(x)[:1999])

        streamed = tmp_path / "streamed.wav"
        body = (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        fmt = _chunk(b"fmt ", _fmt(channels=1, rate=self.AUDIO_RATE))
        streamed.write_bytes(
            b"RIFF"
            + struct.pack("<I", 0xFFFFFFFF)
            + b"WAVE"
            + fmt
            + b"data"
            + struct.pack("<I", 0xFFFFFFFF)
            + body
        )
        samples, meta = load_audio_file(streamed)
        assert meta.num_samples == 3000
        np.testing.assert_array_equal(samples, self._stored(x))

    def test_rf64_and_wave64_from_other_programs(self, tmp_path):
        x = _tone(2000)
        info_chunk = _chunk(b"LIST", b"INFOISFT" + struct.pack("<I", 5) + b"test\0")
        rf64 = tmp_path / "other.wav"
        _rf64(rf64, len(x), head=_cs16(x), extra=info_chunk, rate=self.AUDIO_RATE)

        tail = bytes.fromhex("f3acd3118cd100c04f8edb8a")
        body = _fmt(rate=self.AUDIO_RATE)
        fmt = b"fmt " + tail + struct.pack("<Q", 24 + len(body)) + body
        data = b"data" + tail + struct.pack("<Q", 24 + len(x) * FRAME) + _cs16(x)
        riff = b"riff" + bytes.fromhex("2e91cf11a5d628db04c10000")
        size = 40 + len(fmt) + len(data)
        w64 = tmp_path / "take.w64"
        w64.write_bytes(riff + struct.pack("<Q", size) + b"wave" + tail + fmt + data)

        for path in (rf64, w64):
            samples, meta = load_audio_file(path)
            assert samples.shape == (2000, 2)
            np.testing.assert_allclose(samples[:, 0], x.real, atol=STEP)
            np.testing.assert_allclose(samples[:, 1], x.imag, atol=STEP)
            assert (meta.sample_rate, meta.channels) == (self.AUDIO_RATE, 2)

    @pytest.mark.parametrize("channels", [1, 2])
    def test_8_and_24_bit_samples_read_as_before(self, tmp_path, channels):
        rng = np.random.default_rng(7)
        shape = (500, 2) if channels == 2 else (500,)
        x = np.clip(0.5 * rng.standard_normal(shape), -1.0, 1.0)
        u8 = tmp_path / "u8.wav"
        self._take(u8, [x], channels, AudioSampleFormat.UINT8)
        stored = ((x + 1.0) * 127.5).astype(np.uint8)
        expected = (stored.astype(np.float32) - 128) / 128.0
        np.testing.assert_array_equal(load_audio_file(u8)[0], expected)

        s24 = tmp_path / "s24.wav"
        self._take(s24, [x], channels, AudioSampleFormat.INT24)
        stored = (x * 8388607).astype(np.int32)
        expected = (stored / 8388607.0).astype(np.float32)
        samples, meta = load_audio_file(s24)
        np.testing.assert_array_equal(samples, expected)
        assert meta.sample_format is AudioSampleFormat.INT24

    def test_32_bit_samples_are_refused_plainly(self, tmp_path):
        """They were read as 16-bit ones: noise, twice as long."""
        path = tmp_path / "s32.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(4)
            handle.setframerate(self.AUDIO_RATE)
            handle.writeframes(np.zeros(100, "<i4").tobytes())
        with pytest.raises(ValueError, match="32-bit"):
            AudioPlayer(path)

"""M-F1: 24- and 32-bit integer and floating-point WAV I/Q files are read.

Before, a WAV file whose samples weren't 8- or 16-bit integers was refused
in the window ("its samples are 24-bit, and this app reads 8- and 16-bit
I/Q WAV files"; floating point: "only integer (PCM) WAV files are read"),
though SDR Console, SDRuno and HDSDR write them, and the library read 24-
and 32-bit ones as 16-bit samples: noise, one and a half or two times as
many as the file holds, "valid" in detect_format. Now 24- and 32-bit
integer PCM and 32- and 64-bit IEEE floating point are read (format 3 and
the extensible format's float subformat), in classic WAV, RF64, BW64 and
Wave64 alike, scaled so full scale is 1.0 (floating point as stored), and
seeking and counting are right. What still can't be read is refused in
plain words. Writing is as it was.

The files are packed by hand (``tests/wav_builders.py``), not by the code
under test.
"""

import io
import struct

import numpy as np
import pytest

from sdr_module.dsp import wav_file
from sdr_module.dsp.recording import (
    AudioPlayer,
    FileFormat,
    IQPlayer,
    IQRecorder,
    SampleFormat,
    check_iq_samples,
    detect_format,
    get_file_info,
    load_audio_file,
    load_iq_file,
    save_iq_file,
    wav_frames,
    write_sidecar,
)
from tests.wav_builders import (
    encode,
    fmt_body,
    full_scale,
    integers,
    interleave,
    iq_wav,
    write_wav,
)

RATE = 48000
#: (kind, bytes per sample) read now.
WIDE = [("int", 3), ("int", 4), ("float", 4), ("float", 8)]


def _tone(n=4800, freq=3000.0, rate=RATE, amplitude=0.5):
    t = np.arange(n) / rate
    return (amplitude * np.exp(2j * np.pi * freq * t)).astype(np.complex64)


def _step(kind, width):
    """What storing a value in the format (and reading it back in float32)
    can change it by."""
    if kind == "float":
        return 1e-7
    return 1.0 / full_scale(width) + 1e-7


class TestWideAndFloatSamplesRead:
    @pytest.mark.parametrize("kind,width", WIDE)
    @pytest.mark.parametrize("container", ["RIFF", "RF64", "W64"])
    @pytest.mark.parametrize("extensible", [False, True])
    def test_every_sample_reads_back(
        self, tmp_path, kind, width, container, extensible
    ):
        x = _tone()
        ext = ".w64" if container == "W64" else ".wav"
        path = tmp_path / f"{kind}{8 * width}{ext}"
        offset = iq_wav(path, x, kind, width, RATE, container, extensible)
        header = wav_file.read_wav_header(path)
        assert header.container == container
        assert (header.channels, header.sample_width) == (2, width)
        assert header.sample_type == kind
        assert (header.data_offset, header.frames) == (offset, len(x))
        samples, meta = load_iq_file(path)
        assert samples.dtype == np.complex64
        assert len(samples) == len(x)
        np.testing.assert_allclose(samples, x, atol=_step(kind, width))
        assert meta.sample_rate == RATE

    @pytest.mark.parametrize("kind,width", WIDE)
    def test_seeking_and_counting(self, tmp_path, kind, width):
        x = _tone(10001, freq=1234.0)
        path = tmp_path / "take.wav"
        iq_wav(path, x, kind, width)
        with IQPlayer(path) as player:
            assert player.total_samples == len(x)
            assert player.duration_seconds == pytest.approx(len(x) / RATE)
            player.seek(7777)
            np.testing.assert_allclose(
                player.read(1000), x[7777:8777], atol=_step(kind, width)
            )
            assert player.position == 8777
            player.seek(len(x) - 5)
            assert len(player.read(100)) == 5
            assert len(player.read(100)) == 0 and player.eof
            player.seek(0)
            pieces = [player.read(n) for n in (1, 999, 3000, 7000)]
        np.testing.assert_allclose(np.concatenate(pieces), x, atol=_step(kind, width))

    def test_full_scale_is_one(self, tmp_path):
        """The largest integer reads as 1.0 (as 16-bit's 32767 does);
        floating point is as stored, even past 1.0."""
        for width in (3, 4):
            top = full_scale(width)
            data = integers([top, -top - 1, 0, top // 2], width)
            path = tmp_path / f"s{8 * width}.wav"
            write_wav(path, fmt_body(2, RATE, width), data)
            samples = load_iq_file(path)[0]
            np.testing.assert_allclose(
                samples, [1 - (1 + 1 / top) * 1j, 0.5j], atol=1e-7
            )
        path = tmp_path / "f32.wav"
        write_wav(path, fmt_body(2, RATE, 4, "float"), encode([1.5, -2.25], "float", 4))
        np.testing.assert_array_equal(load_iq_file(path)[0], [1.5 - 2.25j])

    def test_24_valid_bits_in_32_bit_containers(self, tmp_path):
        """WAVE_FORMAT_EXTENSIBLE: 32-bit containers, 24 valid bits at the
        top (left-justified, as the format says): full scale is still 1.0."""
        x = _tone(1000)
        ints = np.rint(interleave(x) * full_scale(3)).astype(np.int64) << 8
        fmt = fmt_body(2, RATE, 4, "int", extensible=True, valid=24)
        path = tmp_path / "s24in32.wav"
        write_wav(path, fmt, integers(ints, 4))
        np.testing.assert_allclose(load_iq_file(path)[0], x, atol=1.2e-7)

    @pytest.mark.parametrize("kind,width", WIDE)
    def test_detect_format_and_file_info_say_what_they_are(self, tmp_path, kind, width):
        x = _tone()
        path = tmp_path / "take.wav"
        iq_wav(path, x, kind, width, container="RF64")
        info = detect_format(path)
        assert info.is_valid, info.error_message
        assert info.file_format is FileFormat.WAV
        assert (info.num_samples, info.num_channels) == (len(x), 2)
        assert info.bits_per_sample == 8 * width
        assert info.sample_rate == RATE
        assert info.metadata["wav_sample_type"] == kind
        assert info.metadata["wav_container"] == "RF64"
        # (The I/Q format nearest it: exactly cf32 and cf64 for floats.)
        if kind == "float":
            assert info.sample_format is (
                SampleFormat.FLOAT32 if width == 4 else SampleFormat.FLOAT64
            )
        details = get_file_info(path)
        assert details["is_valid"] and details["error"] is None
        assert details["num_samples"] == len(x)
        assert details["bits_per_sample"] == 8 * width
        assert details["metadata"]["wav_sample_type"] == kind

    def test_a_sidecar_gives_the_center_as_for_16_bit_files(self, tmp_path):
        x = _tone()
        path = tmp_path / "take.wav"
        offset = iq_wav(path, x, "int", 3)
        write_sidecar(path, SampleFormat.INT16, RATE, 145.5e6, header_bytes=offset)
        with IQPlayer(path) as player:
            assert player.center_frequency == 145.5e6
            assert player.sample_rate == RATE
            np.testing.assert_allclose(player.read(len(x)), x, atol=_step("int", 3))
        assert detect_format(path).center_frequency == 145.5e6

    def test_8_and_16_bit_files_read_as_before(self, tmp_path):
        rng = np.random.default_rng(3)
        x = (0.4 * (rng.standard_normal(500) + 1j * rng.standard_normal(500))).astype(
            np.complex64
        )
        for width, scale, zero, fmt in (
            (1, 128.0, 128.0, SampleFormat.UINT8),
            (2, 32767.0, 0.0, SampleFormat.INT16),
        ):
            path = tmp_path / f"w{width}.wav"
            iq_wav(path, x, "int", width)
            raw = np.frombuffer(
                path.read_bytes()[44:], np.uint8 if width == 1 else "<i2"
            ).astype(np.float32)
            expected = ((raw - np.float32(zero)) / np.float32(scale)).view(np.complex64)
            samples = load_iq_file(path)[0]
            np.testing.assert_array_equal(samples, expected)
            assert detect_format(path).sample_format is fmt
            with IQPlayer(path) as player:
                assert player.sample_format is fmt


class TestStillRefusedInPlainWords:
    def _write(self, path, tag, bits, width, channels=2, extensible_sub=None):
        body = struct.pack(
            "<HHIIHH",
            tag,
            channels,
            RATE,
            RATE * channels * width,
            channels * width,
            bits,
        )
        if extensible_sub is not None:
            body += struct.pack("<HHI", 22, bits, 3) + extensible_sub
        data = bytes(channels * width * 10)
        write_wav(path, body, data)

    @pytest.mark.parametrize(
        "tag,bits,width,said",
        [
            (1, 64, 8, "64-bit integers"),
            (3, 16, 2, "16-bit floating point"),
            (7, 8, 1, "WAV format 7"),  # mu-law
            (2, 4, 1, "WAV format 2"),  # ADPCM
        ],
    )
    def test_formats_that_cant_be_read(self, tmp_path, tag, bits, width, said):
        path = tmp_path / "other.wav"
        self._write(path, tag, bits, width)
        with pytest.raises(wav_file.WavError, match=said):
            wav_file.read_wav_header(path)
        info = detect_format(path)
        assert not info.is_valid and said in info.error_message
        with pytest.raises(ValueError, match=said):
            IQPlayer(path)

    def test_an_unknown_extensible_subformat(self, tmp_path):
        path = tmp_path / "other.wav"
        self._write(path, 0xFFFE, 16, 2, extensible_sub=bytes(16))
        with pytest.raises(wav_file.WavError, match="extensible"):
            wav_file.read_wav_header(path)

    @pytest.mark.parametrize("channels", [2, 1])
    @pytest.mark.parametrize("data", ["packed", "floats"])
    def test_samples_in_wider_words_than_their_bits(self, tmp_path, channels, data):
        """files-3: a plain PCM fmt chunk saying 24 bits a sample with a
        block align of 4 bytes a channel (24-bit samples in 32-bit words, as
        ALSA's arecord wrote S24_LE) was read as packed 3-byte samples: a
        third more of them than the file holds, every one wrong (noise),
        with no word of it (the sample check trusts a WAV header). Where in
        its word each sample lies the header doesn't say: R8-11 reads it
        from the data (TestSamplesInWiderWords). Words that fit neither
        layout (packed samples under a wrong block align, or Cool Edit's
        floats labelled PCM: libsndfile's other readings of such a header)
        are still refused in plain words."""
        x = _tone() if channels == 2 else _tone().real
        if data == "packed":
            raw = encode(interleave(x), "int", 3)
        else:
            raw = encode(interleave(x), "float", 4)
        path = tmp_path / "s24_le.wav"
        write_wav(path, fmt_body(channels, RATE, 4, bits=24), raw)
        said = "24-bit samples are stored in 32-bit words"
        with pytest.raises(wav_file.WavError, match=said):
            wav_file.read_wav_header(path)
        info = detect_format(path)
        assert not info.is_valid and said in info.error_message
        with pytest.raises(ValueError, match=said):
            IQPlayer(path)
        check = check_iq_samples(path)
        assert not check.ok and said in check.problem
        with pytest.raises(ValueError, match=said):
            AudioPlayer(path)

    def test_samples_in_words_they_fill_still_read(self, tmp_path):
        """(Unchanged: packed 24-bit samples, the extensible format's 32-bit
        words with 24 valid bits, whose wBitsPerSample is the word's, and a
        block align of 0, a slip whose samples are packed.)"""
        x = _tone()
        ints = np.rint(interleave(x) * full_scale(3)).astype(np.int64)
        packed = tmp_path / "packed.wav"
        write_wav(packed, fmt_body(2, RATE, 3), integers(ints, 3))
        ext = tmp_path / "ext.wav"
        write_wav(
            ext, fmt_body(2, RATE, 4, extensible=True, valid=24), integers(ints << 8, 4)
        )
        no_align = tmp_path / "align0.wav"
        body = bytearray(fmt_body(2, RATE, 2))
        body[12:14] = b"\0\0"  # nBlockAlign 0
        write_wav(no_align, bytes(body), encode(interleave(x), "int", 2))
        for path, width, step in (
            (packed, 3, 2e-7),
            (ext, 4, 2e-7),
            (no_align, 2, 4e-5),
        ):
            assert wav_file.read_wav_header(path).sample_width == width
            with IQPlayer(path) as player:
                assert player.total_samples == len(x)
                np.testing.assert_allclose(player.read(len(x) + 10), x, atol=step)

    def test_more_than_two_channels(self, tmp_path):
        """Four channels aren't I and Q: refused, not read as garbage."""
        path = tmp_path / "quad.wav"
        write_wav(path, fmt_body(4, RATE, 2), bytes(4 * 2 * 100))
        assert wav_file.read_wav_header(path).channels == 4  # (it is WAV)
        info = detect_format(path)
        assert not info.is_valid
        assert "4 channels" in info.error_message
        with pytest.raises(ValueError, match="4 channels"):
            IQPlayer(path)


def _s24_words(x, justified, top="sign"):
    """``x`` (full scale 1.0) as 24-bit samples in 32-bit words: in each
    word's top three bytes ("left", its low byte zero) or its low three
    ("right", as ALSA's arecord writes S24_LE), its top byte then a copy
    of the sign ("sign") or zero ("zero")."""
    ints = np.rint(interleave(x) * full_scale(3)).astype(np.int64)
    if justified == "left":
        return integers(ints << 8, 4)
    words = ints if top == "sign" else ints & 0xFFFFFF
    return words.astype("<u4" if top == "zero" else "<i4").tobytes()


class TestSamplesInWiderWords:
    """R8-11: 24-bit samples in 32-bit words (a plain PCM header saying 24
    bits a sample and 4 bytes a channel in its block align: ALSA's arecord
    wrote S24_LE so) were refused (R7 files-3: the header doesn't say where
    in each word a sample lies). They are read now, where the data shows
    them to be, as libsndfile looks: each word's low byte zero, the sample
    is in its top three bytes (read as a 32-bit sample); its top byte a copy
    of the sign of the three below it, or zero, it is in its low three
    (sign-extended from bit 23). The same in every reader."""

    LAYOUTS = [("left", "sign"), ("right", "sign"), ("right", "zero")]

    @pytest.mark.parametrize("channels", [2, 1])
    @pytest.mark.parametrize("justified,top", LAYOUTS)
    def test_every_reader_reads_them(self, tmp_path, channels, justified, top):
        x = _tone(10001, freq=1234.0)
        if channels == 1:
            x = x.real
        path = tmp_path / "s24_le.wav"
        offset = write_wav(
            path, fmt_body(channels, RATE, 4, bits=24), _s24_words(x, justified, top)
        )
        step = _step("int", 3)
        header = wav_file.read_wav_header(path)
        assert (header.sample_width, header.bits) == (4, 24)
        assert header.justified == justified
        assert header.sample_name == "24-bit integer"
        assert (header.data_offset, header.frames) == (offset, len(x))
        # Every reader: the header's, the frames', I/Q, audio, the check.
        values = np.concatenate(list(wav_frames(path, 777)))
        np.testing.assert_allclose(values, interleave(x), atol=step)
        with AudioPlayer(path) as audio:
            assert audio.total_samples == len(x)
            assert audio.metadata.sample_format.name == "INT24"
            audio.seek(5000)
            got = audio.read(1000)
            expected = interleave(x).reshape(len(x), channels)[5000:6000]
            np.testing.assert_allclose(got.reshape(expected.shape), expected, atol=step)
        info = detect_format(path)
        assert info.is_valid, info.error_message
        assert info.bits_per_sample == 24
        assert info.metadata["wav_sample_bits"] == 24
        assert info.metadata["wav_sample_justified"] == justified
        check = check_iq_samples(path)
        assert check.ok and not check.note, check
        if channels == 2:
            with IQPlayer(path) as player:
                assert player.total_samples == len(x)
                player.seek(7777)
                np.testing.assert_allclose(player.read(1000), x[7777:8777], atol=step)
            np.testing.assert_allclose(load_iq_file(path)[0], x, atol=step)
        else:
            # (A real signal, read as I/Q as from packed samples.)
            packed = tmp_path / "packed.wav"
            iq_wav(packed, x, "int", 3, RATE)
            with IQPlayer(path) as player, IQPlayer(packed) as reference:
                assert player.total_samples == reference.total_samples
                player.seek(1000)
                reference.seek(1000)
                np.testing.assert_allclose(
                    player.read(2000), reference.read(2000), atol=1e-6
                )

    def test_full_scale_is_one_as_for_packed_samples(self, tmp_path):
        top = full_scale(3)
        ints = np.array([top, -top - 1, 0, top // 2], np.int64)
        expected = load_iq_file(_packed(tmp_path, ints))[0]
        np.testing.assert_allclose(expected, [1 - (1 + 1 / top) * 1j, 0.5j], atol=1e-7)
        for name, words in (
            ("left", (ints << 8).astype("<i4")),
            ("right", ints.astype("<i4")),
            ("zero", (ints & 0xFFFFFF).astype("<u4")),
        ):
            path = tmp_path / f"{name}.wav"
            write_wav(path, fmt_body(2, RATE, 4, bits=24), words.tobytes())
            np.testing.assert_array_equal(load_iq_file(path)[0], expected)

    def test_silence_reads_as_silence(self, tmp_path):
        """(Every word zero: either layout gives zeros.)"""
        path = tmp_path / "silence.wav"
        write_wav(path, fmt_body(2, RATE, 4, bits=24), bytes(8 * 1000))
        with IQPlayer(path) as player:
            samples = player.read(2000)
        assert len(samples) == 1000 and not np.any(samples)

    def test_the_data_is_looked_at_throughout(self, tmp_path):
        """(A take silent at first, its sound later: the words looked at
        are spread through the file, not only its start.)"""
        x = _tone(200000, freq=1234.0)
        x[:150000] = 0
        path = tmp_path / "late.wav"
        write_wav(path, fmt_body(2, RATE, 4, bits=24), _s24_words(x, "right", "zero"))
        assert wav_file.read_wav_header(path).justified == "right"
        with IQPlayer(path) as player:
            player.seek(160000)
            np.testing.assert_allclose(
                player.read(1000), x[160000:161000], atol=_step("int", 3)
            )

    def test_16_bit_sound_in_the_low_bytes(self, tmp_path):
        """(16-bit sound arecord wrote as S24_LE, each word's low byte
        zero: both layouts fit when its top byte copies the sign, and the
        two readings differ only in level, 48 dB: read as 32-bit samples,
        as libsndfile reads them. Zero there, read as 32-bit samples its
        negative half would turn positive: it is read from the low bytes.)"""
        rng = np.random.default_rng(7)
        s16 = rng.integers(-20000, 20000, 4000).astype(np.int64)
        x = s16 / 32767.0
        for top, justified, scale in (("sign", "left", 2**-8), ("zero", "right", 1)):
            words = s16 << 8
            if top == "zero":
                words = (words & 0xFFFFFF).astype("<u4")
            path = tmp_path / f"s16_{top}.wav"
            write_wav(
                path, fmt_body(1, RATE, 4, bits=24), words.astype("<i4").tobytes()
            )
            assert wav_file.read_wav_header(path).justified == justified
            values = np.concatenate(list(wav_frames(path)))
            np.testing.assert_allclose(values, scale * x, atol=2e-4 * scale)

    @pytest.mark.parametrize("container", ["RIFF", "RF64", "W64"])
    def test_a_file_open_on_it_is_left_where_its_samples_start(
        self, tmp_path, container
    ):
        """(read_wav_header of a binary file open on a WAV file leaves it
        where the samples start, as it does for every other WAV file: the
        words looked at to see where the samples lie left it after the
        last of them.)"""
        x = _tone(20000, freq=1234.0)
        data = _s24_words(x, "right")
        path = tmp_path / "s24_le.wav"
        offset = write_wav(path, fmt_body(2, RATE, 4, bits=24), data, container)
        with open(path, "rb") as handle:
            assert wav_file.read_wav_header(handle).justified == "right"
            assert handle.tell() == offset
            assert handle.read(8) == data[:8]
        handle = io.BytesIO(path.read_bytes())
        assert wav_file.read_wav_header(handle).justified == "right"
        assert handle.tell() == offset


def _packed(tmp_path, ints):
    path = tmp_path / "packed.wav"
    write_wav(path, fmt_body(2, RATE, 3), integers(ints, 3))
    return path


class TestTheSampleCheck:
    def test_clipped_24_bit_samples_are_noted(self, tmp_path):
        x = np.clip(interleave(_tone(4800, amplitude=1.4)), -1, 1)
        path = tmp_path / "loud.wav"
        write_wav(path, fmt_body(2, RATE, 3), encode(x, "int", 3))
        check = check_iq_samples(path)
        assert check.ok
        # (Every I or Q value clipped at full scale is counted, no other.)
        share = check.full_scale / (2 * check.samples)
        assert share == pytest.approx(np.mean(np.abs(x) >= 0.99))
        assert 0.3 < share < 0.6
        assert check.peak_dbfs == pytest.approx(3.0, abs=0.15)  # (|I + jQ| to 1.41)
        assert "clipped" in check.note

    def test_floats_that_arent_numbers(self, tmp_path):
        x = interleave(_tone(1000))
        x[100] = np.nan
        path = tmp_path / "nan.wav"
        write_wav(path, fmt_body(2, RATE, 4, "float"), encode(x, "float", 4))
        check = check_iq_samples(path)
        assert not check.ok and check.nonfinite >= 1
        assert "aren't numbers" in check.problem
        assert "32-bit float" in check.problem

    def test_floats_saved_with_another_scale(self, tmp_path):
        x = interleave(_tone(1000)) * 1000.0  # (e.g. raw ADC counts)
        path = tmp_path / "counts.wav"
        write_wav(path, fmt_body(2, RATE, 8, "float"), encode(x, "float", 8))
        check = check_iq_samples(path)
        assert not check.ok
        assert "far beyond full scale" in check.problem

    def test_samples_that_look_right(self, tmp_path):
        for kind, width in WIDE:
            path = tmp_path / f"{kind}{width}.wav"
            iq_wav(path, _tone(), kind, width)
            check = check_iq_samples(path)
            assert check.ok and not check.note, (kind, width, check)
            assert check.peak_dbfs == pytest.approx(-6.02, abs=0.05)


class TestAudioPlayer:
    """The library's AudioPlayer reads 32-bit integer and floating-point
    WAV audio too (it read 8, 16 and 24-bit)."""

    @pytest.mark.parametrize("kind,width", [("int", 4), ("float", 4), ("float", 8)])
    @pytest.mark.parametrize("channels", [1, 2])
    def test_read(self, tmp_path, kind, width, channels):
        rng = np.random.default_rng(5)
        shape = (400, 2) if channels == 2 else (400,)
        x = np.clip(0.5 * rng.standard_normal(shape), -1.0, 1.0)
        path = tmp_path / "audio.wav"
        write_wav(
            path, fmt_body(channels, RATE, width, kind), encode(x.ravel(), kind, width)
        )
        samples, meta = load_audio_file(path)
        assert samples.shape == shape
        np.testing.assert_allclose(samples, x, atol=_step(kind, width))
        assert (meta.sample_rate, meta.channels) == (RATE, channels)
        with AudioPlayer(path) as player:
            assert player.total_samples == 400
            player.seek(300)
            np.testing.assert_allclose(
                player.read(50), x[300:350], atol=_step(kind, width)
            )

    def test_what_it_cant_read_is_refused_plainly(self, tmp_path):
        path = tmp_path / "s64.wav"
        write_wav(path, fmt_body(1, RATE, 8, "int"), bytes(80))
        with pytest.raises(ValueError, match="64-bit"):
            AudioPlayer(path)


class TestCompatibleApi:
    def test_wav_header_is_built_as_before(self):
        header = wav_file.WavHeader("RIFF", 2, 2, RATE, 44, 400)
        assert header.sample_type == "int"
        assert (header.frame_bytes, header.frames) == (4, 100)

    def test_writing_is_as_it_was(self, tmp_path):
        """A WAV recording is still 16-bit integer PCM (or 8-bit)."""
        path = tmp_path / "out.wav"
        save_iq_file(path, _tone(), RATE, 0.0, SampleFormat.FLOAT32, FileFormat.WAV)
        tag, channels, rate, _br, align, bits = struct.unpack(
            "<HHIIHH", path.read_bytes()[20:36]
        )
        assert (tag, channels, rate, align, bits) == (1, 2, RATE, 4, 16)
        recorder = IQRecorder(
            RATE, sample_format=SampleFormat.UINT8, file_format=FileFormat.WAV
        )
        recorder.start(tmp_path / "u8.wav", expected_samples=10)
        recorder.write(_tone(10))
        recorder.stop()
        assert struct.unpack("<H", (tmp_path / "u8.wav").read_bytes()[34:36])[0] == 8

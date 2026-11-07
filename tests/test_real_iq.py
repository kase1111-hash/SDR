"""M-F2: one-channel (real) WAV files are read as I/Q, not mirrored.

Before, a one-channel WAV file (audio from a scanner or another program, or
a receiver's IF or RF sampled directly) was refused in the window ("it has
one channel: it is audio, not I/Q"), and the library read it as I = the
samples, Q = 0: every signal twice, at its frequency and mirrored below 0.
Now it is read the way a receiver turns a real IF into I/Q: the analytic
signal at half the file's rate, centred on a quarter of it, so the I/Q
holds the file's own frequencies (0 Hz to half its rate), each once.

These tests check the conversion against the signal it should give (a
tone at its own frequency, its mirror image at least 100 dB down over the
inner 80% of the span the receiver tunes), and that reading in pieces,
seeking and counting agree with reading the file at once.
"""

import numpy as np
import pytest

from sdr_module.dsp import real_iq
from sdr_module.dsp.recording import (
    FileFormat,
    FormatConverter,
    IQPlayer,
    SampleFormat,
    check_iq_samples,
    detect_format,
    load_iq_file,
    write_sidecar,
)
from tests.wav_builders import encode, fmt_body, iq_wav, write_wav

RATE = 48000


def _cos(n, freq, rate=RATE, amplitude=0.5, phase=0.3):
    return amplitude * np.cos(2 * np.pi * freq * np.arange(n) / rate + phase)


def _components(y, offset_hz, rate, first=100, count=4096):
    """The complex amplitude of ``y`` (I/Q at ``rate``) at ``+offset_hz``
    and at its mirror ``-offset_hz``, over whole cycles of both."""
    m = np.arange(first, first + count)
    part = y[first : first + count].astype(np.complex128)
    turn = np.exp(-2j * np.pi * offset_hz * m / rate)
    return np.mean(part * turn), np.mean(part * np.conj(turn))


class TestTheConversion:
    @pytest.mark.parametrize("fraction", [0.06, 0.1, 0.2, 0.3, 0.4, 0.44])
    def test_a_tone_is_one_tone_at_its_own_frequency(self, fraction):
        # A whole number of cycles of the tone and its mirror in 4096 I/Q
        # samples (the I/Q rate is half the file's).
        count, iq_rate = 4096, RATE / 2
        offset = round((fraction - 0.25) * RATE / iq_rate * count) * iq_rate / count
        freq = RATE / 4 + offset  # (the tone's own frequency in the file)
        frames = 2 * (count + 200) + 1
        x = _cos(frames, freq, amplitude=0.5)
        # (The file's frames with the zeros before and after it that the
        # filter reaches.)
        start, stop = real_iq.frame_span(0, real_iq.iq_count(frames))
        padded = np.zeros(stop - start)
        padded[-start : -start + frames] = x
        y = real_iq.real_to_iq(padded, 0)
        assert y.dtype == np.complex64 and len(y) == real_iq.iq_count(frames)
        wanted, mirror = _components(y, offset, iq_rate)
        # The real tone's amplitude, once, at its own frequency...
        assert abs(wanted) == pytest.approx(0.5, rel=2e-4)
        # ... and nothing at the mirror image.
        assert 20 * np.log10(abs(mirror) / abs(wanted)) < -100.0

    def test_i_is_the_samples_themselves(self):
        """I is every other real sample (signs turning with the quarter
        rate mix); only Q is filtered."""
        rng = np.random.default_rng(1)
        x = rng.uniform(-1, 1, 2 * 500 + 2 * real_iq.CONTEXT - 1)
        y = real_iq.real_to_iq(x, 7)  # (from I/Q sample 7: odd, so -1 first)
        even = x[real_iq.CONTEXT : real_iq.CONTEXT + 1000 : 2]
        signs = np.where(np.arange(7, 507) % 2, -1.0, 1.0)
        np.testing.assert_allclose(y.real, (signs * even).astype(np.float32))

    def test_the_spans_the_reader_needs(self):
        assert real_iq.frame_span(0, 10) == (-real_iq.CONTEXT, 20 + real_iq.CONTEXT - 1)
        assert real_iq.frame_span(5, 1) == (10 - real_iq.CONTEXT, 11 + real_iq.CONTEXT)
        assert [real_iq.iq_count(n) for n in (0, 1, 2, 3, 10)] == [0, 0, 1, 1, 5]
        assert real_iq.iq_rate(48000) == 24000
        assert real_iq.iq_center(48000) == 12000


class TestReadingAOneChannelFile:
    def _file(self, tmp_path, frames=24001, kind="int", width=2, rate=RATE):
        rng = np.random.default_rng(2)
        x = 0.3 * rng.standard_normal(frames)
        x += _cos(frames, 5000.0, rate, 0.4)
        x = np.clip(x, -1, 1)
        path = tmp_path / "audio.wav"
        iq_wav(path, x, kind, width, rate)
        return path, x

    def test_rate_center_and_count(self, tmp_path):
        path, x = self._file(tmp_path, frames=9601)
        with IQPlayer(path) as player:
            assert player.sample_rate == RATE / 2
            assert player.center_frequency == RATE / 4
            assert player.total_samples == 4800
            assert player.duration_seconds == pytest.approx(9600 / RATE)
            assert player.wav_header.channels == 1
            meta = player.metadata
            assert (meta.sample_rate, meta.center_frequency) == (RATE / 2, RATE / 4)
        info = detect_format(path)
        assert info.is_valid, info.error_message
        assert info.num_channels == 1
        assert (info.sample_rate, info.center_frequency) == (RATE / 2, RATE / 4)
        assert info.num_samples == 4800
        assert info.metadata["wav_frame_rate"] == RATE
        samples, meta = load_iq_file(path)
        assert len(samples) == 4800 and meta.sample_rate == RATE / 2

    def test_pieces_and_seeks_read_the_same_samples(self, tmp_path):
        path, _x = self._file(tmp_path)
        with IQPlayer(path) as player:
            whole = player.read(player.total_samples)
            assert player.eof is False
            assert len(player.read(10)) == 0 and player.eof
            rng = np.random.default_rng(4)
            for _ in range(20):
                start = int(rng.integers(0, len(whole)))
                count = int(rng.integers(1, 3000))
                player.seek(start)
                np.testing.assert_array_equal(
                    player.read(count), whole[start : start + count]
                )
                assert player.position == min(len(whole), start + count)
            player.seek(0)
            pieces = []
            while not player.eof:
                pieces.append(player.read(int(rng.integers(1, 700))))
        np.testing.assert_array_equal(np.concatenate(pieces), whole)

    def test_the_tone_shows_once_in_the_file(self, tmp_path):
        """Read through the file reader too (16-bit samples): the 5 kHz
        tone at 5 kHz (center 12 kHz, -7 kHz), its mirror (19 kHz) deep
        down."""
        path, _x = self._file(tmp_path)
        y = load_iq_file(path)[0]
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(y[:8192] * np.hanning(8192))))
        freqs = np.fft.fftshift(np.fft.fftfreq(8192, 1 / (RATE / 2))) + RATE / 4
        tone = spectrum[np.argmin(np.abs(freqs - 5000.0))]
        mirror = spectrum[np.argmin(np.abs(freqs - 19000.0))]
        floor = np.median(spectrum)
        assert 20 * np.log10(tone / floor) > 30
        assert 20 * np.log10(mirror / floor) < 6  # (the noise's own level)

    @pytest.mark.parametrize(
        "kind,width,step",
        [("int", 1, 1 / 128), ("int", 3, 2e-7), ("float", 4, 1e-6), ("float", 8, 1e-6)],
    )
    def test_every_format_reads_alike(self, tmp_path, kind, width, step):
        frames = 4001
        x = _cos(frames, 3000.0, amplitude=0.6)
        exact = tmp_path / "f64.wav"
        iq_wav(exact, x, "float", 8)
        other = tmp_path / f"{kind}{width}.wav"
        iq_wav(other, x, kind, width)
        a, b = load_iq_file(exact)[0], load_iq_file(other)[0]
        # (A step of the format, through the Hilbert filter's gain.)
        np.testing.assert_allclose(b, a, atol=3 * step)

    @pytest.mark.parametrize("frames,count", [(0, 0), (1, 0), (2, 1), (3, 1), (40, 20)])
    def test_tiny_files(self, tmp_path, frames, count):
        path = tmp_path / "tiny.wav"
        write_wav(path, fmt_body(1, RATE, 2), encode(np.full(frames, 0.25), "int", 2))
        with IQPlayer(path) as player:
            assert player.total_samples == count
            got = player.read(100)
            assert len(got) == count
            if count:
                assert abs(got[0].real - 0.25) < 1e-4  # (I is the samples)

    def test_a_full_scale_sine_is_full_scale_iq(self, tmp_path):
        path = tmp_path / "loud.wav"
        iq_wav(path, _cos(9600, 3000.0, amplitude=1.0), "float", 4)
        y = load_iq_file(path)[0][100:-100]
        assert np.abs(y).max() == pytest.approx(1.0, abs=1e-3)
        assert np.abs(y).min() == pytest.approx(1.0, abs=1e-3)

    def test_the_sample_check_counts_clipping_in_the_samples(self, tmp_path):
        clipped = tmp_path / "clipped.wav"
        iq_wav(clipped, np.clip(_cos(9600, 3000.0, amplitude=1.5), -1, 1), "int", 2)
        check = check_iq_samples(clipped)
        assert check.ok and "clipped" in check.note
        # A square wave at 0.9: no sample near full scale, though its
        # Hilbert part (Q) overshoots past 1.0.
        square = tmp_path / "square.wav"
        wave = 0.9 * np.sign(np.sin(2 * np.pi * 1000.0 * np.arange(9600) / RATE + 0.1))
        iq_wav(square, wave, "int", 2)
        check = check_iq_samples(square)
        assert np.abs(load_iq_file(square)[0].imag).max() > 1.0
        assert check.ok and check.full_scale == 0 and not check.note

    def test_the_file_says_its_format(self, tmp_path):
        """Like any WAV file, its format is its own (the check says so)."""
        path, _x = self._file(tmp_path)
        with IQPlayer(path, SampleFormat.UINT8) as player:
            assert player.sample_format is SampleFormat.INT16  # (not cu8)
            assert player.total_samples == 12000

    def test_a_sidecar_gives_the_center_and_none_leaves_its_own(self, tmp_path):
        path, _x = self._file(tmp_path)
        write_sidecar(path, SampleFormat.INT16, RATE, 7.1e6, header_bytes=44)
        with IQPlayer(path) as player:
            assert player.center_frequency == 7.1e6
            assert player.sample_rate == RATE / 2  # (the header's rate)
        assert detect_format(path).center_frequency == 7.1e6
        write_sidecar(path, SampleFormat.INT16, RATE, 0.0, header_bytes=44)
        with IQPlayer(path) as player:
            assert player.center_frequency == RATE / 4
        assert detect_format(path).center_frequency == RATE / 4

    def test_the_callers_values_win(self, tmp_path):
        path, _x = self._file(tmp_path)
        with IQPlayer(path, center_frequency=10.7e6) as player:
            assert player.center_frequency == 10.7e6
            assert player.sample_rate == RATE / 2

    def test_rf64_and_wave64(self, tmp_path):
        x = _cos(5001, 3000.0)
        ref = tmp_path / "classic.wav"
        iq_wav(ref, x, "int", 3)
        expected = load_iq_file(ref)[0]
        for container, name in (("RF64", "big.wav"), ("W64", "take.w64")):
            path = tmp_path / name
            iq_wav(path, x, "int", 3, container=container)
            np.testing.assert_array_equal(load_iq_file(path)[0], expected)

    def test_converting_keeps_its_rate_and_center(self, tmp_path):
        """detect_format and the reader agree, so a converted file is
        labelled with the I/Q's rate and center (not the file's rate)."""
        path, _x = self._file(tmp_path)
        out = tmp_path / "converted.sigmf-data"
        FormatConverter().convert(path, out, output_file_format=FileFormat.SIGMF)
        with IQPlayer(out) as player:
            assert player.sample_rate == RATE / 2
            assert player.center_frequency == RATE / 4
            np.testing.assert_allclose(
                player.read(player.total_samples), load_iq_file(path)[0], atol=1e-6
            )

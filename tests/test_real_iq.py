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

    def test_audio_tones_leave_no_ghost_at_the_far_edge(self, tmp_path):
        """In 48 kHz audio a Morse tone at 700 Hz (1.5% of the rate) had a
        ghost at 23.3 kHz only 14 dB down with the shortest filter; the
        filter is longer at audio rates: 700 Hz's image over 100 dB down,
        300 Hz's over 50 dB."""
        path = tmp_path / "tones.wav"
        x = _cos(int(1.2 * RATE), 700.0, amplitude=0.4)
        x += _cos(len(x), 300.0, amplitude=0.4, phase=1.1)
        iq_wav(path, x, "float", 8)
        y = load_iq_file(path)[0]
        for tone, floor_db in ((700.0, -100.0), (300.0, -50.0)):
            wanted, mirror = _components(y, tone - RATE / 4, RATE / 2, 1000, 24000)
            assert abs(wanted) == pytest.approx(0.4, rel=1e-2)
            assert 20 * np.log10(abs(mirror) / abs(wanted)) < floor_db, tone

    def test_the_filter_is_sized_by_the_rate(self):
        """Longer where it costs little (audio), the shortest at high
        rates, never more work a second than the budget allows."""
        assert real_iq.context_for(48000) == 211
        assert real_iq.context_for(2.4e6) == 39
        assert real_iq.context_for(4.8e6) == real_iq.CONTEXT
        assert real_iq.context_for(20e6) == real_iq.CONTEXT
        for rate in (8000, 11025, 44100, 96000, 192000, 1e6, 2.4e6, 10e6):
            context = real_iq.context_for(rate)
            assert context % 2 == 1 and context >= real_iq.CONTEXT
            work = (context + 1) * rate / 2
            assert work <= real_iq.WORK_PER_S or context == real_iq.CONTEXT
            assert len(real_iq.hilbert_taps(context)) == context + 1

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


class TestAOneChannelFileWithoutARate:
    """files-2: a one-channel WAV file whose header gives no sample rate
    (0). The rate it gets from elsewhere, its sidecar's or the caller's,
    was taken as its I/Q rate, though a sidecar gives the file's own rate
    (the rate of its real samples, as a header does, and as SigMF gives a
    real dataset's): every frequency came out doubled and shifted, the
    duration halved, the center 0 Hz and the filter the shortest (a 300 Hz
    tone's mirror image 5.6 dB down, against 54 dB). Now its I/Q is worked
    out from that rate as from a header's."""

    def _file(self, tmp_path, name="rate0.wav", seconds=2.0):
        frames = int(seconds * RATE)
        x = _cos(frames, 1000.0, amplitude=0.4)
        x += _cos(frames, 300.0, amplitude=0.4, phase=1.1)
        path = tmp_path / name
        write_wav(path, fmt_body(1, 0, 2), encode(x, "int", 2))
        return path, frames

    @staticmethod
    def _tones_right(y):
        """1 kHz and 300 Hz at their own frequencies (I/Q at 24 kS/s
        centred on 12 kHz), each once: the 300 Hz tone's mirror image over
        50 dB down (the filter sized for 48 kHz)."""
        for tone, floor_db in ((1000.0, -90.0), (300.0, -50.0)):
            wanted, mirror = _components(y, tone - RATE / 4, RATE / 2, 1000, 24000)
            assert abs(wanted) == pytest.approx(0.4, rel=1e-2), tone
            assert 20 * np.log10(abs(mirror) / abs(wanted)) < floor_db, tone

    def test_its_sidecars_rate_is_its_own(self, tmp_path):
        path, frames = self._file(tmp_path)
        write_sidecar(path, SampleFormat.INT16, RATE, 0.0, header_bytes=44)
        info = detect_format(path)
        assert info.is_valid, info.error_message
        assert (info.sample_rate, info.center_frequency) == (RATE / 2, RATE / 4)
        assert info.num_samples == frames // 2
        assert info.duration_seconds == pytest.approx(frames / RATE)
        assert info.metadata["wav_frame_rate"] == RATE
        with IQPlayer(path) as player:
            assert (player.sample_rate, player.center_frequency) == (RATE / 2, RATE / 4)
            assert player.duration_seconds == pytest.approx(frames / RATE)
            y = player.read(player.total_samples)
        self._tones_right(y)

    def test_a_sidecar_can_still_place_it(self, tmp_path):
        path, _frames = self._file(tmp_path)
        write_sidecar(path, SampleFormat.INT16, RATE, 10.7e6, header_bytes=44)
        assert detect_format(path).center_frequency == 10.7e6
        assert detect_format(path).sample_rate == RATE / 2
        with IQPlayer(path) as player:
            assert (player.sample_rate, player.center_frequency) == (RATE / 2, 10.7e6)

    def test_the_callers_rate_is_its_iq_rate_as_ever(self, tmp_path):
        """(IqFileOptions.open_player passes the I/Q rate, half the file's
        own.) Its own frequencies and filter follow from it: centred on a
        quarter of the file's rate unless the caller gives a center."""
        path, frames = self._file(tmp_path)
        with IQPlayer(path, sample_rate=RATE / 2) as player:
            assert (player.sample_rate, player.center_frequency) == (RATE / 2, RATE / 4)
            assert player.duration_seconds == pytest.approx(frames / RATE)
            y = player.read(player.total_samples)
        self._tones_right(y)
        with IQPlayer(path, sample_rate=RATE / 2, center_frequency=7.1e6) as player:
            assert player.center_frequency == 7.1e6
        with IQPlayer(path) as player:  # (nothing says it: unknown, as before)
            assert (player.sample_rate, player.center_frequency) == (0.0, 0.0)

    def test_two_channels_take_a_sidecars_rate_as_their_iq_rate(self, tmp_path):
        """(As before: a two-channel file's samples are I/Q, a sidecar's
        rate is theirs. detect_format's duration now counts it too.)"""
        x = np.exp(2j * np.pi * 3000.0 * np.arange(4800) / RATE).astype(np.complex64)
        path = tmp_path / "iq_rate0.wav"
        offset = write_wav(
            path, fmt_body(2, 0, 2), encode(np.column_stack([x.real, x.imag]), "int", 2)
        )
        write_sidecar(path, SampleFormat.INT16, RATE, 145e6, header_bytes=offset)
        info = detect_format(path)
        assert (info.sample_rate, info.center_frequency) == (RATE, 145e6)
        assert info.duration_seconds == pytest.approx(len(x) / RATE)
        with IQPlayer(path) as player:
            assert (player.sample_rate, player.total_samples) == (RATE, len(x))

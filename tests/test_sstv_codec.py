"""SSTV encoder and decoder (hm:HAM-01, hm:HAM-14, hm:GAP-11).

Every mode in SSTV_MODES round-trips a test card through
sdr_module.ham.sstv_encoder and SSTVDecoder, fed in the blocks the main
window feeds (33 ms of audio), with noise and a sound-card clock error;
an independent hand-written signal checks the line layouts (so encoder and
decoder can't share a mistake). Similarity is the correlation and PSNR
against the card.
"""

from __future__ import annotations

import tracemalloc

import numpy as np
import pytest

from sdr_module.ham.sstv import (
    SSTV_MODES,
    SSTVDecoder,
    find_mode,
    freq_to_value,
    line_templates,
    preamble_segments,
    rgb_to_ycc,
    vis_segments,
    ycc_to_rgb,
)
from sdr_module.ham.sstv_encoder import encode_sstv, segment_plan, sstv_test_card

FS = 48000.0
GUI_BLOCK = 1600  # 33 ms at 48 kHz: what the main window feeds per frame
KEY_MODES = ("Robot 36", "Martin 1", "Scottie 1", "PD 120")
# Standard line (sync-to-sync) periods, ms.
LINE_MS = {
    "Robot 36": 150.0,
    "Robot 72": 300.0,
    "Martin 1": 446.446,
    "Martin 2": 226.798,
    "Scottie 1": 428.22,
    "Scottie 2": 277.692,
    "Scottie DX": 1050.3,
    "PD 90": 703.04,
    "PD 120": 508.48,
    "PD 160": 804.416,
    "PD 180": 754.24,
    "PD 240": 1000.0,
    "PD 290": 937.28,
    "Wraase SC2-180": 711.0225,
}


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(float).ravel() - a.mean()
    b = b.astype(float).ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b) + 1e-12))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return float(10 * np.log10(255.0**2 / max(mse, 1e-9)))


class Events:
    def __init__(self, decoder: SSTVDecoder):
        self.modes: list = []
        self.complete: list = []
        self.lost: list = []
        decoder.set_on_mode_detected(lambda m: self.modes.append(m.name))
        decoder.set_on_image_complete(lambda im: self.complete.append(im.copy()))
        decoder.set_on_image_lost(lambda im, rows: self.lost.append((im.copy(), rows)))


def feed(decoder: SSTVDecoder, audio: np.ndarray, block: int = GUI_BLOCK) -> None:
    for i in range(0, len(audio), block):
        decoder.process_audio(audio[i : i + block])


def padded(audio: np.ndarray, fs: float = FS, before=0.3, after=0.5) -> np.ndarray:
    return np.concatenate(
        [
            np.zeros(int(before * fs), np.float32),
            audio,
            np.zeros(int(after * fs), np.float32),
        ]
    )


def add_noise(audio: np.ndarray, snr_db: float, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    power = np.mean(audio[np.abs(audio) > 0] ** 2)
    sigma = np.sqrt(power / 10 ** (snr_db / 10))
    return (audio + rng.normal(0.0, sigma, len(audio))).astype(np.float32)


def card_for(name: str) -> np.ndarray:
    spec = find_mode(name)
    return sstv_test_card(spec.width, spec.height)


# --------------------------------------------------------------------------
# Mode table and encoder
# --------------------------------------------------------------------------


class TestModeTable:
    @pytest.mark.parametrize("name", sorted(LINE_MS))
    def test_line_layout_adds_up_to_the_standard_period(self, name):
        spec = find_mode(name)
        assert spec.scan_time_ms == pytest.approx(LINE_MS[name], abs=1e-6)
        for template in line_templates(spec):
            assert sum(seg[1] for seg in template) == pytest.approx(
                spec.scan_time_ms, abs=1e-6
            )

    def test_every_table_mode_has_a_layout(self):
        assert {spec.name for spec in SSTV_MODES.values()} == set(LINE_MS)

    def test_image_durations(self):
        assert find_mode("Robot 36").duration_s == pytest.approx(36.0)
        assert find_mode("Martin 1").duration_s == pytest.approx(114.29, abs=0.01)
        # Scottie starts with one extra sync pulse.
        assert find_mode("Scottie 1").duration_s == pytest.approx(109.633, abs=0.01)
        assert find_mode("PD 120").duration_s == pytest.approx(126.1, abs=0.01)
        assert find_mode("PD 120").lines_per_sync == 2

    def test_find_mode_by_name_code_and_enum(self):
        spec = find_mode("Martin 1")
        assert find_mode(44) is spec
        assert find_mode(spec.mode) is spec
        assert find_mode("martin_1") is spec
        with pytest.raises(KeyError):
            find_mode("Martin 9")

    def test_vis_bits_lsb_first_with_even_parity(self):
        for code in (8, 44, 60, 95, 99):
            tones = [freq for _k, _ms, freq in vis_segments(code)]
            assert tones[:4] == [1900.0, 1200.0, 1900.0, 1200.0]
            assert tones[-1] == 1200.0  # stop bit
            bits = [1 if f == 1100.0 else 0 for f in tones[4:12]]
            assert sum(b << i for i, b in enumerate(bits[:7])) == code
            assert sum(bits[:7]) % 2 == bits[7]

    def test_robot36_separators_and_scottie_starting_sync(self):
        even, odd = line_templates(find_mode("Robot 36"))
        assert ("parity", 4.5, 1500.0) in even and ("V", 44.0, 0.0) in even
        assert ("parity", 4.5, 2300.0) in odd and ("U", 44.0, 0.0) in odd
        assert preamble_segments(find_mode("Scottie 1")) == [("sync", 9.0, 1200.0)]
        assert preamble_segments(find_mode("Martin 1")) == []

    def test_robot36_sends_every_lines_own_luma(self):
        """Only R-Y / B-Y are shared by a line pair; Y is per line."""
        stripes = np.zeros((240, 320, 3), np.uint8)
        stripes[1::2] = 255  # odd lines white, even lines black
        freqs, durs = segment_plan(stripes, "Robot 36", vis=False, lines=2)
        luma = freqs[np.isclose(durs, 88.0 / 320)]
        first, second = luma[:320], luma[320:640]
        assert np.allclose(first, 1500 + 800 * 16 / 255.0)  # black: Y = 16
        assert np.allclose(second, 1500 + 800 * 235 / 255.0, atol=0.5)  # white

    def test_encoded_length_is_vis_plus_image(self):
        spec = find_mode("Robot 36")
        audio = encode_sstv(card_for("Robot 36"), spec, FS)
        expected = 0.910 + spec.duration_s
        assert len(audio) / FS == pytest.approx(expected, abs=1.0 / FS)
        assert np.max(np.abs(audio)) <= 1.0

    def test_colour_conversion_round_trips(self):
        card = sstv_test_card(64, 48).astype(float)
        y, cr, cb = rgb_to_ycc(card)
        assert np.max(np.abs(ycc_to_rgb(y, cr, cb).astype(float) - card)) <= 1.0
        values = freq_to_value(np.array([1500.0, 1900.0, 2300.0, 2500.0]))
        assert values == pytest.approx([0.0, 127.5, 255.0, 255.0])


# --------------------------------------------------------------------------
# Round trips
# --------------------------------------------------------------------------

# (correlation, PSNR dB) each key mode must reach; measured: Robot 36
# 0.989 / 24.3, Martin 1 0.998 / 32.3, Scottie 1 0.998 / 31.7, PD 120
# 0.995 / 27.7 (clean); about 1 dB less with 20 dB noise.
KEY_THRESHOLDS = {
    "Robot 36": (0.98, 22.0),
    "Martin 1": (0.99, 29.0),
    "Scottie 1": (0.99, 28.0),
    "PD 120": (0.99, 25.0),
}


class TestRoundTrip:
    @pytest.mark.parametrize("name", KEY_MODES)
    def test_full_image_in_gui_blocks(self, name):
        """A whole transmission (VIS, every line), fed 33 ms at a time."""
        card = card_for(name)
        decoder = SSTVDecoder(FS)
        events = Events(decoder)
        feed(decoder, padded(encode_sstv(card, name, FS)))
        assert events.modes == [name]
        assert len(events.complete) == 1, "the image completes"
        image = events.complete[0]
        min_corr, min_psnr = KEY_THRESHOLDS[name]
        assert correlation(image, card) >= min_corr
        assert psnr(image, card) >= min_psnr
        assert not decoder.is_receiving()

    @pytest.mark.parametrize("name", sorted(LINE_MS))
    def test_every_mode_decodes(self, name):
        """Every mode: VIS and the first 24 line periods, 33 ms blocks at
        48 kHz, 20 dB SNR and the sender's clock 0.05% fast."""
        spec = find_mode(name)
        card = card_for(name)
        decoder = SSTVDecoder(FS)
        events = Events(decoder)
        audio = encode_sstv(card, spec, FS * 1.0005, lines=24)
        feed(decoder, add_noise(padded(audio, after=0.3), 20.0, seed=len(name)))
        rows = 24 * spec.lines_per_sync
        assert events.modes == [name]
        assert decoder.state.current_line >= rows - 1
        got = decoder.get_image()[: rows - 1]
        assert correlation(got, card[: rows - 1]) >= 0.97

    @pytest.mark.parametrize("name", ("Martin 2", "PD 90"))
    def test_other_sample_rates(self, name):
        """The decoder works at the audio rate it is given (16 kHz here)."""
        fs = 16000.0
        spec = find_mode(name)
        card = card_for(name)
        decoder = SSTVDecoder(fs)
        events = Events(decoder)
        audio = encode_sstv(card, spec, fs, lines=16)
        feed(decoder, padded(audio, fs, after=0.3), block=int(fs / 30))
        rows = 16 * spec.lines_per_sync - 1
        assert events.modes == [name]
        assert correlation(decoder.get_image()[:rows], card[:rows]) >= 0.97

    @pytest.mark.parametrize("name", ("Robot 36", "Scottie 1"))
    def test_noise_and_clock_error_full_image(self, name):
        """20 dB SNR, and the sender's clock 0.05% fast: the line timing
        follows (no slant) and the picture survives the noise."""
        card = card_for(name)
        audio = encode_sstv(card, name, FS * 1.0005)
        decoder = SSTVDecoder(FS)
        events = Events(decoder)
        feed(decoder, add_noise(padded(audio), 20.0))
        assert len(events.complete) == 1
        image = events.complete[0]
        min_corr, min_psnr = KEY_THRESHOLDS[name]
        assert correlation(image, card) >= min_corr - 0.005
        assert psnr(image, card) >= min_psnr - 1.5
        # No slant: the bottom rows line up with the top rows.
        for rows in (slice(0, 40), slice(-40, None)):
            assert correlation(image[rows], card[rows]) >= 0.97

    @pytest.mark.parametrize("name", ("Martin 1", "PD 120"))
    def test_noise_and_clock_error(self, name):
        spec = find_mode(name)
        card = card_for(name)
        audio = encode_sstv(card, spec, FS * 0.9995, lines=60)
        decoder = SSTVDecoder(FS)
        Events(decoder)
        feed(decoder, add_noise(padded(audio, after=0.3), 20.0, seed=5))
        rows = 60 * spec.lines_per_sync - 1
        assert decoder.state.current_line >= rows
        assert correlation(decoder.get_image()[:rows], card[:rows]) >= 0.98

    def test_block_size_does_not_matter(self):
        audio = padded(encode_sstv(card_for("Martin 2"), "Martin 2", FS, lines=30))
        images = []
        for block in (GUI_BLOCK, 4801, len(audio)):
            decoder = SSTVDecoder(FS)
            feed(decoder, audio, block)
            images.append(decoder.get_image().copy())
        assert np.array_equal(images[0], images[1])
        assert np.array_equal(images[0], images[2])

    def test_back_to_back_images(self):
        one = encode_sstv(card_for("Robot 36"), "Robot 36", FS)
        audio = np.concatenate([one, np.zeros(int(2 * FS), np.float32), one])
        decoder = SSTVDecoder(FS)
        events = Events(decoder)
        feed(decoder, padded(audio))
        assert events.modes == ["Robot 36", "Robot 36"]
        assert len(events.complete) == 2

    def test_parity_error_is_rejected(self):
        spec = find_mode("Martin 2")
        freqs, durs = segment_plan(card_for("Martin 2"), spec, lines=4)
        freqs = freqs.copy()
        parity = 4 + 7  # leader, break, leader, start bit, 7 data bits
        freqs[parity] = 2400.0 - freqs[parity]  # 1100 <-> 1300 Hz
        from sdr_module.ham.sstv_encoder import synthesize_tones

        decoder = SSTVDecoder(FS)
        events = Events(decoder)
        feed(decoder, padded(synthesize_tones(freqs, durs, FS)))
        assert events.modes == []


class TestIndependentReference:
    """Signals generated here from the published timings, not from
    line_templates/segment_plan: a layout mistake shared by the encoder and
    the decoder would show up."""

    @staticmethod
    def tones(segments):
        bounds = np.rint(np.cumsum([ms for _f, ms in segments]) * FS / 1000.0)
        counts = np.diff(np.concatenate(([0.0], bounds))).astype(int)
        inst = np.repeat([f for f, _ms in segments], counts)
        return np.sin(2 * np.pi * np.cumsum(inst) / FS).astype(np.float32)

    @staticmethod
    def vis(code):
        bits = [(code >> i) & 1 for i in range(7)]
        out = [(1900, 300), (1200, 10), (1900, 300), (1200, 30)]
        out += [(1100 if b else 1300, 30) for b in bits]
        out += [(1100 if sum(bits) % 2 else 1300, 30), (1200, 30)]
        return out

    @staticmethod
    def pixels(values, total_ms):
        values = np.asarray(values, dtype=float)
        return [(1500 + 800 * v / 255.0, total_ms / len(values)) for v in values]

    def test_martin1(self):
        card = sstv_test_card(320, 256)
        segs = self.vis(44)
        for y in range(40):
            segs += [(1200, 4.862), (1500, 0.572)]
            for ch in (1, 2, 0):  # G, B, R
                segs += self.pixels(card[y, :, ch], 146.432) + [(1500, 0.572)]
        decoder = SSTVDecoder(FS)
        feed(decoder, padded(self.tones(segs), after=0.3))
        assert decoder.get_mode().name == "Martin 1"
        assert correlation(decoder.get_image()[:39], card[:39]) >= 0.98

    def test_scottie1(self):
        card = sstv_test_card(320, 256)
        segs = self.vis(60) + [(1200, 9.0)]
        for y in range(30):
            segs += [(1500, 1.5)] + self.pixels(card[y, :, 1], 138.24)
            segs += [(1500, 1.5)] + self.pixels(card[y, :, 2], 138.24)
            segs += [(1200, 9.0), (1500, 1.5)] + self.pixels(card[y, :, 0], 138.24)
        decoder = SSTVDecoder(FS)
        feed(decoder, padded(self.tones(segs), after=0.3))
        assert decoder.get_mode().name == "Scottie 1"
        assert correlation(decoder.get_image()[:29], card[:29]) >= 0.98

    def test_robot36(self):
        card = sstv_test_card(320, 240)
        y_, cr, cb = rgb_to_ycc(card.astype(float))
        segs = self.vis(8)
        for y in range(40):
            segs += [(1200, 9.0), (1500, 3.0)] + self.pixels(y_[y], 88.0)
            pair = slice(y - y % 2, y - y % 2 + 2)
            if y % 2 == 0:  # R-Y, 160 samples (as some encoders send)
                chroma = cr[pair].mean(axis=0).reshape(-1, 2).mean(axis=1)
                segs += [(1500, 4.5), (1900, 1.5)] + self.pixels(chroma, 44.0)
            else:
                chroma = cb[pair].mean(axis=0).reshape(-1, 2).mean(axis=1)
                segs += [(2300, 4.5), (1900, 1.5)] + self.pixels(chroma, 44.0)
        decoder = SSTVDecoder(FS)
        feed(decoder, padded(self.tones(segs), after=0.3))
        assert decoder.get_mode().name == "Robot 36"
        # Chroma sent at half the horizontal resolution: a little softer.
        assert correlation(decoder.get_image()[:40], card[:40]) >= 0.96

    def test_pd120(self):
        card = sstv_test_card(640, 496)
        y_, cr, cb = rgb_to_ycc(card.astype(float))
        segs = self.vis(95)
        for p in range(12):
            a, b = 2 * p, 2 * p + 1
            segs += [(1200, 20.0), (1500, 2.08)]
            segs += self.pixels(y_[a], 121.6)
            segs += self.pixels(0.5 * (cr[a] + cr[b]), 121.6)
            segs += self.pixels(0.5 * (cb[a] + cb[b]), 121.6)
            segs += self.pixels(y_[b], 121.6)
        decoder = SSTVDecoder(FS)
        feed(decoder, padded(self.tones(segs), after=0.3))
        assert decoder.get_mode().name == "PD 120"
        assert correlation(decoder.get_image()[:24], card[:24]) >= 0.98


# --------------------------------------------------------------------------
# Without a VIS, losing the signal, memory
# --------------------------------------------------------------------------


def best_alignment(image: np.ndarray, card: np.ndarray, rows: int, step: int) -> float:
    return max(
        correlation(image[:rows], card[offset : offset + rows])
        for offset in range(0, card.shape[0] - rows + 1, step)
    )


class TestWithoutVis:
    @pytest.mark.parametrize("name", KEY_MODES)
    def test_start_now_mid_transmission(self, name):
        """The VIS was missed: Start now picks up from the line syncs."""
        spec = find_mode(name)
        card = card_for(name)
        audio = encode_sstv(card, spec, FS, vis=False)
        start = int(len(audio) * 0.3)
        decoder = SSTVDecoder(FS)
        Events(decoder)
        feed(decoder, audio[start : start + int(3 * FS)])
        assert decoder.start_now(spec)
        stop = min(len(audio), start + int((3 + 30) * FS))
        feed(decoder, audio[start + int(3 * FS) : stop])
        rows = decoder.state.current_line - 2
        assert rows >= 20
        assert (
            best_alignment(decoder.get_image(), card, rows, spec.lines_per_sync) >= 0.97
        )

    def test_mode_guess_from_sync_pulses(self):
        audio = encode_sstv(card_for("Martin 1"), "Martin 1", FS, vis=False)
        decoder = SSTVDecoder(FS)
        feed(decoder, audio[int(20 * FS) : int(24 * FS)])
        guess = decoder.guess_mode_from_syncs()
        assert guess is not None and guess[0].name == "Martin 1"
        assert guess[1] == pytest.approx(446.4, abs=1.0)
        assert "Martin 1" in decoder.get_tone()["text"]
        assert decoder.start_now()  # Auto: the guessed mode
        assert decoder.get_mode().name == "Martin 1"

    def test_start_now_needs_a_mode(self):
        decoder = SSTVDecoder(FS)
        feed(decoder, np.zeros(int(FS), np.float32))
        assert not decoder.start_now()
        assert not decoder.is_receiving()

    def test_forced_mode_starts_on_line_syncs(self):
        spec = find_mode("Scottie 2")
        card = card_for("Scottie 2")
        decoder = SSTVDecoder(FS)
        decoder.set_forced_mode(spec)
        events = Events(decoder)
        feed(decoder, padded(encode_sstv(card, spec, FS, vis=False, lines=60)))
        assert events.modes == ["Scottie 2"]
        rows = decoder.state.current_line - 1
        assert rows >= 40
        assert best_alignment(decoder.get_image(), card, rows, 1) >= 0.97

    def test_forced_mode_overrides_the_vis(self):
        decoder = SSTVDecoder(FS)
        decoder.set_forced_mode(find_mode("Martin 2"))
        events = Events(decoder)
        feed(
            decoder, padded(encode_sstv(card_for("Martin 2"), "Martin 2", FS, lines=4))
        )
        assert events.modes == ["Martin 2"]
        assert decoder.state.vis_code == 40


class TestLosingTheSignal:
    def test_fade_ends_the_image_as_partial(self):
        spec = find_mode("Martin 2")
        audio = encode_sstv(card_for("Martin 2"), spec, FS)
        cut = int(len(audio) * 0.6)
        rng = np.random.default_rng(3)
        noise = (0.3 * rng.standard_normal(int(12 * FS))).astype(np.float32)
        decoder = SSTVDecoder(FS)
        events = Events(decoder)
        feed(decoder, np.concatenate([audio[:cut], noise]))
        assert events.complete == []
        assert len(events.lost) == 1
        image, rows = events.lost[0]
        expected = (cut / FS - 0.91) * 1000.0 / spec.scan_time_ms
        assert abs(rows - expected) <= 2
        assert not image[rows + 1 :].any(), "lines after the fade are blank"
        assert not decoder.is_receiving()

    def test_audio_gap_relocks(self):
        """0.31 s of audio lost mid-image: the line timing re-locks."""
        spec = find_mode("Scottie 1")
        card = card_for("Scottie 1")
        audio = encode_sstv(card, spec, FS, lines=150)
        gap = int((0.91 + 0.009 + 60 * spec.scan_time_ms / 1000.0) * FS)
        audio = np.concatenate([audio[:gap], audio[gap + int(0.31 * FS) :]])
        decoder = SSTVDecoder(FS)
        Events(decoder)
        feed(decoder, padded(audio, after=0.5))
        image = decoder.get_image()
        assert correlation(image[:58], card[:58]) >= 0.99
        assert best_alignment(image[70:140], card, 70, 1) >= 0.99

    def test_stop_returns_the_line_count(self):
        decoder = SSTVDecoder(FS)
        feed(decoder, encode_sstv(card_for("Robot 36"), "Robot 36", FS, lines=50))
        assert decoder.is_receiving()
        assert decoder.stop() >= 48
        assert not decoder.is_receiving()
        assert decoder.get_image() is not None


class TestBoundedMemory:
    def test_long_noise_stays_bounded_and_quiet(self):
        """hm:HAM-14: hours of listening must not grow the buffers."""
        decoder = SSTVDecoder(FS)
        events = Events(decoder)
        rng = np.random.default_rng(7)
        block = (0.2 * rng.standard_normal(int(30 * FS))).astype(np.float32)
        feed(decoder, block)
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            feed(decoder, block)
            feed(decoder, block)
            grown = tracemalloc.get_traced_memory()[0] - before
        finally:
            tracemalloc.stop()
        # (The old decoder kept every 5 ms estimate: 0.4 MB a minute.)
        assert grown < 150_000, f"grew {grown} bytes in a minute"
        assert len(decoder._freq.data) <= 3.0 * FS
        assert len(decoder._coarse.data) <= 3000
        assert events.modes == []
        assert decoder.get_tone()["text"] == "Noise only, no SSTV tones"


class TestToneReadout:
    def test_leader_sync_picture_and_silence(self):
        t = np.arange(int(0.4 * FS)) / FS
        decoder = SSTVDecoder(FS)
        feed(decoder, np.zeros(int(0.2 * FS), np.float32))
        assert decoder.get_tone()["text"] == "No audio"
        feed(decoder, np.sin(2 * np.pi * 1900 * t).astype(np.float32))
        tone = decoder.get_tone()
        assert tone["text"] == "Hearing 1900 Hz leader…"
        assert tone["tone_hz"] == pytest.approx(1900, abs=5)
        assert tone["activity"] == "leader"
        feed(decoder, np.sin(2 * np.pi * 2000 * t).astype(np.float32))
        assert "picture tones" in decoder.get_tone()["text"]
        feed(decoder, np.sin(2 * np.pi * 1200 * t).astype(np.float32))
        assert decoder.get_tone()["text"] == "Hearing 1200 Hz sync tone…"

    def test_bad_samples_do_not_poison_the_decoder(self):
        decoder = SSTVDecoder(FS)
        feed(decoder, np.array([np.nan, np.inf, -np.inf, 0.5] * 400, np.float32))
        t = np.arange(int(0.4 * FS)) / FS
        feed(decoder, np.sin(2 * np.pi * 1900 * t).astype(np.float32))
        assert decoder.get_tone()["tone_hz"] == pytest.approx(1900, abs=5)

    def test_status_keys(self):
        decoder = SSTVDecoder(FS)
        feed(decoder, encode_sstv(card_for("Robot 36"), "Robot 36", FS, lines=10))
        status = decoder.get_status()
        assert status["mode"] == "Robot 36"
        assert status["is_receiving"] and status["sync_locked"]
        assert status["vis_code"] == 8
        assert status["current_line"] >= 9 and status["total_lines"] == 240
        assert 0.0 < status["progress"] < 0.1
        assert decoder.get_tone()["activity"] == "image"

#!/usr/bin/env python3
"""
The audio notch (K-A1, the audit's FG-G14): removes a heterodyne whistle
from AM, SSB, CW or narrow FM audio, at a frequency set by hand or (Auto)
at the strongest steady tone it finds and follows, without touching voice.

Plain NumPy (``sdr_module.dsp.audio_filters``): the streaming IIR engine
it runs on, the notch's design, the tone finder and the notch itself, on
synthetic voice with a real pitch (``tests/voice_synth.py``) plus a tone.
"""

import math
import unittest

import numpy as np

from tests.voice_synth import band_power, pitched_voice, tone, tone_level

RATE = 48000.0
FRAME = 1600  # 33 ms at 48 kHz: one display frame of audio


def _direct_form(sos, x, switch_at=None, sos2=None):
    """Reference: the cascade of biquads a sample at a time (direct form I),
    switching to ``sos2`` (same sections, state kept) at ``switch_at``."""
    rows = [list(r) for r in np.asarray(sos, float)]
    rows2 = None if sos2 is None else [list(r) for r in np.asarray(sos2, float)]
    state = [[0.0, 0.0, 0.0, 0.0] for _ in rows]
    out = np.empty(len(x))
    for n, sample in enumerate(np.asarray(x, float)):
        use = rows2 if (switch_at is not None and n >= switch_at) else rows
        v = sample
        for i, (b0, b1, b2, a0, a1, a2) in enumerate(use):
            x1, x2, y1, y2 = state[i]
            y = (b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0
            state[i] = [v, x1, y, y1]
            v = y
        out[n] = v
    return out


def _run(process, x, sizes):
    """Feed ``x`` to ``process`` in blocks of the given sizes (cycled)."""
    out, pos, i = [], 0, 0
    while pos < len(x):
        size = sizes[i % len(sizes)]
        out.append(np.asarray(process(x[pos : pos + size]), dtype=float))
        pos += size
        i += 1
    return np.concatenate(out)


class TestSosFilter(unittest.TestCase):
    """Second-order sections a block at a time, without a Python loop per
    sample: the same output as the textbook recursion, any block size."""

    def test_matches_the_recursion_for_any_block_size(self):
        from sdr_module.dsp.audio_filters import SosFilter, highpass_sos, notch_sos

        x = np.random.default_rng(1).standard_normal(9000)
        for sos in (
            notch_sos(1000.0, RATE),
            notch_sos(250.0, RATE, 40.0),
            highpass_sos(300.0, RATE),
            highpass_sos(100.0, RATE),
        ):
            ref = _direct_form(sos, x)
            for sizes in ([1, 2, 3, 1000], [9000], [1600], [7, 5001, 1]):
                f = SosFilter(sos)
                y = _run(f.process, x, sizes)
                self.assertLess(np.max(np.abs(y - ref)), 1e-8, sizes)

    def test_new_coefficients_keep_the_state(self):
        from sdr_module.dsp.audio_filters import SosFilter, notch_sos

        x = np.random.default_rng(2).standard_normal(4000)
        a, b = notch_sos(1000.0, RATE), notch_sos(1003.0, RATE)
        ref = _direct_form(a, x, switch_at=2500, sos2=b)
        f = SosFilter(a)
        first = f.process(x[:2500])
        f.set_sos(b)
        y = np.concatenate((first, f.process(x[2500:])))
        self.assertLess(np.max(np.abs(y - ref)), 1e-8)

    def test_empty_blocks_and_reset(self):
        from sdr_module.dsp.audio_filters import SosFilter, notch_sos

        f = SosFilter(notch_sos(1000.0, RATE))
        self.assertEqual(len(f.process(np.zeros(0))), 0)
        x = np.random.default_rng(3).standard_normal(500)
        once = f.process(x)
        f.reset()
        self.assertTrue(np.allclose(f.process(x), once))


class TestNotchDesign(unittest.TestCase):
    def test_deep_at_its_frequency_and_flat_elsewhere(self):
        from sdr_module.dsp.audio_filters import (
            NOTCH_WIDTH_HZ,
            notch_sos,
            sos_response,
        )

        sos = notch_sos(1000.0, RATE)

        def gain(f):
            return 20 * math.log10(abs(sos_response(sos, [f], RATE)[0]) + 1e-30)

        self.assertLess(gain(1000.0), -80.0)
        self.assertLess(gain(1000.25), -38.0)  # Auto finds it within 0.25 Hz
        self.assertLess(gain(999.5), -32.0)  # by hand: 1 Hz steps
        # -3 dB over NOTCH_WIDTH_HZ, then flat: voice is left alone.
        self.assertAlmostEqual(gain(1000.0 + NOTCH_WIDTH_HZ / 2), -3.0, delta=0.3)
        for f in (100.0, 700.0, 900.0, 1100.0, 1300.0, 3000.0, 10000.0):
            self.assertGreater(gain(f), -0.5, f)


class TestToneFinder(unittest.TestCase):
    """Auto's ear: the strongest steady tone, never a voice's harmonics."""

    def finder(self, **kw):
        from sdr_module.dsp.audio_filters import ToneFinder

        return ToneFinder(RATE, **kw)

    def feed(self, finder, x):
        found = []
        for pos in range(0, len(x), FRAME):
            found.append(finder.process(x[pos : pos + FRAME]))
        return found

    def test_voice_alone_is_never_taken_for_a_tone(self):
        for seed in range(8):
            f0 = 85.0 + 22.0 * seed  # deep to high voices
            voice = pitched_voice(15.0, RATE, seed=seed, f0=f0)
            found = [f for f in self.feed(self.finder(), voice) if f is not None]
            self.assertEqual(found, [], f"voice f0={f0}")

    def test_a_steady_tone_under_voice_is_found_and_held(self):
        for freq, amplitude in ((1000.0, 0.2), (612.3, 0.3), (2800.0, 0.1)):
            voice = pitched_voice(6.0, RATE, seed=3)
            x = voice + tone(len(voice), freq, amplitude)
            found = self.feed(self.finder(), x)
            first = next(i for i, f in enumerate(found) if f is not None)
            self.assertLess(first * FRAME / RATE, 1.0, freq)  # within a second
            held = found[first:]
            self.assertTrue(all(f is not None for f in held), freq)
            errors = [abs(f - freq) for f in held]
            self.assertLess(max(errors), 0.5, freq)
            self.assertLess(float(np.median(errors)), 0.05, freq)

    def test_a_weak_whistle_is_found_too(self):
        voice = pitched_voice(8.0, RATE, seed=3)
        x = voice + tone(len(voice), 1234.5, 0.05)  # 22 dB under the voice peaks
        found = self.feed(self.finder(), x)
        late = [f for f in found[int(3.0 * RATE / FRAME) :]]
        self.assertTrue(all(f is not None and abs(f - 1234.5) < 0.5 for f in late))

    def test_it_follows_a_drifting_whistle(self):
        voice = pitched_voice(10.0, RATE, seed=5)
        t = np.arange(len(voice)) / RATE
        freq = 900.0 + 2.0 * t  # 2 Hz a second
        x = voice + 0.2 * np.sin(2 * np.pi * np.cumsum(freq) / RATE)
        found = self.feed(self.finder(), x)
        errors = [
            abs(f - freq[min((i + 1) * FRAME, len(freq) - 1)])
            for i, f in enumerate(found)
            if f is not None
        ]
        self.assertGreater(len(errors), 0.8 * len(found))
        self.assertLess(max(errors), 1.0)

    def test_it_lets_go_once_the_whistle_stops(self):
        voice = pitched_voice(8.0, RATE, seed=2)
        x = voice.copy()
        n = int(4.0 * RATE)
        x[:n] += tone(n, 1500.0, 0.2)
        found = self.feed(self.finder(), x)

        def frame_of(seconds):
            return int(seconds * RATE / FRAME)

        self.assertIsNotNone(found[frame_of(3.5)])
        self.assertTrue(all(f is None for f in found[frame_of(5.5) :]))

    def test_only_its_band_and_not_the_part_it_skips(self):
        voice = pitched_voice(5.0, RATE, seed=1)
        x = voice + tone(len(voice), 700.0, 0.3)
        # (CW: the note at the CW pitch is the signal, not a whistle.)
        finder = self.finder()
        finder.set_band(200.0, 1200.0, skip=(550.0, 850.0))
        self.assertTrue(all(f is None for f in self.feed(finder, x)))
        finder = self.finder()
        finder.set_band(1000.0, 3000.0)
        self.assertTrue(all(f is None for f in self.feed(finder, x)))


class TestAudioNotch(unittest.TestCase):
    """The notch in the audio: Off, Auto or a frequency set by hand."""

    def notch(self, mode, hz=1000.0):
        from sdr_module.dsp.audio_filters import AudioNotch

        notch = AudioNotch(RATE)
        notch.mode = mode
        notch.manual_hz = hz
        return notch

    def check_removed_and_voice_kept(self, out, voice, freq, start_s):
        start = int(start_s * RATE)
        out, clean = out[start:], voice[start:]
        # The whistle is 40 dB or more down...
        self.assertLess(tone_level(out, freq), 0.2 * 0.01)
        # ...and the voice away from it is as it was (within 0.1 dB).
        for low, high in ((150, freq - 120), (freq + 120, 4000)):
            ratio = band_power(out, RATE, low, high) / band_power(
                clean, RATE, low, high
            )
            self.assertAlmostEqual(10 * math.log10(ratio), 0.0, delta=0.1)
        # (Near the notch its phase turns, which no one hears; further
        # away the waveform itself is the voice's.)
        residual = out - clean
        away = band_power(residual, RATE, 150, freq - 400) + band_power(
            residual, RATE, freq + 400, 4000
        )
        self.assertLess(away, 1e-2 * band_power(clean, RATE, 150, 4000))

    def test_off_leaves_the_audio_alone(self):
        voice = pitched_voice(2.0, RATE, seed=4)
        x = voice + tone(len(voice), 1000.0, 0.2)
        out = _run(self.notch("off").process, x, [FRAME])
        self.assertTrue(np.array_equal(out, x.astype(np.float32)))

    def test_by_hand_it_removes_the_whistle_at_its_frequency(self):
        voice = pitched_voice(4.0, RATE, seed=4)
        x = voice + tone(len(voice), 1000.0, 0.2)
        notch = self.notch("manual", 1000.0)
        out = _run(notch.process, x, [FRAME])
        self.assertEqual(notch.frequency_hz, 1000.0)
        self.check_removed_and_voice_kept(out, voice, 1000.0, 0.1)

    def test_auto_finds_and_removes_the_whistle(self):
        voice = pitched_voice(6.0, RATE, seed=6)
        x = voice + tone(len(voice), 1347.0, 0.2)
        notch = self.notch("auto")
        out = _run(notch.process, x, [FRAME])
        self.assertAlmostEqual(notch.frequency_hz, 1347.0, delta=0.3)
        self.check_removed_and_voice_kept(out, voice, 1347.0, 1.5)

    def test_auto_leaves_voice_alone(self):
        for seed in (0, 7):
            voice = pitched_voice(10.0, RATE, seed=seed, f0=110.0 + 60 * seed / 7)
            notch = self.notch("auto")
            out = _run(notch.process, voice, [FRAME])
            self.assertIsNone(notch.frequency_hz)
            self.assertTrue(np.array_equal(out, voice.astype(np.float32)))

    def test_any_block_size_gives_the_same_audio(self):
        voice = pitched_voice(3.0, RATE, seed=8)
        x = voice + tone(len(voice), 800.0, 0.2)
        for mode in ("auto", "manual"):
            whole = _run(self.notch(mode, 800.0).process, x, [len(x)])
            split = _run(self.notch(mode, 800.0).process, x, [1600, 7, 3000, 1])
            self.assertLess(np.max(np.abs(whole - split)), 1e-5, mode)

    def test_it_never_clicks_when_it_engages_or_changes(self):
        """Engaging, moving and switching off ramp over a few ms: the
        audio never steps."""
        n = int(3.0 * RATE)
        x = tone(n, 1000.0, 0.5)
        notch = self.notch("off")
        pieces = []
        for i, pos in enumerate(range(0, n, FRAME)):
            if i == 10:
                notch.mode = "manual"  # on
            if i == 30:
                notch.manual_hz = 1001.0  # a step of the spin box
            if i == 50:
                notch.manual_hz = 2500.0  # typed: somewhere else
            if i == 70:
                notch.mode = "off"
            pieces.append(np.asarray(notch.process(x[pos : pos + FRAME]), float))
        out = np.concatenate(pieces)
        # A 1 kHz tone at 0.5 moves at most 0.065 a sample; a click is a
        # jump far beyond that.
        self.assertLess(np.max(np.abs(np.diff(out))), 0.08)

    def test_new_stream_keeps_what_auto_found(self):
        voice = pitched_voice(3.0, RATE, seed=9)
        x = voice + tone(len(voice), 1500.0, 0.2)
        notch = self.notch("auto")
        _run(notch.process, x, [FRAME])
        self.assertIsNotNone(notch.frequency_hz)
        notch.reset_stream()  # (the squelch closed and opened again)
        self.assertAlmostEqual(notch.frequency_hz, 1500.0, delta=0.3)
        notch.reset()  # (another channel, another mode)
        self.assertIsNone(notch.frequency_hz)


if __name__ == "__main__":
    unittest.main()

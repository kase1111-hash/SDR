"""The time-stretch (WSOLA) behind the sound of a recording played at 0.5x,
2x or 4x (M-P1): faster or slower, its pitch kept, without a click where
two frames or two blocks meet, mono and stereo, streamed a block at a time.

Before, a recording's sound played at 1x only: at any other speed the
speaker was paused ("Playing at 2x: audio plays at 1x only").
"""

import unittest
from unittest import mock

import numpy as np

from sdr_module.dsp.time_stretch import (
    FRAME_S,
    SEARCH_S,
    TimeStretch,
    time_stretch,
)
from tests.voice_synth import pitched_voice

RATE = 48000.0
RATIOS = (0.5, 2.0, 4.0)


def _tone(seconds, freq, amplitude=0.5, rate=RATE, phase=0.3):
    t = np.arange(int(round(seconds * rate))) / rate
    return (amplitude * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def _vowel(f0, seconds=3.0, rate=RATE, seed=0):
    """A steady voiced vowel (formants at 700, 1700 and 2600 Hz) at ``f0``."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    x = np.zeros(len(t))
    for h in range(1, int(4000 / f0)):
        fh = h * f0
        gain = (
            1 / (1 + ((fh - 700) / 250) ** 2)
            + 0.6 / (1 + ((fh - 1700) / 300) ** 2)
            + 0.25 / (1 + ((fh - 2600) / 400) ** 2)
        )
        x += gain / h * np.sin(2 * np.pi * fh * t + rng.uniform(0, 2 * np.pi))
    return (0.5 * x / np.max(np.abs(x))).astype(np.float32)


def _peak_hz(y, rate=RATE):
    """The strongest frequency, interpolated between bins."""
    y = np.asarray(y, dtype=float)
    pad = 8 * len(y)
    s = np.abs(np.fft.rfft(y * np.hanning(len(y)), pad))
    k = int(np.argmax(s[1:])) + 1
    a, b, c = np.log(s[k - 1 : k + 2] + 1e-30)
    return (k + 0.5 * (a - c) / (a - 2 * b + c)) * rate / pad


def _cents(found, wanted):
    return 1200.0 * np.log2(found / wanted)


def _off_tone_db(y, freq, rate=RATE):
    """Power away from ``freq`` (+-60 Hz) against all of it."""
    y = np.asarray(y, dtype=float)
    s = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / rate)
    near = np.abs(f - freq) < 60
    return 10 * np.log10(s[~near].sum() / s.sum())


def _harmonic_db(y, f0, rate=RATE):
    """Power on the harmonics of ``f0`` (+-15 Hz) against the rest, up to
    4 kHz: how clean a stretched vowel stays."""
    y = np.asarray(y, dtype=float)
    s = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    f = np.fft.rfftfreq(len(y), 1 / rate)
    band = f < 4000
    near = np.zeros(len(f), dtype=bool)
    for h in range(1, int(4000 / f0) + 1):
        near |= np.abs(f - h * f0) < 15
    return 10 * np.log10(s[band & near].sum() / s[band & ~near].sum())


def _steps(y, freq, amplitude=0.5, rate=RATE):
    """The largest second difference against a clean sine's: 1.0 for a
    sine with no step in it; a click (a step) is many times that."""
    y = np.asarray(y, dtype=float)
    if y.ndim == 2:
        y = y[:, 0]
    clean = amplitude * (2 * np.pi * freq / rate) ** 2
    return float(np.max(np.abs(np.diff(y, 2)))) / clean


def _in_blocks(stretch, x, ratio, rng, sizes=(1, 4000), pulls=(0, 3000)):
    """Feed ``x`` in random blocks, pulling random amounts in between; then
    the rest."""
    out, i = [], 0
    while i < len(x):
        k = int(rng.integers(*sizes))
        stretch.feed(x[i : i + k])
        i += k
        out.append(stretch.pull(int(rng.integers(*pulls)), ratio))
    out.append(stretch.pull(10**9, ratio))
    return np.concatenate(out)


class TestPitchAndLength(unittest.TestCase):
    """Faster or slower, the pitch stays where it was."""

    def test_tones_keep_their_pitch(self):
        for freq in (100.0, 440.0, 1000.0, 3000.0):
            x = _tone(3.0, freq)
            for ratio in RATIOS:
                with self.subTest(freq=freq, ratio=ratio):
                    y = time_stretch(x, ratio)
                    middle = y[len(y) // 4 : 3 * len(y) // 4]
                    self.assertLess(abs(_cents(_peak_hz(middle), freq)), 0.1)
                    # Nothing but the tone: no warble, no beating.
                    self.assertLess(_off_tone_db(middle, freq), -55.0)

    def test_the_length_follows_the_ratio(self):
        x = _tone(3.0, 440.0)
        for ratio in RATIOS:
            with self.subTest(ratio=ratio):
                y = time_stretch(x, ratio)
                # (Within a frame: the end is faded out over its last hop.)
                self.assertAlmostEqual(len(y) / RATE, 3.0 / ratio, delta=FRAME_S)

    def test_a_voice_keeps_its_pitch_and_stays_clean(self):
        # Down to a 90 Hz voice: the search reaches half its period.
        for f0 in (90.0, 140.0, 220.0):
            x = _vowel(f0)
            for ratio in RATIOS:
                with self.subTest(f0=f0, ratio=ratio):
                    y = time_stretch(x, ratio)
                    middle = y[len(y) // 6 : -len(y) // 6]
                    self.assertGreater(_harmonic_db(middle, f0), 35.0)

    def test_speech_keeps_its_formants(self):
        # A voice whose pitch glides: its spectrum's shape (the formants
        # that make the vowel) is the same at every speed.
        x = pitched_voice(4.0, RATE, seed=3).astype(np.float32)

        def shape(y):
            s = np.abs(np.fft.rfft(np.asarray(y, float) * np.hanning(len(y)))) ** 2
            f = np.fft.rfftfreq(len(y), 1 / RATE)
            edges = np.arange(200.0, 4001.0, 200.0)
            bands = np.array(
                [s[(f >= lo) & (f < lo + 200.0)].sum() for lo in edges[:-1]]
            )
            return 10 * np.log10(bands / bands.sum())

        before = shape(x)
        for ratio in RATIOS:
            with self.subTest(ratio=ratio):
                after = shape(time_stretch(x, ratio))
                self.assertLess(float(np.max(np.abs(after - before))), 3.0)


class TestNoClicks(unittest.TestCase):
    """Where frames meet and where blocks meet, the sound never steps."""

    def test_frames_join_without_a_step(self):
        for freq in (440.0, 1000.0):
            x = _tone(2.0, freq)
            for ratio in RATIOS:
                with self.subTest(freq=freq, ratio=ratio):
                    y = time_stretch(x, ratio)
                    self.assertLess(_steps(y[: -int(2 * FRAME_S * RATE)], freq), 1.05)

    def test_any_block_sizes_give_the_same_sound(self):
        x = pitched_voice(3.0, RATE, seed=5).astype(np.float32)
        rng = np.random.default_rng(1)
        for ratio in RATIOS:
            with self.subTest(ratio=ratio):
                whole = TimeStretch(RATE)
                whole.feed(x)
                expected = whole.pull(10**9, ratio)
                got = _in_blocks(TimeStretch(RATE), x, ratio, rng)
                self.assertEqual(len(got), len(expected))
                np.testing.assert_array_equal(got, expected)

    def test_a_ratio_that_changes_doesnt_click(self):
        x = _tone(4.0, 440.0)
        stretch = TimeStretch(RATE)
        stretch.feed(x)
        out = []
        for ratio in (2.0, 1.3, 0.5, 4.0, 0.8, 2.5):
            out.append(stretch.pull(int(0.25 * RATE), ratio))
        y = np.concatenate(out)
        self.assertLess(_steps(y, 440.0), 1.05)
        self.assertLess(abs(_cents(_peak_hz(y), 440.0)), 0.1)

    def test_it_starts_on_the_first_sample(self):
        # The first frame starts at full level (no fade-in of its own): a
        # stream taken over mid-sound continues it without a dip.
        x = _tone(0.5, 440.0)
        y = time_stretch(x, 2.0)
        hop = TimeStretch(RATE).hop
        np.testing.assert_array_equal(y[:hop], x[:hop])

    def test_silence_stays_silent(self):
        y = time_stretch(np.zeros(int(RATE), np.float32), 2.0)
        self.assertEqual(float(np.max(np.abs(y))), 0.0)


class TestStereo(unittest.TestCase):
    def test_left_and_right_stay_apart_and_in_step(self):
        left = _tone(2.0, 440.0)
        right = _tone(2.0, 660.0, amplitude=0.3)
        x = np.stack([left, right], axis=1)
        for ratio in RATIOS:
            with self.subTest(ratio=ratio):
                y = time_stretch(x, ratio)
                self.assertEqual(y.ndim, 2)
                self.assertEqual(y.shape[1], 2)
                middle = y[len(y) // 4 : 3 * len(y) // 4]
                self.assertLess(abs(_cents(_peak_hz(middle[:, 0]), 440.0)), 0.1)
                self.assertLess(abs(_cents(_peak_hz(middle[:, 1]), 660.0)), 0.1)
                # One frame position for both sides: what is the same on
                # both stays the same on both.
                same = time_stretch(np.stack([left, left], axis=1), ratio)
                np.testing.assert_array_equal(same[:, 0], same[:, 1])

    def test_mono_and_stereo_blocks_in_one_stream(self):
        # Broadcast FM's stereo comes and goes: the stream turns stereo at
        # its first stereo block (mono plays on both sides), without a step.
        mono = _tone(1.0, 440.0)
        n = len(mono) // 2
        stretch = TimeStretch(RATE)
        out = [stretch.pull(0, 2.0)]
        stretch.feed(mono[:n])
        out.append(stretch.pull(int(0.2 * RATE), 2.0))
        stretch.feed(np.stack([mono[n:], mono[n:]], axis=1))
        self.assertEqual(stretch.channels, 2)
        out.append(stretch.pull(10**9, 2.0))
        both = [np.repeat(o[:, None], 2, axis=1) if o.ndim == 1 else o for o in out]
        y = np.concatenate(both)
        self.assertLess(_steps(y, 440.0), 1.05)
        np.testing.assert_array_equal(y[:, 0], y[:, 1])


class TestStreamEnds(unittest.TestCase):
    def test_finish_plays_the_rest_and_fades_out(self):
        x = _tone(1.0, 440.0)
        stretch = TimeStretch(RATE)
        stretch.feed(x)
        played = stretch.pull(int(0.1 * RATE), 2.0)
        rest = stretch.finish(2.0)
        total = (len(played) + len(rest)) / RATE
        self.assertAlmostEqual(total, 0.5, delta=FRAME_S)
        self.assertLess(abs(float(rest[-1])), 1e-3)  # (down to silence)
        self.assertEqual(stretch.pending, 0)  # (and ready for the next)
        y = np.concatenate([played, rest])
        self.assertLess(_steps(y[: -int(2 * FRAME_S * RATE)], 440.0), 1.05)

    def test_drain_hands_the_input_back_exactly(self):
        # Stretched for a while, then unstretched again: the drained output
        # continues the input sample for sample where the last frame left
        # it, so the blocks that follow, played as they are, join without
        # a step.
        x = _tone(2.0, 440.0)
        cut = int(1.0 * RATE)
        stretch = TimeStretch(RATE)
        stretch.feed(x[:cut])
        played = stretch.pull(int(0.6 * RATE), 0.8)
        drained = stretch.drain()
        y = np.concatenate([played, drained, x[cut:]])
        self.assertLess(_steps(y, 440.0), 1.05)
        # What is drained ends exactly where the next block starts.
        np.testing.assert_allclose(drained[-200:], x[cut - 200 : cut], atol=1e-6)
        self.assertEqual(stretch.pending, 0)

    def test_drain_before_any_frame_gives_the_input_as_it_is(self):
        x = _tone(0.01, 440.0)
        stretch = TimeStretch(RATE)
        stretch.feed(x)
        np.testing.assert_array_equal(stretch.drain(), x)

    def test_reset_forgets_the_stream(self):
        stretch = TimeStretch(RATE)
        stretch.feed(np.ones((1000, 2), np.float32))
        stretch.reset()
        self.assertEqual(stretch.channels, 1)
        self.assertEqual(stretch.pending, 0)
        self.assertEqual(len(stretch.pull(100, 1.0)), 0)


class TestWhatTheFollowerAsks(unittest.TestCase):
    def test_pending_counts_the_input_not_yet_played(self):
        stretch = TimeStretch(RATE)
        stretch.feed(np.zeros(9600, np.float32))
        self.assertEqual(stretch.pending, 9600)
        stretch.pull(stretch.hop * 4, 2.0)
        self.assertEqual(stretch.pending, 9600 - 4 * stretch.hop * 2)

    def test_most_ratio_never_asks_for_input_not_there(self):
        stretch = TimeStretch(RATE)
        stretch.feed(np.zeros(int(0.2 * RATE), np.float32))
        count = 3 * stretch.hop
        fastest = stretch.most_ratio(count)
        self.assertGreater(fastest, 0.0)
        out = stretch.pull(count, fastest)
        self.assertEqual(len(out), count)
        # A frame too far: none can be made at all.
        empty = TimeStretch(RATE)
        empty.feed(np.zeros(10, np.float32))
        self.assertEqual(empty.most_ratio(count), 0.0)

    def test_settings(self):
        stretch = TimeStretch(RATE)
        self.assertEqual(stretch.frame, 2 * stretch.hop)
        self.assertAlmostEqual(stretch.frame / RATE, FRAME_S, delta=1 / RATE)
        self.assertAlmostEqual(stretch.search / RATE, SEARCH_S, delta=1 / RATE)
        self.assertEqual(stretch.lookahead, stretch.search + stretch.frame)
        with self.assertRaises(ValueError):
            stretch.pull(10, 0.0)
        with self.assertRaises(ValueError):
            stretch.pull(10, float("nan"))
        with self.assertRaises(ValueError):
            stretch.feed(np.zeros((10, 3), np.float32))


class TestCost(unittest.TestCase):
    def test_the_work_follows_the_sound_played_not_the_speed(self):
        # Two forward FFTs (and an inverse) per hop of output whatever the
        # ratio: 4x costs what 1x does for the same seconds played
        # (counted, not timed).
        calls = []
        real = np.fft.rfft

        def counted(*args, **kwargs):
            calls.append(1)
            return real(*args, **kwargs)

        x = _tone(8.0, 440.0)
        counts = {}
        for ratio in (0.5, 4.0):
            calls.clear()
            stretch = TimeStretch(RATE)
            stretch.feed(x)
            with mock.patch.object(np.fft, "rfft", counted):
                stretch.pull(int(1.0 * RATE), ratio)
            counts[ratio] = len(calls)
        hops = RATE / TimeStretch(RATE).hop
        self.assertLessEqual(counts[4.0], 2 * hops + 2)
        self.assertEqual(counts[0.5], counts[4.0])


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
The audio notch with several whistles (M-A3): Auto finds and removes up
to three steady whistles at once, each engaged and released on its own
without a click, the strongest first; voice and the CW note are still
never notched; Manual stays one frequency set by hand.

Plain NumPy (``sdr_module.dsp.audio_filters``), on synthetic voice with a
real pitch (``tests/voice_synth.py``) plus steady tones.
"""

import math
import unittest

import numpy as np

from tests.voice_synth import band_power, pitched_voice, tone, tone_level

RATE = 48000.0
FRAME = 1600  # 33 ms at 48 kHz: one display frame of audio


def _run(process, x, sizes=(FRAME,)):
    """Feed ``x`` to ``process`` in blocks of the given sizes (cycled)."""
    out, pos, i = [], 0, 0
    while pos < len(x):
        size = sizes[i % len(sizes)]
        out.append(np.asarray(process(x[pos : pos + size]), dtype=float))
        pos += size
        i += 1
    return np.concatenate(out)


def _notch(mode="auto", hz=1000.0):
    from sdr_module.dsp.audio_filters import AudioNotch

    notch = AudioNotch(RATE)
    notch.mode = mode
    notch.manual_hz = hz
    return notch


def _whistles(n, specs):
    """Steady tones ``[(hz, amplitude), ...]``."""
    return sum(
        tone(n, hz, amp, RATE, phase=0.4 + k) for k, (hz, amp) in enumerate(specs)
    )


def _feed(notch, x, size=FRAME):
    """Output, and what the notch says it removes after each block."""
    out, said = [], []
    for p in range(0, len(x), size):
        out.append(np.asarray(notch.process(x[p : p + size]), dtype=float))
        said.append(tuple(notch.frequencies_hz))
    return np.concatenate(out), said


class TestSeveralWhistles(unittest.TestCase):
    TWO = ((700.0, 0.2), (1900.0, 0.08))
    THREE = ((612.3, 0.2), (1347.0, 0.1), (2800.0, 0.05))

    def check(self, specs, seed, seconds=6.0):
        voice = pitched_voice(seconds, RATE, seed=seed)
        x = voice + _whistles(len(voice), specs)
        notch = _notch()
        out, said = _feed(notch, x)
        # Every one found within about a second, and held from then on
        # (a voice passing over one moves it by up to a hertz for a moment,
        # as it did with one notch).
        first = int(1.0 * RATE / FRAME)
        errors = {hz: [] for hz, _amp in specs}
        for frame in said[first:]:
            self.assertEqual(len(frame), len(specs), frame)
            for (hz, _amp), found in zip(sorted(specs), sorted(frame), strict=True):
                errors[hz].append(abs(found - hz))
        for hz, error in errors.items():
            self.assertLess(max(error), 1.5, hz)
            self.assertLess(float(np.median(error)), 0.05, hz)
        # Each at least 40 dB down from a second on...
        tail, clean = out[int(1.0 * RATE) :], voice[int(1.0 * RATE) :]
        for hz, amp in specs:
            self.assertLess(tone_level(tail, hz), amp * 0.01, hz)
        # ...and the voice as it was, but for the notches' own skirts: more
        # than 200 Hz from every notch within 0.1 dB, and everywhere but
        # their +/-60 Hz what notches fixed at those frequencies leave of
        # it (Auto changes nothing else).
        found = sorted(notch.frequencies_hz)
        for guard, reference, delta in (
            (200.0, clean, 0.1),
            (60.0, self.through_fixed_notches(voice, found)[int(1.0 * RATE) :], 0.03),
        ):
            edges = [150.0]
            for hz in found:
                edges += [hz - guard, hz + guard]
            edges.append(4000.0)
            for low, high in zip(edges[::2], edges[1::2], strict=True):
                if high - low < 50.0:
                    continue
                ratio = band_power(tail, RATE, low, high) / band_power(
                    reference, RATE, low, high
                )
                self.assertAlmostEqual(
                    10 * math.log10(ratio), 0.0, delta=delta, msg=(guard, low, high)
                )
        return notch

    @staticmethod
    def through_fixed_notches(x, freqs):
        from sdr_module.dsp.audio_filters import SosFilter, notch_sos

        y = np.asarray(x, dtype=float)
        for hz in freqs:
            y = SosFilter(notch_sos(hz, RATE)).process(y)
        return y

    def test_two_whistles_are_both_removed(self):
        for seed in (3, 6):
            with self.subTest(seed=seed):
                self.check(self.TWO, seed)

    def test_three_whistles_are_all_removed(self):
        for seed in (4, 8):
            with self.subTest(seed=seed):
                self.check(self.THREE, seed)

    def test_the_strongest_first(self):
        notch = self.check(self.THREE, 4)
        found = notch.frequencies_hz
        self.assertEqual([round(f) for f in found], [612, 1347, 2800])
        # The one-whistle API: the strongest.
        self.assertAlmostEqual(notch.frequency_hz, 612.3, delta=0.3)
        self.assertAlmostEqual(notch.finder.frequency_hz, 612.3, delta=0.3)
        self.assertEqual(notch.finder.frequencies_hz, found)

    def test_with_more_whistles_than_notches_the_strongest_go(self):
        specs = ((500.0, 0.2), (1100.0, 0.15), (1700.0, 0.1), (2600.0, 0.02))
        voice = pitched_voice(6.0, RATE, seed=5)
        x = voice + _whistles(len(voice), specs)
        notch = _notch()
        out, _said = _feed(notch, x)
        self.assertEqual([round(f) for f in notch.frequencies_hz], [500, 1100, 1700])
        tail = out[int(1.5 * RATE) :]
        for hz, amp in specs[:3]:
            self.assertLess(tone_level(tail, hz), amp * 0.01, hz)
        # (The weakest plays on: three notches at most.)
        self.assertGreater(tone_level(tail, 2600.0), 0.9 * 0.02)

    def test_a_much_stronger_new_whistle_takes_the_weakest_ones_notch(self):
        specs = ((500.0, 0.05), (1100.0, 0.04), (1700.0, 0.03))
        n = int(5.0 * RATE)
        hiss = 0.002 * np.random.default_rng(2).standard_normal(n)
        x = hiss + _whistles(n, specs)
        late = int(2.0 * RATE)
        x[late:] += tone(n - late, 2300.0, 0.25, RATE, phase=0.0)
        notch = _notch()
        out, _said = _feed(notch, x)
        self.assertEqual([round(f) for f in notch.frequencies_hz], [2300, 500, 1100])
        tail = out[int(3.5 * RATE) :]
        self.assertLess(tone_level(tail, 2300.0), 0.25 * 0.01)

    def test_each_comes_and_goes_on_its_own_without_a_click(self):
        """A from the start, B from 1.5 s to 4 s, C from 2.5 s: each notch
        engages and lets go by itself (A stays removed throughout) and the
        audio never steps."""
        n = int(6.0 * RATE)
        t = np.arange(n) / RATE
        a = 0.2 * np.sin(2 * np.pi * 800.0 * t)
        b_on, b_hz = int(1.5 * RATE), 1500.0
        b_off = b_on + int(round(2.5 * b_hz)) * int(RATE / b_hz)  # (whole cycles)
        b = np.zeros(n)
        b[b_on:b_off] = 0.15 * np.sin(2 * np.pi * b_hz * (t[b_on:b_off] - t[b_on]))
        c_on = int(2.5 * RATE)
        c = np.zeros(n)
        c[c_on:] = 0.1 * np.sin(2 * np.pi * 2400.0 * (t[c_on:] - t[c_on]))
        x = a + b + c
        notch = _notch()
        out, said = _feed(notch, x)

        def at(seconds):
            return said[int(seconds * RATE / FRAME)]

        self.assertEqual([round(f) for f in at(1.4)], [800])
        self.assertEqual(sorted(round(f) for f in at(2.4)), [800, 1500])
        self.assertEqual(sorted(round(f) for f in at(3.9)), [800, 1500, 2400])
        self.assertEqual(sorted(round(f) for f in at(5.9)), [800, 2400])
        # A is removed throughout, while B and C come and go.
        removed = out[int(1.0 * RATE) :]
        self.assertLess(tone_level(removed, 800.0), 0.2 * 0.01)
        # No click: no step bigger than the tones' own.
        natural = 2 * sum(
            amp * math.sin(math.pi * hz / RATE)
            for hz, amp in ((800.0, 0.2), (b_hz, 0.15), (2400.0, 0.1))
        )
        self.assertLess(float(np.max(np.abs(np.diff(out)))), natural)

    def test_any_block_size_gives_the_same_audio(self):
        voice = pitched_voice(4.0, RATE, seed=8)
        x = voice + _whistles(len(voice), self.THREE)
        notch = _notch()
        whole = _run(notch.process, x, (len(x),))
        self.assertEqual(len(notch.frequencies_hz), 3)
        split = _run(_notch().process, x, (1600, 7, 3000, 1))
        self.assertLess(np.max(np.abs(whole - split)), 1e-5)


class TestAFinderOfItsOwn(unittest.TestCase):
    def test_a_finder_following_one_tone_uses_one_notch(self):
        # (A library user's own ToneFinder: the notches it has no tone for
        # stay idle.)
        from sdr_module.dsp.audio_filters import ToneFinder

        notch = _notch()
        notch.finder = ToneFinder(RATE, max_tones=1)
        specs = ((800.0, 0.2), (1500.0, 0.1))
        x = _whistles(int(3.0 * RATE), specs)
        out, _said = _feed(notch, x)
        found = notch.frequencies_hz
        self.assertEqual(len(found), 1)
        notch.reset_stream()  # (every notch given what it is to remove)
        tail = out[int(1.5 * RATE) :]
        for hz, amp in specs:
            if abs(hz - found[0]) < 1.0:
                self.assertLess(tone_level(tail, hz), amp * 0.01)
            else:
                self.assertGreater(tone_level(tail, hz), 0.9 * amp)

    def test_a_finder_following_four_tones_gets_four_notches(self):
        # (What it says it removes is what it removes: a notch per place.)
        from sdr_module.dsp.audio_filters import ToneFinder

        notch = _notch()
        notch.finder = ToneFinder(RATE, max_tones=4)
        specs = ((600.0, 0.2), (1100.0, 0.15), (1700.0, 0.1), (2500.0, 0.08))
        x = _whistles(int(3.0 * RATE), specs)
        out, _said = _feed(notch, x)
        self.assertEqual(len(notch.frequencies_hz), 4)
        tail = out[int(1.5 * RATE) :]
        for hz, amp in specs:
            self.assertLess(tone_level(tail, hz), amp * 0.01, hz)


class TestStillNeverVoiceOrTheCwNote(unittest.TestCase):
    """The finder's protections hold for every notch: a voice's harmonics
    glide and never count, and in CW the note is the signal."""

    def test_voice_with_one_whistle_never_gets_a_second_notch(self):
        for seed, f0 in ((0, 95.0), (7, 180.0)):
            with self.subTest(seed=seed):
                voice = pitched_voice(30.0, RATE, seed=seed, f0=f0)
                x = voice + tone(len(voice), 1234.5, 0.1, RATE)
                _out, said = _feed(_notch(), x)
                found = [f for frame in said for f in frame]
                self.assertTrue(found)
                self.assertTrue(all(abs(f - 1234.5) < 0.5 for f in found))
                self.assertTrue(all(len(frame) <= 1 for frame in said))

    def test_voice_alone_is_never_notched(self):
        for seed in range(4):
            voice = pitched_voice(12.0, RATE, seed=10 + seed, f0=100.0 + 25 * seed)
            notch = _notch()
            out, said = _feed(notch, voice)
            self.assertTrue(all(frame == () for frame in said), seed)
            self.assertTrue(np.array_equal(out, voice.astype(np.float32)))

    def test_the_cw_note_is_left_alone_while_whistles_go(self):
        from sdr_module.gui.audio_chain import AudioOptions, ListeningFilters

        filters = ListeningFilters(RATE)
        options = AudioOptions()
        options.notch = "auto"
        pitch, width = 700.0, 500.0
        band = (pitch - width / 2, pitch + width / 2)
        n = int(5.0 * RATE)
        t = np.arange(n) / RATE
        # A keyed CW note (dits and dahs) and two steady whistles.
        keying = (np.sin(2 * np.pi * 1.7 * t) > -0.3).astype(float)
        note = 0.3 * keying * np.sin(2 * np.pi * pitch * t)
        whistles = _whistles(n, ((480.0, 0.1), (920.0, 0.08)))
        out = _run(
            lambda block: filters.process(block, options, True, band, pitch),
            note + whistles,
        )
        found = sorted(round(f) for f in filters.notch_frequencies_hz)
        self.assertEqual(found, [480, 920])
        tail = slice(int(2.0 * RATE), None)
        for hz, amp in ((480.0, 0.1), (920.0, 0.08)):
            self.assertLess(tone_level(out[tail], hz), amp * 0.01)
        heard = tone_level(out[tail], pitch) / tone_level(note[tail], pitch)
        self.assertAlmostEqual(20 * math.log10(heard), 0.0, delta=0.3)


class TestManualIsOneFrequency(unittest.TestCase):
    def test_manual_removes_the_frequency_set_and_nothing_else(self):
        voice = pitched_voice(3.0, RATE, seed=3)
        x = voice + _whistles(len(voice), ((1000.0, 0.2), (1900.0, 0.1)))
        notch = _notch("manual", 1000.0)
        out, said = _feed(notch, x)
        self.assertTrue(all(frame == (1000.0,) for frame in said))
        self.assertEqual(notch.frequency_hz, 1000.0)
        tail = out[int(0.2 * RATE) :]
        self.assertLess(tone_level(tail, 1000.0), 0.2 * 0.01)
        self.assertGreater(tone_level(tail, 1900.0), 0.09)

    def test_off_removes_nothing(self):
        notch = _notch("off")
        self.assertEqual(notch.frequencies_hz, ())
        self.assertIsNone(notch.frequency_hz)

    def test_from_auto_to_manual_and_back_without_a_click(self):
        n = int(6.0 * RATE)
        x = _whistles(n, ((800.0, 0.2), (1500.0, 0.15), (2400.0, 0.1)))
        notch = _notch()
        pieces = []
        for i, p in enumerate(range(0, n, FRAME)):
            if i == 60:
                notch.mode = "manual"
                notch.manual_hz = 1500.0
            if i == 120:
                notch.mode = "auto"
            pieces.append(np.asarray(notch.process(x[p : p + FRAME]), float))
        out = np.concatenate(pieces)
        natural = 2 * sum(
            amp * math.sin(math.pi * hz / RATE)
            for hz, amp in ((800.0, 0.2), (1500.0, 0.15), (2400.0, 0.1))
        )
        self.assertLess(float(np.max(np.abs(np.diff(out)))), natural)
        self.assertEqual(len(notch.frequencies_hz), 3)


class TestTheChain(unittest.TestCase):
    """The receiver chain says every frequency it notches (the one-whistle
    property stays: the strongest)."""

    def test_notch_frequencies_through_the_chain(self):
        try:
            from sdr_module.gui.main_window import _ReceiverChain
        except ImportError:  # pragma: no cover - environment-dependent
            self.skipTest("PyQt6 not available")
        from sdr_module.gui.audio_chain import AudioOptions

        rate = 2.4e6
        chain = _ReceiverChain(rate, "USB", 2.7e3)
        chain.options = AudioOptions()
        chain.options.notch = "auto"
        chain.options.agc_speed = "off"
        n = int(3.0 * rate)
        t = np.arange(n, dtype=np.float64) / rate
        iq = 0.01 * np.exp(2j * np.pi * 700.0 * t) + 0.006 * np.exp(
            2j * np.pi * 1900.0 * t
        )
        iq = iq.astype(np.complex64)
        for p in range(0, n, 80000):
            chain.demodulate(chain.channelize(iq[p : p + 80000]), 0.0)
        found = chain.notch_frequencies_hz
        self.assertEqual([round(f) for f in found], [700, 1900])
        self.assertAlmostEqual(chain.notch_frequency_hz, found[0])


class TestTheFinderInDetail(unittest.TestCase):
    """Several whistles there from the start, coming together, changing
    level or ending; and a notch moved while it still fades in."""

    def test_found_together_the_three_strongest_go(self):
        # Four steady whistles within 6 dB, all there from the start (as on
        # tuning to a busy channel): the three strongest are removed. (The
        # first spectrum of a stream can't refine a peak's frequency: one
        # away from its bin's centre, as 2500 Hz is, counted a spectrum
        # later, and the three others took every notch.)
        n = int(3.0 * RATE)
        cases = (
            (((600.0, 0.10), (1100.0, 0.13), (1700.0, 0.16), (2500.0, 0.19)), None),
            (((530.0, 0.10), (1213.0, 0.125), (1790.0, 0.19), (2644.0, 0.15)), 6),
        )
        for specs, seed in cases:
            with self.subTest(specs=specs, voice=seed):
                if seed is None:
                    under = 0.002 * np.random.default_rng(2).standard_normal(n)
                else:
                    under = pitched_voice(3.0, RATE, seed=seed)
                notch = _notch()
                out, _said = _feed(notch, under + _whistles(n, specs))
                ranked = sorted(specs, key=lambda spec: -spec[1])
                self.assertEqual(
                    sorted(round(f) for f in notch.frequencies_hz),
                    sorted(round(hz) for hz, _amp in ranked[:3]),
                )
                tail = out[int(1.5 * RATE) :]
                for hz, amp in ranked[:3]:
                    self.assertLess(tone_level(tail, hz), amp * 0.01, hz)

    def test_two_that_come_within_two_bins_count_as_one(self):
        # B, 27 Hz above A, is found as a whistle of its own; drifting to
        # 20 Hz from A it would be a second notch on top of A's: it is let
        # go (A's notch takes it partly), never two notches that close.
        n = int(9.0 * RATE)
        t = np.arange(n) / RATE
        b_hz = np.interp(t, [0.0, 2.0, 6.0, 9.0], [1027.0, 1027.0, 1020.0, 1020.0])
        x = 0.15 * np.sin(2 * np.pi * 1000.0 * t)
        x = x + 0.12 * np.sin(2 * np.pi * np.cumsum(b_hz) / RATE)
        x = x + 0.001 * np.random.default_rng(1).standard_normal(n)
        notch = _notch()
        _out, said = _feed(notch, x)
        self.assertEqual(
            sorted(round(f) for f in said[int(1.9 * RATE / FRAME)]), [1000, 1027]
        )
        apart = 2 * RATE / 4096  # (two bins of the finder's spectrum)
        for frame in said:
            for i, one in enumerate(frame):
                for other in frame[i + 1 :]:
                    self.assertGreaterEqual(abs(one - other), apart, frame)
        self.assertEqual([round(f) for f in notch.frequencies_hz], [1000])

    def test_the_strongest_first_as_their_levels_change(self):
        # A (800 Hz) fades from 0.2 to 0.02 while B (1900 Hz) stays at 0.1:
        # B is then the strongest (what the tooltip lists first).
        n = int(6.0 * RATE)
        t = np.arange(n) / RATE
        fading = np.interp(t, [0.0, 2.0, 3.0, 6.0], [0.2, 0.2, 0.02, 0.02])
        x = fading * np.sin(2 * np.pi * 800.0 * t)
        x = x + 0.1 * np.sin(2 * np.pi * 1900.0 * t + 1.0)
        x = x + 0.001 * np.random.default_rng(3).standard_normal(n)
        notch = _notch()
        _out, said = _feed(notch, x)
        self.assertEqual([round(f) for f in said[int(1.5 * RATE / FRAME)]], [800, 1900])
        self.assertEqual([round(f) for f in notch.frequencies_hz], [1900, 800])
        self.assertAlmostEqual(notch.frequency_hz, 1900.0, delta=0.5)

    def test_one_ending_leaves_the_others_notched_throughout(self):
        # B (600 Hz, found first: the stronger) ends at 2 s; A (1300 Hz)
        # stays removed in every 20 ms of it while B's notch lets go.
        n = int(5.0 * RATE)
        t = np.arange(n) / RATE
        end = int(2.0 * RATE) // 80 * 80  # (whole cycles of B)
        b = np.zeros(n)
        b[:end] = 0.2 * np.sin(2 * np.pi * 600.0 * t[:end])
        x = 0.1 * np.sin(2 * np.pi * 1300.0 * t) + b
        x = x + 0.001 * np.random.default_rng(4).standard_normal(n)
        notch = _notch()
        out, _said = _feed(notch, x)
        self.assertEqual([round(f) for f in notch.frequencies_hz], [1300])
        window = int(0.02 * RATE)
        for start in range(int(1.0 * RATE), n - window, window // 2):
            level = tone_level(out[start : start + window], 1300.0)
            self.assertLess(level, 0.1 * 10 ** (-30 / 20), start / RATE)

    def test_a_notch_moved_while_it_fades_in(self):
        # Moved more than RETUNE_HZ before it has fully engaged (a whistle
        # replaced, or Manual's box changed within 20 ms), it fades out at
        # once and in at the new frequency, for any block sizes. (Rounding
        # left a sliver of the old one; the next step out was then zero
        # samples long and the block failed: no audio, the chain reset.)
        x = tone(4000, 1000.0, 0.1, RATE)
        for first, second in ((2, 100), (3, 7), (13, 7), (14, 100)):
            with self.subTest(first=first, second=second):
                notch = _notch("manual", 1000.0)
                notch.process(x[:first])
                notch.process(x[first : first + second])
                notch.manual_hz = 1500.0
                out = np.asarray(notch.process(x[:1600]), dtype=float)
                self.assertEqual(out.size, 1600)
                self.assertEqual(notch.frequencies_hz, (1500.0,))
                self.assertTrue(np.all(np.isfinite(out)))

    def test_a_notch_tuned_afresh_starts_in_step_with_the_audio(self):
        # Moved to a whistle already playing, a notch's filter first runs
        # over the last WARM_S of its audio: once it has faded in (20 ms
        # out, 20 ms in) the whistle is gone at once, not after the
        # filter's own settling (45 dB down where it is 90 dB down).
        n = int(1.0 * RATE)
        x = tone(n, 1500.0, 0.2, RATE) + tone(n, 700.0, 0.1, RATE, phase=1.0)
        notch = _notch("manual", 1000.0)
        pieces = []
        for index, start in enumerate(range(0, n, FRAME)):
            if index == 10:
                notch.manual_hz = 1500.0
                moved = start
            pieces.append(np.asarray(notch.process(x[start : start + FRAME]), float))
        out = np.concatenate(pieces)
        after = out[moved + int(0.04 * RATE) : moved + int(0.06 * RATE)]
        self.assertLess(tone_level(after, 1500.0), 0.2 * 10 ** (-70 / 20))


if __name__ == "__main__":
    unittest.main()

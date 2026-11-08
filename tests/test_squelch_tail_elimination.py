#!/usr/bin/env python3
"""
Squelch Tail Elimination: a switch for the CTCSS reverse burst (STE).

The CTCSS reverse burst (M-Q2) closes the tone squelch as a station stops
talking, before the noise of its dropped carrier (the squelch tail). But a
voice pitched near the tone, sent with little filtering, can sound like a
burst now and then: the station is then muted for about 0.8 s in the middle
of a transmission, which the app never did before (the squelch lane's check
measured 5 in 12.7 minutes of such voice high-passed at first order, 4 in
6 minutes with no high-pass). Radios let you switch it off (their menus
call it STE); so does this: ``CTCSSDetector(reverse_burst=...)`` and
``ToneSquelch(tail_elimination=...)``, both on by default and settable while
audio runs. Off, a CTCSS transmission is held to its end as before M-Q2.

The DCS turn-off code isn't covered: it is part of the DCS standard (every
DCS radio ends with it), the code itself has nothing at 134.4 Hz, and no
false turn-off was measured in 42 minutes of voice; switching it off would
only bring the DCS squelch tail back.
Plain NumPy.
"""

import unittest

import numpy as np

from tests.test_ctcss_reverse_burst import (
    LEVEL,
    RATE,
    ending,
    tone_with_burst,
    voice,
)
from tests.voice_synth import CLOSE_VOWEL

BLOCK = 1600  # 33 ms, as the window feeds it


def turning_tone(tone_hz=100.0, seconds=4.0, at=1.5, turn_s=0.1, shift_deg=180.0):
    """A tone that keeps going, its phase turned by ``shift_deg`` for
    ``turn_s`` from ``at`` and then back (what a voice pitched near the
    tone can sound like to the burst rule: the carrier stays up and the
    tone goes on)."""
    n = int(round(seconds * RATE))
    t = np.arange(n) / RATE
    up = np.clip((t - at) / 0.002, 0.0, 1.0)
    down = np.clip((t - at - turn_s) / 0.002, 0.0, 1.0)
    shift = np.deg2rad(shift_deg) * (0.5 - 0.5 * np.cos(np.pi * (up - down)))
    return LEVEL * np.cos(2 * np.pi * tone_hz * t + 0.4 + shift)


def near_pitch_voice(tone_hz, f0, seed):
    """A CTCSS tone under a voice pitched near it, high-passed at first
    order (the least a radio does), 4 s."""
    v = voice(4.0, seed=seed, f0=f0, formants=CLOSE_VOWEL, high_pass="gentle")
    t = np.arange(len(v)) / RATE
    return LEVEL * np.cos(2 * np.pi * tone_hz * t + 1.1) + v


def held_over(detector, x, tone_hz, block=BLOCK):
    """Per block: (end time s, whether the tone is held)."""
    out = []
    for pos in range(0, len(x), block):
        detector.process(x[pos : pos + block])
        out.append(((pos + block) / RATE, detector.is_present(tone_hz)))
    return out


def plays_over(squelch, x, block=BLOCK):
    """Per block: (end time s, whether the tone squelch lets it play)."""
    out = []
    for pos in range(0, len(x), block):
        piece = x[pos : pos + block]
        squelch.feed_dcs(piece)
        out.append(((pos + block) / RATE, bool(squelch.process(piece))))
    return out


def muted_after_opening(plays, until):
    """Whether the squelch, once open, closed again before ``until`` s."""
    opened = False
    for at, on in plays:
        if at > until:
            break
        if on:
            opened = True
        elif opened:
            return True
    return False


class TestTheDetector(unittest.TestCase):
    def test_on_by_default(self):
        from sdr_module.dsp.filters import CTCSSDetector

        detector = CTCSSDetector(RATE)
        self.assertTrue(detector.reverse_burst)
        plays = held_over(detector, ending(100.0), 100.0)
        let_go = next(at for at, on in plays if at > 2.0 and not on)
        self.assertLessEqual(let_go - 2.0, 0.10 + 1e-9)
        self.assertEqual(detector.bursts, 1)

    def test_off_the_tone_is_held_through_the_burst(self):
        """As before M-Q2: the burst is the tone, its phase turned; it is
        held to the end of the transmission (the carrier drops at 2.15 s;
        in the window the noise squelch then closes)."""
        from sdr_module.dsp.filters import CTCSSDetector

        detector = CTCSSDetector(RATE, reverse_burst=False)
        self.assertFalse(detector.reverse_burst)
        plays = held_over(detector, ending(100.0), 100.0)
        self.assertTrue(all(on for at, on in plays if 0.6 < at <= 2.15))
        self.assertEqual(detector.bursts, 0)
        self.assertFalse(detector.burst_heard)

    def test_a_turn_of_the_phase_mid_transmission(self):
        """On, the tone is let go at a short turn of its phase and found
        again only about 0.8 s later (the station muted that long); off, it
        is held throughout."""
        from sdr_module.dsp.filters import CTCSSDetector

        for shift in (180.0, 120.0):
            x = turning_tone(shift_deg=shift)
            detector = CTCSSDetector(RATE)
            plays = held_over(detector, x, 100.0)
            self.assertEqual(detector.bursts, 1, shift)
            gap = [at for at, on in plays if 1.5 < at and not on]
            self.assertGreater(max(gap) - min(gap), 0.7, shift)
            self.assertTrue(plays[-1][1], shift)
            detector = CTCSSDetector(RATE, reverse_burst=False)
            plays = held_over(detector, x, 100.0)
            self.assertEqual(detector.bursts, 0, shift)
            self.assertTrue(all(on for at, on in plays if at > 0.6), shift)

    def test_switched_while_it_runs(self):
        """Off for a block and on again: what it heard before isn't taken
        for the windows before the new samples (the tone's phase moved on
        meanwhile: a turn of 120 degrees between them, no burst). A burst
        after that is heard."""
        from sdr_module.dsp.filters import CTCSSDetector

        x = ending(100.0, seconds=3.0, talk=False)
        for gap in (1, 2, 3):
            detector = CTCSSDetector(RATE)
            plays = []
            for k, pos in enumerate(range(0, len(x), BLOCK)):
                detector.reverse_burst = not 30 <= k < 30 + gap  # (1.0 s on)
                detector.process(x[pos : pos + BLOCK])
                plays.append(((pos + BLOCK) / RATE, detector.is_present(100.0)))
            self.assertTrue(all(on for at, on in plays if 0.6 < at <= 3.0), gap)
            self.assertEqual(detector.bursts, 1, gap)  # (the real one)
            let_go = next(at for at, on in plays if at > 3.0 and not on)
            self.assertLessEqual(let_go - 3.0, 0.10 + 1e-9, gap)

    def test_off_a_burst_isnt_looked_for_at_all(self):
        """Off costs what the detector cost before M-Q2: the CTCSS band's
        filter isn't run."""
        from sdr_module.dsp.filters import CTCSSDetector

        detector = CTCSSDetector(RATE, reverse_burst=False)
        calls = []
        detector._burst_in = lambda low: calls.append(low.size)
        for _ in range(60):  # (2 s)
            detector.process(np.zeros(BLOCK))
        self.assertEqual(calls, [])


class TestTheToneSquelch(unittest.TestCase):
    def squelch(self, **kwargs):
        from sdr_module.gui.audio_chain import ToneSquelch

        squelch = ToneSquelch(RATE, **kwargs)
        squelch.tone_hz = 100.0
        return squelch

    def test_on_by_default(self):
        squelch = self.squelch()
        self.assertTrue(squelch.tail_elimination)
        plays = plays_over(squelch, ending(100.0))
        shut = next(at for at, on in plays if at > 2.0 and not on)
        self.assertLessEqual(shut - 2.0, 0.10 + 1e-9)

    def test_off_it_plays_until_the_carrier_drops(self):
        for squelch in (self.squelch(tail_elimination=False), self.squelch()):
            squelch.tail_elimination = False  # (also while it runs)
            self.assertFalse(squelch.tail_elimination)
            plays = plays_over(squelch, ending(100.0))
            self.assertTrue(all(on for at, on in plays if 0.6 < at <= 2.15))

    def test_a_voice_that_sounds_like_a_burst(self):
        """Voices pitched near the tone (180 Hz on 177.3 Hz, 240 Hz on
        225.7 Hz), high-passed at first order: the burst rule takes a
        moment of each for a burst and mutes the station for about 0.8 s
        mid-transmission; with tail elimination off they play throughout."""
        for tone, f0, seed in ((177.3, 180.0, 25), (225.7, 240.0, 30)):
            x = near_pitch_voice(tone, f0, seed)
            squelch = self.squelch()
            squelch.tone_hz = tone
            plays = plays_over(squelch, x)
            # (The case this switch is for: muted mid-transmission.)
            self.assertTrue(muted_after_opening(plays, 4.0), tone)
            squelch = self.squelch(tail_elimination=False)
            squelch.tone_hz = tone
            plays = plays_over(squelch, x)
            self.assertFalse(muted_after_opening(plays, 4.0), tone)

    def test_reversed_and_off_it_stays_muted_through_the_burst(self):
        """The station the reverse tone squelch mutes: with tail
        elimination off it stays muted to the end of its burst (as on)."""
        squelch = self.squelch(tail_elimination=False)
        squelch.reverse = True
        plays = plays_over(squelch, ending(100.0))
        self.assertNotIn(True, [on for at, on in plays if at <= 2.15])
        # Another station (no tone) is heard after about half a second.
        squelch = self.squelch(tail_elimination=False)
        squelch.reverse = True
        plays = plays_over(squelch, voice(2.0))
        self.assertLessEqual(next(at for at, on in plays if on), 0.6 + 1e-9)

    def test_the_dcs_turn_off_code_still_closes_it(self):
        """Not covered by the switch: a DCS code's turn-off code closes the
        squelch within 100 ms either way."""
        from sdr_module.dsp.dcs import dcs_transmission

        for on in (True, False):
            squelch = self.squelch(tail_elimination=on)
            squelch.tone_hz = None
            squelch.dcs = (23, False)
            sent = 0.15 * dcs_transmission(23, rate=RATE, seconds=1.5)
            gone = 0.6 * np.random.default_rng(4).standard_normal(int(0.5 * RATE))
            plays = plays_over(squelch, np.concatenate((sent, gone)))
            self.assertTrue(any(on_ for at, on_ in plays if at < 1.5))
            shut = next(at for at, on_ in plays if at > 1.0 and not on_)
            self.assertLessEqual(shut - 1.5, 0.10 + BLOCK / RATE, on)


class TestTheBurstItselfIsUnchanged(unittest.TestCase):
    """On (the default) the burst is heard exactly as M-Q2 hears it: the
    switch adds nothing to the rule."""

    def test_every_shift_as_before(self):
        from sdr_module.dsp.filters import CTCSSDetector

        for shift in (180.0, 120.0, 135.0, 240.0):
            x = tone_with_burst(127.3, 2.0, shift, 0.15)
            x = np.concatenate(
                (x, 0.6 * np.random.default_rng(2).standard_normal(24000))
            )
            a, b = CTCSSDetector(RATE), CTCSSDetector(RATE, reverse_burst=True)
            self.assertEqual(held_over(a, x, 127.3), held_over(b, x, 127.3), shift)
            self.assertEqual((a.bursts, b.bursts), (1, 1), shift)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
The squelch lane's check, without a window (M-Q1, M-Q2): the guards of the
DCS turn-off code and of the CTCSS reverse burst that the lane's own tests
didn't pin (each could be taken out with every test still passing).

* DCS: the turn-off code must hold most of the window's power (a weak
  signal's noise, which leaves little of the code's own power at times, is
  no turn-off code), and be loud enough to be a tone (a code that stops
  without one, its carrier still up and quiet, is no turn-off code either).
* CTCSS: a reverse burst keeps the tone's level (another station taking
  the channel over on the same tone, much weaker or stronger, is none).
Plain NumPy.
"""

import unittest

import numpy as np

RATE = 48000.0
LEVEL = 0.15  # DCS at 15 % of the deviation


def _turn_offs(x, block=1600):
    from sdr_module.dsp.dcs import DcsDetector

    detector = DcsDetector(RATE)
    for pos in range(0, len(x), block):
        detector.process(x[pos : pos + block])
    return detector


class TestDcsTurnOffGuards(unittest.TestCase):
    def test_a_weak_signal_is_no_turn_off_code(self):
        """4 dB above the noise in the DCS band, the code still read: its
        noise leaves windows with little of the code's power and a little
        134.4 Hz, but never mostly 134.4 Hz (without that rule, these five
        stretches turned the squelch off 7 times)."""
        from sdr_module.dsp.dcs import dcs_waveform

        sigma = LEVEL * 10 ** (-4.0 / 20) * np.sqrt(RATE / 2 / 300.0)
        for seed in (1, 6, 17, 21, 29):
            code = (23, 365, 754)[seed % 3]
            rng = np.random.default_rng(1000 + seed)
            x = LEVEL * dcs_waveform(code, rate=RATE, seconds=10.0)
            x = x + sigma * rng.standard_normal(len(x))
            detector = _turn_offs(x)
            self.assertEqual(detector.turn_offs, 0, seed)

    def test_a_code_that_stops_without_one_turns_nothing_off(self):
        """A transmitter without the turn-off code: the code stops, the
        carrier stays up a moment, quiet. Its code is let go as before (after
        HOLD_S), and no turn-off code is reported (without the level rule, 3
        of these 40 were)."""
        from sdr_module.dsp.dcs import dcs_waveform

        for seed in range(40):
            rng = np.random.default_rng(seed)
            code = (23, 116, 172, 365, 445, 631, 754)[seed % 7]
            x = np.concatenate(
                (LEVEL * dcs_waveform(code, rate=RATE, seconds=1.0), np.zeros(28800))
            )
            x = x + 0.002 * rng.standard_normal(len(x))
            detector = _turn_offs(x)
            self.assertEqual(detector.turn_offs, 0, seed)
            self.assertFalse(detector.turned_off, seed)
            self.assertIsNone(detector.detected, seed)


class TestCtcssBurstGuards(unittest.TestCase):
    def test_a_station_taking_over_at_another_level_is_no_burst(self):
        """A burst is the same transmitter's tone shifted, at its level. A
        much weaker or stronger station on the same tone taking the channel
        over with no gap (its tone at another phase) is no burst: it would
        be muted for most of a second (without the level rule, each of
        these is taken for one)."""
        from sdr_module.dsp.filters import CTCSSDetector

        level = 0.12
        t = np.arange(int(1.5 * RATE)) / RATE
        for ratio in (0.3, 3.0):
            for turn in (0.6, 0.8, 0.95):
                first = level * np.cos(2 * np.pi * 100.0 * t)
                then = (
                    ratio * level * np.cos(2 * np.pi * 100.0 * (t + 1.5) + np.pi * turn)
                )
                x = np.concatenate((first, then))
                detector = CTCSSDetector(RATE)
                for pos in range(0, len(x), 1600):
                    detector.process(x[pos : pos + 1600])
                self.assertEqual(detector.bursts, 0, (ratio, turn))
                self.assertTrue(detector.is_present(100.0), (ratio, turn))


if __name__ == "__main__":
    unittest.main()

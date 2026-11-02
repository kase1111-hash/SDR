#!/usr/bin/env python3
"""
The low cut (a high-pass for voice audio): what removes hum and rumble
(Low cut 100, 200 or 300 Hz, K-A2) and, with a tone squelch set, the
CTCSS tone or the DCS bits under narrow FM voice (K-A3).

``sdr_module.dsp.audio_filters``: the inverse-Chebyshev design and the
streaming :class:`LowCut`.
"""

import math
import unittest

import numpy as np

from tests.voice_synth import band_power, pitched_voice, tone, tone_level

RATE = 48000.0
FRAME = 1600


def _gain_db(sos, freq):
    from sdr_module.dsp.audio_filters import sos_response

    return 20 * math.log10(abs(sos_response(sos, [freq], RATE)[0]) + 1e-30)


class TestHighpassDesign(unittest.TestCase):
    def test_minus_3_db_at_the_corner_and_40_db_below_its_stopband(self):
        from sdr_module.dsp.audio_filters import highpass_sos

        for corner in (100.0, 200.0, 300.0):
            sos = highpass_sos(corner, RATE)
            self.assertAlmostEqual(_gain_db(sos, corner), -3.0, delta=0.3)
            for f in np.linspace(5.0, 0.81 * corner, 40):
                self.assertLess(_gain_db(sos, f), -38.0, (corner, f))
            for f in (1.3 * corner, 2 * corner, 1000.0, 3000.0, 12000.0):
                self.assertGreater(_gain_db(sos, f), -0.1, (corner, f))

    def test_300_hz_keeps_every_ctcss_tone_out(self):
        from sdr_module.dsp.audio_filters import highpass_sos
        from sdr_module.dsp.filters import CTCSS_FREQUENCIES

        sos = highpass_sos(300.0, RATE)
        for f in CTCSS_FREQUENCIES:
            limit = -38.0 if f < 245.0 else -25.0  # (250.3 and 254.1 Hz)
            self.assertLess(_gain_db(sos, f), limit, f)


class TestLowCut(unittest.TestCase):
    def cut(self):
        from sdr_module.dsp.audio_filters import LowCut

        return LowCut(RATE)

    def run_cut(self, cut, x, corner):
        return np.concatenate(
            [
                np.asarray(cut.process(x[p : p + FRAME], corner), float)
                for p in range(0, len(x), FRAME)
            ]
        )

    def test_off_leaves_the_audio_alone(self):
        x = pitched_voice(1.0, RATE, seed=2)
        out = self.run_cut(self.cut(), x, None)
        self.assertTrue(np.array_equal(out, x.astype(np.float32)))

    def test_hum_goes_and_voice_stays(self):
        voice = pitched_voice(3.0, RATE, seed=2)
        n = len(voice)
        for corner, hums in ((100.0, (50.0, 60.0)), (200.0, (100.0, 120.0, 150.0))):
            hum = sum(tone(n, f, 0.1, phase=0.0) for f in hums)
            out = self.run_cut(self.cut(), voice + hum, corner)
            settled = slice(int(0.5 * RATE), None)
            for f in hums:
                self.assertLess(tone_level(out[settled], f), 0.1 * 10 ** (-38 / 20), f)
            low = 1.6 * corner
            ratio = band_power(out[settled], RATE, low, 4000) / band_power(
                voice[settled], RATE, low, 4000
            )
            self.assertAlmostEqual(10 * math.log10(ratio), 0.0, delta=0.15)

    def test_300_hz_takes_the_dcs_bits_out_of_voice(self):
        from sdr_module.dsp.dcs import dcs_waveform

        dcs = 0.15 * dcs_waveform(23, rate=RATE, seconds=3.0)
        out = self.run_cut(self.cut(), dcs, 300.0)[int(0.5 * RATE) :]
        before = band_power(dcs[int(0.5 * RATE) :], RATE, 0, 24000)
        after = band_power(out, RATE, 0, 24000)
        self.assertLess(10 * math.log10(after / before), -30.0)

    def test_switching_never_clicks(self):
        """On, another corner, off: crossfaded, no step in the audio."""
        n = int(2.0 * RATE)
        x = tone(n, 100.0, 0.5) + tone(n, 1000.0, 0.2)
        cut = self.cut()
        pieces = []
        for i, pos in enumerate(range(0, n, FRAME)):
            corner = None if i < 10 or i >= 45 else (300.0 if i < 30 else 100.0)
            pieces.append(np.asarray(cut.process(x[pos : pos + FRAME], corner), float))
        out = np.concatenate(pieces)
        # The tones move at most 0.026 + 0.007 a sample; a step is far more.
        self.assertLess(np.max(np.abs(np.diff(out))), 0.045)
        self.assertIsNone(cut.corner_hz)


if __name__ == "__main__":
    unittest.main()

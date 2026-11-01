"""Final finish: the last refinements of the polish round's fixes (N-01 to
N-08 of the final pass).

Each test names the problem it covers; most run the real window offscreen
on the demo device (or the part on its own where that is enough).
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import require_pyqt6  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signal_synth as S  # noqa: E402

_DEVIATIONS_HZ = (2.5e3, 5e3, 12.5e3, 25e3, 75e3)


def _hz(text):
    """ "12.5 kHz" -> 12500.0"""
    value, unit = text.split()
    return float(value) * {"Hz": 1.0, "kHz": 1e3}[unit]


class TestIdentifyHearsVoiceAtItsDeviation(unittest.TestCase):
    """N-01: Identify held the measured peak with 10 % to spare, so a 2 m
    voice station whose radio limits it at about +/-5 kHz (measured peak
    4.6-5 kHz) was suggested 12.5 kHz, and Apply Mode made it about 8 dB
    quieter. What the window's clipping warning measures is the share of
    the audio beyond the deviation (it warns above 5 %): voice reaches its
    peak only now and then, a tone stays near it. Here the window's own
    receiver chain measures that share at the deviation Identify suggests
    (it stays under 5 %) and at the next smaller choice (over 5 %: the
    suggestion is no wider than it needs to be)."""

    RATE = 240e3
    SECONDS = 1.5

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import main_window

        self.mw = main_window
        self.n = int(self.SECONDS * self.RATE)

    def clip_fraction(self, x, deviation_hz, width_hz):
        """What the clipping warning measures: the share of audio samples
        beyond full scale (30 blocks a second, as the display loop hands
        them over)."""
        chain = self.mw._ReceiverChain(self.RATE, "FM", width_hz)
        over = total = 0.0
        block = int(self.RATE / 30)
        for start in range(0, len(x) - block + 1, block):
            audio = chain.demodulate(chain.channelize(x[start : start + block]), deviation_hz)
            over += chain.last_clip_fraction * len(audio)
            total += len(audio)
        return over / max(total, 1.0)

    def check(self, signal, occupied_hz, width_hz, snr_db):
        from sdr_module.dsp.signal_id import identify

        x, _ = S.add_noise(signal, self.RATE, snr_db, occupied_hz, seed=3)
        x = x.astype(np.complex64)
        ident = identify(
            x[: int(self.RATE)], self.RATE, 146.52e6, search_hz=0.75 * width_hz
        )
        self.assertEqual(ident.mode, "FM", ident.summary())
        deviation = _hz(ident.fm_deviation)
        limit = self.mw._CLIP_FRACTION
        clipped = self.clip_fraction(x, deviation, width_hz)
        self.assertLessEqual(clipped, limit, (ident.fm_deviation, clipped))
        smaller = [d for d in _DEVIATIONS_HZ if d < deviation]
        if smaller:
            self.assertGreater(
                self.clip_fraction(x, smaller[-1], width_hz),
                limit,
                f"{ident.fm_deviation} is wider than it needs to be",
            )
        return ident

    def limited_voice(self, seed):
        """Voice through a radio's limiter: driven well past it, clipped,
        then the filter after the limiter (which overshoots a little)."""
        audio = np.clip(4.0 * S.voice(self.n, self.RATE, seed=seed), -1.0, 1.0)
        audio = S._lowpass(audio, 3000.0, self.RATE)
        return audio / np.max(np.abs(audio))

    def test_a_2_m_station_limited_at_5_khz(self):
        # (Its overshoot reaches +/-5.4 kHz; the limiter holds about 4.8.)
        signal = S.fm(self.n, self.RATE, 5.4e3, self.limited_voice(3))
        ident = self.check(signal, 16.8e3, 25e3, snr_db=30.0)
        self.assertEqual(ident.fm_deviation, "5 kHz")
        peak = ident.result.features["freq_peak_hz"]
        self.assertGreater(1.1 * peak, 5e3)  # (the old margin's 12.5 kHz)
        self.assertEqual(ident.filter_hz, 25e3)
        self.assertIn(f"The swing peaks at about ±{peak / 1e3:.1f} kHz", ident.details)

    def test_a_tone_near_a_choice_keeps_the_margin(self):
        """A 2.5 kHz test tone at +/-5.1 kHz reads a little low (under
        5 kHz), stays near its peak and clips about a tenth of the time at
        5 kHz: the warning would go off, so it is suggested 12.5 kHz."""
        tone = S.tones(self.n, self.RATE, freqs=(2500.0,), amps=(1.0,))
        ident = self.check(S.fm(self.n, self.RATE, 5.1e3, tone), 15.2e3, 25e3, 35.0)
        self.assertEqual(ident.fm_deviation, "12.5 kHz")
        self.assertLess(ident.result.features["freq_peak_hz"], 5e3)
        self.assertIn("and stays near it: 5 kHz could clip", ident.details[-1])


if __name__ == "__main__":
    unittest.main()

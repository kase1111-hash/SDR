#!/usr/bin/env python3
"""
How far a weak broadcast FM station blends toward mono (M-A2): Radio >
Audio > Stereo Blend, Less, Normal (today's: full stereo while its hiss
stays 50 dB below full scale, mono below 30 dB) or More, each moving the
blend's thresholds by 6 dB; read by the receiver chain from its
AudioOptions (``stereo_blend``) on every block, a change taking effect
over 20 ms.
"""

import math
import unittest
from unittest import mock

import numpy as np

from sdr_module.dsp.fm_stereo import BLEND_OFFSETS_DB, StereoDecoder

RATE = 240e3


def _multiplex(left, right, rate=RATE, pilot=0.09, phase=0.4):
    phi = 2 * np.pi * 19e3 * np.arange(len(left)) / rate + phase
    return 0.9 * (
        (left + right) / 2 + (left - right) / 2 * np.sin(2 * phi)
    ) + pilot * np.sin(phi)


def _weak_station(seconds=0.6, density=2e-10, seed=4):
    """A stereo multiplex with white noise of ``density`` per Hz: full
    stereo's hiss would be about 47 dB below full scale (between the Less
    and Normal thresholds)."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    left = 0.4 * np.sin(2 * np.pi * 1000 * t)
    noise = rng.standard_normal(n) * math.sqrt(density * RATE)
    return _multiplex(left, 0 * left, RATE) + noise


def _quick_rise():
    return mock.patch.object(StereoDecoder, "RISE_S", 0.05)


class TestTheChoices(unittest.TestCase):
    def test_less_normal_and_more(self):
        self.assertEqual(BLEND_OFFSETS_DB, {"less": -6.0, "normal": 0.0, "more": 6.0})
        decoder = StereoDecoder(RATE)
        self.assertEqual(decoder.blending, "normal")  # (as before)
        self.assertEqual((decoder.full_snr_db, decoder.mono_snr_db), (50.0, 30.0))
        for name, (full, mono) in (
            ("less", (44.0, 24.0)),
            ("more", (56.0, 36.0)),
            ("normal", (50.0, 30.0)),
        ):
            decoder.blending = name
            self.assertEqual(decoder.blending, name)
            self.assertEqual((decoder.full_snr_db, decoder.mono_snr_db), (full, mono))
        self.assertEqual(StereoDecoder(RATE, blending="more").blending, "more")

    def test_anything_else_is_refused(self):
        decoder = StereoDecoder(RATE)
        for bad in ("loud", "", None, 3):
            with self.assertRaises(ValueError):
                decoder.blending = bad
            with self.assertRaises(ValueError):
                StereoDecoder(RATE, blending=bad)
        self.assertEqual(decoder.blending, "normal")

    def test_the_gui_choices_and_what_a_damaged_value_gives(self):
        from sdr_module.gui.audio_chain import AudioOptions
        from sdr_module.gui.stereo_audio import (
            DEFAULT_STEREO_BLEND,
            STEREO_BLEND_CHOICES,
            usable_blend,
        )

        self.assertEqual(
            [key for key, _label, _tip in STEREO_BLEND_CHOICES],
            ["less", "normal", "more"],
        )
        for _key, label, tip in STEREO_BLEND_CHOICES:
            self.assertTrue(label)
            self.assertTrue(tip)
        self.assertEqual(DEFAULT_STEREO_BLEND, "normal")
        self.assertEqual(AudioOptions().stereo_blend, "normal")
        for good in ("less", " More ", "NORMAL"):
            self.assertEqual(usable_blend(good), good.strip().lower())
        for bad in ("loud", "", None, 6, float("nan")):
            self.assertEqual(usable_blend(bad), "normal")


class TestTheBlend(unittest.TestCase):
    """The same weak station with each choice (the decoder on its own)."""

    def decode(self, blending, mpx=None):
        with _quick_rise():
            decoder = StereoDecoder(RATE, phase_step=False, blending=blending)
            decoder.process(_weak_station() if mpx is None else mpx)
        return decoder

    def test_less_keeps_more_stereo_and_more_blends_sooner(self):
        less, normal, more = (self.decode(name) for name in ("less", "normal", "more"))
        # (The same noise: the same estimate of full stereo's hiss.)
        self.assertAlmostEqual(less.snr_db, normal.snr_db, places=6)
        self.assertGreater(normal.snr_db, 44.5)  # (between the thresholds)
        self.assertLess(normal.snr_db, 49.5)
        self.assertEqual(less.blend, 1.0)
        self.assertGreater(normal.blend, 0.5)
        self.assertLess(normal.blend, 0.9)
        # (6 dB more blending: about half as much of the difference.)
        self.assertGreater(more.blend / normal.blend, 0.4)
        self.assertLess(more.blend / normal.blend, 0.65)
        # Each keeps the hiss it lets through at its own level: full
        # stereo's hiss times the blend squared.
        for decoder in (less, normal, more):
            hiss_db = -decoder.snr_db + 20 * math.log10(max(decoder.blend, 1e-9))
            self.assertLessEqual(hiss_db, -decoder.full_snr_db + 0.01)

    def test_a_clean_station_is_full_stereo_with_every_choice(self):
        rng = np.random.default_rng(1)
        n = int(0.5 * RATE)
        t = np.arange(n) / RATE
        left = 0.4 * np.sin(2 * np.pi * 1000 * t)
        clean = _multiplex(left, 0 * left) + 1e-6 * rng.standard_normal(n)
        for name in ("less", "normal", "more"):
            self.assertEqual(self.decode(name, clean).blend, 1.0, name)

    def test_changed_while_it_runs_it_moves_over_20_ms(self):
        mpx = _weak_station(seconds=0.9)
        with _quick_rise():
            switched = StereoDecoder(RATE, phase_step=False)
            switched.process(mpx[: int(0.5 * RATE)])
            before = switched.blend
            switched.blending = "more"
            switched.process(mpx[int(0.5 * RATE) : int(0.505 * RATE)])
            halfway = switched.blend  # (5 ms on: on its way down)
            switched.process(mpx[int(0.505 * RATE) :])
            more = StereoDecoder(RATE, phase_step=False, blending="more")
            more.process(mpx)
        self.assertLess(halfway, before)
        self.assertGreater(halfway, more.blend)
        self.assertAlmostEqual(switched.blend, more.blend, delta=1e-6)

    def test_a_reset_starts_with_the_choice(self):
        decoder = StereoDecoder(RATE, phase_step=False)
        decoder.blending = "less"
        decoder.reset()
        self.assertEqual(decoder.blending, "less")
        self.assertEqual(decoder.full_snr_db, 44.0)


class TestTheChain(unittest.TestCase):
    """The receiver chain gives its stereo decoder the options' choice."""

    def setUp(self):
        try:
            import PyQt6  # noqa: F401  (main_window imports it)
        except ImportError:  # pragma: no cover - environment-dependent
            self.skipTest("PyQt6 not available")

    def test_the_options_choice_reaches_the_decoder(self):
        from sdr_module.gui.audio_chain import AudioOptions
        from sdr_module.gui.main_window import _ReceiverChain

        rate = 2.4e6
        n = int(0.3 * rate)
        t = np.arange(n) / rate
        left = 0.4 * np.sin(2 * np.pi * 1000 * t)
        mpx = _multiplex(left, 0 * left, rate)
        iq = np.exp(1j * 2 * np.pi * 75e3 * np.cumsum(mpx) / rate).astype(np.complex64)
        chain = _ReceiverChain(rate, "FM", 200e3)
        chain.options = AudioOptions()
        chain.stereo = True
        for value, expected in (("more", "more"), ("less", "less"), ("loud", "normal")):
            chain.options.stereo_blend = value
            for start in range(0, n, 80000):
                chain.demodulate(chain.channelize(iq[start : start + 80000]), 75e3)
            self.assertEqual(chain.stereo_decoder.blending, expected, value)


if __name__ == "__main__":
    unittest.main()

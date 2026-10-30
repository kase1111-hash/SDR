#!/usr/bin/env python3
"""
The receiver chain after the channel filter: filter widths for every mode,
clean FM and SSB audio, the CW pitch, the audio AGC speed and the helpers
in ``gui/audio_chain.py``.

Regression tests for RX-03 (FM DC), RX-11 / UX-B10 / G-09 (SSB and CW
widths), RX-12 (carrier leaking into SSB audio) and HAM-07 (the S-meter
measured a wide channel in narrow modes).
"""

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, require_pyqt6  # noqa: E402

RATE = 2.4e6
AUDIO_RATE = 48000.0


def _mw():
    from sdr_module.gui import main_window

    return main_window


def _fm(audio, rate, deviation, offset_hz=0.0):
    """Complex FM of ``audio`` (peak 1 = ``deviation``) at ``offset_hz``."""
    phase = 2 * np.pi * np.cumsum(deviation * audio + offset_hz) / rate
    return np.exp(1j * phase).astype(np.complex64)


def _receive(chain, iq, deviation=5e3, block=80000):
    parts = []
    for i in range(0, len(iq), block):
        parts.append(chain.demodulate(chain.channelize(iq[i : i + block]), deviation))
    return np.concatenate(parts)


def _tone_level(audio, freq, rate=AUDIO_RATE):
    """Amplitude of the ``freq`` component of ``audio``."""
    t = np.arange(len(audio)) / rate
    ref = np.exp(-2j * np.pi * freq * t)
    return 2 * abs(np.mean(audio * ref))


def _audio_band(chain, seconds=2.0, seed=1):
    """(low, high) audio frequencies where white noise through ``chain``
    comes out within 6 dB of its strongest."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    noise *= 0.01
    audio = _receive(chain, noise)[4800:]
    spectrum = np.abs(np.fft.rfft(audio.reshape(-1, 4800) * np.hanning(4800), axis=1))
    power = np.mean(spectrum**2, axis=0)
    freqs = np.fft.rfftfreq(4800, 1 / AUDIO_RATE)
    smooth = np.convolve(power, np.ones(5) / 5, mode="same")
    inside = freqs[smooth >= smooth.max() / 4]
    return float(inside.min()), float(inside.max())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestFmAudioIsCentred(unittest.TestCase):
    """RX-03: a tuning error must not become a DC offset that clips."""

    def setUp(self):
        require_pyqt6(self)
        self.mw = _mw()

    def test_mistuned_narrowband_fm_has_no_dc_and_does_not_clip(self):
        t = np.arange(int(1.0 * RATE)) / RATE
        tone = np.sin(2 * np.pi * 1e3 * t)
        for offset in (0.0, 2e3, 3e3):
            chain = self.mw._ReceiverChain(RATE, "FM", 25e3)
            iq = _fm(tone, RATE, 3e3, offset)
            audio = _receive(chain, iq, 5e3)[int(0.25 * AUDIO_RATE) :]
            self.assertLess(abs(float(np.mean(audio))), 0.02, offset)
            clipped = float(np.mean(np.abs(audio) >= 0.999))
            self.assertLess(clipped, 0.005, offset)  # was 2.1 % / 27 %
            # The tone itself is untouched.
            self.assertAlmostEqual(_tone_level(audio, 1e3), 0.6, delta=0.05)

    def test_broadcast_fm_keeps_its_bass(self):
        t = np.arange(int(1.0 * RATE)) / RATE
        chain = self.mw._ReceiverChain(RATE, "FM", 200e3)
        audio = _receive(chain, _fm(0.5 * np.sin(2 * np.pi * 60 * t), RATE, 75e3), 75e3)
        level = _tone_level(audio[int(0.5 * AUDIO_RATE) :], 60)
        self.assertGreater(level, 0.5 * 0.85)  # within ~1.4 dB at 60 Hz


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestSidebandWidths(unittest.TestCase):
    """RX-11, UX-B10, G-09: the Bandwidth control sets the SSB/CW filter."""

    def setUp(self):
        require_pyqt6(self)
        self.mw = _mw()

    def test_band_edges_follow_the_width(self):
        cases = {
            ("USB", 1.8e3): (200.0, 2000.0),
            ("USB", 2.7e3): (200.0, 2900.0),
            ("USB", 3.6e3): (200.0, 3800.0),
            ("LSB", 2.4e3): (-2600.0, -200.0),
            ("CW", 500.0): (-250.0, 250.0),
            ("CW", 100.0): (-50.0, 50.0),
            ("CW", 1000.0): (-500.0, 500.0),
        }
        for (mode, width), edges in cases.items():
            chain = self.mw._ReceiverChain(RATE, mode, width)
            self.assertEqual(chain.band_edges(), edges, (mode, width))
            self.assertEqual(self.mw._channel_edges(mode, width, RATE), edges)

    def test_audio_band_follows_the_ssb_width(self):
        narrow = _audio_band(self.mw._ReceiverChain(RATE, "USB", 1.8e3))
        wide = _audio_band(self.mw._ReceiverChain(RATE, "USB", 3.0e3))
        self.assertAlmostEqual(narrow[1], 2000, delta=150)
        self.assertAlmostEqual(wide[1], 3200, delta=150)
        self.assertGreater(narrow[0], 120)
        self.assertGreater(wide[0], 120)

    def test_cw_filter_width_and_pitch(self):
        chain = self.mw._ReceiverChain(RATE, "CW", 250.0)
        low, high = _audio_band(chain)
        self.assertAlmostEqual(low, 700 - 125, delta=60)
        self.assertAlmostEqual(high, 700 + 125, delta=60)
        from sdr_module.gui.audio_chain import AudioOptions

        options = AudioOptions()
        options.cw_pitch_hz = 450.0
        chain = self.mw._ReceiverChain(RATE, "CW", 500.0)
        chain.options = options
        carrier = np.full(int(0.5 * RATE), 0.01, dtype=np.complex64)
        audio = _receive(chain, carrier)[4800:]
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio))))
        peak = np.fft.rfftfreq(len(audio), 1 / AUDIO_RATE)[np.argmax(spectrum)]
        self.assertAlmostEqual(peak, 450, delta=10)

    def test_old_wide_widths_still_build_a_chain(self):
        # Callers that still pass a channel width (10 kHz) get a valid,
        # wide filter instead of an error.
        chain = self.mw._ReceiverChain(RATE, "USB", 10e3)
        low, high = chain.band_edges()
        self.assertEqual(low, 200.0)
        self.assertLessEqual(high, 10.2e3)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestNoCarrierInSsbAudio(unittest.TestCase):
    """RX-12: a carrier or DC spike at the tuned frequency must not reach
    the SSB audio."""

    def setUp(self):
        require_pyqt6(self)
        self.mw = _mw()

    def test_carrier_at_zero_hz_is_far_below_the_passband(self):
        n = int(0.5 * RATE)
        t = np.arange(n) / RATE
        for mode, sign in (("USB", 1), ("LSB", -1)):
            chain = self.mw._ReceiverChain(RATE, mode, 2.7e3)
            dc = chain.channelize(np.full(n, 0.1, dtype=np.complex64))[2000:]
            chain = self.mw._ReceiverChain(RATE, mode, 2.7e3)
            tone = 0.1 * np.exp(sign * 2j * np.pi * 1500 * t).astype(np.complex64)
            inband = chain.channelize(tone)[2000:]
            ratio = np.mean(np.abs(dc) ** 2) / np.mean(np.abs(inband) ** 2)
            self.assertLess(10 * np.log10(ratio), -40, mode)  # was -13.4 dB

    def test_usb_on_an_am_carrier_has_no_dc(self):
        n = int(1.0 * RATE)
        t = np.arange(n) / RATE
        am = (0.02 * (1 + 0.5 * np.sin(2 * np.pi * 1000 * t))).astype(np.complex64)
        for mode in ("USB", "LSB"):
            chain = self.mw._ReceiverChain(RATE, mode, 2.7e3)
            audio = _receive(chain, am)[int(0.3 * AUDIO_RATE) :]
            self.assertLess(abs(float(np.mean(audio))), 0.01, mode)  # was 0.21
            self.assertGreater(_tone_level(audio, 1000), 0.2, mode)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestNarrowChannelIsThePassband(unittest.TestCase):
    """HAM-07: in USB, LSB and CW the channel (what the S-meter, LEVEL and
    the squelch measure) is the demodulated passband, not +/-12 kHz."""

    def setUp(self):
        require_pyqt6(self)
        self.mw = _mw()

    def _power_db(self, mode, width, offset):
        n = int(0.2 * RATE)
        t = np.arange(n) / RATE
        tone = (0.01 * np.exp(2j * np.pi * offset * t)).astype(np.complex64)
        chain = self.mw._ReceiverChain(RATE, mode, width)
        channel = chain.channelize(tone)[3000:]
        return 10 * np.log10(np.mean(np.abs(channel) ** 2) + 1e-30)

    def test_signal_outside_the_passband_does_not_count(self):
        for mode, width, inside, outside in (
            ("USB", 2.7e3, 1500.0, 8000.0),
            ("USB", 2.7e3, 1500.0, -1500.0),  # opposite sideband
            ("LSB", 2.4e3, -1500.0, 1500.0),
            ("CW", 500.0, 100.0, 6000.0),
            ("CW", 250.0, 0.0, 400.0),
        ):
            inside_db = self._power_db(mode, width, inside)
            outside_db = self._power_db(mode, width, outside)
            self.assertAlmostEqual(inside_db, -40.0, delta=0.5)
            self.assertLess(outside_db, inside_db - 40, (mode, width, outside))

    def test_noise_bandwidth_matches_the_filter(self):
        for mode, width in (("USB", 2.7e3), ("CW", 500.0), ("FM", 25e3)):
            chain = self.mw._ReceiverChain(RATE, mode, width)
            enbw = chain.noise_bandwidth()
            # (Filters are their width wide at -6 dB, or flat across it;
            # the transition counts for less in noise power.)
            self.assertGreater(enbw, 0.8 * width, mode)
            self.assertLess(enbw, 1.3 * width, mode)
            # And it is what white noise measures through the channel.
            rng = np.random.default_rng(5)
            n = int(0.5 * RATE)
            noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
            channel = chain.channelize(noise.astype(np.complex64))[4000:]
            measured = np.mean(np.abs(channel) ** 2) * RATE
            self.assertAlmostEqual(
                10 * np.log10(measured / enbw), 0.0, delta=0.5, msg=mode
            )


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestAudioAgcSpeed(unittest.TestCase):
    """FG-G14: the audio AGC speed of AM, SSB and CW."""

    def setUp(self):
        require_pyqt6(self)
        self.mw = _mw()

    def _after_burst(self, speed, delay_s=0.5):
        """Audio level ``delay_s`` after a loud burst ends, with the quiet
        signal 40 dB below it (0.5 = fully recovered)."""
        from sdr_module.gui.audio_chain import AudioAgc

        agc = AudioAgc(AUDIO_RATE, speed)
        t = np.arange(int(2.0 * AUDIO_RATE)) / AUDIO_RATE
        amplitude = np.where(t < 0.5, 0.5, 0.005)
        out = agc.process((amplitude * np.sin(2 * np.pi * 1000 * t)).astype(np.float32))
        start = int((0.5 + delay_s) * AUDIO_RATE)
        return float(np.max(np.abs(out[start : start + 480])))

    def test_speeds_set_the_release_time(self):
        # The level falls as exp(-t / release) until it meets the quiet
        # signal: fast (0.1 s) has recovered after 0.5 s, medium (0.5 s)
        # and slow (2 s) still hold the gain down by 31 and 38 dB.
        self.assertGreater(self._after_burst("fast"), 0.4)
        self.assertAlmostEqual(self._after_burst("medium"), 0.0136, delta=0.003)
        self.assertAlmostEqual(self._after_burst("slow"), 0.0064, delta=0.0015)

    def test_chain_uses_the_options_speed(self):
        from sdr_module.gui.audio_chain import AudioOptions

        levels = {}
        for speed in ("fast", "slow"):
            options = AudioOptions()
            options.agc_speed = speed
            chain = self.mw._ReceiverChain(RATE, "USB", 2.7e3)
            chain.options = options
            t = np.arange(int(1.0 * RATE)) / RATE
            tone = np.exp(2j * np.pi * 1000 * t).astype(np.complex64)
            loud = np.where(t < 0.5, 0.3, 0.003).astype(np.float32)
            audio = _receive(chain, tone * loud)
            levels[speed] = float(np.max(np.abs(audio[int(0.95 * AUDIO_RATE) :])))
        self.assertGreater(levels["fast"], 0.4)
        self.assertLess(levels["slow"], 0.1)

    def test_agc_off_keeps_a_fixed_gain(self):
        from sdr_module.gui.audio_chain import AudioOptions

        options = AudioOptions()
        options.agc_speed = "off"
        levels = []
        for amplitude in (0.001, 0.004):
            chain = self.mw._ReceiverChain(RATE, "USB", 2.7e3)
            chain.options = options
            t = np.arange(int(0.5 * RATE)) / RATE
            tone = amplitude * np.exp(2j * np.pi * 1000 * t)
            audio = _receive(chain, tone.astype(np.complex64))[4800:]
            levels.append(_tone_level(audio, 1000))
        self.assertAlmostEqual(levels[1] / levels[0], 4.0, delta=0.2)

    def test_am_level_follows_modulation_not_signal_strength(self):
        t = np.arange(int(1.0 * RATE)) / RATE
        levels = []
        for carrier in (0.001, 0.1):
            chain = self.mw._ReceiverChain(RATE, "AM", 10e3)
            am = (carrier * (1 + 0.5 * np.sin(2 * np.pi * 800 * t))).astype(
                np.complex64
            )
            audio = _receive(chain, am)[int(0.3 * AUDIO_RATE) :]
            levels.append(_tone_level(audio, 800))
        for level in levels:
            self.assertAlmostEqual(level, 0.5, delta=0.05)


class TestAudioHelpers(unittest.TestCase):
    """gui/audio_chain.py building blocks (no Qt)."""

    def test_channel_power_matches_a_tone(self):
        from sdr_module.gui.audio_chain import FLOOR_DB, channel_power_db

        tone = 0.1 * np.exp(2j * np.pi * 0.01 * np.arange(4096))
        self.assertAlmostEqual(channel_power_db(tone.astype(np.complex64)), -20, 3)
        self.assertAlmostEqual(channel_power_db(tone.astype(np.complex128)), -20, 3)
        self.assertEqual(channel_power_db(np.zeros(0, np.complex64)), FLOOR_DB)
        self.assertEqual(channel_power_db(np.zeros(10, np.complex64)), FLOOR_DB)

    def test_dc_blocker_streams_like_one_block(self):
        from sdr_module.gui.audio_chain import DcBlocker

        rng = np.random.default_rng(2)
        x = (0.3 + 0.1 * rng.standard_normal(20000)).astype(np.float32)
        whole = DcBlocker(20.0, AUDIO_RATE).process(x)
        blocker = DcBlocker(20.0, AUDIO_RATE)
        parts = np.concatenate(
            [blocker.process(x[i : i + 777]) for i in range(0, len(x), 777)]
        )
        np.testing.assert_allclose(parts, whole, atol=1e-5)
        # It starts at zero (no step) and ends with no DC.
        self.assertLess(abs(whole[0]), 1e-6)
        self.assertLess(abs(float(np.mean(whole[-5000:]))), 0.01)
        # Reference recursion.
        a = DcBlocker(20.0, AUDIO_RATE).a
        y = np.zeros(len(x))
        prev_x, prev_y = float(x[0]), 0.0
        for i, value in enumerate(x.astype(np.float64)):
            prev_y = value - prev_x + a * prev_y
            prev_x = value
            y[i] = prev_y
        np.testing.assert_allclose(whole, y, atol=1e-5)

    def test_join_never_steps(self):
        from sdr_module.gui.audio_chain import AudioJoin

        join = AudioJoin(AUDIO_RATE)
        first = join.play(np.full(1000, 0.5, dtype=np.float32))
        self.assertLess(abs(first[0]), 0.01)  # faded in from silence
        self.assertAlmostEqual(float(first[-1]), 0.5, places=5)
        # A new chain whose audio starts at 0: the join starts at 0.5.
        second = join.join(np.zeros(1000, dtype=np.float32))
        self.assertAlmostEqual(float(second[0]), 0.5, delta=0.01)
        self.assertLess(np.max(np.abs(np.diff(second))), 0.01)
        tail = join.fade_out(np.full(1000, 0.4, dtype=np.float32))
        self.assertEqual(len(tail), 240)
        self.assertLess(abs(float(tail[-1])), 0.01)
        self.assertFalse(join.playing)

    def test_agc_never_overshoots_a_sudden_burst(self):
        from sdr_module.gui.audio_chain import AudioAgc

        agc = AudioAgc(AUDIO_RATE, "slow")
        quiet = 0.001 * np.sin(np.arange(48000) * 0.1).astype(np.float32)
        agc.process(quiet)
        burst = np.concatenate(
            [quiet[:1000], 0.5 * np.sin(np.arange(4000) * 0.1).astype(np.float32)]
        )
        out = agc.process(burst)
        self.assertLessEqual(float(np.max(np.abs(out))), 0.75)


if __name__ == "__main__":
    unittest.main()

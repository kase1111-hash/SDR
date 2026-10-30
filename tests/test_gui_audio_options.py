#!/usr/bin/env python3
"""
The Audio group and the I/Q options of the control panel.

* fg:FG-G12 / ux:UX-G11: noise reduction Off / Light / Strong.
* fg:FG-G13 / ux:UX-G09: CTCSS tone squelch, and the tone heard.
* fg:FG-G14 / fg:FG-U07 (audio part): the audio AGC speed.
* fg:FG-G15 / rx:G-06: Remove DC spike, Balance I/Q.
"""

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_squelch import (  # noqa: E402
    FRAME,
    RATE,
    Frames,
    carrier,
    fm,
    noise,
    pretend_audio_output,
)


def _voice(seconds, tone_hz=None, seed=4):
    """Voice-like 300-3000 Hz audio (peak 1), with a CTCSS sub-tone."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE / 50)  # built at 48 kHz, then held
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / 48000)
    spec[(f < 300) | (f > 3000)] = 0
    voice = np.fft.irfft(spec, n)
    voice /= np.max(np.abs(voice))
    voice = 0.8 * np.repeat(voice, 50)
    if tone_hz:
        t = np.arange(len(voice)) / RATE
        voice = voice + 0.15 * np.sin(2 * np.pi * tone_hz * t)
    return voice


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestAudioGroup(_WindowTestCase):
    def setUp(self):
        super().setUp()
        pretend_audio_output(self.win)
        self.panel = self.win._control_panel

    def test_controls_are_there_with_tooltips(self):
        p = self.panel
        for widget in (
            p._nr_combo,
            p._audio_agc_combo,
            p._tone_combo,
            p._dc_check,
            p._iq_check,
        ):
            self.assertTrue(widget.toolTip(), widget)
        labels = [p._nr_combo.itemText(i) for i in range(p._nr_combo.count())]
        self.assertEqual(labels, ["Off", "Light", "Strong"])
        agc = [
            p._audio_agc_combo.itemText(i) for i in range(p._audio_agc_combo.count())
        ]
        self.assertEqual(agc, ["Off", "Fast", "Medium", "Slow"])
        self.assertEqual(p.audio_agc_speed(), "medium")
        self.assertEqual(p._tone_combo.count(), 51)
        self.assertEqual(p._tone_combo.itemText(13), "100.0 Hz (1Z)")

    def test_agc_and_tone_only_where_they_apply(self):
        p = self.panel
        self.win._set_demod_mode("FM")
        self.assertFalse(p._audio_agc_combo.isEnabled())
        self.assertTrue(p._tone_combo.isEnabled())
        self.win._set_demod_mode("USB")
        self.assertTrue(p._audio_agc_combo.isEnabled())
        self.assertFalse(p._tone_combo.isEnabled())
        self.win._set_demod_mode("None (I/Q)")
        self.assertFalse(p._nr_combo.isEnabled())
        self.assertFalse(p._squelch_check.isEnabled())

    def test_options_reach_the_receiver_and_are_saved(self):
        p = self.panel
        p.set_noise_reduction("strong")
        p.set_audio_agc_speed("fast")
        p.set_tone_squelch_hz(100.0)
        p.set_iq_options(True, True)
        options = self.win._audio_options
        self.assertIsNotNone(options.noise_reduction)
        self.assertEqual(options.agc_speed, "fast")
        self.assertEqual(self.win._tone_squelch.tone_hz, 100.0)
        self.assertTrue(self.win._iq_conditioner.remove_dc)
        self.assertTrue(self.win._iq_conditioner.balance_iq)
        values = self.store.values
        self.assertEqual(values["audio/noise_reduction"], "strong")
        self.assertEqual(values["audio/agc_speed"], "fast")
        self.assertEqual(values["audio/tone_squelch"], 100.0)
        self.assertIs(values["audio/remove_dc"], True)
        self.assertIs(values["audio/balance_iq"], True)
        p.set_noise_reduction("off")
        self.assertIsNone(options.noise_reduction)
        # A new window restores them.
        p.set_noise_reduction("light")
        self.win._persist_state()
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        self.addCleanup(other.close)
        self.assertEqual(other._control_panel.noise_reduction(), "light")
        self.assertIsNotNone(other._audio_options.noise_reduction)
        self.assertEqual(other._audio_options.agc_speed, "fast")
        self.assertEqual(other._tone_squelch.tone_hz, 100.0)
        self.assertTrue(other._iq_conditioner.remove_dc)

    def test_noise_reduction_quiets_an_empty_ssb_channel(self):
        self.win.set_frequency(14.2e6)
        self.win._set_demod_mode("USB")
        self.panel.set_squelch_enabled(False)
        self.panel.set_audio_agc_speed("off")  # (a fixed gain: compare levels)
        iq = noise(120 * FRAME, -60.0, seed=21)
        quiet = {}
        for level in ("off", "strong"):
            self.panel.set_noise_reduction(level)
            self.win._reset_demodulator()
            frames = Frames(self, iq).run()
            audio = frames.played()[int(2.0 * 48000) :]
            quiet[level] = float(np.sqrt(np.mean(audio**2)))
        self.assertLess(quiet["strong"], 0.5 * quiet["off"])

    def test_tone_squelch_and_the_tone_heard(self):
        self.win.set_frequency(146.94e6)
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")
        self.panel.set_squelch_db(-90)
        iq = fm(_voice(3.0, tone_hz=100.0), 5e3, -50.0) + noise(int(3.0 * RATE), -95.0)
        n = len(iq) - len(iq) % FRAME
        frames = Frames(self, iq[:n]).run()
        # The readout shows the tone the station sends.
        self.assertEqual(self.panel._tone_heard.text(), "100.0 Hz (1Z)")
        self.assertIn("100.0 Hz (1Z) CTCSS tone", self.panel._tone_heard.toolTip())
        self.assertTrue(any(len(a) for a in frames.audio))
        # A tone squelch on another tone keeps the speaker quiet...
        self.panel.set_tone_squelch_hz(103.5)
        self.win._reset_demodulator()
        frames = Frames(self, iq[:n]).run()
        self.assertFalse(any(len(a) > 240 for a in frames.audio))
        # ...and on the right one it plays (after the tone is recognised).
        self.panel.set_tone_squelch_hz(100.0)
        self.win._reset_demodulator()
        frames = Frames(self, iq[:n]).run()
        self.assertTrue(all(len(a) for a in frames.audio[-30:]))

    def test_remove_dc_spike(self):
        # G-06: a DC spike at the center played as a constant CW tone.
        self.win.set_frequency(7.03e6)
        self.win._set_demod_mode("CW", bandwidth="500 Hz")
        self.panel.set_squelch_db(-85)
        iq = noise(40 * FRAME, -100.0, seed=22) + np.complex64(0.002 + 0.001j)
        spike = Frames(self, iq).run()
        self.assertTrue(spike.states[-1])  # the spike opens the squelch
        self.panel.set_iq_options(True, False)
        self.win._reset_demodulator()
        clean = Frames(self, iq).run()
        self.assertFalse(any(clean.states[10:]))
        self.assertLess(self.win._last_peak_db, -95)

    def test_balance_iq_removes_the_mirror_image(self):
        self.win.set_frequency(100.0e6)
        self.win._set_demod_mode("None (I/Q)")
        n = 40 * FRAME
        tone = carrier(n, 300e3, -20.0)
        i, q = tone.real, 1.1 * np.imag(tone * np.exp(1j * np.radians(4.0)))
        iq = (i + 1j * q).astype(np.complex64) + noise(n, -90.0, seed=23)

        spectra = []
        compute = self.win._power_spectrum_dbfs

        def capture(samples):
            spectra.append(compute(samples))
            return spectra[-1]

        self.win._power_spectrum_dbfs = capture

        def image_db():
            power = spectra[-1]
            bin_hz = RATE / len(power)
            center = len(power) // 2
            mirror = center + int(round(-300e3 / bin_hz))
            main = center + int(round(300e3 / bin_hz))
            return float(
                power[mirror - 2 : mirror + 3].max() - power[main - 2 : main + 3].max()
            )

        Frames(self, iq).run()
        before = image_db()
        self.panel.set_iq_options(False, True)
        Frames(self, iq).run()
        after = image_db()
        self.assertGreater(before, -30)  # a 10 % / 4 degree error: about -24 dBc
        self.assertLess(after, -45)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestRecordingStaysRaw(_WindowTestCase):
    def test_recordings_keep_the_raw_samples(self):
        # The I/Q corrections are for listening and looking; a recording is
        # "raw I/Q" and keeps the spike.
        self.win._control_panel.set_iq_options(True, True)
        captured = []
        self.win._capture = captured.append
        self.win._recording = True
        try:
            iq = noise(FRAME, -80.0, seed=24) + np.complex64(0.01)
            self.win._process_block(iq)
        finally:
            self.win._recording = False
        self.assertIs(captured[0], iq)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestDemoRepeaterTone(_WindowTestCase):
    """The demo's 2 m repeater (145.290 MHz) sends a 100.0 Hz CTCSS tone,
    so Tone squelch and Tone heard can be tried without hardware."""

    def _listen(self, tone_hz, seconds=12.0):
        from sdr_module.gui.device_dialog import MockDevice

        pretend_audio_output(self.win)
        device = MockDevice()
        device.set_frequency(145.29e6)
        device.start_rx()
        self.win.set_frequency(145.29e6)
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")
        self.win._control_panel.set_tone_squelch_hz(tone_hz)
        heard, written = set(), []
        self.win._audio.write = written.append
        for _ in range(int(seconds / (FRAME / RATE))):
            self.win._process_block(device.read_samples(FRAME))
            heard.add(self.win._control_panel._tone_heard.text())
        return heard, sum(len(a) for a in written) / 48000.0

    def test_the_tone_is_heard_and_gates_the_audio(self):
        heard, played = self._listen(0.0)
        self.assertIn("100.0 Hz (1Z)", heard)
        self.assertGreater(played, 4.0)  # 8 s on in every 12
        self.win._reset_demodulator()
        heard, played = self._listen(103.5)  # the neighbouring tone
        self.assertIn("100.0 Hz (1Z)", heard)
        self.assertLess(played, 0.05)  # (fade-outs at most)
        self.win._reset_demodulator()
        _heard, played = self._listen(100.0)
        self.assertGreater(played, 4.0)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestAudioOptionChangesMidStream(_WindowTestCase):
    def setUp(self):
        super().setUp()
        pretend_audio_output(self.win)
        self.panel = self.win._control_panel
        self.win.set_frequency(146.94e6)
        self.win._set_demod_mode("FM", "5 kHz", "25 kHz")
        self.panel.set_squelch_db(-90)

    def test_switching_noise_reduction_does_not_click(self):
        t = np.arange(30 * FRAME) / RATE
        iq = fm(0.8 * np.sin(2 * np.pi * 700 * t), 5e3, -50.0) + noise(
            30 * FRAME, -95.0
        )
        first = Frames(self, iq[: 15 * FRAME]).run()
        self.panel.set_noise_reduction("light")
        second = Frames(self, iq[15 * FRAME :]).run()
        before, after = first.played(), second.played()
        seam = np.abs(np.diff(np.concatenate((before[-3:], after[:400]))))
        self.assertLess(float(seam.max()), 0.1)  # faded, not a step

    def test_the_lamp_says_when_the_tone_squelch_mutes(self):
        from sdr_module.gui.level_meter import STATES

        self.panel.set_tone_squelch_hz(123.0)
        iq = fm(_voice(2.0, tone_hz=100.0), 5e3, -50.0)
        n = len(iq) - len(iq) % FRAME
        Frames(self, iq[:n]).run()
        meter = self.win._level_meter
        self.assertEqual(meter.state(), "tone")
        self.assertEqual(STATES["tone"][0], "TONE")
        self.win._refresh_info_panel()
        self.assertIn("tone 123.0 Hz", self.win._info_panel.value("squelch"))

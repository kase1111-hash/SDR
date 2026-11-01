#!/usr/bin/env python3
"""
Low cut in the window (K-A2): Audio group > Low cut (Off, 100, 200 or
300 Hz) and Radio > Audio > Low Cut, for AM, SSB and narrow FM (not CW,
broadcast FM or I/Q), what the receiver does with it, and its setting.
"""

import math
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_final_window import _IniSettings  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_squelch import (  # noqa: E402
    FRAME,
    RATE,
    Frames,
    fm,
    noise,
    pretend_audio_output,
)
from tests.voice_synth import band_power, pitched_voice, tone_level  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication

AUDIO_RATE = 48000.0


def _menu(win, *path):
    actions = win.menuBar().actions()
    found = None
    for title in path:
        found = next(
            a for a in actions if a.text().replace("&", "").split("\t")[0] == title
        )
        actions = found.menu().actions() if found.menu() is not None else []
    return found


def _with_hum(seconds, hum_hz=60.0, seed=2):
    """Voice (from 300 Hz up, as a microphone channel passes it) with mains
    hum and its second harmonic, as audio at 48 kHz."""
    from sdr_module.dsp.audio_filters import SosFilter, highpass_sos

    voice = pitched_voice(seconds, AUDIO_RATE, seed=seed)
    voice = SosFilter(highpass_sos(300.0, AUDIO_RATE)).process(voice)
    t = np.arange(len(voice)) / AUDIO_RATE
    hum = 0.15 * np.sin(2 * np.pi * hum_hz * t) + 0.08 * np.sin(4 * np.pi * hum_hz * t)
    return voice, voice + hum


def _held(audio):
    held = np.repeat(audio, int(RATE / AUDIO_RATE))
    return held[: len(held) - len(held) % FRAME]


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestLowCutControls(_WindowTestCase):
    def setUp(self):
        super().setUp()
        self.panel = self.win._control_panel

    def test_the_audio_group_has_a_low_cut(self):
        combo = self.panel._low_cut_combo
        self.assertEqual(
            [combo.itemText(i) for i in range(combo.count())],
            ["Off", "100 Hz", "200 Hz", "300 Hz"],
        )
        self.assertEqual(combo.accessibleName(), "Low cut")
        self.assertTrue(combo.toolTip())
        self.assertEqual(self.panel.low_cut(), "off")

    def test_only_where_it_applies(self):
        p, win = self.panel, self.win
        for mode in ("AM", "USB", "LSB"):
            win._set_demod_mode(mode)
            self.assertTrue(p._low_cut_combo.isEnabled(), mode)
        win._set_demod_mode("FM", "5 kHz", "25 kHz")
        self.assertTrue(p._low_cut_combo.isEnabled())
        win._set_demod_mode("CW")
        self.assertFalse(p._low_cut_combo.isEnabled())
        self.assertIn("CW", p._low_cut_combo.toolTip())
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        self.assertFalse(p._low_cut_combo.isEnabled())
        self.assertIn("broadcast", p._low_cut_combo.toolTip())
        win._set_demod_mode("None (I/Q)")
        self.assertFalse(p._low_cut_combo.isEnabled())

    def test_radio_audio_low_cut_follows(self):
        win, p = self.win, self.panel
        win._set_demod_mode("AM")
        _menu(win, "Radio", "Audio", "Low Cut", "200 Hz").trigger()
        self.assertEqual(p.low_cut(), "200")
        p.set_low_cut("300")
        self.assertTrue(_menu(win, "Radio", "Audio", "Low Cut", "300 Hz").isChecked())
        win._set_demod_mode("CW")
        self.assertFalse(_menu(win, "Radio", "Audio", "Low Cut").isEnabled())

    def test_it_reaches_the_receiver_and_is_saved(self):
        p, win = self.panel, self.win
        win._set_demod_mode("AM")
        p.set_low_cut("200")
        self.assertEqual(win._audio_options.low_cut_hz, 200.0)
        self.assertEqual(self.store.values["audio/low_cut"], "200")
        p.set_low_cut("off")
        self.assertEqual(win._audio_options.low_cut_hz, 0.0)
        p.set_low_cut("100")
        win._persist_state()
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        self.addCleanup(other.close)
        self.assertEqual(other._control_panel.low_cut(), "100")
        self.assertEqual(other._audio_options.low_cut_hz, 100.0)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestLowCutInTheReceiver(_WindowTestCase):
    def setUp(self):
        super().setUp()
        pretend_audio_output(self.win)
        self.panel = self.win._control_panel
        self.panel.set_squelch_enabled(False)
        self.panel.set_audio_agc_speed("medium")

    def listen(self, iq):
        self.win._reset_demodulator()
        return Frames(self, iq).run().played()

    def test_am_hum_goes_and_the_voice_stays(self):
        self.win.set_frequency(6.0e6)  # (shortwave: the LO is on it)
        self.win._set_demod_mode("AM")
        voice, audio = _with_hum(3.0)
        iq = 10 ** (-40 / 20) * (1.0 + 0.5 * _held(audio))
        iq = iq.astype(np.complex64) + noise(len(iq), -100.0)
        plain = self.listen(iq)
        self.panel.set_low_cut("200")
        cut = self.listen(iq)
        settled = slice(int(1.0 * AUDIO_RATE), int(3.0 * AUDIO_RATE))
        for hz in (60.0, 120.0):
            before = tone_level(plain[settled], hz, AUDIO_RATE)
            after = tone_level(cut[settled], hz, AUDIO_RATE)
            self.assertGreater(before, 0.01, hz)
            self.assertLess(20 * math.log10(after / before), -35.0, hz)
        ratio = band_power(cut[settled], AUDIO_RATE, 400, 3000) / band_power(
            plain[settled], AUDIO_RATE, 400, 3000
        )
        self.assertAlmostEqual(10 * math.log10(ratio), 0.0, delta=0.5)

    def test_narrow_fm_rumble_goes(self):
        self.win.set_frequency(446.1e6)
        self.win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        voice, audio = _with_hum(3.0, hum_hz=50.0)
        iq = fm(_held(0.6 * audio), 5e3, -50.0) + noise(len(_held(audio)), -95.0)
        plain = self.listen(iq)
        self.panel.set_low_cut("100")
        cut = self.listen(iq)
        settled = slice(int(1.0 * AUDIO_RATE), int(3.0 * AUDIO_RATE))
        before = tone_level(plain[settled], 50.0, AUDIO_RATE)
        after = tone_level(cut[settled], 50.0, AUDIO_RATE)
        self.assertLess(20 * math.log10(after / before), -35.0)
        self.assertEqual(self.win._receiver_chain()._listening.low_cut_hz, 100.0)

    def test_a_tone_squelch_raises_it_to_300_hz(self):
        self.win.set_frequency(446.1e6)
        self.win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        self.panel.set_low_cut("100")
        self.panel.set_tone_squelch_hz(0.0)
        self.listen(fm(np.zeros(FRAME * 4), 5e3, -50.0) + noise(FRAME * 4, -95.0))
        self.assertEqual(self.win._receiver_chain()._listening.low_cut_hz, 100.0)
        self.panel.set_dcs_squelch(23)
        self.panel.set_squelch_enabled(True)
        self.panel.set_squelch_db(-120)
        from sdr_module.dsp.dcs import dcs_waveform

        code = 0.15 * dcs_waveform(23, rate=AUDIO_RATE, seconds=1.0)
        self.listen(fm(_held(code), 5e3, -50.0) + noise(len(_held(code)), -95.0))
        self.assertEqual(self.win._receiver_chain()._listening.low_cut_hz, 300.0)

    def test_broadcast_fm_keeps_its_bass(self):
        self.panel.set_low_cut("300")
        self.win.set_frequency(100.1e6)
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")
        n = int(2.0 * RATE) // FRAME * FRAME
        program = 0.5 * np.sin(2 * np.pi * 100.0 * np.arange(n) / RATE)
        audio = self.listen(fm(program, 75e3, -30.0) + noise(n, -90.0))
        level = tone_level(audio[int(0.5 * AUDIO_RATE) :], 100.0, AUDIO_RATE)
        self.assertGreater(level, 0.4)
        self.assertIsNone(self.win._receiver_chain()._listening.low_cut_hz)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestBroadcastFmIsLeftAlone(unittest.TestCase):
    """Broadcast FM's audio leaves the listening filters exactly as it came
    in, whatever its shape (stereo too: they used to flatten it into one
    channel), and they start afresh once they apply again."""

    def test_untouched_then_afresh(self):
        from sdr_module.gui import main_window as mw
        from sdr_module.gui.audio_chain import AudioOptions

        options = AudioOptions()
        options.notch, options.notch_hz, options.low_cut_hz = "manual", 1000.0, 300.0
        chain = mw._ReceiverChain(RATE, "FM", 200e3)
        chain.options = options
        rng = np.random.default_rng(3)
        mono = (0.1 * rng.standard_normal(FRAME // 50)).astype(np.float32)
        chain._listen(mono)  # (narrow FM: both apply)
        self.assertEqual(chain.notch_frequency_hz, 1000.0)
        self.assertEqual(chain._listening.low_cut_hz, 300.0)
        chain._deemphasis = True  # (as demodulate() sets it at 75 kHz)
        stereo = (0.1 * rng.standard_normal((FRAME // 50, 2))).astype(np.float32)
        for audio in (mono, stereo):
            self.assertIs(chain._listen(audio), audio)
        self.assertIsNone(chain.notch_frequency_hz)
        self.assertIsNone(chain._listening.low_cut_hz)
        chain._deemphasis = False
        out = chain._listen(mono)
        self.assertEqual(out.shape, mono.shape)
        self.assertEqual(chain._listening.low_cut_hz, 300.0)


class TestDamagedLowCutSetting(_IniSettings):
    def window(self, **values):
        from unittest import mock

        from sdr_module.gui import bookmarks_panel, main_window
        from sdr_module.gui.settings_store import GuiSettings

        self.write(first_run_done="true", **values)
        for module in (main_window, bookmarks_panel):
            patcher = mock.patch.object(module, "GuiSettings", GuiSettings)
            patcher.start()
            self.addCleanup(patcher.stop)
        win = main_window.SDRMainWindow(demo_mode=False)
        for timer in (win._display_timer, win._status_timer, win._hotplug_timer):
            timer.stop()

        def close():
            win.close()
            win.deleteLater()
            QApplication.processEvents()

        self.addCleanup(close)
        return win

    def test_an_unknown_corner_is_off_and_logged(self):
        for raw in ("loud", "150", "-1"):
            with self.subTest(raw=raw):
                with self.assertLogs("sdr_module.gui.main_window", "WARNING") as logs:
                    win = self.window(audio__low_cut=raw)
                self.assertEqual(win._control_panel.low_cut(), "off")
                self.assertEqual(win._audio_options.low_cut_hz, 0.0)
                self.assertTrue(any("audio/low_cut" in m for m in logs.output))


if __name__ == "__main__":
    unittest.main()

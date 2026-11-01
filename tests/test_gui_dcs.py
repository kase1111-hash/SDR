#!/usr/bin/env python3
"""
DCS in the window (K-A3): the Tone squelch list offers CTCSS tones or DCS
codes (normal and inverted), the squelch opens only on the code chosen,
Tone heard names the code a station sends, the DCS bits (and a CTCSS
tone) are filtered out of what is heard while a tone squelch is set, the
choice is saved, and a damaged saved code means none.
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


def _voice_above_300(seconds, seed=1):
    """Voice as a radio sends it with a tone squelch: nothing under 300 Hz."""
    from sdr_module.dsp.audio_filters import SosFilter, highpass_sos

    voice = pitched_voice(seconds, AUDIO_RATE, seed=seed)
    return SosFilter(highpass_sos(300.0, AUDIO_RATE)).process(voice)


def narrow_fm(seconds, code=None, inverted=False, ctcss_hz=None, level_db=-50.0):
    """Narrow FM (5 kHz) voice with a DCS code or a CTCSS tone at 15 % of
    the deviation, at 2.4 MS/s with a little noise."""
    from sdr_module.dsp.dcs import dcs_waveform

    audio = 0.6 * _voice_above_300(seconds)
    if code is not None:
        audio = audio + 0.15 * dcs_waveform(
            code, inverted, rate=AUDIO_RATE, seconds=seconds
        )
    if ctcss_hz is not None:
        t = np.arange(len(audio)) / AUDIO_RATE
        audio = audio + 0.15 * np.sin(2 * np.pi * ctcss_hz * t)
    held = np.repeat(audio, int(RATE / AUDIO_RATE))
    n = len(held) - len(held) % FRAME
    return fm(held[:n], 5e3, level_db) + noise(n, -95.0)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestToneSquelchOffersDcs(_WindowTestCase):
    def setUp(self):
        super().setUp()
        self.panel = self.win._control_panel

    def test_the_list_has_the_ctcss_tones_then_the_dcs_codes(self):
        from sdr_module.dsp.dcs import DCS_CODES

        combo = self.panel._tone_combo
        texts = [combo.itemText(i) for i in range(combo.count())]
        self.assertEqual(texts[0], "Off")
        self.assertEqual(texts[13], "100.0 Hz (1Z)")  # (where it always was)
        first = texts.index("DCS 023 N")
        self.assertGreater(first, texts.index("254.1 Hz"))
        normal = [t for t in texts if t.startswith("DCS ") and t.endswith(" N")]
        inverted = [t for t in texts if t.startswith("DCS ") and t.endswith(" I")]
        self.assertEqual(len(normal), len(DCS_CODES))
        self.assertEqual(len(inverted), len(DCS_CODES))
        self.assertLess(texts.index("DCS 754 N"), texts.index("DCS 023 I"))
        tip = combo.itemData(first, 3)  # Qt.ItemDataRole.ToolTipRole
        self.assertIn("normal polarity", tip)
        self.assertIn("DCS", combo.toolTip())

    def test_a_code_reaches_the_receiver_and_is_saved(self):
        p, win = self.panel, self.win
        p.set_dcs_squelch(754, True)
        self.assertEqual(p.dcs_squelch(), (754, True))
        self.assertEqual(p.tone_squelch_hz(), 0.0)
        self.assertEqual(win._tone_squelch.dcs, (754, True))
        self.assertIsNone(win._tone_squelch.tone_hz)
        self.assertEqual(self.store.values["audio/dcs_squelch"], "D754I")
        self.assertEqual(self.store.values["audio/tone_squelch"], 0.0)
        # A CTCSS tone replaces it, and Off clears both.
        p.set_tone_squelch_hz(100.0)
        self.assertIsNone(p.dcs_squelch())
        self.assertIsNone(win._tone_squelch.dcs)
        self.assertEqual(win._tone_squelch.tone_hz, 100.0)
        self.assertEqual(self.store.values["audio/dcs_squelch"], "")
        p.set_dcs_squelch(23)
        win._persist_state()
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        self.addCleanup(other.close)
        self.assertEqual(other._control_panel.dcs_squelch(), (23, False))
        self.assertEqual(other._tone_squelch.dcs, (23, False))
        p.set_dcs_squelch(None)
        self.assertIsNone(p.dcs_squelch())
        self.assertEqual(p._tone_combo.currentText(), "off")

    def test_radio_audio_tone_squelch_says_dcs_too(self):
        action = _menu(self.win, "Radio", "Audio", "Tone Squelch...")
        self.assertIn("DCS", action.toolTip())
        self.win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        self.win.activateWindow()
        action.trigger()
        self.settle(5)
        self.assertIs(QApplication.focusWidget(), self.panel._tone_combo)

    def test_a_dcs_code_on_broadcast_fm_says_where_it_works(self):
        self.win.set_frequency(100.1e6)
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")
        self.panel.set_dcs_squelch(23)
        text = self.win._message_label.text()
        self.assertIn("narrow FM", text)
        self.assertIn("plays as before", text)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestDcsInTheReceiver(_WindowTestCase):
    """A 70 cm channel (FM, 5 kHz) through the real receiver chain."""

    def setUp(self):
        super().setUp()
        pretend_audio_output(self.win)
        self.panel = self.win._control_panel
        self.win.set_frequency(446.1e6)
        self.win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        self.panel.set_squelch_db(-90)

    def listen(self, iq):
        self.win._reset_demodulator()
        frames = Frames(self, iq)
        states = []
        while frames.pos + FRAME <= len(iq):
            frames.run(frames=1)
            states.append(self.win._squelch_state(True))
        return frames.played(), states

    def test_tone_heard_names_the_code(self):
        self.listen(narrow_fm(1.5, code=23))
        heard = self.panel._tone_heard
        self.assertEqual(heard.text(), "DCS 023 N")
        self.assertIn("DCS code 023", heard.toolTip())
        # An inverted code: its normal twin, which sends the same bits.
        self.listen(narrow_fm(1.5, code=47, inverted=True))
        self.assertEqual(heard.text(), "DCS 023 N")
        self.assertIn("DCS 047 I", heard.toolTip())

    def test_the_squelch_opens_only_on_its_code(self):
        self.panel.set_dcs_squelch(23)
        audio, states = self.listen(narrow_fm(2.0, code=23))
        self.assertGreater(float(np.sqrt(np.mean(audio[-24000:] ** 2))), 0.05)
        self.assertEqual(states[-1], "open")
        self.assertTrue(self.win._scanner_host.scan_signal_present())
        self.panel.set_dcs_squelch(25)
        audio, states = self.listen(narrow_fm(2.0, code=23))
        self.assertLess(float(np.max(np.abs(audio), initial=0.0)), 1e-3)
        self.assertEqual(states[-1], "tone")
        self.assertFalse(self.win._scanner_host.scan_signal_present())
        # (The same bits: 023 normal is 047 inverted.)
        self.panel.set_dcs_squelch(47, True)
        audio, states = self.listen(narrow_fm(2.0, code=23))
        self.assertEqual(states[-1], "open")

    def test_the_dcs_bits_are_filtered_out_of_what_is_heard(self):
        iq = narrow_fm(3.0, code=23)
        self.panel.set_tone_squelch_hz(0.0)  # no tone squelch: all of it
        plain, _states = self.listen(iq)
        self.panel.set_dcs_squelch(23)
        heard, _states = self.listen(iq)
        settled = slice(int(1.5 * AUDIO_RATE), int(3.0 * AUDIO_RATE))
        low = band_power(heard[settled], AUDIO_RATE, 0, 200)
        before = band_power(plain[settled], AUDIO_RATE, 0, 200)
        self.assertLess(10 * math.log10(low / before), -30.0)
        # ...and the voice stays (within half a dB above 400 Hz).
        ratio = band_power(heard[settled], AUDIO_RATE, 400, 3000) / band_power(
            plain[settled], AUDIO_RATE, 400, 3000
        )
        self.assertAlmostEqual(10 * math.log10(ratio), 0.0, delta=0.5)

    def test_so_is_a_ctcss_tone(self):
        iq = narrow_fm(3.0, ctcss_hz=100.0)
        plain, _states = self.listen(iq)
        self.panel.set_tone_squelch_hz(100.0)
        heard, _states = self.listen(iq)
        settled = slice(int(1.5 * AUDIO_RATE), int(3.0 * AUDIO_RATE))
        before = tone_level(plain[settled], 100.0, AUDIO_RATE)
        after = tone_level(heard[settled], 100.0, AUDIO_RATE)
        self.assertGreater(before, 0.05)
        self.assertLess(20 * math.log10(after / before), -40.0)

    def test_a_ctcss_squelch_stays_shut_on_a_dcs_station(self):
        self.panel.set_tone_squelch_hz(136.5)
        audio, states = self.listen(narrow_fm(2.0, code=65))
        self.assertEqual(states[-1], "tone")
        self.assertLess(float(np.max(np.abs(audio), initial=0.0)), 1e-3)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestInfoTabNamesTheCode(_WindowTestCase):
    def test_the_squelch_line(self):
        self.win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        self.win._control_panel.set_dcs_squelch(23)
        text, _tone = self.win._squelch_description(False)
        self.assertIn("DCS 023 N", text)


class TestDamagedDcsSetting(_IniSettings):
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

    def test_an_unknown_code_is_no_squelch_and_logged(self):
        for raw in ("D999X", "D024N", "garbage"):
            with self.subTest(raw=raw):
                with self.assertLogs("sdr_module.gui.main_window", "WARNING") as logs:
                    win = self.window(audio__dcs_squelch=raw)
                self.assertIsNone(win._control_panel.dcs_squelch())
                self.assertIsNone(win._tone_squelch.dcs)
                self.assertTrue(any("audio/dcs_squelch" in m for m in logs.output))

    def test_a_good_code_beats_a_stale_tone(self):
        win = self.window(audio__tone_squelch="100.0", audio__dcs_squelch="D023N")
        self.assertEqual(win._control_panel.dcs_squelch(), (23, False))
        self.assertIsNone(win._tone_squelch.tone_hz)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
Radio > Audio > Stereo Blend in the window (M-A2): Less, Normal (the
default: as before) or More, saved (audio/stereo_blend), safe when
damaged, applied at once, and the STEREO badge's tooltip saying which
one a blended or mono station is heard with.
"""

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sdr_module.dsp.fm_stereo import StereoDecoder  # noqa: E402
from tests.test_final_window import _IniSettings  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_squelch import pretend_audio_output  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtGui import QActionGroup
    from PyQt6.QtWidgets import QApplication

LABELS = {
    "less": "Less (more stereo, more hiss)",
    "normal": "Normal",
    "more": "More (quieter weak stations)",
}


def _menu(win, *path):
    """The menu (or action) at ``path`` of menu titles, '&' ignored."""
    actions = win.menuBar().actions()
    found = None
    for title in path:
        found = next(
            a for a in actions if a.text().replace("&", "").split("\t")[0] == title
        )
        actions = found.menu().actions() if found.menu() is not None else []
    return found


def _choice(win, key):
    return _menu(win, "Radio", "Audio", "Stereo Blend", LABELS[key])


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheMenu(_WindowTestCase):
    def test_radio_audio_offers_less_normal_and_more(self):
        menu = _menu(self.win, "Radio", "Audio", "Stereo Blend").menu()
        actions = [a for a in menu.actions() if not a.isSeparator()]
        self.assertEqual(
            [a.text().replace("&", "") for a in actions], list(LABELS.values())
        )
        for action in actions:
            self.assertTrue(action.isCheckable())
            self.assertTrue(action.toolTip())
            self.assertIsInstance(action.actionGroup(), QActionGroup)
            self.assertTrue(action.actionGroup().isExclusive())
        self.assertEqual([a.isChecked() for a in actions], [False, True, False])
        self.assertEqual(self.win._audio_options.stereo_blend, "normal")
        self.assertIn("mono", menu.menuAction().toolTip())
        audio = [
            a.text().replace("&", "")
            for a in _menu(self.win, "Radio", "Audio").menu().actions()
            if not a.isSeparator()
        ]
        self.assertEqual(audio.index("Stereo Blend"), audio.index("Stereo") + 1)

    def test_choosing_applies_at_once_and_is_saved(self):
        for key in ("less", "more", "normal"):
            with self.subTest(key=key):
                with self.assertNoLogs(level="WARNING"):
                    _choice(self.win, key).trigger()
                self.assertTrue(_choice(self.win, key).isChecked())
                self.assertEqual(self.win._audio_options.stereo_blend, key)
                self.assertEqual(self.store.values["audio/stereo_blend"], key)
                self.assertIn("Stereo blend", self.win._message_label.text())

    def test_a_saved_choice_is_restored(self):
        self.store.values["audio/stereo_blend"] = "more"
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        self.addCleanup(other.close)
        for timer in (other._display_timer, other._status_timer, other._hotplug_timer):
            timer.stop()
        self.assertEqual(other._audio_options.stereo_blend, "more")
        self.assertTrue(_choice(other, "more").isChecked())


class TestADamagedSetting(_IniSettings):
    def test_anything_but_a_choice_is_normal(self):
        from sdr_module.gui import bookmarks_panel, main_window
        from sdr_module.gui.settings_store import GuiSettings

        self.write(first_run_done="true", audio__stereo_blend="loud")
        for module in (main_window, bookmarks_panel):
            patcher = mock.patch.object(module, "GuiSettings", GuiSettings)
            patcher.start()
            self.addCleanup(patcher.stop)
        with self.assertLogs("sdr_module.gui.main_window", "WARNING") as logs:
            win = main_window.SDRMainWindow(demo_mode=False)
        for timer in (win._display_timer, win._status_timer, win._hotplug_timer):
            timer.stop()

        def close():
            win.close()
            win.deleteLater()
            QApplication.processEvents()

        self.addCleanup(close)
        self.assertEqual(win._audio_options.stereo_blend, "normal")
        self.assertTrue(_choice(win, "normal").isChecked())
        self.assertTrue(any("audio/stereo_blend" in m for m in logs.output))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestAWeakStation(_WindowTestCase):
    """The demo's 88.5 MHz stereo station: full stereo's hiss would be about
    47 dB below full scale, so Normal blends it partly toward mono, Less
    plays it in full stereo and More in mono; the badge says which."""

    FRAME = 80_000

    def setUp(self):
        super().setUp()
        from sdr_module.gui import device_dialog

        for target, name, value in (
            (device_dialog, "_MAX_LAG_S", 1e9),
            (StereoDecoder, "RISE_S", 0.05),
        ):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        pretend_audio_output(self.win)
        self.win._audio.write = lambda *_a: None

    def listen(self, key, frames=14):
        from sdr_module.gui.device_dialog import MockDevice

        _choice(self.win, key).trigger()
        self.win.set_frequency(88.5e6)
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")
        self.win._reset_demodulator()
        device = MockDevice()
        device.set_sample_rate(2.4e6)
        device.set_frequency(88.5e6)
        device._running = True
        device._rng = np.random.default_rng(11)
        for _ in range(frames):
            self.win._process_block(device.read_samples(self.FRAME))
        decoder = self.win._receiver_chain().stereo_decoder
        return decoder, self.win._stereo_badge

    def test_normal_blends_it_partly(self):
        decoder, badge = self.listen("normal")
        self.assertEqual(decoder.blending, "normal")
        self.assertGreater(decoder.blend, 0.5)
        self.assertLess(decoder.blend, 0.95)
        self.assertEqual(badge.text(), "STEREO")
        self.assertIn("blended partly", badge.toolTip())
        self.assertIn("Stereo Blend: Normal", badge.toolTip())

    def test_less_plays_it_in_full_stereo(self):
        decoder, badge = self.listen("less")
        self.assertEqual(decoder.blending, "less")
        self.assertGreaterEqual(decoder.blend, 0.95)
        self.assertEqual(badge.text(), "STEREO")
        self.assertIn("fully apart", badge.toolTip())

    def test_more_plays_it_in_mono(self):
        decoder, badge = self.listen("more")
        self.assertEqual(decoder.blending, "more")
        self.assertLess(decoder.blend, 0.5)
        self.assertEqual(badge.text(), "MONO")
        self.assertIn("too weak", badge.toolTip())
        self.assertIn("Stereo Blend: More", badge.toolTip())


if __name__ == "__main__":
    unittest.main()

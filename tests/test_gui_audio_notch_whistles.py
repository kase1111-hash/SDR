#!/usr/bin/env python3
"""
Several whistles in the window (M-A3): Auto removes up to three steady
whistles at once, and the Audio group's Whistle row shows every frequency
it removes (in order of frequency; its tooltip lists them strongest
first), without widening the right column.
"""

import math
import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_audio_notch import (  # noqa: E402
    AUDIO_RATE,
    _analytic,
    _upsample,
    _whole_frames,
    _window_of,
)
from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_squelch import (  # noqa: E402
    Frames,
    carrier,
    noise,
    pretend_audio_output,
)
from tests.voice_synth import pitched_voice, tone_level  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication


def usb_with_two_whistles(seconds, seed=6):
    """USB voice at -50 dBFS and carriers 700 and 1900 Hz above the tuned
    frequency (whistles at 700 and 1900 Hz) at -56 and -62 dBFS."""
    voice = pitched_voice(seconds, AUDIO_RATE, seed=seed)
    iq = 10 ** (-50 / 20) * _upsample(_analytic(voice))
    iq = iq + carrier(len(iq), 700.0, -56.0) + carrier(len(iq), 1900.0, -62.0)
    return _whole_frames(iq + noise(len(iq), -100.0, seed=seed))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheWhistleRow(_WindowTestCase):
    def setUp(self):
        super().setUp()
        self.panel = self.win._control_panel
        self.win._set_demod_mode("USB")
        self.panel.set_notch_mode("auto")

    def test_it_shows_every_frequency_removed(self):
        p = self.panel
        p.show_notch_found((1347.2, 612.3, 2800.4))  # (strongest first)
        self.assertEqual(p._notch_found.text(), "612, 1347, 2800 Hz")
        self.assertEqual(p._notch_label.text(), "Whistles:")
        tip = p._notch_found.toolTip()
        self.assertIn("3 steady whistles", tip)
        self.assertLess(tip.index("1347.2 Hz"), tip.index("612.3 Hz"))
        self.assertLess(tip.index("612.3 Hz"), tip.index("2800.4 Hz"))
        self.assertIn("strongest first", tip)
        p.show_notch_found([700.0, 1900.0])
        self.assertEqual(p._notch_found.text(), "700, 1900 Hz")
        # One frequency, as before (a number or a sequence of one).
        for one in (1300.0, (1300.0,)):
            p.show_notch_found(one)
            self.assertEqual(p._notch_found.text(), "1300 Hz")
            self.assertEqual(p._notch_label.text(), "Whistle:")
            self.assertIn("1300.0 Hz", p._notch_found.toolTip())
        for none in (None, (), []):
            p.show_notch_found(none)
            self.assertEqual(p._notch_found.text(), "None")
            self.assertEqual(p._notch_label.text(), "Whistle:")
            self.assertIn("up to three", p._notch_found.toolTip())
        # (Any sequence, a NumPy array too.)
        p.show_notch_found(np.array([1900.0, 700.0]))
        self.assertEqual(p._notch_found.text(), "700, 1900 Hz")

    def test_manual_is_one_whistle_again(self):
        p = self.panel
        p.show_notch_found((700.0, 1900.0))
        self.assertEqual(p._notch_label.text(), "Whistles:")
        for mode in ("manual", "off", "auto"):
            p.set_notch_mode(mode)
            self.assertEqual(p._notch_label.text(), "Whistle:", mode)

    def test_auto_says_it_removes_up_to_three(self):
        combo = self.panel._notch_combo
        auto = combo.itemData(combo.findData("auto"), 3)  # (ToolTipRole)
        for tip in (auto, combo.toolTip()):
            self.assertIn("up to three", tip)

    def test_three_frequencies_take_no_width_from_the_right_column(self):
        from sdr_module.gui import themes

        p = self.panel
        qapp = QApplication.instance()
        old_sheet, old_palette = qapp.styleSheet(), qapp.palette()
        old_theme = themes.current_theme()
        try:
            for theme in (None, *themes.THEMES):
                if theme is not None:
                    themes.apply_theme(qapp, theme)
                p.show_notch_found((4812.0, 3999.0, 2047.0))
                self.settle(3)
                widest = p._nr_combo.minimumSizeHint().width()
                self.assertLessEqual(
                    p._notch_row.minimumSizeHint().width(), widest + 6, theme
                )
                self.assertLessEqual(p.minimumSizeHint().width(), 400, theme)
        finally:
            qapp.setStyleSheet(old_sheet)
            qapp.setPalette(old_palette)
            themes.apply_theme(None, old_theme)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTwoWhistlesInTheReceiver(_WindowTestCase):
    """What is played (48 kHz audio captured from the speaker's write)."""

    def setUp(self):
        super().setUp()
        pretend_audio_output(self.win)
        self.panel = self.win._control_panel
        self.panel.set_squelch_enabled(False)
        self.panel.set_audio_agc_speed("off")  # (a fixed gain: compare levels)

    def test_auto_removes_both_and_says_so(self):
        self.win.set_frequency(14.2e6)
        self.win._set_demod_mode("USB")
        self.panel.set_notch_mode("auto")
        self.win._reset_demodulator()
        audio = Frames(self, usb_with_two_whistles(4.0)).run().played()
        for hz in (700.0, 1900.0):
            before = tone_level(_window_of(audio, 0.1, 0.4), hz)
            after = tone_level(_window_of(audio, 2.0, 4.0), hz)
            self.assertGreater(before, 0.005, hz)
            self.assertLess(20 * math.log10(after / before), -30.0, hz)
        self.assertEqual(self.panel._notch_found.text(), "700, 1900 Hz")
        chain = self.win._receiver_chain()
        found = chain.notch_frequencies_hz
        self.assertEqual([round(f) for f in found], [700, 1900])  # (strongest first)
        self.assertAlmostEqual(chain.notch_frequency_hz, found[0])


if __name__ == "__main__":
    unittest.main()

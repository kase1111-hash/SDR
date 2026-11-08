"""The Second Receiver's Whistle readout names every whistle Auto removes
(post-merge, rx2 with the audio lane's notch for several whistles, M-A3).

After the merge the Second Receiver's chain removes up to three steady
whistles, as the main window's does, but its Whistle row named only the
strongest (``Frame.notch_hz`` is ``chain.notch_frequency_hz``). Now each
frame carries them all (``Frame.notches``, the chain's
``notch_frequencies_hz``: the strongest first) and the row says them as the
main window's Audio group does (``control_panel.show_notch_found``): in
order of frequency, in Hz, "Whistles:" for several, the strongest first in
its tooltip; within the window's width at its minimum size.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_second_receiver import _FakeSink, _SecondCase  # noqa: E402
from tests.test_gui_second_receiver_filters import (  # noqa: E402
    BLOCK,
    RATE,
    _DspCase,
    _usb,
)

#: Three steady whistles (audio Hz, level), the loudest first: 1900 Hz,
#: then 700 Hz, then 2600 Hz.
WHISTLES = [(700.0, 0.02), (1900.0, 0.05), (2600.0, 0.01)]


class TestEachFrameSaysThemAll(_DspCase):
    def test_the_strongest_first(self):
        dsp = self.dsp()
        dsp.options.notch = "auto"
        frames, _audio = self.listen(dsp, _usb(2.0, WHISTLES))
        found = frames[-1].notches
        self.assertEqual([round(f) for f in found], [1900, 700, 2600])
        self.assertAlmostEqual(frames[-1].notch_hz, 1900.0, delta=1.0)  # (as before)

    def test_manual_names_its_own_and_off_none(self):
        dsp = self.dsp()
        dsp.options.notch, dsp.options.notch_hz = "manual", 1000.0
        frames = [dsp.process(_usb(0.2, WHISTLES)[:BLOCK], True) for _ in range(2)]
        self.assertEqual(frames[-1].notches, (1000.0,))
        dsp = self.dsp()
        frames = [dsp.process(_usb(0.2, WHISTLES)[:BLOCK], True) for _ in range(2)]
        self.assertEqual(frames[-1].notches, ())

    def test_nothing_demodulated_nothing_said(self):
        dsp = self.dsp()
        dsp.options.notch = "auto"
        iq = _usb(1.5, WHISTLES)
        for start in range(0, len(iq), BLOCK):
            dsp.process(iq[start : start + BLOCK], True)
        frame = dsp.process(iq[:BLOCK], False)  # (the speaker stopped)
        frame = dsp.process(iq[:BLOCK], False)
        self.assertEqual(frame.notches, ())


class TestTheWording(unittest.TestCase):
    def test_the_main_windows_wording(self):
        try:
            from sdr_module.gui.second_receiver_audio import notch_found_text
        except ImportError:  # pragma: no cover - environment-dependent
            self.skipTest("PyQt6 not available")
        text, tip, tone = notch_found_text((1900.4, 700.0, 2600.0))
        self.assertEqual(text, "700, 1900, 2600 Hz")  # (by frequency: no jumping)
        self.assertEqual(tone, "success")
        self.assertIn("3 steady whistles, the strongest first", tip)
        self.assertIn("1900.4 Hz, 700.0 Hz, 2600.0 Hz", tip)
        text, tip, tone = notch_found_text((1350.0,))
        self.assertEqual(text, "1350 Hz")
        self.assertEqual(tip, "Auto removes a steady whistle at 1350.0 Hz")
        self.assertEqual(notch_found_text(1350.0)[0], "1350 Hz")  # (one number)
        for none in (None, ()):
            text, tip, tone = notch_found_text(none)
            self.assertEqual((text, tone), ("None", "muted"))
            self.assertIn("up to three", tip)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheWhistleRow(_SecondCase):
    def setUp(self):
        super().setUp()
        main_sink = _FakeSink()
        main_sink.open = True
        self.win._audio = main_sink
        self.win._audio_enabled = True
        self.win._audio_problem = None
        self.sink()
        rx = self.rx
        self.assertTrue(rx.start())
        rx._frame_timer.stop()
        rx.set_listen("second")
        rx.set_mode("USB")
        rx._squelch_check.setChecked(False)
        rx._notch_combo.setCurrentIndex(rx._notch_combo.findData("auto"))
        rx._sync_audio_output()

    def listen(self, seconds=2.0):
        rx = self.rx
        self.assertEqual(rx._rate(), RATE)
        iq = _usb(seconds, WHISTLES, rate=rx._rate())
        for start in range(0, len(iq), BLOCK):
            rx._process(iq[start : start + BLOCK])

    def test_it_names_every_whistle_in_hz(self):
        rx = self.rx
        self.listen()
        found = rx._notch_found
        self.assertEqual(found.text(), "700, 1900, 2600 Hz")
        self.assertEqual(found.property("tone"), "success")
        self.assertIn("the strongest first: 1900", found.toolTip())
        self.assertEqual(rx._notch_label.text(), "Whistles:")
        # One whistle again: "Whistle:".
        rx._show_notch_found((1350.0,))
        self.assertEqual(rx._notch_label.text(), "Whistle:")
        self.assertEqual(found.text(), "1350 Hz")

    def test_the_notch_tooltip_says_up_to_three(self):
        tip = self.rx._notch_combo.toolTip()
        self.assertIn("up to three", tip)
        # (The main window's Audio group's, where the notch applies there.)
        self.win._set_demod_mode("USB")
        self.assertEqual(tip, self.win._control_panel._notch_combo.toolTip())

    def test_it_fits_the_window_at_its_minimum_size(self):
        from PyQt6.QtWidgets import QApplication

        rx = self.rx
        for size in ((1024, 640), (860, 560)):
            rx.resize(*size)
            rx._show_notch_found((1900.0, 2600.0, 4800.0))  # (the widest: 4 digits)
            for _ in range(10):  # (the row lays itself out again)
                QApplication.processEvents()
            found = rx._notch_found
            self.assertEqual(found.text(), "1900, 2600, 4800 Hz")
            self.assertGreaterEqual(found.width(), found.sizeHint().width(), size)
            viewport = rx._controls_area.viewport()
            right = found.mapTo(viewport, found.rect().topRight()).x()
            self.assertLessEqual(right, viewport.width(), size)

    def test_stopped_or_off_says_none(self):
        rx = self.rx
        self.listen(1.5)
        self.assertNotEqual(rx._notch_found.text(), "None")
        rx.stop()
        self.assertEqual(rx._notch_found.text(), "None")
        self.assertEqual(rx._notch_label.text(), "Whistle:")

    def choose(self, key):
        combo = self.rx._notch_combo
        combo.setCurrentIndex(combo.findData(key))

    def test_manual_says_whistle_after_auto_found_several(self):
        # ui-4: after Auto named several whistles ("Whistles:"), Manual's
        # one frequency box kept that caption, frame after frame. It says
        # "Whistle:", as the main window's Audio group does.
        rx = self.rx
        self.listen()
        self.assertEqual(rx._notch_found.text(), "700, 1900, 2600 Hz")
        self.assertEqual(rx._notch_label.text(), "Whistles:")
        self.choose("manual")
        self.assertIs(rx._notch_stack.currentWidget(), rx._notch_spin)
        self.assertEqual(rx._notch_label.text(), "Whistle:")
        self.listen(0.5)
        self.assertEqual(rx._notch_label.text(), "Whistle:")
        # Auto again searches anew (no old list meanwhile) and names them
        # all once found.
        self.choose("auto")
        self.assertEqual(rx._notch_found.text(), "None")
        self.listen(1.5)
        self.assertEqual(rx._notch_label.text(), "Whistles:")
        self.assertEqual(rx._notch_found.text(), "700, 1900, 2600 Hz")

    def test_off_then_manual_says_whistle(self):
        rx = self.rx
        self.listen()
        self.assertEqual(rx._notch_label.text(), "Whistles:")
        self.choose("off")
        self.choose("manual")
        self.assertEqual(rx._notch_label.text(), "Whistle:")

    def test_the_main_windows_audio_group_says_the_same(self):
        rx, panel = self.rx, self.win._control_panel
        self.listen()
        self.choose("manual")
        self.win._set_demod_mode("USB")
        panel._notch_combo.setCurrentIndex(panel._notch_combo.findData("auto"))
        panel.show_notch_found((1900.0, 700.0, 2600.0))
        self.assertEqual(panel._notch_label.text(), "Whistles:")
        panel._notch_combo.setCurrentIndex(panel._notch_combo.findData("manual"))
        self.assertEqual(rx._notch_label.text(), panel._notch_label.text())


if __name__ == "__main__":
    unittest.main()

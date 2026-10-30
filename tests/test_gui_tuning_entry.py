#!/usr/bin/env python3
"""
Typing a frequency (rx:RX-13, ux:UX-B13, mc:MC-03, mc:MC-U17): unit
suffixes and e-notation are understood, a bare number in Hz typed into the
MHz field is read as Hz, out-of-range text is kept (never cut down to a
different frequency) and refused with a message on Enter, and Esc cancels
the edit and returns to the plots.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest
tests/test_gui_tuning_entry.py``
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase, require_pyqt6  # noqa

if HAS_PYQT6:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication


class _EntryCase(_WindowTestCase):
    demo = True

    def type_frequency(self, text, key=None):
        """Ctrl+L, type ``text``, then press ``key`` (Enter by default)."""
        self.win._focus_frequency_entry()
        self.settle()
        field = QApplication.focusWidget()
        self.assertIs(field, self.win._frequency_field())
        QTest.keyClicks(field, text)
        QTest.keyClick(field, key if key is not None else Qt.Key.Key_Return)
        self.settle()
        return field

    def error(self):
        label = self.win._message_label
        return label.text() if label.property("tone") == "danger" else ""


class TestTypedFrequencies(_EntryCase):
    def test_unit_suffixes_and_notations(self):
        for text, hz in (
            ("433.92M", 433.92e6),
            ("7074k", 7.074e6),
            ("1.09 GHz", 1.09e9),
            ("145.8 mhz", 145.8e6),
            ("7.074e6", 7.074e6),
            ("144,39", 144.39e6),
            ("162550000", 162.55e6),  # Hz typed into the MHz field
            ("146.52", 146.52e6),
        ):
            self.type_frequency(text)
            self.assertAlmostEqual(self.freq(), hz, places=0, msg=text)
            self.assertEqual(self.error(), "", text)
            # Enter after Ctrl+L returns to the plots.
            self.assertTrue(self.win._spectrum.hasFocus(), text)

    def test_digits_are_never_dropped(self):
        self.win._focus_frequency_entry()
        self.settle()
        field = QApplication.focusWidget()
        QTest.keyClicks(field, "16255")
        self.assertEqual(field.text(), "16255")  # 16255 MHz: kept, red
        self.assertEqual(field.property("tone"), "danger")
        QTest.keyClicks(field, "0000")
        self.assertEqual(field.text(), "162550000")  # now clearly Hz
        self.assertFalse(field.property("tone"))
        QTest.keyClick(field, Qt.Key.Key_Return)
        self.assertEqual(self.freq(), 162.55e6)

    def test_characters_that_cant_be_a_frequency_are_not_typed(self):
        self.win._focus_frequency_entry()
        self.settle()
        field = QApplication.focusWidget()
        QTest.keyClicks(field, "1x4-6;.5")
        self.assertEqual(field.text(), "146.5")

    def test_out_of_range_is_refused_with_a_message_and_kept_to_fix(self):
        field = self.type_frequency("7000")
        self.assertEqual(self.freq(), 100.1e6)  # never tuned to 700 MHz
        self.assertEqual(self.win._device.frequency, 100.1e6)
        self.assertEqual(
            self.error(),
            "7000 MHz is out of range: the receiver tunes "
            "0.001–6000 MHz. For 7000 kHz, type 7000k.",
        )
        self.assertEqual(field.text(), "100.100")
        # The field keeps the focus (text selected) to type it again ...
        self.assertTrue(field.hasFocus())
        QTest.keyClicks(field, "7000k")
        QTest.keyClick(field, Qt.Key.Key_Return)
        self.settle()
        self.assertEqual(self.freq(), 7e6)
        # ... and then Enter returns to the plots as usual.
        self.assertTrue(self.win._spectrum.hasFocus())

    def test_unfinished_text_is_refused_too(self):
        self.type_frequency("7.074e")
        self.assertEqual(self.freq(), 100.1e6)
        self.assertIn("Can't read “7.074e” as a frequency", self.error())

    def test_leaving_the_field_with_bad_text_says_so(self):
        self.win._focus_frequency_entry()
        self.settle()
        field = QApplication.focusWidget()
        QTest.keyClicks(field, "9000")
        self.win._waterfall.setFocus()
        self.settle()
        self.assertIn("9000 MHz is out of range", self.error())
        self.assertEqual(field.text(), "100.100")
        self.assertEqual(self.freq(), 100.1e6)

    def test_grouped_units(self):
        entry = self.win._control_panel._freq_input
        entry._unit_combo.setCurrentText("kHz")
        self.type_frequency("7,074")
        self.assertEqual(self.freq(), 7.074e6)
        entry._unit_combo.setCurrentText("Hz")
        self.type_frequency("145,800,000")
        self.assertEqual(self.freq(), 145.8e6)
        entry._unit_combo.setCurrentText("MHz")


class TestEscCancelsTheEdit(_EntryCase):
    def test_esc_restores_the_frequency_and_returns_to_the_plots(self):
        for typed in ("433", "4"):
            field = self.type_frequency(typed, key=Qt.Key.Key_Escape)
            self.assertEqual(self.freq(), 100.1e6, typed)
            self.assertEqual(self.win._device.frequency, 100.1e6, typed)
            self.assertEqual(field.text(), "100.100", typed)
            self.assertEqual(self.win._freq_label.text(), "100.100 MHz", typed)
            self.assertTrue(self.win._spectrum.hasFocus(), typed)
            self.assertEqual(self.error(), "", typed)

    def test_esc_on_out_of_range_text_is_quiet(self):
        self.type_frequency("9000", key=Qt.Key.Key_Escape)
        self.assertEqual(self.freq(), 100.1e6)
        self.assertEqual(self.error(), "")
        self.assertFalse(self.win._frequency_field().property("tone"))

    def test_enter_still_tunes_after_an_esc(self):
        self.type_frequency("433", key=Qt.Key.Key_Escape)
        self.type_frequency("433.92")
        self.assertEqual(self.freq(), 433.92e6)


class TestFieldSteps(_EntryCase):
    def test_the_arrows_step_by_the_tuning_step(self):
        field = self.win._frequency_field()
        self.win._set_step(12.5e3)
        field.stepBy(1)
        self.assertEqual(self.freq(), 100.1125e6)
        field.stepBy(10)  # Page Up: ten steps
        self.assertEqual(self.freq(), 100.2375e6)
        self.win._set_step(100.0)
        field.stepBy(-3)
        self.assertEqual(self.freq(), 100.2372e6)
        self.assertIn("step 100 Hz", field.toolTip())


class TestFrequencyInputAlone(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui.control_panel import FrequencyInput

        self.entry = FrequencyInput()
        self.spin = self.entry._freq_input
        self.rejected = []
        self.tuned = []
        self.entry.entry_rejected.connect(self.rejected.append)
        self.entry.frequency_changed.connect(self.tuned.append)

    def test_the_device_range_and_its_explanation(self):
        self.entry.set_tuning_range(0.5e6, 1.766e9, "RTL-SDR tunes 0.5–1766 MHz")
        self.assertEqual(self.entry.tuning_range(), (0.5e6, 1.766e9))
        self.assertEqual((self.spin.minimum(), self.spin.maximum()), (0.5, 1766.0))
        self.assertIn("RTL-SDR tunes 0.5–1766 MHz", self.spin.toolTip())
        self.spin.lineEdit().selectAll()
        QTest.keyClicks(self.spin, "2400")
        QTest.keyClick(self.spin, Qt.Key.Key_Return)
        self.assertEqual(
            self.rejected, ["2400 MHz is out of range: RTL-SDR tunes 0.5–1766 MHz."]
        )
        self.assertEqual(self.tuned, [])
        # Bad ranges are ignored.
        self.entry.set_tuning_range(5e6, 1e6)
        self.assertEqual(self.entry.tuning_range(), (0.5e6, 1.766e9))

    def test_the_step(self):
        self.entry.set_step_hz(100.0)
        self.assertEqual(self.entry.step_hz(), 100.0)
        self.assertAlmostEqual(self.spin.singleStep(), 0.0001)
        self.entry._unit_combo.setCurrentText("kHz")
        self.assertAlmostEqual(self.spin.singleStep(), 0.1)
        self.entry.set_step_hz(float("nan"))
        self.assertEqual(self.entry.step_hz(), 100.0)

    def test_cancel_edit(self):
        self.spin.lineEdit().selectAll()
        QTest.keyClicks(self.spin, "433")
        self.entry.cancel_edit()
        self.assertEqual(self.spin.text(), "100.100")
        self.assertEqual(self.tuned, [])


if __name__ == "__main__":
    unittest.main()

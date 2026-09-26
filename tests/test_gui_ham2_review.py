"""Review fixes for the S-Meter panel and the AM/FM Radio Tuner window."""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QFontMetricsF, QWheelEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - PyQt6 missing
    HAS_PYQT6 = False

_APP = None

# Height of the S-Meter tab page in the main window at its default 1400x900.
_TAB_PAGE_HEIGHT_1400x900 = 305


def _app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _settle(frames: int = 3) -> None:
    for _ in range(frames):
        _app().processEvents()


def _tone(amplitude: float, n: int = 4096) -> np.ndarray:
    t = np.arange(n) / 48000.0
    return (amplitude * np.exp(2j * np.pi * 1000 * t)).astype(np.complex64)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestSignalMeterPanelFits(unittest.TestCase):
    """The whole report fits under the gauge in the default-size tab."""

    def setUp(self):
        _app()
        from sdr_module.ham.gui.signal_meter_widget import SignalMeterPanel

        self.panel = SignalMeterPanel()

    def tearDown(self):
        self.panel._update_timer.stop()
        self.panel.deleteLater()
        _settle()

    def test_report_fits_default_tab_height(self):
        self.assertLessEqual(self.panel.sizeHint().height(), _TAB_PAGE_HEIGHT_1400x900)

    def test_mode_combo_sits_in_the_rst_row(self):
        panel = self.panel
        panel.resize(360, _TAB_PAGE_HEIGHT_1400x900)
        panel.show()
        _settle()
        rst = panel._rst_label.geometry()
        combo = panel._mode_combo.geometry()
        self.assertLess(abs(rst.center().y() - combo.center().y()), 6)
        self.assertGreater(combo.left(), rst.right())
        # Measurements line up under the S-meter / RST / Mode columns.
        self.assertEqual(panel._dbm_label.x(), panel._s_meter_label.x())
        self.assertEqual(panel._snr_label.x(), panel._rst_label.x())
        self.assertEqual(panel._peak_label.x(), combo.x())
        for w in (panel._mode_combo, panel._peak_label):
            self.assertLessEqual(w.geometry().bottom(), panel.height())

    def test_mode_names_are_short_with_explanatory_tooltips(self):
        combo = self.panel._mode_combo
        self.assertEqual(
            [combo.itemText(i) for i in range(combo.count())],
            ["Phone", "CW", "Digital"],
        )
        for i in range(combo.count()):
            tip = combo.itemData(i, Qt.ItemDataRole.ToolTipRole)
            self.assertTrue(tip and "report" in tip.lower())

    def test_live_readouts_still_update(self):
        self.panel.update_samples(_tone(0.05))
        self.panel._update_display()
        self.assertTrue(self.panel._s_meter_label.text().startswith("S"))
        self.assertTrue(self.panel._snr_label.text().endswith("dB"))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestGaugeUsesWidth(unittest.TestCase):
    """A short, wide face still gets a (nearly) full-width scale."""

    def test_short_face_keeps_most_of_the_width(self):
        _app()
        from sdr_module.ham.gui.signal_meter_widget import AnalogMeterWidget

        meter = AnalogMeterWidget()
        meter.resize(400, 138)
        fm = QFontMetricsF(meter._label_font())
        _cx, _cy, radius = meter._geometry(fm)
        # ~0.87 with the current layout; the old needle-room rule gave ~0.71.
        self.assertGreaterEqual(radius, 0.8 * meter._radius_for_width(400.0, fm))
        meter.deleteLater()

    def test_compact_meter_shows_no_reading_before_data(self):
        _app()
        from sdr_module.ham.gui.signal_meter_widget import CompactSignalMeter

        compact = CompactSignalMeter()
        self.assertNotIn("S1", compact._bar_label.text())
        self.assertIn("--", compact._bar_label.text())
        compact.deleteLater()


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestRadioTunerReview(unittest.TestCase):
    """Tuner window: resizing, wheel tuning and long-press storing."""

    def setUp(self):
        _app()
        from sdr_module.ham.gui.radio_tuner import RadioTunerWidget

        self.tuner = RadioTunerWidget(None, 2.4e6)
        self.emitted = []
        self.tuner.frequency_changed.connect(lambda f, b: self.emitted.append((f, b)))

    def tearDown(self):
        self.tuner.close()
        self.tuner.deleteLater()
        _settle()

    def test_no_size_grip_over_the_close_button(self):
        self.assertFalse(self.tuner.isSizeGripEnabled())
        self.assertNotEqual(self.tuner.minimumSize(), self.tuner.maximumSize())

    def test_enlarged_window_keeps_rows_at_natural_height(self):
        t = self.tuner
        t.show()
        t.resize(900, 620)
        _settle()
        self.assertLessEqual(t._seek_up_btn.height(), t._tuning_dial.height() + 2)
        self.assertLessEqual(
            t._fm_btn.height() + t._am_btn.height(), t._freq_display.height() + 12
        )
        close = [b for b in t.findChildren(type(t._fm_btn)) if b.text() == "Close"]
        self.assertTrue(close)
        bottom = close[0].mapTo(t, close[0].rect().bottomLeft()).y()
        self.assertGreater(bottom, t.height() - 30)

    def _wheel(self, delta: int) -> None:
        pos = QPointF(40, 30)
        event = QWheelEvent(
            pos,
            pos,
            QPoint(0, delta),
            QPoint(0, delta),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.ScrollUpdate,
            False,
        )
        _app().sendEvent(self.tuner._tuning_dial, event)

    def test_wheel_steps_one_channel_per_notch(self):
        start = self.tuner.get_frequency()
        self._wheel(120)
        self.assertAlmostEqual(self.tuner.get_frequency(), start + 100e3)
        self._wheel(-240)
        self.assertAlmostEqual(self.tuner.get_frequency(), start - 100e3)

    def test_touchpad_deltas_accumulate(self):
        start = self.tuner.get_frequency()
        for _ in range(14):
            self._wheel(8)  # 112 units: not yet a notch
        self.assertAlmostEqual(self.tuner.get_frequency(), start)
        self._wheel(8)  # 120 units in total
        self.assertAlmostEqual(self.tuner.get_frequency(), start + 100e3)

    def test_long_press_stores_without_clicking(self):
        t = self.tuner
        t.show()
        _settle()
        t._seek_up_btn.click()
        freq = t.get_frequency()
        self.emitted.clear()
        button = t._preset_buttons[3]
        QTest.mousePress(button, Qt.MouseButton.LeftButton)
        QTest.qWait(850)
        QTest.mouseRelease(button, Qt.MouseButton.LeftButton)
        self.assertEqual(button.get_preset().frequency_hz, freq)
        self.assertFalse(button.isDown())
        self.assertEqual(self.emitted, [])  # storing does not retune
        # A later ordinary click on another preset still tunes.
        t._preset_buttons[0].click()
        self.assertEqual(self.emitted[-1], (101.1e6, "FM"))


if __name__ == "__main__":
    unittest.main()

"""Behavior tests for the S-Meter panel and the AM/FM Radio Tuner window."""

from __future__ import annotations

import os
import re
import time
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QPoint, QRectF, Qt
    from PyQt6.QtGui import QFontMetricsF
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - PyQt6 missing
    HAS_PYQT6 = False

_SRC = Path(__file__).resolve().parents[1] / "src" / "sdr_module" / "ham" / "gui"
_OWNED = ("signal_meter_widget.py", "radio_tuner.py")
_APP = None


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


class TestNoHardCodedColors(unittest.TestCase):
    """Both widgets take every color from the theme palette."""

    def test_no_hex_or_named_colors(self):
        hex_color = re.compile(r"#[0-9a-fA-F]{3,8}\b")
        named = re.compile(
            r"\b(gray|grey|green|red|orange|blue|yellow|white|black)\b\s*[;\"']"
        )
        rgb_call = re.compile(r"\bQColor\(\s*\d")
        for name in _OWNED:
            text = (_SRC / name).read_text(encoding="utf-8")
            with self.subTest(file=name):
                self.assertIsNone(hex_color.search(text))
                self.assertIsNone(named.search(text))
                self.assertIsNone(rgb_call.search(text))
                self.assertNotIn("setStyleSheet", text)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestAnalogMeterGeometry(unittest.TestCase):
    """The gauge scale is geometrically consistent at any size."""

    def setUp(self):
        _app()
        from sdr_module.ham.gui.signal_meter_widget import AnalogMeterWidget

        self.meter = AnalogMeterWidget()

    def tearDown(self):
        self.meter.deleteLater()

    def test_scale_is_monotonic_and_s9_at_sixty_percent(self):
        m = self.meter
        angles = [m._s_to_angle(s / 2.0) for s in range(0, 39)]
        self.assertTrue(all(a > b for a, b in zip(angles, angles[1:], strict=False)))
        self.assertAlmostEqual(m._s_to_fraction(9.0), 0.6)
        self.assertAlmostEqual(m._s_to_fraction(9.0 + 60.0 / 6.0), 1.0)
        # One S-unit and 10 dB over S9 span the same arc.
        self.assertAlmostEqual(
            m._s_to_fraction(9.0) - m._s_to_fraction(8.0),
            m._s_to_fraction(9.0 + 10.0 / 6.0) - m._s_to_fraction(9.0),
        )

    def test_value_clamps_to_scale(self):
        self.meter.set_value(40.0, -3.0)
        self.assertEqual(self.meter._s_units, 19.0)  # S9+60
        self.assertEqual(self.meter._peak_s_units, 0.0)

    def test_labels_fit_and_never_overlap(self):
        for w, h in ((240, 120), (360, 160), (380, 120), (450, 200), (700, 260)):
            with self.subTest(size=(w, h)):
                self.meter.resize(w, h)
                fm = QFontMetricsF(self.meter._label_font())
                cx, cy, radius = self.meter._geometry(fm)
                labels = self.meter._scale_labels(fm, cx, cy, radius)
                texts = [t for t, _, _ in labels]
                self.assertIn("9", texts)
                self.assertIn("+60", texts)
                face = QRectF(0, 0, w, h)
                boxes = [box for _, box, _ in labels]
                for box in boxes:
                    self.assertTrue(face.contains(box), f"{box} outside {face}")
                for i, a in enumerate(boxes):
                    for b in boxes[i + 1 :]:
                        self.assertFalse(a.intersects(b), f"{a} overlaps {b}")

    def test_paints_in_both_themes(self):
        from sdr_module.gui import themes

        self.meter.resize(360, 160)
        self.meter.set_value(7.0, 12.0)
        for theme in ("light", "dark"):
            themes.apply_theme(_app(), theme)
            self.assertFalse(self.meter.grab().isNull())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestSignalMeterPanel(unittest.TestCase):
    """Signal Report readouts, tones and empty states."""

    def setUp(self):
        _app()
        from sdr_module.ham.gui.signal_meter_widget import SignalMeterPanel

        self.panel = SignalMeterPanel()

    def tearDown(self):
        self.panel._update_timer.stop()
        self.panel.deleteLater()

    def test_empty_state_before_any_samples(self):
        self.assertEqual(self.panel._s_meter_label.text(), "—")
        self.assertEqual(self.panel._rst_label.text(), "—")
        self.assertIn("Start the receiver", self.panel._verbal_label.text())
        self.assertEqual(self.panel._verbal_label.property("role"), "hint")

    def test_readouts_and_tone_follow_strength(self):
        self.panel.update_samples(_tone(0.5))
        self.panel._update_display()
        self.assertTrue(self.panel._s_meter_label.text().startswith("S9+"))
        self.assertIn(self.panel._s_meter_label.property("tone"), ("warning", "danger"))
        self.assertTrue(self.panel._dbm_label.text().endswith("dBm"))
        self.assertTrue(self.panel._peak_label.text().startswith("S"))
        self.assertEqual(self.panel._verbal_label.property("role"), "muted")

    def test_cw_mode_gives_three_digit_rst(self):
        self.panel.update_samples(_tone(0.05))
        self.panel._update_display()
        self.assertEqual(len(self.panel._rst_label.text()), 2)
        self.panel._mode_combo.setCurrentIndex(1)  # CW
        self.assertEqual(len(self.panel._rst_label.text()), 3)

    def test_stale_reading_returns_to_empty_state(self):
        self.panel.update_samples(_tone(0.05))
        self.panel._update_display()
        self.panel._last_reading.timestamp = time.time() - 60
        self.panel._update_display()
        self.assertEqual(self.panel._s_meter_label.text(), "—")
        self.assertFalse(self.panel._analog_meter._active)

    def test_gauge_can_shrink_in_a_short_tab(self):
        # A wrapping label would make the panel height-for-width, which stops
        # the tab's scroll area from shrinking the gauge.
        self.assertFalse(self.panel.layout().hasHeightForWidth())
        gauge = self.panel._analog_meter
        self.assertLess(gauge.minimumSizeHint().height(), gauge.sizeHint().height())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestRadioTuner(unittest.TestCase):
    """AM/FM Radio Tuner window behavior."""

    def setUp(self):
        _app()
        from sdr_module.ham.gui import radio_tuner

        self.rt = radio_tuner
        self.tuner = radio_tuner.RadioTunerWidget(None, 2.4e6)
        self.emitted = []
        self.tuner.frequency_changed.connect(lambda f, b: self.emitted.append((f, b)))

    def tearDown(self):
        self.tuner.close()
        self.tuner.deleteLater()
        _settle()

    def test_window_title_and_resizable(self):
        self.assertEqual(self.tuner.windowTitle(), "AM/FM Radio Tuner")
        self.assertNotEqual(self.tuner.minimumSize(), self.tuner.maximumSize())

    def test_no_dead_controls(self):
        # Tone / balance / power / volume sliders did nothing in the app.
        for name in ("_tone_knob", "_balance_knob", "_power_btn", "_vol_knob"):
            self.assertFalse(hasattr(self.tuner, name), name)

    def test_preset_click_tunes_and_checks_button(self):
        self.tuner._preset_buttons[2].click()
        self.assertEqual(self.tuner.get_frequency(), 97.1e6)
        self.assertEqual(self.emitted[-1], (97.1e6, "FM"))
        checked = [b.isChecked() for b in self.tuner._preset_buttons]
        self.assertEqual(checked, [False, False, True, False, False, False])

    def test_band_switch_restores_each_bands_station(self):
        self.tuner._preset_buttons[1].click()  # 93.3 MHz
        self.tuner._am_btn.click()
        self.assertEqual(self.tuner.get_band(), self.rt.RadioBand.AM)
        self.assertTrue(self.rt.AM_RANGE[0] < self.tuner.get_frequency() < 1700e3)
        self.assertEqual(self.emitted[-1][1], "AM")
        self.tuner._fm_btn.click()
        self.assertEqual(self.tuner.get_frequency(), 93.3e6)
        self.assertTrue(self.tuner._fm_btn.isChecked())
        self.assertFalse(self.tuner._am_btn.isChecked())

    def test_step_buttons_and_keys_move_one_channel(self):
        start = self.tuner.get_frequency()
        self.tuner._seek_up_btn.click()
        self.assertAlmostEqual(self.tuner.get_frequency(), start + 100e3)
        QTest.keyClick(self.tuner._tuning_dial, Qt.Key.Key_Left)
        self.assertAlmostEqual(self.tuner.get_frequency(), start)
        self.tuner._am_btn.click()
        am = self.tuner.get_frequency()
        self.tuner._seek_down_btn.click()
        self.assertAlmostEqual(self.tuner.get_frequency(), am - 10e3)

    def test_dial_click_snaps_to_channel_raster(self):
        dial = self.tuner._tuning_dial
        dial.resize(333, 68)
        QTest.mouseClick(dial, Qt.MouseButton.LeftButton, pos=QPoint(171, 30))
        freq = self.tuner.get_frequency()
        self.assertAlmostEqual(freq / 100e3, round(freq / 100e3), places=6)

    def test_am_dial_labels_use_khz(self):
        self.tuner._am_btn.click()
        dial = self.tuner._tuning_dial
        dial.resize(400, 68)
        self.assertFalse(dial._is_mhz())
        major, minor = dial._tick_steps()
        self.assertGreaterEqual(major, 100e3)
        self.assertLess(minor, major)

    def test_store_preset_gives_feedback(self):
        self.tuner._seek_up_btn.click()
        freq = self.tuner.get_frequency()
        self.tuner.store_preset(4)
        button = self.tuner._preset_buttons[4]
        self.assertEqual(button.get_preset().frequency_hz, freq)
        self.assertTrue(button.isChecked())
        self.assertIn("Stored", self.tuner._hint_label.text())
        self.assertEqual(self.tuner._hint_label.property("tone"), "success")

    def test_empty_preset_does_not_crash(self):
        button = self.tuner._preset_buttons[0]
        button.set_preset(None)
        self.assertIn("Empty", button.text())
        self.tuner._fm_presets[0] = None
        button.click()
        self.assertIn("empty", self.tuner._hint_label.text())
        # Band round trips keep working with an empty slot.
        self.tuner._am_btn.click()
        self.tuner._fm_btn.click()
        self.assertIn("Empty", button.text())

    def test_set_frequency_picks_band_without_emitting(self):
        self.emitted.clear()
        self.tuner.set_frequency(1.2e6)
        self.assertEqual(self.tuner.get_band(), self.rt.RadioBand.AM)
        self.assertEqual(self.tuner.get_frequency(), 1.2e6)
        self.assertEqual(self.emitted, [])

    def test_process_samples_still_demodulates(self):
        audio = self.tuner.process_samples(_tone(0.5))
        self.assertEqual(audio.dtype, np.float32)
        self.tuner.set_muted(True)
        self.assertFalse(np.any(self.tuner.process_samples(_tone(0.5))))

    def test_paints_in_both_themes(self):
        from sdr_module.gui import themes

        self.tuner.set_stereo(True)
        self.tuner.show()
        for theme in ("light", "dark"):
            themes.apply_theme(_app(), theme)
            _settle()
            self.assertFalse(self.tuner.grab().isNull())


if __name__ == "__main__":
    unittest.main()

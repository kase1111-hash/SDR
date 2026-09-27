#!/usr/bin/env python3
"""
Review fixes for the ham radio panels: one watt / dBm format in the QRP
panel, S-meter readouts that agree with each other and fit a short tab, the
SSTV viewer's image conversion and empty state, and the AM/FM tuner's dial
pointer, off-band state and preset menus.

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_integration_ham.py``
"""

import os
import time
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QCoreApplication, QEvent, QPoint, QRect, Qt, QTimer
    from PyQt6.QtGui import QColor, QContextMenuEvent, QFontMetricsF
    from PyQt6.QtTest import QSignalSpy, QTest
    from PyQt6.QtWidgets import QApplication, QFrame, QMenu, QScrollArea

    HAS_PYQT6 = True
    app = QApplication.instance() or QApplication([])
except ImportError:  # pragma: no cover - environment
    HAS_PYQT6 = False

# Height of the Ham Radio tab's page at the 1024x640 minimum window size.
_SHORT_TAB_HEIGHT = 188
_TAB_WIDTH = 376


def settle(n: int = 3) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _tone(amplitude: float, n: int = 4096) -> "np.ndarray":
    t = np.arange(n) / 48000.0
    return (amplitude * np.exp(2j * np.pi * 1000 * t)).astype(np.complex64)


def _near(a: "QColor", b: "QColor", tol: int = 40) -> bool:
    return (
        abs(a.red() - b.red()) + abs(a.green() - b.green()) + abs(a.blue() - b.blue())
        <= tol
    )


# ---------------------------------------------------------------------------
# QRP panel
# ---------------------------------------------------------------------------


class TestQRPFormats(unittest.TestCase):
    def test_format_watts(self):
        from sdr_module.ham.gui.qrp_panel import format_watts

        cases = {
            1.0: "1 W",
            5.0119: "5 W",
            1.2589: "1.3 W",
            12.589: "12.6 W",
            100.0: "100 W",
            2.2: "2.2 W",
            0.5: "500 mW",
            0.1: "100 mW",
            0.001: "1 mW",
            0.0012589: "1.3 mW",
            0.99949: "999 mW",
            0.9996: "1 W",  # not "1000 mW"
            1e-5: "10 µW",
            0.0: "0 W",
        }
        for watts, text in cases.items():
            with self.subTest(watts=watts):
                self.assertEqual(format_watts(watts), text)

    def test_format_dbm(self):
        from sdr_module.ham.gui.qrp_panel import format_dbm

        self.assertEqual(format_dbm(30.0), "+30 dBm")
        self.assertEqual(format_dbm(30.5), "+30.5 dBm")
        self.assertEqual(format_dbm(-3.2), "-3.2 dBm")
        self.assertEqual(format_dbm(0.0), "0 dBm")
        self.assertEqual(format_dbm(-0.01), "0 dBm")

    def test_classification_text_uses_the_same_format(self):
        from sdr_module.ham.gui.qrp_panel import classify_qrp

        self.assertEqual(classify_qrp(0.5)[2], "QRPp: 1 W or less")
        self.assertIn("5 W QRP limit", classify_qrp(4.0, "CW")[2])
        self.assertIn("100 W", classify_qrp(500.0, "SSB")[2])


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestQRPPanelConsistency(unittest.TestCase):
    def setUp(self):
        from sdr_module.ham.gui.qrp_panel import QRPPanel

        self.panel = QRPPanel()
        self.panel.resize(360, 700)
        self.panel.show()
        settle()

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()

    def test_one_watt_and_dbm_format_everywhere(self):
        p = self.panel
        calc = p._amp_calc
        self.assertEqual(p._power_display._watts_label.text(), "1 W")
        self.assertEqual(p._power_display._dbm_label.text(), "+30 dBm")
        self.assertEqual(calc._pa_out.text(), "1 W")
        self.assertEqual(calc._output_label.text(), "1 W (+30 dBm)")
        self.assertEqual(calc._input_watts.text(), "1 mW")
        self.assertEqual(calc._driver_out.text(), "100 mW")
        # The watt spin boxes drop trailing zeros too (not "5.000 W").
        self.assertEqual(p._limit_spin.text(), "5 W")
        self.assertEqual(p._mpw_power_spin.text(), "5 W")
        p._mpw_power_spin.setValue(0.25)
        self.assertEqual(p._mpw_power_spin.text(), "0.25 W")
        self.assertEqual(
            [b.text() for b in p._preset_buttons], ["QRPp 1 W", "CW 5 W", "SSB 10 W"]
        )

    def test_limit_status_and_log_use_the_format(self):
        p = self.panel
        p._preset_buttons[1].click()
        self.assertIn("5 W limit", p._limit_status.text())
        p._log_qso_btn.click()
        self.assertIn("on 5 W", p._mpw_feedback.text())

    def test_typed_watts_still_parse(self):
        spin = self.panel._mpw_power_spin
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin, "2.5")
        QTest.keyClick(spin, Qt.Key.Key_Return)
        self.assertAlmostEqual(spin.value(), 2.5)
        self.assertEqual(spin.text(), "2.5 W")

    def test_scroll_area_is_not_a_tab_stop(self):
        self.assertEqual(self.panel._scroll.focusPolicy(), Qt.FocusPolicy.NoFocus)

    def test_gain_spin_boxes_have_accessible_names(self):
        calc = self.panel._amp_calc
        self.assertEqual(calc._driver_spin.accessibleName(), "Driver gain")
        self.assertEqual(calc._pa_spin.accessibleName(), "PA gain")


# ---------------------------------------------------------------------------
# S-meter
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestSignalMeterReadouts(unittest.TestCase):
    def setUp(self):
        from sdr_module.ham.gui.signal_meter_widget import SignalMeterPanel

        self.panel = SignalMeterPanel()

    def tearDown(self):
        self.panel._update_timer.stop()
        self.panel.deleteLater()

    def test_empty_readouts_use_the_apps_em_dash(self):
        from sdr_module.ham.gui.signal_meter_widget import CompactSignalMeter

        p = self.panel
        for label in (
            p._s_meter_label,
            p._rst_label,
            p._dbm_label,
            p._snr_label,
            p._peak_label,
        ):
            self.assertEqual(label.text(), "—")
        compact = CompactSignalMeter()
        self.assertTrue(compact._bar_label.text().endswith("—"))
        self.assertEqual(compact._rst_label.text(), "RST: —")
        self.assertNotIn("--", compact._bar_label.text() + compact._rst_label.text())
        compact.deleteLater()

    def test_s_meter_peak_hold_and_needle_agree_on_a_steady_tone(self):
        from sdr_module.ham.gui.signal_meter_widget import (
            dbm_to_s_units,
            format_s_units,
        )

        p = self.panel
        for amplitude in (0.5, 0.2, 0.05, 0.01):
            with self.subTest(amplitude=amplitude):
                meter = p.get_meter()
                meter._peak_hold_dbm = -200.0
                p.update_samples(_tone(amplitude))
                p._update_display()
                reading = p._last_reading
                needle = p._analog_meter._s_units
                self.assertEqual(
                    p._s_meter_label.text(),
                    format_s_units(dbm_to_s_units(reading.power_dbm)),
                )
                self.assertEqual(p._s_meter_label.text(), format_s_units(needle))
                # Steady tone: the peak is the current reading, never below it.
                self.assertEqual(p._peak_label.text(), p._s_meter_label.text())

    def test_stale_data_returns_to_em_dashes(self):
        p = self.panel
        p.update_samples(_tone(0.05))
        p._update_display()
        p._last_reading.timestamp = time.time() - 60
        p._update_display()
        self.assertEqual(p._s_meter_label.text(), "—")
        self.assertEqual(p._peak_label.text(), "—")


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestSignalMeterShortTab(unittest.TestCase):
    """At the minimum window size the S-METER / RST row is in view."""

    def setUp(self):
        from sdr_module.ham.gui.signal_meter_widget import SignalMeterPanel

        self.panel = SignalMeterPanel()
        self.area = QScrollArea()
        self.area.setWidgetResizable(True)
        self.area.setFrameShape(QFrame.Shape.NoFrame)
        self.area.setWidget(self.panel)
        self.area.resize(_TAB_WIDTH, _SHORT_TAB_HEIGHT)
        self.area.show()
        self.panel.update_samples(_tone(0.5))
        self.panel._update_display()
        settle(5)

    def tearDown(self):
        self.panel._update_timer.stop()
        self.area.close()
        self.area.deleteLater()

    def test_reading_row_is_visible_without_scrolling(self):
        viewport = self.area.viewport()
        self.assertEqual(self.area.verticalScrollBar().value(), 0)
        for widget in (self.panel._s_meter_label, self.panel._rst_label):
            rect = QRect(widget.mapTo(viewport, QPoint(0, 0)), widget.size())
            self.assertLessEqual(
                rect.bottom(), viewport.height(), f"{widget.text()} is cut off"
            )

    def test_gauge_shrinks_but_keeps_a_wide_scale(self):
        meter = self.panel._analog_meter
        self.assertLess(meter.height(), 100)
        fm = QFontMetricsF(meter._label_font())
        _cx, _cy, radius = meter._geometry(fm)
        self.assertGreaterEqual(
            radius, 0.6 * meter._radius_for_width(meter.width(), fm)
        )

    def test_roomy_tab_keeps_the_full_size_gauge(self):
        self.area.resize(_TAB_WIDTH, 420)
        settle(5)
        meter = self.panel._analog_meter
        self.assertEqual(meter.height(), meter.sizeHint().height())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestCompactGaugeGeometry(unittest.TestCase):
    def test_labels_fit_and_never_overlap_when_short(self):
        from PyQt6.QtCore import QRectF

        from sdr_module.ham.gui.signal_meter_widget import AnalogMeterWidget

        meter = AnalogMeterWidget()
        self.assertEqual(meter.minimumSizeHint().height(), meter.minimumHeight())
        for w, h in ((354, meter.minimumHeight()), (240, 96), (500, 110)):
            with self.subTest(size=(w, h)):
                meter.resize(w, h)
                fm = QFontMetricsF(meter._label_font())
                cx, cy, radius = meter._geometry(fm)
                labels = meter._scale_labels(fm, cx, cy, radius)
                face = QRectF(0, 0, w, h)
                boxes = [box for _, box, _ in labels]
                self.assertIn("+60", [t for t, _, _ in labels])
                for box in boxes:
                    self.assertTrue(face.contains(box), f"{box} outside {face}")
                for i, a in enumerate(boxes):
                    for b in boxes[i + 1 :]:
                        self.assertFalse(a.intersects(b), f"{a} overlaps {b}")
        meter.deleteLater()


# ---------------------------------------------------------------------------
# SSTV viewer
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestSSTVImageConversion(unittest.TestCase):
    def test_single_channel_array_becomes_full_size_rgb(self):
        from sdr_module.ham.gui.sstv_panel import rgb_to_qimage

        gray = np.full((256, 320), 200, dtype=np.uint8)
        image = rgb_to_qimage(gray)
        self.assertEqual((image.width(), image.height()), (320, 256))
        self.assertEqual(image.pixelColor(10, 10).getRgb()[:3], (200, 200, 200))

    def test_rgba_drops_alpha(self):
        from sdr_module.ham.gui.sstv_panel import rgb_to_qimage

        rgba = np.zeros((4, 5, 4), dtype=np.uint8)
        rgba[..., 0] = 255
        image = rgb_to_qimage(rgba)
        self.assertEqual((image.width(), image.height()), (5, 4))
        self.assertEqual(image.pixelColor(0, 0).getRgb()[:3], (255, 0, 0))

    def test_bad_shapes_raise_instead_of_reading_past_the_buffer(self):
        from sdr_module.ham.gui.sstv_panel import ImageDisplayWidget, rgb_to_qimage

        for shape in ((10,), (4, 5, 2), (2, 3, 4, 3)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    rgb_to_qimage(np.zeros(shape, dtype=np.uint8))
        display = ImageDisplayWidget()
        display.set_image(np.zeros((16, 20), dtype=np.uint8))
        self.assertTrue(display.has_image())
        display.deleteLater()


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestSSTVPlaceholder(unittest.TestCase):
    def test_uses_the_shared_plot_placeholder(self):
        from sdr_module.gui import spectrum_widget
        from sdr_module.ham.gui.sstv_panel import ISS_SSTV_HINT, ImageDisplayWidget

        display = ImageDisplayWidget()
        display.resize(320, 256)
        with mock.patch.object(
            spectrum_widget, "draw_placeholder", wraps=spectrum_widget.draw_placeholder
        ) as draw:
            display.grab()
        self.assertTrue(draw.called)
        _painter, rect, title, hint, font, _p = draw.call_args.args
        self.assertEqual(title, "No image yet")
        self.assertFalse(title.endswith("."))
        self.assertEqual(hint, ISS_SSTV_HINT)
        # The helper sizes its title in points: a pixel-sized font would make
        # the title smaller than the hint.
        self.assertGreater(font.pointSizeF(), 0)
        self.assertFalse(font.italic())
        self.assertLessEqual(rect.width(), display.width())
        display.deleteLater()


# ---------------------------------------------------------------------------
# AM/FM radio tuner
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTuningDialPointer(unittest.TestCase):
    def setUp(self):
        from sdr_module.ham.gui.radio_tuner import TuningDial

        self.dial = TuningDial()
        self.dial.resize(360, 68)
        self.dial.set_frequency(99.9e6)  # right beside the "100" label
        self.dial.show()
        settle()

    def tearDown(self):
        self.dial.close()
        self.dial.deleteLater()

    def _pointer_pixels(self, token: str, rows) -> int:
        from sdr_module.gui import themes

        color = themes.get_palette().qcolor(token)
        image = self.dial.grab().toImage()
        x = int(round(self.dial._x_for(self.dial.get_frequency())))
        return sum(
            _near(image.pixelColor(px, py), color)
            for py in rows
            for px in range(x - 1, x + 2)
        )

    def test_pointer_stops_above_the_scale_labels(self):
        h = self.dial.height()
        base_y = h * 0.52
        label_top = int(base_y + 6.0) + 1
        above = range(10, int(base_y))
        labels = range(label_top, h - 2)
        self.assertGreater(self._pointer_pixels("plot_marker", above), 0)
        self.assertEqual(self._pointer_pixels("plot_marker", labels), 0)

    def test_inactive_pointer_is_dimmed(self):
        self.dial.set_active(False)
        settle()
        above = range(10, int(self.dial.height() * 0.52))
        self.assertEqual(self._pointer_pixels("plot_marker", above), 0)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestRadioTunerOffBand(unittest.TestCase):
    """Outside AM/FM the tuner says where the receiver is, not a stale station."""

    def setUp(self):
        from sdr_module.ham.gui.radio_tuner import RadioTunerWidget

        self.tuner = RadioTunerWidget(None, 2.4e6)
        self.tuner.show()
        settle()
        self.tuner.set_frequency(98.7e6)
        self.spy = QSignalSpy(self.tuner.frequency_changed)
        self.tuner.set_frequency(146.52e6)
        settle()

    def tearDown(self):
        self.tuner.close()
        self.tuner.deleteLater()

    def test_off_band_state(self):
        t = self.tuner
        self.assertTrue(t.is_off_band())
        self.assertEqual(len(self.spy), 0)  # following the receiver never emits
        self.assertAlmostEqual(t.get_frequency(), 98.7e6)  # its own station kept
        self.assertFalse(t._fm_btn.isChecked())
        self.assertFalse(t._am_btn.isChecked())
        self.assertFalse(any(b.isChecked() for b in t._preset_buttons))
        self.assertFalse(t._tuning_dial._active)
        self.assertEqual(t._freq_display._digits(), ("146.520", "MHz"))
        self.assertIn("146.520 MHz", t._hint_label.text())
        self.assertEqual(t._hint_label.property("tone"), "info")
        self.assertFalse(t.grab().isNull())

    def test_receiver_back_in_band_leaves_the_state_without_emitting(self):
        t = self.tuner
        t.set_frequency(1010e3)
        self.assertFalse(t.is_off_band())
        self.assertEqual(len(self.spy), 0)
        self.assertTrue(t._am_btn.isChecked())
        self.assertTrue(t._tuning_dial._active)
        self.assertEqual(t._freq_display._digits(), ("1010", "kHz"))
        self.assertNotIn("receiver is on", t._hint_label.text())

    def test_stepping_the_dial_tunes_the_receiver_to_a_station(self):
        t = self.tuner
        QTest.keyClick(t._tuning_dial, Qt.Key.Key_Right)
        self.assertEqual(len(self.spy), 1)
        freq, band = self.spy[0]
        self.assertAlmostEqual(freq, 98.8e6)
        self.assertEqual(band, "FM")
        self.assertFalse(t.is_off_band())
        self.assertTrue(t._fm_btn.isChecked())

    def test_clicking_the_dimmed_pointer_tunes_there(self):
        t = self.tuner
        dial = t._tuning_dial
        x = int(round(dial._x_for(dial.get_frequency())))
        QTest.mouseClick(dial, Qt.MouseButton.LeftButton, pos=QPoint(x, 30))
        self.assertEqual(len(self.spy), 1)
        self.assertAlmostEqual(self.spy[0][0], 98.7e6)
        self.assertFalse(t.is_off_band())

    def test_preset_and_band_buttons_leave_the_state(self):
        t = self.tuner
        t._preset_buttons[1].click()
        self.assertFalse(t.is_off_band())
        self.assertAlmostEqual(self.spy[-1][0], 93.3e6)
        self.assertTrue(t._preset_buttons[1].isChecked())
        t.set_frequency(446.0e6)
        self.assertTrue(t.is_off_band())
        t._am_btn.click()
        self.assertFalse(t.is_off_band())
        self.assertTrue(t._am_btn.isChecked())
        self.assertFalse(t._fm_btn.isChecked())
        self.assertEqual(self.spy[-1][1], "AM")

    def test_feedback_returns_to_the_off_band_hint(self):
        t = self.tuner
        t.store_preset(5)
        self.assertIn("Stored", t._hint_label.text())
        self.assertFalse(t._preset_buttons[5].isChecked())
        t._reset_hint()  # what the hint timer does
        self.assertIn("146.520 MHz", t._hint_label.text())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestRadioTunerCleanup(unittest.TestCase):
    def setUp(self):
        from sdr_module.ham.gui.radio_tuner import RadioTunerWidget

        self.tuner = RadioTunerWidget(None, 2.4e6)
        self.tuner.show()
        settle()

    def tearDown(self):
        self.tuner.close()
        self.tuner.deleteLater()

    def test_preset_menu_is_freed_after_each_right_click(self):
        button = self.tuner._preset_buttons[0]

        def close_menu():
            menu = QApplication.activePopupWidget()
            if menu is not None:
                menu.close()

        for _ in range(3):
            QTimer.singleShot(0, close_menu)
            pos = QPoint(5, 5)
            event = QContextMenuEvent(
                QContextMenuEvent.Reason.Mouse, pos, button.mapToGlobal(pos)
            )
            button.contextMenuEvent(event)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        self.assertEqual(button.findChildren(QMenu), [])

    def test_dead_members_are_gone(self):
        from sdr_module.ham.gui.radio_tuner import PresetButton, RadioTunerWidget

        self.assertFalse(hasattr(RadioTunerWidget, "audio_output"))
        self.assertFalse(hasattr(self.tuner, "_powered"))
        self.assertFalse(hasattr(PresetButton, "_format_freq"))
        # Audio processing still works without the old power flag.
        audio = self.tuner.process_samples(_tone(0.5))
        self.assertEqual(audio.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()

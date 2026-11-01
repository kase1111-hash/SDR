"""The S-meter tells the truth.

hm:HAM-05 / ux:UX-B04 (a real noise floor: an empty channel is not "S5,
RST 55, SNR 23" in green), hm:HAM-06 (peak hold decays and resets),
hm:HAM-12 / hm:UX-08 (empty state on stop and with no device), hm:HAM-11
(compact layout at the window's minimum size), fg:FG-G24 / hm:GAP-01
(calibration per device type, dBFS until calibrated), hm:GAP-02 (copy the
report), hm:GAP-03 (level history) and hm:UX-07 (report format follows the
demodulation).

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_smeter.py``
"""

from __future__ import annotations

import json
import os
import time
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtCore import QPoint, QRect, Qt
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtTest import QSignalSpy, QTest
    from PyQt6.QtWidgets import QApplication, QFrame, QScrollArea

from sdr_module.ham import signal_meter as sm  # noqa: E402
from sdr_module.ham.signal_meter import SignalMeter  # noqa: E402

RNG = np.random.default_rng(11)


def noise(n: int = 4096, sigma: float = 1e-3) -> np.ndarray:
    return ((RNG.standard_normal(n) + 1j * RNG.standard_normal(n)) * sigma).astype(
        np.complex64
    )


def tone(amplitude: float, n: int = 4096) -> np.ndarray:
    t = np.arange(n) / 48000.0
    return (amplitude * np.exp(2j * np.pi * 1000 * t)).astype(np.complex64)


def power_dbfs(x: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(np.abs(x) ** 2)))


class FakeClock:
    """Patches time.time in the signal meter modules."""

    def __init__(self, start: float = 1000.0):
        self.now = start

    def __call__(self) -> float:
        return self.now


def settle(n: int = 3) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _ham_radio() -> bool:
    from sdr_module.gui.main_window import HAS_HAM_RADIO

    return HAS_HAM_RADIO


# --------------------------------------------------------------------------
# Backend
# --------------------------------------------------------------------------


class TestNoiseReference(unittest.TestCase):
    def test_empty_channel_reads_noise_not_a_signal(self):
        """hm:HAM-05: the old fixed -120 dBm floor made noise 'S5, 55'."""
        meter = SignalMeter()
        hiss = noise()
        for _ in range(12):
            reading = meter.update(hiss, power_dbfs(hiss))
        self.assertFalse(reading.signal_present)
        self.assertEqual(reading.readability, 1)
        self.assertLess(reading.snr_db, 1.0)
        self.assertAlmostEqual(reading.noise_dbm, reading.power_dbm, delta=0.5)

    def test_signal_over_the_noise(self):
        meter = SignalMeter()
        hiss = noise()
        floor = power_dbfs(hiss)
        for _ in range(12):
            reading = meter.update(tone(0.03) + hiss, floor)
        expected = 10 * np.log10(0.03**2) - floor  # about 26 dB
        self.assertAlmostEqual(reading.snr_db, expected, delta=0.5)
        self.assertTrue(reading.signal_present)
        self.assertEqual(reading.readability, 5)
        self.assertEqual(meter.get_rst()[0], "5")

    def test_spectral_estimate_without_a_reference(self):
        meter = SignalMeter()
        hiss = noise()
        for _ in range(12):
            reading = meter.update(hiss)
        self.assertLess(reading.snr_db, 1.0)
        self.assertAlmostEqual(reading.noise_dbm + 30, power_dbfs(hiss), delta=1.0)
        for _ in range(12):
            reading = meter.update(tone(0.05) + hiss)
        self.assertGreater(reading.snr_db, 25)


class TestPeakHold(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        patcher = mock.patch.object(sm.time, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.meter = SignalMeter()
        self.meter._settle_until = 0.0

    def run_for(self, seconds: float, samples: np.ndarray, step: float = 0.1):
        reading = None
        for _ in range(int(round(seconds / step))):
            self.clock.now += step
            reading = self.meter.update(samples)
        return reading

    def test_holds_two_seconds_then_falls_twenty_db_a_second(self):
        """hm:HAM-06: it fell 1 dB per 2 s (S9+15 twenty seconds later)."""
        strong = self.run_for(1.0, tone(0.5))
        peak = strong.peak_hold_dbm
        self.meter._avg_window.clear()
        held = self.run_for(1.5, tone(0.005))
        self.assertAlmostEqual(held.peak_hold_dbm, peak, places=6)
        falling = self.run_for(1.0, tone(0.005))
        self.assertAlmostEqual(falling.peak_hold_dbm, peak - 10.0, delta=2.5)
        floor = self.run_for(3.0, tone(0.005))
        self.assertAlmostEqual(floor.peak_hold_dbm, floor.power_dbm, places=6)

    def test_reset_starts_afresh(self):
        self.run_for(1.0, tone(0.5))
        self.meter.reset_peak()
        self.assertTrue(self.meter.settling)
        # Old-channel samples still arriving just after the retune...
        self.run_for(0.2, tone(0.5))
        self.clock.now += 0.4
        # ... are neither held nor averaged once the new level arrives.
        reading = self.meter.update(tone(0.005))
        self.assertFalse(self.meter.settling)
        self.assertAlmostEqual(reading.power_dbm, -76.0, delta=0.05)
        self.assertAlmostEqual(reading.peak_hold_dbm, reading.power_dbm, places=6)

    def test_old_channel_at_the_start_of_the_first_block_is_not_a_signal(self):
        """ux:UX-B04: the first block after a retune starts with the channel
        filter's memory of the old station. Averaged in, it made an empty
        channel read 'S7, RST 37, Three and seven' in green for several
        frames."""
        hiss = noise(16000)
        floor = power_dbfs(hiss)
        for _ in range(12):
            self.clock.now += 0.033
            self.meter.update(tone(0.3, 16000) + hiss, floor)
        self.meter.reset_peak()  # retuned to an empty channel
        first = hiss.copy()
        first[:800] += tone(0.3, 800)  # the old station's last samples
        blocks = [first] + [noise(16000) for _ in range(8)]
        for block in blocks:
            self.clock.now += 0.033
            reading = self.meter.update(block, floor)
            self.assertFalse(reading.signal_present)
            self.assertEqual(reading.readability, 1)
            self.assertLess(reading.power_dbfs, floor + 1.0)


class TestCalibration(unittest.TestCase):
    def test_uncalibrated_is_dbfs_minus_30(self):
        meter = SignalMeter()
        reading = meter.update(tone(0.1))
        self.assertFalse(reading.calibrated)
        self.assertAlmostEqual(reading.power_dbfs, -20.0, places=3)
        self.assertAlmostEqual(reading.power_dbm, -50.0, places=3)

    def test_calibrate_to_a_known_level_and_follow_the_gain(self):
        meter = SignalMeter()
        meter.set_gain(20.0)
        meter.update(tone(0.1))
        offset = meter.offset_for_level(-73.0)
        meter.set_calibration(offset, reference_gain_db=20.0)
        self.assertAlmostEqual(meter.update(tone(0.1)).power_dbm, -73.0, places=3)
        # 10 dB more gain: the samples are 10 dB stronger, the dBm reading
        # stays where it was.
        meter.set_gain(30.0)
        meter._avg_window.clear()
        reading = meter.update(tone(0.1 * 10 ** (10 / 20)))
        self.assertAlmostEqual(reading.power_dbm, -73.0, places=3)
        self.assertAlmostEqual(reading.power_dbfs, -10.0, places=3)
        meter.clear_calibration()
        self.assertFalse(meter.is_calibrated)

    def test_log_report(self):
        meter = SignalMeter()
        for _ in range(3):
            meter.update(tone(0.05) + noise())
        text = meter.get_log_report(14.25e6, when=0.0)
        self.assertEqual(text, "RST 59 · S9+20 · 14.250 MHz · 00:00 UTC")
        meter.reset_peak()
        for _ in range(3):
            meter.update(noise())
        self.assertTrue(meter.get_log_report().startswith("No signal above the noise"))


# --------------------------------------------------------------------------
# Panel
# --------------------------------------------------------------------------


class _PanelCase(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.ham.gui.signal_meter_widget import SignalMeterPanel

        self.panel = SignalMeterPanel()
        self.panel._update_timer.stop()

    def tearDown(self):
        if hasattr(self, "panel"):
            self.panel.deleteLater()
            settle()

    def show_reading(self, samples, noise_dbfs=None, times=3):
        for _ in range(times):
            self.panel.update_samples(samples, noise_dbfs)
        self.panel._update_display()


class TestPanelStates(_PanelCase):
    def test_stop_empties_at_once(self):
        """hm:HAM-12: the reading stayed for 3 s after Stop."""
        self.panel.set_receiving(True)
        self.show_reading(tone(0.5))
        self.assertTrue(self.panel._s_meter_label.text().startswith("S9"))
        self.panel.set_receiving(False)
        self.assertEqual(self.panel._s_meter_label.text(), "—")
        self.assertEqual(self.panel._peak_label.text(), "—")
        self.assertIn("Receiver stopped", self.panel._verbal_label.text())
        self.assertEqual(
            self.panel._analog_meter.placeholder(),
            ("Receiver stopped", "Press Start (Space) to measure."),
        )
        self.assertIsNone(self.panel._last_reading)
        self.assertFalse(self.panel._analog_meter._active)

    def test_empty_gauge_fades_the_dial_without_its_numbers(self):
        """The empty-state message box cut through the scale's numbers
        ('+40' read as '0', half a '3')."""
        meter = self.panel._analog_meter
        meter.resize(390, 140)
        self.panel.set_receiving(False, has_device=False)
        with mock.patch.object(
            meter, "_scale_labels", wraps=meter._scale_labels
        ) as labels:
            self.assertFalse(meter.grab().isNull())
            self.assertFalse(labels.called)
            self.panel.set_receiving(True)
            self.show_reading(tone(0.05) + noise())
            self.assertFalse(meter.grab().isNull())
            self.assertTrue(labels.called)

    def test_compact_peak_reading_resets_on_click(self):
        """hm:UX-08: in the compact layout (what 1366x768 and 1024x640 get)
        the peak is on the summary line, whose tooltip said 'Click to reset',
        but clicking it did nothing."""
        p = self.panel
        p.set_compact(True)
        self.show_reading(tone(0.5))
        self.show_reading(tone(0.005), times=10)
        self.assertIn("peak S9+", p._summary_label.text())
        self.assertIn("Click to reset", p._summary_label.toolTip())
        QTest.mouseClick(p._summary_label, Qt.MouseButton.LeftButton)
        self.show_reading(tone(0.005), times=1)
        self.assertNotIn("peak S9+", p._summary_label.text())

    def test_peak_hold_tooltips_say_the_history_resets_too(self):
        tip = self.panel._peak_label.toolTip()
        self.assertIn("Click to reset", tip)
        self.assertIn("level history", tip)
        self.assertIn("peak hold", self.panel._sparkline.toolTip())

    def test_context_menu_is_freed_after_use(self):
        from PyQt6.QtCore import QCoreApplication, QEvent
        from PyQt6.QtWidgets import QMenu

        before = len(self.panel.findChildren(QMenu))
        with mock.patch.object(QMenu, "exec", lambda *args, **kwargs: None):
            for _ in range(3):
                self.panel._s_meter_label.customContextMenuRequested.emit(QPoint(2, 2))
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        self.assertEqual(len(self.panel.findChildren(QMenu)), before)

    def test_calibration_is_on_the_right_click_menu(self):
        """hm:HAM-11 / hm:GAP-01: at the window's minimum size the compact
        layout hides the Copy / Calibrate row; Copy stayed on the readouts'
        right-click menu but Calibrate could not be reached."""
        from PyQt6.QtWidgets import QMenu

        shown = []

        def fake_exec(menu, *args, **kwargs):
            shown.append([action.text() for action in menu.actions()])

        self.panel.set_compact(True)
        with mock.patch.object(QMenu, "exec", fake_exec):
            self.panel._summary_label.customContextMenuRequested.emit(QPoint(2, 2))
        self.assertIn("Copy Report", shown[0])
        self.assertIn("Calibrate", shown[0])
        self.assertIn("calibrate", self.panel._summary_label.toolTip())

    def test_no_device_says_what_to_do(self):
        """hm:UX-08."""
        self.panel.set_receiving(False, has_device=False)
        self.assertEqual(
            self.panel._analog_meter.placeholder(),
            ("No receiver", "Device › Use Demo Device or Connect…"),
        )
        self.assertEqual(
            self.panel._summary_label.text(),
            "No receiver: Device › Use Demo Device or Connect…",
        )
        self.assertEqual(self.panel._verbal_label.text(), "No receiver connected.")

    def test_noise_is_muted_without_a_report(self):
        hiss = noise()
        self.show_reading(hiss, power_dbfs(hiss), times=12)
        p = self.panel
        self.assertEqual(p._rst_label.text(), "—")
        self.assertEqual(p._s_meter_label.property("tone"), "muted")
        self.assertEqual(p._verbal_label.text(), "No signal above the noise")
        self.assertEqual(p._snr_label.text(), "0 dB")

    def test_clicking_peak_hold_resets_it(self):
        self.show_reading(tone(0.5))
        self.show_reading(tone(0.005), times=10)
        self.assertTrue(self.panel._peak_label.text().startswith("S9+"))
        QTest.mouseClick(self.panel._peak_label, Qt.MouseButton.LeftButton)
        self.show_reading(tone(0.005), times=1)
        self.assertFalse(self.panel._peak_label.text().startswith("S9+"))
        self.assertEqual(
            self.panel._peak_label.text(), self.panel._s_meter_label.text()
        )
        self.assertIn("Click to reset", self.panel._peak_label.toolTip())


class TestReportMode(_PanelCase):
    def test_auto_follows_the_demodulation(self):
        """hm:UX-07: CW gives a 3-digit RST without touching the combo."""
        combo = self.panel._mode_combo
        self.assertEqual(combo.currentText(), "Auto (follow demodulation)")
        self.show_reading(tone(0.05))
        self.assertEqual(len(self.panel._rst_label.text()), 2)
        self.assertEqual(combo.face_text(), "Auto · Phone")
        self.panel.set_demod_mode("CW")
        self.assertEqual(len(self.panel._rst_label.text()), 3)
        self.assertEqual(combo.face_text(), "Auto · CW")
        self.panel.set_demod_mode("None (I/Q)")
        self.assertEqual(combo.face_text(), "Auto · Digital")
        self.panel.set_demod_mode("CW")
        combo.setCurrentIndex(combo.findText("Phone"))  # manual override
        self.assertEqual(len(self.panel._rst_label.text()), 2)
        self.assertEqual(combo.face_text(), "Phone")


class TestUnitsAndCalibration(_PanelCase):
    def test_dbfs_until_calibrated(self):
        p = self.panel
        changed = QSignalSpy(p.calibration_changed)
        self.show_reading(tone(0.1))
        self.assertEqual(p._dbm_label.text(), "-20 dBFS")
        self.assertEqual(p._power_caption.text(), "POWER (UNCAL.)")
        self.assertEqual(p._cal_btn.text(), "Calibrate")
        self.assertIn("~dBm", p._analog_meter.toolTip())
        self.assertTrue(p.calibrate_to(-73.0))
        self.show_reading(tone(0.1))
        self.assertEqual(p._dbm_label.text(), "-73 dBm")
        self.assertEqual(p._s_meter_label.text(), "S9")
        self.assertEqual(p._power_caption.text(), "POWER")
        self.assertEqual(p._cal_btn.text(), "Calibrated")
        self.assertAlmostEqual(changed[-1][0], -23.0, places=3)  # -50 -> -73
        p.clear_calibration()
        self.assertIsNone(changed[-1][0])
        self.show_reading(tone(0.1))
        self.assertEqual(p._dbm_label.text(), "-20 dBFS")

    def test_offset_spin_box_applies(self):
        p = self.panel
        self.show_reading(tone(0.1))
        p._sync_cal_menu()
        p._cal_spin.setValue(-10.0)
        p._on_cal_spin_done()
        self.assertTrue(p.is_calibrated())
        self.show_reading(tone(0.1))
        self.assertEqual(p._dbm_label.text(), "-60 dBm")

    def test_calibrating_needs_a_reading(self):
        self.assertFalse(self.panel.calibrate_to(-73.0))


class TestCopyReport(_PanelCase):
    def test_copy_report_rst_and_contest(self):
        """hm:GAP-02."""
        p = self.panel
        copied = QSignalSpy(p.report_copied)
        self.assertIsNone(p.copy_report())
        self.assertIn("Nothing to copy", copied[-1][0])
        p.set_frequency_source(lambda: 145.8e6)
        self.show_reading(tone(0.05) + noise())
        text = p.copy_report()
        # The S reading shown (S9+17), not rounded to S9+20 (HAM-E2E-4).
        self.assertRegex(text, r"^RST 59 · S9\+17 · 145\.800 MHz · \d\d:\d\d UTC$")
        self.assertEqual(QGuiApplication.clipboard().text(), text)
        self.assertEqual(copied[-1][0], f"Copied: {text}")
        self.assertEqual(p.copy_rst(), "59")
        p.set_demod_mode("CW")
        self.assertEqual(p.copy_contest_report(), "599")
        self.assertEqual(
            [a.text() for a in p._copy_menu().actions()],
            ["Copy Report", "Copy RST", "Copy Contest Report (59/599)"],
        )


class TestLevelHistory(_PanelCase):
    def test_history_shows_min_avg_max_and_clears(self):
        """hm:GAP-03."""
        from sdr_module.ham.gui import signal_meter_widget as smw

        clock = FakeClock()
        with (
            mock.patch.object(smw.time, "time", clock),
            mock.patch.object(sm.time, "time", clock),
        ):
            self.panel._meter._settle_until = 0.0
            for amplitude in (0.5, 0.05, 0.005):
                for _ in range(5):
                    clock.now += 0.1
                    self.panel.update_samples(tone(amplitude))
                    self.panel._update_display()
            values = self.panel._sparkline.values()
            self.assertEqual(len(values), 15)
            self.assertRegex(
                self.panel._stats_label.text(),
                r"^min S[\d+]+ · avg S[\d+]+ · max S9\+\d+$",
            )
            self.assertIn("dBm", self.panel._stats_label.toolTip())
            self.panel.reset_peak()  # what a retune does
            self.assertEqual(self.panel._sparkline.values(), [])
            self.assertEqual(self.panel._stats_label.text(), "—")
            # Over 30 s, old points fall off.
            for _ in range(400):
                clock.now += 0.1
                self.panel.update_samples(tone(0.05))
                self.panel._update_display()
            self.assertLessEqual(len(self.panel._sparkline.values()), 301)

    def test_gauge_empty_state_paints_in_both_themes(self):
        from sdr_module.gui import spectrum_widget, themes

        meter = self.panel._analog_meter
        meter.resize(360, 120)
        self.panel.set_receiving(False, has_device=False)
        saved = themes._current_theme
        try:
            with mock.patch.object(
                spectrum_widget,
                "draw_placeholder",
                wraps=spectrum_widget.draw_placeholder,
            ) as draw:
                for name in ("dark", "light"):
                    themes._current_theme = name
                    self.assertFalse(meter.grab().isNull())
            self.assertEqual(draw.call_args.args[2], "No receiver")
        finally:
            themes._current_theme = saved

    def test_sparkline_paints_in_both_themes(self):
        from sdr_module.gui import themes
        from sdr_module.ham.gui.signal_meter_widget import LevelSparkline

        spark = LevelSparkline()
        spark.resize(200, 24)
        for i in range(30):
            spark.add(100.0 + i * 0.1, -80.0 + 10 * np.sin(i / 3), -95.0)
        saved = themes._current_theme
        try:
            for name in ("dark", "light"):
                themes._current_theme = name
                self.assertFalse(spark.grab().isNull())
        finally:
            themes._current_theme = saved
            spark.deleteLater()


class TestCompactLayout(_PanelCase):
    def area_at(self, height):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.Shape.NoFrame)
        area.setWidget(self.panel)
        area.resize(376, height)
        area.show()
        settle(6)
        self.addCleanup(area.deleteLater)
        self.addCleanup(area.takeWidget)
        return area

    def visible_in(self, area, widget) -> bool:
        viewport = area.viewport()
        rect = QRect(widget.mapTo(viewport, QPoint(0, 0)), widget.size())
        return not widget.isHidden() and rect.bottom() <= viewport.height()

    def test_shortest_tab_keeps_every_reading_in_view(self):
        self.show_reading(tone(0.05) + noise())
        area = self.area_at(157)
        settle(4)
        p = self.panel
        self.assertTrue(p.is_compact())
        self.assertEqual(area.verticalScrollBar().maximum(), 0)
        for widget in (p._s_meter_label, p._rst_label, p._summary_label, p._mode_combo):
            self.assertTrue(self.visible_in(area, widget))
        self.assertRegex(p._summary_label.text(), r"dBFS · SNR \d+ dB · peak S")
        self.assertTrue(p._rst_label.text().startswith("RST "))

    def test_rows_come_back_as_room_allows(self):
        self.show_reading(tone(0.05) + noise())
        area = self.area_at(250)
        p = self.panel
        self.assertTrue(p.is_compact())
        self.assertTrue(self.visible_in(area, p._copy_btn))
        self.assertTrue(self.visible_in(area, p._sparkline))
        area.resize(376, 420)
        settle(6)
        self.assertFalse(p.is_compact())
        self.assertTrue(self.visible_in(area, p._peak_label))
        self.assertEqual(p._rst_label.text(), "59")


# --------------------------------------------------------------------------
# Real window
# --------------------------------------------------------------------------


class TestSignalMeterInTheWindow(_WindowTestCase):
    demo = True

    def setUp(self):
        super().setUp()
        if not _ham_radio():
            self.skipTest("ham radio panels not installed")
        self.panel = self.win._signal_meter_panel
        self.panel._update_timer.stop()
        # Every display frame reads one 30 Hz frame of demo samples, and the
        # demo never skips ahead to the wall clock, however long the test
        # takes. (Its signals come and go on its clock: the carrier at
        # 100.715 MHz, at the edge of a 200 kHz channel on 100.6 MHz, is on
        # from 4.5 s. On a loaded machine the stream had got there and the
        # empty channel read 2.5 dB over its noise.)
        from sdr_module.gui import device_dialog

        patcher = mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.win._pacer.demo_request = lambda rate, now: int(rate / 30)

    def settle_on(self, freq, mode="FM", deviation="75 kHz", bandwidth="200 kHz"):
        self.win.set_frequency(freq)
        self.win._set_demod_mode(mode, deviation, bandwidth)
        for _ in range(12):
            self.win._update_display()
        self.panel._update_display()

    def test_empty_channel_versus_strong_station(self):
        """hm:HAM-05 / ux:UX-B04: 100.6 MHz (nothing) read 'S5, RST 55,
        Five and five, SNR 23 dB' in green; 100.1 MHz had SNR 72 = power +
        120."""
        p = self.panel
        self.settle_on(100.6e6)
        self.assertEqual(p._rst_label.text(), "—")
        self.assertEqual(p._s_meter_label.property("tone"), "muted")
        self.assertEqual(p._verbal_label.text(), "No signal above the noise")
        self.assertLessEqual(p._last_reading.snr_db, 3.0)
        self.settle_on(100.1e6)
        self.assertEqual(p._rst_label.text(), "59")
        self.assertIn(p._s_meter_label.property("tone"), ("warning", "danger"))
        # Channel SNR (200 kHz of noise), not "power + 120".
        self.assertGreater(p._last_reading.snr_db, 40)
        self.assertLess(p._last_reading.snr_db, 60)

    def test_retune_from_a_station_never_flashes_a_report(self):
        """ux:UX-B04: just after retuning from the S9+25 station to an empty
        channel, the meter read 'S6-S8, RST 26-59' in green for a few
        frames (the old station in the channel filter's memory)."""
        p = self.panel
        self.settle_on(100.1e6)
        self.assertEqual(p._rst_label.text(), "59")
        self.win.set_frequency(100.6e6)
        for frame in range(12):
            self.win._update_display()
            p._update_display()
            reading = p._last_reading
            self.assertFalse(reading.signal_present, f"frame {frame}")
            self.assertEqual(p._rst_label.text(), "—", f"frame {frame}")
            self.assertEqual(p._s_meter_label.property("tone"), "muted")

    def test_noise_reference_matches_an_empty_channel(self):
        for freq, bandwidth in ((100.6e6, "200 kHz"), (145.5e6, "25 kHz")):
            self.settle_on(freq, bandwidth=bandwidth)
            reading = self.panel._last_reading
            self.assertAlmostEqual(reading.noise_dbm, reading.power_dbm, delta=1.5)

    def test_retune_mode_change_and_stop_reset_the_peak(self):
        """hm:HAM-06: the peak stayed at S9+24 for seconds after retuning."""
        p = self.panel
        self.settle_on(100.1e6)
        self.assertTrue(p._peak_label.text().startswith("S9+"))
        self.settle_on(100.6e6)
        self.assertFalse(p._peak_label.text().startswith("S9"))
        self.settle_on(100.1e6)
        with mock.patch.object(p, "reset_peak", wraps=p.reset_peak) as reset:
            self.win._set_demod_mode("AM", None, "10 kHz")
            self.assertTrue(reset.called)
        self.win._stop_acquisition()
        self.assertEqual(p._peak_label.text(), "—")
        self.assertIn("Receiver stopped", p._verbal_label.text())

    def test_no_device_message(self):
        self.win._disconnect_device()
        self.assertEqual(
            self.panel._analog_meter.placeholder(),
            ("No receiver", "Device › Use Demo Device or Connect…"),
        )

    def test_calibration_is_kept_per_device_type(self):
        """fg:FG-G24 / hm:GAP-01."""
        p = self.panel
        self.settle_on(100.1e6)
        self.assertTrue(p.calibrate_to(-50.0))
        saved = json.loads(self.store.values["ham/smeter_cal/demo"])
        self.assertAlmostEqual(saved["offset_db"], p._meter.calibration_offset_db)
        self.assertIn("calibrated", self.win._message_label.text())
        # Another session with the same settings starts calibrated.
        from sdr_module.gui import main_window

        other = main_window.SDRMainWindow(demo_mode=True)
        try:
            for timer in (
                other._display_timer,
                other._status_timer,
                other._hotplug_timer,
            ):
                timer.stop()
            self.assertTrue(other._signal_meter_panel.is_calibrated())
            self.assertAlmostEqual(
                other._signal_meter_panel._meter.calibration_offset_db,
                saved["offset_db"],
            )
        finally:
            other._stop_acquisition()
            other.close()
            other.deleteLater()
        p.clear_calibration()
        self.assertEqual(self.store.values["ham/smeter_cal/demo"], "")

    def test_calibrated_reading_does_not_move_with_the_gain(self):
        """fg:FG-G24: the same station read S7 at 10 dB gain, S9+20 at 40."""
        p = self.panel
        self.win._control_panel.set_agc_enabled(False)
        self.win.set_gain(20)
        self.settle_on(100.1e6)
        p.calibrate_to(-50.0)
        self.settle_on(100.1e6)
        before = p._last_reading.power_dbm
        self.win.set_gain(30)
        self.settle_on(100.1e6)
        after = p._last_reading
        self.assertAlmostEqual(after.power_dbm, before, delta=1.5)
        self.assertGreater(
            after.power_dbfs - (before - p._meter._dbfs_to_dbm_offset() - 10), -1.5
        )

    def test_auto_report_mode_follows_the_window(self):
        self.win._set_demod_mode("CW")
        self.assertEqual(self.panel._mode_combo.face_text(), "Auto · CW")
        self.win._set_demod_mode("USB")
        self.assertEqual(self.panel._mode_combo.face_text(), "Auto · Phone")

    def test_copied_report_names_the_frequency(self):
        self.settle_on(100.1e6)
        text = self.panel.copy_report()
        self.assertIn("100.100 MHz", text)
        self.assertIn("Copied", self.win._message_label.text())

    def test_readings_fit_the_minimum_window(self):
        """hm:HAM-11: at 1024x640 (after 1366x768) the S-METER and RST values
        were cut off in the S-Meter tab."""
        self.win.resize(1366, 768)
        self.settle(5)
        self.win.resize(1024, 640)
        self.win._show_panel("s_meter")
        self.settle(8)
        self.settle_on(100.1e6)
        self.settle(4)
        page = self.win._panel_pages["s_meter"]
        viewport = page.viewport()
        p = self.panel
        self.assertEqual(page.verticalScrollBar().value(), 0)
        for widget in (p._s_meter_label, p._rst_label, p._mode_combo):
            rect = QRect(widget.mapTo(viewport, QPoint(0, 0)), widget.size())
            self.assertLessEqual(rect.bottom(), viewport.height(), widget.objectName())
        reading_line = p._summary_label if p.is_compact() else p._peak_label
        rect = QRect(reading_line.mapTo(viewport, QPoint(0, 0)), reading_line.size())
        self.assertLessEqual(rect.bottom(), viewport.height())


class TestNoiseEstimator(unittest.TestCase):
    def test_busy_band_does_not_raise_the_floor(self):
        from sdr_module.gui.main_window import _noise_per_bin

        rng = np.random.default_rng(3)
        bins = rng.exponential(1e-9, 2000)
        busy = bins.copy()
        busy[:600] *= 1e3  # stations in 30% of the bins
        truth = 10 * np.log10(1e-9)
        self.assertAlmostEqual(_noise_per_bin(10 * np.log10(bins)), truth, delta=0.3)
        self.assertAlmostEqual(_noise_per_bin(10 * np.log10(busy)), truth, delta=0.6)
        plain = 10 * np.log10(np.median(busy) / np.log(2))
        self.assertGreater(plain - truth, 2.0)  # what the plain median read

    def test_estimate_is_cheap(self):
        from sdr_module.gui.main_window import _noise_per_bin

        spectrum = 10 * np.log10(np.random.default_rng(1).exponential(1e-9, 1600))
        started = time.perf_counter()
        for _ in range(100):
            _noise_per_bin(spectrum)
        self.assertLess((time.perf_counter() - started) / 100, 2e-3)

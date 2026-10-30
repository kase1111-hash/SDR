#!/usr/bin/env python3
"""
Integration tests for the dialogs: keyboard use (no Tab traps, Enter does
what the focused list implies), the scanner's reported frequencies and its
error handling, the demo device's automatic gain and band plan, and the
welcome screen's hardware and driver states.

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_integration_dialogs.py``
"""

import logging
import os
import re
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
    from PyQt6.QtCore import QCoreApplication, QEvent, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QAbstractButton, QApplication, QLabel, QWidget


def _settle(n=3):
    for _ in range(n):
        QApplication.processEvents()


def _flush_deletes():
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def _tab_stops(dialog, start, count=12, key=None):
    """The widgets Tab (or ``key``) visits, starting from ``start``."""
    key = key or Qt.Key.Key_Tab
    dialog.show()
    dialog.activateWindow()
    _settle()
    start.setFocus()
    _settle()
    seen = []
    for _ in range(count):
        widget = QApplication.focusWidget()
        seen.append(widget)
        QTest.keyClick(widget, key)
        _settle(1)
    return seen


def _button_texts(widgets):
    return [
        w.text().replace("&", "") for w in widgets if isinstance(w, QAbstractButton)
    ]


class _DialogCase(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        self._dialogs = []

    def keep(self, dialog):
        self._dialogs.append(dialog)
        return dialog

    def tearDown(self):
        for dialog in reversed(self._dialogs):
            stop = getattr(dialog, "_stop_worker", None)
            if callable(stop):
                stop()
            dialog.close()
            dialog.deleteLater()
        _flush_deletes()


# --------------------------------------------------------------------------- #
# Keyboard
# --------------------------------------------------------------------------- #
class TestDeviceDialogKeyboard(_DialogCase):
    def test_tab_leaves_the_device_list_and_reaches_every_button(self):
        from sdr_module.gui.device_dialog import DeviceDialog

        dialog = self.keep(DeviceDialog())
        table = dialog._device_table
        self.assertFalse(table.tabKeyNavigation())
        self.assertEqual(table.accessibleName(), "Available devices")
        stops = _tab_stops(dialog, table, 6)
        self.assertIs(stops[1], dialog._rate_combo)
        # Every button once, in the style's layout order. (With Connect as the
        # default, the button box's focus proxy made Tab skip the buttons
        # laid out before it, such as Refresh.)
        self.assertEqual(
            sorted(_button_texts(stops[2:5])), ["Cancel", "Connect", "Refresh"]
        )
        self.assertIs(stops[5], table)

    def test_settings_have_mnemonics(self):
        from sdr_module.gui.device_dialog import DeviceDialog

        dialog = self.keep(DeviceDialog())
        form = dialog._settings_form
        for field in (dialog._rate_combo, dialog._ppm_spin, dialog._direct_combo):
            label = form.labelForField(field)
            self.assertIn("&", label.text())
            self.assertIs(label.buddy(), field)


class TestErrorLogKeyboard(_DialogCase):
    def setUp(self):
        super().setUp()
        from sdr_module.gui import error_log_dialog

        self.mod = error_log_dialog
        self.mod._HISTORY.clear_history()
        self.logger = logging.getLogger("sdr_module.test_integration_dialogs")
        self.logger.addHandler(self.mod._HISTORY)
        self.logger.propagate = False

    def tearDown(self):
        self.logger.removeHandler(self.mod._HISTORY)
        self.mod._HISTORY.clear_history()
        super().tearDown()

    def test_tab_reaches_the_buttons_past_a_full_table(self):
        for i in range(3):
            self.logger.warning(f"warning {i}")
        dialog = self.keep(self.mod.ErrorLogDialog())
        self.assertFalse(dialog._table.tabKeyNavigation())
        self.assertEqual(dialog._table.accessibleName(), "Logged warnings and errors")
        self.assertEqual(dialog._details.accessibleName(), "Message details")
        stops = _tab_stops(dialog, dialog._table, 8)
        self.assertEqual(
            set(_button_texts(stops)), {"Copy All", "Export...", "Clear", "Close"}
        )
        dialog.done(0)


class TestHelpDialogKeyboard(_DialogCase):
    def test_filter_then_close_and_page_keys_scroll(self):
        from sdr_module.gui.help_dialog import HelpDialog

        dialog = self.keep(HelpDialog())
        dialog.resize(420, 360)
        self.assertEqual(dialog._filter.accessibleName(), "Filter shortcuts")
        stops = _tab_stops(dialog, dialog._filter, 3)
        self.assertIs(stops[0], dialog._filter)
        self.assertEqual(_button_texts(stops[1:2]), ["Close"])
        self.assertIs(stops[2], dialog._filter)
        bar = dialog._scroll.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0)
        QTest.keyClick(dialog._filter, Qt.Key.Key_PageDown)
        self.assertGreater(bar.value(), 0)
        QTest.keyClick(dialog._filter, Qt.Key.Key_PageUp)
        self.assertEqual(bar.value(), 0)
        # Enter in the filter still never closes the dialog.
        QTest.keyClick(dialog._filter, Qt.Key.Key_Return)
        self.assertTrue(dialog.isVisible())

    def test_keys_use_the_keycap_role(self):
        from sdr_module.gui.help_dialog import HelpDialog

        dialog = self.keep(HelpDialog())
        roles = {
            label.property("role")
            for caps in dialog._caps
            for label in caps.findChildren(QLabel)
            if label.text() not in ("+", "/")
        }
        self.assertEqual(roles, {"keycap"})


class TestScannerKeyboard(_DialogCase):
    """The Scanner tab from the keyboard (it replaced the scanner dialog)."""

    def _panel_with_results(self):
        from sdr_module.gui.scan_engine import Found
        from tests.test_gui_dialogs import _scanner_panel

        panel, host, _dev = _scanner_panel()
        self.keep(panel)
        panel.resize(420, 560)
        for freq in (100.1e6, 101.1e6, 102.7e6):
            panel._on_found(Found(freq, -30.0, -88.0, 150e3, ("FM", "75 kHz", "")))
        return panel, host

    def test_enter_on_a_result_tunes_instead_of_rescanning(self):
        panel, host = self._panel_with_results()
        panel.show()
        _settle()
        panel._table.setFocus()
        panel._table.selectRow(1)
        QTest.keyClick(panel._table, Qt.Key.Key_Return)
        _settle()
        self.assertEqual(host.listened, [101.1e6])
        self.assertFalse(panel.engine.active)
        self.assertEqual(panel._table.rowCount(), 3)
        self.assertIn("Listening to 101.100 MHz", panel._status.text())
        # Arrow keys move through the results; Enter tunes the current one.
        QTest.keyClick(panel._table, Qt.Key.Key_Down)
        QTest.keyClick(panel._table, Qt.Key.Key_Enter)
        self.assertEqual(host.listened, [101.1e6, 102.7e6])
        self.assertIn("Enter", panel._table.toolTip())

    def test_tab_leaves_the_results_for_the_buttons(self):
        panel, _host = self._panel_with_results()
        panel._table.selectRow(0)
        self.assertFalse(panel._table.tabKeyNavigation())
        self.assertEqual(panel._table.accessibleName(), "Scan results")
        stops = _tab_stops(panel, panel._table, 5)
        self.assertEqual(
            _button_texts(stops[1:5]),
            ["Listen", "Bookmark", "Bookmark All", "⋯"],
        )

    def test_no_mnemonic_is_used_twice(self):
        panel, _host = self._panel_with_results()
        keys = {}
        for widget in panel.findChildren((QLabel, QAbstractButton)):
            match = re.search(r"&([^&])", widget.text())
            if match:
                keys.setdefault(match.group(1).lower(), []).append(widget.text())
        self.assertEqual({k: v for k, v in keys.items() if len(v) > 1}, {})

    def test_no_device_is_said_once(self):
        from tests.test_gui_dialogs import _scanner_panel

        reason = "No receiver: use the demo device or connect an SDR to scan."
        panel, _host, _dev = _scanner_panel(blocker=("device", reason))
        self.keep(panel)
        self.assertEqual(panel._status.text(), reason)
        self.assertEqual(panel._status.property("tone"), "warning")
        self.assertNotIn("No receiver", panel._empty.text())
        self.assertIn("Connect...", panel._empty.text())


# --------------------------------------------------------------------------- #
# Scanner measurements
# --------------------------------------------------------------------------- #
class TestScannerFrequencies(_DialogCase):
    def _sweep(self, start, end, step, threshold):
        from sdr_module.gui.device_dialog import MockDevice
        from sdr_module.gui.scan_engine import ScanEngine, ScanSettings
        from tests.test_gui_scanner import _DemoHost

        dev = MockDevice()
        dev.start_rx()
        engine = ScanEngine(_DemoHost(dev))
        hits = []
        engine.on_found = lambda found: hits.append(found.frequency_hz)
        engine.start(
            ScanSettings("once", start, end, step, threshold), dev.sample_rate, 0.0
        )
        for _ in range(100):
            if not engine.active:
                break
            engine.frame(dev.read_samples(int(dev.sample_rate * 0.034)))
        return hits

    def test_broadcast_stations_are_reported_on_the_channel_raster(self):
        hits = self._sweep(99.3e6, 100.3e6, 200e3, -50.0)
        self.assertEqual(sorted(set(hits)), [99.5e6, 100.1e6])

    def test_narrow_signals_are_rounded_to_a_kilohertz(self):
        hits = self._sweep(162.4e6, 162.55e6, 25e3, -70.0)
        for truth in (162.4e6, 162.475e6, 162.55e6):
            self.assertTrue(any(abs(f - truth) <= 1e3 for f in hits), (truth, hits))
        self.assertTrue(all(f % 1e3 == 0 for f in hits), hits)

    def test_report_frequency_rules(self):
        from sdr_module.gui.scanner_dialog import _report_frequency

        # Wide signal, broadcast-sized step: nearest 100 kHz channel.
        self.assertEqual(_report_frequency(88.4861e6, 90e3, 200e3), 88.5e6)
        # A narrow carrier in the same sweep keeps its own frequency.
        self.assertEqual(_report_frequency(433.9204e6, 5e3, 200e3), 433.92e6)
        # Voice-channel steps never snap to 100 kHz.
        self.assertEqual(_report_frequency(146.5204e6, 90e3, 12.5e3), 146.52e6)
        self.assertEqual(_report_frequency(146.5204e6, 90e3, None), 146.52e6)


class TestScannerRobustness(_DialogCase):
    def test_a_short_capture_is_a_failed_step_not_a_crash(self):
        from sdr_module.gui.scan_engine import MEASURE, ScanEngine, ScanSettings
        from tests.test_gui_scanner import _demo, _DemoHost

        dev = _demo()
        engine = ScanEngine(_DemoHost(dev))
        finished = []
        engine.on_finished = finished.append
        engine.start(ScanSettings("once", 88e6, 88.4e6, 200e3), dev.sample_rate, 0.0)
        engine.frame(np.ones(1, dtype=np.complex64))  # dropped after the retune
        for _ in range(20):
            engine.frame(np.ones(1, dtype=np.complex64))
        # One sample a block never makes a measurement; it waits for more.
        self.assertEqual(engine.state, MEASURE)
        self.assertEqual(engine.steps_measured, 0)
        self.assertEqual(finished, [])

    def test_an_analysis_error_fails_only_that_step(self):
        from sdr_module.dsp import sweep
        from sdr_module.gui import scan_engine

        # The measurement code is robust to a damaged spectrum...
        readings, _noise, _thr = sweep.measure_page(
            np.full(2048, np.nan), 100e6, 2.4e6, [100e6], 200e3
        )
        self.assertEqual(readings, [])
        # ...and an error inside it stops the scan, not the display loop.
        from sdr_module.gui.scan_engine import ScanEngine, ScanSettings
        from tests.test_gui_scanner import _demo, _DemoHost

        dev = _demo()
        engine = ScanEngine(_DemoHost(dev))
        engine.start(ScanSettings("once", 100e6, 100.2e6, 200e3), dev.sample_rate, 0.0)
        with (
            mock.patch.object(
                scan_engine, "measure_page", side_effect=ValueError("bad")
            ),
            self.assertLogs("sdr_module.gui.scan_engine", "WARNING"),
        ):
            for _ in range(10):
                engine.frame(dev.read_samples(80000))
        self.assertFalse(engine.active)

    def test_nothing_escapes_the_thread_and_the_dialog_says_why(self):
        from sdr_module.gui import scan_engine
        from tests.test_gui_dialogs import _feed, _scanner_panel

        panel, _host, dev = _scanner_panel()
        self.keep(panel)
        with (
            mock.patch.object(
                scan_engine, "measure_page", side_effect=RuntimeError("tuner went away")
            ),
            self.assertLogs("sdr_module.gui", "WARNING"),
        ):
            panel.start_scan()
            _feed(panel, dev, seconds=1.0)
        self.assertFalse(panel.engine.active)
        self.assertIn("tuner went away", panel._status.text())
        self.assertEqual(panel._status.property("tone"), "warning")
        self.assertTrue(panel._start_btn.isEnabled())


# --------------------------------------------------------------------------- #
# Demo device
# --------------------------------------------------------------------------- #
class TestDemoDevice(unittest.TestCase):
    def test_automatic_gain_avoids_clipping_and_restores_the_manual_gain(self):
        from sdr_module.gui.device_dialog import MockDevice

        dev = MockDevice()
        dev.start_rx()
        dev.set_frequency(100.1e6)
        dev.set_gain(49)
        clipped = np.mean(np.abs(dev.read_samples(8192).real) >= 0.999)
        self.assertGreater(clipped, 0.1)

        self.assertTrue(dev.set_gain_mode(True))
        self.assertEqual(dev.gain_mode, "auto")
        self.assertLess(dev.gain, 40.0)
        samples = dev.read_samples(8192)
        self.assertLess(float(np.max(np.abs(samples))), 0.5)  # about -10 dBFS
        # A manual gain set meanwhile applies once AGC is off again.
        dev.set_gain(33)
        self.assertLess(dev.gain, 40.0)
        dev.set_gain_mode(False)
        self.assertEqual((dev.gain_mode, dev.gain), ("manual", 33.0))

    def test_automatic_gain_lifts_a_quiet_band(self):
        from sdr_module.gui.device_dialog import MockDevice

        dev = MockDevice()
        dev.set_frequency(162.5e6)
        dev.set_gain(10)
        dev.set_gain_mode(True)
        self.assertGreater(dev.gain, 30.0)

    def test_every_band_preset_lands_on_a_demo_signal(self):
        from sdr_module.gui import main_window
        from sdr_module.gui.device_dialog import _DEMO_SIGNALS
        from sdr_module.gui.first_run_wizard import BAND_PRESETS

        demo = [freq for freq, *_rest in _DEMO_SIGNALS]
        presets = list(BAND_PRESETS.values())
        presets += [p.frequency_hz for p in main_window.BAND_PRESETS]
        if main_window.HAS_HAM_RADIO:
            from sdr_module.ham.gui.radio_tuner import RadioTunerWidget

            presets += [p.frequency_hz for p in RadioTunerWidget.DEFAULT_FM_PRESETS]
        for freq in presets:
            self.assertTrue(any(abs(freq - f) < 1e3 for f in demo), freq)
        # The strongest station is the FM Broadcast preset.
        strongest = max(_DEMO_SIGNALS, key=lambda s: s[2])
        self.assertEqual(strongest[0], BAND_PRESETS["FM Broadcast (88–108 MHz)"])


class TestDemoAgcInTheWindow(_WindowTestCase):
    demo = True

    def _level(self):
        """Mean power (dB) of the per-frame channel peaks over 30 frames.

        One frame's peak bin of FM music swings by 10 dB or more, so a single
        reading made this comparison flaky; the linear mean is stable.
        """
        levels = []
        for _ in range(30):
            self.win._update_display()
            levels.append(self.win._last_peak_db)
        return 10.0 * np.log10(np.mean(10.0 ** (np.asarray(levels) / 10.0)))

    def test_agc_changes_what_the_demo_receives(self):
        self.win.set_frequency(100.1e6)
        self.win.set_gain(45)
        manual = self._level()
        self.win._control_panel.set_agc_enabled(True)
        self.assertEqual(self.win._device.gain_mode, "auto")
        agc = self._level()
        self.assertLess(agc, manual - 6.0)
        self.assertLess(self.win._device.gain, 45.0 - 6.0)
        self.win._control_panel.set_agc_enabled(False)
        self.assertEqual(self.win._device.gain, 45.0)


# --------------------------------------------------------------------------- #
# Welcome screen
# --------------------------------------------------------------------------- #
class TestWelcomeStates(_DialogCase):
    def wizard(self, *args, **kwargs):
        from sdr_module.gui.first_run_wizard import FirstRunWizard

        return self.keep(FirstRunWizard(*args, **kwargs))

    def test_missing_driver_is_explained(self):
        missing = [("RTL-SDR", 'pip install "sdr-module[rtlsdr]"')]
        wiz = self.wizard(hardware_found=False, missing_drivers=missing)
        text = wiz._status.text()
        self.assertTrue(text.startswith("ⓘ "))
        self.assertIn("driver isn't installed", text)
        self.assertIn('pip install "sdr-module[rtlsdr]"', text)
        self.assertEqual(wiz._status.property("tone"), "info")
        self.assertTrue(wiz.wants_demo_mode())

    def test_missing_drivers_are_looked_up_by_default(self):
        from sdr_module.gui import device_dialog

        with mock.patch.object(device_dialog, "_driver_installed", lambda n: False):
            wiz = self.wizard(hardware_found=False)
        self.assertIn("sdr-module[hackrf]", wiz._status.text())
        with mock.patch.object(device_dialog, "_driver_installed", lambda n: True):
            wiz = self.wizard(hardware_found=False)
        self.assertTrue(wiz._status.text().startswith("ⓘ No SDR hardware"))

    def test_found_hardware_is_connected_on_get_started(self):
        class Window(QWidget):
            _device = None
            _demo_mode = False
            calls = []

            def _show_device_dialog(self, start_after=False):
                self.calls.append((start_after, self._wizard.isVisible()))

        win = self.keep(Window())
        wiz = self.wizard(win, hardware_found=True, device_name="HackRF One")
        win._wizard = wiz
        self.assertTrue(wiz._status.text().startswith("✓ Found HackRF One."))
        self.assertIn("Get Started connects it", wiz._status.text())
        self.assertIn("HackRF One", wiz._connect_check.text())
        self.assertFalse(wiz._connect_check.isHidden())
        self.assertTrue(wiz._demo_check.isHidden())
        self.assertTrue(wiz.wants_connect())
        self.assertFalse(wiz.wants_demo_mode())
        wiz.show()
        _settle()
        wiz._on_accept()
        # Connected after the welcome screen closed, receiving straight away.
        self.assertEqual(win.calls, [(True, False)])

    def test_unticking_connect_updates_the_callout(self):
        wiz = self.wizard(hardware_found=True, device_name="HackRF One")
        emitted = []
        wiz.connect_requested.connect(lambda: emitted.append(True))
        wiz._connect_check.setChecked(False)
        self.assertIn("Device > Connect", wiz._status.text())
        wiz._on_accept()
        self.assertEqual(emitted, [])

    def test_hardware_already_connected_offers_nothing_to_connect(self):
        class Window(QWidget):
            _device = object()
            _demo_mode = False

        wiz = self.wizard(self.keep(Window()), hardware_found=True)
        self.assertEqual(wiz._status.text(), "✓ Connected to your SDR.")
        self.assertTrue(wiz._connect_check.isHidden())
        self.assertTrue(wiz._demo_check.isHidden())
        self.assertFalse(wiz.wants_connect())

    def test_tips_use_the_keycap_role(self):
        wiz = self.wizard(hardware_found=False, missing_drivers=[])
        caps = [
            label
            for label in wiz.findChildren(QLabel)
            if label.property("role") in ("keycap", "badge")
        ]
        self.assertTrue(caps)
        self.assertEqual({label.property("role") for label in caps}, {"keycap"})


class TestWelcomeInTheWindow(_WindowTestCase):
    def test_get_started_connects_the_found_hardware(self):
        from sdr_module.gui.first_run_wizard import FirstRunWizard

        opened = []

        def fake_exec(wiz):
            wiz._on_accept()
            return wiz.result()

        self.win._scan_hardware = lambda: [("RTLSDRDevice:1#1", "RTL-SDR #0")]
        with (
            mock.patch.object(FirstRunWizard, "exec", fake_exec),
            mock.patch.object(
                self.win,
                "_show_device_dialog",
                lambda start_after=False: opened.append(start_after) or True,
            ),
        ):
            self.win._run_first_run_wizard()
        _flush_deletes()
        self.assertEqual(opened, [True])
        self.assertFalse(self.win._demo_mode)


if __name__ == "__main__":
    unittest.main()

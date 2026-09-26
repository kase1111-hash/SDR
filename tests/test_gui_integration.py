#!/usr/bin/env python3
"""
Integration tests for the main window's wiring to its panels, dialogs and
devices: bookmarks, SSTV audio, Ham ID, the radio tuner, the license class,
gain/AGC, the first-run wizard, the device dialog, hot-plug polling, the
scanner, dialog lifetimes and the ``--sample-rate`` launcher flag.

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_integration.py``
"""

import json
import logging
import os
import sys
import traceback
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _FakeSettings,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtCore import QCoreApplication, QEvent, QTimer
    from PyQt6.QtWidgets import QApplication, QDialog, QMessageBox


_saved_style = None


def setUpModule():  # noqa: N802 - unittest hook
    """Run on the unstyled app, whatever an earlier module left applied.

    These are wiring tests, and building a main window under the full theme
    stylesheet costs ~0.2 s instead of ~0.03 s. The theme test below applies
    both themes itself.
    """
    global _saved_style
    if not HAS_PYQT6:
        return
    qapp = QApplication.instance()
    _saved_style = (qapp.styleSheet(), qapp.palette())
    qapp.setStyleSheet("")


def tearDownModule():  # noqa: N802 - unittest hook
    if not HAS_PYQT6 or _saved_style is None:
        return
    qapp = QApplication.instance()
    qapp.setStyleSheet(_saved_style[0])
    qapp.setPalette(_saved_style[1])


def _ham_radio() -> bool:
    from sdr_module.gui.main_window import HAS_HAM_RADIO

    return HAS_HAM_RADIO


def _flush_deletes() -> None:
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)


def _recording_device(base=None):
    """A device that records the calls the main window makes."""
    from sdr_module.devices.base import SDRDevice

    class Recorder(SDRDevice):
        def __init__(self):
            super().__init__()
            self._state.sample_rate = 2.4e6
            self.calls = []

        def _log(self, name, *args):
            self.calls.append((name, *args))
            return True

        def open(self, index=0):
            return self._log("open", index)

        def close(self):
            self._log("close")

        def set_frequency(self, freq_hz):
            return self._log("set_frequency", freq_hz)

        def set_sample_rate(self, rate_hz):
            self._state.sample_rate = rate_hz
            return self._log("set_sample_rate", rate_hz)

        def set_bandwidth(self, bw_hz):
            return self._log("set_bandwidth", bw_hz)

        def set_gain(self, gain_db):
            return self._log("set_gain", gain_db)

        def set_gain_mode(self, auto):
            return self._log("set_gain_mode", auto)

        def start_rx(self, callback=None):
            self._state.is_streaming = True
            return self._log("start_rx")

        def stop_rx(self):
            self._state.is_streaming = False
            return self._log("stop_rx")

        def names(self):
            return [c[0] for c in self.calls]

    return Recorder()


class _ExclusiveDongle:
    """Builds a driver class whose one device can be open only once, like a
    real USB dongle ("in use" until the holder closes it)."""

    def __init__(self):
        from sdr_module.devices.base import DeviceInfo, SDRDevice

        owner = self
        self.in_use = False
        self.opened = []

        class Dongle(SDRDevice):
            def __init__(self):
                super().__init__()
                self._state.sample_rate = 2.4e6
                self.started = False

            def open(self, index=0):
                if owner.in_use:
                    return False
                owner.in_use = True
                owner.opened.append(self)
                self._is_open = True
                self._info = DeviceInfo(
                    "Fake Dongle", "0001", "Test", "Dongle", index=index
                )
                return True

            def close(self):
                if self._is_open:
                    owner.in_use = False
                self._is_open = False

            def set_frequency(self, freq_hz):
                return True

            def set_sample_rate(self, rate_hz):
                self._state.sample_rate = rate_hz
                return True

            def set_bandwidth(self, bw_hz):
                return True

            def set_gain(self, gain_db):
                return True

            def set_gain_mode(self, auto):
                return True

            def start_rx(self, callback=None):
                self.started = True
                self._state.is_streaming = True
                return True

            def stop_rx(self):
                self._state.is_streaming = False
                return True

            def read_samples(self, num_samples, timeout=1.0):
                return np.zeros(num_samples, dtype=np.complex64)

        self.cls = Dongle


class TestBookmarksFollowTuning(_WindowTestCase):
    def test_add_field_shows_the_tuned_frequency(self):
        field = self.win._bookmarks_panel._freq_input
        self.win.set_frequency(146.52e6)
        self.assertAlmostEqual(field.value(), 146.52, places=3)
        self.win._spectrum.frequency_clicked.emit(433.92e6)
        self.assertAlmostEqual(field.value(), 433.92, places=3)
        # Quick-tune buttons in the control panel tune through the signal.
        self.win._control_panel._quick_tune(1e6)
        self.assertAlmostEqual(field.value(), 434.92, places=3)


class TestSSTVAudio(_WindowTestCase):
    demo = True

    def setUp(self):
        super().setUp()
        if not _ham_radio():
            self.skipTest("ham radio panels not installed")
        self.panel = self.win._sstv_panel
        self.fed = []
        self.panel.process_audio = lambda audio, sample_rate=None: self.fed.append(
            (np.asarray(audio), sample_rate)
        )
        self.spoken = []
        self.win._audio.write = self.spoken.append
        self.win._set_audio_enabled(False)

    def tearDown(self):
        if HAS_PYQT6 and getattr(self, "panel", None) and self.panel.is_receiving():
            self.panel._start_btn.click()  # stop its poll timer
        super().tearDown()

    def test_decoder_gets_continuous_fm_audio_with_speaker_off(self):
        self.win._set_demod_mode("FM")
        self.panel._start_btn.click()
        self.assertTrue(self.panel.is_receiving())
        # Below squelch too: the decoder needs every block for its timing.
        self.win._control_panel.set_squelch_db(0)
        # (The demo device is read in real time: blocks vary in length.)
        device = self.win._device
        read, reads = device.read_samples, []
        device.read_samples = lambda n: reads.append(read(n)) or reads[-1]
        for _ in range(3):
            self.win._update_display()
        self.assertEqual(len(self.fed), 3)
        self.assertEqual({rate for _a, rate in self.fed}, {48000.0})
        for audio, _rate in self.fed:
            self.assertEqual(audio.dtype, np.float32)
        # Gap-free: the I/Q blocks give one audio sample per 50, the
        # decimation phase carried into the next block.
        total = sum(len(a) for a, _r in self.fed)
        self.assertEqual(total, -(-sum(len(r) for r in reads) // 50))
        self.assertEqual(self.spoken, [])  # speaker stays silent

    def test_non_fm_mode_hints_and_feeds_nothing(self):
        self.win._set_demod_mode("AM")
        self.panel._start_btn.click()
        self.assertIn("FM", self.win._message_label.text())
        self.assertEqual(self.win._message_label.property("tone"), "warning")
        self.win._update_display()
        self.assertEqual(self.fed, [])

    def test_switching_away_from_fm_while_listening_hints(self):
        self.win._set_demod_mode("FM")
        self.panel._start_btn.click()
        self.win._clear_status_message()
        self.win._set_demod_mode("USB")
        self.assertIn("SSTV", self.win._message_label.text())

    def test_start_while_stopped_says_to_start_receiving(self):
        self.win._stop_acquisition()
        self.win._set_demod_mode("FM")
        self.panel._start_btn.click()
        self.assertIn("Start receiving", self.win._message_label.text())

    def test_speaker_gets_the_same_audio_above_squelch(self):
        self.win._audio_enabled = True
        self.win._set_demod_mode("FM")
        self.win._control_panel.set_squelch_db(-120)
        self.win._update_display()
        self.assertEqual(len(self.spoken), 1)
        self.assertEqual(self.fed, [])  # the decoder isn't listening


class TestHamId(_WindowTestCase):
    def setUp(self):
        super().setUp()
        if not _ham_radio():
            self.skipTest("ham radio panels not installed")

    def test_send_id_needs_a_hackrf(self):
        from sdr_module.devices.hackrf import HackRFDevice

        panel = self.win._callsign_panel
        self.assertIs(panel._tx_available, False)
        self.assertEqual(
            panel._tx_unavailable_reason, "Connect a HackRF One to transmit an ID."
        )
        self.win._start_demo_mode()
        self.assertIs(panel._tx_available, False)
        self.win._release_device()

        self.win._device = HackRFDevice()  # not opened; no hardware touched
        self.win._refresh_state_ui()
        self.assertIs(panel._tx_available, True)
        self.win._device = None
        self.win._refresh_state_ui()
        self.assertIs(panel._tx_available, False)

    def test_settings_are_saved_and_restored(self):
        panel = self.win._callsign_panel
        panel.set_callsign("w1aw")
        panel._wpm_spin.setValue(25)
        saved = json.loads(self.store.values["ham_id"])
        self.assertEqual(saved["callsign"], "W1AW")
        self.assertEqual(saved["cw_wpm"], 25)

        other = self.mw.SDRMainWindow()
        try:
            restored = other._callsign_panel
            self.assertEqual(restored.get_callsign(), "W1AW")
            self.assertEqual(restored._wpm_spin.value(), 25)
            # Restoring doesn't rewrite the saved settings half-applied.
            self.assertEqual(json.loads(self.store.values["ham_id"]), saved)
        finally:
            other.close()
            other.deleteLater()

    def test_unreadable_saved_settings_are_ignored(self):
        self.store.values["ham_id"] = "{not json"
        other = self.mw.SDRMainWindow()
        try:
            self.assertEqual(other._callsign_panel.get_callsign(), "")
        finally:
            other.close()
            other.deleteLater()


class TestRadioTuner(_WindowTestCase):
    def setUp(self):
        super().setUp()
        if not _ham_radio():
            self.skipTest("ham radio panels not installed")

    def tearDown(self):
        if HAS_PYQT6 and getattr(self.win, "_radio_tuner", None) is not None:
            self.win._radio_tuner.close()
        super().tearDown()

    def test_tuner_opens_on_the_received_station(self):
        self.win.set_frequency(97.1e6)
        self.win._show_radio_tuner()
        tuner = self.win._radio_tuner
        self.assertAlmostEqual(tuner.get_frequency(), 97.1e6)
        self.assertEqual(tuner.get_band().value, "FM")
        # While open, it follows tuning inside the broadcast bands...
        self.win.set_frequency(1010e3)
        self.assertAlmostEqual(tuner.get_frequency(), 1010e3)
        self.assertEqual(tuner.get_band().value, "AM")
        # ...and ignores frequencies outside them.
        self.win.set_frequency(146.52e6)
        self.assertAlmostEqual(tuner.get_frequency(), 1010e3)

    def test_outside_the_broadcast_bands_the_tuner_keeps_its_station(self):
        self.win.set_frequency(146.52e6)
        self.win._show_radio_tuner()
        self.assertAlmostEqual(self.win._radio_tuner.get_frequency(), 101.1e6)
        self.assertAlmostEqual(self.freq(), 146.52e6)  # nothing emitted

    def test_tuning_the_radio_sets_the_receiver_mode(self):
        self.win._show_radio_tuner()
        tuner = self.win._radio_tuner
        panel = self.win._control_panel
        tuner.frequency_changed.emit(1010e3, "AM")
        self.assertAlmostEqual(self.freq(), 1010e3)
        self.assertEqual(panel._demod_combo.currentText(), "AM")
        tuner.frequency_changed.emit(93.3e6, "FM")
        self.assertAlmostEqual(self.freq(), 93.3e6)
        self.assertEqual(panel._demod_combo.currentText(), "FM")
        self.assertEqual(panel.get_fm_deviation(), 75e3)
        self.assertEqual(self.store.values["demod_mode"], "FM")


class TestLicenseClass(_WindowTestCase):
    def setUp(self):
        super().setUp()
        from sdr_module.core.frequency_manager import get_frequency_manager

        self._saved_class = get_frequency_manager().get_license_class()

    def tearDown(self):
        if HAS_PYQT6:
            from sdr_module.core.frequency_manager import get_frequency_manager

            get_frequency_manager().set_license_class(self._saved_class)
        super().tearDown()

    def _window_with(self, saved):
        self.store.values["license_class"] = saved
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        self.addCleanup(other.close)
        return other

    def test_change_is_saved(self):
        from sdr_module.core.frequency_manager import LicenseClass

        self.win._control_panel.set_license_class(LicenseClass.GENERAL)
        self.assertEqual(self.store.values["license_class"], "general")

    def test_saved_class_is_restored(self):
        from sdr_module.core.frequency_manager import (
            LicenseClass,
            get_frequency_manager,
        )

        other = self._window_with("technician")
        self.assertEqual(
            other._control_panel._selected_license(), LicenseClass.TECHNICIAN
        )
        self.assertEqual(
            get_frequency_manager().get_license_class(), LicenseClass.TECHNICIAN
        )

    def test_unknown_saved_class_falls_back_to_none(self):
        from sdr_module.core.frequency_manager import LicenseClass

        other = self._window_with("wizard")
        self.assertEqual(other._control_panel._selected_license(), LicenseClass.NONE)


class TestGainAndAgc(_WindowTestCase):
    def test_set_gain_does_not_override_agc(self):
        dev = _recording_device()
        self.win._control_panel.set_agc_enabled(True)
        self.win._device = dev
        self.win.set_gain(35)
        self.assertNotIn("set_gain", dev.names())
        # Switching AGC off applies the slider's gain.
        self.win._control_panel.set_agc_enabled(False)
        self.assertIn(("set_gain_mode", False), dev.calls)
        self.assertEqual(dev.calls[-1], ("set_gain", 35.0))
        self.win._device = None

    def test_new_device_gets_agc_mode_then_gain(self):
        panel = self.win._control_panel
        panel.set_gain(30)

        panel.set_agc_enabled(True)
        dev = _recording_device()
        self.win._device = dev
        self.win._apply_controls_to_device()
        self.assertIn(("set_gain_mode", True), dev.calls)
        self.assertNotIn("set_gain", dev.names())

        panel.set_agc_enabled(False)
        dev = _recording_device()
        self.win._device = dev
        self.win._apply_controls_to_device()
        names = dev.names()
        self.assertLess(names.index("set_gain_mode"), names.index("set_gain"))
        self.assertIn(("set_gain", 30.0), dev.calls)
        self.win._device = None

    def test_demo_device_follows_the_gain_slider(self):
        self.win._control_panel.set_gain(34)
        self.win._start_demo_mode()
        self.assertEqual(self.win._device.gain, 34.0)


class TestFirstRunWizard(_WindowTestCase):
    def _run(self, band, demo):
        from sdr_module.gui.first_run_wizard import FirstRunWizard

        def fake_exec(wiz):
            wiz._band.setCurrentText(band)
            wiz._demo_check.setChecked(demo)
            wiz._on_accept()
            return wiz.result()

        with mock.patch.object(FirstRunWizard, "exec", fake_exec):
            self.win._run_first_run_wizard()
        _flush_deletes()

    def test_airband_with_demo_mode(self):
        from sdr_module.gui.first_run_wizard import FirstRunWizard

        self._run("Airband AM (118–137 MHz)", demo=True)
        self.assertTrue(self.win._demo_mode)
        self.assertTrue(self.win._is_running)
        self.assertAlmostEqual(self.freq(), 125e6)
        self.assertEqual(self.win._control_panel._demod_combo.currentText(), "AM")
        self.assertEqual(self.win.findChildren(FirstRunWizard), [])

    def test_noaa_selects_narrowband_fm(self):
        self._run("NOAA Weather (162 MHz)", demo=False)
        self.assertIsNone(self.win._device)
        self.assertAlmostEqual(self.freq(), 162.55e6)
        panel = self.win._control_panel
        self.assertEqual(panel._demod_combo.currentText(), "FM")
        self.assertEqual(panel.get_fm_deviation(), 5e3)

    def test_fm_broadcast_selects_wideband_fm(self):
        self._run("FM Broadcast (88–108 MHz)", demo=False)
        self.assertEqual(self.win._control_panel.get_fm_deviation(), 75e3)


class TestDeviceDialog(_WindowTestCase):
    """Connecting replaces the current device; Cancel keeps it."""

    def setUp(self):
        super().setUp()
        from sdr_module.gui import device_dialog

        self.dd = device_dialog
        self.dongle = _ExclusiveDongle()
        cls = self.dongle.cls
        patcher = mock.patch.object(
            device_dialog.DeviceDialog,
            "_device_class",
            staticmethod(lambda dev_type: cls if dev_type == "rtlsdr" else None),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.seen = {}

    def _connect(self, rate=None, accept=True):
        """Run Device > Connect..., choosing the fake dongle."""
        seen = self.seen
        win = self.win

        def fake_exec(dialog):
            dialog._add_device("RTL-SDR", "Fake Dongle", "0001", "Available")
            dialog._devices.append({"type": "rtlsdr", "index": 0, "info": {}})
            dialog._device_table.selectRow(dialog._device_table.rowCount() - 1)
            seen["preselected_rate"] = dialog._rate_combo.currentData()
            if rate is not None:
                dialog._rate_combo.setCurrentIndex(dialog._rate_combo.findData(rate))
            if not accept:
                return 0
            with mock.patch.object(self.dd.QMessageBox, "warning") as warn:
                dialog._on_accept()
                seen["badge_during_open"] = win._state_badge.text()
                if warn.called:
                    dialog.reject()
            return dialog.result()

        with mock.patch.object(self.dd.DeviceDialog, "exec", fake_exec):
            result = self.win._show_device_dialog()
        _flush_deletes()
        return result

    def _open_first_dongle(self):
        first = self.dongle.cls()
        first.open(0)
        self.win._device = first
        self.win._start_acquisition()
        self.assertTrue(self.win._is_running)
        return first

    def test_reselecting_the_open_dongle_reopens_it_with_the_new_rate(self):
        first = self._open_first_dongle()
        self.assertTrue(self._connect(rate=1.8e6))
        dev = self.win._device
        self.assertIsNot(dev, first)
        self.assertFalse(first.is_open)
        self.assertEqual(dev.state.sample_rate, 1.8e6)
        # The dialog offered the current rate; the old device was released
        # (and the badge said so) before the new one was opened.
        self.assertEqual(self.seen["preselected_rate"], 2.4e6)
        self.assertEqual(self.seen["badge_during_open"], "NO DEVICE")
        # Receiving resumes on the new device, and the chrome agrees.
        self.assertTrue(dev.started)
        self.assertTrue(self.win._is_running)
        self.assertEqual(self.win._state_badge.text(), "RUNNING")
        self.assertEqual(self.win._rate_label.text(), "1.8 MS/s")
        self.assertIn("Fake Dongle", self.win.windowTitle())
        self.win._refresh_info_panel()
        self.assertEqual(self.win._info_panel.value("sample_rate"), "1.8 MS/s")
        self.assertEqual(self.win.findChildren(self.dd.DeviceDialog), [])

    def test_cancel_keeps_the_current_device(self):
        first = self._open_first_dongle()
        self.assertFalse(self._connect(accept=False))
        self.assertIs(self.win._device, first)
        self.assertTrue(first.is_open)
        self.assertTrue(self.win._is_running)
        self.assertEqual(self.win._state_badge.text(), "RUNNING")

    def test_failed_reopen_then_cancel_leaves_nothing_connected(self):
        self._open_first_dongle()
        self.dongle.cls.open = lambda dev, index=0: False  # now unplugged
        self.assertFalse(self._connect())
        self.assertIsNone(self.win._device)
        self.assertFalse(self.win._is_running)
        self.assertEqual(self.win._state_badge.text(), "NO DEVICE")
        self.assertEqual(self.win.windowTitle(), "SDR Module")
        self.assertIn("Disconnected", self.win._message_label.text())

    def test_demo_to_hardware_releases_the_demo_device(self):
        self.win._start_demo_mode()
        demo = self.win._device
        self.assertTrue(self._connect())
        self.assertIsNot(self.win._device, demo)
        self.assertFalse(self.win._demo_mode)
        self.assertFalse(demo.is_streaming)
        self.assertNotIn("Demo", self.win.windowTitle())


class TestHotplug(_WindowTestCase):
    def test_missing_drivers_are_not_polled_and_log_nothing(self):
        from sdr_module.core import device_manager
        from sdr_module.gui import main_window

        self.win._hardware_classes = None
        with (
            mock.patch.object(main_window, "_driver_importable", return_value=False),
            mock.patch.object(
                device_manager.DeviceManager,
                "scan_devices",
                side_effect=AssertionError("must not scan"),
            ),
            self.assertNoLogs("sdr_module", level=logging.INFO),
        ):
            for _ in range(3):
                self.win._poll_hotplug()
        self.assertEqual(self.win._hardware_classes, [])

    def test_polling_this_environment_logs_no_warnings(self):
        self.win._hardware_classes = None
        with self.assertNoLogs("sdr_module", level=logging.WARNING):
            for _ in range(3):
                self.win._poll_hotplug()

    def test_driver_probe(self):
        from sdr_module.gui.main_window import _driver_importable

        self.assertTrue(_driver_importable("json", "dumps"))
        self.assertTrue(_driver_importable("importlib", "util"))
        self.assertFalse(_driver_importable("no_such_sdr_driver_pkg", "X"))
        self.assertFalse(_driver_importable("json", "no_such_name"))

    def test_dongles_sharing_a_serial_are_both_listed(self):
        from sdr_module.devices.base import DeviceInfo

        class Driver:
            @staticmethod
            def list_devices():
                return [
                    DeviceInfo(f"RTL-SDR #{i}", "00000001", "x", "y", index=i)
                    for i in range(2)
                ]

        self.win._hardware_classes = [Driver]
        found = self.win._scan_hardware()
        self.assertEqual([name for _k, name in found], ["RTL-SDR #0", "RTL-SDR #1"])
        self.assertEqual(len({key for key, _n in found}), 2)

    def test_scan_menu_lists_names_and_missing_drivers(self):
        from sdr_module.gui import main_window

        shown = []
        with (
            mock.patch.object(main_window, "_driver_importable", return_value=False),
            mock.patch.object(
                main_window.QMessageBox,
                "information",
                side_effect=lambda *a: shown.append(a),
            ),
        ):
            self.win._refresh_devices()
        self.assertEqual(shown[0][1], "No Devices Found")
        self.assertIn('sdr-module[rtlsdr]"', shown[0][2])
        self.assertIn('sdr-module[hackrf]"', shown[0][2])


class TestScanner(_WindowTestCase):
    demo = True

    def _run(self, display_active):
        from sdr_module.gui.scanner_dialog import ScannerDialog

        seen = {}
        if display_active:
            self.win._display_timer.start()

        def fake_exec(dialog):
            seen["display_active"] = self.win._display_timer.isActive()
            dialog.frequency_selected.emit(433.92e6)
            return 0

        with mock.patch.object(ScannerDialog, "exec", fake_exec):
            self.win._show_scanner()
        _flush_deletes()
        self.assertEqual(self.win.findChildren(ScannerDialog), [])
        return seen

    def test_display_pauses_and_tuning_a_result_tunes_the_receiver(self):
        seen = self._run(display_active=True)
        self.assertFalse(seen["display_active"])
        self.assertTrue(self.win._display_timer.isActive())
        self.assertEqual(self.win._display_timer.interval(), 33)
        self.assertAlmostEqual(self.freq(), 433.92e6)
        self.assertAlmostEqual(self.win._device.frequency, 433.92e6)
        self.win._display_timer.stop()

    def test_a_stopped_display_stays_stopped(self):
        self._run(display_active=False)
        self.assertFalse(self.win._display_timer.isActive())


class TestDialogLifetime(_WindowTestCase):
    """Modal dialogs are deleted after use instead of piling up."""

    def _count_after(self, open_dialog, patch_target):
        before = set(map(id, self.win.findChildren(QDialog)))
        with mock.patch.object(patch_target, "exec", return_value=0):
            open_dialog()
        _flush_deletes()
        return [d for d in self.win.findChildren(QDialog) if id(d) not in before]

    def test_help_error_history_and_prompt_are_deleted(self):
        from sdr_module.gui.error_log_dialog import ErrorLogDialog
        from sdr_module.gui.help_dialog import HelpDialog

        for opener, cls in (
            (self.win._show_help, HelpDialog),
            (self.win._show_error_history, ErrorLogDialog),
            (self.win._prompt_no_device, QMessageBox),
        ):
            for _ in range(3):
                self.assertEqual(self._count_after(opener, cls), [], cls.__name__)

    def test_prompt_still_reports_the_choice(self):
        with mock.patch.object(QMessageBox, "exec", return_value=0):
            self.assertEqual(self.win._prompt_no_device(), "cancel")

    def test_device_dialog_is_deleted_when_cancelled(self):
        from sdr_module.gui.device_dialog import DeviceDialog

        self.assertEqual(
            self._count_after(self.win._show_device_dialog, DeviceDialog), []
        )


class TestCleanups(_WindowTestCase):
    demo = True

    def test_export_confirms_once(self):
        calls = []
        self.win._bookmarks_panel.export_csv = lambda: calls.append(1) or 3
        self.win._clear_status_message()
        self.win._export_channels_csv()
        self.assertEqual(calls, [1])
        self.assertEqual(self.win._message_label.text(), "")

    def test_waterfall_gets_center_and_span(self):
        self.win.set_frequency(145.8e6)
        self.win.set_sample_rate(2.048e6)
        self.assertAlmostEqual(self.win._waterfall._center_freq, 145.8e6)
        self.assertAlmostEqual(self.win._waterfall._sample_rate, 2.048e6)

    def test_band_preset_sets_fm_deviation(self):
        self.win._apply_band_preset(100.1e6, "FM", "FM Broadcast")
        self.assertEqual(self.win._control_panel.get_fm_deviation(), 75e3)
        self.win._apply_band_preset(146.52e6, "FM", "2m Ham (146.52)")
        self.assertEqual(self.win._control_panel.get_fm_deviation(), 5e3)
        self.win._apply_band_preset(125e6, "AM", "Airband AM")
        self.assertEqual(self.win._control_panel._demod_combo.currentText(), "AM")


class TestSampleRateFlag(_WindowTestCase):
    demo = True

    def test_applies_to_the_demo_device(self):
        self.win.set_sample_rate(1.024e6)
        self.assertEqual(self.win._device.sample_rate, 1.024e6)
        self.assertEqual(self.win._rate_label.text(), "1.024 MS/s")
        self.win._refresh_info_panel()
        self.assertEqual(self.win._info_panel.value("sample_rate"), "1.024 MS/s")

    def test_is_kept_for_a_later_demo_device_and_the_connect_dialog(self):
        from sdr_module.gui.device_dialog import DeviceDialog

        self.win._disconnect_device()
        self.win.set_sample_rate(1.8e6)
        seen = []

        def fake_exec(dialog):
            seen.append(dialog._rate_combo.currentData())
            return 0

        with mock.patch.object(DeviceDialog, "exec", fake_exec):
            self.win._show_device_dialog()
        self.assertEqual(seen, [1.8e6])
        self.win._start_demo_mode()
        self.assertEqual(self.win._device.sample_rate, 1.8e6)

    def test_bad_values_are_ignored(self):
        self.win.set_sample_rate(0)
        self.win.set_sample_rate("fast")
        self.assertEqual(self.win._device.sample_rate, 2.4e6)

    def test_launcher_flag_detection(self):
        from sdr_module.gui.app import SDRApplication

        def should(*argv, value=2.4e6):
            launcher = SDRApplication(args=["sdr", *argv])
            return launcher._should_apply({"sample_rate": value}, "sample_rate")

        self.assertFalse(should("--demo"))
        self.assertTrue(should("-s", "2.4e6"))
        self.assertTrue(should("--sample-rate=2.4e6"))
        self.assertTrue(should(value=1.024e6))


class TestEveryMenuActionAndThemeSwitching(unittest.TestCase):
    """Open every dialog and tool from the menus in one session, then switch
    themes repeatedly: no exception may escape (Qt only prints exceptions
    raised in slots, so sys.excepthook collects them)."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import bookmarks_panel, main_window, themes

        self.themes = themes
        self.store = _FakeSettings()
        for module in (main_window, bookmarks_panel):
            patcher = mock.patch.object(module, "GuiSettings", lambda: self.store)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.errors = []
        old_hook = sys.excepthook
        sys.excepthook = lambda *exc: self.errors.append(
            "".join(traceback.format_exception(*exc))
        )
        self.addCleanup(setattr, sys, "excepthook", old_hook)
        qapp = QApplication.instance()
        old_sheet, old_palette = qapp.styleSheet(), qapp.palette()
        old_theme = themes.current_theme()

        def restore_theme():
            qapp.setStyleSheet(old_sheet)
            qapp.setPalette(old_palette)
            themes.apply_theme(None, old_theme)

        self.addCleanup(restore_theme)

    @staticmethod
    def _actions(menu):
        for action in menu.actions():
            if action.menu() is not None:
                yield from TestEveryMenuActionAndThemeSwitching._actions(action.menu())
            elif not action.isSeparator():
                yield action

    def test_open_everything_then_switch_themes(self):
        from sdr_module.gui.main_window import SDRMainWindow

        qapp = QApplication.instance()
        win = SDRMainWindow(demo_mode=True)
        self.addCleanup(win.deleteLater)
        self.addCleanup(win.close)
        for timer in (win._display_timer, win._status_timer, win._hotplug_timer):
            timer.stop()
        win._persist_state = lambda: None
        win.resize(1280, 720)
        win.show()

        opened = []

        def close_modal():
            modal = QApplication.activeModalWidget()
            if modal is not None:
                opened.append(type(modal).__name__)
                if isinstance(modal, QDialog):
                    modal.reject()
                else:
                    modal.close()

        closer = QTimer()
        closer.timeout.connect(close_modal)
        closer.start(20)
        self.addCleanup(closer.stop)

        triggered = 0
        for top in win.menuBar().actions():
            for action in self._actions(top.menu()):
                name = action.text().replace("&", "").split("\t")[0]
                if name == "Exit" or not action.isEnabled():
                    continue
                action.trigger()
                triggered += 1
                qapp.processEvents()
                win._update_display()
                if win._radio_tuner is not None and win._radio_tuner.isVisible():
                    opened.append("RadioTunerWidget")
                    win._radio_tuner.close()
        closer.stop()
        _flush_deletes()

        self.assertGreater(triggered, 30)
        for expected in (
            "DeviceDialog",
            "ScannerDialog",
            "HelpDialog",
            "ErrorLogDialog",
            "RadioTunerWidget",
        ):
            if expected == "RadioTunerWidget" and not _ham_radio():
                continue
            self.assertIn(expected, opened)
        # Only the reusable radio tuner window is kept.
        leftovers = {type(d).__name__ for d in win.findChildren(QDialog)}
        self.assertLessEqual(leftovers, {"RadioTunerWidget"})

        for i in range(4):
            self.themes.apply_theme(qapp, "light" if i % 2 == 0 else "dark")
            qapp.processEvents()
            win._update_display()
        self.assertEqual(self.errors, [])


if __name__ == "__main__":
    unittest.main()

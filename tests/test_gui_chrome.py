#!/usr/bin/env python3
"""
Tests for the main window "chrome": layout, toolbar, status bar, menus,
keyboard handling, the Info tab and the application launcher.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_chrome.py``
"""

import os
import re
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QAction, QShortcut
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import (
        QApplication,
        QLabel,
        QPushButton,
        QScrollArea,
        QSplitter,
    )

    HAS_PYQT6 = True
    PYQT6_IMPORT_ERROR = ""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
except ImportError as _import_error:  # pragma: no cover - environment
    HAS_PYQT6 = False
    PYQT6_IMPORT_ERROR = str(_import_error)

GUI_DIR = Path(__file__).resolve().parent.parent / "src" / "sdr_module" / "gui"


def require_pyqt6(test_case: unittest.TestCase) -> None:
    """Skip when PyQt6 is unavailable (fail instead if SDR_REQUIRE_GUI is set)."""
    if HAS_PYQT6:
        return
    if os.environ.get("SDR_REQUIRE_GUI"):
        test_case.fail(
            f"PyQt6 is required but could not be imported: {PYQT6_IMPORT_ERROR}"
        )
    test_case.skipTest("PyQt6 not available")


class _FakeSettings:
    """In-memory stand-in for GuiSettings so tests never touch QSettings."""

    def __init__(self):
        self.values = {}
        self.bookmarks = []

    def get_float(self, key, default):
        return float(self.values.get(key, default))

    def get_int(self, key, default):
        return int(self.values.get(key, default))

    def get_bool(self, key, default):
        return bool(self.values.get(key, default))

    def get_str(self, key, default):
        return str(self.values.get(key, default))

    def set(self, key, value):
        self.values[key] = value

    def sync(self):
        pass

    def get_bookmarks(self):
        return list(self.bookmarks)

    def set_bookmarks(self, bookmarks):
        self.bookmarks = list(bookmarks)

    def is_first_run(self):
        return False

    def mark_first_run_done(self):
        pass

    def save_geometry(self, name, geometry):
        self.values[f"geom/{name}"] = geometry

    def load_geometry(self, name):
        return self.values.get(f"geom/{name}")


class _WindowTestCase(unittest.TestCase):
    """Builds a real SDRMainWindow with in-memory settings and no timers."""

    demo = False

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import bookmarks_panel, main_window

        self.mw = main_window
        self._patched = [
            (main_window, "GuiSettings", main_window.GuiSettings),
            (bookmarks_panel, "GuiSettings", bookmarks_panel.GuiSettings),
        ]
        self.store = _FakeSettings()
        main_window.GuiSettings = lambda: self.store
        bookmarks_panel.GuiSettings = lambda: self.store

        self.win = main_window.SDRMainWindow(demo_mode=self.demo)
        for timer in (
            self.win._display_timer,
            self.win._status_timer,
            self.win._hotplug_timer,
        ):
            timer.stop()
        # Recording a new take or closing with unsaved samples asks first
        # (a modal box); tests answer "Don't Save" unless they test the box.
        self.win._confirm_discard_recording = lambda *_args: True
        self.win.resize(1280, 720)
        self.win.show()
        self.win.activateWindow()
        self.settle()

    def tearDown(self):
        if not HAS_PYQT6:
            return
        self.win.close()
        self.win.deleteLater()
        self.settle()
        for module, name, value in self._patched:
            setattr(module, name, value)

    @staticmethod
    def settle(n=5):
        for _ in range(n):
            QApplication.processEvents()

    def freq(self):
        return self.win._control_panel._freq_input.get_frequency()


class TestLayout(_WindowTestCase):
    def test_right_column_is_vertical_splitter_with_scrolling_controls(self):
        right = self.win._right_splitter
        self.assertIsInstance(right, QSplitter)
        self.assertEqual(right.orientation(), Qt.Orientation.Vertical)
        top = right.widget(0)
        # Either the chrome wrapped the control panel, or it scrolls itself.
        self.assertTrue(
            isinstance(top, QScrollArea) or top.findChild(QScrollArea) is not None
        )
        self.assertIs(right.widget(1), self.win._right_tabs)
        # Not wrapped twice.
        if isinstance(top, QScrollArea):
            self.assertIsNone(self.win._control_panel.findChild(QScrollArea))

    def test_minimum_size_fits_small_laptops(self):
        self.assertLessEqual(self.win.minimumWidth(), 1024)
        self.assertLessEqual(self.win.minimumHeight(), 640)

    def test_right_column_keeps_a_sane_width(self):
        width = self.win._main_splitter.sizes()[1]
        self.assertGreaterEqual(width, 340)
        self.assertLessEqual(width, 480)

    def test_top_level_tabs_fit_without_hiding(self):
        tabs = self.win._right_tabs
        titles = [tabs.tabText(i) for i in range(tabs.count())]
        self.assertIn("Decoder", titles)
        self.assertIn("Info", titles)
        total = sum(tabs.tabBar().tabRect(i).width() for i in range(tabs.count()))
        self.assertLessEqual(total, 360)
        for i in range(tabs.count()):
            self.assertTrue(tabs.tabToolTip(i))

    def test_info_tab_is_populated(self):
        self.win._show_panel("info")
        self.settle()
        self.win._refresh_info_panel()
        info = self.win._info_panel
        self.assertEqual(info.value("frequency"), "100.100 MHz")  # the default
        self.assertEqual(info.value("state"), "Not connected")
        self.assertTrue(info.value("version"))

    def test_show_panel_selects_nested_ham_panel(self):
        from sdr_module.gui.main_window import HAS_HAM_RADIO

        if not HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")
        self.win._show_panel("s_meter")
        self.assertIs(self.win._right_tabs.currentWidget(), self.win._ham_tabs)
        self.assertIs(
            self.win._ham_tabs.currentWidget(), self.win._panel_pages["s_meter"]
        )

    def test_plot_toggles_keep_one_plot_visible(self):
        self.win._spectrum_action.setChecked(False)
        self.assertFalse(self.win._waterfall_action.isEnabled())
        self.win._spectrum_action.setChecked(True)
        self.assertTrue(self.win._waterfall_action.isEnabled())


class TestShortcuts(_WindowTestCase):
    def test_no_key_sequence_is_bound_twice(self):
        seen = {}
        for action in self.win.findChildren(QAction):
            for seq in action.shortcuts():
                text = seq.toString()
                self.assertNotIn(text, seen, f"{text} bound twice")
                seen[text] = action.text()
        for shortcut in self.win.findChildren(QShortcut):
            text = shortcut.key().toString()
            self.assertNotIn(text, seen, f"{text} bound twice")
            seen[text] = "QShortcut"

    def test_ctrl_b_bookmarks_current_frequency(self):
        before = len(self.store.bookmarks)
        QTest.keyClick(
            self.win._spectrum, Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier
        )
        self.settle()
        self.assertEqual(len(self.store.bookmarks), before + 1)

    def test_arrows_tune_when_plot_has_focus(self):
        self.win._spectrum.setFocus()
        self.settle()
        start = self.freq()
        QTest.keyClick(self.win._spectrum, Qt.Key.Key_Right)
        self.assertAlmostEqual(self.freq(), start + 10e3)
        QTest.keyClick(
            self.win._spectrum, Qt.Key.Key_Left, Qt.KeyboardModifier.ShiftModifier
        )
        self.assertAlmostEqual(self.freq(), start - 90e3)
        QTest.keyClick(
            self.win._spectrum, Qt.Key.Key_Right, Qt.KeyboardModifier.ControlModifier
        )
        self.assertAlmostEqual(self.freq(), start + 910e3)
        self.assertEqual(
            self.win._freq_label.text(), self.mw.format_frequency(start + 910e3)
        )

    def test_focused_slider_keeps_its_arrow_keys(self):
        slider = self.win._control_panel._gain_slider
        slider.setFocus()
        self.settle()
        start_freq, start_value = self.freq(), slider.value()
        QTest.keyClick(slider, Qt.Key.Key_Right)
        self.assertEqual(self.freq(), start_freq)
        self.assertEqual(slider.value(), start_value + 1)

    def test_space_on_focused_button_presses_it_not_start(self):
        buttons = [
            b
            for b in self.win._control_panel.findChildren(QPushButton)
            if b.isVisible()
            and b.isEnabled()
            and b.focusPolicy() != Qt.FocusPolicy.NoFocus
            and not b.isCheckable()
        ]
        self.assertTrue(buttons)
        button = buttons[0]
        clicks = []
        button.clicked.connect(lambda *_: clicks.append(1))
        button.setFocus()
        self.settle()
        prompted = []
        self.win._prompt_no_device = lambda: prompted.append(1) or "cancel"
        QTest.keyClick(button, Qt.Key.Key_Space)
        self.assertEqual(len(clicks), 1)
        self.assertEqual(prompted, [])

    def test_space_with_plot_focus_toggles_receiving(self):
        prompted = []
        self.win._prompt_no_device = lambda: prompted.append(1) or "cancel"
        self.win._spectrum.setFocus()
        self.settle()
        QTest.keyClick(self.win._spectrum, Qt.Key.Key_Space)
        self.assertEqual(prompted, [1])


class TestActionsAndState(_WindowTestCase):
    def test_start_without_device_offers_demo(self):
        self.win._prompt_no_device = lambda: "demo"
        self.win._start_button.click()
        self.assertTrue(self.win._is_running)
        self.assertTrue(self.win._demo_mode)
        self.assertTrue(self.win._start_button.text().endswith("Stop"))
        # Plain (not red) while running: red is reserved for Record.
        self.assertEqual(self.win._start_button.property("role") or "", "")
        self.assertEqual(self.win._state_badge.text(), "RUNNING")
        self.assertIn("Demo", self.win.windowTitle())
        self.win._start_button.click()
        self.assertFalse(self.win._is_running)
        self.assertEqual(self.win._start_button.property("role"), "primary")

    def test_start_without_device_cancel_does_nothing(self):
        self.win._prompt_no_device = lambda: "cancel"
        self.win._toggle_acquisition()
        self.assertFalse(self.win._is_running)
        self.assertIsNone(self.win._device)
        self.assertEqual(self.win._state_badge.text(), "NO DEVICE")

    def test_decoder_action_switches_tab_instead_of_dialog(self):
        self.win._show_panel("info")
        self.win._show_decoder_config()
        self.assertIs(
            self.win._right_tabs.currentWidget(), self.win._panel_pages["decoder"]
        )

    def test_status_messages_are_toned(self):
        self.win._show_status_message("Saved", "success")
        self.assertEqual(self.win._message_label.text(), "Saved")
        self.assertEqual(self.win._message_label.property("tone"), "success")
        self.win._show_status_error("Boom")
        self.assertEqual(self.win._message_label.property("tone"), "danger")
        self.win._clear_status_message()
        self.assertEqual(self.win._message_label.text(), "")

    def test_bookmark_feedback_is_not_an_error(self):
        self.win._bookmark_current_frequency()
        self.assertEqual(self.win._message_label.property("tone"), "success")

    def test_click_to_tune_updates_controls_and_axis(self):
        self.win._spectrum.frequency_clicked.emit(146.52e6)
        self.assertAlmostEqual(self.freq(), 146.52e6)
        self.assertEqual(self.win._freq_label.text(), "146.520 MHz")
        self.assertAlmostEqual(self.win._spectrum._center_freq, 146.52e6)
        self.assertAlmostEqual(self.win._waterfall._center_freq, 146.52e6)

    def test_agc_toggle_passes_a_boolean(self):
        calls = []

        class Dev:
            def set_gain_mode(self, auto):
                calls.append(auto)

        self.win._device = Dev()
        self.win._on_agc_changed(False)
        self.win._on_agc_changed(True)
        self.win._device = None
        self.assertEqual(calls, [False, True])

    def test_record_button_menu_and_panel_stay_in_sync(self):
        self.win._record_button.click()
        self.assertTrue(self.win._recording)
        self.assertTrue(self.win._record_action.isChecked())
        self.assertEqual(self.win._record_button.property("role"), "danger")
        self.assertFalse(self.win._recording_label.isHidden())
        self.assertTrue(self.win._control_panel._record_btn.isChecked())
        self.win._record_action.trigger()
        self.assertFalse(self.win._recording)
        self.assertFalse(self.win._record_button.isChecked())
        self.assertTrue(self.win._recording_label.isHidden())

    def test_theme_toggle_updates_menu_and_persists(self):
        from sdr_module.gui import themes

        qapp = QApplication.instance()
        old_sheet, old_palette = qapp.styleSheet(), qapp.palette()
        old_theme = themes.current_theme()
        try:
            themes.apply_theme(qapp, "dark")
            self.win._toggle_theme()
            self.assertEqual(themes.current_theme(), "light")
            self.assertEqual(self.win._theme, "light")
            self.assertTrue(self.win._theme_actions["light"].isChecked())
            self.assertEqual(self.store.values.get("theme"), "light")
        finally:
            qapp.setStyleSheet(old_sheet)
            qapp.setPalette(old_palette)
            themes.apply_theme(None, old_theme)


class TestDemoWindow(_WindowTestCase):
    demo = True

    def test_display_update_feeds_readouts(self):
        for _ in range(3):
            self.win._update_display()
        self.assertTrue(self.win._level_label.text().endswith("dBFS"))
        self.assertIsNotNone(self.win._last_peak_db)

    def test_s_meter_receives_samples(self):
        from sdr_module.gui.main_window import HAS_HAM_RADIO

        if not HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")
        self.win._update_display()
        self.assertIsNotNone(self.win._signal_meter_panel._last_reading)

    def test_disconnect_returns_to_no_device_state(self):
        self.win._disconnect_device()
        self.assertIsNone(self.win._device)
        self.assertFalse(self.win._disconnect_action.isEnabled())
        self.assertTrue(self.win._demo_action.isEnabled())
        self.assertEqual(self.win.windowTitle(), "SDR Module")


class TestReviewFixes(_WindowTestCase):
    """Fixes from the chrome review: clamping, pause, save format, status bar."""

    demo = True

    def test_out_of_range_frequency_is_clamped_everywhere(self):
        self.win.set_frequency(20e9)
        cp = self.freq()
        self.assertLess(cp, 20e9)
        from sdr_module.gui.main_window import format_frequency

        self.assertEqual(self.win._freq_label.text(), format_frequency(cp))
        self.assertAlmostEqual(self.win._spectrum._center_freq, cp)
        self.win.set_frequency(100e6)
        self.win._nudge_frequency(-500e6)
        self.assertEqual(self.win._freq_label.text(), format_frequency(self.freq()))

    def test_start_and_record_are_not_both_red(self):
        self.assertTrue(self.win._is_running)
        self.win._record_button.click()
        self.assertEqual(self.win._record_button.property("role"), "danger")
        self.assertNotEqual(self.win._start_button.property("role"), "danger")
        self.assertIn("Stop recording", self.win._record_button.toolTip())
        self.win._record_button.click()
        self.assertNotIn("Stop recording", self.win._record_button.toolTip())

    def test_recording_pause_button_works(self):
        panel = self.win._control_panel
        self.win._record_button.click()
        self.assertTrue(panel._pause_btn.isEnabled())
        self.win._update_display()
        captured = len(self.win._samples_buffer)
        self.assertEqual(captured, 1)

        panel._pause_btn.click()
        self.assertTrue(self.win._recording_paused)
        self.win._update_display()
        self.assertEqual(len(self.win._samples_buffer), captured)
        self.win._update_status()
        self.assertTrue(self.win._recording_label.text().startswith("PAUSED"))
        self.assertEqual(self.win._recording_label.property("tone"), "warning")

        panel._pause_btn.click()
        self.assertFalse(self.win._recording_paused)
        self.win._update_display()
        self.assertEqual(len(self.win._samples_buffer), captured + 1)
        self.win._update_status()
        self.assertTrue(self.win._recording_label.text().startswith("REC "))
        self.win._record_button.click()
        self.assertFalse(self.win._recording_paused)

    def test_recording_without_receiver_is_shown_as_armed(self):
        self.win._stop_acquisition()
        self.win._record_button.click()
        self.assertEqual(self.win._recording_label.text(), "REC ARMED")
        self.assertEqual(self.win._recording_elapsed(), 0.0)
        self.win._record_button.click()
        # Nothing captured: say so instead of staying silent.
        self.assertEqual(self.win._message_label.property("tone"), "warning")

    def test_resolve_save_format(self):
        resolve = self.win._resolve_save_format
        wav_filter = next(f[0] for f in self.win._SAVE_FORMATS if f[1] == ".wav")
        name, spec = resolve("rec/x", wav_filter)
        self.assertEqual((name, spec[2]), ("rec/x.wav", "WAV"))
        # A typed extension wins over the selected filter.
        name, spec = resolve("rec/x.cf32", wav_filter)
        self.assertEqual((name, spec[3]), ("rec/x.cf32", "FLOAT32"))
        name, spec = resolve("rec/x.sigmf-meta", "")
        self.assertEqual((name, spec[2]), ("rec/x.sigmf-data", "SIGMF"))

    def test_save_uses_the_recording_panel_format(self):
        import tempfile

        from sdr_module.gui import main_window

        panel = self.win._control_panel
        panel._format_combo.setCurrentText("SigMF (.sigmf-data)")
        self.win._record_button.click()
        self.win._update_display()
        self.win._record_button.click()

        seen = {}
        folder = tempfile.mkdtemp(prefix="sdr-chrome-")

        def fake_save(parent, caption, directory, filters, initial):
            seen.update(directory=directory, initial=initial)
            return os.path.join(folder, "capture"), initial

        with mock.patch.object(
            main_window.QFileDialog, "getSaveFileName", side_effect=fake_save
        ):
            self.win._save_recording()
        self.assertIn("SigMF", seen["initial"])
        self.assertTrue(seen["directory"].endswith(".sigmf-data"))
        self.assertTrue(os.path.exists(os.path.join(folder, "capture.sigmf-data")))
        self.assertTrue(os.path.exists(os.path.join(folder, "capture.sigmf-meta")))

    def test_status_bar_elides_long_messages(self):
        label = self.win._message_label
        text = "A very long status message " * 20
        self.win._show_status_message(text)
        self.settle()
        self.assertEqual(label.text(), text)
        self.assertTrue(QLabel.text(label).endswith("…"))
        self.assertFalse(self.win.statusBar().isSizeGripEnabled())
        self.assertLessEqual(self.win.minimumSizeHint().width(), 1024)

    def test_exit_has_a_shortcut(self):
        exits = [
            a
            for a in self.win.findChildren(QAction)
            if a.text().replace("&", "") == "Exit"
        ]
        self.assertEqual(len(exits), 1)
        self.assertFalse(exits[0].shortcut().isEmpty())

    def test_ctrl_l_and_readout_click_focus_the_frequency_field(self):
        field = self.win._control_panel._freq_input._freq_input
        self.win._spectrum.setFocus()
        self.settle()
        QTest.keyClick(
            self.win._spectrum, Qt.Key.Key_L, Qt.KeyboardModifier.ControlModifier
        )
        self.settle()
        self.assertTrue(field.hasFocus())

        self.win._spectrum.setFocus()
        self.settle()
        QTest.mouseClick(self.win._freq_label, Qt.MouseButton.LeftButton)
        self.settle()
        self.assertTrue(field.hasFocus())

    def test_app_icon_is_drawn(self):
        from sdr_module.gui.app import make_app_icon

        icon = make_app_icon()
        self.assertFalse(icon.isNull())
        self.assertFalse(icon.pixmap(32, 32).isNull())

    def test_decoder_is_rebuilt_for_a_new_sample_rate(self):
        self.win._decoder_panel._proto_combo.setCurrentText("POCSAG")
        first = self.win._decoder
        self.assertIsNotNone(first)

        class State:
            sample_rate = 1.024e6

        self.win._device.state = State()
        self.win._refresh_state_ui()
        self.assertIsNot(self.win._decoder, first)
        self.assertEqual(self.win._decoder_rate, 1.024e6)


def _fake_hardware_device(blocks, streaming=True, error=None):
    """An SDRDevice subclass that hands out ``blocks`` like a USB stream."""
    from sdr_module.devices.base import SDRDevice

    class FakeHardware(SDRDevice):
        rx_error = error

        def __init__(self):
            super().__init__()
            self._state.sample_rate = 2.048e6
            self._state.is_streaming = streaming
            self.timeouts = []

        def read_samples(self, num_samples, timeout=1.0):
            self.timeouts.append(timeout)
            return blocks.pop(0) if blocks else None

        def open(self, index=0):
            return True

        def close(self):
            pass

        def set_frequency(self, freq_hz):
            return True

        def set_sample_rate(self, rate_hz):
            return True

        def set_bandwidth(self, bw_hz):
            return True

        def set_gain(self, gain_db):
            return True

        def set_gain_mode(self, auto):
            return True

        def start_rx(self, callback=None):
            return True

        def stop_rx(self):
            return True

    return FakeHardware()


class TestHardwareStream(_WindowTestCase):
    """The display loop must never wait on (or silently lose) a USB stream."""

    def test_reads_do_not_block_and_plots_use_a_fixed_fft(self):
        import numpy as np

        from sdr_module.gui.main_window import DISPLAY_BLOCK

        block = (np.ones(256 * 1024) * 0.1).astype(np.complex64)
        dev = _fake_hardware_device([block])
        self.win._device = dev
        self.win._is_running = True
        self.win._recording = True
        seen = []
        self.win._spectrum.update_spectrum = seen.append
        self.win._update_display()
        # Every queued transfer is taken, none waited for.
        self.assertTrue(dev.timeouts)
        self.assertEqual(set(dev.timeouts), {0.0})
        self.assertEqual(len(seen[0]), DISPLAY_BLOCK)
        # The recording keeps the whole block, not just the plotted part.
        self.assertEqual(len(self.win._samples_buffer[0]), len(block))
        self.win._recording = False
        self.win._is_running = False
        self.win._device = None

    def test_dead_stream_stops_and_reports(self):
        dev = _fake_hardware_device([], streaming=False, error="USB unplugged")
        self.win._device = dev
        self.win._is_running = True
        self.win._update_display()
        self.assertFalse(self.win._is_running)
        self.assertEqual(self.win._state_badge.text(), "STOPPED")
        self.assertIn("USB unplugged", self.win._message_label.text())
        self.assertEqual(self.win._message_label.property("tone"), "danger")
        self.win._device = None

    def test_hotplug_announces_first_device_after_empty_start(self):
        from sdr_module.devices.base import DeviceInfo

        dongle = DeviceInfo("RTL-SDR #0", "00000001", "RTL-SDR Blog", "RTL2832U")
        found = [[], [dongle]]

        class Driver:
            @staticmethod
            def list_devices():
                return found.pop(0)

        self.win._hardware_classes = [Driver]
        self.win._poll_hotplug()  # baseline: nothing attached
        self.win._poll_hotplug()  # a dongle was plugged in
        text = self.win._message_label.text()
        self.assertIn("RTL-SDR #0", text)
        self.assertNotIn("DeviceInfo(", text)  # a name, not a repr


class TestFatalErrorDialog(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_message_does_not_suggest_a_wrong_flag(self):
        from PyQt6.QtWidgets import QMessageBox

        from sdr_module.gui.app import SDRApplication

        shown = []
        with mock.patch.object(
            QMessageBox, "critical", side_effect=lambda *a, **k: shown.append(a)
        ):
            launcher = SDRApplication(args=["sdr"])
            launcher._app = QApplication.instance()
            launcher._show_fatal_error(RuntimeError("boom"))
        self.assertEqual(len(shown), 1)
        text = shown[0][2]
        self.assertIn("boom", text)
        self.assertNotIn("-v", text)


class TestFormatting(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_format_frequency(self):
        from sdr_module.gui.main_window import format_frequency

        self.assertEqual(format_frequency(100e6), "100.000 MHz")
        self.assertEqual(format_frequency(146.5125e6), "146.5125 MHz")
        self.assertEqual(format_frequency(1090e6), "1090.000 MHz")
        self.assertEqual(format_frequency(7.074e6), "7.074 MHz")

    def test_format_rate_and_bandwidth(self):
        from sdr_module.gui.main_window import format_bandwidth, format_rate

        self.assertEqual(format_rate(2.4e6), "2.4 MS/s")
        self.assertEqual(format_rate(2.048e6), "2.048 MS/s")
        self.assertEqual(format_bandwidth(2.4e6), "2.4 MHz")
        self.assertEqual(format_bandwidth(1757.8), "1.76 kHz")


class TestLauncherSettings(unittest.TestCase):
    """Launcher defaults must not clobber the persisted frequency/gain."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui.app import SDRApplication

        self.make = lambda *argv: SDRApplication(args=["sdr", *argv])

    def test_default_values_not_typed_are_ignored(self):
        from sdr_module.gui.settings_store import DEFAULT_FREQUENCY_HZ

        app_ = self.make("--demo")
        self.assertFalse(
            app_._should_apply({"frequency": DEFAULT_FREQUENCY_HZ}, "frequency")
        )
        self.assertFalse(app_._should_apply({"gain": 20.0}, "gain"))
        self.assertFalse(app_._should_apply({"frequency": None}, "frequency"))

    def test_explicit_values_apply(self):
        self.assertTrue(self.make()._should_apply({"frequency": 144.8e6}, "frequency"))
        self.assertTrue(
            self.make("-f", "100e6")._should_apply({"frequency": 100e6}, "frequency")
        )
        self.assertTrue(
            self.make("--freq=100e6")._should_apply({"frequency": 100e6}, "frequency")
        )
        self.assertTrue(self.make("-g20")._should_apply({"gain": 20.0}, "gain"))


class TestNoHardCodedColors(unittest.TestCase):
    """Chrome files must take every color from the theme palette."""

    PATTERN = re.compile(
        r"#[0-9a-fA-F]{3,8}\b|rgba?\(|\b(gray|grey|green|red|orange|blue|yellow|"
        r"white|black)\b\s*[;\"']"
    )

    def test_chrome_sources_have_no_literal_colors(self):
        for name in ("main_window.py", "app.py"):
            text = (GUI_DIR / name).read_text(encoding="utf-8")
            hits = [
                line.strip()
                for line in text.splitlines()
                if self.PATTERN.search(line) and "noqa: color" not in line
            ]
            self.assertEqual(hits, [], f"{name} has literal colors: {hits}")

    def test_no_inline_stylesheets(self):
        text = (GUI_DIR / "main_window.py").read_text(encoding="utf-8")
        self.assertNotIn("setStyleSheet(", text)


if __name__ == "__main__":
    unittest.main()

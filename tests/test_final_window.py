"""Final fixes to the window chrome, the devices and the saved settings
(F-window): the problems the end-to-end journeys found (DEV-E2E-*, ROB-*,
LISTEN-02, DEC-E2E-12, PLOTS-E2E-15) and the leftovers of the earlier
rounds. Each test failed before its fix.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest
tests/test_final_window.py``
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, require_pyqt6  # noqa: E402
from tests.test_gui_device_settings import _DeviceCase  # noqa: E402
from tests.test_gui_second_receiver import _FakeSink, _SecondCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import QSettings, Qt
    from PyQt6.QtWidgets import QApplication, QMessageBox


def _block(n=4096, seed=1):
    """A block of quiet noise, as a device delivers it."""
    rng = np.random.default_rng(seed)
    noise = 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    return noise.astype(np.complex64)


class _Answers:
    """Stands in for the window's ``_exec_dialog``: answers each question
    box with the button named, and runs Device > Connect... choosing a row
    (by its name) through the dialog's own Connect."""

    def __init__(self, win, answer="Cancel", row=None, rows=()):
        self.win = win
        self.answer = answer
        self.row = row
        self.rows = list(rows)
        self.boxes = []
        self.dialogs = []

    def __call__(self, box):
        from sdr_module.gui.device_dialog import DeviceDialog

        if isinstance(box, QMessageBox):
            self.boxes.append(box.text())
            for button in box.buttons():
                if button.text().replace("&", "") == self.answer:
                    button.click()
                    break
            return 0
        if isinstance(box, DeviceDialog):
            self.dialogs.append(box)
            for label, name, kind, index in self.rows:
                box._add_device(label, name, "0001", "Available")
                box._devices.append(
                    {"type": kind, "index": index, "info": {}, "name": name}
                )
            getattr(box, "_mark_busy_rows", lambda: None)()
            for row in range(box._device_table.rowCount()):
                if box._device_table.item(row, 1).text() == self.row:
                    box._device_table.selectRow(row)
                    break
            else:
                raise AssertionError(f"no row {self.row!r}")
            box._on_accept()
            return box.result()
        return box.exec()


class TestDeviceSwitchWhileRecording(_DeviceCase):
    """DEV-E2E-01: Device > Connect... and Disconnect ask before a recording
    taking its samples from the device is ended; the next device never adds
    its samples to the same take (one take holds one device and rate)."""

    def recording_on_rtl(self):
        dev = self.rtl()
        win = self.win
        win._start_acquisition()
        win._toggle_recording(True)
        win._process_block(_block())
        self.assertTrue(win._recording)
        self.assertFalse(win._capture_pending)
        return dev, win._buffer_sample_count()

    def test_connect_asks_and_cancel_keeps_the_device_and_the_take(self):
        dev, count = self.recording_on_rtl()
        win = self.win
        answers = _Answers(win, "Cancel", row="Demo Device")
        win._exec_dialog = answers
        self.assertFalse(win._show_device_dialog())
        self.assertEqual(
            answers.boxes,
            ["Stop recording and switch from RTL-SDR #0 to the demo device?"],
        )
        # Nothing changed: the dialog stayed open (not accepted), the device
        # and the take are as they were.
        self.assertIs(win._device, dev)
        self.assertTrue(dev.is_open)
        self.assertTrue(win._is_running)
        self.assertTrue(win._recording)
        self.assertEqual(win._buffer_sample_count(), count)

    def test_stop_and_switch_ends_the_take_before_the_next_device(self):
        dev, count = self.recording_on_rtl()
        win = self.win
        win._exec_dialog = _Answers(win, "Stop and Switch", row="Demo Device")
        self.assertTrue(win._show_device_dialog())
        self.assertTrue(win._demo_mode)
        self.assertFalse(dev.is_open)
        self.assertFalse(win._recording)
        # The demo device's samples don't join the RTL-SDR's take.
        win._process_block(_block(seed=2))
        self.assertEqual(win._buffer_sample_count(), count)
        self.assertTrue(win._buffer_unsaved)  # (kept, to save)

    def test_the_same_dongle_at_another_rate_asks_before_it_is_closed(self):
        dev, count = self.recording_on_rtl()
        win = self.win
        answers = _Answers(
            win,
            "Cancel",
            row="RTL-SDR #0",
            rows=[("RTL-SDR", "RTL-SDR #0", "rtlsdr", 0)],
        )
        win._exec_dialog = answers
        self.assertFalse(win._show_device_dialog())
        self.assertEqual(answers.boxes, ["Stop recording and reconnect RTL-SDR #0?"])
        # Asked before the dongle was released to be opened again.
        self.assertIs(win._device, dev)
        self.assertTrue(dev.is_open)
        self.assertTrue(win._recording)

    def test_disconnect_asks_and_ends_the_take(self):
        dev, count = self.recording_on_rtl()
        win = self.win
        answers = _Answers(win, "Cancel")
        win._exec_dialog = answers
        win._disconnect_device()
        self.assertEqual(answers.boxes, ["Stop recording and disconnect RTL-SDR #0?"])
        self.assertIs(win._device, dev)
        self.assertTrue(win._recording)
        answers.answer = "Stop and Disconnect"
        win._disconnect_device()
        self.assertIsNone(win._device)
        # Ended, not waiting for the next device (no REC WAITING).
        self.assertFalse(win._recording)
        self.assertNotIn("WAITING", win._recording_label.text())
        self.assertEqual(win._buffer_sample_count(), count)
        # The demo device now starts a take of its own (none is armed).
        win._exec_dialog = _Answers(win, "Stop and Switch")
        win._use_demo_device()
        win._process_block(_block(seed=3))
        self.assertEqual(win._buffer_sample_count(), count)

    def test_releasing_the_device_ends_a_take_that_has_samples(self):
        _dev, count = self.recording_on_rtl()
        win = self.win
        win._release_device()
        self.assertFalse(win._recording)
        self.assertEqual(win._buffer_sample_count(), count)

    def test_an_armed_recording_waits_for_the_device_connected(self):
        # Record pressed before receiving: Connect... starts it on the
        # device chosen, without a question.
        dev = self.rtl()
        win = self.win
        win._toggle_recording(True)
        self.assertTrue(win._capture_pending)
        answers = _Answers(win, "Cancel", row="Demo Device")
        win._exec_dialog = answers
        self.assertTrue(win._show_device_dialog(start_after=True))
        self.assertEqual(answers.boxes, [])
        self.assertTrue(win._recording)
        self.assertFalse(dev.is_open)
        win._process_block(_block())
        self.assertFalse(win._capture_pending)
        self.assertGreater(win._buffer_sample_count(), 0)

    def test_a_rate_change_is_not_called_a_retune(self):
        _dev, _count = self.recording_on_rtl()
        win = self.win
        win._buffer_meta = (win._lo_frequency(), 1.8e6)
        win._process_block(_block(seed=4))
        text = win._message_label.text()
        self.assertIn("The sample rate changed while recording", text)
        self.assertNotIn("Retuned", text)


class TestConnectDialogBusyDevice(_DeviceCase):
    """DEV-E2E-03: a device the second receiver has is listed as in use and
    can't be connected; an expected refusal is no error."""

    def dialog(self, busy_index=1):
        from sdr_module.gui.device_dialog import DeviceDialog

        dialog = DeviceDialog(self.win)
        self.addCleanup(dialog.deleteLater)
        dialog._device_table.setRowCount(0)
        dialog._devices.clear()
        for index in (0, 1):
            dialog._add_device("RTL-SDR", f"RTL-SDR #{index}", "0001", "Available")
            dialog._devices.append(
                {"type": "rtlsdr", "index": index, "info": {}, "name": f"RTL #{index}"}
            )
        reason = "RTL-SDR #1 is in use by the second receiver. Stop it there."

        def busy(entry):
            if entry.get("index") == busy_index:
                return ("In use by the second receiver", reason)
            return None

        dialog.set_busy_check(busy)
        return dialog, reason

    def test_the_row_says_in_use_and_connect_is_off(self):
        dialog, reason = self.dialog()
        table = dialog._device_table
        self.assertEqual(table.item(1, 3).text(), "In use by the second receiver")
        self.assertEqual(table.item(1, 1).toolTip(), reason)
        self.assertEqual(table.item(0, 3).text(), "Available")
        # Greyed out: it can't be selected (the tooltip says why).
        self.assertFalse(table.item(1, 1).flags() & Qt.ItemFlag.ItemIsEnabled)
        table.selectRow(1)
        self.assertIsNone(dialog._selected_entry())
        dialog._update_setting_controls(dialog._devices[1])
        self.assertFalse(dialog._connect_btn.isEnabled())
        self.assertEqual(dialog._connect_btn.toolTip(), reason)
        dialog._update_setting_controls(dialog._devices[0])
        self.assertTrue(dialog._connect_btn.isEnabled())

    def test_the_first_free_row_is_selected(self):
        dialog, _reason = self.dialog(busy_index=0)
        self.assertEqual(dialog._selected_entry()["index"], 1)
        self.assertTrue(dialog._connect_btn.isEnabled())

    def test_a_refusal_is_information_not_an_error(self):
        from sdr_module.gui import device_dialog

        dialog, reason = self.dialog()
        opened = []
        dialog._open_device = lambda entry: opened.append(entry)
        with (
            mock.patch.object(device_dialog.QMessageBox, "information") as info,
            mock.patch.object(device_dialog.QMessageBox, "critical") as critical,
            self.assertNoLogs("sdr_module.gui.device_dialog", "ERROR"),
        ):
            # A row marked busy while selected (Enter or a double-click).
            dialog._device_table.selectRow(0)
            dialog._devices[0]["busy"] = reason
            dialog._on_accept()
            # Taken after the list was made: the window's open refuses.
            raised = device_dialog.ConnectRefused("in use elsewhere")

            def refuse(entry):
                raise raised

            dialog._open_device = refuse
            dialog._devices[0]["busy"] = ""
            dialog._on_accept()
        self.assertEqual(opened, [])
        self.assertEqual(info.call_count, 2)
        self.assertEqual(info.call_args_list[0].args[2], reason)
        self.assertEqual(info.call_args_list[1].args[2], "in use elsewhere")
        critical.assert_not_called()
        self.assertNotEqual(dialog.result(), 1)  # (still open)


# --------------------------------------------------------------------------- #
# Saved settings read back at startup (ROB-02, ROB-07, ROB-08, ROB-09, ROB-13)
# --------------------------------------------------------------------------- #
class _IniSettings(unittest.TestCase):
    """Real GuiSettings on a QSettings INI file in a temporary folder,
    holding what a damaged or hand-edited settings file can hold."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import settings_store

        folder = tempfile.mkdtemp(prefix="fw-settings-")
        self.addCleanup(shutil.rmtree, folder, True)
        self.path = os.path.join(folder, "settings.ini")
        path = self.path
        patcher = mock.patch.object(
            settings_store,
            "QSettings",
            lambda *_args: QSettings(path, QSettings.Format.IniFormat),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, **values):
        """Store ``values`` as QSettings writes them (text, or a list)."""
        raw = QSettings(self.path, QSettings.Format.IniFormat)
        for key, value in values.items():
            raw.setValue(key.replace("__", "/"), value)
        raw.sync()

    def settings(self):
        from sdr_module.gui.settings_store import GuiSettings

        return GuiSettings()


class TestDamagedValuesGiveTheDefault(_IniSettings):
    """ROB-07 / ROB-02: a value that isn't of the key's type gives the
    key's default (not False, not True, not an overflow), logged once."""

    def test_booleans(self):
        self.write(
            audio_enabled="garbage",
            squelch__enabled="nope",
            tuning__keep_centered=["a", "b"],
            audio__balance_iq=["y"],
            ok_true="true",
            ok_off="off",
        )
        s = self.settings()
        with self.assertLogs("sdr_module.gui.settings_store", "WARNING") as logs:
            self.assertTrue(s.get_bool("audio_enabled", True))
            self.assertTrue(s.get_bool("squelch/enabled", True))
            self.assertFalse(s.get_bool("tuning/keep_centered", False))
            self.assertFalse(s.get_bool("audio/balance_iq", False))
            s.get_bool("audio_enabled", True)  # (said once per key)
        self.assertEqual(len(logs.records), 4)
        self.assertIn("audio_enabled", logs.output[0])
        self.assertTrue(s.get_bool("ok_true", False))
        self.assertFalse(s.get_bool("ok_off", True))
        self.assertTrue(s.get_bool("missing", True))

    def test_numbers(self):
        self.write(
            audio__cw_pitch="2147483648",
            big="99999999999999999999",
            small="-99999999999999999999",
            gain_db="nan",
            squelch_db="inf",
            fine="-7",
        )
        s = self.settings()
        with self.assertLogs("sdr_module.gui.settings_store", "WARNING"):
            self.assertEqual(s.get_int("audio/cw_pitch", 700), 700)
            self.assertEqual(s.get_int("big", 5), 5)
            self.assertEqual(s.get_int("small", 5), 5)
            self.assertEqual(s.get_float("gain_db", 20.0), 20.0)
            self.assertEqual(s.get_float("squelch_db", -80.0), -80.0)
        self.assertEqual(s.get_int("fine", 0), -7)
        self.assertEqual(s.get_float("fine", 0.0), -7.0)

    def test_a_list_is_no_text(self):
        self.write(recording__folder=["a", "b"])
        with self.assertLogs("sdr_module.gui.settings_store", "WARNING"):
            self.assertEqual(self.settings().get_str("recording/folder", ""), "")

    def test_a_damaged_first_run_flag_is_not_a_first_run(self):
        self.assertTrue(self.settings().is_first_run())  # (nothing saved)
        self.write(first_run_done="garbage")
        with self.assertLogs("sdr_module.gui.settings_store", "WARNING"):
            self.assertFalse(self.settings().is_first_run())


class TestDamagedSettingsInTheWindow(_IniSettings):
    """The window starts with each damaged value at its default (ROB-02:
    the app couldn't start at all with a 64-bit CW pitch)."""

    def window(self, **values):
        from sdr_module.gui import bookmarks_panel, main_window
        from sdr_module.gui.settings_store import GuiSettings

        self.write(first_run_done="true", **values)
        for module in (main_window, bookmarks_panel):
            patcher = mock.patch.object(module, "GuiSettings", GuiSettings)
            patcher.start()
            self.addCleanup(patcher.stop)
        win = main_window.SDRMainWindow(demo_mode=False)
        for timer in (win._display_timer, win._status_timer, win._hotplug_timer):
            timer.stop()
        win._confirm_discard_recording = lambda *_a: True

        def close():
            win.close()
            win.deleteLater()
            QApplication.processEvents()

        self.addCleanup(close)
        return win

    def test_a_cw_pitch_beyond_32_bits_starts_at_the_default(self):
        for raw in ("2147483648", "99999999999999999999", "-99999999999999999999"):
            with self.subTest(raw=raw):
                win = self.window(audio__cw_pitch=raw)
                self.assertEqual(win._control_panel.cw_pitch(), 700)
        # In range but beyond the spin box: its nearest end, as before.
        win = self.window(audio__cw_pitch="100000")
        self.assertEqual(win._control_panel.cw_pitch(), 1000)

    def test_set_cw_pitch_takes_anything(self):
        panel = self.window()._control_panel
        for value, pitch in ((2**40, 1000), (-(2**40), 400), (float("nan"), 700)):
            panel.set_cw_pitch(value)
            self.assertEqual(panel.cw_pitch(), pitch)

    def test_damaged_switches_keep_their_defaults(self):
        win = self.window(
            audio_enabled="garbage",
            squelch__enabled="garbage",
            tuning__band_plan_mode="garbage",
            decoder__rds_auto="garbage",
            layout__spectrum_visible="garbage",
            tuning__keep_centered=["a", "b"],
            audio__balance_iq=["y"],
        )
        # (audio_enabled: no output here to tell; see test_booleans.)
        self.assertTrue(win._control_panel.is_squelch_enabled())
        self.assertTrue(win._band_plan_action.isChecked())
        self.assertTrue(win._rds_auto_action.isChecked())
        self.assertTrue(win._spectrum_action.isChecked())
        self.assertFalse(win._keep_centered_action.isChecked())
        self.assertFalse(win._control_panel.iq_options()[1])

    def test_a_huge_squelch_threshold_is_the_default(self):
        # ROB-08: 0 dBFS (the slider's end) kept the squelch shut for good.
        for raw in ("1e12", "50", "nan"):
            with self.subTest(raw=raw):
                with self.assertLogs(level="WARNING") as logs:
                    win = self.window(squelch_db=raw, squelch__auto="false")
                self.assertEqual(win._control_panel.get_squelch_db(), -80.0)
                self.assertTrue(any("squelch_db" in line for line in logs.output))

    def test_a_nan_gain_is_the_default(self):
        win = self.window(gain_db="nan")
        self.assertAlmostEqual(win._control_panel.gain_db(), 20.0, delta=1.0)

    def test_unknown_choices_are_logged(self):
        with self.assertLogs("sdr_module.gui.main_window", "WARNING") as logs:
            win = self.window(audio__noise_reduction="loud", audio__agc_speed="x")
        self.assertEqual(win._control_panel.noise_reduction(), "off")
        self.assertEqual(win._control_panel.audio_agc_speed(), "medium")
        self.assertTrue(any("audio/noise_reduction" in m for m in logs.output))

    def test_unreadable_display_settings_are_logged(self):
        with self.assertLogs("sdr_module.gui.main_window", "WARNING") as logs:
            self.window(spectrum_display="garbage{", waterfall_display="[[[")
        text = "\n".join(logs.output)
        self.assertIn("unreadable spectrum settings", text)
        self.assertIn("unreadable waterfall settings", text)


class TestSavedDeviceGain(_DeviceCase):
    """ROB-08: a damaged saved gain ({"db": NaN, "agc": "x"}) keeps the
    defaults instead of 0 dB with Tuner AGC on."""

    def test_nan_gain_and_a_non_bool_agc(self):
        panel = self.panel
        self.win._release_device()
        self.store.values["device/demo/gain"] = '{"db": NaN, "agc": "x"}'
        before = panel.gain_db()
        with self.assertLogs("sdr_module.gui.device_controls", "WARNING"):
            self.win._start_demo_mode()
        self.assertEqual(panel.gain_db(), before)
        self.assertFalse(panel.is_agc_enabled())


class TestRecordingsFolder(unittest.TestCase):
    """ROB-09: only a full path is a recordings folder; anything else is
    the default (recordings never land where the app was started)."""

    def folder(self, value):
        from sdr_module.gui import file_dialogs

        class Store:
            def get_str(self, key, default):
                return value

        return file_dialogs.recordings_folder(Store())

    def test_relative_and_damaged_values(self):
        from sdr_module.gui import file_dialogs

        default = file_dialogs.default_recordings_folder()
        for bad in ("garbage", "['a', 'b']", "rec/sub"):
            with self.subTest(bad=bad):
                with self.assertLogs("sdr_module.gui.file_dialogs", "WARNING"):
                    self.assertEqual(self.folder(bad), default)
        self.assertEqual(self.folder("/data/rec"), "/data/rec")
        self.assertEqual(self.folder("~/rec"), os.path.expanduser("~/rec"))
        self.assertEqual(self.folder(""), default)


class TestRecentFrequencies(unittest.TestCase):
    """ROB-13: saved recent frequencies outside 1 kHz-300 GHz are dropped,
    an unknown mode or FM deviation is left out of the entry; logged."""

    def test_damaged_entries(self):
        from sdr_module.gui.tuning import TunedEntry, TuningHistory

        raw = json.dumps(
            [
                {"freq_hz": 1e30, "mode": "AM"},
                {"freq_hz": 145290000, "mode": "XYZ"},
                {"freq_hz": 145290001, "mode": "FM", "deviation": "banana"},
                {"freq_hz": 7000e6, "mode": "FM", "deviation": "5 kHz"},
                {"freq_hz": 7.074e6, "mode": "USB"},
            ]
        )
        with self.assertLogs("sdr_module.gui.tuning", "WARNING"):
            entries = TuningHistory.entries_from_json(raw)
        self.assertEqual(
            entries,
            [
                TunedEntry(145290000.0, "", ""),
                TunedEntry(145290001.0, "FM", ""),
                TunedEntry(7000e6, "FM", "5 kHz"),
                TunedEntry(7.074e6, "USB", ""),
            ],
        )

    def test_the_names_agree_with_the_control_panel(self):
        require_pyqt6(self)
        from sdr_module.gui.control_panel import DEMOD_MODES, FM_DEVIATIONS
        from sdr_module.gui.tuning import ENTRY_DEVIATIONS, ENTRY_MODES

        self.assertEqual(ENTRY_MODES, tuple(name for name, _tip in DEMOD_MODES))
        self.assertEqual(ENTRY_DEVIATIONS, tuple(v for v, _t, _tip in FM_DEVIATIONS))


# --------------------------------------------------------------------------- #
# The Audio button says what is really going on (DEV-E2E-02, ROB-12)
# --------------------------------------------------------------------------- #
class TestAudioButtonStates(_SecondCase):
    demo = True

    def setUp(self):
        super().setUp()
        win = self.win
        win._audio = _FakeSink()
        win._audio.open = True
        win._audio_enabled = True
        win._audio_problem = None  # (as with an output device)
        win._sync_audio_ui()

    def shown(self):
        button = self.win._audio_button
        return button.text(), button.toolTip(), button.accessibleName()

    def test_listen_this_receiver_mutes_the_main_window_and_says_so(self):
        win, rx = self.win, self.rx
        rx.start()
        rx.set_listen("second")
        self.assertEqual(win._audio_state(), "second")
        text, tip, name = self.shown()
        self.assertEqual(text, " Muted (2nd RX)")
        self.assertIn("Listen in the Second Receiver window", tip)
        self.assertEqual(name, "Audio muted by the second receiver")
        self.assertFalse(win._mute_action.isChecked())  # (not muted by you)
        self.assertEqual(
            win._audio_status()[0],
            "Muted by the Second Receiver (Listen: This receiver)",
        )
        rx.set_listen("both")
        self.assertEqual(win._audio_state(), "on")
        self.assertEqual(self.shown()[0], " Audio")
        rx.set_listen("second")
        rx.close()  # (closing the window gives the main window its audio)
        self.settle()
        win._update_status()
        self.assertEqual(self.shown()[0], " Audio")

    def test_paused_while_the_signal_isnt_real_time(self):
        # ROB-12: "Audio" while nothing played (the receiver couldn't keep
        # up), and the hint gone from the status bar after 10 s.
        win = self.win
        win._realtime_hint_shown = True
        win._realtime_hint_text = "The demo device can't keep up in real time."
        win._update_status()  # (the status timer)
        self.assertEqual(win._audio_state(), "paused")
        text, tip, name = self.shown()
        self.assertEqual(text, " Paused")
        self.assertIn("can't keep up", tip)
        self.assertEqual(win._audio_button.property("tone"), "warning")
        win._realtime_hint_shown = False
        win._update_status()
        self.assertEqual(self.shown()[0], " Audio")
        self.assertFalse(win._audio_button.property("tone"))

    def test_the_new_texts_fit_the_button(self):
        button = self.win._audio_button
        for text in (" Muted (2nd RX)", " Paused"):
            self.assertGreaterEqual(
                button.minimumWidth(), button.fontMetrics().horizontalAdvance(text)
            )


# --------------------------------------------------------------------------- #
# Second receiver: Auto gain, rates and plot defaults (DEV-E2E-09, -10)
# --------------------------------------------------------------------------- #
class TestSecondReceiverDevices(_SecondCase):
    def test_auto_gain_reads_auto(self):
        rx = self.rx
        rx.start()  # (the demo device)
        rx._gain_auto.setChecked(True)
        rx._apply_gain()
        rx._show_gain()
        self.assertTrue(rx._gain_label.text().startswith("Auto"))
        rx._gain_auto.setChecked(False)
        self.assertTrue(rx._gain_label.text().endswith("dB"))
        self.assertFalse(rx._gain_label.text().startswith("Auto"))

    def test_the_main_windows_rates(self):
        from sdr_module.gui import device_settings
        from sdr_module.gui.second_receiver import SAMPLE_RATES

        for kind in ("demo", "rtlsdr", "hackrf"):
            self.assertEqual(SAMPLE_RATES[kind], device_settings.sample_rates_for(kind))
        self.assertIn(2.56e6, SAMPLE_RATES["hackrf"])
        self.assertIn(3.2e6, SAMPLE_RATES["hackrf"])

    def test_the_waterfall_starts_as_the_main_windows(self):
        shown = self.rx._waterfall.display_settings()
        self.assertEqual(shown.get("history_s"), 60.0)
        self.assertTrue(shown.get("auto_levels"))


if __name__ == "__main__":
    unittest.main()

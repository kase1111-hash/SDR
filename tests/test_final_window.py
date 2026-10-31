"""Final fixes to the window chrome, the devices and the saved settings
(F-window): the problems the end-to-end journeys found (DEV-E2E-*, ROB-*,
LISTEN-02, DEC-E2E-12, PLOTS-E2E-15) and the leftovers of the earlier
rounds. Each test failed before its fix.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest
tests/test_final_window.py``
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_device_settings import _DeviceCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QMessageBox


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


if __name__ == "__main__":
    unittest.main()

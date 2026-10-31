"""The Recording group finishes the job.

rx:UX-06: after Stop the group showed "Ready 00:00:00", with no sign of an
unsaved take and no Save or Discard; it listed 3 of the 5 formats.
rx:G-11 / mc:MC-U07: no automatic saving, dialogs didn't remember folders.
mc:MC-U08: "REC ARMED" even after a recording had captured samples.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
)

if HAS_PYQT6:
    from PyQt6.QtWidgets import QMessageBox


class _GroupCase(_WindowTestCase):
    demo = True

    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="sdr-group-")
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.win._recording_folder = os.path.join(self.folder, "cache")
        self.rec = os.path.join(self.folder, "rec")
        self.store.set("recording/folder", self.rec)
        self.panel = self.win._control_panel

    def record(self, frames=3):
        self.win._record_button.click()
        for _ in range(frames):
            self.win._update_display()
        self.win._record_button.click()


class TestTakeLine(_GroupCase):
    def test_unsaved_take_with_save_and_discard(self):
        self.assertEqual(self.panel.take_text(), "")
        self.record()
        text = self.panel.take_text()
        self.assertRegex(text, r"^Unsaved · 00:0\d · [\d.]+ [kMG]B @ 100\.100 MHz$")
        self.assertTrue(self.panel._take_save_btn.isVisibleTo(self.panel))
        self.assertTrue(self.panel._take_discard_btn.isVisibleTo(self.panel))
        self.assertEqual(self.panel._take_label.property("tone"), "warning")
        self.assertEqual(self.panel._record_status.text(), "Ready")
        # Save... saves it (File > Save Recording).
        target = os.path.join(self.folder, "t.sigmf-data")
        with mock.patch(
            "sdr_module.gui.main_window.QFileDialog.getSaveFileName",
            return_value=(target, ""),
        ):
            self.panel._take_save_btn.click()
        self.assertTrue(os.path.exists(target))
        self.assertEqual(self.panel.take_text(), "Saved t.sigmf-data")
        self.assertTrue(self.panel._take_folder_btn.isVisibleTo(self.panel))
        with mock.patch(
            "sdr_module.gui.file_dialogs.QDesktopServices.openUrl", return_value=True
        ) as opened:
            self.panel._take_folder_btn.click()
        self.assertEqual(opened.call_args[0][0].toLocalFile(), self.folder)

    def test_discard_asks_then_deletes(self):
        self.record()
        path = self.win._samples_buffer.path
        with mock.patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Cancel
        ):
            self.panel._take_discard_btn.click()
        self.assertTrue(os.path.exists(path))
        with mock.patch.object(
            QMessageBox, "question", return_value=QMessageBox.StandardButton.Discard
        ):
            self.panel._take_discard_btn.click()
        self.assertFalse(os.path.exists(path))
        self.assertEqual(self.win._buffer_sample_count(), 0)
        self.assertEqual(self.panel.take_text(), "")


class TestFormats(_GroupCase):
    def test_all_five_sigmf_first_and_changeable_while_recording(self):
        combo = self.panel._format_combo
        items = [combo.itemText(i) for i in range(combo.count())]
        self.assertEqual(len(items), 5)
        self.assertEqual(items[0], "SigMF, keeps frequency and rate")
        self.assertEqual(combo.currentIndex(), 0)
        self.win._record_button.click()
        self.assertTrue(combo.isEnabled())
        combo.setCurrentIndex(2)  # cs16
        self.assertEqual(self.win._panel_recording_extension(), ".cs16")
        self.win._record_button.click()


class TestAutoSave(_GroupCase):
    def test_stop_saves_to_the_folder_with_an_automatic_name(self):
        self.panel.set_auto_save(True)
        self.assertTrue(self.store.get_bool("recording/auto_save", False))
        self.record()
        names = sorted(os.listdir(self.rec))
        self.assertEqual(len(names), 2, names)
        data = [n for n in names if n.endswith(".sigmf-data")][0]
        self.assertRegex(data, r"^iq_100\.100MHz_2\.4MSps_\d{8}_\d{6}\.sigmf-data$")
        with open(os.path.join(self.rec, data[: -len("data")] + "meta")) as handle:
            meta = json.load(handle)
        self.assertEqual(meta["global"]["core:sample_rate"], 2.4e6)
        self.assertEqual(meta["captures"][0]["core:frequency"], 100.1e6)
        self.assertFalse(self.win._buffer_unsaved)
        self.assertTrue(self.panel.take_text().startswith("Saved iq_100.100MHz_"))
        self.assertIn("Saved the recording to", self.win._message_label.text())
        # A second take gets its own name, never overwriting the first.
        self.record()
        self.assertEqual(len(os.listdir(self.rec)), 4)

    def test_closing_with_a_take_running_saves_without_asking(self):
        from sdr_module.gui.main_window import SDRMainWindow

        win = self.win
        win._confirm_discard_recording = (
            lambda doing: SDRMainWindow._confirm_discard_recording(win, doing)
        )
        win._exec_dialog = mock.Mock(side_effect=AssertionError("asked"))
        self.panel.set_auto_save(True)
        win._record_button.click()
        win._update_display()
        win.close()
        self.assertEqual(len([n for n in os.listdir(self.rec) if "sigmf" in n]), 2)

    def test_disk_full_stop_saves_after_the_frame_and_keeps_the_warning(self):
        from collections import namedtuple

        self.panel.set_auto_save(True)
        self.panel._format_combo.setCurrentIndex(2)  # cs16: a real write
        win = self.win
        win._record_button.click()
        win._update_display()
        win._next_space_check = 0.0
        usage = namedtuple("Usage", "total used free")(10**12, 0, 420 * 10**6)
        with mock.patch.object(self.mw.shutil, "disk_usage", return_value=usage):
            win._update_display()
        self.assertFalse(win._recording)
        # Not inside the frame (no nested event loop in the display loop)...
        self.assertFalse(os.path.isdir(self.rec) and os.listdir(self.rec))
        self.settle()
        # ...but right after it.
        names = sorted(os.listdir(self.rec))
        self.assertEqual(len([n for n in names if n.endswith(".cs16")]), 1, names)
        text = win._message_label.text()
        self.assertIn("Recording stopped: only 420 MB", text)
        self.assertIn("It was saved to", text)
        self.assertEqual(win._message_label.property("tone"), "danger")

    def test_folder_button(self):
        other = os.path.join(self.folder, "other")
        os.makedirs(other)
        with mock.patch(
            "sdr_module.gui.main_window.file_dialogs.QFileDialog.getExistingDirectory",
            return_value=other,
        ):
            self.panel._folder_btn.click()
        self.assertEqual(self.store.get_str("recording/folder", ""), other)
        self.assertEqual(self.panel._folder_btn.text(), "other")
        self.assertIn(other, self.panel._folder_btn.toolTip())


class TestWaitingBadge(_GroupCase):
    def test_rec_waiting_after_capturing_rec_armed_before(self):
        win = self.win
        win._stop_acquisition()
        win._record_button.click()
        self.assertEqual(win._recording_label.text(), "REC ARMED")
        self.assertIn("Armed", self.panel._record_status.text())
        win._record_button.click()
        win._start_acquisition()
        win._record_button.click()
        win._update_display()
        win._stop_acquisition()  # stopped mid-recording
        self.assertRegex(win._recording_label.text(), r"^REC WAITING \d\d:\d\d:\d\d$")
        self.assertEqual(win._recording_label.property("tone"), "warning")
        self.assertIn("resumes", win._recording_label.toolTip())
        self.assertIn("Waiting", self.panel._record_status.text())
        win._start_acquisition()
        win._update_display()
        win._update_status()
        self.assertTrue(win._recording_label.text().startswith("REC "))
        self.assertNotIn("WAITING", win._recording_label.text())
        win._record_button.click()


if __name__ == "__main__":
    unittest.main()

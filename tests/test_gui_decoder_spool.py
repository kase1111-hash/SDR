"""K-D3: Export saves every message since Clear, not only the table's 1,000.

The Decoder tab's table (and Log) keep the newest 1,000 messages for speed.
Export saved only those, so a long session lost the rest. Now every
message is also appended to a spool file in the temporary folder
(``gui/decoder_spool.py``); Export saves them all, Clear and quitting
delete the spool, and a full disk is reported.
"""

import csv
import errno
import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication


def _settle(n=3):
    for _ in range(n):
        QApplication.processEvents()


def _rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


class _SpoolCase(unittest.TestCase):
    """A Decoder panel on its own, its spool in a temporary folder."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import decoder_panel, decoder_spool

        self.mod = decoder_panel
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = os.path.join(self.tmp.name, "spool")
        patch = mock.patch.object(decoder_spool, "spool_folder", lambda: self.folder)
        patch.start()
        self.addCleanup(patch.stop)
        self.panel = decoder_panel.DecoderPanel()
        self.panel.resize(420, 520)
        self.panel.show()
        _settle()
        self.infos = []
        box = mock.patch.object(
            decoder_panel.QMessageBox,
            "information",
            side_effect=lambda *a, **k: self.infos.append(a[2]),
        )
        box.start()
        self.addCleanup(box.stop)

    def tearDown(self):
        self.panel.discard_spool()
        self.panel.close()
        self.panel.deleteLater()
        _settle(1)
        self.tmp.cleanup()

    def add(self, count, start=0):
        for i in range(start, start + count):
            self.panel.add_message("POCSAG", f"{i} (F0)", f"page {i}")

    def export(self):
        path = os.path.join(self.tmp.name, "out.csv")
        count = self.panel._export_messages(path)
        return count, _rows(path)


class TestExportSavesEveryMessage(_SpoolCase):
    def test_every_message_since_clear_is_exported(self):
        p = self.panel
        p._max_messages = 5  # (1,000 in the app)
        self.add(12)
        self.assertEqual(p._table.rowCount(), 5)
        count, rows = self.export()
        self.assertEqual(count, 12)
        self.assertEqual(
            rows[0][:5], ["Date", "Time", "Protocol", "Address", "Content"]
        )
        self.assertEqual([r[3] for r in rows[1:]], [f"{i} (F0)" for i in range(12)])
        self.assertEqual(rows[1][4], "page 0")
        self.assertIn("Exported 12 messages", self.infos[-1])
        self.assertNotIn("newest", self.infos[-1])

    def test_the_lines_that_said_only_the_newest_are_kept(self):
        p = self.panel
        p._max_messages = 5
        self.add(8)
        # The table still shows the newest 5 and says so, but Export keeps
        # them all.
        self.assertIn("newest 5 of 8 messages", p._table.toolTip())
        self.assertIn("Export saves all 8", p._table.toolTip())
        self.assertNotIn("Export keep", p._table.toolTip())
        self.assertEqual(p._kept_label.text(), "5 (the newest)")
        self.assertEqual(
            p._export_btn.toolTip(),
            "Save all 8 decoded messages since Clear to a CSV file",
        )
        caption = p._kept_label.parent().findChildren(type(p._kept_label))
        tips = " ".join(label.toolTip() for label in caption)
        self.assertIn("Export saves every message since Clear", tips)
        self.assertNotIn("Export keep the newest", tips)

    def test_clear_starts_again_and_deletes_the_spool(self):
        p = self.panel
        p._max_messages = 5
        self.add(7)
        spool = p._spool.path
        self.assertTrue(spool and os.path.exists(spool))
        p.clear()
        self.assertFalse(os.path.exists(spool))
        self.add(3, start=100)
        count, rows = self.export()
        self.assertEqual(count, 3)
        self.assertEqual([r[3] for r in rows[1:]], ["100 (F0)", "101 (F0)", "102 (F0)"])

    def test_messages_from_a_recording_and_invalid_ones_are_exported_too(self):
        p = self.panel
        p._max_messages = 2
        p.add_message("FLEX", "7", "bad", valid=False, raw="a5")
        p.add_message("POCSAG", "1 (F0)", "from file", source="pocsag.sigmf-data")
        p.add_message("POCSAG", "2 (F0)", "live", freq_hz=152.24e6, sample_rate=2.4e6)
        count, rows = self.export()
        self.assertEqual(count, 3)
        self.assertEqual(rows[1][2:7], ["FLEX", "7", "bad", "False", "a5"])
        self.assertEqual(rows[2][2], "POCSAG (file)")
        self.assertEqual(rows[3][-2:], ["152.240000", "2.4"])

    def test_a_long_session_keeps_memory_bounded(self):
        p = self.panel
        p._max_messages = 50
        self.add(2000)
        self.assertEqual(len(p._messages), 50)
        self.assertEqual(p._spool.count, 2000)
        count, rows = self.export()
        self.assertEqual(count, 2000)
        self.assertEqual(len(rows), 2001)
        self.assertEqual(rows[-1][3], "1999 (F0)")

    def test_export_runs_as_a_job(self):
        # A long session's spool can be large: Export copies it in a worker
        # thread (with progress and Cancel), titled as its command.
        from sdr_module.gui import file_jobs

        calls = []
        real = file_jobs.run_job

        def spy(parent, title, label, work, *args, **kwargs):
            calls.append(title)
            return real(parent, title, label, work, *args, **kwargs)

        self.add(3)
        with mock.patch.object(file_jobs, "run_job", spy):
            count, _rows = self.export()
        self.assertEqual(count, 3)
        self.assertEqual(calls, ["Export Decoded Messages"])

    def test_a_cancelled_export_leaves_no_file(self):
        from sdr_module.gui import file_jobs

        self.add(3)
        path = os.path.join(self.tmp.name, "out.csv")
        with mock.patch.object(
            file_jobs,
            "run_job",
            lambda *_a, **_k: file_jobs.JobResult(cancelled=True),
        ):
            self.assertEqual(self.panel._export_messages(path), 0)
        self.assertFalse(os.path.exists(path))


class TestFullDisk(_SpoolCase):
    def fill_the_disk(self):
        """Make the spool's next write fail as on a full disk."""
        spool = self.panel._spool
        real = spool._open

        def full():
            handle = real()
            writer = mock.Mock(wraps=handle)
            writer.write.side_effect = OSError(errno.ENOSPC, "No space left on device")
            return writer

        spool._open = full

    def test_a_full_disk_is_reported_once_and_export_says_what_it_has(self):
        p = self.panel
        p._max_messages = 4
        self.add(6)
        shown = []
        p.spool_failed.connect(shown.append)
        self.fill_the_disk()
        with self.assertLogs("sdr_module.gui.decoder_spool", "WARNING") as logs:
            self.add(10, start=6)
        self.assertEqual(len(logs.records), 1)  # (one entry in Error History)
        self.assertIn("No space left on device", logs.records[0].getMessage())
        self.assertEqual(len(shown), 1)
        self.assertIn("No space left on device", shown[0])
        self.assertIn("newest 4", shown[0])
        # The table and Export say what is true now.
        self.assertIn(
            "the first 6 and the newest 4 of the 16 decoded messages",
            p._export_btn.toolTip(),
        )
        self.assertIn("No space left on device", p._table.toolTip())
        count, rows = self.export()
        # The 6 spooled before the disk filled, and the table's newest 4
        # (messages 12-15); 6-11 are lost and the box says so.
        self.assertEqual(count, 10)
        self.assertEqual(
            [r[3] for r in rows[1:]],
            [f"{i} (F0)" for i in list(range(6)) + list(range(12, 16))],
        )
        self.assertIn("Exported 10 of 16 messages", self.infos[-1])
        self.assertIn("6 messages could not be kept", self.infos[-1])
        self.assertIn("No space left on device", self.infos[-1])

    def test_clear_tries_again(self):
        p = self.panel
        self.fill_the_disk()
        with self.assertLogs("sdr_module.gui.decoder_spool", "WARNING"):
            self.add(2)
        self.assertTrue(p._spool.problem)
        p._spool._open = type(p._spool)._open.__get__(p._spool)
        p.clear()
        self.assertEqual(p._spool.problem, "")
        self.add(3)
        self.assertEqual(p._spool.count, 3)


class TestStaleSpools(unittest.TestCase):
    def test_a_crashed_sessions_spool_is_deleted(self):
        from sdr_module.gui.decoder_spool import MessageSpool

        with tempfile.TemporaryDirectory() as folder:
            stale = os.path.join(folder, "sdr-decoded-99999-deadbeef.csv")
            with open(stale, "w") as handle:
                handle.write("old\n")
            other = os.path.join(folder, "notes.csv")  # not ours
            with open(other, "w") as handle:
                handle.write("keep\n")
            spool = MessageSpool(folder)
            spool.append(["a", "b"])
            self.assertFalse(os.path.exists(stale))
            self.assertTrue(os.path.exists(other))
            # A spool in use is not touched by another session's cleanup.
            second = MessageSpool(folder)
            second.append(["c"])
            self.assertTrue(os.path.exists(spool.path))
            spool.discard()
            second.discard()
            self.assertEqual(os.listdir(folder), ["notes.csv"])


class TestQuittingDeletesTheSpool(_WindowTestCase):
    def test_closing_the_window(self):
        panel = self.win._decoder_panel
        with tempfile.TemporaryDirectory() as folder:
            from sdr_module.gui.decoder_spool import MessageSpool

            panel._spool = MessageSpool(folder)
            panel.add_message("POCSAG", "1 (F0)", "page")
            path = panel._spool.path
            self.assertTrue(os.path.exists(path))
            self.win.close()
            self.assertFalse(os.path.exists(path))

    def test_a_full_disk_shows_in_the_status_bar(self):
        # Once: the spool logged it already (one entry in Error History).
        panel = self.win._decoder_panel
        with self.assertNoLogs(level="WARNING"):
            panel.spool_failed.emit("Decoder: could not keep every message")
        self.assertIn("could not keep every message", self.win._message_label.text())
        self.assertEqual(self.win._message_label.property("tone"), "warning")

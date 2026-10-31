#!/usr/bin/env python3
"""
The last fixes to the Scanner, the bookmarks and CHIRP files, found by
using the finished app end to end (the "scan" area of the final wave).

Each test reproduces one reported problem: it failed before its fix.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_final_scan.py``
"""

import json
import logging
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _FakeSettings,
    _WindowTestCase,
    require_pyqt6,
)
from tests.test_gui_scanner import FRAME_S, _ScanWindowCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import QSettings, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication


def _settle(n=3):
    for _ in range(n):
        QApplication.processEvents()


# A settings file damaged by hand (or by another program): entries that
# aren't bookmarks at all, and bookmarks without a usable frequency.
_DAMAGED = [
    None,
    1,
    "x",
    ["a"],
    {"label": "no frequency"},
    {"label": "text", "freq_hz": "abc"},
    {"label": "zero", "freq_hz": 0},
    {"label": "nan", "freq_hz": float("nan")},
    {"label": "ok", "freq_hz": 100100000},
    {"label": 5, "freq_hz": 145290000.0},  # a bad name is still a bookmark
]


# --------------------------------------------------------------------------- #
# ROB-01: damaged saved bookmarks never stop the app starting
# --------------------------------------------------------------------------- #
class TestDamagedSavedBookmarks(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def _settings_file(self, value):
        """A GuiSettings over a throwaway INI file holding ``value`` as the
        saved bookmarks (as the app stores them: JSON text)."""
        from sdr_module.gui.settings_store import GuiSettings

        folder = tempfile.mkdtemp(prefix="sdr-bookmarks-")
        path = os.path.join(folder, "settings.ini")
        raw = QSettings(path, QSettings.Format.IniFormat)
        raw.setValue("bookmarks", value)
        raw.sync()
        settings = GuiSettings()
        settings._s = QSettings(path, QSettings.Format.IniFormat)
        return settings

    def test_get_bookmarks_skips_what_isnt_a_bookmark_and_says_so(self):
        settings = self._settings_file(json.dumps(_DAMAGED))
        with self.assertLogs("sdr_module.gui.settings_store", "WARNING") as logs:
            bookmarks = settings.get_bookmarks()
        # The bookmarks stay (a frequency that isn't a number reads 0 Hz, to
        # be edited); what isn't a bookmark goes. One line per kind.
        self.assertEqual(
            [b["label"] for b in bookmarks],
            ["no frequency", "text", "zero", "nan", "ok", 5],
        )
        self.assertEqual(
            [b["freq_hz"] for b in bookmarks],
            [0.0, 0.0, 0, 0.0, 100100000, 145290000.0],
        )
        self.assertEqual(len(logs.records), 2)
        self.assertIn("Ignored 4 damaged entries", logs.output[0])
        self.assertIn("3 saved bookmarks have no usable frequency", logs.output[1])

    def test_a_frequency_too_big_for_a_float_doesnt_stop_the_app(self):
        # (A whole number of 400 digits: json reads it, float() can't.)
        settings = self._settings_file(
            '[{"label": "big", "freq_hz": 1%s}]' % ("0" * 400)
        )
        with self.assertLogs("sdr_module.gui.settings_store", "WARNING"):
            bookmarks = settings.get_bookmarks()
        self.assertEqual(bookmarks, [{"label": "big", "freq_hz": 0.0}])

    def test_a_list_of_nothing_but_damage_gives_no_bookmarks(self):
        for value in ("[1, 2, 3]", '[null, 1, "x"]'):
            settings = self._settings_file(value)
            with self.assertLogs("sdr_module.gui.settings_store", "WARNING"):
                self.assertEqual(settings.get_bookmarks(), [])

    def test_good_bookmarks_load_without_a_warning(self):
        good = [{"label": "ok", "freq_hz": 100.1e6, "demod": "FM"}]
        settings = self._settings_file(json.dumps(good))
        logger = logging.getLogger("sdr_module.gui.settings_store")
        with self.assertNoLogs(logger, "WARNING"):
            self.assertEqual(settings.get_bookmarks(), good)

    def test_the_bookmarks_panel_skips_what_isnt_a_bookmark(self):
        from sdr_module.gui import bookmarks_panel

        store = _FakeSettings()
        store.bookmarks = list(_DAMAGED)
        saved = bookmarks_panel.GuiSettings
        bookmarks_panel.GuiSettings = lambda: store
        try:
            with self.assertLogs("sdr_module.gui.settings_store", "WARNING"):
                panel = bookmarks_panel.BookmarksPanel()
        finally:
            bookmarks_panel.GuiSettings = saved
        try:
            self.assertEqual(panel.bookmark_count(), 6)
            self.assertEqual(panel.groups(), [])
            self.assertEqual(panel._list.topLevelItemCount(), 6)
            self.assertEqual(
                panel._list.topLevelItem(3).text(1), "0.000 MHz"
            )  # NaN: listed at 0 Hz
        finally:
            panel.deleteLater()
            _settle()


class TestTheWindowStartsWithDamagedBookmarks(_WindowTestCase):
    """The window (as SDRApplication.run builds it) starts and keeps the
    good bookmark."""

    def setUp(self):
        require_pyqt6(self)
        from tests import test_gui_chrome

        # Seed the in-memory settings before the window reads them.
        original = test_gui_chrome._FakeSettings

        class _Damaged(original):
            def __init__(self):
                super().__init__()
                self.bookmarks = [None, 1, "x", {"label": "ok", "freq_hz": 100.1e6}]

        test_gui_chrome._FakeSettings = _Damaged
        try:
            with self.assertLogs("sdr_module.gui.settings_store", "WARNING"):
                super().setUp()
        finally:
            test_gui_chrome._FakeSettings = original

    def test_starts_with_the_good_bookmark(self):
        panel = self.win._bookmarks_panel
        self.assertEqual([b["label"] for b in panel.bookmarks()], ["ok"])
        self.assertEqual(self.win._scanner_host.scan_bookmark_counts(), (1, 0))


# --------------------------------------------------------------------------- #
# SCAN-01: typing a frequency while a scan runs
# --------------------------------------------------------------------------- #
class TestTypingAFrequencyDuringAScan(_ScanWindowCase):
    demo = True

    def _sweep(self):
        panel = self.win._scanner_panel
        panel._preset.setCurrentIndex(1)  # FM broadcast
        panel.set_scan_mode("continuous")
        self.assertTrue(panel.start_scan())
        self.frames(0.2)
        return panel

    def _type(self, field, keys, seconds_between=0.3):
        """Type at a person's speed: the display loop runs between keys."""
        for key in keys:
            QTest.keyClicks(field, key)
            self.frames(seconds_between)

    def test_typing_stops_the_scan_and_enter_tunes_what_was_typed(self):
        win = self.win
        panel = self._sweep()
        field = win._frequency_field()
        line = field.lineEdit()
        win._focus_frequency_entry()  # Ctrl+L
        self.settle()
        self.assertTrue(field.hasFocus())
        # The scan goes on retuning (and FREQ follows it) until a key is
        # pressed; the field stays selected, ready to be typed over.
        tuned = set()
        for _ in range(6):
            self.frames(FRAME_S * 3)
            tuned.add(round(win._current_frequency()))
            self.assertEqual(line.selectedText(), line.text())
        self.assertGreater(len(tuned), 1)
        self._type(field, "8")
        self.assertFalse(panel.engine.active)  # the first key stopped it
        self.assertEqual(
            panel._status.text(), "Scan stopped: you are typing a frequency."
        )
        self._type(field, "9.3")
        self.assertEqual(line.text(), "89.3")
        QTest.keyClick(field, Qt.Key.Key_Return)
        self.settle()
        self.assertAlmostEqual(win._current_frequency(), 89.3e6)
        self.assertFalse(panel.engine.active)

    def test_a_retune_while_typing_keeps_the_typed_text(self):
        win = self.win
        field = win._frequency_field()
        line = field.lineEdit()
        win._focus_frequency_entry()
        self.settle()
        QTest.keyClicks(field, "14")
        cursor = line.cursorPosition()
        # Something else retunes meanwhile (the scan's last step, Track
        # Signal, a device that moved): FREQ follows, the typing stays.
        self.assertTrue(win._tune_to(100.5e6))
        self.assertEqual(line.text(), "14")
        self.assertEqual(line.cursorPosition(), cursor)
        QTest.keyClicks(field, "5.29")
        QTest.keyClick(field, Qt.Key.Key_Return)
        self.settle()
        self.assertAlmostEqual(win._current_frequency(), 145.29e6)

    def test_esc_after_a_retune_shows_the_frequency_tuned(self):
        win = self.win
        field = win._frequency_field()
        win._focus_frequency_entry()
        self.settle()
        QTest.keyClicks(field, "14")
        self.assertTrue(win._tune_to(100.5e6))
        QTest.keyClick(field, Qt.Key.Key_Escape)
        self.settle()
        self.assertAlmostEqual(win._current_frequency(), 100.5e6)
        self.assertEqual(win._control_panel._freq_input.get_frequency(), 100.5e6)
        self.assertAlmostEqual(field.value(), 100.5)


# --------------------------------------------------------------------------- #
# SCAN-05: Listen on a bookmark-scan result keeps the bookmark's tone squelch
# --------------------------------------------------------------------------- #
class TestBookmarkResultsKeepTheirTone(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_the_engine_reports_the_bookmark_tone(self):
        from sdr_module.gui.scan_engine import LISTEN, ScanEngine, ScanSettings
        from tests.test_gui_scanner import _demo, _DemoHost, _run

        dev = _demo(145.29e6)
        host = _DemoHost(dev, active=[145.29e6])
        engine = ScanEngine(host)
        found = []
        engine.on_found = found.append
        visit = {
            "index": 0,
            "label": "Rpt 100",
            "freq_hz": 145.29e6,
            "demod": "FM",
            "fm_deviation": "5 kHz",
            "bandwidth": "",
            "tone_hz": 100.0,
        }
        settings = ScanSettings(mode="bookmarks", bookmarks=[visit])
        self.assertEqual(engine.start(settings, dev.sample_rate, 0.0), "")
        _run(engine, dev, 2.0, until=lambda: engine.state == LISTEN)
        engine.stop()
        self.assertEqual([(f.label, f.tone_hz) for f in found], [("Rpt 100", 100.0)])

    def test_results_keep_the_tone_and_two_tones_are_two_results(self):
        from sdr_module.gui.scan_engine import Found
        from sdr_module.gui.scanner_panel import ScanHit, ScannerPanel
        from tests.test_gui_scanner import FakeWindow

        store = _FakeSettings()
        panel = ScannerPanel(settings=store)
        panel.set_host(FakeWindow())
        try:
            panel.set_scan_mode("bookmarks")
            panel.engine.settings.mode = "bookmarks"
            listening = ("FM", "5 kHz", "")
            for tone in (100.0, 88.5, 100.0):
                panel._on_found(
                    Found(145.29e6, -50.0, -90.0, 10e3, listening, "Rpt", tone_hz=tone)
                )
            hits = panel.hits()
            self.assertEqual(sorted(h.tone_hz for h in hits), [88.5, 100.0])
            self.assertEqual([h.count for h in hits if h.tone_hz == 100.0], [2])
            # Kept between sessions (and a damaged saved tone means none).
            again = ScanHit.from_dict(hits[0].to_dict())
            self.assertEqual(again.tone_hz, hits[0].tone_hz)
            data = dict(hits[0].to_dict(), tone_hz="nan")
            self.assertEqual(ScanHit.from_dict(data).tone_hz, 0.0)
            self.assertEqual(ScanHit.from_dict({"frequency_hz": 1e8}).tone_hz, 0.0)
        finally:
            panel.engine.stop()
            panel.deleteLater()
            _settle()


class TestListenToABookmarkResult(_ScanWindowCase):
    demo = True

    def test_listen_restores_the_bookmark_tone_squelch(self):
        from sdr_module.gui.scan_engine import LISTEN

        win, panel = self.win, self.win._scanner_panel
        bookmarks = win._bookmarks_panel
        index = bookmarks.add_bookmark("Rpt 100", 145.29e6, "FM", "5 kHz")
        bookmarks.set_tone(index, 100.0)  # the demo repeater's tone
        panel.set_scan_mode("bookmarks")
        self.assertTrue(panel.start_scan())
        self.assertTrue(self.frames(4.0, until=lambda: panel.engine.state == LISTEN))
        panel.stop_scan()
        self.assertEqual([h.label for h in panel.hits()], ["Rpt 100"])
        # Somewhere else, with no tone squelch; then Listen to the result.
        win.set_frequency(100.1e6)
        win._control_panel.set_tone_squelch_hz(0.0)
        messages = []
        real = win._show_status_message
        win._show_status_message = lambda text, *a, **k: (
            messages.append(text),
            real(text, *a, **k),
        )
        panel._table.selectRow(0)
        panel.listen_selected()
        self.assertAlmostEqual(win._current_frequency(), 145.29e6)
        self.assertEqual(win._control_panel.tone_squelch_hz(), 100.0)
        self.assertIn("Tuned to Rpt 100 (FM, tone squelch 100.0 Hz)", messages)


# --------------------------------------------------------------------------- #
# SCAN-06: Bookmark All asks first; many bookmarks are removed at once
# --------------------------------------------------------------------------- #
class TestBookmarkAllAndBulkRemove(_ScanWindowCase):
    demo = True

    def _hits(self, count):
        from sdr_module.gui.scanner_panel import ScanHit

        return [
            ScanHit(440.0e6 + i * 25e3, -60.0, 20.0, 5e3, "FM", "5 kHz", "25 kHz")
            for i in range(count)
        ]

    def test_many_results_ask_first_and_go_in_a_group(self):
        win, panel = self.win, self.win._scanner_panel
        bookmarks = win._bookmarks_panel
        panel._hits = self._hits(30)
        panel._rebuild_table()
        asked = []
        panel._confirm_bookmark_all = lambda count, group: (
            asked.append((count, group)),
            False,
        )[1]
        panel.bookmark_all()
        self.assertEqual(bookmarks.bookmark_count(), 0)  # cancelled
        self.assertEqual(asked[0][0], 30)
        self.assertTrue(asked[0][1].startswith("Scan "))
        panel._confirm_bookmark_all = lambda count, group: True
        panel.bookmark_all()
        self.assertEqual(bookmarks.bookmark_count(), 30)
        group = asked[0][1][:15]  # "Scan YYYY-MM-DD"
        self.assertTrue(
            all(b["group"].startswith(group) for b in bookmarks.bookmarks())
        )
        self.assertIn("in the group", panel._status.text())
        # A few results: no question, no group (as before).
        bookmarks._bookmarks = []
        bookmarks._save()
        panel._hits = self._hits(3)
        panel._rebuild_table()
        panel._confirm_bookmark_all = lambda count, group: self.fail("asked")
        panel.bookmark_all()
        self.assertEqual(bookmarks.bookmark_count(), 3)
        self.assertNotIn("group", bookmarks.bookmark(0))

    def test_a_group_is_removed_with_one_question(self):
        bookmarks = self.win._bookmarks_panel
        bookmarks.add_bookmark("Mine", 145.5e6, "FM", "5 kHz")
        bookmarks.add_unique_many(
            [(f"S{i}", 440e6 + i * 25e3, "FM", "5 kHz", "") for i in range(25)],
            group="Scan 2026-10-31 22:00",
        )
        self.assertEqual(bookmarks.groups(), ["Scan 2026-10-31 22:00"])
        folder = bookmarks._list.topLevelItem(0)
        self.assertEqual(folder.childCount(), 25)
        menu = bookmarks._group_menu(folder.text(0))
        remove = next(a for a in menu.actions() if a.text().startswith("Remove Group"))
        asked = []
        bookmarks._confirm_remove_many = lambda count, group="": (
            asked.append((count, group)),
            True,
        )[1]
        remove.trigger()
        menu.deleteLater()
        self.assertEqual(asked, [(25, "Scan 2026-10-31 22:00")])
        self.assertEqual([b["label"] for b in bookmarks.bookmarks()], ["Mine"])

    def test_several_selected_bookmarks_are_removed_with_one_question(self):
        from PyQt6.QtWidgets import QAbstractItemView

        bookmarks = self.win._bookmarks_panel
        for i in range(5):
            bookmarks.add_bookmark(f"B{i}", 146.5e6 + i * 25e3, "FM", "5 kHz")
        tree = bookmarks._list
        self.assertEqual(
            tree.selectionMode(), QAbstractItemView.SelectionMode.ExtendedSelection
        )
        tree.clearSelection()
        for row in (1, 2, 4):
            tree.topLevelItem(row).setSelected(True)
        self.assertEqual(bookmarks._selected_indices(), [1, 2, 4])
        self.assertTrue(bookmarks._remove_btn.isEnabled())
        self.assertFalse(bookmarks._tune_btn.isEnabled())  # which one?
        self.assertIn("3 selected", bookmarks._remove_btn.toolTip())
        asked = []
        bookmarks._confirm_remove_many = lambda count, group="": (
            asked.append(count),
            True,
        )[1]
        bookmarks._confirm_remove = lambda _b: self.fail("asked one by one")
        tree.setFocus()
        QTest.keyClick(tree, Qt.Key.Key_Delete)
        self.assertEqual(asked, [3])
        self.assertEqual([b["label"] for b in bookmarks.bookmarks()], ["B0", "B3"])
        # The right-click menu of a selection removes the selection too.
        tree.clearSelection()
        for row in (0, 1):
            tree.topLevelItem(row).setSelected(True)
        menu = bookmarks._context_menu(1)
        remove = next(a for a in menu.actions() if a.text().startswith("Remove"))
        self.assertTrue(remove.text().startswith("Remove 2 Bookmarks"))
        remove.trigger()
        menu.deleteLater()
        self.assertEqual(bookmarks.bookmarks(), [])


if __name__ == "__main__":
    unittest.main()

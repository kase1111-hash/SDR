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


# --------------------------------------------------------------------------- #
# SCAN-02: at 1366x768 the results' buttons are in view after a scan
# --------------------------------------------------------------------------- #
class TestResultsButtonsInView(_ScanWindowCase):
    demo = True

    def setUp(self):
        super().setUp()
        self.win.resize(1366, 768)
        self.win._show_panel("scanner")
        self.settle()
        # The tool tabs' default share of a 1366x768 window (305 px; the
        # tab's scroll area shows 276 of them).
        splitter = self.win._right_splitter
        total = sum(splitter.sizes())
        splitter.setSizes([total - 305, 305])
        self.settle()

    def _area(self):
        from PyQt6.QtWidgets import QScrollArea

        area = self.win._scanner_panel.parentWidget()
        while area is not None and not isinstance(area, QScrollArea):
            area = area.parentWidget()
        return area

    def _shown(self, *widgets):
        """The widgets wholly inside the tab's visible part."""
        from PyQt6.QtCore import QPoint

        viewport = self._area().viewport()
        out = []
        for widget in widgets:
            top = widget.mapTo(viewport, QPoint(0, 0)).y()
            out.append(top >= 0 and top + widget.height() <= viewport.height())
        return out

    def test_after_a_sweep_the_status_and_the_buttons_are_in_view(self):
        panel = self.win._scanner_panel
        self.assertLess(self._area().viewport().height(), 300)  # (a short tab)
        panel._preset.setCurrentIndex(1)  # FM broadcast
        panel.set_scan_mode("once")
        self.assertTrue(panel.start_scan())
        self.frames(6.0, until=lambda: not panel.engine.active)
        self.settle()
        self.assertGreater(len(panel.hits()), 10)
        self.assertEqual(
            self._shown(
                panel._run_row,
                panel._listen_btn,
                panel._bookmark_all_btn,
                panel._results_btn,
            ),
            [True] * 4,
        )

    def test_while_listening_skip_hold_and_the_buttons_are_in_view(self):
        from sdr_module.gui.scan_engine import LISTEN

        panel = self.win._scanner_panel
        panel._preset.setCurrentIndex(1)
        panel.set_scan_mode("stop")
        self.assertTrue(panel.start_scan())
        self.assertTrue(self.frames(15.0, until=lambda: panel.engine.state == LISTEN))
        self.settle()
        self.assertEqual(
            self._shown(panel._skip_btn, panel._hold_btn, panel._listen_btn),
            [True] * 3,
        )
        panel.stop_scan()


# --------------------------------------------------------------------------- #
# SCAN-03: a running scan shows (and can be stopped) from any tab
# --------------------------------------------------------------------------- #
class TestScanChipInTheStatusBar(_ScanWindowCase):
    demo = True

    def _chip(self):
        chip = self.win._scanner_panel.status_chip()
        self.assertIn(chip, self.win.statusBar().findChildren(type(chip)))
        return chip

    def _action(self, chip, text):
        return next(
            a for a in chip.menu().actions() if a.text().replace("&", "") == text
        )

    def test_the_chip_shows_while_scanning_and_stops_the_scan(self):
        win, panel = self.win, self.win._scanner_panel
        chip = self._chip()
        self.assertTrue(chip.isHidden())
        panel._preset.setCurrentIndex(1)
        panel.set_scan_mode("continuous")
        self.assertTrue(panel.start_scan())
        win._show_panel("bookmarks")  # (elsewhere)
        self.frames(0.5)
        self.assertFalse(chip.isHidden())
        self.assertTrue(chip.text().startswith("Scanning"))
        self.assertIn("Stop", chip.toolTip())
        self.assertFalse(self._action(chip, "Skip").isEnabled())  # (sweeping)
        self._action(chip, "Show Scanner").trigger()
        self.assertEqual(win._current_panel_key(), "scanner")
        win._show_panel("bookmarks")
        self._action(chip, "Stop Scan").trigger()
        self.assertFalse(panel.engine.active)
        self.assertTrue(chip.isHidden())
        # (Stopped from another tab: the status bar says so.)
        self.assertIn("Scan stopped", win._message_label.text())

    def test_skip_and_hold_from_the_chip(self):
        from sdr_module.gui.scan_engine import LISTEN

        panel = self.win._scanner_panel
        chip = self._chip()
        panel._preset.setCurrentIndex(1)
        panel.set_scan_mode("stop")
        self.assertTrue(panel.start_scan())
        self.assertTrue(self.frames(15.0, until=lambda: panel.engine.state == LISTEN))
        self.assertIn("88.500 MHz", chip.text())
        hold = self._action(chip, "Hold")
        self.assertTrue(hold.isEnabled())
        hold.trigger()
        self.assertTrue(panel.engine.held)
        self.assertTrue(panel._hold_btn.isChecked())
        self.assertIn("held", chip.text())
        heard = panel.engine.current.frequency_hz
        self._action(chip, "Skip").trigger()
        self.assertFalse(panel.engine.held)
        self.frames(0.5)
        self.assertNotEqual(getattr(panel.engine.current, "frequency_hz", 0), heard)
        panel.stop_scan()
        self.assertTrue(chip.isHidden())


# --------------------------------------------------------------------------- #
# SCAN-04: bookmark-scan results show the bookmark's name
# --------------------------------------------------------------------------- #
class TestBookmarkResultsShowTheirNames(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_the_frequency_cell_names_the_bookmark_and_sorts_by_frequency(self):
        from sdr_module.gui.scan_engine import Found
        from sdr_module.gui.scanner_panel import COL_FREQ, ScannerPanel
        from tests.test_gui_scanner import FakeWindow

        panel = ScannerPanel(settings=_FakeSettings())
        panel.set_host(FakeWindow())
        try:
            panel.engine.settings.mode = "bookmarks"
            fm = ("FM", "5 kHz", "")
            for freq, name, tone in (
                (162.4e6, "NOAA 1", 0.0),
                (145.29e6, "Rpt 100", 100.0),
                (145.29e6, "Rpt 88.5", 88.5),
                (125.0e6, "Tower", 0.0),
            ):
                found = Found(freq, -50.0, -90.0, 10e3, fm, name, tone_hz=tone)
                panel._on_found(found)
            table = panel._table
            table.sortItems(COL_FREQ)
            texts = [table.item(r, COL_FREQ).text() for r in range(table.rowCount())]
            # (Sorted by frequency, whatever the names.)
            self.assertEqual(texts[0], "Tower · 125.000 MHz")
            self.assertEqual(texts[-1], "NOAA 1 · 162.400 MHz")
            self.assertEqual(
                sorted(texts[1:3]), ["Rpt 100 · 145.290 MHz", "Rpt 88.5 · 145.290 MHz"]
            )
            # A sweep's results are just the frequency, as before.
            panel.engine.settings.mode = "once"
            panel._hits = []
            panel._rebuild_table()
            panel._on_found(Found(100.1e6, -40.0, -90.0, 150e3, fm))
            self.assertEqual(table.item(0, COL_FREQ).text(), "100.100 MHz")
        finally:
            panel.engine.stop()
            panel.deleteLater()
            _settle()


# --------------------------------------------------------------------------- #
# SCAN-07: Export Results starts in Documents, then the folder used last
# --------------------------------------------------------------------------- #
class TestExportResultsFolder(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_documents_first_then_the_folder_used_last(self):
        from unittest import mock

        from sdr_module.gui import file_dialogs
        from sdr_module.gui.scanner_panel import ScanHit, ScannerPanel
        from tests.test_gui_scanner import FakeWindow

        panel = ScannerPanel(settings=_FakeSettings())
        panel.set_host(FakeWindow())
        panel._hits = [ScanHit(100.1e6, -40.0, 40.0, 150e3, "FM", "75 kHz")]
        panel._rebuild_table()
        offered = []
        folder = tempfile.mkdtemp(prefix="scan-export-")

        def save(_parent, _caption, start, _filters, _selected=""):
            offered.append(start)
            return os.path.join(folder, "mine.csv"), "CSV (*.csv)"

        try:
            target = "sdr_module.gui.file_dialogs.QFileDialog.getSaveFileName"
            with mock.patch(target, side_effect=save):
                self.assertEqual(panel.export_csv(), 1)
                self.assertEqual(panel.export_csv(), 1)
            self.assertEqual(
                os.path.dirname(offered[0]), file_dialogs.documents_folder()
            )
            self.assertTrue(os.path.basename(offered[0]).startswith("scan-"))
            self.assertEqual(os.path.dirname(offered[1]), folder)
            self.assertTrue(os.path.exists(os.path.join(folder, "mine.csv")))
        finally:
            panel.deleteLater()
            _settle()


# --------------------------------------------------------------------------- #
# The Scanner tab's smaller problems (SCAN-12, 13, 14, 19, 21, 24, 26)
# --------------------------------------------------------------------------- #
class _ScanPanelCase(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from tests.test_gui_scanner import FakeWindow

        self.store = _FakeSettings()
        self.host = FakeWindow()
        self._panels = []

    def tearDown(self):
        for panel in self._panels:
            panel.engine.stop()
            panel.close()
            panel.deleteLater()
        _settle()

    def panel(self, width=394):
        from sdr_module.gui.scanner_panel import ScannerPanel

        panel = ScannerPanel(settings=self.store)
        panel.set_host(self.host)
        panel.resize(width, 560)
        panel.show()
        _settle()
        self._panels.append(panel)
        return panel

    def run_scan(self, panel, seconds=10.0):
        dev = self.host.dev
        for _ in range(int(seconds / FRAME_S)):
            if not panel.engine.active:
                break
            panel.engine.frame(dev.read_samples(int(dev.sample_rate * FRAME_S)))


class TestScannerTabDetails(_ScanPanelCase):
    def test_scan_13_one_bookmark_marked_skip_is_passed_by(self):
        self.host.bookmarks = [
            {"index": 0, "label": "A", "freq_hz": 145.29e6, "demod": "FM"},
        ]
        self.host.scan_bookmark_counts = lambda: (2, 1)
        panel = self.panel()
        panel.set_scan_mode("bookmarks")
        self.assertIn("1 marked Skip is passed by.", panel._hint.text())
        self.host.scan_bookmark_counts = lambda: (3, 2)
        panel.bookmarks_changed()
        self.assertIn("2 marked Skip are passed by.", panel._hint.text())

    def test_scan_14_copy_frequency_says_so_in_the_status_bar(self):
        from sdr_module.gui.scanner_panel import ScanHit

        panel = self.panel()
        panel._hits = [ScanHit(104.3e6, -40.0, 40.0, 150e3, "FM", "75 kHz")]
        panel._rebuild_table()
        panel._copy("104.300 MHz")
        self.assertEqual(self.host.messages[-1], ("Copied 104.300 MHz", "info"))
        self.assertEqual(QApplication.clipboard().text(), "104.300 MHz")

    def test_scan_19_the_progress_bar_empties_when_a_scan_stops(self):
        panel = self.panel()
        panel._preset.setCurrentIndex(1)  # FM broadcast
        panel.set_scan_mode("continuous")
        self.assertTrue(panel.start_scan())
        self.run_scan(panel, 0.6)
        self.assertGreater(panel._progress.value(), 0)
        panel.stop_scan()
        self.assertEqual(panel._progress.value(), 0)
        # A sweep that ran to its end shows a full bar, as before.
        panel.set_scan_mode("once")
        self.assertTrue(panel.start_scan())
        self.run_scan(panel)
        self.assertEqual(panel._progress.value(), 1000)

    def test_scan_24_a_header_click_during_a_scan_sorts_the_results(self):
        from PyQt6.QtCore import QPoint

        from sdr_module.gui.scanner_panel import COL_LEVEL

        panel = self.panel()
        panel._preset.setCurrentIndex(1)
        panel.set_scan_mode("continuous")
        self.assertTrue(panel.start_scan())
        self.run_scan(panel, 0.35)
        table = panel._table
        self.assertGreater(table.rowCount(), 2)
        self.assertLess(table.rowCount(), 18)  # (more to come)
        header = table.horizontalHeader()

        def levels():
            return [
                float(table.item(r, COL_LEVEL).data(Qt.ItemDataRole.UserRole))
                for r in range(table.rowCount())
            ]

        x = header.sectionViewportPosition(COL_LEVEL) + 5
        QTest.mouseClick(header.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(x, 5))
        self.assertEqual(header.sortIndicatorSection(), COL_LEVEL)
        order = header.sortIndicatorOrder()
        expected = sorted(levels(), reverse=order == Qt.SortOrder.DescendingOrder)
        self.assertEqual(levels(), expected)
        # New results join in order (the tab's refresh keeps it sorted).
        count = table.rowCount()
        self.run_scan(panel, 1.5)
        panel.refresh()
        self.assertGreater(table.rowCount(), count)
        expected = sorted(levels(), reverse=order == Qt.SortOrder.DescendingOrder)
        self.assertEqual(levels(), expected)
        self.assertTrue(panel.engine.active)

    def test_scan_26_scan_from_here_leaves_the_tabs_own_mode_and_group(self):
        self.host.groups = ["Wx"]
        self.host.bookmarks = [
            {
                "index": 0,
                "label": "NOAA 1",
                "freq_hz": 162.4e6,
                "demod": "FM",
                "group": "Wx",
            },
            {
                "index": 1,
                "label": "NOAA 2",
                "freq_hz": 162.55e6,
                "demod": "FM",
                "group": "Wx",
            },
        ]
        panel = self.panel()
        panel.set_scan_mode("continuous")
        panel.set_scan_mode("bookmarks")
        panel._group.setCurrentIndex(panel._group.findData("Wx"))
        panel.set_scan_mode("continuous")
        self.assertEqual(self.store.values["scanner/bookmark_group"], "Wx")
        self.assertEqual(self.store.values["scanner/mode"], "continuous")
        self.assertTrue(panel.scan_bookmarks_from(1))
        self.assertEqual(panel.engine.mode, "bookmarks")
        self.assertEqual(panel.scan_mode(), "bookmarks")  # (what runs, shown)
        self.assertFalse(panel._group.isEnabled())
        panel.stop_scan()
        # Back as the user left them, and never saved otherwise.
        self.assertEqual(panel.scan_mode(), "continuous")
        self.assertEqual(self.store.values["scanner/mode"], "continuous")
        self.assertEqual(self.store.values["scanner/bookmark_group"], "Wx")
        panel.set_scan_mode("bookmarks")
        self.assertEqual(panel._group.currentData(), "Wx")


class TestScannerCombosFit(_ScanPanelCase):
    def _fits(self, combo):
        from PyQt6.QtWidgets import QStyle, QStyleOptionComboBox

        option = QStyleOptionComboBox()
        combo.initStyleOption(option)
        field = combo.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            option,
            QStyle.SubControl.SC_ComboBoxEditField,
            combo,
        )
        metrics = combo.fontMetrics()
        return {
            combo.itemText(i): metrics.horizontalAdvance(combo.itemText(i))
            <= field.width()
            for i in range(combo.count())
        }

    def test_scan_12_band_mode_and_group_texts_fit_a_390_px_column(self):
        from sdr_module.gui.scanner_dialog import SCAN_PRESETS

        self.host.groups = []
        panel = self.panel(width=390)
        self.assertTrue(
            all(self._fits(panel._preset).values()), self._fits(panel._preset)
        )
        self.assertTrue(all(self._fits(panel._mode).values()), self._fits(panel._mode))
        panel.set_scan_mode("bookmarks")
        _settle()
        self.assertTrue(
            all(self._fits(panel._group).values()), self._fits(panel._group)
        )
        # The band's range is in its tooltip; the setting keeps the full name.
        noaa = next(p for p in SCAN_PRESETS if p.label.startswith("NOAA"))
        index = panel._preset.findData(noaa.label)
        self.assertEqual(panel._preset.itemText(index), "NOAA Weather")
        self.assertIn(
            "162.4", panel._preset.itemData(index, Qt.ItemDataRole.ToolTipRole)
        )
        panel._preset.setCurrentIndex(index)
        self.assertEqual(self.store.values["scanner/preset"], noaa.label)

    def test_a_band_saved_by_name_or_full_label_is_restored(self):
        from sdr_module.gui.scanner_dialog import SCAN_PRESETS

        airband = next(p for p in SCAN_PRESETS if p.label.startswith("Airband"))
        self.store.values["scanner/preset"] = airband.label  # (as saved before)
        panel = self.panel()
        self.assertEqual(panel.preset(), airband)


class TestBookmarkVisitsAreCached(_ScanPanelCase):
    def test_scan_21_refresh_doesnt_rebuild_the_visit_list(self):
        calls = []
        real = self.host.scan_bookmark_list

        def counting(group=""):
            calls.append(group)
            return real(group)

        self.host.scan_bookmark_list = counting
        self.host.bookmarks = [
            {"index": i, "label": f"B{i}", "freq_hz": 146e6 + i * 25e3, "demod": "FM"}
            for i in range(50)
        ]
        panel = self.panel()
        panel.set_scan_mode("bookmarks")
        calls.clear()
        for _ in range(5):
            panel.refresh()  # (the tab's 500 ms timer)
        self.assertLessEqual(len(calls), 1)
        # Changed bookmarks are seen at once.
        self.host.bookmarks = self.host.bookmarks[:10]
        panel.bookmarks_changed()
        self.assertIn("Visits 10 bookmarks", panel._hint.text())
        # A scan starts from the bookmarks as they are now.
        self.host.bookmarks = self.host.bookmarks[:3]
        self.assertTrue(panel.start_scan())
        self.assertEqual(len(panel.engine.settings.bookmarks), 3)


# --------------------------------------------------------------------------- #
# The Bookmarks tab and its Edit dialog (SCAN-08, 09, 15, 22, 23, DEV-E2E-18,
# leftover 18) and Ctrl+B (SCAN-10)
# --------------------------------------------------------------------------- #
class _BookmarksCase(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import bookmarks_panel

        self.mod = bookmarks_panel
        self.store = _FakeSettings()
        self._saved = bookmarks_panel.GuiSettings
        bookmarks_panel.GuiSettings = lambda: self.store
        self.panels = []
        self.messages = []

    def tearDown(self):
        self.mod.GuiSettings = self._saved
        for panel in self.panels:
            panel.close()
            panel.deleteLater()
        _settle()

    def make(self, bookmarks=(), width=400):
        self.store.bookmarks = [dict(b) for b in bookmarks]
        panel = self.mod.BookmarksPanel()
        panel.message.connect(lambda text, tone: self.messages.append((text, tone)))
        panel.resize(width, 420)
        panel.show()
        _settle()
        self.panels.append(panel)
        return panel


class TestBookmarksTabDetails(_BookmarksCase):
    RPT = {
        "label": "Rpt 100",
        "freq_hz": 145.29e6,
        "demod": "FM",
        "fm_deviation": "5 kHz",
        "mode": "FM",
        "tone_mode": "TSQL",
        "ctone": 100.0,
    }

    def test_scan_08_a_typed_name_isnt_lost_and_can_be_a_second_bookmark(self):
        panel = self.make([self.RPT])
        panel.set_mode_source(lambda: ("FM", "5 kHz"))
        panel.set_current_frequency(145.29e6)
        asked = []
        panel._confirm_add_another = lambda name, new, freq: (
            asked.append((name, new, freq)),
            False,
        )[1]
        panel._label_input.setText("Rpt 88.5")
        panel._add_btn.click()
        self.assertEqual(asked, [("Rpt 100", "Rpt 88.5", "145.290 MHz")])
        self.assertEqual(panel.bookmark_count(), 1)
        self.assertEqual(panel._label_input.text(), "Rpt 88.5")  # kept
        self.assertEqual(self.messages[-1][0], "Already bookmarked: Rpt 100")
        # Yes: a second bookmark on the channel (another tone's users).
        panel._confirm_add_another = lambda name, new, freq: True
        panel._add_btn.click()
        self.assertEqual(
            [b["label"] for b in panel.bookmarks()], ["Rpt 100", "Rpt 88.5"]
        )
        self.assertEqual(panel._label_input.text(), "")
        # No name typed: no question, as before.
        panel._confirm_add_another = lambda *a: self.fail("asked")
        panel._add_btn.click()
        self.assertEqual(panel.bookmark_count(), 2)

    def test_scan_23_the_name_placeholder_reads_in_full(self):
        panel = self.make([], width=400)
        field = panel._label_input
        need = field.fontMetrics().horizontalAdvance(field.placeholderText())
        # With the theme, the tab of a 1024-1366 px window gives the field
        # its minimum width (80 px): "Name (optional)" read "Name (op...".
        # (About 40 px go to the frame, the padding and the clear button.)
        self.assertLessEqual(need, field.minimumWidth() - 40)
        self.assertIn("optional", field.toolTip())

    def test_scan_15_the_csv_buttons_import_says_how_many(self):
        panel = self.make([])
        path = os.path.join(tempfile.mkdtemp(), "two.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                "Name,Frequency,Mode\nCafe 1,146.520000,FM\nWx,162.550000,FM\n"
            )
        self.assertEqual(panel.import_csv(path), 2)
        self.assertEqual(self.messages[-1], ("Imported 2 bookmarks", "success"))

    def test_scan_22_scan_and_export_follow_the_order_shown(self):
        from sdr_module.core.chirp_csv import list_order, read_chirp_csv
        from sdr_module.gui.bookmarks_panel import COL_NAME

        bookmarks = [
            {"label": "FM 100.1", "freq_hz": 100.1e6},
            {"label": "Rpt", "freq_hz": 145.29e6, "group": "Ham"},
            {"label": "NOAA", "freq_hz": 162.4e6, "group": "Weather"},
            {"label": "Tower", "freq_hz": 125.0e6},
        ]
        panel = self.make(bookmarks)
        tree = panel._list
        shown = []
        for i in range(tree.topLevelItemCount()):
            top = tree.topLevelItem(i)
            if top.childCount():
                shown += [top.child(c).text(COL_NAME) for c in range(top.childCount())]
            else:
                shown.append(top.text(COL_NAME))
        self.assertEqual(shown, ["Rpt", "NOAA", "FM 100.1", "Tower"])
        self.assertEqual(list_order(panel.bookmarks()), [1, 2, 0, 3])
        from unittest import mock

        path = os.path.join(tempfile.mkdtemp(), "out.csv")
        with mock.patch.object(self.mod.QMessageBox, "information"):
            panel.export_csv(path)  # (a box says it is done)
        names = [c.name for c in read_chirp_csv(path).channels]
        self.assertEqual(names, shown)

    def test_l18_editing_a_bookmark_above_6_ghz_keeps_its_frequency(self):
        from sdr_module.gui.bookmark_editor import BookmarkEditDialog

        b = {"label": "10 GHz beacon", "freq_hz": 10368.1e6, "demod": "USB"}
        dialog = BookmarkEditDialog(b, ("USB", ""), [])
        try:
            self.assertEqual(dialog.bookmark()["freq_hz"], 10368.1e6)
        finally:
            dialog.deleteLater()


class TestKeepTheReceiversMode(_BookmarksCase):
    def test_scan_09_an_own_bookmark_set_to_keep_has_no_chirp_mode(self):
        from sdr_module.core.chirp_csv import is_chirp_channel
        from sdr_module.gui.bookmark_editor import BookmarkEditDialog

        own = {
            "label": "FM Broadcast 99.5",
            "freq_hz": 99.5e6,
            "demod": "FM",
            "fm_deviation": "75 kHz",
            "mode": "WFM",
        }
        self.assertFalse(is_chirp_channel(own))
        self.assertFalse(is_chirp_channel(dict(own, tone_mode="TSQL", ctone=100.0)))
        dialog = BookmarkEditDialog(own, ("FM", "75 kHz"), [])
        try:
            dialog._mode.setCurrentIndex(0)  # Keep the receiver's mode
            edited = dialog.bookmark()
        finally:
            dialog.deleteLater()
        self.assertEqual(edited, {"label": "FM Broadcast 99.5", "freq_hz": 99.5e6})
        self.assertEqual(self.mod._details(edited), "")
        # A CHIRP channel keeps a CHIRP Mode: "Auto".
        chirp = {"label": "Rptr", "freq_hz": 146.94e6, "mode": "NFM", "location": 4}
        self.assertTrue(is_chirp_channel(chirp))
        self.assertTrue(is_chirp_channel({"label": "x", "freq_hz": 1e8, "mode": "FM"}))
        dialog = BookmarkEditDialog(chirp, ("FM", "5 kHz"), [])
        try:
            dialog._mode.setCurrentIndex(0)
            self.assertEqual(dialog.bookmark()["mode"], "Auto")
        finally:
            dialog.deleteLater()


class TestCtrlBSaysTheTone(_WindowTestCase):
    def test_scan_10_the_message_names_the_saved_tone(self):
        win = self.win
        win.set_frequency(146.52e6)
        win._set_demod_mode("FM", "5 kHz", "25 kHz")
        win._control_panel.set_tone_squelch_hz(100.0)
        win._bookmark_current_frequency()
        saved = self.store.bookmarks[-1]
        self.assertEqual((saved["tone_mode"], saved["ctone"]), ("TSQL", 100.0))
        self.assertIn("(FM, tone squelch 100.0 Hz)", win._message_label.text())

    def test_scan_15_file_import_says_it_once(self):
        from unittest import mock

        path = os.path.join(tempfile.mkdtemp(), "two.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("Name,Frequency\nA,146.520000\nB,162.550000\n")
        shown = []
        real = self.win._show_status_message
        self.win._show_status_message = lambda text, *a, **k: (
            shown.append(text),
            real(text, *a, **k),
        )
        target = "sdr_module.gui.file_dialogs.QFileDialog.getOpenFileName"
        with mock.patch(target, return_value=(path, "")):
            self.win._import_channels_csv()
        self.assertEqual(
            [t for t in shown if t.startswith("Imported")], ["Imported 2 bookmarks"]
        )
        self.assertEqual(self.win._current_panel_key(), "bookmarks")


# --------------------------------------------------------------------------- #
# CHIRP files (SCAN-16, 17, 18) and sdr-scan channels import (leftover 33)
# --------------------------------------------------------------------------- #
_NATIVE = (
    "Location,Name,Frequency,Duplex,Offset,Tone,rToneFreq,cToneFreq,DtcsCode,"
    "DtcsPolarity,RxDtcsCode,CrossMode,Mode,TStep,Skip,Power,Comment\n"
    "1,NFM TSQL,462.562500,,0.000000,TSQL,88.5,88.5,023,NN,023,Tone->Tone,NFM,"
    "5.00,,,\n"
    "2,FM rptr,146.940000,-,0.600000,,88.5,88.5,023,NN,023,Tone->Tone,FM,5.00,,,\n"
    "3,CROSS,146.940000,,0.000000,Cross,100.0,123.0,023,NN,023,Tone->Tone,FM,"
    "5.00,,,\n"
    "4,TO DCS,147.000000,,0.000000,Cross,100.0,123.0,023,NN,023,Tone->DTCS,FM,"
    "5.00,,,\n"
    "8,,,,,,,,,,,,,,,,\n"
    "9,,,,,,,,,,,,,,,,\n"
    "10,Zero,0,,,,,,,,,,FM,,,,\n"
)


class TestChirpFiles(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def channels(self):
        import io

        from sdr_module.core.chirp_csv import read_chirp_csv

        report = read_chirp_csv(io.StringIO(_NATIVE), strict=False)
        return {c.name: c.to_bookmark() for c in report.channels}, report

    def test_scan_16_nfm_is_heard_narrow(self):
        from sdr_module.gui.bookmarks_panel import listening

        marks, _report = self.channels()
        self.assertEqual(
            listening(marks["NFM TSQL"]), ("FM", "2.5 kHz", "12.5 kHz", 88.5)
        )
        self.assertEqual(listening(marks["FM rptr"]), ("FM", "5 kHz", "", 0.0))
        # A filter saved with the channel wins; the app's own FM is as before.
        wide = dict(marks["NFM TSQL"], bandwidth="25 kHz")
        self.assertEqual(listening(wide)[2], "25 kHz")
        own = {"label": "x", "freq_hz": 1.5e8, "demod": "FM", "fm_deviation": "2.5 kHz"}
        self.assertEqual(listening(own), ("FM", "2.5 kHz", "", 0.0))

    def test_scan_17_a_cross_channel_listens_for_its_receive_tone(self):
        from sdr_module.gui import bookmarks_panel
        from sdr_module.gui.bookmark_editor import bookmark_tone_hz

        marks, _report = self.channels()
        self.assertEqual(bookmark_tone_hz(marks["CROSS"]), 123.0)
        self.assertEqual(bookmarks_panel.listening(marks["CROSS"])[3], 123.0)
        for mode in ("->Tone", "DTCS->Tone"):
            self.assertEqual(
                bookmark_tone_hz(dict(marks["CROSS"], cross_mode=mode)), 123.0, mode
            )
        # Receiving with a DCS code isn't supported: no tone squelch, and the
        # tooltip says so.
        self.assertEqual(bookmark_tone_hz(marks["TO DCS"]), 0.0)
        tip = bookmarks_panel._tooltip(marks["TO DCS"])
        self.assertIn("not supported here", tip)
        self.assertIn("123 Hz", bookmarks_panel._tooltip(marks["CROSS"]))

    def test_scan_18_empty_memory_slots_are_not_unreadable_rows(self):
        marks, report = self.channels()
        self.assertEqual(len(marks), 4)
        self.assertEqual(report.skipped, ["line 8: missing or zero frequency"])


class TestCliImportSkipBad(unittest.TestCase):
    def test_l33_each_skipped_row_is_said_once(self):
        import contextlib
        import io
        from unittest import mock

        from sdr_module import cli
        from sdr_module.gui import settings_store

        path = os.path.join(tempfile.mkdtemp(), "bad.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("Name,Frequency\nGood,146.520000\nBad,abc\n")
        store = _FakeSettings()
        out, err = io.StringIO(), io.StringIO()
        logger = logging.getLogger("sdr_module.core.chirp_csv")
        with (
            mock.patch.object(settings_store, "GuiSettings", lambda: store),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
            self.assertNoLogs(logger, "WARNING"),
        ):
            rc = cli.main(["channels", "import", path, "--skip-bad"])
        self.assertEqual(rc, 0)
        said = (out.getvalue() + err.getvalue()).count("line 3")
        self.assertEqual(said, 1, out.getvalue() + err.getvalue())
        self.assertEqual([b["label"] for b in store.bookmarks], ["Good"])


class TestCliExportFollowsTheList(unittest.TestCase):
    def test_scan_22_channels_export_writes_the_list_order(self):
        import contextlib
        import io
        from unittest import mock

        from sdr_module import cli
        from sdr_module.core.chirp_csv import read_chirp_csv
        from sdr_module.gui import settings_store

        store = _FakeSettings()
        store.bookmarks = [
            {"label": "FM 100.1", "freq_hz": 100.1e6},
            {"label": "Rpt", "freq_hz": 145.29e6, "group": "Ham"},
            {"label": "Tower", "freq_hz": 125.0e6},
        ]
        path = os.path.join(tempfile.mkdtemp(), "out.csv")
        with (
            mock.patch.object(settings_store, "GuiSettings", lambda: store),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(cli.main(["channels", "export", path]), 0)
        names = [c.name for c in read_chirp_csv(path).channels]
        self.assertEqual(names, ["Rpt", "FM 100.1", "Tower"])


class TestBookmarkScanFollowsTheList(_ScanWindowCase):
    demo = True

    def test_scan_22_the_scan_visits_in_the_order_shown(self):
        win, panel = self.win, self.win._scanner_panel
        bookmarks = win._bookmarks_panel
        for label, freq, group in (
            ("FM 100.1", 100.1e6, ""),
            ("Rpt", 145.29e6, "Ham"),
            ("NOAA", 162.4e6, "Weather"),
            ("Tower", 125.0e6, ""),
        ):
            index = bookmarks.add_bookmark(label, freq, "FM", "5 kHz")
            if group:
                b = bookmarks._bookmarks[index]
                b["group"] = group
        bookmarks._save()
        visits = win._scanner_host.scan_bookmark_list()
        self.assertEqual(
            [v["label"] for v in visits], ["Rpt", "NOAA", "FM 100.1", "Tower"]
        )
        # Scan from Here on "FM 100.1" (stored first, listed third).
        self.assertTrue(panel.scan_bookmarks_from(0))
        self.assertEqual(panel.engine.current.label, "FM 100.1")
        panel.stop_scan()


# --------------------------------------------------------------------------- #
# L15: the steps a sweep's receiver refuses are one Error History entry
# --------------------------------------------------------------------------- #
class TestRefusedStepsAreToldOncePerSweep(unittest.TestCase):
    """The engine: one line for a pass's refused steps, and a continuous
    scan that keeps meeting the same ones doesn't say it every pass."""

    def setUp(self):
        require_pyqt6(self)

    def test_one_line_a_sweep_and_not_again_for_the_same_steps(self):
        from sdr_module.gui.scan_engine import ScanEngine, ScanSettings
        from tests.test_gui_scanner import _demo, _DemoHost, _run

        dev = _demo()
        host = _DemoHost(dev)
        host.refuse_above = 100e6
        engine = ScanEngine(host)
        settings = ScanSettings("once", 98e6, 104e6, 200e3)
        with self.assertLogs("sdr_module.gui.scan_engine", "WARNING") as logs:
            self.assertEqual(engine.start(settings, dev.sample_rate, 0.0), "")
            _run(engine, dev)
        self.assertFalse(engine.active)
        refused = engine.steps_refused
        self.assertGreater(refused, 1)
        self.assertEqual(len(logs.records), 1, logs.output)
        self.assertIn(f"{refused} steps", logs.records[0].getMessage())
        # Continuously: the same steps every pass, said once.
        settings = ScanSettings("continuous", 98e6, 104e6, 200e3)
        with self.assertLogs("sdr_module.gui.scan_engine", "WARNING") as logs:
            self.assertEqual(engine.start(settings, dev.sample_rate, 0.0), "")
            _run(engine, dev, until=lambda: engine.pass_number > 3)
            self.assertGreater(engine.pass_number, 3)
            engine.stop()
        self.assertEqual(len(logs.records), 1, logs.output)


class TestRefusedStepsInTheWindow(_ScanWindowCase):
    demo = True

    def setUp(self):
        super().setUp()
        from sdr_module.gui.error_log_dialog import install_history_handler

        self.history = install_history_handler()
        self.history.clear_history()
        self.addCleanup(self.history.clear_history)
        win = self.win
        win.set_frequency(95.0e6)  # (where the scan goes back to tunes)
        dev = win._device
        tune = dev.set_frequency
        # The receiver refuses the top of the band, as a tuner that can't
        # lock there does.
        dev.set_frequency = lambda freq: tune(freq) if freq < 100e6 else False

    def entries(self):
        return [text for _t, _level, _name, text in self.history.entries()]

    def sweep(self, mode, seconds, until=None):
        panel = self.win._scanner_panel
        panel._preset.setCurrentIndex(1)  # FM broadcast, 88-108 MHz
        panel.set_scan_mode(mode)
        self.assertTrue(panel.start_scan())
        self.frames(seconds, until=until or (lambda: not panel.engine.active))
        return panel

    def test_a_sweep_is_one_entry_not_one_per_refused_step(self):
        panel = self.sweep("once", 10.0)
        self.assertFalse(panel.engine.active)
        refused = panel.engine.steps_refused
        self.assertGreater(refused, 2)
        self.assertIn(
            f"{refused} steps the receiver couldn't tune were skipped",
            panel._status.text(),
        )
        entries = self.entries()
        self.assertEqual(len(entries), 1, entries)
        self.assertIn(f"{refused} steps", entries[0])
        first = min(p.lo_hz for p in panel.engine.pages if p.lo_hz >= 100e6)
        self.assertIn(f"skipped: {first / 1e6:.3f}, ", entries[0])
        # Back where it began (95 MHz tunes): nothing more to say.
        self.assertAlmostEqual(self.win._current_frequency(), 95.0e6)

    def test_a_continuous_sweep_says_it_once(self):
        panel = self.win._scanner_panel
        self.sweep("continuous", 30.0, until=lambda: panel.engine.pass_number > 2)
        self.assertGreater(panel.engine.pass_number, 2)
        panel.stop_scan()
        self.assertEqual(len(self.entries()), 1, self.entries())

    def test_a_steps_driver_error_is_in_the_entry_not_an_entry_itself(self):
        dev = self.win._device
        tune = dev.set_frequency

        def refuse(freq):
            if freq >= 100e6:
                logging.getLogger("sdr_module.devices.rtlsdr").error(
                    "Failed to set frequency: PLL not locked"
                )
                return False
            return tune(freq)

        dev.set_frequency = refuse
        self.sweep("once", 10.0)
        entries = self.entries()
        self.assertEqual(len(entries), 1, entries)
        self.assertIn("PLL not locked", entries[0])

    def test_a_refused_tune_outside_a_scan_is_still_logged(self):
        self.assertFalse(self.win._tune_to(104.0e6, recenter=True))
        entries = self.entries()
        self.assertEqual(len(entries), 1, entries)
        self.assertIn("104.000 MHz", entries[0])


if __name__ == "__main__":
    unittest.main()

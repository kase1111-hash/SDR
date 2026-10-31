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

if HAS_PYQT6:
    from PyQt6.QtCore import QSettings
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

    def test_get_bookmarks_keeps_only_usable_bookmarks_and_says_so(self):
        settings = self._settings_file(json.dumps(_DAMAGED))
        with self.assertLogs("sdr_module.gui.settings_store", "WARNING") as logs:
            bookmarks = settings.get_bookmarks()
        self.assertEqual([b["label"] for b in bookmarks], ["ok", 5])
        self.assertEqual(len(logs.records), 1)  # one line, not one per entry
        self.assertIn("8 damaged bookmarks", logs.output[0])

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
            with self.assertLogs("sdr_module.gui.bookmarks_panel", "WARNING"):
                panel = bookmarks_panel.BookmarksPanel()
        finally:
            bookmarks_panel.GuiSettings = saved
        try:
            self.assertEqual(panel.bookmark_count(), 2)
            self.assertEqual(panel.groups(), [])
            self.assertEqual(panel._list.topLevelItemCount(), 2)
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
            with self.assertLogs("sdr_module.gui.bookmarks_panel", "WARNING"):
                super().setUp()
        finally:
            test_gui_chrome._FakeSettings = original

    def test_starts_with_the_good_bookmark(self):
        panel = self.win._bookmarks_panel
        self.assertEqual([b["label"] for b in panel.bookmarks()], ["ok"])
        self.assertEqual(self.win._scanner_host.scan_bookmark_counts(), (1, 0))


if __name__ == "__main__":
    unittest.main()

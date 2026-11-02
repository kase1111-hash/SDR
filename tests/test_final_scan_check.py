#!/usr/bin/env python3
"""
What the check of the scan area's final fixes found: damaged saved
bookmarks that still stopped the app starting or broke ``sdr-scan
channels``, a bookmark scan that began in the middle of the list, PMR446
results listed off their channel, and a tooltip that said too much.

Each test failed before its fix.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_final_scan_check.py``
"""

import contextlib
import io
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
from tests.test_gui_scanner import _ScanWindowCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication

#: A whole number of 400 digits: json reads it, float() can't.
_HUGE = 10**400

# Saved bookmarks damaged by hand: numbers no float holds, text where a
# number belongs, a frequency below zero.
_BIG_TONE = {
    "label": "big tone",
    "freq_hz": 145.29e6,
    "demod": "FM",
    "fm_deviation": "5 kHz",
    "tone_mode": "TSQL",
    "ctone": _HUGE,
}
_BIG_SHIFT = {"label": "big shift", "freq_hz": 146.94e6, "duplex": "-"}
_BIG_SHIFT["offset_hz"] = _HUGE
_BIG_RTONE = {"label": "big rtone", "freq_hz": 146.52e6, "tone_mode": "Tone"}
_BIG_RTONE["rtone"] = _HUGE


def _settle(n=3):
    for _ in range(n):
        QApplication.processEvents()


# --------------------------------------------------------------------------- #
# A saved tone or shift too big for a float stopped the app starting
# --------------------------------------------------------------------------- #
class TestTheWindowStartsWithHugeNumbersInABookmark(_WindowTestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import bookmarks_panel, main_window
        from tests import test_gui_chrome

        # (If the window can't be built, setUp fails and tearDown never
        # runs: put the real settings back anyway.)
        for module in (bookmarks_panel, main_window):
            self.addCleanup(setattr, module, "GuiSettings", module.GuiSettings)
        original = test_gui_chrome._FakeSettings

        class _Damaged(original):
            def __init__(self):
                super().__init__()
                self.bookmarks = [
                    dict(_BIG_TONE),
                    dict(_BIG_SHIFT),
                    dict(_BIG_RTONE),
                    {"label": "ok", "freq_hz": 100.1e6},
                ]

        test_gui_chrome._FakeSettings = _Damaged
        try:
            super().setUp()
        finally:
            test_gui_chrome._FakeSettings = original

    def test_it_starts_and_keeps_every_bookmark(self):
        from sdr_module.gui.bookmarks_panel import listening

        panel = self.win._bookmarks_panel
        self.assertEqual(
            [b["label"] for b in panel.bookmarks()],
            ["big tone", "big shift", "big rtone", "ok"],
        )
        self.assertEqual(self.win._scanner_host.scan_bookmark_counts(), (4, 0))
        # A tone that isn't a tone squelches nothing.
        self.assertEqual(listening(panel.bookmark(0))[3], 0.0)

    def test_edit_opens_on_a_shift_too_big_for_a_float(self):
        from sdr_module.gui.bookmark_editor import BookmarkEditDialog

        dialog = BookmarkEditDialog(dict(_BIG_SHIFT), parent=self.win)
        try:
            self.assertEqual(dialog._offset.value(), 0.0)
            self.assertEqual(dialog.bookmark()["freq_hz"], 146.94e6)
        finally:
            dialog.deleteLater()
            _settle()


class TestNumbersTooBigForAFloat(unittest.TestCase):
    def test_the_list_texts_and_the_tone_squelch(self):
        require_pyqt6(self)
        from sdr_module.gui import bookmarks_panel
        from sdr_module.gui.bookmark_editor import bookmark_tone_hz

        self.assertEqual(bookmarks_panel._num(_HUGE, 7.0), 7.0)
        self.assertEqual(bookmark_tone_hz(_BIG_TONE), 0.0)
        for bookmark in (_BIG_TONE, _BIG_SHIFT, _BIG_RTONE):
            self.assertIn(bookmark["label"], bookmarks_panel._tooltip(bookmark))
            bookmarks_panel._details(bookmark)  # (raised OverflowError)


# --------------------------------------------------------------------------- #
# sdr-scan channels list/export and the Bookmarks export with damaged ones
# --------------------------------------------------------------------------- #
class TestDamagedBookmarksStillList(unittest.TestCase):
    DAMAGED = [
        {"label": "below zero", "freq_hz": -5.0},
        {"label": "text shift", "freq_hz": 146.94e6, "duplex": "-", "offset_hz": "x"},
        dict(_BIG_SHIFT),
        {
            "label": "text fields",
            "freq_hz": 146.52e6,
            "tone_mode": "TSQL",
            "rtone": "x",
            "ctone": [1],
            "dtcs": "abc",
            "rx_dtcs": _HUGE,
            "tuning_step_khz": "?",
            "location": "a",
        },
        {"label": "ok", "freq_hz": 100.1e6},
    ]

    def test_a_frequency_below_zero_is_listed_at_0_hz(self):
        require_pyqt6(self)
        from sdr_module.gui.settings_store import usable_bookmarks

        with self.assertLogs("sdr_module.gui.settings_store", "WARNING") as logs:
            kept = usable_bookmarks([{"label": "below zero", "freq_hz": -5.0}])
        self.assertEqual(kept, [{"label": "below zero", "freq_hz": 0.0}])
        self.assertIn("no usable frequency", logs.output[0])

    def test_channels_are_made_from_them(self):
        from sdr_module.core.chirp_csv import Channel

        channels = [Channel.from_bookmark(b) for b in self.DAMAGED[1:]]
        self.assertEqual([c.offset_hz for c in channels[:2]], [0.0, 0.0])
        odd = channels[2]
        self.assertEqual((odd.rtone, odd.ctone), (88.5, 88.5))
        self.assertEqual((odd.dtcs, odd.rx_dtcs), (23, 23))
        self.assertEqual(odd.tuning_step_khz, 5.0)
        self.assertIsNone(odd.location)
        # What was saved well is kept.
        good = Channel.from_bookmark(
            {
                "label": "Rpt",
                "freq_hz": 146.94e6,
                "duplex": "-",
                "offset_hz": 600e3,
                "tone_mode": "TSQL",
                "ctone": 100.0,
                "dtcs": 23,
                "location": 7,
            }
        )
        self.assertEqual(
            (good.offset_hz, good.ctone, good.dtcs, good.location),
            (600e3, 100.0, 23, 7),
        )

    def test_sdr_scan_channels_list_and_export(self):
        require_pyqt6(self)
        import json

        from PyQt6.QtCore import QSettings

        from sdr_module import cli
        from sdr_module.core.chirp_csv import read_chirp_csv
        from sdr_module.gui import settings_store

        # The saved bookmarks as the app keeps them: JSON text in its INI.
        folder = tempfile.mkdtemp(prefix="sdr-bookmarks-")
        ini = os.path.join(folder, "settings.ini")
        raw = QSettings(ini, QSettings.Format.IniFormat)
        raw.setValue("bookmarks", json.dumps(self.DAMAGED))
        raw.sync()
        store = settings_store.GuiSettings()
        store._s = QSettings(ini, QSettings.Format.IniFormat)
        path = os.path.join(folder, "out.csv")
        out = io.StringIO()
        with (
            mock.patch.object(settings_store, "GuiSettings", lambda: store),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(io.StringIO()),
            self.assertLogs("sdr_module.gui.settings_store", "WARNING"),
        ):
            self.assertEqual(cli.main(["channels", "list"]), 0)
            self.assertEqual(cli.main(["channels", "export", path]), 0)
        self.assertIn("5 channel(s)", out.getvalue())
        names = [c.name for c in read_chirp_csv(path, strict=False).channels]
        self.assertEqual(names, ["text shift", "big shift", "text fields", "ok"])


# --------------------------------------------------------------------------- #
# A bookmark scan begins where the list begins
# --------------------------------------------------------------------------- #
class TestBookmarkScanBeginsAtTheTopOfTheList(_ScanWindowCase):
    demo = True

    def setUp(self):
        super().setUp()
        bookmarks = self.win._bookmarks_panel
        # Stored A, B (Skip), C, D; D is in a group, listed first.
        for label, freq, group, skip in (
            ("A", 100.1e6, "", ""),
            ("B", 125.0e6, "", "S"),
            ("C", 145.29e6, "", ""),
            ("D", 162.4e6, "Wx", ""),
        ):
            index = bookmarks.add_bookmark(label, freq, "FM", "5 kHz")
            if group:
                bookmarks._bookmarks[index]["group"] = group
            if skip:
                bookmarks._bookmarks[index]["skip"] = skip
        bookmarks._save()
        listed = [bookmarks.bookmark(i)["label"] for i in bookmarks.list_order()]
        self.assertEqual(listed, ["D", "A", "B", "C"])

    def test_start_scan_begins_on_the_first_bookmark_listed(self):
        panel = self.win._scanner_panel
        panel.set_scan_mode("bookmarks")
        self.assertTrue(panel.start_scan())
        self.assertEqual(panel.engine.current.label, "D")
        panel.stop_scan()

    def test_from_here_on_one_marked_skip_begins_on_the_next_listed(self):
        panel = self.win._scanner_panel
        self.assertTrue(panel.scan_bookmarks_from(1))  # B, marked Skip
        self.assertEqual(panel.engine.current.label, "C")
        panel.stop_scan()
        # On one that isn't: that one, as before.
        self.assertTrue(panel.scan_bookmarks_from(0))
        self.assertEqual(panel.engine.current.label, "A")
        panel.stop_scan()


# --------------------------------------------------------------------------- #
# PMR446 (6.25 kHz off the 5 kHz raster) keeps its own frequency
# --------------------------------------------------------------------------- #
class TestPmr446ResultsStayOnTheirChannel(unittest.TestCase):
    def test_a_reading_on_a_pmr446_channel_isnt_moved_to_the_5_khz_raster(self):
        from sdr_module.dsp.sweep import report_frequency

        # Channel 1 (446.00625) and 2 (446.01875) were listed 1.25 kHz off,
        # at 446.005 and 446.020; to the kHz they are 446.006 and 446.019.
        self.assertEqual(report_frequency(446.00625e6, 8e3, 12.5e3), 446.006e6)
        self.assertEqual(report_frequency(446.01875e6, 8e3, 12.5e3), 446.019e6)
        self.assertEqual(report_frequency(446.0066e6, 8e3, 12.5e3), 446.007e6)

    def test_5_khz_channels_still_snap(self):
        from sdr_module.dsp.sweep import report_frequency

        self.assertEqual(report_frequency(145.0488e6, 8e3, 12.5e3), 145.05e6)
        self.assertEqual(report_frequency(146.9389e6, 10e3, 12.5e3), 146.94e6)
        self.assertEqual(report_frequency(146.5342e6, 10e3, 12.5e3), 146.535e6)
        self.assertEqual(report_frequency(446.0256e6, 10e3, 12.5e3), 446.025e6)


# --------------------------------------------------------------------------- #
# A Cross channel that sends a tone but receives without one
# --------------------------------------------------------------------------- #
class TestCrossToneTooltip(unittest.TestCase):
    def test_a_cross_channel_without_a_receive_squelch_says_so(self):
        require_pyqt6(self)
        from sdr_module.gui import bookmarks_panel

        for cross in ("Tone->", "DTCS->"):
            tip = bookmarks_panel._tooltip(
                {
                    "label": "x",
                    "freq_hz": 146.94e6,
                    "tone_mode": "Cross",
                    "cross_mode": cross,
                }
            )
            self.assertNotIn("not supported", tip, cross)
            self.assertIn(f"Cross ({cross}): no receive squelch", tip)
        # One receiving with a DCS code: its code (K-A3: DCS is supported
        # now); reverse DCS still isn't.
        tip = bookmarks_panel._tooltip(
            {"label": "x", "freq_hz": 1e8, "tone_mode": "Cross", "cross_mode": "->DTCS"}
        )
        self.assertIn("Cross (->DTCS): receives DCS 023 N", tip)
        tip = bookmarks_panel._tooltip(
            {"label": "x", "freq_hz": 1e8, "tone_mode": "DTCS-R"}
        )
        self.assertIn("not supported here", tip)


if __name__ == "__main__":
    unittest.main()

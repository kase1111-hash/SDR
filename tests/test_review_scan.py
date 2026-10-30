#!/usr/bin/env python3
"""
Regression tests for the code review of the scanner and bookmarks (area
"scan"): one test (or a few) per confirmed finding, named after its id.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_review_scan.py``
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, require_pyqt6  # noqa: E402
from tests.test_gui_scanner import (  # noqa: E402
    _demo,
    _DemoHost,
    _run,
    _ScanWindowCase,
)

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication


def _preset_index(panel, prefix):
    """The Band list's index of the preset whose label starts with ``prefix``."""
    for i in range(panel._preset.count()):
        if panel._preset.itemText(i).startswith(prefix):
            return i
    raise ValueError(prefix)


# --------------------------------------------------------------------------- #
# SCAN-01: a sweep started with a wide filter measured around the wrong LO
# --------------------------------------------------------------------------- #
class TestScan01SweepWithAWideFilter(_ScanWindowCase):
    def test_each_step_is_measured_around_the_lo_it_was_planned_on(self):
        """I/Q with a 2 MHz filter (what Listen on an ISM 915 result
        selects): the nearest channel doesn't fit beside the LO, so tuning
        the VFO there moved the LO by half a step, and every result was
        off by it (125.000 MHz listed as 124.988 MHz)."""
        win, panel, host = self.win, self.win._scanner_panel, self.win._scanner_host
        win._set_demod_mode("None (I/Q)", None, "2 MHz")
        self.assertEqual(win._control_panel._bw_combo.currentText(), "2 MHz")
        moved = []
        tune_lo = host.scan_tune_lo

        def spy(center, vfo=None):
            ok = tune_lo(center, vfo)
            moved.append(round(win._lo_frequency() - center))
            return ok

        host.scan_tune_lo = spy
        panel._preset.setCurrentIndex(_preset_index(panel, "Airband"))
        panel.set_scan_mode("once")
        self.assertTrue(panel.start_scan(), panel._status.text())
        self.frames(10.0, until=lambda: not panel.engine.active)
        self.assertFalse(panel.engine.active)
        self.assertEqual(set(moved), {0})
        found = {round(h.frequency_hz) for h in panel.hits()}
        self.assertIn(125_000_000, found)  # the demo's steady AM station
        self.assertFalse({124_988_000, 124_987_500} & found, sorted(found))

    def test_listen_on_a_result_found_with_a_wide_filter_hears_it(self):
        win, panel = self.win, self.win._scanner_panel
        win._set_demod_mode("None (I/Q)", None, "2 MHz")
        panel._preset.setCurrentIndex(_preset_index(panel, "Airband"))
        panel.set_scan_mode("once")
        self.assertTrue(panel.start_scan())
        self.frames(10.0, until=lambda: not panel.engine.active)
        rows = [
            r
            for r in range(panel._table.rowCount())
            if panel._table.item(r, 0).text() == "125.000 MHz"
        ]
        self.assertEqual(len(rows), 1, [h.frequency_hz for h in panel.hits()])
        panel._table.selectRow(rows[0])
        panel.listen_selected()
        self.settle()
        self.assertAlmostEqual(win._current_frequency(), 125.0e6)
        self.assertEqual(win._control_panel.get_demod_mode(), "AM")
        self.frames(1.0)
        self.assertTrue(win._squelch.is_open)


class TestScan01EngineMeasuresAroundTheReceiversLo(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def test_a_host_that_puts_the_lo_elsewhere_still_gets_true_frequencies(self):
        """The engine measures each page around the LO the host reports, not
        the one it planned: results stay right even when the receiver can't
        put its LO exactly there."""
        from sdr_module.gui.scan_engine import ScanEngine, ScanSettings

        class Shifted(_DemoHost):
            def scan_tune_lo(self, center, vfo=None):
                return super().scan_tune_lo(center + 12.5e3, vfo)

        dev = _demo(162.4e6)
        engine = ScanEngine(Shifted(dev))
        found = []
        engine.on_found = found.append
        settings = ScanSettings("once", 162.4e6, 162.55e6, 25e3)
        self.assertEqual(engine.start(settings, dev.sample_rate, 0.0), "")
        _run(engine, dev)
        self.assertFalse(engine.active)
        noaa = sorted({round(f.frequency_hz) for f in found})
        self.assertEqual(noaa, [162400000, 162475000, 162550000])


# --------------------------------------------------------------------------- #
# SCAN-03: Edit... rewrote an imported channel's CHIRP Mode
# --------------------------------------------------------------------------- #
_CHIRP_MODES = (
    "NFM",
    "DV",
    "DMR",
    "P25",
    "DN",
    "PKT",
    "RTTY",
    "FSK",
    "NAM",
    "CWR",
    "NCW",
    "WFM",
    "AM",
    "FM",
    "USB",
    "Auto",
)


def _chirp_csv(modes):
    rows = "".join(
        f"{i},CH{m},{145.0 + i * 0.025:.6f},,0.000000,,88.5,88.5,023,NN,{m},5.00,,,\n"
        for i, m in enumerate(modes, 1)
    )
    return (
        "Location,Name,Frequency,Duplex,Offset,Tone,rToneFreq,cToneFreq,DtcsCode,"
        "DtcsPolarity,Mode,TStep,Skip,Comment,URCALL\n" + rows
    )


class _BookmarksCase(unittest.TestCase):
    """A Bookmarks panel with in-memory settings."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import bookmarks_panel
        from tests.test_gui_chrome import _FakeSettings

        self.mod = bookmarks_panel
        self.store = _FakeSettings()
        self._real_settings = bookmarks_panel.GuiSettings
        bookmarks_panel.GuiSettings = lambda: self.store
        self._panels = []

    def tearDown(self):
        self.mod.GuiSettings = self._real_settings
        for panel in self._panels:
            panel.close()
            panel.deleteLater()
        QApplication.processEvents()

    def panel(self, bookmarks=()):
        self.store.bookmarks = [dict(b) for b in bookmarks]
        panel = self.mod.BookmarksPanel()
        self._panels.append(panel)
        return panel

    def edit_with(self, panel, index, change):
        """Run the panel's Edit... on ``index``, with ``change(dialog)``
        done in the dialog before Save."""
        from sdr_module.gui.bookmark_editor import BookmarkEditDialog

        def dialog(parent, bookmark, listening, groups):
            box = BookmarkEditDialog(bookmark, listening, groups, parent)
            try:
                change(box)
                return box.bookmark()
            finally:
                box.deleteLater()

        real = self.mod._edit_dialog
        self.mod._edit_dialog = dialog
        try:
            return panel.edit_bookmark(index)
        finally:
            self.mod._edit_dialog = real


class TestScan03EditKeepsTheChirpMode(_BookmarksCase):
    def test_renaming_an_imported_channel_changes_nothing_else(self):
        import io

        from sdr_module.core.chirp_csv import export_bookmarks_csv, read_chirp_csv

        channels = read_chirp_csv(io.StringIO(_chirp_csv(_CHIRP_MODES))).channels
        panel = self.panel([c.to_bookmark() for c in channels])
        for index, mode in enumerate(_CHIRP_MODES):
            before = panel.bookmark(index)

            def rename(box):
                box._name.setText(box._name.text() + " renamed")

            self.assertTrue(self.edit_with(panel, index, rename), mode)
            after = panel.bookmark(index)
            self.assertEqual(after.pop("label"), before.pop("label") + " renamed")
            self.assertEqual(after, before, mode)
        out = io.StringIO()
        export_bookmarks_csv(out, panel.bookmarks())
        out.seek(0)
        self.assertEqual([c.mode for c in read_chirp_csv(out)], list(_CHIRP_MODES))

    def test_a_new_filter_width_keeps_the_mode_and_survives_a_round_trip(self):
        import io

        from sdr_module.core.chirp_csv import export_bookmarks_csv, read_chirp_csv

        channels = read_chirp_csv(io.StringIO(_chirp_csv(("DV", "NFM")))).channels
        panel = self.panel([c.to_bookmark() for c in channels])
        for index in (0, 1):
            self.edit_with(
                panel,
                index,
                lambda box: box._width.setCurrentIndex(box._width.findData("12.5 kHz")),
            )
        dv, nfm = panel.bookmarks()
        self.assertEqual((dv["mode"], dv["bandwidth"]), ("DV", "12.5 kHz"))
        self.assertEqual((nfm["mode"], nfm["bandwidth"]), ("NFM", "12.5 kHz"))
        self.assertEqual(self.mod.listening(dv), ("FM", "5 kHz", "12.5 kHz", 0.0))
        out = io.StringIO()
        export_bookmarks_csv(out, [dv, nfm])
        out.seek(0)
        back = [c.to_bookmark() for c in read_chirp_csv(out)]
        self.assertEqual([b["mode"] for b in back], ["DV", "NFM"])
        self.assertEqual(
            [self.mod.listening(b) for b in back],
            [
                ("FM", "5 kHz", "12.5 kHz", 0.0),
                ("FM", "5 kHz", "12.5 kHz", 0.0),
            ],
        )

    def test_a_new_mode_or_deviation_still_sets_the_chirp_mode(self):
        import io

        from sdr_module.core.chirp_csv import read_chirp_csv

        channels = read_chirp_csv(io.StringIO(_chirp_csv(("DV", "FM", "NFM")))).channels
        panel = self.panel([c.to_bookmark() for c in channels])
        self.edit_with(
            panel, 0, lambda box: box._mode.setCurrentIndex(box._mode.findData("AM"))
        )
        self.edit_with(panel, 1, lambda box: box._deviation.setCurrentText("2.5 kHz"))

        # Away and back again is no change.
        def away_and_back(box):
            box._mode.setCurrentIndex(box._mode.findData("USB"))
            box._mode.setCurrentIndex(box._mode.findData("FM"))

        self.edit_with(panel, 2, away_and_back)
        am, narrow, nfm = panel.bookmarks()
        self.assertEqual((am["demod"], am["mode"]), ("AM", "AM"))
        self.assertEqual(
            (narrow["demod"], narrow["fm_deviation"], narrow["mode"]),
            ("FM", "2.5 kHz", "NFM"),
        )
        self.assertEqual(nfm["mode"], "NFM")
        self.assertNotIn("demod", nfm)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

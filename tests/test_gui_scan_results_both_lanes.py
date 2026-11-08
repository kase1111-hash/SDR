"""Scan results carrying two lanes' fields at once (post-merge follow-up).

The squelch lane gave a Scan Bookmarks result a reverse tone squelch
(TSQL-R / DTCS-R: ``ScanHit.reverse``), the scan-decode lane gave a sweep
with both receivers' results whose reading they are (``ScanHit.receiver``
and ``ScanHit.readings``). Both are saved with the results
(``scanner/results``) and both come back in one table.

What is checked here, and what changed:

- Restored after a restart, each keeps its fields, the RX column shows for
  the list (empty for a bookmark's result), and the tooltips say both the
  squelch Listen applies and whose reading the level is.
- Listen on a restored reversed result applies the reverse squelch, with
  the receiver that tunes it.
- Export Results (CSV) didn't say a result's tone squelch at all: two
  bookmarks on one frequency told apart only by their tone (a repeater's
  users, a reversed and a plain one) were two identical rows. A "Squelch"
  column now says it as the tooltip does ("reverse tone squelch 123.0 Hz",
  "DCS 023 N"), only when a result has one (other exports are as they
  were), before the Receiver and Level from columns of a sweep with both.
- Damaged or mismatched saved values are dropped, the result kept: a
  reverse with no tone or code, a receiver not among the readings, wrong
  types; a reading saved as true/false is no level (it read as 1 dBFS).
"""

import csv
import json
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

try:
    from PyQt6.QtWidgets import QApplication
except ImportError:  # pragma: no cover - the GUI tests need PyQt6
    pass

OLD_CSV_HEADER = [
    "Frequency (MHz)",
    "Level (dBFS)",
    "SNR (dB)",
    "Width (kHz)",
    "Mode",
    "FM deviation",
    "Filter",
    "Band",
    "Name",
    "First heard",
    "Last heard",
    "Times heard",
]
SEEN = 1.7e9  # (a time any clock shows: 2023-11-14)


def _saved(freq, label="", mode="FM", **fields):
    from sdr_module.gui.scanner_panel import ScanHit

    hit = ScanHit(freq, -50.0, 30.0, 12e3, mode, "5 kHz", "12.5 kHz", label)
    hit.first_seen = hit.last_seen = SEEN
    for name, value in fields.items():
        setattr(hit, name, value)
    return hit.to_dict()


#: A Scan Bookmarks result with a reverse tone squelch, one with a reverse
#: DCS code, a sweep result both receivers read, and one carrying both
#: lanes' fields (as a saved list can).
BOTH_LANES = [
    _saved(145.29e6, "Rpt reversed", tone_hz=123.0, reverse=True),
    _saved(442.1e6, "DCS reversed", dcs="D025N", reverse=True),
    _saved(1766e6, receiver="second", readings={"main": -55.0, "second": -48.0}),
    _saved(
        146.94e6,
        "Both lanes",
        tone_hz=100.0,
        reverse=True,
        receiver="second",
        readings={"second": -45.0},
    ),
]


class _PanelCase(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import scanner_panel

        self.mod = scanner_panel
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.panels = []

    def tearDown(self):
        for panel in self.panels:
            panel.close()
            panel.deleteLater()
        QApplication.processEvents()

    def restored(self, saved):
        """A Scanner tab started with ``saved`` as its saved results (as
        the app starts after a restart)."""
        from tests.test_gui_scanner import FakeWindow

        store = _FakeSettings()
        store.values["scanner/results"] = json.dumps(saved)
        panel = self.mod.ScannerPanel(settings=store)
        panel.set_host(FakeWindow())
        panel.resize(420, 600)
        panel.show()
        QApplication.processEvents()
        self.panels.append(panel)
        return panel

    @staticmethod
    def by_freq(panel):
        return {round(h.frequency_hz / 1e3): h for h in panel.hits()}

    def row(self, panel, freq):
        table = panel._table
        for row in range(table.rowCount()):
            item = table.item(row, self.mod.COL_FREQ)
            hit = panel._hits[item.data(self.mod._HIT_ROLE)]
            if abs(hit.frequency_hz - freq) < 1.0:
                return row
        raise AssertionError(f"no row for {freq}")

    def export(self, panel):
        path = os.path.join(self.tmp.name, "scan.csv")
        panel.export_csv(path)
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.reader(fh))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestRestored(_PanelCase):
    def test_both_lanes_fields_come_back(self):
        mod = self.mod
        panel = self.restored(BOTH_LANES)
        hits = self.by_freq(panel)
        self.assertEqual(len(hits), 4)
        rpt, dcs = hits[145290], hits[442100]
        both, mixed = hits[1766000], hits[146940]
        self.assertEqual((rpt.tone_hz, rpt.reverse, rpt.receiver), (123.0, True, ""))
        self.assertEqual((dcs.dcs, dcs.reverse, dcs.readings), ("D025N", True, {}))
        self.assertEqual((both.receiver, both.read_by), ("second", "Both"))
        self.assertEqual(both.readings, {"main": -55.0, "second": -48.0})
        self.assertFalse(both.reverse)
        self.assertEqual(
            (mixed.tone_hz, mixed.reverse, mixed.receiver, mixed.read_by),
            (100.0, True, "second", "Second"),
        )
        # The RX column: shown for the list, empty for a bookmark's result.
        table = panel._table
        self.assertFalse(table.isColumnHidden(mod.COL_RECEIVER))
        rx = {
            freq: table.item(self.row(panel, freq), mod.COL_RECEIVER).text()
            for freq in (145.29e6, 442.1e6, 1766e6, 146.94e6)
        }
        self.assertEqual(
            rx, {145.29e6: "", 442.1e6: "", 1766e6: "Both", 146.94e6: "2nd"}
        )
        # The tooltips say the squelch Listen applies, and whose reading.
        tip = table.item(self.row(panel, 146.94e6), mod.COL_FREQ).toolTip()
        self.assertIn("reverse tone squelch 100.0 Hz", tip)
        self.assertIn("Read by the second receiver", tip)
        tip = table.item(self.row(panel, 442.1e6), mod.COL_FREQ).toolTip()
        self.assertIn("reverse DCS 025 N", tip)
        self.assertNotIn("Read by", tip)

    def test_damaged_or_mismatched_values_go_the_result_stays(self):
        saved = [
            _saved(146.0e6, reverse=True),  # (nothing to reverse)
            dict(_saved(146.1e6, tone_hz=100.0), reverse="yes"),
            dict(_saved(146.2e6, tone_hz=100.0), reverse=1),
            dict(_saved(146.3e6), receiver="main", readings={"second": -40.0}),
            dict(_saved(146.4e6), receiver=["main"], readings={"main": -40.0}),
            dict(_saved(146.5e6), receiver="main", readings=[["main", -40.0]]),
            dict(_saved(146.6e6), receiver="main", readings={"main": "loud"}),
            dict(_saved(146.7e6), receiver="second", readings={"second": True}),
            dict(_saved(146.8e6), receiver="main", readings={"main": False}),
            dict(_saved(146.9e6, tone_hz=100.0), reverse=True, receiver="third"),
            dict(_saved(147.0e6), dcs=23, reverse=True),
            dict(_saved(147.1e6), readings={"main": -40.0, "second": -41.0}),
        ]
        panel = self.restored(saved)
        hits = self.by_freq(panel)
        self.assertEqual(len(hits), len(saved))  # (every result kept)
        for freq in (146000, 146100, 146200, 147000):
            self.assertFalse(hits[freq].reverse, freq)
        for freq in (146300, 146400, 146500, 146600, 146700, 146800, 147100):
            self.assertEqual((hits[freq].receiver, hits[freq].readings), ("", {}), freq)
        # (The squelch is kept without a receiver that isn't one.)
        self.assertEqual(
            (hits[146900].tone_hz, hits[146900].reverse, hits[146900].receiver),
            (100.0, True, ""),
        )
        # None of them reads as a sweep with both: no RX column.
        self.assertTrue(panel._table.isColumnHidden(self.mod.COL_RECEIVER))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestExport(_PanelCase):
    def test_it_says_each_results_squelch(self):
        panel = self.restored(
            [
                _saved(145.29e6, "Rpt", tone_hz=100.0),
                _saved(145.30e6, "Rpt reversed", tone_hz=123.0, reverse=True),
                _saved(442.1e6, "DCS", dcs="D023N"),
                _saved(442.2e6, "DCS reversed", dcs="D025N", reverse=True),
                _saved(146.52e6, "Simplex"),
            ]
        )
        rows = self.export(panel)
        self.assertEqual(rows[0], OLD_CSV_HEADER + ["Squelch"])
        squelch = {row[8]: row[-1] for row in rows[1:]}
        self.assertEqual(
            squelch,
            {
                "Rpt": "tone squelch 100.0 Hz",
                "Rpt reversed": "reverse tone squelch 123.0 Hz",
                "DCS": "DCS 023 N",
                "DCS reversed": "reverse DCS 025 N",
                "Simplex": "",
            },
        )

    def test_both_lanes_columns_together(self):
        rows = self.export(self.restored(BOTH_LANES))
        self.assertEqual(
            rows[0], OLD_CSV_HEADER + ["Squelch", "Receiver", "Level from"]
        )
        by_freq = {row[0]: row[-3:] for row in rows[1:]}
        self.assertEqual(
            by_freq,
            {
                "145.290000": ["reverse tone squelch 123.0 Hz", "", ""],
                "146.940000": ["reverse tone squelch 100.0 Hz", "Second", "Second"],
                "442.100000": ["reverse DCS 025 N", "", ""],
                "1766.000000": ["", "Both", "Second"],
            },
        )

    def test_exports_without_a_squelch_are_as_they_were(self):
        sweep = self.export(self.restored([_saved(100.1e6), _saved(88.5e6)]))
        self.assertEqual(sweep[0], OLD_CSV_HEADER)
        both = self.export(
            self.restored([_saved(1850e6, receiver="second", readings={"second": -4})])
        )
        self.assertEqual(both[0], OLD_CSV_HEADER + ["Receiver", "Level from"])
        # (A squelch kept only where Listen applies it: FM.)
        am = self.export(self.restored([_saved(119.1e6, mode="AM", tone_hz=100.0)]))
        self.assertEqual(am[0], OLD_CSV_HEADER)


# --------------------------------------------------------------------------- #
# In the window: restored after a restart, listened to
# --------------------------------------------------------------------------- #


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestInTheWindow(_WindowTestCase):
    demo = True

    def restart(self):
        """The app started again with these results saved."""
        self.store.values["scanner/results"] = json.dumps(BOTH_LANES)
        self.win.close()
        self.win.deleteLater()
        self.settle()
        self.win = self.mw.SDRMainWindow(demo_mode=True)
        for timer in (
            self.win._display_timer,
            self.win._status_timer,
            self.win._hotplug_timer,
        ):
            timer.stop()
        self.win._confirm_discard_recording = lambda *_args: True
        self.win.resize(1280, 720)
        self.win.show()
        self.settle()
        return self.win._scanner_panel

    def listen(self, panel, freq):
        from sdr_module.gui import scanner_panel as mod

        table = panel._table
        for row in range(table.rowCount()):
            hit = panel._hits[table.item(row, mod.COL_FREQ).data(mod._HIT_ROLE)]
            if abs(hit.frequency_hz - freq) < 1.0:
                table.selectRow(row)
                break
        panel.listen_selected()
        self.settle()

    def test_restored_and_listened_with_the_reverse_squelch(self):
        panel = self.restart()
        win = self.win
        self.assertEqual(len(panel.hits()), 4)
        controls = win._control_panel
        self.listen(panel, 145.29e6)
        self.assertAlmostEqual(win._current_frequency(), 145.29e6, delta=1.0)
        self.assertEqual(controls.tone_squelch_hz(), 123.0)
        self.assertTrue(controls.tone_reverse())
        self.assertIn("reverse tone squelch 123.0 Hz", win._message_label.text())
        # The one carrying both lanes' fields: the second receiver read it,
        # this window's receiver tunes it, with its reverse squelch.
        self.listen(panel, 146.94e6)
        self.assertAlmostEqual(win._current_frequency(), 146.94e6, delta=1.0)
        self.assertEqual(controls.tone_squelch_hz(), 100.0)
        self.assertTrue(controls.tone_reverse())
        self.listen(panel, 442.1e6)
        self.assertEqual(controls.dcs_squelch(), (25, False))
        self.assertTrue(controls.tone_reverse())
        self.assertIn("reverse DCS 025 N", win._message_label.text())


if __name__ == "__main__":
    unittest.main()

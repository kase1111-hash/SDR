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
    FRAME_S,
    _demo,
    _DemoHost,
    _run,
    _ScanWindowCase,
)

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication  # noqa: F401


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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

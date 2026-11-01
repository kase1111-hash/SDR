"""Checks of the scan area's final fixes next to the other areas' (found on
the merged build). tests/test_final_integration.py holds the integrator's;
the scan area's own tests are in tests/test_final_scan.py.

F-window's rule (leftover 9): Tools > Error History lists problems only,
not a notice of the app doing what was asked. F-scan's SCAN-03: a scan that
ends while the Scanner tab is out of sight says why in the status bar. The
status bar logs what it shows in the warning tone, so a scan stopped from
another tab by what the user did (Space, Open Recording, a new sample rate)
was listed in Error History as a problem, and a scan that broke was listed
once more than with the tab in view.
"""

from __future__ import annotations

import logging
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_review_mainstate import _MainStateCase  # noqa: E402


class _Records(logging.Handler):
    """The WARNING records of the app's loggers (what reaches Tools > Error
    History)."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestScanNoticesStayOutOfTheErrorHistory(_MainStateCase):
    def setUp(self):
        super().setUp()
        self.scanner = self.win._scanner_panel
        self.records = _Records()
        logger = logging.getLogger("sdr_module")
        logger.addHandler(self.records)
        self.addCleanup(logger.removeHandler, self.records)

    def scan_out_of_sight(self):
        scanner = self.scanner
        scanner._preset.setCurrentIndex(1)  # FM Broadcast
        scanner.set_scan_mode("continuous")
        self.assertTrue(scanner.start_scan())
        self.win._show_panel("bookmarks")  # (the Scanner tab out of sight)
        self.frames(0.3)
        self.assertTrue(scanner.engine.active)
        self.assertFalse(scanner.isVisible())
        del self.records.records[:]

    def assert_notice(self, text):
        """The status bar says ``text`` (amber), and nothing is listed."""
        self.assertFalse(self.scanner.engine.active)
        self.assertEqual(self.status(), (text, "warning"))
        self.assertEqual(self.records.records, [])

    def test_space_stopping_the_receiver_is_a_notice(self):
        self.scan_out_of_sight()
        self.win._toggle_acquisition()  # (Space)
        self.scanner.refresh()  # (the tab's 500 ms timer)
        self.assert_notice("Scan stopped: the receiver was stopped.")

    def test_opening_a_recording_is_a_notice(self):
        self.scan_out_of_sight()
        self.messages.clear()
        self.assertTrue(self.win._open_recording(self.recording(100.1e6, 2.4e6)))
        self.assertFalse(self.scanner.engine.active)
        self.assertIn(
            ("Scan stopped: you opened a recording.", "warning"), self.messages
        )
        self.assertEqual([r for r in self.records.records if "Scan stopped" in r], [])
        self.win._playback.close(restore=True)

    def test_a_new_sample_rate_is_a_notice(self):
        self.scan_out_of_sight()
        self.assertTrue(self.win.set_sample_rate(2.048e6))
        self.frames(0.1)
        self.assertFalse(self.scanner.engine.active)
        self.assertIn(
            ("Scan stopped: the sample rate changed. Start it again.", "warning"),
            self.messages,
        )
        self.assertEqual([r for r in self.records.records if "Scan stopped" in r], [])

    def broken_scan_records(self):
        """What a scan step that breaks lists in Error History."""
        engine = self.scanner.engine

        def broken():
            raise RuntimeError("the receiver broke")

        engine._safely(broken)
        self.assertFalse(engine.active)
        return [r for r in self.records.records if "the receiver broke" in r]

    def test_a_scan_that_breaks_is_listed_as_with_the_tab_in_view(self):
        self.scan_out_of_sight()
        out_of_sight = self.broken_scan_records()
        self.assertIn("Details are in Tools > Error History", self.status()[0])
        # The same scan with the Scanner tab in view.
        self.win._show_panel("scanner")
        self.assertTrue(self.scanner.start_scan())
        self.frames(0.3)
        del self.records.records[:]
        in_view = self.broken_scan_records()
        self.assertTrue(in_view)
        self.assertEqual(len(out_of_sight), len(in_view), out_of_sight)

    def test_a_receiver_that_returns_no_samples_is_still_listed(self):
        """A problem (not a notice): listed once, whichever tab is shown."""
        from sdr_module.gui.scanner_panel import NO_BLOCKS_S

        for shown in ("bookmarks", "scanner"):
            with self.subTest(tab=shown):
                self.scan_out_of_sight()
                self.win._show_panel(shown)
                self.scanner.engine.last_block_at -= NO_BLOCKS_S + 1
                self.scanner.refresh()
                self.assertFalse(self.scanner.engine.active)
                listed = [r for r in self.records.records if "no samples" in r]
                self.assertEqual(len(listed), 1, self.records.records)


if __name__ == "__main__":
    unittest.main()

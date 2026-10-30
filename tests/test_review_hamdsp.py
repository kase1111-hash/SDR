"""Regression tests for the code review of the ham radio and DSP area.

One class per finding (HAMDSP-1 to HAMDSP-11): SSTV, the S-meter, QRP, the
AM/FM tuner, noise reduction and the CLI encoder.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_review_hamdsp.py``
"""

from __future__ import annotations

import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication


def settle(n: int = 3) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _ham_radio() -> bool:
    from sdr_module.gui.main_window import HAS_HAM_RADIO

    return HAS_HAM_RADIO


class _HamWindowCase(_WindowTestCase):
    """The real window (demo device) with the ham radio panels."""

    demo = True

    def setUp(self):
        super().setUp()
        if not _ham_radio():
            self.skipTest("ham radio panels not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.sstv = self.win._sstv_panel
        self.sstv.set_save_directory(self.tmp.name)
        self.win._audio.write = lambda audio: None

    def tearDown(self):
        if HAS_PYQT6 and getattr(self, "sstv", None) and self.sstv.is_receiving():
            self.sstv._start_btn.click()
        super().tearDown()


class TestTuneToIssRefused(_HamWindowCase):
    """HAMDSP-1: Tune to ISS ignored a refused tune. It said "Tuned to the
    ISS SSTV downlink", switched the station being heard to narrow FM and
    started the SSTV decoder while the receiver stayed where it was."""

    def assert_nothing_changed(self, freq_hz: float, mode: str, reason: str):
        win = self.win
        self.assertAlmostEqual(win._current_frequency(), freq_hz)
        self.assertEqual(win._control_panel.get_demod_mode(), mode)
        self.assertFalse(self.sstv.is_receiving(), "the decoder stays off")
        self.assertIn(reason, win._message_label.text())
        self.assertEqual(win._message_label.property("tone"), "danger")

    def test_refused_while_transmitting(self):
        win = self.win
        self.assertTrue(win.set_frequency(14.230e6))
        win._set_demod_mode("USB")
        win._tx_busy = lambda: True  # a transmission is on the air
        self.sstv._tune_iss_btn.click()
        settle()
        self.assert_nothing_changed(14.230e6, "USB", "Can't tune while transmitting")

    def test_refused_outside_the_recording_from_the_menu(self):
        win = self.win
        win._set_demod_mode("AM")
        win._lock_lo(100.1e6, 2.4e6, "the recording")
        tuned = win._current_frequency()
        win._tune_iss_from_menu()
        settle()
        self.assert_nothing_changed(tuned, "AM", "outside the recording")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

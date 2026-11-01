"""Checks that need two areas' final fixes together (found while merging
them): each area's own tests live in tests/test_final_<area>.py.

A HackRFDevice runs on the threaded fake ``python_hackrf`` backend
(tests/hackrf_fakes.py) where something is transmitted.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_station_tx import _TxCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestOnAirNoticesStayOutOfTheErrorHistory(_TxCase):
    """The on-air notices (Record I/Q armed for after the transmission,
    Space refused while transmitting) say what happens next: they are not
    problems, so like the other notices they stay out of Tools > Error
    History (and the console)."""

    def _send_slow_id(self):
        self.win._spectrum.update_spectrum(np.full(1024, -90.0))
        self.win._waterfall.add_line(np.full(1024, -90.0))
        self.panel.set_callsign("W1AW")
        self.panel._wpm_spin.setValue(5)  # (a long ID: about 10 s)
        self.station.send_station_id()
        self.assertTrue(self.station.busy)
        QApplication.processEvents()

    def quietly(self, action, shown):
        with self.assertNoLogs("sdr_module.gui.main_window", level="WARNING"):
            action()
        self.assertIn(shown, self.win._message_label.text())

    def test_recording_armed_and_space_refused_on_the_air(self):
        self.hackrf(pace=True)
        self.win._start_acquisition()  # (receiving resumes after the ID)
        self._send_slow_id()
        self.quietly(self.win._start_recording, "Recording armed:")
        self.win._stop_recording()
        self.quietly(self.win._start_acquisition, "Can't receive while transmitting")
        self.station.stop()
        self.wait_done(5)
        QApplication.processEvents()
        self.win._stop_acquisition()

    def test_when_the_receiver_was_off(self):
        self.hackrf(pace=True)
        self._send_slow_id()
        self.quietly(self.win._start_recording, "Recording armed:")
        self.win._stop_recording()
        self.quietly(self.win._start_acquisition, "Can't receive while transmitting")
        self.station.stop()
        self.wait_done(5)
        QApplication.processEvents()

    def test_a_stopped_transmission_is_not_a_problem(self):
        self.hackrf(pace=True)
        self._send_slow_id()
        with self.assertNoLogs("sdr_module.gui.main_window", level="WARNING"):
            self.station.stop()
            self.wait_done(5)
        self.assertTrue(
            self.win._message_label.text().startswith("Stopped DE W1AW"),
            self.win._message_label.text(),
        )
        self.assertEqual(self.win._message_label.property("tone"), "warning")


if __name__ == "__main__":
    unittest.main()

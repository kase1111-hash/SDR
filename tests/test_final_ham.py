"""Final fixes in the ham area (the end-to-end journeys' problems): the
window while on the air, SSTV Start Now with Auto, the S-meter's copied
report, the SSTV hints, the Ham ID preview's audio output, the second
receiver's spectrum defaults and transmit monitor line, the 60 m wording,
the SSTV history after a restart, the radio tuner without a device and the
Ham ID tooltips.

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
    from PyQt6.QtWidgets import QApplication  # noqa: F401

ON_AIR_TEXT = "Transmitting - receiving resumes when it ends"


class TestWindowWhileOnTheAir(_TxCase):
    """HAM-E2E-1: while transmitting, the plots say receiving resumes when
    it ends and Start is off (the HackRF is half duplex), so nothing
    invites a press of Space that would only be refused."""

    def _show_a_frame(self):
        """The plots have something to dim (the fake HackRF receives
        nothing)."""
        self.win._spectrum.update_spectrum(np.full(1024, -90.0))
        self.win._waterfall.add_line(np.full(1024, -90.0))

    def _send_slow_id(self):
        self._show_a_frame()
        self.panel.set_callsign("W1AW")
        self.panel._wpm_spin.setValue(5)  # (a long ID: about 10 s)
        self.station.send_station_id()
        self.assertTrue(self.station.busy)
        QApplication.processEvents()

    def _assert_on_air_state(self):
        win = self.win
        self.assertEqual(win._spectrum.stale_text(), ON_AIR_TEXT)
        self.assertEqual(win._waterfall.stale_text(), ON_AIR_TEXT)
        button = win._start_button
        self.assertFalse(button.isEnabled())
        self.assertNotEqual(button.property("role"), "primary")
        self.assertIn("transmission ends", button.toolTip())
        self.assertFalse(win._start_action.isEnabled())
        self.assertEqual(win._state_badge.text(), "ON AIR")

    def test_receiver_running_before(self):
        self.hackrf(pace=True)
        self.win._start_acquisition()
        self.assertTrue(self.win._is_running)
        self._send_slow_id()
        self._assert_on_air_state()
        # Anything that refreshes the state keeps it (e.g. a theme or
        # device-controls sync while on the air).
        self.win._refresh_state_ui()
        self._assert_on_air_state()
        self.station.stop()
        self.wait_done(5)
        QApplication.processEvents()
        # Receiving resumed: live plots, Stop offered.
        self.assertTrue(self.win._is_running)
        self.assertIsNone(self.win._spectrum.stale_text())
        self.assertIsNone(self.win._waterfall.stale_text())
        self.assertTrue(self.win._start_button.isEnabled())
        self.assertTrue(self.win._start_action.isEnabled())
        self.win._stop_acquisition()

    def test_receiver_stopped_before(self):
        self.hackrf(pace=True)
        self.assertFalse(self.win._is_running)
        self._send_slow_id()
        self._assert_on_air_state()
        self.station.stop()
        self.wait_done(5)
        QApplication.processEvents()
        self.assertFalse(self.win._is_running)
        self.assertEqual(self.win._spectrum.stale_text(), "Stopped - press Space")
        self.assertEqual(self.win._waterfall.stale_text(), "Stopped - press Space")
        button = self.win._start_button
        self.assertTrue(button.isEnabled())
        self.assertEqual(button.property("role"), "primary")
        self.assertEqual(button.toolTip(), "Start receiving (Space)")
        self.assertTrue(self.win._start_action.isEnabled())

    def test_recording_armed_on_the_air_says_when_it_captures(self):
        self.hackrf(pace=True)
        self._send_slow_id()
        self.assertTrue(self.win._start_recording())
        text = self.win._message_label.text()
        self.assertIn("when receiving resumes after the transmission", text)
        self.assertNotIn("(Space)", text)
        self.win._stop_recording()
        self.station.stop()
        self.wait_done(5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

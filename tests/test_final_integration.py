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

from tests import test_final_window as _final_window  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_station_tx import _TxCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication


class _DamagedWindow(_final_window._IniSettings):
    """A window built on a real settings file holding damaged values (the
    helper of test_final_window's damaged-settings tests)."""

    window = _final_window.TestDamagedSettingsInTheWindow.window


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


class TestSavedChoicesOfAnotherShape(_DamagedWindow):
    """A saved value that is valid JSON but not the object the window
    saves (a list, a number, a string) is damaged too: like unreadable
    JSON, it gives the defaults and is logged (Tools > Error History),
    as the second receiver's plot choices already are."""

    def test_plot_choices_of_another_shape_are_logged(self):
        with self.assertLogs("sdr_module.gui.main_window", "WARNING") as logs:
            self.window(spectrum_display="[1, 2]", waterfall_display="42")
        text = "\n".join(logs.output)
        self.assertIn("unreadable spectrum settings", text)
        self.assertIn("unreadable waterfall settings", text)

    def test_ham_choices_of_another_shape_are_logged(self):
        from sdr_module.gui import main_window

        if not main_window.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")
        with self.assertLogs("sdr_module.gui.main_window", "WARNING") as logs:
            self.window(ham_id='"W1AW"', ham__qrp="[]")
        text = "\n".join(logs.output)
        self.assertIn("unreadable saved Ham ID settings", text)
        self.assertIn("unreadable saved QRP settings", text)

    def test_radio_presets_of_another_shape_are_logged(self):
        """(Read when Tools > AM/FM Radio Tuner... first opens.)"""
        from sdr_module.gui import main_window

        if not main_window.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")
        win = self.window(ham__radio_presets="[1, 2]")
        with self.assertLogs("sdr_module.gui.main_window", "WARNING") as logs:
            win._show_radio_tuner()
        self.assertIn("unreadable saved radio presets", "\n".join(logs.output))
        win._radio_tuner.close()


if __name__ == "__main__":
    unittest.main()

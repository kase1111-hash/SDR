"""Checks that need the ham area's final fixes together with another area's
(found while merging them; the ham area's own tests are in
tests/test_final_ham.py).

Tools > Error History lists problems only (F-window, leftover 9): a notice
in amber - what to do first, or what the app did as asked - is not logged.
The ham area's notices follow that rule too, among them the radio tuner's
new "No receiver" note (HAM-E2E-13), whose SSTV twin ("Tuned to 145.800
MHz FM. No receiver: ...") F-window had already made a notice.

A HackRFDevice runs on the threaded fake ``python_hackrf`` backend
(tests/hackrf_fakes.py) where something is transmitted.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_station_tx import _TxCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestHamNoticesStayOutOfTheErrorHistory(_TxCase):
    """Each shows in amber in the status bar, and nothing is logged (no
    WARNING from any logger: Tools > Error History lists them all)."""

    def quietly(self, action, shown):
        with self.assertNoLogs(level="WARNING"):
            action()
        self.assertIn(shown, self.win._message_label.text())
        self.assertEqual(self.win._message_label.property("tone"), "warning")

    def test_a_station_tuned_with_no_receiver(self):
        win = self.win
        self.assertIsNone(win._device)
        win._show_radio_tuner()
        tuner = win._radio_tuner
        self.addCleanup(tuner.close)
        self.quietly(tuner._preset_buttons[0].click, "No receiver")
        self.assertTrue(win._message_label.text().startswith("Tuned to "))

    def test_the_sstv_mode_hint(self):
        win = self.win
        win._sstv_panel.start_listening()
        self.addCleanup(win._sstv_panel._on_stop_clicked)  # (Stop Decoder)
        self.assertTrue(win._sstv_listening())
        self.quietly(lambda: win._set_demod_mode("AM"), "SSTV decoding needs FM audio")

    def test_send_id_without_a_callsign(self):
        self.hackrf()
        self.quietly(self.station.send_station_id, "enter your callsign")

    def test_already_transmitting(self):
        self.hackrf(pace=True)
        self.panel.set_callsign("W1AW")
        self.panel._wpm_spin.setValue(5)  # (a long ID: about 10 s)
        self.assertTrue(self.station.send_station_id())
        self.assertTrue(self.station.busy)
        QApplication.processEvents()
        job = self.station.make_id_job(146.52e6)
        self.quietly(lambda: self.station.transmit(job), "Already transmitting")
        self.station.stop()
        self.wait_done(5)

    def test_transmit_text_notices(self):
        from sdr_module.protocols.text_modes import MAX_TEXT_CHARS

        self.win._show_transmit_text()
        dialog = self.win._transmit_text
        self.addCleanup(dialog.close)
        self.quietly(
            lambda: dialog.insert_macro("CQ CQ CQ DE {call}"), "Enter your callsign"
        )
        self.panel.set_callsign("W1AW")
        dialog.set_text("E" * MAX_TEXT_CHARS)
        self.quietly(lambda: dialog.insert_macro("DE {call}"), "The macro doesn't fit")
        self.quietly(
            lambda: dialog.set_text("E" * (MAX_TEXT_CHARS + 5)),
            "what didn't fit was left out",
        )


if __name__ == "__main__":
    unittest.main()

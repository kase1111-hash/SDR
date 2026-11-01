"""Final review of the late fixes taken together (the polish and finish
rounds): messages they made that could say something no longer true.

Each test names the problem it covers; they run the real window offscreen
on the demo device.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_recording_files import _FilesCase  # noqa: E402


class TestTheDecoderOffNoteHoldsOnlyWhileItIsOff(_FilesCase):
    """P-07 leads the next message with a preset decoder's "off" note
    ("ACARS decoder off · Tuned to ...") while the "off" message is in the
    status bar. Ticking Decode again before that tune left the note to lead
    it all the same: "ACARS decoder off · Tuned to NOAA Weather (FM)" with
    ACARS decoding."""

    def decoding(self):
        panel = self.win._decoder_panel
        return panel._proto_combo.currentText(), panel._enabled_check.isChecked()

    def tune_away_from_acars(self):
        win = self.win
        self.assertTrue(win._apply_preset_named("ACARS"))
        win.set_frequency(125e6)  # (no band plan line: the "off" message stands)
        self.assertTrue(
            win._message_label.text().startswith("ACARS decoder off: tuned away")
        )
        self.assertEqual(self.decoding(), ("ACARS", False))

    def test_decode_ticked_again(self):
        win = self.win
        self.tune_away_from_acars()
        win._decoder_panel._enabled_check.setChecked(True)  # (you, in the tab)
        self.assertEqual(self.decoding(), ("ACARS", True))
        win._on_bookmark_tune(162.55e6, "NOAA Weather", "FM", "5 kHz")
        self.assertEqual(win._message_label.text(), "Tuned to NOAA Weather (FM)")
        self.assertEqual(self.decoding(), ("ACARS", True))

    def test_sstv_started_again(self):
        from sdr_module.gui.main_window import HAS_HAM_RADIO

        if not HAS_HAM_RADIO:
            self.skipTest("no ham radio tools")
        win = self.win
        if not win._is_running:
            win._toggle_acquisition()
        self.assertTrue(win._apply_preset_named("ISS SSTV/Voice"))
        win.set_frequency(146.52e6)  # (2 m too: no band plan line)
        self.assertFalse(win._sstv_listening())
        stopped = win._message_label.text()
        self.assertTrue(stopped.startswith("SSTV decoder stopped: tuned away"))
        win._sstv_panel._on_start_clicked()  # (you, in the SSTV tab)
        self.assertTrue(win._sstv_listening())
        self.assertEqual(win._message_label.text(), stopped)
        win._on_bookmark_tune(162.55e6, "NOAA Weather", "FM", "5 kHz")
        self.assertEqual(win._message_label.text(), "Tuned to NOAA Weather (FM)")

    def test_still_off_it_leads_the_message(self):
        """(As P-07 has it, while the decoder stays off.)"""
        win = self.win
        self.tune_away_from_acars()
        win._on_bookmark_tune(162.55e6, "NOAA Weather", "FM", "5 kHz")
        self.assertEqual(
            win._message_label.text(), "ACARS decoder off · Tuned to NOAA Weather (FM)"
        )

    def test_another_decoder_chosen(self):
        """ACARS is still off with another protocol decoding."""
        win = self.win
        self.tune_away_from_acars()
        win._decoder_panel._proto_combo.setCurrentText("POCSAG")
        self.settle(2)
        win._on_bookmark_tune(162.55e6, "NOAA Weather", "FM", "5 kHz")
        self.assertEqual(
            win._message_label.text(), "ACARS decoder off · Tuned to NOAA Weather (FM)"
        )


if __name__ == "__main__":
    unittest.main()

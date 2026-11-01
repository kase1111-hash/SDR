"""Final review of the late fixes taken together (the polish and finish
rounds): messages they made that could say something no longer true.

Each test names the problem it covers; they run the real window offscreen
on the demo device.
"""

import os
import unittest
from unittest import mock

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


class TestTheAudioTipNamesTheJobOnlyWhileItCounts(_FilesCase):
    """P-04 names this app's own job as the reason audio is paused ("Decode
    Recording keeps this computer busy, so audio is paused meanwhile."),
    for a few seconds after it too. The Audio button's tooltip took that
    reason when the pause began and kept it while the pause lasted: on a
    computer still too slow once the job was done, it went on naming the
    job long after it had ended."""

    def setUp(self):
        super().setUp()
        from tests.test_final_perf import _Clock

        self.clock = _Clock()
        patcher = mock.patch.object(self.mw, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        # (No sound card here: the button's state is that of a computer
        # with one, whose audio is paused.)
        patcher = mock.patch.object(
            type(self.win), "_audio_state", return_value="paused"
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def tip(self):
        return self.win._audio_button.toolTip()

    def test_the_tip_follows_the_reason(self):
        from sdr_module.gui import file_jobs

        win = self.win
        with file_jobs.running("Decode Recording"):
            win._update_status()  # (the status bar's timer)
        self.assertTrue(
            self.tip().startswith(
                "Decode Recording keeps this computer busy, so audio is paused"
            ),
            self.tip(),
        )
        # Done with it, the computer is still too slow: once what the job
        # skipped no longer counts, the tip says why the audio is paused now.
        for _ in range(20):
            self.clock.advance(0.2)
            win._update_status()
        self.assertNotIn("Decode Recording", self.tip())
        self.assertTrue(self.tip().startswith(win._realtime_hint()), self.tip())

    def test_the_same_reason_is_not_redrawn(self):
        """(Each tick of the status timer while nothing changed leaves the
        button alone.)"""
        win = self.win
        win._update_status()
        with mock.patch.object(win, "_sync_audio_ui") as sync:
            for _ in range(5):
                self.clock.advance(0.2)
                win._update_status()
        sync.assert_not_called()


if __name__ == "__main__":
    unittest.main()

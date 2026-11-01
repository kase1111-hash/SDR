"""The Decoder next to the other areas' final fixes (found on the merged
build): each area's own tests live in tests/test_final_<area>.py.
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestAPresetsPauseIsNotTheUsersChoice(_WindowTestCase):
    """Tuning away from a preset turns off the decoder it turned on
    (LISTEN-05: Decode unticked). The Decoder took that untick for the
    user's own pause, so every protocol the user chose next started
    paused, with no "Tune to" (DEC-E2E-01) and a "paused" chip."""

    demo = True

    def decoding(self):
        panel = self.win._decoder_panel
        return panel.selection(), panel._enabled_check.isChecked()

    def test_the_next_protocol_chosen_decodes(self):
        win, panel = self.win, self.win._decoder_panel
        self.assertTrue(win._apply_preset_named("ACARS"))
        self.assertEqual(self.decoding(), ("ACARS", True))
        self.assertTrue(win.set_frequency(100.1e6))  # away from the preset
        self.assertEqual(self.decoding(), ("ACARS", False))  # (LISTEN-05)
        # (Off, as its message says: no "paused" chip left behind, W-06.)
        self.assertTrue(win._decoder_chip.isHidden())
        panel._proto_combo.setCurrentText("RDS")
        self.assertEqual(self.decoding(), ("RDS", True))
        self.assertEqual(win._decoder_chip.text(), "RDS · 0 msgs")
        panel._proto_combo.setCurrentText("POCSAG")
        self.assertEqual(self.decoding(), ("POCSAG", True))
        self.assertEqual(panel._tune_offer[0], "Tune to 152.240 MHz")

    def test_back_to_the_paused_protocol_it_decodes_too(self):
        win, panel = self.win, self.win._decoder_panel
        self.assertTrue(win._apply_preset_named("ADS-B"))
        self.assertTrue(win.set_frequency(100.1e6))
        self.assertEqual(self.decoding(), ("ADS-B", False))
        panel._proto_combo.setCurrentText("Off")
        panel._proto_combo.setCurrentText("ADS-B")
        self.assertEqual(self.decoding(), ("ADS-B", True))

    def test_a_pause_of_the_users_own_stays(self):
        win, panel = self.win, self.win._decoder_panel
        panel._proto_combo.setCurrentText("POCSAG")
        panel._enabled_check.click()  # untick, as a user does
        panel._proto_combo.setCurrentText("FLEX")
        self.assertEqual(self.decoding(), ("FLEX", False))
        # A preset turns its decoder on (ticked: the pause is over); tuning
        # away turns it off; the user's next protocol decodes.
        self.assertTrue(win._apply_preset_named("ACARS"))
        self.assertEqual(self.decoding(), ("ACARS", True))
        self.assertTrue(win.set_frequency(100.1e6))
        panel._proto_combo.setCurrentText("RDS")
        self.assertEqual(self.decoding(), ("RDS", True))

    def test_ticking_decode_after_the_pause_is_the_users_choice(self):
        win, panel = self.win, self.win._decoder_panel
        self.assertTrue(win._apply_preset_named("ACARS"))
        self.assertTrue(win.set_frequency(100.1e6))
        panel._enabled_check.click()  # ticked again: ACARS decodes here
        self.assertEqual(self.decoding(), ("ACARS", True))
        panel._enabled_check.click()  # ... and unticked: the user's pause
        panel._proto_combo.setCurrentText("RDS")
        self.assertEqual(self.decoding(), ("RDS", False))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestTuneToADSBSaysOnceWhatItSet(_WindowTestCase):
    """The Decoder's ADS-B "Tune to" from broadcast FM left "None (I/Q):
    200 kHz filter, no audio" (LISTEN-13's note for I/Q) in the status
    history under "Tuned to ADS-B (1090.000 MHz, I/Q)", while the tune set
    the ADS-B preset's 2.4 MHz filter right after (DEC-E2E-14). (ACARS
    and bookmarks: test_final_listen's TestBookmarkAndDecoderTunesSayOnce.)"""

    demo = True

    def history(self):
        return [entry.text for entry in self.win._status_history.entries()]

    def test_ads_b_from_broadcast_fm(self):
        win, panel = self.win, self.win._decoder_panel
        self.assertTrue(win.set_frequency(100.1e6))
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        win._status_history.clear()
        panel._proto_combo.setCurrentText("ADS-B")
        panel._tune_to_protocol()  # (the Aircraft view's "Tune to")
        self.assertEqual(win._control_panel._bw_combo.currentText(), "2.4 MHz")
        self.assertEqual(self.history(), ["Tuned to ADS-B (1090.000 MHz, I/Q)"])

    def test_choosing_i_q_by_hand_still_says_what_it_listens_with(self):
        win = self.win
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        win._status_history.clear()
        win._set_demod_mode("None (I/Q)")
        self.assertTrue(self.history()[0].startswith("None (I/Q): "), self.history())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestTheDecoderChipGivesWayAt1024Px(_WindowTestCase):
    """The status bar keeps 220 px for messages at 1024 px with every chip
    (F-window, DEV-E2E-07; its test shows REC, AUDIO, AFC and the decoder
    chip "ADS-B · 6 msgs"). The Decoder's chip says more: the station's
    name, the Morse speed, counts in the thousands. It never gave way, so
    with "CW (Morse) · 15 WPM · 12,345 msgs" the window's minimum width
    became 1,079 px: wider than the screen it was made to fit."""

    demo = True
    LONG = (
        "AX.25/APRS · 12,345 msgs",
        "RDS · SDR DEMO · 1,234 msgs",
        "CW (Morse) · 15 WPM · 12,345 msgs",
    )

    def test_its_long_texts_leave_the_window_at_1024_px(self):
        from tests.test_gui_coherent_window import _ThemedSizes

        _ThemedSizes.each_theme(self, self.check_room)

    def check_room(self, theme):
        from PyQt6.QtWidgets import QLabel

        from sdr_module.gui.themes import set_role

        win = self.win
        win.resize(1024, 640)
        win._recording_label.setText("REC 00:12")
        win._recording_label.setVisible(True)
        win._recording_info_label.setText("26.7 MB · 17.7 GB free")
        win._recording_info_label.setVisible(True)
        win._audio_rec_label.setText("AUDIO 00:05")
        win._audio_rec_label.setVisible(True)
        afc = QLabel("AFC +2.2 kHz")  # (as Track Signal shows it)
        set_role(afc, "badge", "info")
        win.statusBar().addPermanentWidget(afc)
        self.addCleanup(afc.deleteLater)
        chip = win._decoder_chip
        for text in self.LONG:
            chip.setText(text)
            chip.setVisible(True)
            self.settle(10)
            where = (theme, text)
            self.assertLessEqual(win.minimumSizeHint().width(), 1024, where)
            self.assertEqual(win.width(), 1024, where)
            self.assertGreaterEqual(win._message_label.width(), 220, where)
            self.assertEqual(chip.text(), text)  # (whole: only drawn short)
            self.assertGreater(chip.width(), 40, where)  # (its protocol shows)
        win.statusBar().removeWidget(afc)
        # With room, the chip shows its whole text.
        win._recording_label.setVisible(False)
        win._recording_info_label.setVisible(False)
        win._audio_rec_label.setVisible(False)
        self.settle(10)
        self.assertGreaterEqual(chip.width(), chip.sizeHint().width(), theme)


if __name__ == "__main__":
    unittest.main()

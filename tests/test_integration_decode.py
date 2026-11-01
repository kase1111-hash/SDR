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
        self.assertEqual(win._decoder_chip.text(), "ACARS · paused")
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


if __name__ == "__main__":
    unittest.main()

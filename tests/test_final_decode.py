"""Decoder fixes from the end-to-end journeys (DEC-E2E-*) and the review
leftovers (L27-L29): what users of the finished app ran into in the
Decoder tab, the status-bar chip, File > Decode Recording and the decoders
themselves.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication


def _settle(n=5):
    for _ in range(n):
        QApplication.processEvents()


def _status(name, unit="message", **fields):
    """A live decoder's status, as the window passes it to the panel."""
    from sdr_module.dsp.live_decoders import LiveDecoderStatus

    base = dict(name=name, unit=unit, signal_measured=True)
    base.update(fields)
    return LiveDecoderStatus(**base)


class _PanelCase(unittest.TestCase):
    """A Decoder panel on its own, shown, with a clock the test sets."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import decoder_panel

        self.mod = decoder_panel
        self.now = 1000.0
        self.panel = decoder_panel.DecoderPanel()
        self.panel._clock = lambda: self.now
        self.panel.resize(420, 520)
        self.panel.show()
        _settle(2)

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        _settle(1)

    def front(self):
        return self.panel._tabs.currentWidget()


# --------------------------------------------------------------------------- #
# DEC-E2E-02: a protocol's own view comes to the front
# --------------------------------------------------------------------------- #


class TestProtocolViewInFront(_PanelCase):
    def test_straight_from_ads_b_to_cw_and_back(self):
        # The fallback to Messages looked at the tab that was in front
        # before the switch (now hidden) and covered the view just shown.
        p = self.panel
        p._proto_combo.setCurrentText("ADS-B")
        self.assertIs(self.front(), p._aircraft_view)
        p._proto_combo.setCurrentText("CW (Morse)")
        self.assertIs(self.front(), p._cw_view)
        p._proto_combo.setCurrentText("ADS-B")
        self.assertIs(self.front(), p._aircraft_view)
        # Through another protocol it was right already, and stays so.
        p._proto_combo.setCurrentText("POCSAG")
        self.assertIs(self.front(), p._messages_page)
        p._proto_combo.setCurrentText("CW (Morse)")
        self.assertIs(self.front(), p._cw_view)

    def test_auto_follows_the_mode_to_the_text_view(self):
        p = self.panel
        p.set_current_frequency(1090e6)
        p._proto_combo.setCurrentText("Auto")
        self.assertIs(self.front(), p._aircraft_view)
        p.set_demod_mode("CW")  # Auto: Morse code in CW
        self.assertEqual(p.protocol(), "CW (Morse)")
        self.assertIs(self.front(), p._cw_view)

    def test_a_tab_picked_by_hand_stays_while_the_protocol_does(self):
        p = self.panel
        p._proto_combo.setCurrentText("ADS-B")
        p._tabs.setCurrentWidget(p._raw_output)
        p.set_current_frequency(1090.2e6)  # (nothing changes the protocol)
        self.assertIs(self.front(), p._raw_output)


# --------------------------------------------------------------------------- #
# DEC-E2E-01: "Tune to ..." stays offered once messages are listed
# --------------------------------------------------------------------------- #


class TestTuneToStaysOffered(_PanelCase):
    def setUp(self):
        super().setUp()
        self.tuned = []
        self.panel.tune_requested.connect(lambda *args: self.tuned.append(args))
        self.panel.set_receiving(True)

    def test_offered_beside_the_status_line_with_rows_in_the_table(self):
        # The only button was the empty table's: one row (of any protocol)
        # hid it, and only Clear (losing the messages) brought it back.
        p = self.panel
        p.set_current_frequency(144.39e6)
        p._proto_combo.setCurrentText("AX.25/APRS")
        p.add_message("AX.25/APRS", "N0CALL-9", "49.0583, -72.0292")
        p._proto_combo.setCurrentText("POCSAG")
        p.set_decoder_status(_status("POCSAG", "page"))  # (as the window does)
        _settle(1)
        self.assertFalse(p._empty.is_visible())  # the table lists the packet
        self.assertTrue(
            p.decoder_status_text().startswith("Not tuned to POCSAG"),
            p.decoder_status_text(),
        )
        button = p._tune_btn
        self.assertTrue(button.isVisible())
        self.assertEqual(button.text(), "Tune to 152.240 MHz")
        self.assertIn("pager channel", button.toolTip())
        button.click()
        self.assertEqual(self.tuned[-1][:3], (152.24e6, "POCSAG (152.240 MHz)", "FM"))
        p.set_current_frequency(152.24e6)  # (the window reports the tune)
        self.assertFalse(button.isVisible())

    def test_one_button_while_the_empty_view_in_front_has_its_own(self):
        p = self.panel
        p.set_current_frequency(100.1e6)
        p._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        aircraft = p._aircraft_view.placeholder()
        self.assertTrue(aircraft.button.isVisible())
        self.assertFalse(p._tune_btn.isVisible())
        p._tabs.setCurrentWidget(p._raw_output)  # the Log: no button there
        self.assertTrue(p._tune_btn.isVisible())
        p._tabs.setCurrentWidget(p._messages_page)  # the empty table has one
        self.assertTrue(p._empty.button.isVisible())
        self.assertFalse(p._tune_btn.isVisible())
        p.add_message("ADS-B", "4840D6", "KLM1023")
        self.assertTrue(p._tune_btn.isVisible())

    def test_not_offered_while_paused_or_off(self):
        p = self.panel
        p.set_current_frequency(100.1e6)
        p._proto_combo.setCurrentText("POCSAG")
        p.add_message("POCSAG", "1 (F0)", "page")
        self.assertTrue(p._tune_btn.isVisible())
        p._enabled_check.setChecked(False)
        self.assertFalse(p._tune_btn.isVisible())
        p._enabled_check.setChecked(True)
        p._proto_combo.setCurrentText("Off")
        self.assertFalse(p._tune_btn.isVisible())


# --------------------------------------------------------------------------- #
# DEC-E2E-04: Decode Recording's frequency follows the protocol chosen
# --------------------------------------------------------------------------- #


def _options(center=144.09e6, rate=1e6, seconds=10.0, name="aprs_1M_offset.cs16"):
    from sdr_module.gui.iq_files import IqFileOptions

    return IqFileOptions(
        path=os.path.join(os.sep, "recordings", name),
        sample_rate=rate,
        center_hz=center,
        total_samples=int(seconds * rate),
        source="a sidecar file",
    )


class TestDecodeDialogFrequency(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def dialog(self, options, protocol="POCSAG", defaults=None):
        from sdr_module.gui.iq_files import DecodeOptionsDialog

        protocols = ["POCSAG", "FLEX", "AX.25/APRS", "ADS-B", "ACARS", "RDS"]
        start = (defaults or {}).get(protocol, options.center_hz)
        dialog = DecodeOptionsDialog(
            options, protocols, protocol, start, None, defaults=defaults
        )
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_choosing_a_protocol_moves_to_its_channel(self):
        # It stayed on the frequency of the protocol it opened with (the
        # file's center, for POCSAG), and APRS at 144.390 MHz was missed.
        defaults = {"POCSAG": 144.09e6, "AX.25/APRS": 144.39e6, "RDS": 144.09e6}
        dialog = self.dialog(_options(), "POCSAG", defaults)
        self.assertEqual(dialog._freq.text(), "144.090 MHz")
        dialog._protocol.setCurrentText("AX.25/APRS")
        self.assertEqual(dialog._freq.text(), "144.390 MHz")
        dialog._on_accept()
        self.assertEqual(dialog.choice()[0], "AX.25/APRS")
        self.assertAlmostEqual(dialog.choice()[1], 300e3, delta=1.0)

    def test_a_frequency_typed_by_hand_stays(self):
        from PyQt6.QtTest import QTest

        defaults = {"POCSAG": 144.09e6, "AX.25/APRS": 144.39e6}
        dialog = self.dialog(_options(), "POCSAG", defaults)
        dialog._freq.selectAll()
        QTest.keyClicks(dialog._freq, "144.2")
        dialog._protocol.setCurrentText("AX.25/APRS")
        self.assertEqual(dialog._freq.text(), "144.2")

    def test_without_defaults_nothing_changes(self):
        dialog = self.dialog(_options(), "POCSAG")
        dialog._protocol.setCurrentText("AX.25/APRS")
        self.assertEqual(dialog._freq.text(), "144.090 MHz")


class TestDecodeRecordingDefaults(_WindowTestCase):
    def test_the_window_gives_each_protocol_its_channel(self):
        from sdr_module.gui import iq_files

        win = self.win
        win._decoder_panel._proto_combo.setCurrentText("Off")
        self.settle(2)
        self.assertTrue(win.set_frequency(100.1e6))
        options = _options()
        seen = {}

        def fake_exec(dialog):
            if isinstance(dialog, iq_files.DecodeOptionsDialog):
                seen["start"] = dialog._freq.text()
                dialog._protocol.setCurrentText("AX.25/APRS")
                seen["aprs"] = dialog._freq.text()
                dialog._protocol.setCurrentText("ACARS")  # (not in the file)
                seen["acars"] = dialog._freq.text()
            return 0  # cancelled: nothing is decoded

        win._exec_dialog = fake_exec
        self.assertFalse(win._decode_recording(options.path, options))
        self.assertEqual(seen["start"], "144.090 MHz")  # POCSAG: the center
        self.assertEqual(seen["aprs"], "144.390 MHz")
        self.assertEqual(seen["acars"], "144.090 MHz")


if __name__ == "__main__":
    unittest.main()

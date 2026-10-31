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


if __name__ == "__main__":
    unittest.main()

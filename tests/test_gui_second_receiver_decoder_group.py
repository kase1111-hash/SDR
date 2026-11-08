"""The Second Receiver's Decoder group after the merge (rx2 with the
scan-decode lane's M-D2): a busy sky said, and no empty row.

* Busy sky: below 2.5 MS/s the ADS-B decoder reads only so many replies a
  second and, in a very busy sky, leaves the least clear ones unread
  (``LiveDecoderStatus.skipped_per_s``). The Decoder tab says so for the
  main receiver ("Busy sky: about ... replies a second weren't read");
  the Second Receiver's own status line didn't. Now it does, in the same
  words, and its tooltip names this window's Sample rate list (not the
  main window's Receiver group).
* The last message's row: with Protocol Off, or before the first message,
  it was an empty line (the rx2 checker's open low issue). With Protocol
  Off it is hidden now (no message can come); once a protocol is chosen it
  says "No message yet" until the first one, so the group doesn't grow (and
  Show in Decoder Tab doesn't move under the pointer) when it comes.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_adsb_skipped_replies import _sky  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_second_receiver_decoders import _DecodeCase  # noqa: E402

RATE = 2.4e6


def _status(**values):
    from sdr_module.dsp.live_decoders import LiveDecoderStatus

    fields = dict(name="ADS-B", unit="message", sample_rate=RATE, frames=900)
    fields.update(values)
    return LiveDecoderStatus(**fields)


def _line(status, **kwargs):
    from sdr_module.gui.second_receiver_decoding import decoder_line

    args = dict(
        decode=True,
        running=True,
        scanning=False,
        rows=3,
        last_time=None,
        now=100.0,
        tuned_hz=1090e6,
    )
    args.update(kwargs)
    return decoder_line("ADS-B", status, **args)


class TestTheBusySkyIsSaid(unittest.TestCase):
    def setUp(self):
        try:
            import PyQt6  # noqa: F401  (the decoder panel's wording)
        except ImportError:  # pragma: no cover - environment-dependent
            self.skipTest("PyQt6 not available")

    def test_in_the_decoder_tabs_words(self):
        from sdr_module.dsp.live_decoders import skipped_text

        text, tone, tip = _line(_status(skipped=9000, skipped_per_s=4519.0))
        self.assertEqual(text, skipped_text(4519.0, RATE))
        self.assertTrue(text.startswith("Busy sky: about 4,500 replies a second"))
        self.assertEqual(tone, "info")
        # The longer story, with what helps here, then the counts.
        self.assertIn("4,500 a second were left", tip)
        self.assertIn("this window's Device group", tip)
        self.assertNotIn("Receiver > Sample rate", tip)
        self.assertIn("frames heard", tip)

    def test_not_while_none_are_skipped_or_at_2_56_ms_s(self):
        text, _tone, _tip = _line(_status(valid=12, last_time=99.0))
        self.assertNotIn("Busy sky", text)
        self.assertIn("ADS-B", text)
        text, _tone, _tip = _line(_status(sample_rate=2.56e6, skipped_per_s=0.0))
        self.assertNotIn("Busy sky", text)

    def test_a_problem_or_a_pause_comes_first(self):
        busy = _status(skipped_per_s=900.0)
        self.assertTrue(_line(busy, decode=False)[0].startswith("Paused"))
        self.assertTrue(_line(busy, running=False)[0].startswith("Stopped"))

    def test_the_main_windows_tip_is_unchanged(self):
        from sdr_module.dsp.live_decoders import skipped_tip

        tip = skipped_tip(873.0, RATE)
        self.assertIn(
            "Receiver > Sample rate (or click the rate in the status bar)", tip
        )


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestABusySkyInTheWindow(_DecodeCase):
    def test_its_status_line_says_it(self):
        rx = self.rx
        self.start_on(1090e6, ("AM", "", ""))
        self.assertEqual(rx._rate(), RATE)
        self.choose("ADS-B")
        sky = _sky(RATE).astype(np.complex64)
        block = int(RATE / 30)
        for start in range(0, len(sky) - block + 1, block):
            rx._process(sky[start : start + block])
        status = rx._work.decoding.status()
        self.assertGreater(status.skipped_per_s, 0)
        line = rx._decoder_line
        self.assertTrue(line.text().startswith("Busy sky: about"), line.text())
        self.assertEqual(line.property("tone"), "info")
        self.assertIn("this window's Device group", line.toolTip())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheLastMessageRow(_DecodeCase):
    def row_shown(self):
        rx = self.rx
        return rx._decoder_form.isRowVisible(rx._decoder_last)

    def button_top(self):
        from PyQt6.QtCore import QPoint
        from PyQt6.QtWidgets import QApplication

        for _ in range(10):  # (the column lays itself out again)
            QApplication.processEvents()
        rx = self.rx
        return rx._decoder_show.mapTo(rx, QPoint(0, 0)).y()

    def test_hidden_with_protocol_off(self):
        rx = self.rx
        self.choose("Off")
        self.assertFalse(self.row_shown())
        self.choose("AX.25/APRS")
        self.assertTrue(self.row_shown())
        self.assertEqual(rx._decoder_last.text(), "No message yet")
        self.assertEqual(rx._decoder_last.property("tone"), "muted")
        self.choose("Off")
        self.assertFalse(self.row_shown())

    def test_nothing_moves_when_the_first_message_comes(self):
        rx = self.rx
        rx.resize(1024, 640)
        self.start_on()
        self.choose("AX.25/APRS")
        before = self.button_top()
        group_height = rx._decoder_show.parentWidget().height()
        self.run_seconds(2.5)
        self.assertIn("N0CALL-9>APZ001", rx._decoder_last.text())
        self.assertNotEqual(rx._decoder_last.property("tone"), "muted")
        self.assertEqual(self.button_top(), before)
        self.assertEqual(rx._decoder_show.parentWidget().height(), group_height)

    def test_clear_and_another_protocol_say_none_yet(self):
        rx = self.rx
        self.start_on()
        self.choose("AX.25/APRS")
        self.run_seconds(2.5)
        self.assertIn("N0CALL", rx._decoder_last.text())
        self.panel.clear()
        self.assertEqual(rx._decoder_last.text(), "No message yet")
        self.choose("POCSAG")
        self.assertEqual(rx._decoder_last.text(), "No message yet")
        self.assertEqual(rx._decoder_last.toolTip(), "")

    def test_restored_off_it_is_hidden_from_the_start(self):
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        self.choose("Off")
        again = self.extra(SecondReceiverWindow(self.win))
        self.assertFalse(again._decoder_form.isRowVisible(again._decoder_last))
        self.store.values["device/second_receiver/decoder"] = "POCSAG"
        again = self.extra(SecondReceiverWindow(self.win))
        self.assertTrue(again._decoder_form.isRowVisible(again._decoder_last))
        self.assertEqual(again._decoder_last.text(), "No message yet")


if __name__ == "__main__":
    unittest.main()

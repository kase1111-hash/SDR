"""M-R4: a live decoder in the Second Receiver window.

Only the main receiver decoded (APRS, POCSAG, FLEX, ACARS, ADS-B, RDS,
CW...): a second receiver parked on 144.390 MHz couldn't decode APRS. Now
the Second Receiver has its own protocol choice and Decode switch (the main
window's live decoders, ``dsp/live_decoders.py``), decoding on its DSP
worker, never the GUI thread, with the main window's refusals and hints when
the rate is too low (worded for this window). Its messages go into the main
window's Decoder tab, marked "2nd RX · " as Decode Recording's rows are
marked "(file)", at this receiver's frequency and rate; the window shows a
compact status line and the last message, and Show in Decoder Tab brings
them up.
"""

from __future__ import annotations

import os
import threading
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_second_receiver import _FakeSink, _SecondCase  # noqa: E402

APRS_ROW = (
    "N0CALL-9>APZ001",
    "49.0583, -72.0292  ·  Car  ·  Demo mobile on 144.390 MHz",
)


def _on_gui_thread():
    return threading.current_thread() is threading.main_thread()


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class _DecodeCase(_SecondCase):
    def setUp(self):
        super().setUp()
        from sdr_module.gui import device_dialog

        patcher = mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9)
        patcher.start()  # (every demo sample is read)
        self.addCleanup(patcher.stop)
        main_sink = _FakeSink()
        main_sink.open = True
        self.win._audio = main_sink
        self.win._audio_enabled = True
        self.win._audio_problem = None
        self.sink()
        self.panel = self.win._decoder_panel

    def start_on(self, freq=144.39e6, mode=("FM", "5 kHz", "25 kHz")):
        rx = self.rx
        self.assertTrue(rx.start())
        rx._frame_timer.stop()
        rx.tune(freq, recenter=True)
        rx.set_mode(*mode)
        rx.device._stream_pos = 0  # (the start of every demo schedule)

    def choose(self, protocol):
        combo = self.rx._decoder_combo
        index = combo.findData(protocol)
        self.assertGreaterEqual(index, 0, protocol)
        combo.setCurrentIndex(index)

    def run_seconds(self, seconds, pipelined=False):
        rx = self.rx
        block = int(rx._rate() / 30)
        for _ in range(int(seconds * 30) + 1):
            samples = rx.device.read_samples(block)
            if pipelined:
                rx._feed(samples)
            else:
                rx._process(samples)
        rx._settle_frames()

    def rows(self):
        return [m for m in self.panel._messages if m["valid"]]


class TestTheControls(_DecodeCase):
    def test_a_protocol_choice_and_a_decode_switch(self):
        from sdr_module.dsp.live_decoders import LIVE_DECODERS

        rx = self.rx
        combo = rx._decoder_combo
        self.assertEqual(combo.itemData(0), "Off")
        self.assertEqual(
            [combo.itemData(i) for i in range(1, combo.count())], list(LIVE_DECODERS)
        )
        self.assertEqual(combo.currentData(), "Off")
        for i in range(1, combo.count()):
            self.assertTrue(combo.itemData(i, 3))  # (what it decodes, where)
        self.assertTrue(combo.toolTip())
        self.assertTrue(combo.accessibleName())
        self.assertTrue(rx._decode_check.isChecked())
        self.assertTrue(rx._decode_check.toolTip())
        self.assertTrue(rx._decoder_show.toolTip())
        self.assertIsNone(rx._work.decoding.decoder)

    def test_choices_are_saved_and_come_back(self):
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        self.choose("POCSAG")
        self.rx._decode_check.setChecked(False)
        values = self.store.values
        self.assertEqual(values["device/second_receiver/decoder"], "POCSAG")
        self.assertIs(values["device/second_receiver/decode"], False)
        again = self.extra(SecondReceiverWindow(self.win))
        self.assertEqual(again._decoder_combo.currentData(), "POCSAG")
        self.assertFalse(again._decode_check.isChecked())

    def test_a_damaged_choice_is_logged_and_gives_off(self):
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        self.store.values["device/second_receiver/decoder"] = "MORSE-ISH"
        with self.assertLogs("sdr_module.gui", "WARNING") as logs:
            again = self.extra(SecondReceiverWindow(self.win))
        self.assertEqual(again._decoder_combo.currentData(), "Off")
        self.assertIn("device/second_receiver/decoder", "\n".join(logs.output))


class TestItDecodes(_DecodeCase):
    def test_aprs_on_144_390_reaches_the_decoder_tab_marked(self):
        rx = self.rx
        self.start_on()
        self.choose("AX.25/APRS")
        self.run_seconds(2.5)
        rows = self.rows()
        self.assertEqual([(m["address"], m["content"]) for m in rows], [APRS_ROW])
        row = rows[0]
        self.assertEqual(row["protocol"], "2nd RX · AX.25/APRS")
        self.assertEqual(row["freq_hz"], 144.39e6)
        self.assertEqual(row["sample_rate"], rx._rate())
        # The main receiver's own counts and decoder are untouched.
        self.assertEqual(self.panel.count_for("AX.25/APRS"), 0)
        self.assertIsNone(self.win._decoder)
        # The window: the last message and a compact status line.
        self.assertIn("N0CALL-9>APZ001", rx._decoder_last.text())
        self.assertIn("Demo mobile", rx._decoder_last.toolTip())
        line = rx._decoder_line.text()
        self.assertIn("AX.25/APRS", line)
        self.assertIn("1 packet", line)
        self.assertEqual(rx._decoder_line.property("tone"), "success")

    def test_decoding_runs_on_the_worker_never_the_gui_thread(self):
        rx = self.rx
        self.start_on()
        self.choose("AX.25/APRS")
        threads = []
        real = rx._work.decoding.decoder.process

        def watched(block):
            threads.append(_on_gui_thread())
            return real(block)

        rx._work.decoding.decoder.process = watched
        self.run_seconds(2.5, pipelined=True)
        self.assertGreater(len(threads), 60)
        self.assertNotIn(True, threads)
        self.assertEqual(len(self.rows()), 1)  # (shown on the GUI thread)

    def test_decode_unticked_pauses_it(self):
        rx = self.rx
        self.start_on()
        self.choose("AX.25/APRS")
        seen = []
        rx._work.decoding.decoder.process = lambda block: seen.append(len(block)) or []
        rx._decode_check.setChecked(False)
        self.run_seconds(0.2)
        self.assertEqual(seen, [])
        self.assertEqual(rx._decoder_line.text(), "Paused: tick Decode to resume")
        rx._decode_check.setChecked(True)
        self.run_seconds(0.2)
        self.assertTrue(seen)

    def test_not_while_a_scan_sweeps_with_it(self):
        rx = self.rx
        self.start_on()
        self.choose("AX.25/APRS")
        seen = []
        rx._work.decoding.decoder.process = lambda block: seen.append(len(block)) or []
        rx.set_scan_feed(lambda samples: None)
        self.run_seconds(0.2)
        self.assertEqual(seen, [])
        self.assertIn("Scanner", rx._decoder_line.text())
        rx.set_scan_feed(None)
        self.run_seconds(0.2)
        self.assertTrue(seen)

    def test_it_follows_the_vfo(self):
        rx = self.rx
        self.start_on()
        self.choose("POCSAG")
        rx.tune(rx.lo_frequency() + 250e3)  # (inside the span: the VFO moves)
        self.run_seconds(0.1)
        decoder = rx._work.decoding.decoder
        self.assertEqual(decoder.offset_hz, 250e3)

    def test_stopped_it_says_how_to_start(self):
        rx = self.rx
        self.choose("POCSAG")
        self.assertIn("Start", rx._decoder_line.text())


class TestRatesTooLow(_DecodeCase):
    def test_ads_b_says_why_and_what_to_do_here(self):
        rx = self.rx
        index = rx._rate_combo.findData(1.0e6)
        self.assertGreaterEqual(index, 0)
        rx._rate_combo.setCurrentIndex(index)
        rx._rate_combo.activated.emit(index)
        self.start_on(1090e6, ("None (I/Q)", "", "2 MHz"))
        with self.assertLogs("sdr_module.gui", "WARNING") as logs:
            self.choose("ADS-B")
        line = rx._decoder_line.text()
        self.assertIn("ADS-B needs at least 2 MS/s", line)
        self.assertIn("Sample rate", line)  # (this window's, not Receiver >)
        self.assertNotIn("Receiver >", line)
        self.assertEqual(rx._decoder_line.property("tone"), "warning")
        self.assertIn("ADS-B needs at least 2 MS/s", rx.message())
        self.assertTrue(any("ADS-B" in line for line in logs.output))
        # A rate it can use: rebuilt, the problem gone.
        index = rx._rate_combo.findData(2.4e6)
        rx._rate_combo.setCurrentIndex(index)
        rx._rate_combo.activated.emit(index)
        decoder = rx._work.decoding.decoder
        self.assertEqual(decoder.sample_rate, 2.4e6)
        self.assertIsNone(decoder.status.problem)
        self.assertNotIn("needs at least", rx._decoder_line.text())

    def test_a_new_rate_too_low_says_why_after_the_span(self):
        # (Review: choosing 1 MS/s with ADS-B on said why, then "1 MS/s:
        # ... in view." took its place at once.)
        rx = self.rx
        self.start_on(1090e6, ("None (I/Q)", "", "2 MHz"))
        self.choose("ADS-B")
        self.assertNotIn("needs at least", rx._decoder_line.text())
        index = rx._rate_combo.findData(1.0e6)
        rx._rate_combo.setCurrentIndex(index)
        with self.assertLogs("sdr_module.gui", "WARNING"):
            rx._rate_combo.activated.emit(index)
        message = rx.message()
        self.assertIn("1 MS/s: ", message)  # (the span, as before)
        self.assertIn("ADS-B needs at least 2 MS/s", message)
        self.assertIn("Choose 2 MS/s or more in Sample rate", message)
        self.assertEqual(rx._message_label.property("tone"), "warning")

    def test_chosen_again_it_is_said_again(self):
        # (Review: once said for a protocol and rate, choosing it again
        # there said nothing in the status line; the main window says it
        # each time its decoder is built.)
        rx = self.rx
        index = rx._rate_combo.findData(1.0e6)
        rx._rate_combo.setCurrentIndex(index)
        rx._rate_combo.activated.emit(index)
        self.start_on(1090e6, ("None (I/Q)", "", "2 MHz"))
        with self.assertLogs("sdr_module.gui", "WARNING"):
            self.choose("ADS-B")
        self.assertIn("ADS-B needs at least 2 MS/s", rx.message())
        rx._clear_message()
        self.choose("Off")
        with self.assertLogs("sdr_module.gui", "WARNING"):
            self.choose("ADS-B")
        self.assertIn("ADS-B needs at least 2 MS/s", rx.message())


class TestTheDecoderTab(_DecodeCase):
    def test_show_in_decoder_tab(self):
        self.win._show_panel("bookmarks")
        self.assertFalse(self.win._decoder_page_visible())
        self.rx._decoder_show.click()
        self.assertTrue(self.win._decoder_page_visible())

    def test_a_minimized_main_window_comes_back(self):
        # (Check: with the main window minimized the button did nothing
        # one could see.)
        win = self.win
        win._show_panel("bookmarks")
        win.showMinimized()
        self.settle()
        self.assertTrue(win.isMinimized())
        self.rx._decoder_show.click()
        self.settle()
        self.assertFalse(win.isMinimized())
        self.assertTrue(win._decoder_page_visible())

    def test_clear_there_clears_the_last_message_here(self):
        rx = self.rx
        self.start_on()
        self.choose("AX.25/APRS")
        self.run_seconds(2.5)
        self.assertTrue(rx._decoder_last.text())
        self.panel.clear()
        # (Gone; the row says none has come since: it keeps its place.)
        self.assertEqual(rx._decoder_last.text(), "No message yet")
        self.assertEqual(rx._decoder_last.toolTip(), "")
        self.assertNotIn("1 packet", rx._decoder_line.text())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestAddMessageReceiver(_WindowTestCase):
    """decoder_panel.add_message's new ``receiver`` keyword (backwards
    compatible): another receiver's rows say so before the protocol, are
    counted apart and stay out of the main receiver's views."""

    def test_rows_of_another_receiver(self):
        from sdr_module.dsp.morse import MorseMessage
        from sdr_module.dsp.protocols import ProtocolType

        panel = self.win._decoder_panel
        panel.add_message("POCSAG", "1234567 (F3)", "HELLO")  # (as before)
        line = MorseMessage(ProtocolType("cw"), 0.0, b"", True, text="CQ CQ")
        panel.add_message("CW (Morse)", "", "CQ CQ", message=line, receiver="2nd RX")
        protocols = [m["protocol"] for m in panel._messages]
        self.assertEqual(protocols, ["POCSAG", "2nd RX · CW (Morse)"])
        self.assertEqual(panel.count_for("CW (Morse)"), 0)
        self.assertEqual(panel._cw_view.plain_text(), "")
        # The main receiver's own Morse still reaches its Text view.
        panel.add_message("CW (Morse)", "", "CQ CQ", message=line)
        self.assertIn("CQ CQ", panel._cw_view.plain_text())

    def test_the_protocol_column_says_whose_rows_they_are(self):
        # With nothing chosen in the selector and one protocol in the table
        # the column hides (it would repeat nothing): not for another
        # receiver's rows, whose mark it carries.
        from sdr_module.gui import decoder_panel as mod

        panel = self.win._decoder_panel
        table = panel._table
        panel.add_message("AX.25/APRS", "N0CALL-9", "a", receiver="2nd RX")
        self.assertIsNone(panel.protocol())
        self.assertFalse(table.isColumnHidden(mod.COL_PROTOCOL))
        self.assertEqual(
            table.item(0, mod.COL_PROTOCOL).toolTip(), "2nd RX · AX.25/APRS"
        )
        panel.clear()
        panel.add_message("AX.25/APRS", "N0CALL-9", "a")  # (as before)
        self.assertTrue(table.isColumnHidden(mod.COL_PROTOCOL))


if __name__ == "__main__":
    unittest.main()

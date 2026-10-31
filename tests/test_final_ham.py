"""Final fixes in the ham area (the end-to-end journeys' problems): the
window while on the air, SSTV Start Now with Auto, the S-meter's copied
report, the SSTV hints, the Ham ID preview's audio output, the second
receiver's spectrum defaults and transmit monitor line, the 60 m wording,
the SSTV history after a restart, the radio tuner without a device and the
Ham ID tooltips.

A HackRFDevice runs on the threaded fake ``python_hackrf`` backend
(tests/hackrf_fakes.py) where something is transmitted.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sdr_module.ham.sstv import find_mode  # noqa: E402
from sdr_module.ham.sstv_encoder import encode_sstv, sstv_test_card  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_smeter import _PanelCase as _SMeterCase  # noqa: E402
from tests.test_gui_smeter import noise, tone  # noqa: E402
from tests.test_gui_sstv import FS, _PanelCase, correlation  # noqa: E402
from tests.test_gui_station_tx import _TxCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication  # noqa: F401

ON_AIR_TEXT = "Transmitting - receiving resumes when it ends"


class TestWindowWhileOnTheAir(_TxCase):
    """HAM-E2E-1: while transmitting, the plots say receiving resumes when
    it ends and Start is off (the HackRF is half duplex), so nothing
    invites a press of Space that would only be refused."""

    def _show_a_frame(self):
        """The plots have something to dim (the fake HackRF receives
        nothing)."""
        self.win._spectrum.update_spectrum(np.full(1024, -90.0))
        self.win._waterfall.add_line(np.full(1024, -90.0))

    def _send_slow_id(self):
        self._show_a_frame()
        self.panel.set_callsign("W1AW")
        self.panel._wpm_spin.setValue(5)  # (a long ID: about 10 s)
        self.station.send_station_id()
        self.assertTrue(self.station.busy)
        QApplication.processEvents()

    def _assert_on_air_state(self):
        win = self.win
        self.assertEqual(win._spectrum.stale_text(), ON_AIR_TEXT)
        self.assertEqual(win._waterfall.stale_text(), ON_AIR_TEXT)
        button = win._start_button
        self.assertFalse(button.isEnabled())
        self.assertNotEqual(button.property("role"), "primary")
        self.assertIn("transmission ends", button.toolTip())
        self.assertFalse(win._start_action.isEnabled())
        self.assertEqual(win._state_badge.text(), "ON AIR")

    def test_receiver_running_before(self):
        self.hackrf(pace=True)
        self.win._start_acquisition()
        self.assertTrue(self.win._is_running)
        self._send_slow_id()
        self._assert_on_air_state()
        # Anything that refreshes the state keeps it (e.g. a theme or
        # device-controls sync while on the air).
        self.win._refresh_state_ui()
        self._assert_on_air_state()
        self.station.stop()
        self.wait_done(5)
        QApplication.processEvents()
        # Receiving resumed: live plots, Stop offered.
        self.assertTrue(self.win._is_running)
        self.assertIsNone(self.win._spectrum.stale_text())
        self.assertIsNone(self.win._waterfall.stale_text())
        self.assertTrue(self.win._start_button.isEnabled())
        self.assertTrue(self.win._start_action.isEnabled())
        self.win._stop_acquisition()

    def test_receiver_stopped_before(self):
        self.hackrf(pace=True)
        self.assertFalse(self.win._is_running)
        self._send_slow_id()
        self._assert_on_air_state()
        self.station.stop()
        self.wait_done(5)
        QApplication.processEvents()
        self.assertFalse(self.win._is_running)
        self.assertEqual(self.win._spectrum.stale_text(), "Stopped - press Space")
        self.assertEqual(self.win._waterfall.stale_text(), "Stopped - press Space")
        button = self.win._start_button
        self.assertTrue(button.isEnabled())
        self.assertEqual(button.property("role"), "primary")
        self.assertEqual(button.toolTip(), "Start receiving (Space)")
        self.assertTrue(self.win._start_action.isEnabled())

    def test_recording_armed_on_the_air_says_when_it_captures(self):
        self.hackrf(pace=True)
        self._send_slow_id()
        self.assertTrue(self.win._start_recording())
        text = self.win._message_label.text()
        self.assertIn("when receiving resumes after the transmission", text)
        self.assertNotIn("(Space)", text)
        self.win._stop_recording()
        self.station.stop()
        self.wait_done(5)


class TestStartNowOnAuto(_PanelCase):
    """HAM-E2E-2: Start Now with Mode "Auto (VIS)" from an idle decoder (or
    just after Stop Decoder) failed at once, as the decoder had heard no
    line syncs yet, while the tone line went on to name the mode. It now
    listens a few seconds for the syncs that tell the mode."""

    def robot36(self):
        spec = find_mode("Robot 36")
        card = sstv_test_card(spec.width, spec.height)
        return spec, card, encode_sstv(card, spec, FS, vis=False)

    def assert_decodes(self, spec, card, min_rows=40, min_corr=0.93):
        """(Robot 36 sends colour at half resolution: started mid-image it
        correlates 0.95-0.97 with the card, as with the mode chosen.)"""
        decoder = self.panel._decoder
        self.assertTrue(decoder.is_receiving())
        self.assertEqual(decoder.get_mode().name, spec.name)
        rows = decoder.state.current_line - 1
        self.assertGreater(rows, min_rows)
        best = max(
            correlation(decoder.get_image()[:rows], card[o : o + rows])
            for o in range(0, spec.height - rows)
        )
        self.assertGreater(best, min_corr)

    def test_from_an_idle_decoder(self):
        spec, card, audio = self.robot36()
        self.assertFalse(self.panel.is_receiving())
        self.assertEqual(self.panel.mode_choice(), "Auto (VIS)")
        self.panel._start_now_btn.click()
        self.assertTrue(self.panel.is_receiving(), "Start Now starts listening")
        status = self.panel._status_label
        self.assertEqual(status.text(), "Listening for line syncs to tell the mode…")
        self.assertNotEqual(status.property("tone"), "warning")
        start = int(len(audio) * 0.25)  # the transmission is under way
        self.feed(audio[start : start + int(1.5 * FS)])
        self.assertTrue(self.panel._decoder.is_receiving())
        self.assertIn("Robot 36", status.text())
        self.assertNotEqual(status.property("tone"), "warning")
        self.feed(audio[start + int(1.5 * FS) : start + int(10 * FS)])
        self.assert_decodes(spec, card)

    def test_right_after_stop_decoder(self):
        """j2_1: Stop Decoder mid-image, then Start Now at once: the rest of
        the image was never decoded."""
        spec = find_mode("Robot 36")
        card = sstv_test_card(spec.width, spec.height)
        audio = encode_sstv(card, spec, FS)
        self.panel._start_btn.click()
        half = len(audio) // 3
        self.feed(audio[:half])
        self.assertTrue(self.panel._decoder.is_receiving())
        self.panel._start_btn.click()  # Stop Decoder (keeps the partial)
        self.assertEqual(len(self.history_texts()), 1)
        self.panel._start_now_btn.click()
        self.feed(audio[half : half + int(10 * FS)])
        self.assert_decodes(spec, card)

    def test_gives_up_when_no_syncs_come_and_clears_later(self):
        from sdr_module.ham.gui.sstv_panel import START_NOW_AUTO_WAIT_S

        rng = np.random.default_rng(3)
        hiss = (0.3 * rng.standard_normal(int(12 * FS))).astype(np.float32)
        status = self.panel._status_label
        self.panel._start_now_btn.click()
        self.feed(hiss[: int((START_NOW_AUTO_WAIT_S - 0.5) * FS)])
        self.assertEqual(status.text(), "Listening for line syncs to tell the mode…")
        self.feed(hiss[int((START_NOW_AUTO_WAIT_S - 0.5) * FS) :])
        self.assertEqual(
            status.text(),
            "Can't tell the mode: no line syncs were heard. Choose the mode in "
            "the Mode list, then press Start Now.",
        )
        self.assertEqual(status.property("tone"), "warning")
        self.assertFalse(self.panel._decoder.is_receiving())
        # Line syncs turn up later (a transmission without its VIS): the
        # warning gives way to what the tone line says.
        _spec, _card, audio = self.robot36()
        self.feed(audio[int(len(audio) * 0.3) : int(len(audio) * 0.3 + 1.5 * FS)])
        self.panel._update_status()
        self.assertEqual(
            status.text(),
            "Robot 36 line syncs heard: press Start Now to decode the image "
            "under way.",
        )
        self.assertNotEqual(status.property("tone"), "warning")
        self.assertFalse(self.panel._decoder.is_receiving())

    def test_stop_decoder_cancels_the_wait(self):
        self.panel._start_now_btn.click()
        self.panel._start_btn.click()  # Stop Decoder
        self.assertFalse(self.panel.is_receiving())
        _spec, _card, audio = self.robot36()
        self.panel._start_btn.click()  # Start Decoder (no Start Now)
        self.feed(audio[int(len(audio) * 0.3) : int(len(audio) * 0.3 + 2 * FS)])
        self.assertFalse(self.panel._decoder.is_receiving())


class TestStartNowWithoutAudio(_PanelCase):
    """L23: Start Now with no audio coming in ended after 1.5 s with "the
    audio stopped for a while" although none ever came."""

    def test_chosen_mode(self):
        self.panel.set_mode_choice("Robot 36")
        self.panel._start_now_btn.click()
        self.assertTrue(self.panel._decoder.is_receiving())
        self.panel._last_audio_time -= 2.0  # 2 s later, still no audio
        self.panel._update_status()
        text = self.panel._status_label.text()
        self.assertNotIn("stopped for a while", text)
        self.assertEqual(
            text,
            "Start Now heard no audio to decode. Start the receiver, tuned to "
            "the SSTV signal, then press Start Now.",
        )
        self.assertFalse(self.panel._decoder.is_receiving())
        self.assertTrue(self.panel.is_receiving(), "still listening")
        self.assertEqual(self.history_texts(), [])

    def test_auto(self):
        self.panel._start_now_btn.click()
        self.panel._last_audio_time -= 2.0
        self.panel._update_status()
        self.assertEqual(
            self.panel._status_label.text(),
            "Start Now heard no audio to decode. Start the receiver, tuned to "
            "the SSTV signal, then press Start Now.",
        )
        # The wait is over: syncs heard later don't start an image.
        spec = find_mode("Robot 36")
        audio = encode_sstv(sstv_test_card(320, 240), spec, FS, vis=False)
        self.feed(audio[int(len(audio) * 0.3) : int(len(audio) * 0.3 + 2 * FS)])
        self.assertFalse(self.panel._decoder.is_receiving())

    def test_audio_that_stops_is_still_said_so(self):
        spec = find_mode("Robot 36")
        audio = encode_sstv(sstv_test_card(320, 240), spec, FS, vis=False)
        self.panel.set_mode_choice("Robot 36")
        self.panel._start_now_btn.click()
        self.feed(audio[int(len(audio) * 0.3) : int(len(audio) * 0.6)])
        self.assertTrue(self.panel._decoder.is_receiving())
        self.panel._last_audio_time -= 2.0  # the receiver stopped
        self.panel._update_status()
        self.assertIn("the audio stopped for a while", self.panel._status_label.text())


class TestCopiedReportIsTheReadingShown(_SMeterCase):
    """HAM-E2E-4: the panel showed S9+25 while Copy put S9+30 in the log
    (the core rounds over S9 to 10 dB; the panel reads 1 dB)."""

    def test_copy_report_gives_the_s_reading_shown(self):
        panel = self.panel
        panel.set_frequency_source(lambda: 100.1e6)
        seen = set()
        for amplitude in (0.05, 0.1, 0.02, 0.08):
            self.show_reading(tone(amplitude) + noise())
            shown = panel._s_meter_label.text()
            seen.add(shown)
            text = panel.copy_report()
            self.assertRegex(
                text, r"^RST 5\d · S9\+\d+ · 100\.100 MHz · \d\d:\d\d UTC$"
            )
            self.assertEqual(text.split(" · ")[1], shown, text)
        self.assertGreater(len(seen), 2)

    def test_the_spoken_report_keeps_ten_db_steps(self):
        self.show_reading(tone(0.1) + noise())  # S9+23
        self.assertEqual(self.panel._s_meter_label.text(), "S9+23")
        self.assertEqual(self.panel._verbal_label.text(), "Five and nine, 20 over")

    def test_library_default_is_unchanged(self):
        from sdr_module.ham.signal_meter import SignalMeter

        meter = SignalMeter()
        for _ in range(3):
            meter.update(tone(0.1) + noise())
        self.assertEqual(
            meter.get_log_report(14.25e6, when=0.0),
            "RST 59 · S9+20 · 14.250 MHz · 00:00 UTC",
        )
        self.assertEqual(
            meter.get_log_report(14.25e6, when=0.0, s_text="S9+23"),
            "RST 59 · S9+23 · 14.250 MHz · 00:00 UTC",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

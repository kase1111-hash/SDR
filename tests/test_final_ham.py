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
import tempfile
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sdr_module.ham.sstv import find_mode  # noqa: E402
from sdr_module.ham.sstv_encoder import encode_sstv, sstv_test_card  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_qrp import _PanelCase as _QrpCase  # noqa: E402
from tests.test_gui_smeter import _PanelCase as _SMeterCase  # noqa: E402
from tests.test_gui_smeter import noise, tone  # noqa: E402
from tests.test_gui_sstv import FS, _PanelCase, correlation  # noqa: E402
from tests.test_gui_station_tx import _TxCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import QPoint, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QScrollArea

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


def settle(n: int = 5) -> None:
    for _ in range(n):
        QApplication.processEvents()


class _SSTVWindowCase(_WindowTestCase):
    """The real window (demo device) at 1366x768 with the SSTV tab shown."""

    demo = True

    def setUp(self):
        super().setUp()
        if not self.mw.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")
        self.tmp = tempfile.TemporaryDirectory()
        self.panel = self.win._sstv_panel
        self.panel.set_save_directory(self.tmp.name)
        self.win._audio.write = lambda audio: None
        self.win.resize(1366, 768)
        self.win._show_panel("sstv")
        settle()
        area = self.panel.parentWidget()
        while area is not None and not isinstance(area, QScrollArea):
            area = area.parentWidget()
        self.area = area

    def tearDown(self):
        if HAS_PYQT6 and getattr(self, "panel", None) and self.panel.is_receiving():
            self.panel._start_btn.click()
        super().tearDown()
        if hasattr(self, "tmp"):
            self.tmp.cleanup()

    def in_view(self, widget) -> bool:
        if not widget.isVisible():
            return False
        top = widget.mapTo(self.area.viewport(), QPoint(0, 0)).y()
        return top >= 0 and top + widget.height() <= self.area.viewport().height()

    def mode(self) -> str:
        return self.win._control_panel._demod_combo.currentText()


class TestSSTVSwitchKeepsThePanelInView(_SSTVWindowCase):
    """HAM-E2E-5: Switch to USB took the keyboard focus to the Auto-save
    check box (Space then un-ticked it) and scrolled the tab to the bottom,
    so the next hint (Switch to LSB) came up out of view."""

    def test_switch_then_the_next_hint(self):
        win, panel = self.win, self.panel
        bar = self.area.verticalScrollBar()
        self.assertGreater(bar.maximum(), 0, "the tab scrolls at 1366x768")
        win.set_frequency(14.23e6)
        win._set_demod_mode("USB")
        panel._start_btn.click()  # Start Decoder
        win._set_demod_mode("AM")
        settle()
        self.assertTrue(self.in_view(panel._fm_callout))
        QTest.mouseClick(panel._switch_fm_btn, Qt.MouseButton.LeftButton)
        settle()
        self.assertEqual(self.mode(), "USB")
        self.assertFalse(panel._fm_callout.isVisible())
        self.assertIsNot(QApplication.focusWidget(), panel._auto_save_check)
        self.assertEqual(bar.value(), 0)
        self.assertTrue(self.in_view(panel._start_btn))
        # Space is the window's Start/Stop again, not a panel control's.
        self.assertTrue(panel._auto_save_check.isChecked())
        # 40 m in AM: the Switch to LSB hint is in view.
        win.set_frequency(7.171e6)
        win._set_demod_mode("AM")
        settle()
        self.assertEqual(panel._switch_fm_btn.text(), "Switch to LSB")
        self.assertTrue(self.in_view(panel._fm_callout))

    def test_a_hint_scrolls_into_view(self):
        win, panel = self.win, self.panel
        bar = self.area.verticalScrollBar()
        win.set_frequency(14.23e6)
        win._set_demod_mode("USB")
        panel._start_btn.click()
        bar.setValue(bar.maximum())  # looking at Received Images
        settle()
        self.assertFalse(self.in_view(panel._start_btn))
        win._set_demod_mode("AM")
        settle()
        self.assertTrue(self.in_view(panel._fm_callout))

    def test_keyboard_switch_leaves_the_focus_in_the_panel(self):
        win, panel = self.win, self.panel
        bar = self.area.verticalScrollBar()
        win.set_frequency(14.23e6)
        win._set_demod_mode("USB")
        panel._start_btn.click()
        win._set_demod_mode("AM")
        settle()
        panel._switch_fm_btn.setFocus(Qt.FocusReason.TabFocusReason)
        settle()
        self.assertTrue(panel._switch_fm_btn.hasFocus())
        QTest.keyClick(panel._switch_fm_btn, Qt.Key.Key_Space)
        settle()
        self.assertEqual(self.mode(), "USB")
        self.assertIs(QApplication.focusWidget(), panel._start_btn)
        self.assertEqual(bar.value(), 0)


class TestSSTVWrongSideband(_SSTVWindowCase):
    """HAM-E2E-6: from 14.230 MHz (USB) to 7.171 MHz the receiver stays in
    USB (on purpose, for FT8), where SSTV is sent in LSB: no hint."""

    def test_40m_in_usb_offers_lsb(self):
        win, panel = self.win, self.panel
        win.set_frequency(14.23e6)
        win._set_demod_mode("USB")
        panel._start_btn.click()
        settle()
        self.assertTrue(panel._fm_callout.isHidden())
        win.set_frequency(7.171e6)
        settle()
        self.assertEqual(self.mode(), "USB")
        self.assertFalse(panel._fm_callout.isHidden())
        self.assertEqual(
            panel._fm_label.text(),
            "SSTV on 40 m is usually sent in LSB; the receiver is in USB.",
        )
        self.assertEqual(panel._switch_fm_btn.text(), "Switch to LSB")
        panel._switch_fm_btn.click()
        settle()
        self.assertEqual(self.mode(), "LSB")
        self.assertTrue(panel._fm_callout.isHidden())
        # Back on 20 m in LSB: USB is offered.
        win.set_frequency(14.23e6)
        settle()
        if self.mode() == "LSB":
            self.assertEqual(panel._switch_fm_btn.text(), "Switch to USB")
            self.assertFalse(panel._fm_callout.isHidden())

    def test_no_hint_when_not_listening_or_on_vhf(self):
        win, panel = self.win, self.panel
        win.set_frequency(7.171e6)
        win._set_demod_mode("USB")
        settle()
        self.assertTrue(panel._fm_callout.isHidden(), "decoder not listening")
        panel._start_btn.click()
        win.set_frequency(144.5e6)
        win._set_demod_mode("USB")  # (SSB SSTV on 2 m is fine)
        settle()
        self.assertTrue(panel._fm_callout.isHidden())
        win.set_frequency(14.23e6)
        win._set_demod_mode("USB")
        settle()
        self.assertTrue(panel._fm_callout.isHidden())


class _FakeSink:
    """Stands in for gui.audio_sink.AudioSink (no sound card here) and
    logs what is asked of it."""

    log: list = []

    def __init__(self):
        self.log.append(("new",))

    def set_volume(self, volume):
        self.log.append(("volume", round(float(volume), 2)))

    def set_output_device(self, device_id, name=""):
        self.log.append(("output", device_id, name))

    def start(self, rate=48000):
        self.log.append(("start", rate))
        return True

    def write(self, samples):
        self.log.append(("write", len(samples)))

    def stop(self):
        self.log.append(("stop",))


class TestPreviewIdPlaysOnThePreferredOutput(_WindowTestCase):
    """HAM-E2E-7: Ham ID > Preview ID opened its own output on the system
    default at a fixed level, unlike the receiver and Transmit Text's
    Preview (the output chosen in Tools > Preferences, the app volume)."""

    def setUp(self):
        super().setUp()
        if not self.mw.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")
        from sdr_module.gui import transmit_text

        _FakeSink.log = []
        old = transmit_text.AudioSink
        transmit_text.AudioSink = _FakeSink
        self.addCleanup(setattr, transmit_text, "AudioSink", old)
        self.store.values["prefs/audio_output"] = "usb-headset-0001"
        self.store.values["prefs/audio_output_name"] = "USB Headset"
        self.store.values["audio_volume"] = 40

    def test_preview_id(self):
        panel = self.win._callsign_panel
        panel.set_callsign("W1AW")
        panel._test_btn.click()
        log = _FakeSink.log
        self.assertIn(("output", "usb-headset-0001", "USB Headset"), log)
        self.assertIn(("volume", 0.4), log)
        self.assertIn(("start", 48000), log)
        self.assertTrue(any(entry[0] == "write" for entry in log))
        self.assertEqual(panel._test_btn.text(), "Stop Preview")
        self.assertTrue(
            panel._status_label.text().startswith("Playing preview: DE W1AW")
        )
        panel._test_btn.click()  # Stop Preview
        self.assertEqual(log[-1], ("stop",))
        self.assertEqual(panel._test_btn.text(), "Preview ID")

    def test_the_preview_ends_by_itself(self):
        panel = self.win._callsign_panel
        panel.set_callsign("W1AW")
        panel._test_btn.click()
        preview = panel._preview
        clock = [preview._clock()]
        preview._clock = lambda: clock[0]
        clock[0] += 60.0  # (long after the ID)
        preview.tick()
        settle()
        self.assertEqual(_FakeSink.log[-1], ("stop",))
        self.assertEqual(panel._test_btn.text(), "Preview ID")
        self.assertNotIn("Playing preview", panel._status_label.text())


class TestSecondReceiverSpectrum(_WindowTestCase):
    """HAM-E2E-8: the second receiver's spectrum started with Peak hold on
    and a fixed range, while the main window's defaults are Peak hold off
    and Range Auto."""

    demo = True

    def open_second(self):
        rx = self.win._second_rx.show()
        rx._status_timer.stop()
        self.addCleanup(rx.shutdown)
        settle()
        return rx

    def test_same_defaults_as_the_main_window(self):
        main = self.win._spectrum.display_settings()
        self.assertFalse(main["peak_hold"])
        self.assertTrue(main["range_auto"])
        rx = self.open_second()
        self.assertFalse(rx._spectrum.peak_hold())
        self.assertTrue(rx._spectrum.display_settings()["range_auto"])

    def test_its_display_choices_are_kept(self):
        rx = self.open_second()
        rx._spectrum._peak_check.setChecked(True)  # as a click does
        rx.shutdown()
        rx.close()
        settle()
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        again = SecondReceiverWindow(self.win)
        self.addCleanup(again.shutdown)
        self.assertTrue(again._spectrum.peak_hold())
        self.assertTrue(again._spectrum.display_settings()["range_auto"])
        # (The main window's own choice is its own.)
        self.assertFalse(self.win._spectrum.peak_hold())


class TestSecondReceiverTxLine(_TxCase):
    """HAM-E2E-9: the second receiver's Transmit line named the engine's
    job class, "On air: RTTY ... on 14.080 MHz (DATA).", where the main
    window says how it is sent ("45.45 Bd, 170 Hz shift")."""

    license = "general"

    def setUp(self):
        super().setUp()
        self.rx = self.win._second_rx.show()
        self.rx._status_timer.stop()
        self.addCleanup(self.rx.shutdown)

    def text_job(self, summary="45.45 Bd, 170 Hz shift"):
        from sdr_module.gui.transmitter import ArraySource, TxJob

        rate = 2e6
        tone = np.exp(2j * np.pi * 1000.0 * np.arange(int(5 * rate)) / rate)
        return TxJob(
            ArraySource(tone.astype(np.complex64)),
            rate,
            14.080e6,
            "DATA",
            250.0,
            "RTTY “TEST DE W1AW”",
            summary,
            identify=False,
        )

    def test_rtty_on_air(self):
        self.hackrf(pace=True, freq=14.080e6)
        self.assertTrue(self.station.transmit(self.text_job(), confirm=False))
        self.rx._update_tx_group()
        text = self.rx._tx_label.text()
        self.assertIn(
            "On air: RTTY “TEST DE W1AW” on 14.080 MHz (45.45 Bd, 170 Hz shift).",
            text,
        )
        self.assertNotIn("DATA", text)
        self.station.stop()
        self.wait_done(5)

    def test_a_job_without_a_summary_reads_in_words(self):
        self.hackrf(pace=True, freq=14.080e6)
        self.assertTrue(self.station.transmit(self.text_job(""), confirm=False))
        self.rx._update_tx_group()
        text = self.rx._tx_label.text()
        self.assertIn("on 14.080 MHz (data on the carrier).", text)
        self.assertNotIn("DATA", text)
        self.station.stop()
        self.wait_done(5)


class _LicenseCase(unittest.TestCase):
    def setUp(self):
        from sdr_module.core import frequency_manager

        self.fm = frequency_manager.get_frequency_manager()
        before = self.fm.get_license_class()
        self.addCleanup(lambda: self.fm.set_license_class(before))

    def license(self, name):
        from sdr_module.core.frequency_manager import LicenseClass

        self.fm.set_license_class(LicenseClass(name))

    @staticmethod
    def problem(freq_hz, width_hz, mode):
        from sdr_module.gui.transmitter import legality_problem

        return legality_problem(freq_hz, width_hz, mode, quiet=True)


class TestSixtyMetreWording(_LicenseCase):
    """HAM-E2E-11: a Technician at 5.3305 MHz was told the license doesn't
    include "the signal's edge at 5.330 MHz": the refused frequency came
    back rounded to 3 decimals, 500 Hz off, and was taken for an edge."""

    def test_a_band_the_class_has_no_part_of_is_named(self):
        self.license("technician")
        self.assertEqual(
            self.problem(5.3305e6, 0.0, "USB"),
            "your license (Technician) doesn't include 60 m (5.3305 MHz)",
        )
        self.assertEqual(
            self.problem(5.3465e6, 2.8e3, "USB"),
            "your license (Technician) doesn't include 60 m (5.3465 MHz)",
        )
        self.assertEqual(
            self.problem(14.2e6, 60.0, "CW"),
            "your license (Technician) doesn't include 20 m (14.200 MHz)",
        )
        # A band with privileges for the class: the frequency alone.
        self.assertEqual(
            self.problem(7.15e6, 2.7e3, "LSB"),
            "your license (Technician) doesn't include 7.150 MHz",
        )

    def test_edges_are_named_to_the_hertz(self):
        self.license("technician")
        # A CW signal 30 Hz above the edge of the Technician's segment.
        self.assertEqual(
            self.problem(7.02503e6, 100.0, "CW"),
            "your license (Technician) doesn't include the signal's edge at "
            "7.02498 MHz",
        )
        self.license("general")
        self.assertEqual(
            self.problem(5.3305e6, 60.0, "CW"),
            "the signal's edge at 5.33047 MHz is outside the amateur and "
            "license-free bands",
        )

    def test_the_frequency_manager_names_the_frequency_exactly(self):
        self.license("technician")
        allowed, reason = self.fm.is_tx_allowed(5.3305e6, 0.0, "USB")
        self.assertFalse(allowed)
        self.assertIn("5.3305 MHz", reason)
        allowed, reason = self.fm.is_tx_allowed(14.2e6, 0.0, "USB")
        self.assertIn("14.200 MHz", reason)


class TestQrpCardDigits(_QrpCase):
    """HAM-E2E-11: the QRP card read 5.330 MHz for the 5.3305 MHz channel."""

    def test_half_khz_channel(self):
        self.license("general")
        self.panel.set_station(5.3305e6, "USB")
        self.assertEqual(
            self.panel._legal_label.text(),
            "At 5.3305 MHz (General, SSB): legal limit 100 W PEP.",
        )
        self.license("technician")
        self.panel.set_station(5.3305e6, "USB")
        self.assertEqual(
            self.panel._legal_label.text(),
            "Not permitted at 5.3305 MHz in SSB: your license (Technician) "
            "doesn't include 60 m (5.3305 MHz).",
        )


class TestSSTVHistoryAfterARestart(_PanelCase):
    """HAM-E2E-12: after a restart Received Images listed the saved images,
    but the viewer said "No image yet" and Save Image was off until a row
    was clicked."""

    def write_image(self, name, height=240):
        from pathlib import Path

        from sdr_module.ham.gui.sstv_panel import save_rgb_image

        path = Path(self.tmp.name) / name
        save_rgb_image(sstv_test_card(320, height), str(path))
        return path

    def open_panel(self):
        from sdr_module.ham.gui.sstv_panel import SSTVPanel

        panel = SSTVPanel()
        self.addCleanup(panel.deleteLater)
        panel.set_save_directory(self.tmp.name)
        panel.resize(360, 900)
        panel.show()
        settle()
        return panel

    def test_the_newest_saved_image_is_shown(self):
        self.write_image("sstv_20261029_101530_Martin_1.png", 256)
        self.write_image("sstv_20261029_111500_Robot_36_partial_180of240.png")
        panel = self.open_panel()
        self.assertEqual(panel._history_list.count(), 2)
        self.assertTrue(panel._image_display.has_image())
        self.assertEqual(panel._history_list.currentRow(), 1)
        self.assertEqual(panel._image_count_label.text(), "2 of 2")
        self.assertEqual(panel._mode_label.text(), "Robot 36")
        self.assertTrue(panel._save_btn.isEnabled())
        # Start Decoder: listening, the old picture stays until a new one.
        panel._start_btn.click()
        self.assertEqual(panel._status_label.text(), "Listening for an SSTV signal…")

    def test_no_saved_images_keeps_the_empty_state(self):
        panel = self.open_panel()
        self.assertFalse(panel._image_display.has_image())
        self.assertEqual(panel._image_count_label.text(), "No images")
        self.assertFalse(panel._save_btn.isEnabled())

    def test_an_unreadable_newest_file_stays_quiet(self):
        from pathlib import Path

        (Path(self.tmp.name) / "sstv_20261029_111500_Robot_36.png").write_bytes(
            b"not a png"
        )
        panel = self.open_panel()
        self.assertFalse(panel._image_display.has_image())
        self.assertEqual(panel._status_label.text(), "")


class TestRadioTunerWithoutADevice(_WindowTestCase):
    """HAM-E2E-13: with no device, the tuner's footer still said "Tunes the
    main receiver. Use the toolbar's Audio button to listen." and a preset
    said "Tuned to 101.100 MHz (FM)", while nothing could play."""

    demo = True

    def setUp(self):
        super().setUp()
        if not self.mw.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")

    def tearDown(self):
        tuner = getattr(self.win, "_radio_tuner", None) if HAS_PYQT6 else None
        if tuner is not None:
            tuner.close()
        super().tearDown()

    NO_RECEIVER = "No receiver: Device › Use Demo Device or Connect…"

    def test_footer_and_status_say_there_is_no_receiver(self):
        win = self.win
        win._disconnect_device()
        win._show_radio_tuner()
        tuner = win._radio_tuner
        footer = tuner._hint_label
        self.assertEqual(footer.text(), self.NO_RECEIVER)
        self.assertEqual(footer.property("tone"), "warning")
        tuner._preset_buttons[0].click()
        status = win._message_label.text()
        self.assertTrue(status.startswith("Tuned to "), status)
        self.assertIn("No receiver", status)
        self.assertEqual(win._message_label.property("tone"), "warning")
        tuner._hint_timer.stop()
        tuner._reset_hint()  # (after any feedback the footer showed)
        self.assertEqual(footer.text(), self.NO_RECEIVER)
        # A device again: the usual footer.
        win._start_demo_mode()
        settle()
        self.assertEqual(footer.text(), tuner._FOOTER_HINT)
        self.assertNotEqual(footer.property("tone"), "warning")

    def test_with_the_demo_device_the_usual_footer(self):
        self.win._show_radio_tuner()
        tuner = self.win._radio_tuner
        self.assertEqual(tuner._hint_label.text(), tuner._FOOTER_HINT)
        tuner._preset_buttons[0].click()
        self.assertNotIn("No receiver", self.win._message_label.text())


class TestHamIdTooltips(_WindowTestCase):
    """HAM-E2E-14: View > Panels > Ham ID (and the tab) said "Station
    identification with your callsign (CW ID, HackRF only)" though the ID
    is sent as an FM tone above 30 MHz and the tab holds the license class
    and TX power too."""

    def test_panel_tip_describes_the_tab(self):
        if not self.mw.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")
        tips = {key: tip for key, _title, tip in self.win._panel_specs}
        tip = tips["ham_id"]
        self.assertNotIn("CW ID, HackRF only", tip)
        for words in ("callsign", "license class", "CW or FM tone", "TX power"):
            self.assertIn(words, tip)
        ham = self.win._ham_tabs
        index = ham.indexOf(self.win._panel_pages["ham_id"])
        self.assertEqual(ham.tabToolTip(index), tip)


class TestSSTVImageEndingWithTheInput(_PanelCase):
    """L23: the GUI never called SSTVDecoder.finish(), so an image that
    ended with a played-back recording waited for audio after its last
    line and was kept as a partial image of N-1 lines."""

    def test_finish_input_completes_the_last_line(self):
        audio = encode_sstv(sstv_test_card(320, 240), "Robot 36", FS)
        self.panel._start_btn.click()
        self.feed(audio)  # (the recording ends with the image)
        self.assertTrue(self.panel._decoder.is_receiving(), "the last line waits")
        self.panel.finish_input()
        self.assertFalse(self.panel._decoder.is_receiving())
        texts = self.history_texts()
        self.assertEqual(len(texts), 1)
        self.assertNotIn("partial", texts[0])
        self.assertTrue(self.panel._status_label.text().startswith("Image received"))

    def test_finish_input_keeps_a_partial_image_and_says_why(self):
        audio = encode_sstv(sstv_test_card(320, 240), "Robot 36", FS)
        self.panel._start_btn.click()
        self.feed(audio[: len(audio) // 2])
        self.panel.finish_input()
        texts = self.history_texts()
        self.assertEqual(len(texts), 1)
        self.assertIn("partial", texts[0])
        self.assertIn("audio ended", self.panel._status_label.text())
        self.assertNotIn("faded", self.panel._status_label.text())

    def test_nothing_to_finish(self):
        self.panel.finish_input()  # not listening
        self.panel._start_btn.click()
        self.panel.finish_input()  # listening, no image
        self.assertEqual(self.history_texts(), [])


class TestSSTVFinishedAtTheEndOfPlayback(_WindowTestCase):
    """L23: the window tells the SSTV panel when a recording played to its
    end (not when the receiver is merely stopped)."""

    demo = True

    def setUp(self):
        super().setUp()
        if not self.mw.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")

    def test_end_of_a_recording_finishes_the_sstv_input(self):
        from unittest import mock

        from tests.test_gui_playback import _fm_tone, _save_sigmf

        folder = tempfile.mkdtemp(prefix="sdr-final-ham-")
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        panel = self.win._sstv_panel
        with mock.patch.object(panel, "finish_input") as finish:
            self.win._stop_acquisition()  # a live receiver stopped
            finish.assert_not_called()
            self.win._start_acquisition()
            path = _save_sigmf(f"{folder}/sstv.sigmf-data", _fm_tone(0.5))
            self.assertTrue(self.win._open_recording(path))
            device = self.win._device
            while not device.finished:
                self.win._process_block(device.read_samples(8000))
            self.win._playback._tick()  # (the playback's timer: the end)
            self.assertFalse(self.win._is_running)
            finish.assert_called_once()


class TestTunerSeekInARecording(_WindowTestCase):
    """L23: during playback, the tuner tuned first and switched to the
    band's mode after, so in a mode wider than the band's filter (I/Q at
    1 MHz, say) only the recording's centre channel fitted: seek counted
    no other station."""

    demo = True

    def setUp(self):
        super().setUp()
        if not self.mw.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")
        from tests.test_gui_playback import _fm_tone, _save_sigmf

        folder = tempfile.mkdtemp(prefix="sdr-final-ham-")
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        path = _save_sigmf(
            f"{folder}/fm.sigmf-data", _fm_tone(1.0, rate=1e6), rate=1e6, center=100.1e6
        )
        self.assertTrue(self.win._open_recording(path))
        self.win._set_demod_mode("None (I/Q)", bandwidth="1 MHz")

    def test_a_station_off_centre_can_be_tuned(self):
        win = self.win
        self.assertTrue(win._tuner_can_tune(100.3e6))
        self.assertFalse(win._tuner_can_tune(100.6e6), "outside the recording")
        win._on_radio_frequency_changed(100.3e6, "FM")
        self.assertFalse(win._tuner_refused)
        self.assertEqual(win._current_frequency(), 100.3e6)
        self.assertEqual(win._control_panel.get_demod_mode(), "FM")
        self.assertEqual(win._control_panel._bw_combo.currentText(), "200 kHz")

    def test_a_refused_station_keeps_the_mode(self):
        win = self.win
        win._on_radio_frequency_changed(100.6e6, "FM")  # outside the file
        self.assertTrue(win._tuner_refused)
        self.assertEqual(win._current_frequency(), 100.1e6)
        self.assertEqual(win._control_panel.get_demod_mode(), "None (I/Q)")


class TestCwOnEveryAmateurFrequency(_LicenseCase):
    """L25: the HF phone segments listed no CW, so a data or SSB
    transmission there couldn't carry its automatic CW ID. FCC 97.305(a):
    CW may be sent on any frequency the operator may transmit on."""

    PHONE = (1.9e6, 3.9e6, 7.2e6, 14.23e6, 18.15e6, 21.3e6, 24.95e6, 28.4e6)

    def test_cw_in_the_phone_segments(self):
        self.license("general")
        for freq in self.PHONE:
            allowed, reason = self.fm.is_tx_allowed(freq, 100.0, "CW")
            self.assertTrue(allowed, f"{freq / 1e6} MHz: {reason}")
        self.license("amateur_extra")
        allowed, reason = self.fm.is_tx_allowed(14.2e6, 100.0, "CW")
        self.assertTrue(allowed, reason)

    def test_only_where_the_license_reaches(self):
        self.license("technician")
        self.assertFalse(self.fm.is_tx_allowed(14.23e6, 100.0, "CW")[0])
        self.license("general")
        self.assertFalse(self.fm.is_tx_allowed(14.2e6, 100.0, "CW")[0])  # Extra
        self.license("none")
        self.assertFalse(self.fm.is_tx_allowed(14.23e6, 100.0, "CW")[0])

    def test_other_modes_keep_their_segments(self):
        self.license("general")
        self.assertFalse(self.fm.is_tx_allowed(14.07e6, 2.7e3, "USB")[0])
        self.assertFalse(self.fm.is_tx_allowed(50.05e6, 2.7e3, "USB")[0])
        self.assertFalse(self.fm.is_tx_allowed(7.2e6, 300.0, "FM")[0])
        self.assertEqual(self.fm.get_power_limit(10.12e6, "CW"), 200.0)


class TestDataJobCarriesItsCwId(_TxCase):
    """L25 through the transmit path: PSK31 on 14.230 MHz (General) with
    automatic ID was refused for its CW ID."""

    license = "general"

    def test_data_in_a_phone_segment_with_a_cw_id(self):
        from sdr_module.gui.transmitter import ArraySource, TxJob

        self.panel.set_callsign("W1AW")
        self.panel._id_end_check.setChecked(True)
        self.hackrf(pace=True, freq=14.23e6)
        rate = 2e6
        tone = np.exp(2j * np.pi * 1000.0 * np.arange(int(2 * rate)) / rate)
        job = TxJob(
            ArraySource(tone.astype(np.complex64)),
            rate,
            14.23e6,
            "DATA",
            100.0,
            "PSK31 “CQ DE W1AW”",
            "31.25 Bd",
            identify=True,
        )
        self.assertTrue(self.station.transmit(job, confirm=False), self.status()[0])
        self.station.stop()
        self.wait_done(5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()

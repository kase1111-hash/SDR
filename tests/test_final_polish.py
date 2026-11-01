"""Final polish: small cross-area problems left open by the post-merge
checks (P-01 to P-10 of the final pass).

Each test names the problem it covers; most run the real window offscreen
on the demo device (or the part on its own where that is enough).
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import require_pyqt6  # noqa: E402
from tests.test_gui_recording_files import _FilesCase  # noqa: E402
from tests.test_gui_signal_tab import _SignalCase  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signal_synth as S  # noqa: E402


def _save_sigmf(path, center, rate=1e6, seconds=0.2):
    from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

    rng = np.random.default_rng(7)
    n = int(rate * seconds)
    noise = 0.001 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    save_iq_file(
        path,
        noise.astype(np.complex64),
        rate,
        center,
        SampleFormat.FLOAT32,
        FileFormat.SIGMF,
    )
    return path


_DEVIATIONS_HZ = (2.5e3, 5e3, 12.5e3, 25e3, 75e3)


def _hz(text):
    """ "12.5 kHz" -> 12500.0"""
    value, unit = text.split()
    return float(value) * {"Hz": 1.0, "kHz": 1e3}[unit]


class TestIdentifiedDeviationDoesNotClip(unittest.TestCase):
    """P-01: Identify suggested "2.5 kHz" for the demo's 100.715 MHz call
    (+/-3.4 kHz) and "5 kHz" for its POCSAG pager (+/-4.5 kHz FSK); half a
    second after Apply Mode the window warned "FM audio is clipping". Here
    the window's own receiver chain measures what the warning measures (the
    share of audio samples beyond full scale, warned about above 5 %) at
    the deviation Identify suggests: it doesn't clip, and the next smaller
    choice would (the suggestion is no wider than it needs to be)."""

    RATE = 240e3
    SECONDS = 1.5

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import main_window

        self.mw = main_window

    def clip_fraction(self, x, deviation_hz, width_hz):
        """What the window's clipping warning measures, block by block (30
        blocks a second, as the display loop hands them over)."""
        chain = self.mw._ReceiverChain(self.RATE, "FM", width_hz)
        over = total = 0.0
        block = int(self.RATE / 30)
        for start in range(0, len(x) - block + 1, block):
            channel = chain.channelize(x[start : start + block])
            audio = chain.demodulate(channel, deviation_hz)
            over += chain.last_clip_fraction * len(audio)
            total += len(audio)
        return over / max(total, 1.0)

    def check(self, signal, occupied_hz, width_hz, snr_db=30.0):
        from sdr_module.dsp.signal_id import identify

        x, _ = S.add_noise(signal, self.RATE, snr_db, occupied_hz, seed=3)
        x = x.astype(np.complex64)
        ident = identify(
            x[: int(self.RATE)], self.RATE, 300e6, search_hz=0.75 * width_hz
        )
        self.assertEqual(ident.mode, "FM", ident.summary())
        deviation = _hz(ident.fm_deviation)
        limit = self.mw._CLIP_FRACTION
        clipped = self.clip_fraction(x, deviation, width_hz)
        self.assertLessEqual(clipped, limit, (ident.fm_deviation, clipped))
        smaller = [d for d in _DEVIATIONS_HZ if d < deviation]
        if smaller:
            self.assertGreater(
                self.clip_fraction(x, smaller[-1], width_hz),
                limit,
                f"{ident.fm_deviation} is wider than it needs to be",
            )
        return ident.fm_deviation

    def test_the_demo_calls_swing(self):
        """Two tones swinging +/-3.4 kHz together, as the demo's narrow FM
        calls do (2.2 kHz + 1.2 kHz)."""
        n = int(self.SECONDS * self.RATE)
        signal = S.fm(n, self.RATE, 3.4e3, S.tones(n, self.RATE))
        self.assertEqual(self.check(signal, 10e3, 12.5e3), "5 kHz")

    def test_a_swing_just_beyond_a_choice(self):
        n = int(self.SECONDS * self.RATE)
        signal = S.fm(n, self.RATE, 2.8e3, S.tones(n, self.RATE))
        self.assertEqual(self.check(signal, 9e3, 12.5e3), "5 kHz")

    def test_wide_voice(self):
        n = int(self.SECONDS * self.RATE)
        audio = S.voice(n, self.RATE, seed=7, band=(300.0, 2500.0))
        signal = S.fm(n, self.RATE, 12e3, audio)
        self.assertEqual(self.check(signal, 30e3, 50e3), "12.5 kHz")

    def test_pager_data(self):
        """FSK at +/-4.5 kHz, 1200 baud: POCSAG as the demo sends it."""
        n = int(self.SECONDS * self.RATE)
        signal = S.fsk(n, self.RATE, 1200, 4500)
        self.assertEqual(self.check(signal, 10.2e3, 25e3), "12.5 kHz")


class TestARecordingTooSlowForTheDecoder(_FilesCase):
    """P-05: with ADS-B chosen by hand, opening a 1 MS/s recording said "ADS-B
    needs at least 2 MS/s; the device runs at 1 MS/s. Choose 2.4 MS/s or
    more in Receiver > Sample rate (or click the rate in the status bar)",
    a rate a playing recording can't change. It says the recording's rate
    is too low, as Decode Recording does; the device keeps its advice."""

    def said(self):
        """Everywhere the decoder's problem shows: the status bar, the
        decoder chip's tooltip, the Decoder's status line and empty table."""
        win = self.win
        panel = win._decoder_panel
        return [
            win._message_label.text(),
            win._decoder_chip.toolTip(),
            panel.decoder_status_text(),
            panel._empty._text,
        ]

    def choose_ads_b(self):
        self.win._decoder_panel._proto_combo.setCurrentText("ADS-B")  # by hand
        self.settle(2)

    def open_file(self, name, rate):
        path = _save_sigmf(self.path(name), 1090e6, rate)
        with self.open_dialog(path):
            self.assertTrue(self.win._open_recording())
        self.settle(2)

    def test_a_recording_says_its_rate_is_too_low(self):
        win = self.win
        self.choose_ads_b()
        self.open_file("slow.sigmf-data", 1e6)
        self.assertTrue(win._decoder_panel.is_decoding())
        self.assertEqual(win._decoder_chip.text(), "ADS-B · can't decode")
        self.assertIn(
            "ADS-B needs at least 2 MS/s; this recording is 1 MS/s.",
            win._message_label.text(),
        )
        self.assertIn("this recording is 1 MS/s", win._decoder_chip.toolTip())
        self.assertIn("this recording is 1 MS/s", win._decoder_panel._empty._text)
        for text in self.said():
            self.assertNotIn("Sample rate", text)
            self.assertNotIn("the device", text)
        # Closed: the demo device (2.4 MS/s) decodes again.
        win._playback.close()
        self.settle(2)
        self.assertEqual(win._decoder_chip.text(), "ADS-B · 0 msgs")

    def test_the_device_and_a_recording_at_the_same_rate(self):
        """Each says it its own way, although the rate stays 1 MS/s."""
        win = self.win
        self.assertTrue(win.set_sample_rate(1e6))
        self.choose_ads_b()
        tip = win._decoder_chip.toolTip()
        self.assertIn("ADS-B needs at least 2 MS/s; the device runs at 1 MS/s", tip)
        self.assertIn("Receiver > Sample rate", tip)
        self.open_file("same.sigmf-data", 1e6)
        tip = win._decoder_chip.toolTip()
        self.assertIn("ADS-B needs at least 2 MS/s; this recording is 1 MS/s", tip)
        self.assertNotIn("Sample rate", tip)
        win._playback.close()
        self.settle(2)
        self.assertEqual(win._device_sample_rate(), 1e6)
        self.assertIn("the device runs at 1 MS/s", win._decoder_chip.toolTip())


class TestLongAndMediumWaveInKilohertz(_SignalCase):
    """P-02: below 2 MHz the window names a frequency in kHz (FREQ "880
    kHz", LISTEN-12), but Identify said "0.880 MHz: AM · confident · SNR
    58 dB" and "at 0.8800 MHz", and the Scanner "Bookmarked 0.660 MHz.".
    Sentences and cards name it as the window does; the Signals and
    results columns (headed MHz, sorted by number) stay in MHz."""

    def said(self):
        texts = []
        show = self.win._show_status_message

        def record(text, *args, **kwargs):
            texts.append(text)
            return show(text, *args, **kwargs)

        patcher = mock.patch.object(self.win, "_show_status_message", record)
        patcher.start()
        self.addCleanup(patcher.stop)
        return texts

    def test_identify_names_880_khz(self):
        said = self.said()
        self.listen(880e3, ("AM", None, "10 kHz"))
        self.assertTrue(self.tools.identify())
        self.assertEqual(self.panel.identify_status.text(), "Listening to 880 kHz…")
        ident = self.identified()
        self.assertEqual(ident.kind, "am", ident.summary())
        status = self.panel.identify_status.text()
        self.assertTrue(status.startswith("880 kHz at "), status)
        self.assertIn("· at 880 kHz", self.panel.listen_label.text())
        self.assertTrue(said[-1].startswith("880 kHz: AM · "), said)
        self.panel.apply_btn.click()
        self.assertTrue(said[-1].startswith("Listening to 880 kHz in AM"), said)
        texts = said + [status, self.panel.listen_label.text()]
        self.assertFalse([t for t in texts if "0.880" in t], texts)

    def test_formatters(self):
        from sdr_module.gui.signal_panel import format_mhz, format_signal_mhz

        self.assertEqual(format_mhz(880e3), "880 kHz")
        self.assertEqual(format_mhz(1260.5e3), "1260.5 kHz")
        self.assertEqual(format_mhz(433.92e6), "433.920 MHz")
        self.assertEqual(format_mhz(146.5125e6), "146.5125 MHz")
        # A wide signal to the kHz, a narrow one to 100 Hz, as above 2 MHz.
        self.assertEqual(format_signal_mhz(880.2e3, 9e3), "880 kHz")
        self.assertEqual(format_signal_mhz(350.43e3, 500.0), "350.4 kHz")
        self.assertEqual(format_signal_mhz(880.02e3, 4e3), "880 kHz")
        self.assertEqual(format_signal_mhz(99.82e6, 500.0), "99.8200 MHz")
        # The Signals list's column is headed MHz.
        self.assertEqual(format_signal_mhz(880e3, 9e3, unit=False), "0.880")

    def test_the_scanner_says_660_khz(self):
        from sdr_module.gui import scanner_panel as sp

        win = self.win
        panel = win._scanner_panel
        panel._hits = [sp.ScanHit(660e3, -40.0, 40.0, 9e3, "AM", "", "10 kHz")]
        panel._rebuild_table()
        panel._table.selectRow(0)
        # (The results column is numeric, headed MHz.)
        self.assertEqual(panel._table.item(0, sp.COL_FREQ).text(), "0.660 MHz")
        panel.bookmark_selected()
        # (Named as Ctrl+B names it, W-04: in kHz too.)
        self.assertEqual(
            panel._status.text(),
            "Bookmarked AM Broadcast 660 kHz (AM). Rename it with F2 in the "
            "Bookmarks tab.",
        )
        panel.listen_selected()
        self.assertEqual(panel._status.text(), "Listening to 660 kHz (AM).")
        self.assertEqual(win._current_frequency(), 660e3)
        self.assertIn("660 kHz (AM Broadcast)", panel._hit_tooltip(panel._hits[0]))


class TestTheScanChipGivesWayAt1024Px(_SignalCase):
    """P-03: the decoder chip gives way at 1024 px (I-decode), the scan
    chip didn't: a held bookmark scan with a long name ("Scanning · NOAA
    Weather Radio 162.… · held") with AUDIO and Track Signal's AFC needed
    1,042-1,071 px. It is cut as the decoder chip is (whole in its
    tooltip), not the window widened. (A scan is refused while I/Q
    records: no REC here.)"""

    HELD = "Scanning · NOAA Weather Radio 162.… · held"
    DECODER = (
        "ADS-B · 6 msgs",
        "AX.25/APRS · 12,345 msgs",
        "CW (Morse) · 15 WPM · 12,345 msgs",
    )

    def test_the_window_keeps_1024_px(self):
        from tests.test_gui_coherent_window import _ThemedSizes

        _ThemedSizes.each_theme(self, self.check_room)

    def check_room(self, theme):
        from PyQt6.QtWidgets import QLabel

        from sdr_module.gui.themes import set_role

        win = self.win
        win.resize(1024, 640)
        win._audio_rec_label.setText("AUDIO 00:05")
        win._audio_rec_label.setVisible(True)
        afc = QLabel("AFC +2.2 kHz")  # (as Track Signal shows it)
        set_role(afc, "badge", "info")
        win.statusBar().addPermanentWidget(afc)
        self.addCleanup(afc.deleteLater)
        scan = win._scanner_panel.status_chip()
        scan.setText(self.HELD)
        scan.setVisible(True)
        chip = win._decoder_chip
        for text in self.DECODER:
            chip.setText(text)
            chip.setVisible(True)
            self.settle(10)
            where = (theme, text)
            self.assertLessEqual(win.minimumSizeHint().width(), 1024, where)
            self.assertEqual(win.width(), 1024, where)
            self.assertGreaterEqual(win._message_label.width(), 220, where)
            self.assertEqual(scan.text(), self.HELD)  # (whole: only drawn short)
            self.assertGreater(scan.width(), 60, where)  # ("Scanning · …")
        win.statusBar().removeWidget(afc)
        # With room, the scan chip shows its whole text.
        chip.setVisible(False)
        win.resize(1366, 768)
        self.settle(10)
        self.assertGreaterEqual(scan.width(), scan.sizeHint().width(), theme)
        scan.setVisible(False)
        win._audio_rec_label.setVisible(False)


def _hardware_case():
    from tests.test_review_mainloop import _HardwareCase

    return _HardwareCase


class TestTheAppsOwnJobIsNamedNotTheDevice(_hardware_case()):
    """P-04: during a busy Decode Recording (and for a few seconds after
    it) the Decoder's line and the status bar said "The demo device can't
    keep up in real time right now, so the ADS-B decoder misses part of
    the signal...": the effect is true, the cause it named is not. The
    display loop runs on during the job (here 1.5 times real time); the
    hints name the job instead."""

    JOB = (
        "Decode Recording keeps this computer busy, so the ADS-B decoder "
        "misses part of the signal meanwhile."
    )
    # (Once it is done: W-10.)
    DONE = (
        "Decode Recording kept this computer busy, so the ADS-B decoder "
        "missed part of the signal meanwhile."
    )

    def setUp(self):
        super().setUp()
        self.win._demo_mode = True  # (the demo device's wording)
        self.win._set_demod_mode("None (I/Q)", bandwidth="2 MHz")
        self.win._decoder_panel._proto_combo.setCurrentText("ADS-B")
        self.settle(1)
        self.win._decoder_panel.set_receiving(True)

    def line(self):
        return self.win._decoder_panel.decoder_status_text()

    def during_a_job(self, frames):
        """Run ``frames`` display frames while Decode Recording's job runs
        (run_job's own event loop), and what was said meanwhile."""
        import threading

        from PyQt6.QtCore import QTimer

        from sdr_module.gui.file_jobs import run_job

        done = threading.Event()
        said = {}

        def frames_meanwhile():
            try:
                self.run_frames(frames)
                said["line"] = self.line()
                said["bar"] = self.win._message_label.text()
                said["audio"] = self.win._realtime_hint()
            finally:
                done.set()

        QTimer.singleShot(0, frames_meanwhile)
        result = run_job(
            self.win, "Decode Recording", "Decoding…", lambda report: done.wait(20)
        )
        self.assertTrue(result.ok, result.error)
        return said

    def test_the_hints_name_the_job(self):
        said = self.during_a_job(40)
        self.assertTrue(self.win._pacer.overloaded)
        self.assertEqual(said["line"], self.JOB)
        self.assertEqual(said["bar"], self.JOB)  # (the status bar says it once)
        self.assertEqual(
            said["audio"],
            "Decode Recording keeps this computer busy, so audio is paused meanwhile.",
        )
        # Done: the computer keeps up again. What it skipped until then
        # was still the job's doing (never the device's), then nothing.
        self.LOAD = 0.3
        lines = []
        for _ in range(300):
            self.run_frames(1)
            if not lines or lines[-1] != self.line():
                lines.append(self.line())
        self.assertFalse(self.win._pacer.overloaded)
        for line in lines:
            self.assertNotIn("device", line)
        self.assertIn(self.DONE, lines)
        self.assertNotIn("misses", lines[-1])
        self.assertNotIn("missed", lines[-1])

    def test_without_a_job_the_device_is_still_named(self):
        self.run_frames(40)
        self.assertTrue(self.line().startswith("The demo device can't keep up"))
        self.assertTrue(
            self.win._realtime_hint().startswith("The demo device can't keep up")
        )


def _demo_take_case():
    from tests.test_final_record import _DemoTakeCase

    return _DemoTakeCase


class TestTheJobIsNamedUntilTheLoopCatchesUp(_demo_take_case()):
    """P-04, on the demo device with the display loop at ten frames a second
    (a busy computer): the job was named for the 2 s real time is judged
    over, counted from its end. But the pacer finds out that frames keep up
    again only at its next probe, up to a second after the job, and the
    one-block frames before that still count for 2 s: what was skipped was
    still measured up to 2 s after the job. After some jobs the Decoder's
    line then said "The demo device can't keep up ... A lower sample rate
    (Receiver > Sample rate) helps." for 3 s."""

    FRAME_S = 0.1
    JOB = (
        "Decode Recording keeps this computer busy, so the POCSAG decoder "
        "misses part of the signal meanwhile."
    )
    # (Once it is done: W-10.)
    DONE = (
        "Decode Recording kept this computer busy, so the POCSAG decoder "
        "missed part of the signal meanwhile."
    )

    def setUp(self):
        super().setUp()
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("POCSAG")
        self.run_frames(0.1)
        panel.set_receiving(True)

    def run_frames(self, seconds, stall_at=None, stall=0.0):
        end = self.clock.now + seconds
        while self.clock.now < end:
            tick = self.clock.now
            self.win._update_display()
            self.clock.now = max(self.clock.now, tick + self.FRAME_S)

    def line(self):
        return self.win._decoder_panel.decoder_status_text()

    def a_busy_job(self, seconds):
        """Decode Recording's job (run_job's own event loop), with
        ``seconds`` of display frames meanwhile."""
        import threading

        from PyQt6.QtCore import QTimer

        from sdr_module.gui.file_jobs import run_job

        done = threading.Event()

        def frames_meanwhile():
            try:
                self.run_frames(seconds)
            finally:
                done.set()

        QTimer.singleShot(0, frames_meanwhile)
        result = run_job(
            self.win, "Decode Recording", "Decoding…", lambda report: done.wait(20)
        )
        self.assertTrue(result.ok, result.error)

    def test_what_the_job_skipped_never_names_the_device(self):
        win = self.win
        # (Each job a frame longer, from a pacer that starts over: the job
        # ends at each moment of its probes' period, a little over 1 s.)
        for k in range(12):
            seconds = 1.0 + k * self.FRAME_S
            win._pacer.reset()
            win._rx_history.clear()
            self.load = 1.5  # (can't keep up, before and during the job)
            self.run_frames(1.0)
            self.assertTrue(win._pacer.overloaded)
            self.a_busy_job(seconds)
            self.load = 0.3  # (done: the computer keeps up again)
            lines = []
            for _ in range(35):
                self.run_frames(self.FRAME_S)
                lines.append(self.line())
            where = (seconds, sorted(set(lines)))
            self.assertIn(self.DONE, lines, where)
            self.assertFalse([t for t in lines if "device" in t], where)
        self.run_frames(3.0)
        self.assertNotIn("misses", self.line())  # (then nothing)
        self.assertNotIn("missed", self.line())

    def test_a_computer_still_too_slow_afterwards_names_the_device(self):
        """...but not for good: still too slow once the job is done, the
        device's wording comes back."""
        self.load = 1.5
        self.run_frames(1.0)
        self.a_busy_job(1.0)
        lines = [self.line()]
        for _ in range(60):
            self.run_frames(self.FRAME_S)
            lines.append(self.line())
        self.assertEqual(lines[0], self.JOB)
        self.assertTrue(lines[-1].startswith("The demo device can't keep up"), lines)


def _tx_case():
    from tests.test_gui_station_tx import _TxCase

    return _TxCase


class TestOnAirComesBackAfterANotice(_tx_case()):
    """P-06: "On air: ... Stop ends it." was posted once at the start of a
    transmission. A notice that replaced it (Space refused, Open Recording
    refused while transmitting) left the status bar empty once it expired,
    for the rest of the transmission. It comes back."""

    def test_the_on_air_text_comes_back(self):
        from PyQt6.QtWidgets import QApplication

        win = self.win
        self.hackrf(pace=True)
        self.panel.set_callsign("W1AW")
        self.panel._wpm_spin.setValue(5)  # (a long ID: about 10 s)
        self.station.send_station_id()
        self.assertTrue(self.station.busy)
        QApplication.processEvents()
        on_air = win._message_label.text()
        self.assertTrue(on_air.startswith("On air: "), on_air)
        win._start_acquisition()  # refused: a notice replaces it
        self.assertIn("Can't receive while transmitting", win._message_label.text())
        # The notice expires (its timer's slot)...
        win._message_timer.stop()
        win._clear_status_message()
        self.assertEqual(win._message_label.text(), "")
        # ...and with the transmission's next progress, "On air" is back.
        self.assertTrue(self.pump(2.0, until=lambda: win._message_label.text()))
        self.assertEqual(self.status(), (on_air, "danger"))
        history = [e.text for e in win._status_history.entries()]
        self.assertEqual(history.count(on_air), 1, history)  # (said once)
        # A notice still shown is left alone.
        win._show_status_message("Something to read", "info", 60000)
        self.pump(0.5)
        self.assertEqual(win._message_label.text(), "Something to read")
        self.station.stop()
        self.wait_done(5)
        QApplication.processEvents()
        self.assertFalse(win._message_label.text().startswith("On air"))


class TestThePresetsDecoderNoteStays(_FilesCase):
    """P-07: typing a frequency in another band right after a preset turned
    its decoder on (or opening a recording) said "ACARS decoder off: tuned
    away from the ACARS preset (131.550 MHz)..." and at once replaced it
    with the band plan's "FM Broadcast: listening in FM...". The note was
    left only in the status history. One message says both now."""

    def type_frequency(self, text):
        from tests.test_final_listen import TestBandPlanListeningSegments

        TestBandPlanListeningSegments.type_frequency(self, text)

    def decoding(self):
        panel = self.win._decoder_panel
        return panel._proto_combo.currentText(), panel._enabled_check.isChecked()

    def test_a_typed_tune_into_another_band(self):
        win = self.win
        self.assertTrue(win._apply_preset_named("ACARS"))
        self.assertEqual(self.decoding(), ("ACARS", True))
        self.type_frequency("100.1")
        self.assertEqual(self.decoding(), ("ACARS", False))
        text = self.message()
        self.assertTrue(
            text.startswith("ACARS decoder off · FM Broadcast: listening in FM"), text
        )
        # Its band's line is still the one cleared when you leave the band
        # (LISTEN-13), note and all; the next tune says only its own.
        self.type_frequency("146.52")
        text = self.message()
        self.assertNotIn("FM Broadcast", text)
        self.assertTrue(text.startswith("2m Calling: listening in FM"), text)

    def test_a_bookmark_without_a_mode(self):
        win = self.win
        self.assertTrue(win._apply_preset_named("ADS-B"))
        win._on_bookmark_tune(100.1e6, "Local FM")
        text = self.message()
        self.assertTrue(
            text.startswith("ADS-B decoder off · FM Broadcast: listening in FM"), text
        )

    def test_a_bookmark_with_its_own_mode(self):
        win = self.win
        self.assertTrue(win._apply_preset_named("ADS-B"))
        win._on_bookmark_tune(146.52e6, "2m Calling", "FM", "5 kHz")
        self.assertEqual(self.message(), "ADS-B decoder off · Tuned to 2m Calling (FM)")
        # Once: the next tune went nowhere near the decoder, and says so.
        win._on_bookmark_tune(162.55e6, "NOAA Weather", "FM", "5 kHz")
        self.assertEqual(self.message(), "Tuned to NOAA Weather (FM)")

    def test_opening_a_recording(self):
        win = self.win
        self.assertTrue(win._apply_preset_named("ADS-B"))
        path = _save_sigmf(self.path("aprs2m.sigmf-data"), 144.39e6)
        with self.open_dialog(path):
            self.assertTrue(win._open_recording())
        text = self.message()
        # (Led by the file it plays: "Playing aprs2m.sigmf-data (...) · ".)
        self.assertTrue(text.startswith("Playing aprs2m.sigmf-data ("), text)
        self.assertIn(" · ADS-B decoder off · APRS: listening in FM (narrow", text)

    def test_a_tune_that_says_nothing_else_keeps_the_note(self):
        win = self.win
        self.assertTrue(win._apply_preset_named("ACARS"))
        win.set_frequency(125e6)  # (no band plan line: the note stands)
        self.assertTrue(self.message().startswith("ACARS decoder off: tuned away"))


def _scan_window_case():
    from tests.test_gui_scanner import _ScanWindowCase

    return _ScanWindowCase


class TestATuneDuringAScanSaysBoth(_scan_window_case()):
    """P-08: with the Scanner tab out of sight, a preset or a bookmark tuned
    from another tab said its message ("ADS-B: 1090.000 MHz ... · ADS-B
    decoder on", the band plan's line), and the next frame replaced it with
    "Scan stopped: you tuned to 1090.000 MHz.". One message says both."""

    def setUp(self):
        super().setUp()
        win, panel = self.win, self.win._scanner_panel
        panel._confirm_long_scan = lambda *_args: True
        panel._preset.setCurrentIndex(1)
        panel.set_scan_mode("continuous")
        self.assertTrue(panel.start_scan())
        win._show_panel("info")  # (the Scanner tab out of sight)
        self.frames(0.2)
        self.assertTrue(panel.engine.active)

    def history(self):
        return [entry.text for entry in self.win._status_history.entries()]

    def test_a_preset(self):
        win = self.win
        self.assertTrue(win._apply_preset_named("ADS-B"))
        said = win._message_label.text()
        self.assertIn("ADS-B decoder on", said)
        self.frames(0.1)
        self.assertFalse(win._scanner_panel.engine.active)
        self.assertEqual(win._message_label.text(), f"Scan stopped · {said}")
        # (Each in the status history once, as it was said.)
        history = self.history()
        self.assertIn("Scan stopped: you tuned to 1090.000 MHz.", history)
        self.assertIn(said, history)
        self.assertNotIn(f"Scan stopped · {said}", history)

    def test_a_bookmark(self):
        win = self.win
        win._on_bookmark_tune(100.1e6, "Local FM")
        said = win._message_label.text()
        self.assertTrue(said)
        self.frames(0.1)
        self.assertFalse(win._scanner_panel.engine.active)
        self.assertEqual(win._message_label.text(), f"Scan stopped · {said}")

    def test_a_tune_that_says_nothing_says_the_scan_stopped(self):
        win = self.win
        win.set_frequency(144.39e6)  # (as a click on the plots: no message)
        self.frames(0.1)
        self.assertEqual(
            win._message_label.text(), "Scan stopped: you tuned to 144.390 MHz."
        )


if __name__ == "__main__":
    unittest.main()

"""Final walk: what the release check's real-window walk found (W-01 to
W-13 of the final pass).

Each test names the problem it covers; most run the real window offscreen
on the demo device (or the part on its own where that is enough).
"""

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import require_pyqt6  # noqa: E402


def _window_case():
    from tests.test_gui_chrome import _WindowTestCase

    return _WindowTestCase


# --------------------------------------------------------------------------- #
# W-01: pagers heard at a deviation that holds them
# --------------------------------------------------------------------------- #


class TestPagersAreHeardAtADeviationThatHoldsThem(unittest.TestCase):
    """W-01: the POCSAG and FLEX presets, the band plan (152.240 or
    931.9375 MHz typed) and the Decoder tab's "Tune to" listened in FM at 5
    kHz deviation. Pagers send FSK at +/-4.5 kHz (POCSAG) and +/-4.8 kHz
    (FLEX), as the demo's do: at 5 kHz their audio clipped and within
    seconds the window warned "FM audio is clipping" (and logged it), while
    Identify said 12.5 kHz. They are heard at 12.5 kHz now."""

    PAGING = ("POCSAG", "FLEX", "POCSAG/FLEX")

    def test_the_presets(self):
        from sdr_module.core.band_plan import preset_listening
        from sdr_module.core.frequency_manager import get_frequency_manager

        for name in self.PAGING:
            preset = get_frequency_manager().get_preset_by_name(name)
            self.assertEqual(
                preset_listening(preset), ("FM", "12.5 kHz", "25 kHz"), name
            )

    def test_the_band_plan(self):
        from sdr_module.core.band_plan import listening_for

        # The presets' channels, and another channel of the 900 MHz band.
        for freq in (152.24e6, 931.9375e6, 929.6e6, 930.5125e6):
            self.assertEqual(listening_for(freq), ("FM", "12.5 kHz", "25 kHz"), freq)

    def test_other_narrow_fm_presets_keep_5_khz(self):
        from sdr_module.core.band_plan import listening_for

        for freq in (146.52e6, 144.39e6, 162.55e6, 156.8e6, 145.8e6):
            self.assertEqual(listening_for(freq)[1], "5 kHz", freq)

    def test_the_decoder_tabs_suggestion(self):
        from sdr_module.gui.decoder_panel import PROTOCOL_TUNE

        self.assertEqual(PROTOCOL_TUNE["POCSAG"][1:3], ("FM", "12.5 kHz"))
        self.assertEqual(PROTOCOL_TUNE["FLEX"][1:3], ("FM", "12.5 kHz"))


class TestTheDemoPagersDontClipAsTheyAreHeard(unittest.TestCase):
    """W-01: the demo device's pagers through the window's own receiver
    chain, block by block as the display loop hands them over, counted as
    the clipping warning counts (a clear signal, more than 5 % of its audio
    beyond full scale; it warns after half a second of that). At the
    deviation their preset listens with they don't clip; at 5 kHz they did
    (the control: the warning came)."""

    RATE = 1.024e6

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import main_window

        self.mw = main_window

    def clipping_seconds(self, freq, start_s, seconds, deviation_hz):
        from sdr_module.gui.device_dialog import MockDevice

        def demo(at_hz, position_s):
            dev = MockDevice()
            dev.set_sample_rate(self.RATE)
            dev.set_frequency(at_hz)
            dev.start_rx()
            dev._stream_rate = self.RATE
            dev._stream_pos = int(position_s * self.RATE)
            return dev

        block = int(self.RATE / 30)
        quiet = demo(300e6, 0.0)  # (the noise the channel has without a signal)
        chain = self.mw._ReceiverChain(self.RATE, "FM", 25e3)
        noise_db = float(
            np.median(
                [
                    10
                    * np.log10(
                        np.mean(
                            np.abs(chain.channelize(quiet.read_samples(block))) ** 2
                        )
                    )
                    for _ in range(10)
                ]
            )
        )
        dev = demo(freq, start_s)
        chain = self.mw._ReceiverChain(self.RATE, "FM", 25e3)
        clipping = heard = 0.0
        for _ in range(int(seconds * 30)):
            channel = chain.channelize(dev.read_samples(block))
            level_db = 10 * np.log10(np.mean(np.abs(channel) ** 2) + 1e-20)
            chain.demodulate(channel, deviation_hz)
            if level_db - noise_db >= self.mw._CLIP_MIN_SNR_DB:
                heard += 1 / 30
                if chain.last_clip_fraction > self.mw._CLIP_FRACTION:
                    clipping += 1 / 30
        self.assertGreater(heard, 1.0)  # (the page was sent meanwhile)
        return clipping

    def deviation_hz(self, freq):
        from sdr_module.core.band_plan import listening_for

        value, unit = listening_for(freq)[1].split()
        self.assertEqual(unit, "kHz")
        return float(value) * 1e3

    def test_pocsag(self):
        # (The demo pages at 3 s and 13 s of every 20.)
        freq = 152.24e6
        limit = self.mw._CLIP_WARN_AFTER_S
        held = self.clipping_seconds(freq, 2.9, 2.0, self.deviation_hz(freq))
        self.assertLess(held, limit)
        self.assertGreater(self.clipping_seconds(freq, 2.9, 2.0, 5e3), limit)

    def test_flex(self):
        # (Its frames at 5 s and 16 s of every 22.)
        freq = 931.9375e6
        limit = self.mw._CLIP_WARN_AFTER_S
        held = self.clipping_seconds(freq, 4.9, 2.2, self.deviation_hz(freq))
        self.assertLess(held, limit)
        self.assertGreater(self.clipping_seconds(freq, 4.9, 2.2, 5e3), limit)


class TestPagingPresetsInTheWindow(_window_case()):
    """W-01 in the window: Radio > Band Presets > Paging, a typed pager
    frequency and the Decoder tab's "Tune to" all listen at 12.5 kHz."""

    demo = True

    def deviation(self):
        return self.win._control_panel.get_fm_deviation_text()

    def test_the_presets(self):
        for name in ("POCSAG", "FLEX"):
            self.assertTrue(self.win._apply_preset_named(name))
            self.assertEqual(self.deviation(), "12.5 kHz", name)
            self.assertEqual(self.win._control_panel._bw_combo.currentText(), "25 kHz")

    def test_typed(self):
        from tests.test_final_listen import TestBandPlanListeningSegments

        win = self.win
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        self.assertTrue(win.set_frequency(100.1e6))
        for text, freq in (("152.24", 152.24e6), ("931.9375", 931.9375e6)):
            win._set_demod_mode("FM", "75 kHz", "200 kHz")
            TestBandPlanListeningSegments.type_frequency(self, text)
            self.assertEqual(win._current_frequency(), freq)
            self.assertEqual(self.deviation(), "12.5 kHz", text)
            self.assertIn("12.5 kHz deviation", win._message_label.text())

    def test_the_decoder_tabs_tune_to(self):
        win, panel = self.win, self.win._decoder_panel
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        panel._proto_combo.setCurrentText("POCSAG")
        panel._tune_to_protocol()
        self.assertEqual(win._current_frequency(), 152.24e6)
        self.assertEqual(self.deviation(), "12.5 kHz")


class TestTheClippingWarningNamesADeviationThatHolds(_window_case()):
    """W-01: the warning advised "75 kHz (broadcast) for an FM radio
    station" whatever was clipping, a pager too. It names the deviation
    that holds what it measured: the next choice for a pager heard at 5
    kHz, 75 kHz (broadcast) for a broadcast station."""

    demo = True

    def warn(self, freq, deviation, width, start_s, seconds):
        from sdr_module.gui import main_window
        from sdr_module.gui.device_dialog import MockDevice

        win = self.win
        win._set_demod_mode("FM", deviation, width)
        rate = 1.024e6
        dev = MockDevice()
        dev.set_sample_rate(rate)
        dev.set_frequency(freq)
        dev.start_rx()
        dev._stream_rate = rate
        dev._stream_pos = int(start_s * rate)
        chain = main_window._ReceiverChain(rate, "FM", float(width.split()[0]) * 1e3)
        said = []
        win._show_status_message = lambda text, *a, **k: said.append(text)
        win._clip_warn_due = 0.0
        block = int(rate / 30)
        for _ in range(int(seconds * 30)):
            channel = chain.channelize(dev.read_samples(block))
            chain.demodulate(channel, win._control_panel.get_fm_deviation())
            win._check_fm_clipping(chain, 1 / 30, 30.0)
            if said:
                break
        return said

    def test_a_pager_heard_at_5_khz(self):
        said = self.warn(152.24e6, "5 kHz", "25 kHz", 2.9, 2.0)
        self.assertEqual(
            said,
            [
                "FM audio is clipping: this station deviates more than 5 kHz. "
                "Choose a larger FM deviation in Demodulation: 12.5 kHz holds it."
            ],
        )

    def test_a_broadcast_station_heard_narrow(self):
        said = self.warn(100.1e6, "5 kHz", "200 kHz", 0.0, 1.5)
        self.assertEqual(
            said,
            [
                "FM audio is clipping: this station deviates more than 5 kHz. "
                "Choose a larger FM deviation in Demodulation, such as 75 kHz "
                "(broadcast) for an FM radio station."
            ],
        )


# --------------------------------------------------------------------------- #
# W-04: one wording for a new bookmark
# --------------------------------------------------------------------------- #


class TestTheScannersBookmarkSaysWhatCtrlBSays(_window_case()):
    """W-04: Ctrl+B and the Signals list's Bookmark said "Bookmarked AM
    Broadcast 660 kHz (AM). Rename it with F2 in the Bookmarks tab.", the
    Scanner's Bookmark only "Bookmarked 660 kHz." for the same bookmark."""

    demo = True

    def bookmark_result(self, hit):
        panel = self.win._scanner_panel
        panel._hits = [hit]
        panel._rebuild_table()
        panel._table.selectRow(0)
        panel.bookmark_selected()
        return panel._status.text()

    def test_a_result(self):
        from sdr_module.gui import scanner_panel as sp

        said = self.bookmark_result(
            sp.ScanHit(660e3, -40.0, 40.0, 9e3, "AM", "", "10 kHz")
        )
        want = "Bookmarked AM Broadcast 660 kHz (AM). Rename it with F2 in the Bookmarks tab."
        self.assertEqual(said, want)
        self.assertEqual(self.win._message_label.text(), want)
        names = [b.get("label") for b in self.win._bookmarks_panel.bookmarks()]
        self.assertIn("AM Broadcast 660 kHz", names)

    def test_as_ctrl_b_says_it(self):
        from sdr_module.gui import scanner_panel as sp

        win = self.win
        win._set_demod_mode("AM", None, "10 kHz")
        self.assertTrue(win.set_frequency(880e3))
        win._bookmark_current_frequency()
        by_key = win._message_label.text()
        said = self.bookmark_result(
            sp.ScanHit(1260e3, -40.0, 40.0, 9e3, "AM", "", "10 kHz")
        )
        self.assertEqual(by_key.replace("880", "1260"), said)

    def test_a_bookmarks_own_name_and_iq(self):
        from sdr_module.gui import scanner_panel as sp

        said = self.bookmark_result(
            sp.ScanHit(
                433.92e6,
                -40.0,
                30.0,
                20e3,
                "None (I/Q)",
                "",
                "500 kHz",
                label="Garage door",
            )
        )
        self.assertEqual(
            said,
            "Bookmarked Garage door (I/Q). Rename it with F2 in the Bookmarks tab.",
        )


# --------------------------------------------------------------------------- #
# W-05: a long channel name in the scan chip and the listen label
# --------------------------------------------------------------------------- #


def _panel_case():
    from tests.test_gui_scanner import _PanelCase

    return _PanelCase


class TestALongNameWhileScanning(_panel_case()):
    """W-05: held on a bookmark named "NOAA Weather Radio Seattle WX1
    162.55", the scan chip cut the name to 23 characters itself, so its
    tooltip (which starts with the chip's whole text when the status bar
    is short of room) never held it either; the Scanner tab's listen label
    was clipped at the Skip button without "…", hiding the frequency, the
    mode and "(held)"."""

    NAME = "NOAA Weather Radio Seattle WX1 162.55"

    def held_on_the_long_name(self):
        from sdr_module.gui.scan_engine import LISTEN
        from tests.test_gui_scanner import FRAME_S, FakeWindow

        host = FakeWindow()
        host.scan_signal_present = lambda: True
        visit = {"demod": "FM", "fm_deviation": "5 kHz", "bandwidth": ""}
        host.bookmarks = [
            dict(visit, index=0, label=self.NAME, freq_hz=162.55e6, tone_hz=0.0)
        ]
        panel = self.panel(host)
        panel.set_scan_mode("bookmarks")
        self.assertTrue(panel.start_scan())
        dev = host.dev
        panel.engine.frame(dev.read_samples(int(dev.sample_rate * FRAME_S)))
        self.assertEqual(panel.engine.state, LISTEN)
        panel._hold_btn.click()
        self.assertTrue(panel.engine.held)
        return panel

    def test_the_chip_keeps_the_whole_name(self):
        panel = self.held_on_the_long_name()
        chip = panel.status_chip()
        self.assertEqual(chip.text(), f"Scanning · {self.NAME} · held")
        # (Short of room, it is cut to its first part, and its tooltip then
        # starts with the whole text: it doesn't ask for more room.)
        self.assertLess(
            chip.minimumSizeHint().width(),
            chip.fontMetrics().horizontalAdvance("Scanning · NOAA Weather"),
        )

    def test_the_listen_label_ends_in_an_ellipsis(self):
        from tests.test_gui_scanner import _settle

        panel = self.held_on_the_long_name()
        label = panel._listen_label
        whole = f"Listening: {self.NAME} · 162.550 MHz · FM (held)"
        self.assertEqual(label.text(), whole)
        self.assertEqual(label.toolTip(), whole)
        panel.resize(300, 560)
        _settle()
        shown = label.displayed_text()
        self.assertTrue(shown.endswith("…"), shown)
        self.assertLessEqual(
            label.fontMetrics().horizontalAdvance(shown), label.width()
        )


# --------------------------------------------------------------------------- #
# W-06 and W-07: a preset's decoder, turned off as the tuning left it
# --------------------------------------------------------------------------- #


class TestAPresetsDecoderOffLeavesNoChip(_window_case()):
    """W-06: Band Presets > Aviation > ACARS, then a tune elsewhere: the
    message said "ACARS decoder off · ...", while the status bar's chip
    said "ACARS · paused" and stayed there from then on."""

    demo = True

    def released(self):
        win = self.win
        self.assertTrue(win._apply_preset_named("ACARS"))
        self.assertFalse(win._decoder_chip.isHidden())
        self.assertTrue(win.set_frequency(100.1e6))
        self.assertTrue(win._message_label.text().startswith("ACARS decoder off"))
        return win._decoder_chip

    def test_tuned_away_from_the_preset(self):
        chip = self.released()
        self.assertTrue(chip.isHidden(), chip.text())
        self.win._update_decoder_chip()  # (as the display loop refreshes it)
        self.assertTrue(chip.isHidden(), chip.text())

    def test_ticking_decode_again_brings_it_back(self):
        chip = self.released()
        self.win._decoder_panel._enabled_check.click()
        self.assertFalse(chip.isHidden())
        self.assertEqual(chip.text(), "ACARS · 0 msgs")

    def test_a_pause_of_your_own_keeps_its_chip(self):
        win, panel = self.win, self.win._decoder_panel
        panel._proto_combo.setCurrentText("POCSAG")
        panel._enabled_check.click()  # untick, as a user does
        self.assertFalse(win._decoder_chip.isHidden())
        self.assertEqual(win._decoder_chip.text(), "POCSAG · paused")


def _files_case():
    from tests.test_gui_recording_files import _FilesCase

    return _FilesCase


class TestCloseRecordingBringsThePresetsDecoderBack(_files_case()):
    """W-07: ACARS preset, then File > Open Recording of an FM capture
    ("ACARS decoder off · FM Broadcast: ..."), then File > Close
    Recording: back on 131.550 MHz in AM with the preset's 15 kHz filter,
    but its decoder stayed paused ("ACARS · paused", "Paused: tick Decode
    to resume"). Going back now brings it back, and says so."""

    def decoding(self):
        panel = self.win._decoder_panel
        return panel.selection(), panel._enabled_check.isChecked()

    def open_fm_capture(self):
        from tests.test_final_polish import _save_sigmf

        win = self.win
        self.assertTrue(win._apply_preset_named("ACARS"))
        path = _save_sigmf(self.path("fm8.sigmf-data"), 100.1e6)
        with self.open_dialog(path):
            self.assertTrue(win._open_recording())
        self.assertEqual(self.decoding(), ("ACARS", False))

    def test_back_on_the_presets_channel(self):
        win = self.win
        self.open_fm_capture()
        win._playback.close(restore=True)  # (File > Close Recording)
        self.assertEqual(win._current_frequency(), 131.55e6)
        self.assertEqual(self.decoding(), ("ACARS", True))
        self.assertEqual(
            self.message(),
            "Closed fm8.sigmf-data. Back to Demo Device (simulated signals). "
            "ACARS decoder on again.",
        )
        self.assertEqual(win._decoder_chip.text(), "ACARS · 0 msgs")
        # (It is the preset's again: tuned away, it goes off.)
        self.assertTrue(win.set_frequency(100.1e6))
        self.assertEqual(self.decoding(), ("ACARS", False))

    def test_a_decoder_chosen_meanwhile_stays(self):
        win = self.win
        self.open_fm_capture()
        win._decoder_panel._proto_combo.setCurrentText("RDS")
        self.assertEqual(self.decoding(), ("RDS", True))
        win._playback.close(restore=True)
        self.assertEqual(self.decoding(), ("RDS", True))
        self.assertNotIn("ACARS", self.message())


# --------------------------------------------------------------------------- #
# W-08: Open Recording in another band says the file is playing
# --------------------------------------------------------------------------- #


class TestOpeningARecordingInAnotherBandNamesIt(_files_case()):
    """W-08: opening an FM recording while tuned elsewhere said only "ACARS
    decoder off · FM Broadcast: listening in FM (broadcast, 75 kHz
    deviation) with a 200 kHz filter, as the band is usually heard ...";
    the same file opened in its own band says "Playing fm8.sigmf-meta
    (98.9–101.3 MHz). Click a signal to tune to it; Space pauses."."""

    def test_from_a_presets_channel(self):
        from tests.test_final_polish import _save_sigmf

        win = self.win
        self.assertTrue(win._apply_preset_named("ACARS"))
        path = _save_sigmf(self.path("fm8.sigmf-data"), 100.1e6)
        with self.open_dialog(path):
            self.assertTrue(win._open_recording())
        text = self.message()
        self.assertTrue(
            text.startswith("Playing fm8.sigmf-data (99.6–100.6 MHz) · "), text
        )
        self.assertIn(
            " · ACARS decoder off · FM Broadcast: listening in FM (broadcast, "
            "75 kHz deviation) with a 200 kHz filter",
            text,
        )
        # One message: not also the band plan's line without the file (a
        # click on the message area lists the messages).
        said = [entry.text for entry in win._status_history.entries()]
        self.assertEqual(said[0], text)
        self.assertEqual([t for t in said[1:] if t in text], [], said)

    def test_in_its_own_band_as_before(self):
        from tests.test_final_polish import _save_sigmf

        win = self.win
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        self.assertTrue(win.set_frequency(100.1e6))
        path = _save_sigmf(self.path("fm8.sigmf-data"), 100.1e6)
        with self.open_dialog(path):
            self.assertTrue(win._open_recording())
        self.assertEqual(
            self.message(),
            "Playing fm8.sigmf-data (99.6–100.6 MHz). Click a signal to tune "
            "to it; Space pauses.",
        )


# --------------------------------------------------------------------------- #
# W-09: another playback speed is said once
# --------------------------------------------------------------------------- #


def _playback_case():
    from tests.test_gui_playback import _PlaybackCase

    return _PlaybackCase


class TestAnotherPlaybackSpeedIsSaidOnce(_playback_case()):
    """W-09: choosing 4x said "Playing at 4x: audio plays at 1x only." and,
    10 ms later, the paused audio's own "Playing at 4x: audio plays at 1x
    only (the spectrum, decoders and recordings keep working).": two
    entries in the status history (a click on the message area) for one
    choice. One message says it."""

    SAID = (
        "Playing at {}x: audio plays at 1x only (the spectrum, decoders and "
        "recordings keep working)."
    )

    def speed_messages(self):
        return [
            entry.text
            for entry in self.win._status_history.entries()
            if entry.text.startswith("Playing at ")
        ]

    def run_frames(self, count=20):
        for _ in range(count):
            self.win._update_display()
            self.settle()

    def test_with_the_speaker_on(self):
        win = self.win
        self.play()
        win._audio_enabled = True  # (no sound card here switched it off)
        win._control_panel.set_squelch_db(-120)
        win._playback_bar.set_speed(4.0)
        self.run_frames()
        self.assertFalse(win._realtime())  # (the audio is paused)
        self.assertEqual(self.message(), self.SAID.format(4))
        self.assertEqual(self.speed_messages(), [self.SAID.format(4)])
        # Back at 1x, as before.
        win._playback_bar.set_speed(1.0)
        self.run_frames()
        self.assertEqual(self.message(), "Playing at 1x.")

    def test_with_the_speaker_off(self):
        """Said all the same (the audio, off, says nothing)."""
        win = self.win
        self.play()
        win._playback_bar.set_speed(2.0)
        self.run_frames()
        self.assertEqual(self.speed_messages(), [self.SAID.format(2)])


# --------------------------------------------------------------------------- #
# W-10: the Decoder's line once the job is done
# --------------------------------------------------------------------------- #


def _demo_take_case():
    from tests.test_final_record import _DemoTakeCase

    return _DemoTakeCase


class TestTheDecodersLineOnceTheJobIsDone(_demo_take_case()):
    """W-10: after a busy Decode Recording the Decoder's line went on saying
    "Decode Recording keeps this computer busy, so the RDS decoder misses
    part of the signal meanwhile." for about 4 s after the job had ended:
    what the job made the display loop skip still counts for a few seconds,
    so it is still the job that is named, but in the present tense. Once the
    job is done it is said in the past tense (the demo device, on a fake
    clock, at ten frames a second: a busy computer)."""

    FRAME_S = 0.1
    BUSY = (
        "Decode Recording keeps this computer busy, so the POCSAG decoder "
        "misses part of the signal meanwhile."
    )
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

    def run_frames(self, seconds):
        end = self.clock.now + seconds
        while self.clock.now < end:
            tick = self.clock.now
            self.win._update_display()
            self.clock.now = max(self.clock.now, tick + self.FRAME_S)

    def line(self):
        return self.win._decoder_panel.decoder_status_text()

    def a_busy_job(self, seconds):
        """Decode Recording's job (run_job's own event loop) with ``seconds``
        of display frames meanwhile; the Decoder's line at its end."""
        import threading

        from PyQt6.QtCore import QTimer

        from sdr_module.gui.file_jobs import run_job

        done = threading.Event()
        said = []

        def frames_meanwhile():
            try:
                self.run_frames(seconds)
                said.append(self.line())
            finally:
                done.set()

        QTimer.singleShot(0, frames_meanwhile)
        result = run_job(
            self.win, "Decode Recording", "Decoding…", lambda report: done.wait(20)
        )
        self.assertTrue(result.ok, result.error)
        return said[0]

    def test_said_in_the_past_once_it_is_done(self):
        self.run_frames(1.0)  # (can't keep up, before and during the job)
        self.assertEqual(self.a_busy_job(1.5), self.BUSY)
        self.load = 0.3  # (done: the computer keeps up again)
        lines = []
        for _ in range(60):
            self.run_frames(self.FRAME_S)
            lines.append(self.line())
            if len(lines) == 1:
                # (The audio's hint, the Audio button's tooltip, alike.)
                self.assertEqual(
                    self.win._realtime_hint(),
                    "Decode Recording kept this computer busy, so audio is "
                    "paused until it catches up.",
                )
        self.assertNotIn(self.BUSY, lines)
        self.assertIn(self.DONE, lines)
        self.assertNotIn("misses", lines[-1])  # (then nothing)
        self.assertNotIn("missed", lines[-1])


if __name__ == "__main__":
    unittest.main()

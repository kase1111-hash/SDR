"""Final finish: the last refinements of the polish round's fixes (N-01 to
N-08 of the final pass).

Each test names the problem it covers; most run the real window offscreen
on the demo device (or the part on its own where that is enough).
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import require_pyqt6  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import signal_synth as S  # noqa: E402

_DEVIATIONS_HZ = (2.5e3, 5e3, 12.5e3, 25e3, 75e3)


def _hz(text):
    """ "12.5 kHz" -> 12500.0"""
    value, unit = text.split()
    return float(value) * {"Hz": 1.0, "kHz": 1e3}[unit]


class TestIdentifyHearsVoiceAtItsDeviation(unittest.TestCase):
    """N-01: Identify held the measured peak with 10 % to spare, so a 2 m
    voice station whose radio limits it at about +/-5 kHz (measured peak
    4.6-5 kHz) was suggested 12.5 kHz, and Apply Mode made it about 8 dB
    quieter. What the window's clipping warning measures is the share of
    the audio beyond the deviation (it warns above 5 %): voice reaches its
    peak only now and then, a tone stays near it. Here the window's own
    receiver chain measures that share at the deviation Identify suggests
    (it stays under 5 %) and at the next smaller choice (over 5 %: the
    suggestion is no wider than it needs to be)."""

    RATE = 240e3
    SECONDS = 1.5

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import main_window

        self.mw = main_window
        self.n = int(self.SECONDS * self.RATE)

    def clip_fraction(self, x, deviation_hz, width_hz):
        """What the clipping warning measures: the share of audio samples
        beyond full scale (30 blocks a second, as the display loop hands
        them over)."""
        chain = self.mw._ReceiverChain(self.RATE, "FM", width_hz)
        over = total = 0.0
        block = int(self.RATE / 30)
        for start in range(0, len(x) - block + 1, block):
            audio = chain.demodulate(
                chain.channelize(x[start : start + block]), deviation_hz
            )
            over += chain.last_clip_fraction * len(audio)
            total += len(audio)
        return over / max(total, 1.0)

    def check(self, signal, occupied_hz, width_hz, snr_db):
        from sdr_module.dsp.signal_id import identify

        x, _ = S.add_noise(signal, self.RATE, snr_db, occupied_hz, seed=3)
        x = x.astype(np.complex64)
        ident = identify(
            x[: int(self.RATE)], self.RATE, 146.52e6, search_hz=0.75 * width_hz
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
        return ident

    def limited_voice(self, seed):
        """Voice through a radio's limiter: driven well past it, clipped,
        then the filter after the limiter (which overshoots a little)."""
        audio = np.clip(4.0 * S.voice(self.n, self.RATE, seed=seed), -1.0, 1.0)
        audio = S._lowpass(audio, 3000.0, self.RATE)
        return audio / np.max(np.abs(audio))

    def test_a_2_m_station_limited_at_5_khz(self):
        # (Its overshoot reaches +/-5.4 kHz; the limiter holds about 4.8.)
        signal = S.fm(self.n, self.RATE, 5.4e3, self.limited_voice(3))
        ident = self.check(signal, 16.8e3, 25e3, snr_db=30.0)
        self.assertEqual(ident.fm_deviation, "5 kHz")
        peak = ident.result.features["freq_peak_hz"]
        self.assertGreater(1.1 * peak, 5e3)  # (the old margin's 12.5 kHz)
        self.assertEqual(ident.filter_hz, 25e3)
        self.assertIn(f"The swing peaks at about ±{peak / 1e3:.1f} kHz", ident.details)

    def test_a_tone_near_a_choice_keeps_the_margin(self):
        """A 2.5 kHz test tone at +/-5.1 kHz reads a little low (under
        5 kHz), stays near its peak and clips about a tenth of the time at
        5 kHz: the warning would go off, so it is suggested 12.5 kHz."""
        tone = S.tones(self.n, self.RATE, freqs=(2500.0,), amps=(1.0,))
        ident = self.check(S.fm(self.n, self.RATE, 5.1e3, tone), 15.2e3, 25e3, 35.0)
        self.assertEqual(ident.fm_deviation, "12.5 kHz")
        self.assertLess(ident.result.features["freq_peak_hz"], 5e3)
        self.assertIn("and stays near it: 5 kHz could clip", ident.details[-1])


def _playback_case():
    from tests.test_gui_playback import _PlaybackCase

    return _PlaybackCase


class _SlowPlaybackCase(_playback_case()):
    """A 1 MS/s recording (looped) on a fake clock; processing a frame costs
    ``load`` times the time its samples cover at 1x (as in test_final_perf's
    TestPlaybackWhileOverloaded)."""

    PLAY_RATE = 1e6

    def recording(self):
        from tests.test_gui_playback import _fm_tone

        return _fm_tone(1.0, rate=self.PLAY_RATE)

    def setUp(self):
        super().setUp()
        from unittest import mock

        from tests.test_final_perf import _Clock

        x = self.recording()
        self.play(self.make_file(x, name="p.sigmf-data", rate=self.PLAY_RATE))
        self.win._playback_bar.set_loop(True)
        self.assertTrue(self.win._playback.active)
        self.clock = _Clock()
        patcher = mock.patch.object(self.mw, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.load = 0.0
        process = self.win._process_block

        def slow_process(samples):
            self.clock.advance(self.load * len(samples) / self.PLAY_RATE)
            process(samples)

        self.win._process_block = slow_process

    def run_frames(self, seconds):
        win = self.win
        win._pacer.restart()
        win._update_display()
        start = self.clock.now
        while self.clock.now - start < seconds:
            tick = self.clock.now
            win._update_display()
            self.clock.now = max(self.clock.now, tick + 0.033)


class TestPlaybackAudioHintGivesAdviceThatApplies(_SlowPlaybackCase):
    """N-02: a recording playing at 1x on a computer that can't keep up in
    real time paused its audio with "This computer can't process 1 MS/s in
    real time right now, so audio is paused until it catches up. A lower
    sample rate (Receiver > Sample rate) helps.", but the rate is refused
    while a file plays."""

    def hints(self):
        """What the window says about the paused audio: the hint (the
        status bar's) and the Audio button's tooltip (no sound card here:
        its state is that of a computer with one)."""
        from unittest import mock

        win = self.win
        with mock.patch.object(type(win), "_audio_state", return_value="paused"):
            win._sync_audio_ui()
        return win._realtime_hint(), win._audio_button.toolTip()

    def test_a_computer_too_slow_for_1x(self):
        # (Under half of real time: audio pauses.)
        self.load = 3.0
        self.run_frames(4.0)
        win = self.win
        self.assertTrue(win._pacer.overloaded)
        self.assertFalse(win._realtime())
        hint, tip = self.hints()
        self.assertEqual(
            hint,
            "This computer can't process the recording's 1 MS/s in real time "
            "right now, so audio is paused until it catches up. Closing other "
            "programs may help.",
        )
        for text in (hint, tip):
            self.assertNotIn("Sample rate", text)
            self.assertNotIn("sample rate", text)
        self.assertIn(hint, tip)

    def test_frames_late_for_a_moment(self):
        """A window hiccup (a stall of the display loop): what the recording
        played meanwhile is behind real time for a moment."""
        self.run_frames(2.0)
        self.assertTrue(self.win._realtime())
        self.clock.advance(1.5)  # (the stall)
        self.run_frames(0.1)
        win = self.win
        self.assertFalse(win._pacer.overloaded)
        self.assertFalse(win._realtime())
        hint, tip = self.hints()
        self.assertEqual(
            hint,
            "The recording isn't playing in real time right now, so audio is "
            "paused until it catches up.",
        )
        self.assertNotIn("Sample rate", tip)

    def test_faster_or_slower_than_1x(self):
        self.win._playback_bar.set_speed(2.0)
        self.assertEqual(
            self.win._realtime_hint(),
            "Playing at 2x: audio plays at 1x only (the spectrum, decoders and "
            "recordings keep working).",
        )

    def test_live_hardware_keeps_its_advice(self):
        """Back on a device, the advice to lower its rate stands."""
        self.load = 3.0
        self.run_frames(2.0)
        self.win._playback.close()
        self.assertFalse(self.win._playback.active)
        self.win._demo_mode = False
        self.win._pacer.overloaded = True
        self.assertIn(
            "A lower sample rate (Receiver > Sample rate) helps.",
            self.win._realtime_hint(),
        )


class TestBurstsLoggedWhileARecordingPlaysSlowly(_SlowPlaybackCase):
    """N-04: the Bursts page paused ("Paused: this computer can't keep up
    right now") whenever the display loop was behind, as if each frame read
    one block of the signal and skipped the rest, which a device does. A
    recording is read in full however slowly it plays (each frame takes
    what follows on, 7433741): its bursts are logged whole."""

    BURSTS_S = (0.30, 0.95, 1.60)
    BURST_LEN_S = 0.04

    def recording(self):
        """Noise with on/off keyed bursts 120 kHz off the centre."""
        rate = self.PLAY_RATE
        n = int(2.0 * rate)
        rng = np.random.default_rng(5)
        x = 0.003 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        t = np.arange(n) / rate
        tone = 0.3 * np.exp(2j * np.pi * 120e3 * t)
        for start in self.BURSTS_S:
            on = slice(int(start * rate), int((start + self.BURST_LEN_S) * rate))
            x[on] += tone[on]
        return x.astype(np.complex64)

    def test_bursts_are_logged_whole(self):
        win = self.win
        panel = win._signal_panel
        tools = win._signal_tools
        win._set_demod_mode("None (I/Q)", bandwidth="200 kHz")
        win._show_signal_page("bursts")
        self.load = 3.0  # (a third of real time)
        self.run_frames(4.0)
        self.assertTrue(win._pacer.overloaded)
        for job, _report, _when in list(tools._burst_jobs):
            job.result(timeout=30)
        tools._collect_bursts()
        self.assertFalse(tools._bursts_paused)
        self.assertNotIn("can't keep up", panel.bursts_empty.text())
        # Each burst played is listed once, whole: as long as it is, give
        # or take the frames it was seen in (up to 50 ms of signal each).
        device = win._device
        self.assertEqual(device.loops, 0)
        played_s = device.position / self.PLAY_RATE
        played = [t for t in self.BURSTS_S if t + self.BURST_LEN_S < played_s]
        self.assertTrue(played)
        listed = [report.burst for _shown, report in panel.bursts()]
        self.assertEqual(len(listed), len(played), (played_s, listed))
        for burst in listed:
            self.assertAlmostEqual(
                burst.center_hz - win._lo_frequency(), 120e3, delta=5e3
            )
            self.assertGreaterEqual(burst.duration_s, self.BURST_LEN_S - 0.005)
            self.assertLessEqual(burst.duration_s, self.BURST_LEN_S + 0.1)

    def test_a_device_still_pauses_them(self):
        """...whereas a device's frames skip what they can't take."""
        win = self.win
        win._playback.close()
        self.assertTrue(win._demo_mode)
        win._show_signal_page("bursts")
        win._pacer.overloaded = True
        win._process_block(win._device.read_samples(2048))
        self.assertTrue(win._signal_tools._bursts_paused)
        self.assertIn("can't keep up", win._signal_panel.bursts_empty.text())


def _files_case():
    from tests.test_gui_recording_files import _FilesCase

    return _FilesCase


class TestADecoderThatCantDecodeIsLoggedOncePerSituation(_files_case()):
    """N-05: since P-05 the decoder is rebuilt when a recording starts or
    stops playing at the same rate (to word its problem for the recording
    or the device). Each rebuild logged another "ADS-B decoder: ..."
    warning to Tools > Error History: every playback start and stop added
    one. One is logged per real change (the decoder, the rate, a recording
    or the device); the same situation again adds none."""

    def open_file(self):
        from tests.test_final_polish import _save_sigmf

        path = _save_sigmf(self.path("same.sigmf-data"), 1090e6, 1e6)
        with self.open_dialog(path):
            self.assertTrue(self.win._open_recording())
        self.settle(2)

    def logged(self, action):
        """The "ADS-B decoder: ..." warnings ``action`` logs."""
        import logging

        records = []
        handler = logging.Handler(logging.WARNING)
        handler.emit = records.append
        log = logging.getLogger("sdr_module")
        log.addHandler(handler)
        try:
            action()
        finally:
            log.removeHandler(handler)
        return [
            r.getMessage()
            for r in records
            if r.getMessage().startswith("ADS-B decoder:")
        ]

    def close_file(self):
        self.win._playback.close()
        self.settle(2)

    def test_playback_started_and_stopped_again(self):
        win = self.win
        self.assertTrue(win.set_sample_rate(1e6))

        def choose():
            win._decoder_panel._proto_combo.setCurrentText("ADS-B")  # by hand
            self.settle(2)

        device = self.logged(choose)
        self.assertEqual(len(device), 1)
        self.assertIn("the device runs at 1 MS/s", device[0])
        recording = self.logged(self.open_file)  # (a real change: logged)
        self.assertEqual(len(recording), 1)
        self.assertIn("this recording is 1 MS/s", recording[0])
        for _ in range(2):  # (the same two situations again: nothing new)
            self.assertEqual(self.logged(self.close_file), [])
            self.assertIn("the device runs at 1 MS/s", win._decoder_chip.toolTip())
            self.assertEqual(self.logged(self.open_file), [])
            self.assertIn("this recording is 1 MS/s", win._decoder_chip.toolTip())
        self.close_file()

    def test_a_real_change_is_logged_again(self):
        win = self.win
        self.assertTrue(win.set_sample_rate(1e6))
        panel = win._decoder_panel

        def choose(name):
            def action():
                panel._proto_combo.setCurrentText(name)
                self.settle(2)

            return action

        self.assertEqual(len(self.logged(choose("ADS-B"))), 1)
        # Another rate and back: the rate changed, said again.
        self.assertEqual(self.logged(lambda: win.set_sample_rate(2.4e6)), [])
        self.assertEqual(len(self.logged(lambda: win.set_sample_rate(1e6))), 1)
        # Another decoder and back: so is the decoder.
        self.logged(choose("POCSAG"))
        self.assertEqual(len(self.logged(choose("ADS-B"))), 1)


class TestEveryJobTitleReadsInTheHints(unittest.TestCase):
    """N-06: while one of this app's jobs keeps the computer busy, the hints
    name it: "<title> keeps this computer busy, so audio is paused
    meanwhile." (and "... so the ADS-B decoder misses part of the signal
    meanwhile."). Transmit Text's save was titled "Saving": "Saving keeps
    this computer busy, ...". Each job is titled with the name of the
    command that runs it, as File > Decode Recording's is."""

    def titles(self):
        """The title of every job the app runs (run_job's second argument)."""
        import ast

        src = Path(__file__).resolve().parent.parent / "src" / "sdr_module"
        found = {}
        for path in sorted(src.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                func = getattr(node, "func", None)
                if isinstance(node, ast.Call) and getattr(func, "id", "") == "run_job":
                    title = node.args[1]
                    self.assertIsInstance(title, ast.Constant, path.name)
                    found[title.value] = path.name
        return found

    def test_each_title_is_a_command_name(self):
        import re

        titles = self.titles()
        self.assertIn("Decode Recording", titles)
        self.assertGreaterEqual(len(titles), 5, titles)
        for title in titles:
            words = title.split()
            # (A command: "Save Text Signal", not "Saving".)
            self.assertRegex(title, re.compile(r"^[A-Z][a-z]+( [A-Z][a-z]+)+$"))
            self.assertFalse(words[0].endswith("ing"), title)

    def test_transmit_texts_save(self):
        """Titled as its file dialog is: "Save Text Signal"."""
        self.assertEqual(self.titles().get("Save Text Signal"), "transmit_text.py")


def _tx_text_case():
    from tests.test_gui_transmit_text import _TextCase

    return _TextCase


class TestTransmitTextsSaveNamedInTheHints(_tx_text_case()):
    """N-06, in the window: the job Transmit Text's Save runs is named in
    the hints as a command."""

    def test_the_hint_names_the_save(self):
        import shutil
        import tempfile
        from unittest import mock

        from PyQt6.QtWidgets import QFileDialog

        from sdr_module.gui import file_jobs, main_window

        folder = tempfile.mkdtemp(prefix="n06_")
        self.addCleanup(shutil.rmtree, folder, True)
        path = os.path.join(folder, "cq.wav")
        self.patch.setattr(
            QFileDialog,
            "getSaveFileName",
            staticmethod(lambda *a: (path, "WAV audio (*.wav)")),
        )
        self.dlg.set_mode("CW")
        self.dlg.set_text("CQ")
        with mock.patch.object(file_jobs, "run_job", wraps=file_jobs.run_job) as job:
            self.dlg._save_button.click()
        title = job.call_args.args[1]
        self.assertEqual(title, "Save Text Signal")
        self.assertTrue(os.path.exists(path))
        with mock.patch.object(main_window, "busy_with", return_value=title):
            self.assertEqual(
                self.win._realtime_hint(),
                "Save Text Signal keeps this computer busy, so audio is paused "
                "meanwhile.",
            )


def _window_case():
    from tests.test_gui_chrome import _WindowTestCase

    return _WindowTestCase


class TestSecondReceiverWaterfallSettingDamaged(_window_case()):
    """N-03: a damaged settings file gives each setting its default and is
    logged at WARNING (Tools > Error History), as the main window's plots
    and the second receiver's spectrum (e493f13) are. The second
    receiver's saved waterfall choices were dropped with a debug line when
    unreadable, and silently when not a set of choices at all."""

    demo = True

    def open_with(self, key, raw):
        from sdr_module.gui import second_receiver

        self.store.values[getattr(second_receiver, key)] = raw
        with self.assertLogs("sdr_module", "WARNING") as logs:
            rx = self.win._second_rx.show()
            self.addCleanup(rx.shutdown)
            rx._status_timer.stop()
        return rx, [r.getMessage() for r in logs.records]

    def test_unreadable_waterfall_settings(self):
        rx, said = self.open_with("WATERFALL_KEY", "{not json")
        self.assertEqual(
            said,
            [
                "Ignoring the second receiver's unreadable waterfall settings: '{not json'"
            ],
        )
        # (The main window's defaults, as with no saved choice.)
        shown = rx._waterfall.display_settings()
        self.assertTrue(shown["auto_levels"])
        self.assertEqual(shown["history_s"], 60.0)

    def test_saved_choices_that_are_not_choices(self):
        rx, said = self.open_with("WATERFALL_KEY", "[1, 2]")
        self.assertEqual(
            said,
            ["Ignoring the second receiver's unreadable waterfall settings: '[1, 2]'"],
        )

    def test_the_spectrums_too(self):
        rx, said = self.open_with("SPECTRUM_KEY", "7")
        self.assertEqual(
            said, ["Ignoring the second receiver's unreadable spectrum settings: '7'"]
        )

    def test_one_line_per_damaged_value(self):
        raw = '{"floor_db": "abc", "colormap": "nope", "history_s": 300}'
        rx, said = self.open_with("WATERFALL_KEY", raw)
        self.assertEqual(
            sorted(said),
            [
                "Ignoring the damaged waterfall setting colormap='nope': using "
                "'turbo' instead.",
                "Ignoring the damaged waterfall setting floor_db='abc': using "
                "-100.0 instead.",
            ],
        )
        self.assertEqual(rx._waterfall.display_settings()["history_s"], 300.0)


class TestWaterfallKeepsUnusualLinesOutOfItsReference(unittest.TestCase):
    """N-07, the waterfall on its own: lines it is told are from frames of
    an unusual length (``usual=False``) are coloured like the others but
    don't set how bright their noise shows, however many there are."""

    BINS = 1024

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        self.wf = WaterfallWidget()
        self.addCleanup(self.wf.deleteLater)
        self.wf.set_frequency_range(100e6, 2.4e6)
        self.wf.set_history_seconds(0.0)  # (a row per line)
        self.rng = np.random.default_rng(5)
        self.t = 0.0

    def lines(self, ffts, count, usual=True):
        """Rows of the strongest of ``ffts`` FFTs of noise; their noise as
        drawn (dB)."""
        wf, shown = self.wf, []
        for _ in range(count):
            power = self.rng.exponential(1.0, (ffts, self.BINS)).max(axis=0)
            self.t += 1 / 30
            wf.add_line(-80.0 + 10 * np.log10(power), self.t, -80.0, usual=usual)
            codes = wf._row_codes[-1][1].astype(float)
            shown.append(float(np.median(codes / 8.0 - 200.0)))
        return np.array(shown)

    def test_a_long_stretch_of_one_block_frames(self):
        usual = float(np.median(self.lines(39, 300)))
        for shown in (self.lines(1, 300, usual=False), self.lines(39, 300)):
            self.assertLess(float(np.max(np.abs(shown - usual))), 0.5)

    def test_only_unusual_lines_since_a_retune(self):
        """They are all there is to go by; the first usual lines take over
        within a few rows."""
        self.lines(39, 60)
        self.wf.set_frequency_range(144e6, 2.4e6)  # (a retune)
        self.lines(1, 40, usual=False)
        shown = self.lines(39, 10)
        normal = float(np.median(self.lines(39, 30)))
        self.assertLess(float(np.max(np.abs(shown[2:] - normal))), 0.5, shown)


class TestWaterfallBrightnessThroughALongOverload(_window_case()):
    """N-07: the waterfall colours each row so its noise shows as bright as
    usual lines' (P-10: the median of the last 301 lines' noise rise). An
    overload longer than half of that (one-block frames for ~5 s) moved the
    median at once: the rows stepped by about 7 dB during it and again
    about 5 s after it. Frames read while this computer can't keep up are
    kept out of that reference, however long it lasts."""

    demo = True
    FRAME_S = 1 / 30

    def shown_db(self):
        """The noise of the newest row as drawn (its median colour, dB)."""
        codes = self.win._waterfall._row_codes[-1][1].astype(float)
        return float(np.median(codes / 8.0 - 200.0))

    def frames(self, count, overloaded):
        """``count`` frames, as a computer that can't keep up reads them
        (one block each) or as usual; the noise of each row as drawn."""
        win = self.win
        rate = win._device_sample_rate()
        shown = []
        for _ in range(count):
            win._pacer.reset()
            win._pacer.overloaded = overloaded
            size = 2048 if overloaded else int(rate * self.FRAME_S)
            win._process_block(win._device.read_samples(size))
            shown.append(self.shown_db())
        return np.array(shown)

    def test_no_step_during_or_after(self):
        from unittest import mock

        from sdr_module.gui import device_dialog

        patcher = mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9)
        patcher.start()  # (every demo sample is read)
        self.addCleanup(patcher.stop)
        win = self.win
        win._device._rng = np.random.default_rng(7)
        win._waterfall.set_history_seconds(0.0)  # (a row per line)
        if not win._is_running:
            win._start_acquisition()
        before = self.frames(160, False)
        usual = float(np.median(before[40:]))
        behind = self.frames(170, True)  # (more than half of 301 lines)
        after = self.frames(170, False)
        # Drawn alike all through (one-block rows read their noise floor a
        # little differently: about 1.4 dB here, as from the first one).
        first, last = np.median(behind[:20]), np.median(behind[-20:])
        self.assertLess(abs(float(last - first)), 0.5, (first, last))
        self.assertLess(float(np.ptp(behind)), 2.0, behind)
        self.assertLess(float(np.max(np.abs(after - usual))), 0.5, after[:5])
        # (The frames behind were one block each: their rows' own levels,
        # unlike their colours, read several dB lower.)
        history = win._waterfall._history
        drop = float(np.median(history[-200]) - np.median(history[-1]))
        self.assertLess(drop, -5.0)


def _scan_window_case():
    from tests.test_gui_scanner import _ScanWindowCase

    return _ScanWindowCase


class TestOnlyTheTunesOwnMessageJoinsTheScanStop(_scan_window_case()):
    """N-08: a tune from another tab during a scan says its own message and
    the scan's stop in one ("Scan stopped · ADS-B: 1090.000 MHz ...",
    P-08). Any message posted after the tune counted as the tune's: one
    from something else before the next frame (or during it) was joined to
    the stop as if the tune had said it."""

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

    STOPPED = "Scan stopped: you tuned to 1090.000 MHz."

    def test_a_message_from_something_else_before_the_next_frame(self):
        win = self.win
        self.assertTrue(win._apply_preset_named("ADS-B"))
        self.settle()  # (back in the event loop: the tune has said its own)
        win._show_status_message("Saved the recording.", "success", 5000)
        self.frames(0.1)
        self.assertFalse(win._scanner_panel.engine.active)
        self.assertEqual(win._message_label.text(), self.STOPPED)

    def test_a_message_from_the_next_frame_itself(self):
        win = self.win
        self.assertTrue(win._apply_preset_named("ADS-B"))
        run = win._run_decoder

        def says_something(*args, **kw):
            win._show_status_message("Something else.", "info", 5000)
            return run(*args, **kw)

        win._run_decoder = says_something  # (before the scan sees the frame)
        engine = win._scanner_panel.engine
        self.assertTrue(self.frames(0.2, until=lambda: not engine.active))
        self.assertEqual(win._message_label.text(), self.STOPPED)

    def test_the_tunes_own_message_still_joins_it(self):
        win = self.win
        self.assertTrue(win._apply_preset_named("ADS-B"))
        said = win._message_label.text()
        self.settle()
        self.frames(0.1)
        self.assertEqual(win._message_label.text(), f"Scan stopped · {said}")


if __name__ == "__main__":
    unittest.main()

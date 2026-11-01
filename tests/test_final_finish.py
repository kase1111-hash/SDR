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


class TestPlaybackAudioHintGivesAdviceThatApplies(_playback_case()):
    """N-02: a recording playing at 1x on a computer that can't keep up in
    real time paused its audio with "This computer can't process 1 MS/s in
    real time right now, so audio is paused until it catches up. A lower
    sample rate (Receiver > Sample rate) helps.", but the rate is refused
    while a file plays. Here processing costs ``load`` times the time its
    samples cover at 1x (a fake clock, as in test_final_perf)."""

    PLAY_RATE = 1e6

    def setUp(self):
        super().setUp()
        from unittest import mock

        from tests.test_final_perf import _Clock
        from tests.test_gui_playback import _fm_tone

        x = _fm_tone(1.0, rate=self.PLAY_RATE)
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


if __name__ == "__main__":
    unittest.main()

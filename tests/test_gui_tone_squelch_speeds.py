#!/usr/bin/env python3
"""
The tone squelch while a recording plays at another speed.

Three lanes meet in the window's _route_audio: the squelch lane's tone
squelch (the CTCSS reverse burst, the DCS turn-off code, the reverse tone
squelch reset as the squelch opens), the playback lane's sound at 0.5x, 2x
and 4x (PlaybackSound) and Record Audio. At 0.5x and 4x a station's reverse
burst or turn-off code closes the squelch as at 1x, before its carrier
drops: nothing of the squelch tail is heard (with Squelch Tail Elimination
off the CTCSS tail is heard, as before the burst was looked for). With the
reverse tone squelch and Record Audio running at 2x, the station it mutes
is neither heard nor recorded, from its first moment.

The display loop runs on a fake clock (the playback lane's harness); the
recordings are made here: narrow FM at 240 kS/s.
"""

import os
import unittest
import wave

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_ctcss_reverse_burst import tone_with_burst  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_playback_sound import _SoundCase  # noqa: E402

RATE = 240e3
AUDIO = 48000.0
CENTER = 146.52e6
BEFORE_S = 1.0  # carrier off before the transmission
ON_S = 3.0  # the transmission, up to its burst or turn-off code
AFTER_S = 1.0  # carrier off after it


def _recording(kind="ctcss", tone_hz=100.0):
    """A transmission with a 1 kHz voice tone and its CTCSS tone (ending
    with a 150 ms reverse burst) or DCS 023 N (ending with its 180 ms
    turn-off code), the carrier off before and after (noise); and when the
    burst or turn-off code starts and the carrier drops (s)."""
    from sdr_module.dsp.dcs import DCS_TURN_OFF_S, dcs_transmission

    if kind == "ctcss":
        lead = 0.15
        sub = tone_with_burst(tone_hz, ON_S, 180.0, lead) / 0.12 * 600.0
    else:
        lead = DCS_TURN_OFF_S
        sub = 750.0 * dcs_transmission(23, rate=AUDIO, seconds=ON_S)
    n = len(sub)
    t = np.arange(n) / AUDIO
    deviation = sub + 2.5e3 * np.sin(2 * np.pi * 1000.0 * t) * (t < ON_S)
    held = np.repeat(deviation, int(RATE / AUDIO))
    phase = 2 * np.pi * np.cumsum(held) / RATE
    on = (0.3 * np.exp(1j * phase)).astype(np.complex64)
    off_before = np.zeros(int(BEFORE_S * RATE), np.complex64)
    off_after = np.zeros(int(AFTER_S * RATE), np.complex64)
    x = np.concatenate((off_before, on, off_after))
    rng = np.random.default_rng(3)
    x += 1e-3 * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    end = BEFORE_S + ON_S
    return x.astype(np.complex64), end, end + lead


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class _SpeedCase(_SoundCase):
    def listen(self, kind, speed, record=False, tone_hz=100.0, reverse=False):
        """Per processed block: (recording time at its end, peak written to
        the speaker, squelch open, tone squelch holding it back)."""
        x, signal_end, dropped = _recording(kind)
        self.open(x, rate=RATE, center=CENTER)
        win = self.win
        panel = win._control_panel
        win._playback_bar.set_loop(False)
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        self.assertTrue(win.set_frequency(CENTER))
        panel.set_squelch_enabled(True)
        panel.set_squelch_auto(True)
        if kind == "dcs":
            panel.set_dcs_squelch(23)
        else:
            panel.set_tone_squelch_hz(tone_hz)
        panel.set_tone_reverse(reverse)
        win._playback.seek(0.0)
        win._playback_bar.set_speed(speed)
        if record:
            self.assertTrue(win._start_audio_recording())
            self.addCleanup(win._stop_audio_recording, True)
        trace = []
        process = win._process_block

        def traced(samples):
            before = len(self.heard)
            process(samples)
            written = [b for _t, b in self.heard[before:] if len(b)]
            trace.append(
                (
                    win._device.position / RATE,
                    max((float(np.max(np.abs(b))) for b in written), default=0.0),
                    bool(win._squelch.is_open),
                    bool(win._tone_blocked),
                )
            )

        win._process_block = traced
        end = self.clock.now + 30.0
        while win._device.position / RATE < dropped + AFTER_S - 0.1:
            self.frames(0.033)
            self.assertLess(self.clock.now, end)
        return trace, signal_end, dropped

    def check_closed_at_the_end(self, trace, signal_end, dropped, speed):
        heard = [p for at, p, _o, _b in trace if BEFORE_S + 0.8 < at < signal_end]
        self.assertGreater(max(heard), 0.05, speed)  # (the voice played)
        shut = next(
            at for at, _p, open_, blocked in trace if at > signal_end and blocked
        )
        self.assertLessEqual(shut, dropped + 0.033 * speed + 1e-6, speed)
        # Nothing of the squelch tail (the dropped carrier's noise).
        tail = [p for at, p, _o, _b in trace if at > dropped + 0.033 * speed]
        self.assertTrue(tail)
        self.assertLess(max(tail), 1e-3, speed)


class TestTheEndOfATransmission(_SpeedCase):
    def test_the_reverse_burst_at_half_and_four_times(self):
        for speed in (0.5, 4.0):
            with self.subTest(speed=speed):
                trace, signal_end, dropped = self.listen("ctcss", speed)
                self.check_closed_at_the_end(trace, signal_end, dropped, speed)

    def test_the_turn_off_code_at_half_and_four_times(self):
        for speed in (0.5, 4.0):
            with self.subTest(speed=speed):
                trace, signal_end, dropped = self.listen("dcs", speed)
                self.check_closed_at_the_end(trace, signal_end, dropped, speed)

    def test_tail_elimination_off_the_tail_plays_at_4x(self):
        """(As before the burst was looked for: the noise squelch ends it.)"""
        self.win._set_tail_elimination(False)
        trace, _signal_end, dropped = self.listen("ctcss", 4.0)
        tail = [p for at, p, _o, _b in trace if at > dropped + 0.15]
        self.assertGreater(max(tail), 1e-3)


class TestReverseWithRecordAudioAt2x(_SpeedCase):
    def wav_peak(self, start_s, end_s):
        path = self.win._stop_audio_recording(quiet=True)
        with wave.open(path) as f:
            channels, rate = f.getnchannels(), f.getframerate()
            pcm = np.frombuffer(f.readframes(f.getnframes()), "<i2")
        pcm = pcm.reshape(-1, channels)[:, 0].astype(float) / 32768.0
        return float(np.max(np.abs(pcm[int(start_s * rate) : int(end_s * rate)])))

    def test_the_station_it_mutes_is_neither_heard_nor_recorded(self):
        trace, signal_end, _dropped = self.listen(
            "ctcss", 2.0, record=True, reverse=True
        )
        during = [p for at, p, _o, _b in trace if BEFORE_S <= at <= signal_end]
        self.assertTrue(during)
        self.assertLess(max(during), 1e-3)
        self.assertLess(self.wav_peak(BEFORE_S, signal_end), 1e-3)

    def test_another_station_is_heard_and_recorded(self):
        trace, signal_end, _dropped = self.listen(
            "ctcss", 2.0, record=True, tone_hz=123.0, reverse=True
        )
        # (Decided 0.55 s into the transmission, as at 1x, to a frame.)
        let_through = next(
            at
            for at, _p, open_, blocked in trace
            if BEFORE_S < at and open_ and not blocked
        )
        self.assertLess(let_through - BEFORE_S, 0.55 + 0.066 + 0.05)
        during = [p for at, p, _o, _b in trace if BEFORE_S + 1.0 < at < signal_end]
        self.assertGreater(max(during), 0.05)
        self.assertGreater(self.wav_peak(BEFORE_S + 1.0, signal_end), 0.05)


if __name__ == "__main__":
    unittest.main()

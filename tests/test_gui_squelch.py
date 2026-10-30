#!/usr/bin/env python3
"""
The squelch: decided on the whole block, open at the threshold, closed only
after the level has stayed 3 dB below it for the hang time, the demodulator
reset only when it really closes, Auto (noise + 7 dB) and on/off.

Regression tests for rx:RX-04 (chatter, chopped signals), rx:RX-05 and
mc:MC-U05 (the default threshold passed noise), ux:UX-B05 and ux:UX-U06.
"""

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

RATE = 2.4e6
FRAME = 80000  # 33 ms at 2.4 MS/s
FRAME_S = FRAME / RATE


def noise(n, power_db, seed=0):
    """Complex white noise with a total power of ``power_db`` dBFS."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(10 ** (power_db / 10) / 2)
    return (scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n))).astype(
        np.complex64
    )


def carrier(n, offset_hz=0.0, level_db=-40.0, start=0):
    t = (np.arange(n) + start) / RATE
    amplitude = 10 ** (level_db / 20)
    return (amplitude * np.exp(2j * np.pi * offset_hz * t)).astype(np.complex64)


def fm(audio, deviation, level_db=-40.0, offset_hz=0.0):
    phase = 2 * np.pi * np.cumsum(deviation * np.asarray(audio) + offset_hz) / RATE
    return (10 ** (level_db / 20) * np.exp(1j * phase)).astype(np.complex64)


class Frames:
    """Feeds a long I/Q array to the window a 33 ms frame at a time and
    records the squelch state, the audio written and resets."""

    def __init__(self, test, iq, frame=FRAME):
        self.test = test
        self.win = test.win
        self.iq = iq
        self.frame = frame
        self.pos = 0
        self.states = []
        self.audio = []
        self.resets = 0
        self.win._audio.write = self._write
        self._frame_audio = []

    def _write(self, audio):
        self._frame_audio.append(np.asarray(audio, dtype=np.float32).copy())

    def run(self, frames=None):
        count = 0
        while self.pos + self.frame <= len(self.iq) and (
            frames is None or count < frames
        ):
            block = self.iq[self.pos : self.pos + self.frame]
            self.pos += self.frame
            chain = self.win._receiver_chain()
            original = chain.reset_demodulator

            def counting(_original=original):
                self.resets += 1
                _original()

            chain.reset_demodulator = counting
            self._frame_audio = []
            self.win._process_block(block)
            chain.reset_demodulator = original
            self.states.append(self.win._squelch.is_open)
            self.audio.append(
                np.concatenate(self._frame_audio)
                if self._frame_audio
                else np.zeros(0, np.float32)
            )
            count += 1
        return self

    @property
    def transitions(self):
        s = self.states
        return sum(1 for a, b in zip(s[:-1], s[1:], strict=True) if a != b)

    def played(self):
        return np.concatenate(self.audio) if self.audio else np.zeros(0)


def pretend_audio_output(win):
    """There is no sound card here: let the window believe there is one
    (what it writes is captured by the tests)."""
    win._audio_output_problem = lambda: None
    win._audio.start = lambda *_args, **_kwargs: True
    win._audio_enabled = True


class _SquelchCase(_WindowTestCase):
    def setUp(self):
        super().setUp()
        pretend_audio_output(self.win)
        self.panel = self.win._control_panel

    def listen(self, mode="FM", deviation="5 kHz", bandwidth="25 kHz"):
        self.win.set_frequency(162.55e6)  # (narrow FM profile)
        if self.panel.get_demod_mode() == mode:
            self.win._set_demod_mode("AM")
        self.win._set_demod_mode(mode, deviation, bandwidth)
        self.win._reset_demodulator()


class TestSquelchGate(unittest.TestCase):
    """The decision itself (no window)."""

    def gate(self, **kwargs):
        from sdr_module.gui.audio_chain import SquelchGate

        kwargs.setdefault("auto", False)
        kwargs.setdefault("manual_db", -60.0)
        return SquelchGate(**kwargs)

    def test_opens_at_the_threshold_at_once(self):
        gate = self.gate()
        self.assertEqual(gate.update(-60.5, None, FRAME_S), "")
        self.assertEqual(gate.update(-60.0, None, FRAME_S), "opened")
        self.assertTrue(gate.is_open)

    def test_hysteresis_keeps_it_open_close_to_the_threshold(self):
        gate = self.gate()
        gate.update(-55, None, FRAME_S)
        for level in (-61, -62.9, -58, -62.5) * 20:  # within 3 dB below
            self.assertEqual(gate.update(level, None, FRAME_S), "")
        self.assertTrue(gate.is_open)

    def test_closes_only_after_the_hang_time(self):
        gate = self.gate(hang_s=0.3)
        gate.update(-50, None, FRAME_S)
        changes = [gate.update(-70, None, FRAME_S) for _ in range(12)]
        closed_at = changes.index("closed")
        self.assertAlmostEqual((closed_at + 1) * FRAME_S, 0.3, delta=FRAME_S)
        # A level back above threshold - 3 dB restarts the hang.
        gate = self.gate(hang_s=0.3)
        gate.update(-50, None, FRAME_S)
        for _ in range(5):
            gate.update(-70, None, FRAME_S)
        gate.update(-62, None, FRAME_S)
        for _ in range(6):
            self.assertEqual(gate.update(-70, None, FRAME_S), "")
        self.assertTrue(gate.is_open)

    def test_auto_threshold_is_the_noise_plus_the_margin(self):
        gate = self.gate(auto=True, margin_db=7.0)
        self.assertEqual(gate.threshold_db(-80.0), -73.0)
        self.assertEqual(gate.update(-74.0, -80.0, FRAME_S), "")
        self.assertEqual(gate.update(-72.0, -80.0, FRAME_S), "opened")
        # Without a noise estimate yet it uses the manual threshold.
        self.assertEqual(gate.threshold_db(None), -60.0)

    def test_disabled_is_always_open(self):
        gate = self.gate(enabled=False)
        self.assertEqual(gate.update(-150, None, FRAME_S), "opened")
        self.assertEqual(gate.update(-150, None, FRAME_S), "")
        self.assertTrue(gate.is_open)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestSquelchInTheWindow(_SquelchCase):
    def test_noise_at_the_threshold_does_not_chatter(self):
        # RX-04 / UX-B05: the threshold right at the noise level used to
        # toggle the audio 16 times a second.
        self.listen()
        n = 90 * FRAME
        iq = noise(n, -70.0, seed=1)
        levels = []
        warm = Frames(self, iq[: 30 * FRAME])
        while warm.pos + FRAME <= len(warm.iq):
            warm.run(1)
            levels.append(self.win._last_peak_db)
        # A threshold in the middle of the noise's own spread: about half
        # the frames are above it.
        self.panel.set_squelch_auto(False)
        self.win._squelch.manual_db = float(np.median(levels))
        self.win._reset_demodulator()
        frames = Frames(self, iq[30 * FRAME :]).run()
        self.assertLessEqual(frames.transitions, 1)  # was 48 in 3 s

    def test_a_burst_opens_once_and_closes_after_the_hang(self):
        self.listen()
        self.panel.set_squelch_db(-70)
        n = 120 * FRAME  # 4 s: 1 s quiet, 1 s carrier, 2 s quiet
        iq = noise(n, -80.0, seed=2)
        on = slice(30 * FRAME, 60 * FRAME)
        iq[on] += carrier(30 * FRAME, 1000.0, -50.0)
        frames = Frames(self, iq).run()
        self.assertEqual(frames.transitions, 2)
        opened = frames.states.index(True)
        closed = opened + frames.states[opened:].index(False)
        self.assertIn(opened, (30, 31))
        # Closes the hang time (0.3 s) after the level fell, which is a
        # frame or two after the burst ends (the channel filter's tail).
        delay = (closed - 60) * FRAME_S
        self.assertGreaterEqual(delay, 0.3)
        self.assertLessEqual(delay, 0.45)
        # The demodulator is reset once, when the squelch really closes
        # (it used to be reset on every quiet frame).
        self.assertEqual(frames.resets, 1)
        # No audio before the burst, audio through it and the hang.
        self.assertFalse(any(len(a) for a in frames.audio[:30]))
        self.assertTrue(all(len(a) for a in frames.audio[31:60]))

    def test_keyed_cw_is_not_chopped(self):
        # RX-04: the end of each dit was cut off because the frame's level
        # came from its last 0.85 ms.
        self.listen("CW", bandwidth="500 Hz")
        self.panel.set_squelch_db(-80)
        dit = int(0.08 * RATE)
        pattern = np.zeros(int(4.0 * RATE), dtype=bool)
        for start in range(int(0.5 * RATE), int(3.0 * RATE), 2 * dit):
            pattern[start : start + dit] = True  # dits, a dit apart
        iq = noise(len(pattern), -95.0, seed=3) + carrier(
            len(pattern), 0.0, -60.0
        ) * pattern.astype(np.float32)
        frames = Frames(self, iq).run()
        self.assertEqual(frames.transitions, 2)  # open once, close once
        start = int(0.5 * RATE) // FRAME + 1
        end = int(3.0 * RATE) // FRAME
        self.assertTrue(all(frames.states[start:end]))

    def test_closed_squelch_writes_no_audio_and_fades_at_the_edges(self):
        self.listen()
        self.panel.set_squelch_db(-70)
        n = 60 * FRAME
        iq = noise(n, -80.0, seed=4)
        audio = np.sin(2 * np.pi * 800 * np.arange(30 * FRAME) / RATE)
        iq[15 * FRAME : 45 * FRAME] = (
            fm(audio, 3e3, -50.0) + iq[15 * FRAME : 45 * FRAME]
        )
        frames = Frames(self, iq).run()
        played = [a for a in frames.audio if len(a)]
        first, last = played[0], played[-1]
        self.assertLess(abs(float(first[0])), 0.02)  # faded in
        self.assertLessEqual(len(last), 240)  # 5 ms fade-out...
        self.assertLess(abs(float(last[-1])), 0.05)  # ...down to silence
        # The tone itself plays without a click (the hang after it plays
        # the channel's noise, as squelch tails do).
        tone = np.concatenate(frames.audio[17:44])
        self.assertLess(float(np.max(np.abs(np.diff(tone)))), 0.1)

    def test_squelch_off_plays_an_empty_channel(self):
        self.listen()
        self.panel.set_squelch_enabled(False)
        self.assertFalse(self.win._squelch.enabled)
        frames = Frames(self, noise(10 * FRAME, -80.0, seed=5)).run()
        self.assertTrue(all(frames.states))
        self.assertTrue(all(len(a) for a in frames.audio[1:]))
        self.assertIs(self.store.values["squelch/enabled"], False)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestAutoSquelch(_SquelchCase):
    def test_auto_follows_the_noise_whatever_the_gain(self):
        # RX-05: the absolute -80 dBFS default passed full-scale hiss from a
        # gain of 25 dB up; Auto sits 7 dB above the channel's noise.
        self.listen()
        self.panel.set_squelch_auto(True)
        for level in (-95.0, -80.0, -70.0, -60.0, -50.0):
            frames = Frames(self, noise(15 * FRAME, level, seed=int(-level))).run()
            self.assertFalse(any(frames.states), level)
            threshold = self.win._squelch.last_threshold_db
            self.assertAlmostEqual(threshold - self.win._last_peak_db, 7, delta=2)
        # ...and the slider shows where Auto put the threshold.
        self.assertEqual(
            self.panel.get_squelch_db(), round(self.win._squelch.last_threshold_db)
        )
        # A signal 20 dB above the noise opens it.
        iq = noise(10 * FRAME, -80.0, seed=9) + carrier(10 * FRAME, 1e3, -60.0)
        self.assertTrue(Frames(self, iq).run().states[-1])

    def test_moving_the_slider_sets_the_threshold_by_hand(self):
        self.listen()
        self.assertTrue(self.panel.is_squelch_auto())
        self.panel._squelch_slider.setValue(-55)
        self.assertFalse(self.panel.is_squelch_auto())
        self.assertFalse(self.win._squelch.auto)
        self.assertEqual(self.win._squelch.manual_db, -55.0)
        self.assertEqual(self.store.values["squelch_db"], -55.0)
        self.assertIs(self.store.values["squelch/auto"], False)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestFirstLaunch(_WindowTestCase):
    """Defaults: the demo's 100.1 MHz station plays at once, an empty
    channel is quiet (and the squelch is on, in Auto)."""

    demo = True

    def _audio_at(self, freq):
        from sdr_module.gui.device_dialog import MockDevice

        spoken = []
        self.win._audio.write = spoken.append
        pretend_audio_output(self.win)
        device = MockDevice()
        device.set_frequency(freq)
        device.start_rx()
        self.win.set_frequency(freq)
        for _ in range(30):
            self.win._process_block(device.read_samples(FRAME))
        return sum(len(a) for a in spoken) / 48000.0

    def test_defaults(self):
        panel = self.win._control_panel
        self.assertTrue(panel.is_squelch_enabled())
        self.assertTrue(panel.is_squelch_auto())
        self.assertEqual(panel.get_demod_mode(), "FM")
        self.assertGreater(self._audio_at(100.1e6), 0.9)  # the station plays
        self.win._reset_demodulator()
        self.assertEqual(self._audio_at(100.5e6), 0.0)  # empty: quiet
        self.win._reset_demodulator()
        self.assertEqual(self._audio_at(96.3e6), 0.0)


if __name__ == "__main__":
    unittest.main()

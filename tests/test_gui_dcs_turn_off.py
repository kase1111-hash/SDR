#!/usr/bin/env python3
"""
The DCS turn-off code in the window and on the demo device (M-Q1).

A DCS transmission that ends with its turn-off code (a 134.4 Hz tone sent
for about 180 ms after the code, the carrier still up) closes the tone
squelch within 100 ms of the tone's start, so the squelch tail (the noise
as the carrier drops) is never heard; Tone heard says None as soon as the
code has gone. Before, the squelch stayed open until 0.35-0.40 s after the
code stopped, and the tail was heard. The demo's 70 cm repeater (442.100
MHz, DCS 023 N) sends it at the end of each transmission.
"""

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_squelch import (  # noqa: E402
    FRAME,
    RATE,
    Frames,
    fm,
    noise,
    pretend_audio_output,
)
from tests.voice_synth import pitched_voice, tone_level  # noqa: E402

AUDIO_RATE = 48000.0
FRAME_S = FRAME / RATE


def _voice_above_300(seconds, seed=1):
    from sdr_module.dsp.audio_filters import SosFilter, highpass_sos

    voice = pitched_voice(seconds, AUDIO_RATE, seed=seed)
    return SosFilter(highpass_sos(300.0, AUDIO_RATE)).process(voice)


def ending_transmission(code=23, seconds=2.0, turn_off=True, after_s=0.6):
    """Narrow FM (5 kHz) voice with DCS ``code`` for ``seconds``, then its
    turn-off code (the voice stopped), then the carrier gone (noise only),
    at 2.4 MS/s; and the sample where the carrier drops."""
    from sdr_module.dsp.dcs import dcs_transmission

    audio = 0.15 * dcs_transmission(
        code,
        rate=AUDIO_RATE,
        seconds=seconds,
        turn_off_s=None if turn_off else 0.0,
    )
    voice = 0.6 * _voice_above_300(seconds)
    audio[: len(voice)] += voice
    held = np.repeat(audio, int(RATE / AUDIO_RATE))
    on = len(held)
    off = int(after_s * RATE)
    total = on + off
    total -= total % FRAME
    iq = np.concatenate((fm(held, 5e3, -50.0), np.zeros(total - on, np.complex64)))
    return iq + noise(total, -95.0), on


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheTurnOffCodeInTheReceiver(_WindowTestCase):
    """A 70 cm channel (FM, 5 kHz) with a DCS 023 N squelch, through the
    real receiver chain."""

    def setUp(self):
        super().setUp()
        pretend_audio_output(self.win)
        self.panel = self.win._control_panel
        self.win.set_frequency(446.1e6)
        self.win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        self.panel.set_squelch_db(-90)

    def listen(self, iq):
        """Per frame: (end time s, peak of the audio played, lamp state,
        Tone heard)."""
        win = self.win
        win._reset_demodulator()
        frames = Frames(self, iq)
        out = []
        while frames.pos + FRAME <= len(iq):
            frames.run(frames=1)
            audio = frames.audio[-1]
            peak = float(np.max(np.abs(audio))) if audio.size else 0.0
            out.append(
                (
                    frames.pos / RATE,
                    peak,
                    win._squelch_state(True),
                    self.panel._tone_heard.text(),
                )
            )
        return out

    def test_the_squelch_tail_is_never_heard(self):
        self.panel.set_dcs_squelch(23)
        iq, dropped = ending_transmission(23)
        frames = self.listen(iq)
        turn_off_at = 2.0
        drop_at = dropped / RATE
        heard = [peak for end, peak, _s, _t in frames if end <= turn_off_at]
        self.assertGreater(max(heard), 0.1)  # (the voice played)
        # Shut within 100 ms of the turn-off code's start: TONE, while the
        # carrier is still there...
        shut = next(
            end
            for end, _p, state, _t in frames
            if end > turn_off_at and state == "tone"
        )
        self.assertLessEqual(shut - turn_off_at, 0.10 + FRAME_S)
        # ...and nothing at all once it has dropped (before: its noise
        # burst played for about 0.2 s).
        tail = [peak for end, peak, _s, _t in frames if end > drop_at]
        self.assertLess(max(tail), 1e-3)

    def test_tone_heard_follows(self):
        iq, _dropped = ending_transmission(23)
        frames = self.listen(iq)
        during = [text for end, _p, _s, text in frames if 1.0 < end <= 2.0]
        self.assertTrue(all(text == "DCS 023 N" for text in during), during)
        after = [text for end, _p, _s, text in frames if end > 2.1]
        self.assertTrue(after)
        # The code has gone: None at once (never a CTCSS tone near 134 Hz).
        self.assertTrue(all(text == "None" for text in after), after)

    def test_a_ctcss_squelch_near_134_hz_doesnt_open_on_it(self):
        """The turn-off code is between the 131.8 and 136.5 Hz CTCSS tones:
        a squelch set to either still doesn't open on a DCS station."""
        for tone in (131.8, 136.5):
            self.panel.set_tone_squelch_hz(tone)
            iq, _dropped = ending_transmission(65)
            frames = self.listen(iq)
            self.assertLess(max(peak for _e, peak, _s, _t in frames), 1e-3, tone)

    def test_without_its_turn_off_code_as_before(self):
        """A radio that sends none: the squelch closes once the code has
        been missing a while, as before."""
        self.panel.set_dcs_squelch(23)
        iq, dropped = ending_transmission(23, turn_off=False)
        frames = self.listen(iq)
        drop_at = dropped / RATE
        late = [end for end, peak, _s, _t in frames if end > drop_at and peak > 1e-3]
        self.assertTrue(late)  # (its tail is still heard: nothing to go by)
        self.assertLess(max(late) - drop_at, 0.45)


def _demo_dcs_signal():
    """(seed, (period, on)) of the demo's DCS repeater."""
    from sdr_module.gui.device_dialog import _DEMO_SIGNALS

    for seed, (freq, kind, _level, duty) in enumerate(_DEMO_SIGNALS):
        if kind == "dcs":
            return seed, duty, freq
    raise AssertionError("no DCS repeater in the demo")


def _first_end_s():
    """When the demo repeater's first transmission ends (its carrier drops),
    in stream seconds."""
    seed, (period, on), _freq = _demo_dcs_signal()
    shift = (seed * 1.7) % period
    end = on - shift
    while end < 1.0:
        end += period
    return end


class TestTheDemoRepeaterSendsIt(unittest.TestCase):
    """The demo device's 70 cm repeater ends each transmission with the
    turn-off code: 180 ms of 134.4 Hz at the code's deviation, then the
    carrier drops."""

    def demodulated(self, start_s, seconds, rate=2.4e6):
        from sdr_module.gui import device_dialog
        from sdr_module.gui.device_dialog import MockDevice

        _seed, _duty, freq = _demo_dcs_signal()
        with mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9):
            device = MockDevice()
            device.set_sample_rate(rate)
            device.set_frequency(freq - 100e3)
            device.start_rx()
            device._stream_pos = int(start_s * rate)
            iq = np.concatenate(
                [device.read_samples(int(rate / 30)) for _ in range(int(seconds * 30))]
            )
        n = np.arange(len(iq))
        x = iq * np.exp(-2j * np.pi * 100e3 * n / rate)
        factor = int(rate // AUDIO_RATE)
        x = x[: len(x) // factor * factor].reshape(-1, factor).mean(axis=1)
        audio = np.angle(x[1:] * np.conj(x[:-1])) * AUDIO_RATE / (2 * np.pi) / 5e3
        level = np.abs(x[1:])
        return audio, level

    def test_the_last_180_ms_are_the_turn_off_code(self):
        from sdr_module.dsp.dcs import DcsDetector

        end = _first_end_s()
        start = end - 1.2
        audio, level = self.demodulated(start, 1.6)
        t = start + np.arange(len(audio)) / AUDIO_RATE
        tail = (t > end - 0.17) & (t < end - 0.01)
        voice = (t > end - 0.8) & (t < end - 0.3)
        # 134.4 Hz at the code's deviation (15 % of 5 kHz), nothing else.
        self.assertAlmostEqual(
            tone_level(audio[tail], 134.4, AUDIO_RATE), 0.15, delta=0.01
        )
        self.assertLess(tone_level(audio[voice], 134.4, AUDIO_RATE), 0.01)
        # In the DCS band (up to 300 Hz) there is nothing besides it: the
        # code and the voice have stopped.
        spectrum = np.fft.rfft(audio[tail])
        f = np.fft.rfftfreq(int(np.count_nonzero(tail)), 1 / AUDIO_RATE)
        spectrum[(f < 20) | (f > 300)] = 0
        low = np.fft.irfft(spectrum, int(np.count_nonzero(tail)))
        k = np.arange(low.size) / AUDIO_RATE
        basis = np.stack((np.cos(2 * np.pi * 134.4 * k), np.sin(2 * np.pi * 134.4 * k)))
        fit = np.linalg.lstsq(basis.T, low, rcond=None)[0] @ basis
        rest = low - fit
        self.assertLess(float(np.sqrt(np.mean(rest[200:-200] ** 2))), 0.015)
        # The carrier then drops.
        self.assertLess(np.median(level[t > end + 0.01]), 0.2 * np.median(level[voice]))
        # A receiver hears the code, then lets go within 100 ms of the tone.
        detector = DcsDetector(AUDIO_RATE)
        closed = None
        opened = False
        for pos in range(0, len(audio), 480):
            detector.process(audio[pos : pos + 480])
            if detector.is_present(23, False):
                opened = True
            elif opened and closed is None:
                closed = start + (pos + 480) / AUDIO_RATE
        self.assertTrue(opened)
        self.assertIsNotNone(closed)
        self.assertLessEqual(closed - (end - 0.18), 0.10)
        self.assertEqual(detector.turn_offs, 1)

    def test_the_rest_of_the_transmission_is_as_before(self):
        """Up to its turn-off code the repeater sends its loop as before
        (the voice and the code, unchanged), and the phase runs on into the
        tone without a jump."""
        from sdr_module.gui.device_dialog import _dcs_modulation, _loops

        seed, duty, _freq = _demo_dcs_signal()
        end = _first_end_s()
        rate = 2.4e6
        loop = _loops("dcs", seed, rate)
        pos, n = int((end - 1.0) * rate), int(0.5 * rate)
        before = _dcs_modulation(loop, duty, seed, rate, pos, n)
        self.assertTrue(np.array_equal(before, loop.read(pos, n)))
        pos, n = int((end - 0.3) * rate), int(0.25 * rate)
        phase = _dcs_modulation(loop, duty, seed, rate, pos, n).astype(np.float64)
        steps = np.abs(np.diff(phase))
        # (No step larger than the loop's own steepest, its voice and code,
        # and the tone's: a jump would be far larger.)
        own = np.abs(np.diff(loop.read(pos, n).astype(np.float64)))
        tone = 2 * np.pi * 750.0 / rate
        self.assertLess(float(np.max(steps)), float(np.max(own)) + 1.01 * tone)
        # Read in pieces, the same phase.
        pieces = np.concatenate(
            [
                _dcs_modulation(loop, duty, seed, rate, pos + k, min(7919, n - k))
                for k in range(0, n, 7919)
            ]
        )
        self.assertTrue(np.allclose(pieces, phase, atol=1e-4))

    def test_reading_stays_cheap(self):
        """The turn-off code is a table read like the loops: reading the
        demo around a transmission's end costs about what any other read
        does."""
        import time

        from sdr_module.gui import device_dialog
        from sdr_module.gui.device_dialog import MockDevice

        _seed, _duty, freq = _demo_dcs_signal()
        end = _first_end_s()
        rate = 2.4e6
        costs = {}
        with mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9):
            device = MockDevice()
            device.set_sample_rate(rate)
            device.set_frequency(freq)
            device.start_rx()
            for name, start in (("voice", end - 1.0), ("turn-off", end - 0.19)):
                device._stream_pos = int(start * rate)
                device.read_samples(int(rate / 30))  # (warm)
                device._stream_pos = int(start * rate)
                best = []
                for _ in range(3):
                    device._stream_pos = int(start * rate)
                    t0 = time.perf_counter()
                    for _ in range(5):
                        device.read_samples(int(rate / 30))
                    best.append(time.perf_counter() - t0)
                costs[name] = min(best)
        self.assertLess(costs["turn-off"], 2.0 * costs["voice"])


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheDemoRepeaterInTheWindow(_WindowTestCase):
    """Listening to the demo repeater with its code set: no squelch tail at
    the end of a transmission."""

    demo = True

    def test_no_tail_at_the_end_of_a_transmission(self):
        from sdr_module.gui import device_dialog

        win = self.win
        pretend_audio_output(win)
        played = []
        win._audio.write = lambda audio: played.append(np.asarray(audio, float).copy())
        win._stop_acquisition()
        win._device.start_rx()
        win.set_frequency(442.1e6)
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        win._control_panel.set_dcs_squelch(23)
        rate = win._device_sample_rate()
        end = _first_end_s()
        start = end - 1.5
        win._device._stream_pos = int(start * rate)
        heard = win._control_panel._tone_heard
        frames = []
        with mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9):
            for _ in range(int(2.0 * 30)):
                win._pacer.reset()
                played.clear()
                block = win._device.read_samples(int(rate / 30))
                at = win._device._stream_pos / rate
                win._process_block(block)
                peak = max(
                    (float(np.max(np.abs(a))) for a in played if a.size), default=0.0
                )
                frames.append((at, peak, heard.text()))
        self.assertGreater(max(p for at, p, _t in frames if at < end - 0.3), 0.05)
        quiet_from = end - 0.18 + 0.10 + 1 / 30  # (within 100 ms, to a frame)
        late = [p for at, p, _t in frames if at > quiet_from]
        self.assertLess(max(late), 1e-3)
        self.assertEqual(frames[-1][2], "None")


if __name__ == "__main__":
    unittest.main()

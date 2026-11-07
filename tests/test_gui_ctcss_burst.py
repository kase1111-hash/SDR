#!/usr/bin/env python3
"""
The CTCSS reverse burst in the window and on the demo device (M-Q2).

A CTCSS transmission that ends with a reverse burst (the tone's phase
shifted 120-180 degrees for 150 ms or more, the carrier still up) closes
the tone squelch within 100 ms of the burst's start, so the squelch tail
(the noise as the carrier drops) is never heard; Tone heard says None as
soon as the tone has been let go. Before, the squelch stayed open until the
noise squelch closed, and the tail was heard (about 0.33 s of noise). The
demo's 2 m repeater (145.290 MHz, 100.0 Hz) sends one at the end of each
transmission.
"""

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_ctcss_reverse_burst import tone_with_burst, voice  # noqa: E402
from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402
from tests.test_gui_squelch import (  # noqa: E402
    FRAME,
    RATE,
    Frames,
    fm,
    noise,
    pretend_audio_output,
)

AUDIO_RATE = 48000.0
FRAME_S = FRAME / RATE


def ending_transmission(tone=100.0, seconds=2.0, burst_s=0.15, after_s=0.6):
    """Narrow FM (5 kHz) voice with a CTCSS ``tone`` for ``seconds`` (the
    voice up to the end), its reverse burst (180 degrees for ``burst_s``;
    0: none), then the carrier gone (noise only), at 2.4 MS/s; and the
    sample where the carrier drops."""
    audio = tone_with_burst(tone, seconds, 180.0, burst_s)
    audio[: int(seconds * AUDIO_RATE)] += voice(seconds, seed=7)
    held = np.repeat(audio, int(RATE / AUDIO_RATE))
    on = len(held)
    total = on + int(after_s * RATE)
    total -= total % FRAME
    iq = np.concatenate((fm(held, 5e3, -50.0), np.zeros(total - on, np.complex64)))
    return iq + noise(total, -95.0), on


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheBurstInTheReceiver(_WindowTestCase):
    """A 2 m channel (FM, 5 kHz) with a 100.0 Hz tone squelch, through the
    real receiver chain."""

    def setUp(self):
        super().setUp()
        pretend_audio_output(self.win)
        self.panel = self.win._control_panel
        self.win.set_frequency(146.52e6)
        self.win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        self.panel.set_squelch_db(-90)
        self.panel.set_tone_squelch_hz(100.0)

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
        iq, dropped = ending_transmission()
        frames = self.listen(iq)
        burst_at, drop_at = 2.0, dropped / RATE
        heard = [peak for end, peak, _s, _t in frames if end <= burst_at]
        self.assertGreater(max(heard), 0.1)  # (the voice played)
        # Shut within 100 ms of the burst's start: TONE, while the carrier
        # is still there...
        shut = next(
            end for end, _p, state, _t in frames if end > burst_at and state == "tone"
        )
        self.assertLessEqual(shut - burst_at, 0.10 + FRAME_S)
        # ...and nothing at all once it has dropped (before: its noise
        # burst played for about a third of a second).
        tail = [peak for end, peak, _s, _t in frames if end > drop_at]
        self.assertLess(max(tail), 1e-3)
        # Tone heard: the tone until the burst, None at once after it.
        during = [text for end, _p, _s, text in frames if 1.0 < end <= 2.0]
        self.assertTrue(all(text == "100.0 Hz (1Z)" for text in during), during)
        after = [text for end, _p, _s, text in frames if end > burst_at + 0.15]
        self.assertTrue(after)
        self.assertTrue(all(text == "None" for text in after), after)

    def test_without_a_burst_as_before(self):
        """A radio that sends none: the tail is heard as before (the noise
        squelch closes it)."""
        iq, dropped = ending_transmission(burst_s=0.0)
        frames = self.listen(iq)
        drop_at = dropped / RATE
        late = [end for end, peak, _s, _t in frames if end > drop_at and peak > 1e-3]
        self.assertTrue(late)  # (nothing to go by)
        self.assertLess(max(late) - drop_at, 0.45)


def _demo_repeater():
    """(seed, (period, on), frequency) of the demo's 2 m repeater."""
    from sdr_module.gui.device_dialog import _DEMO_SIGNALS

    for seed, (freq, kind, _level, duty) in enumerate(_DEMO_SIGNALS):
        if kind == "rpt":
            return seed, duty, freq
    raise AssertionError("no 2 m repeater in the demo")


def _first_end_s():
    """When the demo repeater's first transmission (after a second) ends,
    its carrier dropping, in stream seconds."""
    seed, (period, on), _freq = _demo_repeater()
    end = on - (seed * 1.7) % period
    while end < 1.0:
        end += period
    return end


class TestTheDemoRepeaterSendsIt(unittest.TestCase):
    """The demo device's 2 m repeater ends each transmission with a reverse
    burst: 150 ms of its 100.0 Hz tone shifted by 180 degrees, then the
    carrier drops."""

    def demodulated(self, start_s, seconds, rate=2.4e6):
        from sdr_module.gui import device_dialog
        from sdr_module.gui.device_dialog import MockDevice

        _seed, _duty, freq = _demo_repeater()
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
        return audio, np.abs(x[1:])

    @staticmethod
    def phasor(audio, hz=100.0):
        """The tone's phasor in ``audio`` (0-300 Hz, at AUDIO_RATE)."""
        k = np.arange(len(audio)) / AUDIO_RATE
        return np.sum(audio * np.exp(-2j * np.pi * hz * k)) * 2 / len(audio)

    def test_the_last_150_ms_are_the_tone_shifted(self):
        from sdr_module.dsp.filters import CTCSSDetector

        end = _first_end_s()
        start = end - 1.2
        audio, level = self.demodulated(start, 1.6)
        t = start + np.arange(len(audio)) / AUDIO_RATE

        def tone(lo, hi):
            # (The tone's phase at a common time: lo..hi seen from t = 0.)
            part = (t >= lo) & (t < hi)
            return self.phasor(audio[part]) * np.exp(-2j * np.pi * 100.0 * lo)

        before = tone(end - 0.40, end - 0.20)
        burst = tone(end - 0.12, end - 0.01)
        # The tone at 12 % of the deviation, before and in the burst...
        self.assertAlmostEqual(abs(burst), 0.12, delta=0.012)
        self.assertAlmostEqual(abs(before), 0.12, delta=0.03)  # (voice too)
        # ...shifted by 180 degrees.
        angle = np.degrees(np.angle(burst * np.conj(before)))
        self.assertGreater(abs(angle), 170.0)
        # The voice has stopped: in the CTCSS band (up to 300 Hz) there is
        # nothing besides the tone.
        part = (t > end - 0.12) & (t < end - 0.01)
        size = int(np.count_nonzero(part))
        spectrum = np.fft.rfft(audio[part])
        f = np.fft.rfftfreq(size, 1 / AUDIO_RATE)
        spectrum[(f < 20) | (f > 300)] = 0
        low = np.fft.irfft(spectrum, size)
        k = np.arange(size) / AUDIO_RATE
        basis = np.stack((np.cos(2 * np.pi * 100.0 * k), np.sin(2 * np.pi * 100.0 * k)))
        rest = low - np.linalg.lstsq(basis.T, low, rcond=None)[0] @ basis
        self.assertLess(float(np.sqrt(np.mean(rest[200:-200] ** 2))), 0.015)
        # The carrier then drops.
        voice = (t > end - 0.8) & (t < end - 0.3)
        self.assertLess(np.median(level[t > end + 0.01]), 0.2 * np.median(level[voice]))
        # A receiver lets the tone go within 100 ms of the burst.
        detector = CTCSSDetector(AUDIO_RATE)
        closed = None
        opened = False
        for pos in range(0, len(audio), 480):
            detector.process(audio[pos : pos + 480])
            if detector.is_present(100.0):
                opened = True
            elif opened and closed is None:
                closed = start + (pos + 480) / AUDIO_RATE
        self.assertTrue(opened)
        self.assertIsNotNone(closed)
        self.assertLessEqual(closed - (end - 0.15), 0.10)
        self.assertEqual(detector.bursts, 1)

    def test_the_rest_of_the_transmission_is_as_before(self):
        """Up to its burst the repeater sends its loop as before, and the
        phase runs on into the burst without a jump."""
        from sdr_module.gui.device_dialog import _loops, _rpt_modulation

        seed, duty, _freq = _demo_repeater()
        end = _first_end_s()
        rate = 2.4e6
        loop = _loops("rpt", seed, rate)
        pos, n = int((end - 1.0) * rate), int(0.5 * rate)
        before = _rpt_modulation(loop, duty, seed, rate, pos, n)
        self.assertTrue(np.array_equal(before, loop.read(pos, n)))
        pos, n = int((end - 0.3) * rate), int(0.28 * rate)
        phase = _rpt_modulation(loop, duty, seed, rate, pos, n).astype(np.float64)
        steps = np.abs(np.diff(phase))
        own = np.abs(np.diff(loop.read(pos, n).astype(np.float64)))
        tone = 2 * np.pi * 600.0 / rate
        self.assertLess(float(np.max(steps)), float(np.max(own)) + 2.01 * tone)
        # Read in pieces, the same phase.
        pieces = np.concatenate(
            [
                _rpt_modulation(loop, duty, seed, rate, pos + k, min(7919, n - k))
                for k in range(0, n, 7919)
            ]
        )
        self.assertTrue(np.allclose(pieces, phase, atol=1e-4))

    def test_reading_stays_cheap(self):
        """The burst is a table read like the loops: reading the demo
        during it costs about what the same read without it does."""
        import time

        from sdr_module.gui import device_dialog
        from sdr_module.gui.device_dialog import MockDevice

        _seed, _duty, freq = _demo_repeater()
        start = _first_end_s() - 0.16
        rate = 2.4e6

        def plain(loop, _duty, _seed, _rate, pos, n):
            return loop.read(pos, n)

        def cost():
            device._stream_pos = int(start * rate)
            device.read_samples(int(rate / 30))  # (warm)
            best = []
            for _ in range(3):
                device._stream_pos = int(start * rate)
                t0 = time.perf_counter()
                for _ in range(4):
                    device.read_samples(int(rate / 30))
                best.append(time.perf_counter() - t0)
            return min(best)

        with mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9):
            device = MockDevice()
            device.set_sample_rate(rate)
            device.set_frequency(freq)
            device.start_rx()
            with_burst = cost()
            with mock.patch.object(device_dialog, "_rpt_modulation", plain):
                without = cost()
        self.assertLess(with_burst, 2.0 * without)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheDemoRepeaterInTheWindow(_WindowTestCase):
    """Listening to the demo repeater with its tone set: no squelch tail at
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
        win.set_frequency(145.29e6)
        win._set_demod_mode("FM", "5 kHz", "12.5 kHz")
        win._control_panel.set_tone_squelch_hz(100.0)
        rate = win._device_sample_rate()
        end = _first_end_s()
        win._device._stream_pos = int((end - 1.5) * rate)
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
        quiet_from = end - 0.15 + 0.10 + 1 / 30  # (within 100 ms, to a frame)
        late = [p for at, p, _t in frames if at > quiet_from]
        self.assertLess(max(late), 1e-3)
        self.assertEqual(frames[-1][2], "None")


if __name__ == "__main__":
    unittest.main()

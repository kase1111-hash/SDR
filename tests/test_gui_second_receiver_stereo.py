"""M-R1: the Second Receiver plays broadcast FM in stereo.

It played mono: its speaker opened one channel and its receiver chain's
``stereo`` was never set. Now broadcast FM there goes through the main
window's own stereo path (``_ReceiverChain`` with ``stereo``, the chain's
``StereoPath``), its speaker opens two channels (mono plays on both), it
has a Stereo switch of its own (saved, on by default) and a STEREO lamp
that says, as the main window's badge does, whether stereo plays and why
not. Everything else hears the same mono audio as before.

The stereo signals come from the independent synthesizer of the standard
multiplex in ``tests/test_fm_stereo.py`` and from the demo's 97.1 MHz
station (its 3.7 kHz tone on the left only).
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from sdr_module.dsp.fm_stereo import StereoDecoder  # noqa: E402
from tests.test_fm_stereo import (  # noqa: E402
    _db,
    _fm,
    _multiplex,
    _pre_emphasis,
    _quick_rise,
    _tone,
)
from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_second_receiver import _FakeSink, _SecondCase  # noqa: E402

RATE = 2.4e6
BLOCK = 80_000  # one display frame at 2.4 MS/s
AUDIO = 48000.0


def _station(seconds=0.6, left_hz=1000.0, right_hz=3000.0, rate=RATE):
    """A stereo FM broadcast station at 0 Hz: one tone on each side."""
    n = int(seconds * rate)
    t = np.arange(n) / rate
    left = _pre_emphasis(0.4 * np.sin(2 * np.pi * left_hz * t), rate)
    right = _pre_emphasis(0.4 * np.sin(2 * np.pi * right_hz * t), rate)
    return _fm(_multiplex(left, right, rate), rate)


def _both(frames):
    """The audio of ``frames`` as (n, 2): mono blocks on both sides."""
    out = []
    for frame in frames:
        audio = frame.audio
        if audio is None or not len(audio):
            continue
        out.append(audio if audio.ndim == 2 else np.stack((audio, audio), axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2))


class _DspCase(unittest.TestCase):
    def setUp(self):
        try:
            import PyQt6  # noqa: F401
        except ImportError:  # pragma: no cover - environment-dependent
            self.skipTest("PyQt6 not available")
        patcher = _quick_rise()
        patcher.start()
        self.addCleanup(patcher.stop)

    def dsp(self, stereo=True, deviation=75e3, bandwidth=200e3):
        from sdr_module.gui.second_receiver import ReceiverDsp

        dsp = ReceiverDsp()
        dsp.rate = RATE
        dsp.mode = "FM"
        dsp.bandwidth_hz = bandwidth
        dsp.deviation_hz = deviation
        dsp.stereo = stereo
        return dsp

    @staticmethod
    def run_blocks(dsp, iq, play=True):
        return [
            dsp.process(iq[start : start + BLOCK], play)
            for start in range(0, len(iq), BLOCK)
        ]


class TestTheSignalPathPlaysStereo(_DspCase):
    def test_left_and_right_come_apart(self):
        dsp = self.dsp()
        frames = self.run_blocks(dsp, _station())
        both = _both(frames)
        self.assertTrue(any(f.audio is not None and f.audio.ndim == 2 for f in frames))
        tail = both[int(0.3 * AUDIO) :]
        left, right = tail[:, 0], tail[:, 1]
        self.assertAlmostEqual(_tone(left, 1000, AUDIO), 0.36, delta=0.02)
        self.assertAlmostEqual(_tone(right, 3000, AUDIO), 0.36, delta=0.02)
        # Each tone on its own side, 40 dB and more (it played on both).
        self.assertGreater(_db(_tone(left, 1000, AUDIO), _tone(right, 1000, AUDIO)), 40)
        self.assertGreater(_db(_tone(right, 3000, AUDIO), _tone(left, 3000, AUDIO)), 40)
        text, tone, tip = frames[-1].stereo
        self.assertEqual((text, tone), ("STEREO", "success"))
        self.assertIn("fully apart", tip)

    def test_its_mono_is_the_same_audio_as_before(self):
        # (L+R)/2 is exactly what the receiver played before stereo, and
        # with the Stereo switch off it plays that, one channel.
        iq = _station(0.4)
        mono = _both(self.run_blocks(self.dsp(stereo=False), iq))
        frames = self.run_blocks(self.dsp(stereo=False), iq)
        self.assertTrue(all(f.audio is None or f.audio.ndim == 1 for f in frames))
        frames = self.run_blocks(self.dsp(stereo=True), iq)
        self.assertTrue(any(f.audio is not None and f.audio.ndim == 2 for f in frames))
        both = _both(frames)
        self.assertEqual(both.shape, mono.shape)
        np.testing.assert_allclose(both.mean(axis=1), mono[:, 0], atol=1e-5)

    def test_only_broadcast_fm_is_stereo(self):
        # Narrow FM (a repeater) and AM never decode stereo, and show no lamp.
        calls = []
        real = StereoDecoder.process

        def counted(decoder, mpx):
            calls.append(len(mpx))
            return real(decoder, mpx)

        with mock.patch.object(StereoDecoder, "process", counted):
            dsp = self.dsp(deviation=5e3, bandwidth=25e3)
            frames = self.run_blocks(dsp, _station(0.2))
            dsp = self.dsp(bandwidth=10e3)
            dsp.mode = "AM"
            frames += self.run_blocks(dsp, _station(0.2))
        self.assertEqual(calls, [])
        self.assertTrue(all(f.stereo == ("", None, "") for f in frames))

    def test_the_lamp_says_why_it_plays_mono(self):
        dsp = self.dsp(stereo=False)
        text, tone, tip = self.run_blocks(dsp, _station(0.2))[-1].stereo
        self.assertEqual(text, "MONO")
        self.assertIsNone(tone)
        self.assertIn("Stereo switch is off", tip)
        # A filter too narrow for the stereo signal.
        dsp = self.dsp(bandwidth=100e3)
        frames = self.run_blocks(dsp, _station(0.2))
        self.assertTrue(all(f.audio is None or f.audio.ndim == 1 for f in frames))
        text, _tone_, tip = frames[-1].stereo
        self.assertEqual(text, "MONO")
        self.assertIn("Filter is too narrow", tip)
        # A station without a stereo pilot (a mono broadcast).
        n = int(0.5 * RATE)
        t = np.arange(n) / RATE
        mono = _fm(0.5 * np.sin(2 * np.pi * 1000 * t), RATE)
        text, _tone_, tip = self.run_blocks(self.dsp(), mono)[-1].stereo
        self.assertEqual(text, "MONO")
        self.assertIn("no stereo pilot", tip)

    def test_the_lamp_shows_with_the_speaker_off_while_someone_listens(self):
        # As a radio's lamp with the volume down (the main window's Audio
        # switch off): stereo is decoded for the lamp, nothing plays.
        dsp = self.dsp()
        dsp.heard = True
        frames = self.run_blocks(dsp, _station(), play=False)
        self.assertTrue(all(f.audio is None for f in frames))
        self.assertEqual(frames[-1].stereo[0], "STEREO")

    def test_no_stereo_is_decoded_while_nobody_listens(self):
        # Listen plays the main window only (or the audio is paused): the
        # lamp says nothing, and no stereo is decoded for it.
        calls = []
        real = StereoDecoder.process

        def counted(decoder, mpx):
            calls.append(len(mpx))
            return real(decoder, mpx)

        dsp = self.dsp()
        dsp.heard = False
        with mock.patch.object(StereoDecoder, "process", counted):
            frames = self.run_blocks(dsp, _station(0.3), play=False)
        self.assertEqual(calls, [])
        self.assertTrue(all(f.stereo == ("", None, "") for f in frames))

    def test_switching_stereo_off_fades_to_mono_without_a_click(self):
        dsp = self.dsp()
        iq = _station(0.8)
        frames = self.run_blocks(dsp, iq[: int(0.4 * RATE)])
        self.assertEqual(frames[-1].audio.ndim, 2)
        dsp.stereo = False
        frames += self.run_blocks(dsp, iq[int(0.4 * RATE) :])
        both = _both(frames)
        # No step anywhere: the largest sample-to-sample change stays that
        # of the tones themselves (3 kHz at 0.4: about 0.16 a sample).
        self.assertLess(float(np.max(np.abs(np.diff(both, axis=0)))), 0.25)
        self.assertEqual(frames[-1].audio.ndim, 1)

    def test_the_speaker_stopping_fades_out_in_stereo(self):
        # (Check: the block that fades out when the speaker stops, as when
        # Listen moves to the main window, is the stereo audio too; a mono
        # one after stereo would step each side to (L+R)/2 at once.)
        dsp = self.dsp()
        iq = _station(0.6)
        half = int(0.4 * RATE)
        frames = self.run_blocks(dsp, iq[:half])
        self.assertEqual(frames[-1].audio.ndim, 2)
        last = frames[-1].audio[-1]
        stopped = self.run_blocks(dsp, iq[half:], play=False)
        fade = stopped[0].audio
        self.assertEqual(fade.ndim, 2)
        self.assertTrue(all(f.audio is None for f in stopped[1:]))
        # Each side goes on from where it was (one sample of the tones).
        self.assertLess(float(np.max(np.abs(fade[0] - last))), 0.2)
        self.assertLess(float(np.max(np.abs(fade[-1]))), 0.05)  # (faded out)


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheSpeakerOpensInStereo(_SecondCase):
    def test_its_audio_sink_asks_for_two_channels(self):
        from sdr_module.gui.audio_sink import AudioSink

        sink = self.rx._audio
        self.assertIsInstance(sink, AudioSink)
        self.assertEqual(sink._wanted_channels, 2)


class _StereoWindowCase(_SecondCase):
    """The second receiver on a demo FM station, Listen: This receiver."""

    def setUp(self):
        super().setUp()
        from sdr_module.gui import device_dialog

        for target, name, value in (
            (device_dialog, "_MAX_LAG_S", 1e9),
            (StereoDecoder, "RISE_S", 0.05),
        ):
            patcher = mock.patch.object(target, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        main_sink = _FakeSink()
        main_sink.open = True
        self.win._audio = main_sink
        self.win._audio_enabled = True
        self.win._audio_problem = None
        self.speaker = self.sink()
        rx = self.rx
        self.assertTrue(rx.start())
        rx._frame_timer.stop()
        rx.device._rng = np.random.default_rng(3)
        rx.set_listen("second")
        self.listen()

    def listen(self, freq=97.1e6, mode=("FM", "75 kHz", "200 kHz")):
        rx = self.rx
        rx.tune(freq, recenter=True)
        rx.set_mode(*mode)
        rx._sync_audio_output()

    def stereo_blocks(self):
        return [b for b in self.speaker.blocks if b.ndim == 2]

    def heard(self):
        both = [
            b if b.ndim == 2 else np.stack((b, b), axis=1) for b in self.speaker.blocks
        ]
        return np.concatenate(both) if both else np.zeros((0, 2))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheWindowPlaysStereo(_StereoWindowCase):
    def test_a_tone_on_the_left_is_heard_on_the_left(self):
        self.feed(20)
        self.assertTrue(self.stereo_blocks())
        tail = self.heard()[-int(0.25 * AUDIO) :]
        left = _tone(tail[:, 0], 3.7e3, AUDIO)
        right = _tone(tail[:, 1], 3.7e3, AUDIO)
        self.assertGreater(left, 0.1)
        self.assertGreater(_db(left, right), 35)

    def test_the_stereo_lamp(self):
        rx = self.rx
        badge = rx._stereo_badge
        self.assertFalse(badge.isVisible())  # (nothing heard yet)
        self.feed(14)
        self.assertTrue(badge.isVisible())
        self.assertEqual(badge.text(), "STEREO")
        self.assertEqual(badge.property("tone"), "success")
        self.assertIn("pilot is locked", badge.toolTip())
        self.assertTrue(badge.accessibleName())
        # Its switch off: MONO, and why.
        rx._stereo_check.setChecked(False)
        self.feed(3)
        self.assertEqual(badge.text(), "MONO")
        self.assertIn("Stereo switch is off", badge.toolTip())
        self.assertTrue(all(b.ndim == 1 for b in self.speaker.blocks[-2:]))
        rx._stereo_check.setChecked(True)
        self.feed(10)
        self.assertEqual(badge.text(), "STEREO")
        # Narrow FM (the demo repeater): no lamp.
        self.listen(145.29e6, ("FM", "5 kHz", "25 kHz"))
        self.feed(3)
        self.assertFalse(badge.isVisible())

    def test_no_lamp_and_no_stereo_while_only_the_main_window_plays(self):
        rx = self.rx
        self.feed(14)
        self.assertTrue(rx._stereo_badge.isVisible())
        calls = []
        real = StereoDecoder.process

        def counted(decoder, mpx):
            calls.append(len(mpx))
            return real(decoder, mpx)

        rx.set_listen("main")
        with mock.patch.object(StereoDecoder, "process", counted):
            self.feed(4)
        self.assertEqual(calls, [])
        self.assertFalse(rx._stereo_badge.isVisible())

    def test_the_lamp_shows_with_the_main_windows_audio_off(self):
        # The app's Audio switch off: nothing plays, the lamp still says
        # what is received (as the main window's badge does).
        rx = self.rx
        self.win._audio_enabled = False
        rx._update_status()
        self.assertFalse(self.speaker.open)
        self.feed(14)
        self.assertTrue(rx._stereo_badge.isVisible())
        self.assertEqual(rx._stereo_badge.text(), "STEREO")

    def test_the_main_windows_listen_rules_stay(self):
        # Only this receiver: the main window is muted (and decodes no
        # stereo for its own badge); Both: each plays its own.
        self.assertTrue(self.host.mutes_main())
        self.rx.set_listen("both")
        self.assertFalse(self.host.mutes_main())
        self.feed(14)
        self.assertTrue(self.stereo_blocks())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheStereoSwitch(_StereoWindowCase):
    def test_on_by_default_with_a_tooltip_and_saved(self):
        rx = self.rx
        check = rx._stereo_check
        self.assertTrue(check.isChecked())
        self.assertIn("mono", check.toolTip())
        self.assertTrue(rx._dsp.stereo)
        check.setChecked(False)
        self.assertFalse(rx._dsp.stereo)
        self.assertIs(self.store.values["device/second_receiver/stereo"], False)
        check.setChecked(True)
        self.assertIs(self.store.values["device/second_receiver/stereo"], True)

    def test_it_comes_back_and_is_the_windows_own(self):
        from sdr_module.gui.second_receiver import SecondReceiverWindow

        self.rx._stereo_check.setChecked(False)
        # (The main window's Radio > Audio > Stereo is its own.)
        self.assertTrue(self.win._stereo_enabled)
        again = self.extra(SecondReceiverWindow(self.win))
        self.assertFalse(again._stereo_check.isChecked())
        self.assertFalse(again._dsp.stereo)


if __name__ == "__main__":
    unittest.main()

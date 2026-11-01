#!/usr/bin/env python3
"""
Broadcast FM stereo in the window (K-S1): the audio join, the speaker
output and Record Audio in two channels, Radio > Audio > Stereo, the
STEREO badge in the receiver bar and the radio tuner's STEREO lamp.
"""

import os
import shutil
import tempfile
import unittest
import wave
from unittest import mock

import numpy as np

from sdr_module.dsp.fm_stereo import StereoDecoder
from sdr_module.gui import audio_sink as audio_sink_module
from sdr_module.gui.audio_chain import AudioJoin
from sdr_module.gui.audio_sink import AudioSink
from tests.test_final_window import _IniSettings
from tests.test_gui_audio_sink import Clock, FakeDevice, FakeOutput
from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase
from tests.test_gui_squelch import pretend_audio_output

RATE = 48000


def _stereo(n, left=0.5, right=-0.25):
    out = np.empty((n, 2), dtype=np.float32)
    out[:, 0], out[:, 1] = left, right
    return out


class TestAudioJoinStereo(unittest.TestCase):
    """The fades and crossfades keep each side free of steps."""

    def test_fade_in_ramps_both_sides(self):
        join = AudioJoin(RATE)
        out = join.fade_in(_stereo(1000))
        self.assertEqual(out.shape, (1000, 2))
        self.assertLess(abs(out[0, 0]), 0.01)
        self.assertLess(abs(out[0, 1]), 0.01)
        np.testing.assert_allclose(out[-1], (0.5, -0.25))
        np.testing.assert_allclose(join.last, (0.5, -0.25))

    def test_a_stereo_block_after_mono_starts_from_the_mono_sample(self):
        join = AudioJoin(RATE)
        join.play(np.full(500, 0.2, dtype=np.float32))
        out = join.join(_stereo(1000))
        np.testing.assert_allclose(out[0], (0.2, 0.2), atol=0.01)
        np.testing.assert_allclose(out[-1], (0.5, -0.25))

    def test_mono_after_stereo_crossfades_each_side_from_its_own_sample(self):
        join = AudioJoin(RATE)
        join.play(_stereo(500))
        out = join.join(np.full(1000, 0.1, dtype=np.float32))
        # (Two channels while it crossfades: one value for both would step.)
        self.assertEqual(out.shape, (1000, 2))
        np.testing.assert_allclose(out[0], (0.5, -0.25), atol=0.02)
        np.testing.assert_allclose(out[-1], (0.1, 0.1))
        self.assertLess(np.max(np.abs(np.diff(out, axis=0))), 0.01)

    def test_release_after_stereo_decays_each_side(self):
        join = AudioJoin(RATE)
        join.play(_stereo(500))
        out = join.release()
        self.assertEqual(out.shape[1], 2)
        np.testing.assert_allclose(out[0], (0.5, -0.25), atol=0.01)
        np.testing.assert_allclose(out[-1], (0.0, 0.0), atol=0.01)
        self.assertFalse(join.playing)
        self.assertEqual(join.release().size, 0)

    def test_fade_out_of_a_stereo_block(self):
        join = AudioJoin(RATE)
        join.play(_stereo(500))
        out = join.fade_out(_stereo(5000))
        self.assertEqual(out.shape, (int(0.005 * RATE), 2))
        np.testing.assert_allclose(out[0], (0.5, -0.25), atol=0.01)
        np.testing.assert_allclose(out[-1], (0.0, 0.0), atol=0.01)

    def test_mono_stays_mono(self):
        join = AudioJoin(RATE)
        out = join.join(np.full(100, 0.3, dtype=np.float32))
        self.assertEqual(out.ndim, 1)
        self.assertIsInstance(join.last, float)
        self.assertEqual(join.release().ndim, 1)


class _StereoSinkCase(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        patcher = mock.patch.object(audio_sink_module.time, "monotonic", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make(self, channels=2, buffer_s=0.25, limit=None):
        """A sink whose output opened with ``channels`` channels."""
        sink = AudioSink(channels=2)
        output = FakeOutput(int(buffer_s * RATE) * 2 * channels)
        device = FakeDevice(output, limit)
        sink._sink, sink._io, sink._sample_rate = output, device, RATE
        sink._channels = channels
        return sink, output, device

    @staticmethod
    def frames(received, channels=2):
        return np.frombuffer(bytes(received), dtype=np.int16).reshape(-1, channels)


class TestAudioSinkStereo(_StereoSinkCase):
    def test_left_and_right_are_interleaved(self):
        sink, _output, device = self.make()
        sink.write(_stereo(4000))
        frames = self.frames(device.received)
        audio = frames[np.any(frames != 0, axis=1)]  # (after the cushion)
        self.assertEqual(len(audio), 4000)
        np.testing.assert_array_equal(audio[0], (16383, -8191))
        self.assertEqual(sink.channels, 2)

    def test_mono_plays_on_both_channels(self):
        sink, _output, device = self.make()
        sink.write(np.full(4000, 0.5, dtype=np.float32))
        frames = self.frames(device.received)
        audio = frames[np.any(frames != 0, axis=1)]
        self.assertEqual(len(audio), 4000)
        np.testing.assert_array_equal(audio[:, 0], audio[:, 1])

    def test_stereo_on_a_mono_output_is_mixed_down(self):
        sink, _output, device = self.make(channels=1)
        sink.write(_stereo(4000, 0.5, -0.1))
        samples = np.frombuffer(bytes(device.received), dtype=np.int16)
        audio = samples[samples != 0]
        self.assertEqual(len(audio), 4000)
        self.assertEqual(int(audio[0]), int(0.2 * 32767))

    def test_the_jitter_buffer_counts_frames(self):
        sink, output, _device = self.make()
        self.assertAlmostEqual(sink.queued_seconds(), 0.0)
        sink.write(_stereo(4800))  # 0.1 s, after a 0.12 s cushion's worth
        self.assertAlmostEqual(sink.queued_seconds(), 0.12, delta=0.002)

    def test_drops_and_pads_keep_left_and_right_in_place(self):
        # The output takes odd amounts and the clocks drift both ways:
        # every frame still arrives whole, left first.
        sink, output, device = self.make(limit=1002)
        for i in range(60):
            sink.write(_stereo(1600, 0.25, -0.5))
            self.clock.now += 1600 / RATE
            output.play(int(1600 * (1.4 if i % 10 < 5 else 0.6)) * 4)
            sink._pump(True)
        self.assertGreater(sink.underruns + sink.overruns, 0)
        frames = self.frames(device.received[: len(device.received) // 4 * 4])
        sound = frames[np.any(frames != 0, axis=1)]
        self.assertGreater(len(sound), 0)
        np.testing.assert_array_equal(np.unique(sound[:, 0]), [8191])
        np.testing.assert_array_equal(np.unique(sound[:, 1]), [-16383])

    def test_muting_mid_frame_keeps_left_and_right_in_step(self):
        # The output took half a frame (the left sample) when the speaker
        # was muted: the right one still follows it, so nothing swaps.
        sink, output, device = self.make(limit=1002)
        sink.write(_stereo(1000, 0.25, -0.5))
        self.assertEqual(len(device.received) % 4, 2)  # (half a frame out)
        sink.set_muted(True)
        sink.set_muted(False)
        output.play(output.queued)
        self.clock.now += 1.0
        for _ in range(20):
            sink.write(_stereo(1600, 0.25, -0.5))
            self.clock.now += 1600 / RATE
            output.play(1600 * 4)
            sink._pump(True)
        frames = self.frames(device.received[: len(device.received) // 4 * 4])
        sound = frames[np.any(frames != 0, axis=1)]
        np.testing.assert_array_equal(np.unique(sound[:, 0]), [8191])
        np.testing.assert_array_equal(np.unique(sound[:, 1]), [-16383])

    def test_a_drop_keeps_the_frame_the_output_has_begun(self):
        # The output holds half a frame (a left sample of the cushion's
        # silence) when a burst overfills the queue: the oldest audio is
        # dropped after the right sample that finishes that frame, so no
        # frame gets one side of each.
        sink, output, device = self.make(limit=1002)
        sink.write(_stereo(1000, 0.25, -0.5))
        self.assertEqual(len(device.received) % 4, 2)  # (half a frame out)
        sink.write(_stereo(24000, 0.25, -0.5))  # (a stall's burst)
        self.assertEqual(sink.overruns, 1)
        for _ in range(400):
            self.clock.now += 0.01
            output.play(4000)
            sink._feed()
        frames = self.frames(device.received[: len(device.received) // 4 * 4])
        sound = frames[np.any(frames != 0, axis=1)]
        self.assertGreater(len(sound), 20000)
        np.testing.assert_array_equal(np.unique(sound[:, 0]), [8191])
        np.testing.assert_array_equal(np.unique(sound[:, 1]), [-16383])

    def test_a_sink_is_mono_unless_asked(self):
        # (The second receiver and Transmit Text's preview play mono.)
        self.assertEqual(AudioSink()._wanted_channels, 1)
        self.assertEqual(AudioSink(channels=2)._wanted_channels, 2)


class _Format:
    """QAudioFormat stand-in that remembers what was asked."""

    class SampleFormat:
        Int16 = 2

    def __init__(self):
        self.rate = self.channels = 0

    def setSampleRate(self, rate):  # noqa: N802 (Qt API)
        self.rate = rate

    def setChannelCount(self, count):  # noqa: N802 (Qt API)
        self.channels = count

    def channelCount(self):  # noqa: N802 (Qt API)
        return self.channels

    def setSampleFormat(self, _fmt):  # noqa: N802 (Qt API)
        pass


class _Device:
    def __init__(self, stereo):
        self.stereo = stereo

    def isNull(self):  # noqa: N802 (Qt API)
        return False

    def isFormatSupported(self, fmt):  # noqa: N802 (Qt API)
        return fmt.channels == 1 or self.stereo

    def id(self):
        return b"fake"

    def description(self):
        return "Fake output"


class _QSink:
    opened = []

    def __init__(self, device, fmt):
        _QSink.opened.append(fmt.channels)
        self.size = 0

    def setVolume(self, value):  # noqa: N802 (Qt API)
        pass

    def setBufferSize(self, size):  # noqa: N802 (Qt API)
        self.size = size

    def bufferSize(self):  # noqa: N802 (Qt API)
        return self.size

    def bytesFree(self):  # noqa: N802 (Qt API)
        return self.size

    def start(self):
        return mock.Mock(write=lambda data: len(data))

    def error(self):
        return audio_sink_module.QAudio.Error.NoError

    def stop(self):
        pass


@unittest.skipUnless(audio_sink_module.HAS_QT_AUDIO, "needs QtMultimedia")
class TestAudioSinkOpens(unittest.TestCase):
    def _open(self, stereo_device, channels=2):
        device = _Device(stereo_device)
        media = mock.Mock()
        media.defaultAudioOutput.return_value = device
        media.audioOutputs.return_value = [device]
        _QSink.opened = []
        with mock.patch.multiple(
            audio_sink_module,
            QMediaDevices=media,
            QAudioSink=_QSink,
            QAudioFormat=_Format,
        ):
            sink = AudioSink(channels=channels)
            self.assertTrue(sink.start(RATE))
        return sink

    def test_two_channels_when_the_output_can(self):
        sink = self._open(True)
        self.assertEqual(_QSink.opened, [2])
        self.assertEqual(sink.channels, 2)
        self.assertEqual(sink._sink.size, int(AudioSink.BUFFER_S * RATE) * 4)

    def test_mono_when_it_cannot(self):
        sink = self._open(False)
        self.assertEqual(_QSink.opened, [1])
        self.assertEqual(sink.channels, 1)

    def test_a_mono_sink_opens_mono(self):
        sink = self._open(True, channels=1)
        self.assertEqual(_QSink.opened, [1])
        self.assertEqual(sink.channels, 1)


class TestAudioTakeStereo(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="stereo-take-")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def _take(self, channels):
        from sdr_module.gui.audio_recording import AudioTake

        return AudioTake(
            os.path.join(self.folder, "take.wav"), 100.1e6, "FM", RATE, channels
        )

    def _read(self, path):
        with wave.open(path, "rb") as wav:
            data = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
            return wav.getnchannels(), data.reshape(-1, wav.getnchannels())

    def test_a_stereo_take_records_both_sides(self):
        take = self._take(2)
        self.assertEqual(take.channels, 2)
        take.write(_stereo(4800), True)
        take.write(np.full(4800, 0.1, dtype=np.float32), True)  # mono: both
        take.silence(0.1)
        take.stop()
        channels, frames = self._read(take.path)
        self.assertEqual(channels, 2)
        self.assertEqual(len(frames), 3 * 4800)
        self.assertAlmostEqual(take.seconds, 0.3)
        middle = frames[2400]
        self.assertEqual(tuple(middle), (16383, -8191))
        np.testing.assert_array_equal(frames[7200], (3276, 3276))

    def test_a_mono_take_mixes_stereo_down(self):
        take = self._take(1)
        take.write(_stereo(4800, 0.5, -0.1), True)
        take.stop()
        channels, frames = self._read(take.path)
        self.assertEqual(channels, 1)
        self.assertEqual(int(frames[2400, 0]), int(0.2 * 32767))


class _StereoWindowCase(_WindowTestCase):
    """The window listening to the demo's 100.1 MHz stereo station, a frame
    at a time (the blend widens to stereo in 50 ms here, not 0.75 s)."""

    FRAME = 80_000  # samples of 2.4 MS/s per block (33 ms)

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
        pretend_audio_output(self.win)
        self.written = []
        self.win._audio.write = self.written.append
        self.device = None

    def tune(self, freq=100.1e6, mode=("FM", "75 kHz", "200 kHz")):
        from sdr_module.gui.device_dialog import MockDevice

        self.win.set_frequency(freq)
        self.win._set_demod_mode(*mode)
        self.device = MockDevice()
        self.device.set_sample_rate(2.4e6)
        self.device.set_frequency(freq)
        self.device._running = True
        self.device._rng = np.random.default_rng(11)

    def frames(self, count):
        for _ in range(count):
            self.win._process_block(self.device.read_samples(self.FRAME))

    def stereo_blocks(self):
        return [block for block in self.written if np.ndim(block) == 2]


class TestStereoInTheWindow(_StereoWindowCase):
    def test_the_stereo_switch_is_in_radio_audio(self):
        action = self.win._stereo_action
        menu = next(
            a.menu()
            for a in self.win.menuBar().actions()
            if a.text().replace("&", "") == "Radio"
        )
        audio = next(
            a.menu() for a in menu.actions() if a.text().replace("&", "") == "Audio"
        )
        self.assertIn(action, audio.actions())
        self.assertEqual(action.text(), "&Stereo")
        self.assertTrue(action.isCheckable())
        self.assertTrue(action.isChecked())  # (on by default)
        self.assertIn("mono", action.toolTip())
        with self.assertNoLogs(level="WARNING"):
            action.setChecked(False)
        self.assertIs(self.store.values["audio/stereo"], False)
        self.assertIn("mono", self.win._message_label.text())
        action.setChecked(True)
        self.assertIs(self.store.values["audio/stereo"], True)

    def test_broadcast_fm_plays_in_stereo(self):
        self.tune()
        self.frames(12)
        stereo = self.stereo_blocks()
        self.assertTrue(stereo)
        last = stereo[-1]
        # The demo tune: the arpeggio leans right, the pad left.
        self.assertGreater(float(np.max(np.abs(last[:, 0] - last[:, 1]))), 0.01)
        badge = self.win._stereo_badge
        self.assertTrue(self.win._stereo_badge_action.isVisible())
        self.assertEqual(badge.text(), "STEREO")
        self.assertEqual(badge.property("tone"), "success")
        self.assertTrue(self.win._stereo_lit)
        self.assertIn("Radio > Audio > Stereo", badge.toolTip())

    def test_stereo_off_plays_mono_and_says_so(self):
        self.tune()
        self.frames(8)
        self.win._stereo_action.setChecked(False)
        self.frames(3)  # (the difference fades out within a frame or two)
        self.written.clear()
        self.frames(4)
        self.assertTrue(self.written)
        self.assertEqual(self.stereo_blocks(), [])
        badge = self.win._stereo_badge
        self.assertEqual(badge.text(), "MONO")
        self.assertIn("Radio > Audio > Stereo is off", badge.toolTip())
        self.assertFalse(self.win._stereo_lit)

    def test_the_badge_shows_only_for_broadcast_fm(self):
        self.tune()
        self.frames(6)
        self.assertTrue(self.win._stereo_badge_action.isVisible())
        for mode in (("FM", "5 kHz", "25 kHz"), ("AM", None, "10 kHz")):
            with self.subTest(mode=mode):
                self.win._set_demod_mode(*mode)
                self.frames(2)
                self.assertFalse(self.win._stereo_badge_action.isVisible())
                self.assertFalse(self.win._stereo_lit)
        self.written.clear()
        self.frames(2)
        self.assertEqual(self.stereo_blocks(), [])

    def test_the_badge_says_stereo_with_the_speaker_off(self):
        # As a radio's lamp does: what the station is received as, whether
        # it plays (no audio output, muted) or not.
        self.win._audio_enabled = False
        self.tune()
        self.frames(12)
        self.assertEqual(self.written, [])  # (nothing to the speaker)
        self.assertEqual(self.win._stereo_badge.text(), "STEREO")
        self.assertTrue(self.win._stereo_badge_action.isVisible())
        self.win._stereo_action.setChecked(False)
        self.frames(2)
        self.assertEqual(self.win._stereo_badge.text(), "MONO")
        self.assertTrue(self.win._stereo_badge_action.isVisible())

    def test_a_narrow_filter_says_so(self):
        self.tune(mode=("FM", "75 kHz", "100 kHz"))
        self.frames(8)
        self.assertEqual(self.win._stereo_badge.text(), "MONO")
        self.assertIn("Filter is too narrow", self.win._stereo_badge.toolTip())
        self.assertEqual(self.stereo_blocks(), [])

    def test_no_station_no_badge(self):
        # An empty channel (the squelch closed): nothing is received.
        self.win._control_panel.set_squelch_auto(False)
        self.win._control_panel.set_squelch_enabled(True)
        self.win._control_panel.set_squelch_db(-20.0)
        self.tune(100.75e6)
        self.frames(6)
        self.assertFalse(self.win._stereo_badge_action.isVisible())

    def test_the_badge_goes_when_the_receiver_stops(self):
        self.tune()
        self.frames(6)
        self.win._reset_demodulator()
        self.assertFalse(self.win._stereo_badge_action.isVisible())
        self.assertFalse(self.win._stereo_lit)

    def test_a_saved_choice_is_restored(self):
        from sdr_module.gui import main_window

        self.store.values["audio/stereo"] = False
        other = main_window.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        self.addCleanup(other.close)
        for timer in (other._display_timer, other._status_timer, other._hotplug_timer):
            timer.stop()
        self.assertFalse(other._stereo_action.isChecked())
        self.assertFalse(other._stereo_enabled)


class TestStereoLamp(_StereoWindowCase):
    """The radio tuner's STEREO lamp lights when stereo is decoded, not
    whenever there is a pilot."""

    def setUp(self):
        super().setUp()
        if not self.mw.HAS_HAM_RADIO:
            self.skipTest("ham radio panels not installed")
        self.win._show_radio_tuner()
        self.tuner = self.win._radio_tuner
        self.addCleanup(self.tuner.close)

    def test_the_lamp_follows_what_is_decoded(self):
        self.tune()
        self.frames(12)
        self.assertTrue(self.tuner._stereo)
        self.win._stereo_action.setChecked(False)
        self.frames(6)
        # The pilot is still there (the seek still counts it), but what
        # plays is mono: no STEREO.
        self.assertTrue(self.win._pilot is not None and self.win._pilot.present)
        self.assertFalse(self.tuner._stereo)
        self.win._stereo_action.setChecked(True)
        self.frames(12)
        self.assertTrue(self.tuner._stereo)

    def test_no_lamp_while_the_tuner_is_off_its_bands(self):
        """Tuned outside both broadcast bands, the tuner shows the receiver
        is elsewhere: its lamp stays dark (as before stereo was decoded),
        while the receiver bar still says what plays."""
        self.tune()
        self.frames(12)
        self.assertTrue(self.tuner._stereo)
        with mock.patch.object(self.tuner, "is_off_band", return_value=True):
            self.win._show_stereo(None, False)  # (shown afresh)
            self.frames(3)
            self.assertEqual(self.win._stereo_badge.text(), "STEREO")
            self.assertFalse(self.tuner._stereo)


class TestRecordAudioInStereo(_StereoWindowCase):
    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="stereo-rec-")
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.store.values["recording/folder"] = self.folder

    def _record(self):
        self.tune()
        self.assertTrue(self.win._start_audio_recording())
        message = self.win._message_label.text()
        self.frames(15)
        path = self.win._stop_audio_recording()
        with wave.open(path, "rb") as wav:
            data = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
            return message, wav.getnchannels(), data.reshape(-1, wav.getnchannels())

    def test_broadcast_fm_records_a_stereo_wav(self):
        message, channels, frames = self._record()
        self.assertEqual(channels, 2)
        self.assertIn("stereo", message)
        tail = frames[len(frames) // 2 :].astype(float)
        self.assertGreater(np.max(np.abs(tail[:, 0] - tail[:, 1])), 300)

    def test_with_stereo_off_it_records_mono(self):
        self.win._stereo_action.setChecked(False)
        message, channels, _frames = self._record()
        self.assertEqual(channels, 1)
        self.assertNotIn("stereo", message)


class TestDamagedStereoSetting(_IniSettings):
    """A damaged Radio > Audio > Stereo setting starts with stereo on (the
    default), logged once."""

    def test_a_damaged_value_gives_stereo(self):
        from sdr_module.gui import bookmarks_panel, main_window
        from sdr_module.gui.settings_store import GuiSettings

        self.write(first_run_done="true", audio__stereo="maybe")
        for module in (main_window, bookmarks_panel):
            patcher = mock.patch.object(module, "GuiSettings", GuiSettings)
            patcher.start()
            self.addCleanup(patcher.stop)
        with self.assertLogs("sdr_module.gui.settings_store", "WARNING") as logs:
            win = main_window.SDRMainWindow(demo_mode=False)
        for timer in (win._display_timer, win._status_timer, win._hotplug_timer):
            timer.stop()
        self.addCleanup(win.deleteLater)
        self.addCleanup(win.close)
        self.assertTrue(win._stereo_enabled)
        self.assertTrue(win._stereo_action.isChecked())
        self.assertTrue(any("audio/stereo" in line for line in logs.output))


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestStereoWork(_StereoWindowCase):
    """Stereo costs nothing outside broadcast FM, with the switch off, or
    in the second receiver."""

    def test_no_stereo_work_unless_it_can_play(self):
        calls = []
        real = StereoDecoder.process

        def counted(decoder, mpx):
            calls.append(len(mpx))
            return real(decoder, mpx)

        with mock.patch.object(StereoDecoder, "process", counted):
            for mode in (("FM", "5 kHz", "25 kHz"), ("AM", None, "10 kHz")):
                self.tune(100.1e6, mode)
                self.frames(3)
            self.win._stereo_action.setChecked(False)
            self.tune()
            self.frames(3)
            self.assertEqual(calls, [])
            self.win._stereo_action.setChecked(True)
            self.frames(1)
            self.assertEqual(len(calls), 1)

    def test_a_filter_too_narrow_for_stereo_needs_no_demodulation(self):
        # The speaker off, broadcast FM is demodulated only for the STEREO
        # badge and lamp. Through a filter too narrow for stereo the badge
        # says so without it (the audio went on being demodulated, about
        # half a millisecond a frame, for nothing).
        from sdr_module.gui.main_window import _ReceiverChain

        self.win._audio_enabled = False
        self.tune(mode=("FM", "75 kHz", "100 kHz"))
        calls = []
        real = _ReceiverChain.demodulate

        def counted(chain, channel, deviation):
            calls.append(len(channel))
            return real(chain, channel, deviation)

        with mock.patch.object(_ReceiverChain, "demodulate", counted):
            self.frames(4)
        self.assertEqual(calls, [])
        self.assertTrue(self.win._stereo_badge_action.isVisible())
        self.assertEqual(self.win._stereo_badge.text(), "MONO")
        self.assertIn("Filter is too narrow", self.win._stereo_badge.toolTip())

    def test_the_second_receiver_is_mono(self):
        from sdr_module.gui.main_window import _ReceiverChain

        self.assertFalse(_ReceiverChain.stereo)
        self.tune()
        self.frames(2)
        self.assertTrue(self.win._rx_chain.stereo)
        self.assertFalse(_ReceiverChain(2.4e6, "FM", 200e3).stereo)


if __name__ == "__main__":
    unittest.main()

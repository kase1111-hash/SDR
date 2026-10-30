"""Playing a recording back through the receiver (File > Open Recording).

fg:FG-G01, mc:MC-G01, rx:G-05, ux:UX-G02: recordings could only be
converted, never played. fg:FG-U01: Ctrl+O was "Import Recording (convert
format)". fg:FG-G02 / mc:MC-G03 / ux:UX-G03 (playback part): with a file as
the source, the live decoders decode it.
"""

import os
import shutil
import tempfile
import time
import tracemalloc
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
)

if HAS_PYQT6:
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent
    from PyQt6.QtWidgets import QMessageBox

RATE = 240e3
CENTER = 145.0e6


def _fm_tone(seconds=1.0, tone=1000.0, deviation=5e3, offset=0.0, rate=RATE):
    n = int(seconds * rate)
    t = np.arange(n) / rate
    phase = 2 * np.pi * deviation / tone * np.sin(2 * np.pi * tone * t)
    x = 0.5 * np.exp(1j * (phase + 2 * np.pi * offset * t))
    return x.astype(np.complex64)


def _save_sigmf(path, samples, rate=RATE, center=CENTER):
    from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

    save_iq_file(path, samples, rate, center, SampleFormat.FLOAT32, FileFormat.SIGMF)
    return path


class TestFilePlaybackDevice(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="sdr-playback-")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def device(self, samples, **kw):
        from sdr_module.gui.iq_files import describe_iq_file
        from sdr_module.gui.playback import FilePlaybackDevice

        path = _save_sigmf(os.path.join(self.folder, "x.sigmf-data"), samples, **kw)
        options, missing = describe_iq_file(path)
        self.assertEqual(missing, set())
        return FilePlaybackDevice(options)

    def test_the_mock_device_api(self):
        x = (np.arange(1000) / 1000).astype(np.complex64)
        dev = self.device(x)
        self.assertEqual((dev.frequency, dev.sample_rate), (CENTER, RATE))
        self.assertTrue(dev.paced)
        self.assertTrue(dev.set_frequency(CENTER))
        self.assertFalse(dev.set_frequency(CENTER + 1e3))  # a file can't retune
        self.assertTrue(dev.set_sample_rate(RATE))
        self.assertFalse(dev.set_sample_rate(2.4e6))
        self.assertIsNone(dev.read_samples(10))  # not started
        self.assertTrue(dev.start_rx())
        self.assertTrue(dev.is_streaming)
        np.testing.assert_array_equal(dev.read_samples(600), x[:600])
        np.testing.assert_array_equal(dev.read_samples(600), x[600:])
        self.assertTrue(dev.finished)
        self.assertEqual(len(dev.read_samples(10)), 0)
        dev.stop_rx()
        self.assertTrue(dev.start_rx())  # Play again: from the start
        np.testing.assert_array_equal(dev.read_samples(5), x[:5])
        dev.close()

    def test_loop_wraps_around(self):
        x = (np.arange(1000) / 1000).astype(np.complex64)
        dev = self.device(x)
        dev.loop = True
        dev.start_rx()
        dev.read_samples(900)
        out = dev.read_samples(300)
        np.testing.assert_array_equal(out, np.concatenate([x[900:], x[:200]]))
        self.assertEqual(dev.loops, 1)
        self.assertFalse(dev.finished)

    def test_seek(self):
        x = (np.arange(48000) / 48000).astype(np.complex64)
        dev = self.device(x, rate=48000.0)
        dev.start_rx()
        dev.seek_seconds(0.5)
        self.assertEqual(dev.position, 24000)
        np.testing.assert_array_equal(dev.read_samples(3), x[24000:24003])
        dev.seek(10**9)
        self.assertEqual(dev.position, 48000)

    def test_samples_that_arent_numbers_are_played_as_silence(self):
        x = np.ones(100, np.complex64)
        x[10] = np.nan
        dev = self.device(x)
        dev.start_rx()
        out = dev.read_samples(100)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertEqual(dev.bad_samples, 1)

    def test_it_streams_from_disk(self):
        # A 4 M-sample file (32 MB): playing a block reads only the block.
        x = np.zeros(4_000_000, np.complex64)
        dev = self.device(x)
        dev.start_rx()
        tracemalloc.start()
        dev.read_samples(80_000)
        _now, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertLess(peak, 4e6)


class _PlaybackCase(_WindowTestCase):
    demo = True

    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="sdr-playback-win-")
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.win._recording_folder = self.folder

    def make_file(self, samples=None, name="fm.sigmf-data", **kw):
        if samples is None:
            samples = _fm_tone(2.0)
        return _save_sigmf(os.path.join(self.folder, name), samples, **kw)

    def play(self, path=None):
        path = path or self.make_file()
        with mock.patch(
            "sdr_module.gui.main_window.QFileDialog.getOpenFileName",
            return_value=(path, ""),
        ):
            self.win._open_recording_action.trigger()
        self.settle()
        return path

    def message(self):
        return self.win._message_label.text()


class TestOpenRecording(_PlaybackCase):
    def test_ctrl_o_plays_the_file(self):
        action = self.win._open_recording_action
        self.assertEqual(action.shortcut().toString(), "Ctrl+O")
        self.assertEqual(action.text().replace("&", ""), "Open Recording...")
        before = self.win._device
        path = self.play()
        win = self.win
        self.assertTrue(win._playback.active)
        self.assertIsNot(win._device, before)
        self.assertTrue(win._is_running)
        self.assertEqual(win._device_label.text(), "Playback: fm.sigmf-data")
        self.assertEqual(win._state_badge.text(), "PLAYING")
        self.assertTrue(win._playback_bar.isVisible())
        self.assertIn("fm.sigmf-data", win._playback_bar.file_text())
        self.assertEqual(win._lo_frequency(), CENTER)  # the file's center
        self.assertEqual(win._rate_label.text(), "0.24 MS/s")
        self.assertIn("Playing fm.sigmf-data", self.message())
        self.assertTrue(win._close_recording_action.isEnabled())
        self.assertEqual(win._start_button.text(), "❚❚  Pause")
        self.assertTrue(os.path.exists(path))

    def test_tuning_moves_the_vfo_inside_the_file_and_is_refused_outside(self):
        self.play()
        win = self.win
        # Broadcast FM's 200 kHz channel doesn't fit 50 kHz off the middle of
        # a 240 kHz recording: said as such, not as "outside".
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        self.assertFalse(win.set_frequency(CENTER + 50e3))
        self.assertIn("A 200 kHz channel at 145.050 MHz doesn't fit", self.message())
        win._set_demod_mode("FM", "5 kHz", "25 kHz")
        self.assertTrue(win.set_frequency(CENTER + 50e3))
        self.assertEqual(win._current_frequency(), CENTER + 50e3)
        self.assertEqual(win._lo_frequency(), CENTER)
        self.assertFalse(win.set_frequency(146.0e6))
        self.assertEqual(win._current_frequency(), CENTER + 50e3)
        self.assertIn("outside the recording", self.message())
        self.assertIn("144.88–145.12 MHz", self.message())

    def test_played_samples_reach_spectrum_audio_and_s_meter(self):
        self.play()
        win = self.win
        win._set_demod_mode("FM", "5 kHz", "25 kHz")
        win._audio_enabled = True
        heard = []
        win._audio.write = lambda a: heard.append(np.array(a))
        lines = []
        win._waterfall.add_line = lambda power: lines.append(power)
        for _ in range(12):
            win._process_block(win._device.read_samples(24000))
        audio = np.concatenate(heard)
        self.assertGreater(len(audio), 0.9 * 48000)
        self.assertGreater(float(np.sqrt(np.mean(audio**2))), 0.1)
        self.assertEqual(len(lines), 12)
        self.assertGreater(win._last_peak_db, -40.0)

    def test_squelch_s_meter_and_sstv_work_on_played_samples(self):
        quiet = np.zeros(int(0.5 * RATE), np.complex64)
        self.play(self.make_file(np.concatenate([quiet, _fm_tone(1.0)])))
        win = self.win
        win._set_demod_mode("FM", "5 kHz", "25 kHz")
        panel = win._control_panel
        panel.set_squelch_auto(False)
        panel.set_squelch_enabled(True)
        panel.set_squelch_db(-40.0)
        metered, sstv = [], []
        win._signal_meter_panel.update_samples = lambda ch, noise: metered.append(
            len(ch)
        )
        win._sstv_listening = lambda: True
        win._sstv_panel.process_audio = lambda audio, sample_rate: sstv.append(
            len(audio)
        )
        dev = win._device
        for _ in range(4):  # the quiet half second
            win._process_block(dev.read_samples(int(0.1 * RATE)))
        self.assertFalse(win._squelch.is_open)
        for _ in range(5):  # the FM tone
            win._process_block(dev.read_samples(int(0.1 * RATE)))
        self.assertTrue(win._squelch.is_open)
        self.assertEqual(len(metered), 9)
        self.assertGreater(sum(sstv), 0.8 * 48000)

    def test_the_display_loop_reads_in_real_time(self):
        self.play()
        win = self.win
        dev = win._device
        win._update_display()  # first frame: one block
        start = time.monotonic()
        while time.monotonic() - start < 0.4:
            time.sleep(0.03)
            win._update_display()
        elapsed = time.monotonic() - start
        played = dev.position / RATE
        self.assertAlmostEqual(played, elapsed, delta=0.12)

    def test_speed_reads_faster_and_mutes_the_speaker(self):
        self.play()
        win = self.win
        win._playback_bar.set_speed(2.0)
        self.assertEqual(win._device.speed, 2.0)
        self.assertEqual(win._pacing_rate(), 2 * RATE)
        self.assertFalse(win._realtime())
        self.assertIn("audio plays at 1x only", win._realtime_hint())
        win._playback_bar.set_speed(1.0)
        self.assertEqual(win._pacing_rate(), RATE)

    def test_space_pauses_and_plays(self):
        self.play()
        win = self.win
        win._spectrum.setFocus()
        self.settle()

        def space():
            event = QKeyEvent(
                QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier
            )
            win.keyPressEvent(event)

        space()
        self.assertFalse(win._is_running)
        self.assertEqual(win._state_badge.text(), "PAUSED")
        self.assertEqual(win._start_action.text(), "&Play Recording\tSpace")
        space()
        self.assertTrue(win._is_running)
        self.assertEqual(win._state_badge.text(), "PLAYING")

    def test_the_end_pauses_and_play_starts_again(self):
        self.play(self.make_file(_fm_tone(0.5)))
        win = self.win
        dev = win._device
        win._process_block(dev.read_samples(10**6))
        win._playback._tick()
        self.assertFalse(win._is_running)
        self.assertEqual(win._state_badge.text(), "ENDED")
        self.assertIn("End of fm.sigmf-data", self.message())
        win._toggle_acquisition()
        self.assertTrue(win._is_running)
        self.assertEqual(dev.position, 0)

    def test_loop_and_seek_from_the_strip(self):
        self.play()
        win = self.win
        bar = win._playback_bar
        bar._loop.setChecked(True)
        self.assertTrue(win._device.loop)
        bar._slider.setValue(15)  # tenths of a second, as a key press would
        self.assertAlmostEqual(win._device.position_s, 1.5, places=3)
        self.assertEqual(bar.elapsed_text(), "0:01")

    def test_close_goes_back_to_the_device_and_frequency_from_before(self):
        win = self.win
        demo = win._device
        win.set_frequency(100.3e6)
        lo = win._lo_frequency()
        self.play()
        self.assertFalse(win._control_panel._gain_slider.parentWidget().isEnabled())
        self.assertFalse(win._control_panel._agc_check.isEnabled())
        win._playback_bar._close.click()
        self.settle()
        self.assertFalse(win._playback.active)
        self.assertIs(win._device, demo)
        self.assertTrue(win._is_running)
        self.assertEqual(win._current_frequency(), 100.3e6)
        self.assertEqual(win._lo_frequency(), lo)
        self.assertIsNone(win._lo_lock)
        self.assertFalse(win._playback_bar.isVisible())
        self.assertEqual(win._state_badge.text(), "RUNNING")
        self.assertTrue(win._control_panel._gain_slider.parentWidget().isEnabled())
        self.assertIn("Back to Demo Device", self.message())
        self.assertFalse(win._close_recording_action.isEnabled())
        # The demo works as before: tuning far away retunes it.
        self.assertTrue(win.set_frequency(433.92e6))
        self.assertEqual(demo.frequency, 433.92e6)

    def test_close_ends_recordings_of_the_file(self):
        self.play()
        win = self.win
        win._set_demod_mode("FM", "5 kHz", "25 kHz")
        self.store.set("recording/folder", os.path.join(self.folder, "rec"))
        win._record_button.click()
        win._record_audio_action.trigger()
        for _ in range(3):
            win._process_block(win._device.read_samples(24000))
        count = win._buffer_sample_count()
        win._playback.close()
        self.assertFalse(win._recording)
        self.assertIsNone(win._audio_take)
        self.assertEqual(win._buffer_sample_count(), count)  # kept, to save
        self.assertEqual(win._buffer_meta, (CENTER, RATE))  # the file's
        self.assertIn(
            "Stopped the I/Q recording and the audio recording", self.message()
        )
        win._update_display()
        self.assertEqual(win._buffer_sample_count(), count)  # nothing live added

    def test_disconnect_closes_the_file_and_the_device_before_it(self):
        self.play()
        self.win._disconnect_device()
        self.assertIsNone(self.win._device)
        self.assertFalse(self.win._playback.active)
        self.assertEqual(self.win._state_badge.text(), "NO DEVICE")
        self.assertIn("Closed fm.sigmf-data and disconnected", self.message())

    def test_closing_the_window_saves_the_tuning_from_before(self):
        self.win.set_frequency(100.3e6)
        self.play()
        self.win.set_frequency(CENTER + 20e3)
        self.win.close()
        self.assertAlmostEqual(self.store.get_float("frequency_hz", 0.0), 100.3e6)

    def test_opening_another_file_keeps_the_device_to_go_back_to(self):
        demo = self.win._device
        self.play()
        other = self.make_file(_fm_tone(1.0), name="other.sigmf-data", center=433.92e6)
        self.play(other)
        self.assertEqual(self.win._lo_frequency(), 433.92e6)
        self.assertEqual(self.win._device_label.text(), "Playback: other.sigmf-data")
        self.win._playback.close()
        self.assertIs(self.win._device, demo)


class TestOpenWhileRecording(_PlaybackCase):
    def _answer(self, button_text):
        """Answer the next question box with the button named so."""

        def fake_exec(box):
            for button in box.buttons():
                if button.text().replace("&", "") == button_text:
                    button.click()
                    return 0
            raise AssertionError(f"no {button_text!r} in {box.text()!r}")

        self.win._exec_dialog = fake_exec

    def test_while_receiving_it_just_plays(self):
        asked = []
        self.win._exec_dialog = lambda box: asked.append(box) or 0
        self.play()
        self.assertEqual(asked, [])
        self.assertTrue(self.win._playback.active)

    def test_cancel_keeps_the_recording_going(self):
        win = self.win
        win._record_button.click()
        win._update_display()
        self._answer("Cancel")
        self.play()
        self.assertFalse(win._playback.active)
        self.assertTrue(win._recording)
        self.assertTrue(win._record_button.isChecked())
        win._record_button.click()

    def test_stop_and_play_keeps_the_take(self):
        win = self.win
        win._record_button.click()
        win._update_display()
        count = win._buffer_sample_count()
        self.assertGreater(count, 0)
        self._answer("Stop and Play")
        self.play()
        self.assertTrue(win._playback.active)
        self.assertFalse(win._recording)
        self.assertEqual(win._buffer_sample_count(), count)
        self.assertTrue(win._buffer_unsaved)


class TestHeaderlessPlayback(_PlaybackCase):
    """fg:FG-G03 for playback: an rtl_sdr .bin asks how to read it."""

    def test_bin_asks_and_plays(self):
        from sdr_module.gui import iq_files

        x = _fm_tone(1.0, rate=1.024e6)
        raw = np.empty(2 * len(x), np.uint8)
        raw[0::2] = np.clip(np.rint(x.real * 127.5 + 127.5), 0, 255)
        raw[1::2] = np.clip(np.rint(x.imag * 127.5 + 127.5), 0, 255)
        path = os.path.join(self.folder, "capture_433.92MHz_1.024Msps.bin")
        raw.tofile(path)
        seen = {}

        def fake_exec(dialog):
            if isinstance(dialog, iq_files.ImportOptionsDialog):
                seen["format"] = dialog.sample_format()
                seen["rate"] = dialog.sample_rate()
                seen["center"] = dialog.center_hz()
                seen["check"] = dialog.check_text()
                dialog._on_accept()
                return dialog.result()
            return 0

        self.win._exec_dialog = fake_exec
        self.play(path)
        from sdr_module.dsp.recording import SampleFormat

        self.assertEqual(seen["format"], SampleFormat.UINT8)
        self.assertEqual(seen["rate"], 1.024e6)
        self.assertEqual(seen["center"], 433.92e6)
        self.assertIn("look right", seen["check"])
        self.assertTrue(self.win._playback.active)
        self.assertEqual(self.win._lo_frequency(), 433.92e6)
        self.assertEqual(self.win._device_sample_rate(), 1.024e6)

    def test_an_empty_file_is_refused(self):
        path = os.path.join(self.folder, "empty.cf32")
        open(path, "wb").close()
        with mock.patch.object(QMessageBox, "warning") as warning:
            self.play(path)
        warning.assert_called_once()
        self.assertIn("no I/Q samples", warning.call_args[0][2])
        self.assertFalse(self.win._playback.active)


class TestDecodersOnPlayback(_PlaybackCase):
    def test_aprs_in_a_played_file_decodes(self):
        from sdr_module.dsp import protocol_encoders as enc

        burst = enc.aprs_baseband(
            "N0CALL-9", "APZ001", "!4903.50N/07201.75W>Played back", RATE, ["WIDE1-1"]
        )
        pad = np.zeros(int(0.3 * RATE), np.complex64)
        x = np.concatenate([pad, burst.astype(np.complex64), pad])
        self.play(self.make_file(x, name="aprs.sigmf-data", center=144.39e6))
        win = self.win
        panel = win._decoder_panel
        panel._proto_combo.setCurrentText("AX.25/APRS")
        self.settle(2)
        panel._enabled_check.setChecked(True)
        while not win._device.finished:
            win._process_block(win._device.read_samples(8000))
        texts = [m["content"] for m in panel._messages]
        self.assertTrue(any("Played back" in t for t in texts), texts)


if __name__ == "__main__":
    unittest.main()

"""Record what you hear: Radio > Record Audio (Ctrl+Shift+A).

fg:FG-G04, mc:MC-G02, rx:G-04, ux:UX-G01: only raw I/Q could be recorded.
Now the demodulated audio is written as 48 kHz 16-bit WAV, from the audio
the receiver produces (also with the speaker off, squelched stretches as
silence), auto-named in the recordings folder, with an AUDIO badge.
"""

import os
import shutil
import tempfile
import unittest
import wave

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    _WindowTestCase,
)


def _read_wav(path):
    with wave.open(path, "rb") as handle:
        params = (handle.getframerate(), handle.getsampwidth(), handle.getnchannels())
        frames = handle.readframes(handle.getnframes())
    return params, np.frombuffer(frames, np.int16).astype(np.float32) / 32767.0


class TestAudioTake(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="sdr-audio-")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_heard_audio_and_silence_with_fades(self):
        from sdr_module.gui.audio_recording import AudioTake

        path = os.path.join(self.folder, "a.wav")
        take = AudioTake(path, 145.8e6, "FM")
        tone = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(4800) / 48000)
        take.write(tone.astype(np.float32), open_=True)
        take.write(tone.astype(np.float32), open_=False)  # squelched
        take.silence(0.05)  # nothing demodulated
        take.write(tone.astype(np.float32), open_=True)
        seconds = take.stop()  # (ends with a short fade, not a click)
        from sdr_module.gui.audio_chain import FADE_S

        fade = int(round(FADE_S * 48000))
        self.assertAlmostEqual(seconds, 0.35 + fade / 48000, places=4)
        params, audio = _read_wav(path)
        self.assertEqual(params, (48000, 2, 1))
        self.assertEqual(len(audio), 16800 + fade)
        self.assertLess(abs(audio[-1]), 0.01)
        self.assertLess(abs(audio[0]), 0.01)  # faded in
        self.assertAlmostEqual(float(np.max(np.abs(audio[2000:4800]))), 0.5, places=2)
        self.assertLess(float(np.max(np.abs(audio[6000:12000]))), 1e-4)  # silence
        self.assertAlmostEqual(take.heard_s, 0.2, places=3)

    def test_file_name(self):
        from sdr_module.gui.audio_recording import audio_file_name

        name = audio_file_name(145.8e6, "FM", 0)
        self.assertRegex(name, r"^145\.800MHz_FM_\d{8}_\d{6}\.wav$")
        self.assertTrue(audio_file_name(7.074e6, "USB").startswith("7.074MHz_USB_"))


class _AudioCase(_WindowTestCase):
    demo = True

    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="sdr-audio-win-")
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.win._recording_folder = os.path.join(self.folder, "cache")
        self.store.set("recording/folder", os.path.join(self.folder, "rec"))
        self.spoken = []
        self.win._audio.write = self.spoken.append

    def frames(self, n=10, size=24000):
        for _ in range(n):
            self.win._process_block(self.win._device.read_samples(size))

    def wavs(self):
        folder = os.path.join(self.folder, "rec")
        return sorted(n for n in os.listdir(folder) if n.endswith(".wav"))


class TestRecordAudio(_AudioCase):
    def test_menu_action(self):
        action = self.win._record_audio_action
        self.assertEqual(action.shortcut().toString(), "Ctrl+Shift+A")
        self.assertTrue(action.isCheckable())

    def test_records_what_is_heard_even_with_the_speaker_off(self):
        win = self.win
        win._audio_enabled = False  # speaker off
        win._record_audio_action.trigger()
        take = win._audio_take
        self.assertIsNotNone(take)
        self.assertTrue(win._audio_rec_label.isVisible())
        self.frames(20)  # 0.2 s of the 100.1 MHz station
        win._update_status()
        self.assertRegex(win._audio_rec_label.text(), r"^AUDIO 00:0\d$")
        self.assertEqual(win._audio_rec_label.property("tone"), "danger")
        win._record_audio_action.trigger()
        self.assertIsNone(win._audio_take)
        self.assertFalse(win._audio_rec_label.isVisible())
        names = self.wavs()
        self.assertEqual(len(names), 1)
        self.assertRegex(names[0], r"^100\.100MHz_FM_\d{8}_\d{6}\.wav$")
        params, audio = _read_wav(os.path.join(self.folder, "rec", names[0]))
        # (K-S1: broadcast FM with Radio > Audio > Stereo on records a
        # stereo WAV, left and right interleaved.)
        self.assertEqual(params, (48000, 2, 2))
        self.assertAlmostEqual(len(audio) / 2 / 48000, 20 * 24000 / 2.4e6, delta=0.02)
        self.assertGreater(float(np.sqrt(np.mean(audio**2))), 0.05)  # music
        self.assertEqual(self.spoken, [])  # nothing went to the speaker
        self.assertIn("Saved 00:00 of audio", win._message_label.text())

    def test_squelched_stretches_are_silence(self):
        win = self.win
        win.set_frequency(100.75e6)  # an empty channel in the demo's span
        panel = win._control_panel
        panel.set_squelch_auto(False)
        panel.set_squelch_enabled(True)
        panel.set_squelch_db(-20.0)
        win._record_audio_action.trigger()
        self.frames(10)
        win._record_audio_action.trigger()
        names = self.wavs()
        _params, audio = _read_wav(os.path.join(self.folder, "rec", names[0]))
        self.assertGreater(len(audio), 0.08 * 48000)
        self.assertLess(float(np.max(np.abs(audio))), 1e-4)

    def test_info_tab_shows_it(self):
        self.win._record_audio_action.trigger()
        self.frames(2)
        self.win._refresh_info_panel()
        self.assertEqual(self.win._info_panel.value("recording"), "Audio 00:00")
        self.win._record_audio_action.trigger()
        self.win._refresh_info_panel()
        self.assertEqual(self.win._info_panel.value("recording"), "Idle")

    def test_none_iq_mode_is_refused(self):
        self.win._set_demod_mode("None (I/Q)")
        self.win._record_audio_action.trigger()
        self.assertIsNone(self.win._audio_take)
        self.assertFalse(self.win._record_audio_action.isChecked())
        self.assertIn("None (I/Q) has no audio", self.win._message_label.text())

    def test_nothing_received_leaves_no_file(self):
        self.win._stop_acquisition()
        self.win._record_audio_action.trigger()
        self.assertIn("starts with the receiver", self.win._message_label.text())
        self.win._update_status()
        self.assertEqual(self.win._audio_rec_label.property("tone"), "warning")
        self.win._record_audio_action.trigger()
        self.assertEqual(self.wavs(), [])
        self.assertIn("nothing was saved", self.win._message_label.text())

    def test_recording_group_choice_drives_the_record_buttons(self):
        win = self.win
        panel = win._control_panel
        panel.set_record_kind("audio")
        self.assertFalse(panel._format_combo.isVisibleTo(panel))
        self.assertEqual(win._record_button.accessibleName(), "Record audio")
        win._record_button.click()  # the toolbar's Record
        self.assertIsNotNone(win._audio_take)
        self.frames(3)
        self.assertTrue(win._record_audio_action.isChecked())
        self.assertTrue(panel._record_btn.isChecked())
        self.assertFalse(win._recording)  # not I/Q
        self.assertEqual(panel._record_status.text(), "● Recording audio")
        panel._pause_btn.click()
        self.assertTrue(win._audio_take.paused)
        win._update_status()
        self.assertTrue(win._audio_rec_label.text().startswith("AUDIO PAUSED"))
        panel._pause_btn.click()
        panel._record_btn.click()  # the panel's Stop
        self.assertIsNone(win._audio_take)
        self.assertEqual(len(self.wavs()), 1)
        self.assertEqual(self.store.get_str("recording/kind", ""), "audio")
        panel.set_record_kind("iq")
        self.assertEqual(win._record_button.accessibleName(), "Record I/Q")

    def test_iq_and_audio_together(self):
        win = self.win
        win._record_audio_action.trigger()
        win._record_button.click()  # I/Q (the default choice)
        self.frames(3)
        self.assertTrue(win._recording)
        self.assertIsNotNone(win._audio_take)
        win._record_button.click()
        win._record_audio_action.trigger()
        self.assertGreater(win._buffer_sample_count(), 0)
        self.assertEqual(len(self.wavs()), 1)

    def test_with_audio_chosen_an_iq_recording_leaves_the_group_timer_alone(self):
        # The group's Record buttons and timer are the audio's then: an I/Q
        # recording started from Radio > Record I/Q used to show its time
        # there, next to "Ready".
        win = self.win
        panel = win._control_panel
        panel.set_record_kind("audio")
        win._record_action.trigger()  # Radio > Record I/Q
        self.frames(3)
        self.assertTrue(win._recording)
        win._recording_elapsed = lambda: 65.0
        win._update_status()
        self.assertTrue(win._recording_label.text().startswith("REC 00:01:05"))
        # (Its status line names the I/Q recording and how to stop it, not
        # "Ready": REC-E2E-12.)
        self.assertEqual(
            panel._record_status.text(),
            "● Recording I/Q 01:05 (Ctrl+Shift+R stops it)",
        )
        self.assertEqual(panel._record_time.text(), "00:00:00")
        win._record_action.trigger()

    def test_works_during_playback(self):
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

        rate = 240e3
        t = np.arange(int(0.5 * rate)) / rate
        x = 0.5 * np.exp(1j * 5.0 * np.sin(2 * np.pi * 1000 * t))
        path = os.path.join(self.folder, "fm.sigmf-data")
        save_iq_file(
            path,
            x.astype(np.complex64),
            rate,
            145e6,
            SampleFormat.FLOAT32,
            FileFormat.SIGMF,
        )
        win = self.win
        win._confirm_stop_recordings = lambda *_a: True
        self.assertTrue(win._open_recording(path))
        win._set_demod_mode("FM", "5 kHz", "25 kHz")
        win._record_audio_action.trigger()
        while not win._device.finished:
            win._process_block(win._device.read_samples(12000))
        win._record_audio_action.trigger()
        names = self.wavs()
        self.assertTrue(names[0].startswith("145.000MHz_FM_"))
        _params, audio = _read_wav(os.path.join(self.folder, "rec", names[0]))
        self.assertAlmostEqual(len(audio) / 48000, 0.5, delta=0.02)
        self.assertGreater(float(np.sqrt(np.mean(audio[4800:] ** 2))), 0.1)
        win._playback.close()

    def test_closing_the_window_finishes_the_file(self):
        self.win._record_audio_action.trigger()
        self.frames(3)
        path = self.win._audio_take.path
        self.win.close()
        params, audio = _read_wav(path)
        self.assertEqual(params[0], 48000)
        self.assertGreater(len(audio), 0)


if __name__ == "__main__":
    unittest.main()

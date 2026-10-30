"""Converting, decoding and inspecting recordings, and saving them right.

Covers the T5 audit findings:

* fg:FG-B03, mc:MC-06, rx:RX-06, fg:FG-G03, fg:FG-U06: headerless files
  (rtl_sdr ``.bin``) are read in the format the Import Options dialog says,
  prefilled from the extension (``.bin`` is cu8) and the file name;
* fg:FG-B04, mc:MC-07, rx:UX-14: converting doesn't retune the receiver
  (the summary offers Tune there and Play);
* mc:MC-02, rx:RX-08: converting streams into a disk-backed buffer in a
  worker thread with progress and Cancel;
* mc:MC-08, rx:RX-10: the unsaved-take prompt comes first, and Cancel keeps
  a running recording running;
* mc:MC-16: an empty file is refused, the buffer kept;
* mc:MC-05, rx:RX-07, ux:UX-B12: raw saves keep the rate and frequency in a
  sidecar; unknown values are asked for, never stamped from the receiver;
* fg:FG-G22: Recording Info, cu8/cs8 output;
* fg:FG-G02, mc:MC-G03, tl:G-01, ux:UX-G03: Decode Recording;
* mc:MC-U07: dialogs remember their folder per kind.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _FakeSettings,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtWidgets import QMessageBox

RATE = 240e3


def _write_cu8(path, x):
    raw = np.empty(2 * len(x), np.uint8)
    raw[0::2] = np.clip(np.rint(x.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.rint(x.imag * 127.5 + 127.5), 0, 255)
    raw.tofile(path)


def _tone(n=48000, freq=30e3, rate=RATE, amplitude=0.5):
    t = np.arange(n) / rate
    return (amplitude * np.exp(2j * np.pi * freq * t)).astype(np.complex64)


class _FilesCase(_WindowTestCase):
    demo = True

    def setUp(self):
        super().setUp()
        self.folder = tempfile.mkdtemp(prefix="sdr-files-")
        self.addCleanup(shutil.rmtree, self.folder, True)
        self.win._recording_folder = os.path.join(self.folder, "cache")
        self.store.set("recording/folder", os.path.join(self.folder, "recordings"))
        self.boxes = []
        # Dialogs are answered (a box nobody expected fails the test instead
        # of waiting forever).
        self.answer()

    def path(self, name):
        return os.path.join(self.folder, name)

    def answer(self, *texts, accept_options=True):
        """Answer the dialogs that come up, in order: Import Options is
        accepted as prefilled; message boxes get the button named."""
        from sdr_module.gui import iq_files

        answers = list(texts)

        def fake_exec(dialog):
            self.boxes.append(dialog)
            if isinstance(dialog, iq_files.ImportOptionsDialog):
                if not accept_options:
                    return 0
                dialog._on_accept()
                return dialog.result()
            if isinstance(dialog, iq_files.RecordingDetailsDialog):
                text = answers.pop(0)
                if text == "Cancel":
                    return 0
                rate, center = text
                dialog.fill(rate, center)
                dialog._on_accept()
                return dialog.result()
            if isinstance(dialog, (iq_files.DecodeOptionsDialog,)):
                dialog._on_accept()
                return dialog.result()
            if isinstance(dialog, iq_files.RecordingInfoDialog):
                text = answers.pop(0) if answers else ""
                if text:
                    dialog._choose(text)
                return dialog.result()
            text = answers.pop(0) if answers else "Close"
            for button in dialog.buttons():
                if button.text().replace("&", "") == text:
                    button.click()
                    return 0
            raise AssertionError(f"no {text!r} button in {dialog.text()!r}")

        self.win._exec_dialog = fake_exec

    def open_dialog(self, path):
        return mock.patch(
            "sdr_module.gui.main_window.QFileDialog.getOpenFileName",
            return_value=(path, ""),
        )

    def save_dialog(self, path, chosen=""):
        return mock.patch(
            "sdr_module.gui.main_window.QFileDialog.getSaveFileName",
            return_value=(path, chosen),
        )

    def message(self):
        return self.win._message_label.text()


class TestConvertRecording(_FilesCase):
    def test_menu(self):
        actions = {
            a.text().replace("&", ""): a
            for a in self.win.menuBar().actions()[0].menu().actions()
        }
        convert = actions["Convert Recording..."]
        self.assertEqual(convert.shortcut().toString(), "Ctrl+Shift+O")
        self.assertIn("Recording Info...", actions)
        self.assertIn("Decode Recording...", actions)
        self.assertNotIn("Import Recording (convert format)...", actions)

    def test_rtl_sdr_bin_is_read_as_cu8_and_streamed_to_disk(self):
        x = _tone()
        path = self.path("rtl_433.92MHz_240k.bin")
        _write_cu8(path, x)
        self.answer("Close")
        with self.open_dialog(path):
            self.assertTrue(self.win._convert_recording())
        store = self.win._samples_buffer
        self.assertEqual(store.sample_count, len(x))
        self.assertIsNotNone(store.path)  # on disk, not in memory
        self.assertTrue(store.path.startswith(self.win._recording_folder))
        np.testing.assert_allclose(store.samples(), x, atol=1.0 / 127)
        self.assertEqual(self.win._buffer_meta, (433.92e6, 240e3))
        self.assertFalse(self.win._buffer_unsaved)
        self.assertIn("Loaded rtl_433.92MHz_240k.bin", self.message())
        options = self.boxes[0]
        self.assertEqual(
            options.windowTitle(), "Import Options – rtl_433.92MHz_240k.bin"
        )
        summary = self.boxes[1]
        self.assertIn("Sample format: cu8", summary.informativeText())

    def test_the_receiver_is_not_retuned(self):
        x = _tone()
        path = self.path("far.sigmf-data")
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

        save_iq_file(path, x, RATE, 1090e6, SampleFormat.FLOAT32, FileFormat.SIGMF)
        before = (self.freq(), self.win._lo_frequency(), self.win._device.frequency)
        self.answer("Close")
        with self.open_dialog(path):
            self.win._convert_recording()
        self.assertEqual(
            (self.freq(), self.win._lo_frequency(), self.win._device.frequency), before
        )
        self.assertTrue(self.win._is_running)
        # The summary offers to tune there instead.
        self.answer("Tune There")
        with self.open_dialog(path):
            self.win._convert_recording()
        self.assertEqual(self.freq(), 1090e6)

    def test_play_from_the_summary(self):
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

        path = self.path("p.sigmf-data")
        save_iq_file(path, _tone(), RATE, 145e6, SampleFormat.FLOAT32, FileFormat.SIGMF)
        self.answer("Play")
        with self.open_dialog(path):
            self.win._convert_recording()
        self.assertTrue(self.win._playback.active)
        self.win._playback.close()

    def test_play_from_the_summary_asks_for_what_the_file_doesnt_say(self):
        # Play was greyed out (without a word) for a file without its rate
        # and center; Recording Info's Play asks for them, so does this.
        path = self.path("plain.cs16")
        x = _tone()
        raw = np.empty(2 * len(x), np.int16)
        raw[0::2] = np.rint(x.real * 32767)
        raw[1::2] = np.rint(x.imag * 32767)
        raw.tofile(path)
        self.answer("Play")
        with self.open_dialog(path):
            self.assertTrue(self.win._convert_recording())
        asked = [b for b in self.boxes if type(b).__name__ == "ImportOptionsDialog"]
        self.assertEqual(len(asked), 2)  # to convert, then to play
        self.assertTrue(self.win._playback.active)
        self.win._playback.close()

    def test_a_running_recording_is_asked_about_first_and_cancel_keeps_it(self):
        win = self.win
        win._record_button.click()
        win._update_display()
        count = win._buffer_sample_count()
        path = self.path("x.cu8")
        _write_cu8(path, _tone())
        confirm = mock.Mock(return_value=False)  # Cancel
        win._confirm_discard_recording = confirm
        with self.open_dialog(path):
            self.assertFalse(win._convert_recording())
        confirm.assert_called_once()
        self.assertIn("converting x.cu8", confirm.call_args[0][0])
        self.assertTrue(win._recording)  # still recording
        self.assertTrue(win._record_button.isChecked())
        win._update_display()
        self.assertGreater(win._buffer_sample_count(), count)
        win._record_button.click()

    def test_the_real_prompt_cancel_keeps_recording(self):
        """The unsaved-take box itself: live wording, Cancel keeps going."""
        from sdr_module.gui.main_window import SDRMainWindow

        win = self.win
        win._confirm_discard_recording = (
            lambda doing: SDRMainWindow._confirm_discard_recording(win, doing)
        )
        win._record_button.click()
        win._update_display()
        path = self.path("y.cu8")
        _write_cu8(path, _tone())
        self.answer("Cancel")
        with self.open_dialog(path):
            win._convert_recording()
        box = [b for b in self.boxes if isinstance(b, QMessageBox)][0]
        self.assertIn("A recording is in progress", box.informativeText())
        self.assertTrue(win._recording)
        win._confirm_discard_recording = lambda *_a: True
        win._record_button.click()

    def test_saving_automatically_a_running_recording_is_still_asked_about(self):
        # With Save automatically on, converting used to stop a running
        # recording without a word (it was saved, but it stopped).
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file
        from sdr_module.gui.main_window import SDRMainWindow

        win = self.win
        win._confirm_discard_recording = (
            lambda doing: SDRMainWindow._confirm_discard_recording(win, doing)
        )
        win._control_panel.set_auto_save(True)
        path = self.path("z.sigmf-data")
        save_iq_file(path, _tone(), RATE, 145e6, SampleFormat.FLOAT32, FileFormat.SIGMF)
        win._record_button.click()
        win._update_display()
        self.answer("Cancel")
        with self.open_dialog(path):
            self.assertFalse(win._convert_recording())
        box = [b for b in self.boxes if isinstance(b, QMessageBox)][0]
        self.assertEqual(box.text(), "Stop recording and convert z.sigmf-data?")
        self.assertIn("saved automatically", box.informativeText())
        self.assertTrue(win._recording)  # Cancel: still recording
        win._update_display()
        self.answer("Stop and Convert", "Close")
        with self.open_dialog(path):
            self.assertTrue(win._convert_recording())
        self.assertFalse(win._recording)
        saved = os.listdir(self.path("recordings"))
        self.assertTrue(any(name.endswith(".sigmf-data") for name in saved), saved)
        self.assertEqual(win._buffer_source, "z.sigmf-data")

    def test_an_empty_file_is_refused_and_the_buffer_kept(self):
        win = self.win
        win._record_button.click()
        win._update_display()
        win._record_button.click()
        store = win._samples_buffer
        count = store.sample_count
        path = self.path("empty.cf32")
        open(path, "wb").close()
        with self.open_dialog(path), mock.patch.object(QMessageBox, "warning") as warn:
            self.assertFalse(win._convert_recording())
        warn.assert_called_once()
        self.assertIn("no I/Q samples", warn.call_args[0][2])
        self.assertIs(win._samples_buffer, store)
        self.assertEqual(win._buffer_sample_count(), count)

    def test_cancel_while_reading_keeps_the_buffer(self):
        from sdr_module.gui import file_jobs, main_window

        path = self.path("c.cu8")
        _write_cu8(path, _tone())
        store_before = self.win._samples_buffer
        self.answer()
        with (
            self.open_dialog(path),
            mock.patch.object(
                main_window,
                "run_job",
                return_value=file_jobs.JobResult(cancelled=True),
            ),
        ):
            self.assertFalse(self.win._convert_recording())
        self.assertIs(self.win._samples_buffer, store_before)
        self.assertIn("nothing was loaded", self.message())
        leftovers = (
            [
                n
                for n in os.listdir(self.win._recording_folder)
                if n.startswith("sdr-recording-")
            ]
            if os.path.isdir(self.win._recording_folder)
            else []
        )
        self.assertEqual(leftovers, [])

    def test_garbage_is_warned_about(self):
        path = self.path("rtl.raw")  # .raw suggests cf32: cu8 data is garbage
        _write_cu8(path, _tone())
        from sdr_module.gui import iq_files

        seen = {}

        def fake_exec(dialog):
            if isinstance(dialog, iq_files.ImportOptionsDialog):
                index = dialog._format_combo.findText(
                    iq_files.FORMAT_NAMES[iq_files.SampleFormat.FLOAT32]
                )
                dialog._format_combo.setCurrentIndex(index)
                seen["check"] = dialog.check_text()
                seen["hint"] = dialog._format_hint.text()
                return 0
            return 0

        self.win._exec_dialog = fake_exec
        with self.open_dialog(path):
            self.assertFalse(self.win._convert_recording())
        self.assertIn("probably not cf32 data", seen["check"])
        self.assertIn("cu8", seen["hint"])


class TestSaving(_FilesCase):
    def _convert(self, path):
        self.answer("Close")
        with self.open_dialog(path):
            self.assertTrue(self.win._convert_recording())

    def test_unknown_center_is_asked_for_not_stamped_from_the_receiver(self):
        from sdr_module.dsp.recording import load_iq_file

        path = self.path("capture.cs16")
        x = _tone()
        raw = np.empty(2 * len(x), np.int16)
        raw[0::2] = np.rint(x.real * 32767)
        raw[1::2] = np.rint(x.imag * 32767)
        raw.tofile(path)
        # Import Options: the rate typed, the center left unknown.
        from sdr_module.gui import iq_files

        def fake_exec(dialog):
            if isinstance(dialog, iq_files.ImportOptionsDialog):
                dialog._rate_edit.setText("1M")
                dialog._center_edit.setText("")
                dialog._on_accept()
                return dialog.result()
            if isinstance(dialog, QMessageBox):
                dialog.buttons()[-1].click()  # Close
            return 0

        self.win._exec_dialog = fake_exec
        with self.open_dialog(path):
            self.win._convert_recording()
        self.assertEqual(self.win._buffer_meta, (None, 1e6))
        # Save: asked; Cancel saves nothing.
        self.answer("Cancel")
        with self.save_dialog(self.path("out.sigmf-data")) as dialog:
            self.assertFalse(self.win._save_recording())
        dialog.assert_not_called()
        # Entered: written as entered (not the receiver's 100.1 MHz).
        self.answer(("", "433.92"))
        target = self.path("out.sigmf-data")
        with self.save_dialog(target):
            self.assertTrue(self.win._save_recording())
        _samples, meta = load_iq_file(target)
        self.assertEqual((meta.sample_rate, meta.center_frequency), (1e6, 433.92e6))

    def test_left_unknown_it_is_saved_as_unknown(self):
        path = self.path("u.cs16")
        np.zeros(2000, np.int16).tofile(path)
        from sdr_module.gui import iq_files

        def fake_exec(dialog):
            if isinstance(dialog, iq_files.ImportOptionsDialog):
                dialog._rate_edit.setText("")
                dialog._center_edit.setText("")
                dialog._on_accept()
                return dialog.result()
            if isinstance(dialog, iq_files.RecordingDetailsDialog):
                dialog._on_accept()  # both left blank
                return dialog.result()
            if isinstance(dialog, QMessageBox):
                dialog.buttons()[-1].click()
            return 0

        self.win._exec_dialog = fake_exec
        with self.open_dialog(path):
            self.win._convert_recording()
        target = self.path("u_out.cf32")
        with self.save_dialog(target):
            self.assertTrue(self.win._save_recording())
        with open(target + ".sigmf-meta") as handle:
            meta = json.load(handle)
        self.assertNotIn("core:sample_rate", meta["global"])
        self.assertNotIn("core:frequency", meta["captures"][0])

    def test_a_wav_without_a_rate_is_refused_clearly_and_leaves_no_file(self):
        # A WAV header must hold the rate: saving one left unknown failed
        # with "bad frame rate" and left a broken out.wav behind.
        path = self.path("u.cs16")
        np.zeros(2000, np.int16).tofile(path)
        from sdr_module.gui import iq_files

        def fake_exec(dialog):
            if isinstance(
                dialog, (iq_files.ImportOptionsDialog, iq_files.RecordingDetailsDialog)
            ):
                dialog._on_accept()  # rate and center left unknown
                return dialog.result()
            if isinstance(dialog, QMessageBox):
                dialog.buttons()[-1].click()
            return 0

        self.win._exec_dialog = fake_exec
        with self.open_dialog(path):
            self.win._convert_recording()
        target = self.path("u_out.wav")
        with (
            self.save_dialog(target),
            mock.patch.object(QMessageBox, "warning") as warn,
        ):
            self.assertFalse(self.win._save_recording())
        self.assertIn("needs the sample rate", warn.call_args[0][2])
        self.assertFalse(os.path.exists(target))
        self.assertFalse(os.path.exists(target + ".sigmf-meta"))

    def test_the_receiver_s_values_offered_are_its_center_and_known_rate(self):
        # "Use the receiver's" offered the tuned frequency, not the center
        # of the samples (the LO), and "0 kS/s" with no device connected.
        from PyQt6.QtWidgets import QPushButton

        from sdr_module.gui import iq_files

        win = self.win
        self.assertTrue(win.set_frequency(100.3e6))  # the VFO moves, not the LO
        self.assertEqual(win._lo_frequency(), 100.1e6)
        offered = []

        def fake_exec(dialog):
            if isinstance(dialog, iq_files.RecordingDetailsDialog):
                use = [
                    b
                    for b in dialog.findChildren(QPushButton)
                    if b.text().startswith("Use")
                ]
                offered.append(use[0].text() if use else "")
                if use:
                    use[0].click()
                offered.append((dialog._rate.text(), dialog._center.text()))
            return 0

        win._exec_dialog = fake_exec
        for device in ("demo", None):
            if device is None:
                win._release_device()
                self.assertIsNone(win._device)
            win._replace_buffer(self.mw._SampleStore.from_samples(_tone(1000)))
            win._buffer_meta = (None, None)
            self.assertFalse(win._save_recording())  # (cancelled)
        self.assertEqual(
            offered,
            [
                "Use the receiver's (2.4 MS/s, 100.100 MHz)",
                ("2.4 MS/s", "100.100 MHz"),
                "Use the receiver's (100.100 MHz)",  # no device: no rate
                ("", "100.100 MHz"),
            ],
        )

    def test_raw_saves_keep_rate_and_frequency(self):
        from sdr_module.dsp.recording import load_iq_file

        win = self.win
        win._record_button.click()
        win._update_display()
        win._record_button.click()
        center, rate = win._buffer_meta
        for name in ("take.cf32", "take.cs16", "take.raw", "take.wav", "take.cu8"):
            target = self.path(name)
            with self.save_dialog(target):
                self.assertTrue(win._save_recording(), name)
            self.assertTrue(os.path.exists(target + ".sigmf-meta"), name)
            _samples, meta = load_iq_file(target)
            self.assertEqual((meta.sample_rate, meta.center_frequency), (rate, center))
        # Converting it back recovers them without asking.
        win.set_frequency(162.55e6)
        self.answer("Close")
        with self.open_dialog(self.path("take.cs16")):
            win._convert_recording()
        self.assertEqual(win._buffer_meta, (center, rate))
        self.assertFalse(
            any(type(b).__name__ == "ImportOptionsDialog" for b in self.boxes)
        )

    def test_a_sigmf_save_keeps_when_the_take_was_recorded(self):
        # core:datetime was the time of saving (IQRecorder's start), not of
        # recording, while raw saves' sidecars had the take's start.
        win = self.win
        win._record_button.click()
        win._update_display()
        win._record_button.click()
        win._take_started = "2020-01-02T03:04:05Z"  # (recorded long ago)
        for name, meta_name in (
            ("old.sigmf-data", "old.sigmf-meta"),
            ("old.cf32", "old.cf32.sigmf-meta"),
        ):
            with self.save_dialog(self.path(name)):
                self.assertTrue(win._save_recording())
            with open(self.path(meta_name)) as handle:
                meta = json.load(handle)
            self.assertEqual(
                meta["captures"][0]["core:datetime"], "2020-01-02T03:04:05Z", name
            )

    def test_cu8_and_cs8_can_be_written(self):
        names = [spec[1] for spec in self.win._SAVE_FORMATS]
        self.assertEqual(names[0], ".sigmf-data")  # SigMF first
        self.assertIn(".cu8", names)
        self.assertIn(".cs8", names)
        self.win._record_button.click()
        self.win._update_display()
        self.win._record_button.click()
        whole = np.array(self.win._samples_buffer.samples())
        target = self.path("t.cs8")
        with self.save_dialog(target):
            self.assertTrue(self.win._save_recording())
        from sdr_module.dsp.recording import load_iq_file

        samples, meta = load_iq_file(target)
        self.assertEqual(meta.sample_format.value, "cs8")
        np.testing.assert_allclose(samples, np.clip(whole, -1, 1), atol=2.0 / 127)

    def test_suggested_name_has_frequency_and_rate_and_remembers_the_folder(self):
        self.win._record_button.click()
        self.win._update_display()
        self.win._record_button.click()
        seen = []

        def fake_save(parent, caption, directory, filters, initial):
            seen.append(directory)
            return os.path.join(self.folder, "keep", "a.sigmf-data"), initial

        os.makedirs(self.path("keep"))
        with mock.patch(
            "sdr_module.gui.main_window.QFileDialog.getSaveFileName",
            side_effect=fake_save,
        ):
            self.win._save_recording()
            self.win._save_recording()
        name = os.path.basename(seen[0])
        self.assertRegex(name, r"^iq_100\.100MHz_2\.4MSps_\d{8}_\d{6}\.sigmf-data$")
        self.assertEqual(os.path.dirname(seen[1]), self.path("keep"))
        self.assertEqual(
            self.store.get_str("recording/last_folder/iq", ""), self.path("keep")
        )


class TestUnreadableFiles(_FilesCase):
    def _sigmf(self, name, datatype, data=True):
        meta = {
            "global": {"core:datatype": datatype, "core:sample_rate": 1e6},
            "captures": [{"core:sample_start": 0, "core:frequency": 1e8}],
            "annotations": [],
        }
        with open(self.path(name + ".sigmf-meta"), "w") as handle:
            json.dump(meta, handle)
        if data:
            np.zeros(64, np.complex64).tofile(self.path(name + ".sigmf-data"))
        return self.path(name + ".sigmf-meta")

    def test_big_endian_and_missing_data_are_refused_clearly(self):
        for path, why in (
            (self._sigmf("be", "cf32_be"), "sample type cf32_be isn't one"),
            (self._sigmf("gone", "cf32_le", data=False), "gone.sigmf-data isn't"),
        ):
            with (
                self.open_dialog(path),
                mock.patch.object(QMessageBox, "warning") as warn,
            ):
                self.assertFalse(self.win._open_recording())
            self.assertIn(why, warn.call_args[0][2])
            self.assertFalse(self.win._playback.active)

    def test_an_audio_wav_is_not_taken_for_iq(self):
        from sdr_module.dsp.recording import save_audio_file

        path = self.path("145.800MHz_FM_20261030_120000.wav")
        save_audio_file(path, np.zeros(4800, np.float32), 48000)
        with self.open_dialog(path), mock.patch.object(QMessageBox, "warning") as warn:
            self.assertFalse(self.win._open_recording())
        self.assertIn("it is audio, not I/Q", warn.call_args[0][2])

    def test_wav_samples_wider_than_16_bits_are_refused_not_misread(self):
        # 24- and 32-bit I/Q WAV files (HDSDR can write them) were read as
        # 16-bit: garbage, with "the samples look right".
        import wave

        x = _tone(4800)
        for width in (3, 4):
            full = 2 ** (8 * width - 1) - 1
            ints = np.empty(2 * len(x), np.int32)
            ints[0::2] = np.rint(x.real * full)
            ints[1::2] = np.rint(x.imag * full)
            data = ints.astype("<i4").view(np.uint8).reshape(-1, 4)[:, :width]
            path = self.path(f"iq_{8 * width}bit.wav")
            with wave.open(path, "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(width)
                handle.setframerate(48000)
                handle.writeframes(data.tobytes())
            for action in (self.win._open_recording, self.win._show_recording_info):
                with (
                    self.open_dialog(path),
                    mock.patch.object(QMessageBox, "warning") as warn,
                ):
                    action()
                self.assertIn(f"its samples are {8 * width}-bit", warn.call_args[0][2])
            self.assertFalse(self.win._playback.active)


class TestRecordingInfo(_FilesCase):
    def test_info_of_a_sigmf_file(self):
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file
        from sdr_module.gui.iq_files import RecordingInfoDialog, describe_iq_file

        path = self.path("i.sigmf-data")
        save_iq_file(
            path, _tone(), RATE, 433.92e6, SampleFormat.FLOAT32, FileFormat.SIGMF
        )
        options, missing = describe_iq_file(path)
        dialog = RecordingInfoDialog(options)
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.value("Type"), "SigMF recording")
        self.assertEqual(dialog.value("Sample rate"), "240 kS/s")
        self.assertEqual(dialog.value("Center frequency"), "433.920 MHz")
        self.assertEqual(dialog.value("Duration"), "0:00.2")
        self.assertEqual(dialog.value("Samples"), "48,000")
        self.assertIn("cu8, cs8", dialog.value("Converts to"))
        self.assertEqual(dialog.value("Metadata"), "SigMF metadata")
        self.assertIn("look right", dialog._check.text())

    def test_info_menu_then_play(self):
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

        path = self.path("j.sigmf-data")
        save_iq_file(path, _tone(), RATE, 145e6, SampleFormat.FLOAT32, FileFormat.SIGMF)
        self.answer("play")
        with self.open_dialog(path):
            self.win._show_recording_info()
        self.assertTrue(self.win._playback.active)
        self.win._playback.close()

    def test_info_of_a_headerless_file_asks_first(self):
        path = self.path("h.bin")
        _write_cu8(path, _tone())
        self.answer("")
        with self.open_dialog(path):
            self.win._show_recording_info()
        kinds = [type(b).__name__ for b in self.boxes]
        self.assertEqual(kinds, ["ImportOptionsDialog", "RecordingInfoDialog"])
        info = self.boxes[1]
        self.assertTrue(info.value("Sample format").startswith("cu8"))
        self.assertEqual(
            info.value("Metadata"), "Entered by you (the file doesn't say)"
        )


class TestDecodeRecording(_FilesCase):
    def _aprs_file(self, offset=0.0, center=144.39e6):
        from sdr_module.dsp import protocol_encoders as enc
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

        burst = enc.aprs_baseband(
            "N0CALL-9", "APZ001", "!4903.50N/07201.75W>From a file", RATE, ["WIDE1-1"]
        ).astype(np.complex64)
        pad = np.zeros(int(0.3 * RATE), np.complex64)
        x = np.concatenate([pad, burst, pad, pad])
        if offset:
            x = (x * np.exp(2j * np.pi * offset * np.arange(len(x)) / RATE)).astype(
                np.complex64
            )
        path = self.path("aprs.sigmf-data")
        save_iq_file(path, x, RATE, center, SampleFormat.FLOAT32, FileFormat.SIGMF)
        return path

    def test_decodes_into_the_decoder_table_with_the_file_named(self):
        path = self._aprs_file(offset=-50e3, center=144.44e6)
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("AX.25/APRS")
        self.settle(2)
        self.answer()
        with self.open_dialog(path):
            self.assertTrue(self.win._decode_recording())
        decode = self.boxes[-1]
        # Prefilled on the APRS channel, which is inside the file.
        self.assertAlmostEqual(decode.offset(), -50e3, delta=1.0)
        texts = [m["content"] for m in panel._messages]
        self.assertTrue(
            any("From a file" in t and "from aprs.sigmf-data at" in t for t in texts),
            texts,
        )
        self.assertIn(
            "Decoded 1 AX.25/APRS message(s) from aprs.sigmf-data", self.message()
        )
        self.assertIs(
            self.win._right_tabs.currentWidget(), self.win._panel_pages["decoder"]
        )

    def test_nothing_found_says_what_to_check(self):
        path = self._aprs_file()
        self.win._decoder_panel._proto_combo.setCurrentText("POCSAG")
        self.answer()
        with self.open_dialog(path):
            self.win._decode_recording()
        self.assertIn("No POCSAG messages in aprs.sigmf-data", self.message())

    def test_decoder_tab_button(self):
        panel = self.win._decoder_panel
        self.assertEqual(panel._decode_file_btn.text(), "Decode File...")
        with mock.patch.object(self.win, "_decode_recording") as decode:
            # (Connected with a lambda: the patched method is looked up.)
            panel._decode_file_btn.click()
        decode.assert_called_once_with()

    def test_offset_outside_the_file_is_refused(self):
        from sdr_module.gui.iq_files import DecodeOptionsDialog, describe_iq_file

        options, _missing = describe_iq_file(self._aprs_file())
        dialog = DecodeOptionsDialog(options, ["AX.25/APRS"], "AX.25/APRS", 0.0)
        self.addCleanup(dialog.deleteLater)
        dialog.set_frequency_text("146.0")
        dialog._on_accept()
        self.assertIsNone(dialog.choice())
        self.assertIn("isn't inside the recording", dialog._note.text())


class TestOneLogRecordPerStop(_FilesCase):
    """mc:MC-17: an automatic stop is listed once in Error History."""

    def test_disk_full_stop_logs_once(self):
        from collections import namedtuple

        usage = namedtuple("Usage", "total used free")(10**12, 0, 420 * 1024**2)
        self.win._record_button.click()
        self.win._update_display()
        self.win._next_space_check = 0.0
        with (
            mock.patch.object(self.mw.shutil, "disk_usage", return_value=usage),
            self.assertLogs("sdr_module.gui.main_window", level="WARNING") as logs,
        ):
            self.win._update_display()
        records = [r for r in logs.records if "Recording stopped" in r.getMessage()]
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0].getMessage().startswith("Status bar error"))

    def test_start_failure_logs_once(self):
        self.win._stop_acquisition()
        with (
            mock.patch.object(
                self.win._device, "start_rx", side_effect=RuntimeError("USB error")
            ),
            self.assertLogs("sdr_module.gui.main_window", level="WARNING") as logs,
        ):
            self.win._start_acquisition()
        records = [r for r in logs.records if "USB error" in r.getMessage()]
        self.assertEqual(len(records), 1)


class TestFileDialogFolders(unittest.TestCase):
    """mc:MC-U07: each kind of file remembers its own folder."""

    def setUp(self):
        require_pyqt6(self)
        self.folder = tempfile.mkdtemp(prefix="sdr-dialogs-")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_kinds_remember_their_own_folder(self):
        from sdr_module.gui import file_dialogs

        store = _FakeSettings()
        a = os.path.join(self.folder, "a")
        b = os.path.join(self.folder, "b")
        os.makedirs(a)
        os.makedirs(b)
        file_dialogs.remember_folder(store, "iq", os.path.join(a, "x.cf32"))
        file_dialogs.remember_folder(store, "screenshot", b)
        self.assertEqual(file_dialogs.start_folder(store, "iq"), a)
        self.assertEqual(file_dialogs.start_folder(store, "screenshot"), b)
        shutil.rmtree(a)  # gone: back to the default
        default = file_dialogs.start_folder(store, "iq")
        self.assertNotEqual(default, a)

    def test_defaults(self):
        from sdr_module.gui import file_dialogs

        self.assertTrue(
            file_dialogs.default_folder("iq").endswith(
                os.path.join("Documents", "SDR Recordings")
            )
            or file_dialogs.default_folder("iq").endswith("SDR Recordings")
        )
        self.assertEqual(
            file_dialogs.default_folder("screenshot"), file_dialogs.pictures_folder()
        )
        store = _FakeSettings()
        self.assertEqual(
            file_dialogs.recordings_folder(store),
            file_dialogs.default_recordings_folder(),
        )

    def test_open_and_save_use_and_remember_the_folder(self):
        from sdr_module.gui import file_dialogs

        store = _FakeSettings()
        store.set("recording/last_folder/iq", self.folder)
        target = os.path.join(self.folder, "sub")
        os.makedirs(target)
        with mock.patch.object(
            file_dialogs.QFileDialog,
            "getSaveFileName",
            return_value=(os.path.join(target, "n.wav"), "WAV"),
        ) as dialog:
            path, chosen = file_dialogs.save_file(
                None, store, "iq", "Save", "n.wav", "WAV (*.wav)"
            )
        self.assertEqual(dialog.call_args[0][2], os.path.join(self.folder, "n.wav"))
        self.assertEqual((path, chosen), (os.path.join(target, "n.wav"), "WAV"))
        self.assertEqual(store.get_str("recording/last_folder/iq", ""), target)

    def test_unique_names(self):
        from sdr_module.gui import file_dialogs

        open(os.path.join(self.folder, "a.wav"), "w").close()
        open(os.path.join(self.folder, "b.sigmf-meta"), "w").close()
        self.assertEqual(
            os.path.basename(file_dialogs.unique_path(self.folder, "a.wav")), "a_2.wav"
        )
        self.assertEqual(
            os.path.basename(file_dialogs.unique_path(self.folder, "b.sigmf-data")),
            "b_2.sigmf-data",
        )


class TestFileNameHints(unittest.TestCase):
    def test_patterns(self):
        from sdr_module.dsp.recording import SampleFormat
        from sdr_module.gui.iq_files import guess_from_filename as guess

        cases = {
            "iq_100.100MHz_2.4MSps_20261030_120000.sigmf-data": (2.4e6, 100.1e6),
            "gqrx_20230101_123456_100100000_2400000_fc.raw": (2.4e6, 100.1e6),
            "SDRSharp_20230101_123456Z_100100000Hz_IQ.wav": (None, 100.1e6),
            "HDSDR_20230101_123456Z_100100kHz_RF.wav": (None, 100.1e6),
            "g001_433.92M_250k.cu8": (250e3, 433.92e6),
            "capture_1090MHz_2Msps.bin": (2e6, 1090e6),
            "rtl_sdr_1698765432.bin": (None, None),  # a time stamp
            "capture_433.92MHz": (None, 433.92e6),
        }
        for name, (rate, center) in cases.items():
            hints = guess(name)
            self.assertEqual((hints.sample_rate, hints.center_hz), (rate, center), name)
        self.assertIs(guess("rtl.bin").sample_format, SampleFormat.UINT8)
        self.assertIs(guess("x.cs8").sample_format, SampleFormat.INT8)
        self.assertIsNone(guess("x.iq").sample_format)

    def test_rate_text(self):
        from sdr_module.gui.iq_files import parse_rate_text as parse

        self.assertEqual(parse("2.4"), 2.4e6)
        self.assertEqual(parse("2048000"), 2.048e6)
        self.assertEqual(parse("250k"), 250e3)
        self.assertEqual(parse("250 kS/s"), 250e3)
        self.assertEqual(parse("2,4 MS/s"), 2.4e6)
        self.assertIsNone(parse("fast"))
        self.assertIsNone(parse("500M"))


if __name__ == "__main__":
    unittest.main()


class TestInputWhileAJobRuns(_FilesCase):
    """run_job keeps the window's event loop going; until its progress
    dialog shows (after a moment) nothing else blocked the window, so a
    click on Record during a Convert started a recording that then went on
    into the converted file's buffer."""

    def test_clicks_and_keys_wait_until_the_job_is_done(self):
        import time

        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtTest import QTest

        from sdr_module.gui.file_jobs import run_job

        win = self.win
        win._update_display()
        self.assertTrue(win._is_running)

        def press():
            QTest.mouseClick(win._record_button, Qt.MouseButton.LeftButton)
            win._spectrum.setFocus()
            QTest.keyClick(win._spectrum, Qt.Key.Key_Space)

        def work(report):
            for i in range(30):  # 0.3 s: the progress dialog doesn't show
                time.sleep(0.01)
                report(i / 30)
            return "done"

        QTimer.singleShot(50, press)
        # (The moment before the progress dialog shows is what's tested: with
        # the computer busy, the dialog could show before the press, and
        # offscreen its modality doesn't stop a click made by QTest.)
        result = run_job(win, "Test", "Working…", work, show_after_ms=60_000)
        self.assertEqual(result.value, "done")
        self.assertFalse(win._recording)
        self.assertTrue(win._is_running)
        # Afterwards the window takes input as before.
        QTest.mouseClick(win._record_button, Qt.MouseButton.LeftButton)
        self.assertTrue(win._recording)
        win._record_button.click()


class TestRunJob(unittest.TestCase):
    """The worker thread behind Convert, Save and Decode."""

    def setUp(self):
        require_pyqt6(self)

    def test_cancel_stops_the_work(self):
        import time

        from PyQt6.QtCore import QTimer
        from PyQt6.QtWidgets import QApplication, QProgressDialog

        from sdr_module.gui.file_jobs import run_job

        steps = []

        def work(report):
            for i in range(400):
                time.sleep(0.005)
                steps.append(i)
                report(i / 400)
            return "done"

        def press_cancel():
            for widget in QApplication.topLevelWidgets():
                if isinstance(widget, QProgressDialog) and widget.isVisible():
                    widget.canceled.emit()  # what its Cancel button does
                    return
            QTimer.singleShot(20, press_cancel)

        QTimer.singleShot(100, press_cancel)
        result = run_job(None, "Test", "Working…", work, show_after_ms=0)
        self.assertTrue(result.cancelled)
        self.assertFalse(result.ok)
        self.assertLess(len(steps), 400)

    def test_results_errors_and_partial_results(self):
        from sdr_module.gui.file_jobs import run_job

        result = run_job(None, "Test", "…", lambda report: 42)
        self.assertTrue(result.ok)
        self.assertEqual(result.value, 42)
        failed = run_job(None, "Test", "…", lambda report: 1 / 0)
        self.assertFalse(failed.ok)
        self.assertIsInstance(failed.error, ZeroDivisionError)
        got = []

        def work(report, partial):
            for i in range(3):
                partial(i)
            return None

        run_job(None, "Test", "…", work, on_partial=got.append)
        self.assertEqual(got, [0, 1, 2])

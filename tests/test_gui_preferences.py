"""Tools > Preferences (Ctrl+,) and the audio output device (mc:MC-G07).

The audio output list and QAudioSink are faked (there is no sound card
here): the sink must open the output chosen, fall back to the default when
it is unplugged, and the toolbar's Audio button must say which one plays.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication

from sdr_module.gui import audio_sink as audio_sink_module  # noqa: E402
from sdr_module.gui.audio_sink import AudioSink  # noqa: E402


class FakeAudioDevice:
    def __init__(self, ident, name, null=False):
        self._id = ident.encode()
        self._name = name
        self._null = null

    def id(self):
        return self._id

    def description(self):
        return self._name

    def isNull(self):  # noqa: N802 (Qt API)
        return self._null


SPEAKERS = FakeAudioDevice("alsa.speakers", "Speakers")
HEADSET = FakeAudioDevice("usb.headset", "USB Headset")


class FakeMediaDevices:
    outputs = [SPEAKERS, HEADSET]

    @staticmethod
    def audioOutputs():  # noqa: N802 (Qt API)
        return list(FakeMediaDevices.outputs)

    @staticmethod
    def defaultAudioOutput():  # noqa: N802 (Qt API)
        outputs = FakeMediaDevices.outputs
        return outputs[0] if outputs else FakeAudioDevice("", "", null=True)


class FakeQAudioSink:
    opened = []

    def __init__(self, device, fmt):
        self.device = device
        FakeQAudioSink.opened.append(device.description())

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


def _fake_audio(test):
    """Patch the audio backend for ``test`` (undone afterwards)."""
    FakeMediaDevices.outputs = [SPEAKERS, HEADSET]
    FakeQAudioSink.opened = []
    for name, value in (
        ("QMediaDevices", FakeMediaDevices),
        ("QAudioSink", FakeQAudioSink),
    ):
        patcher = mock.patch.object(audio_sink_module, name, value)
        patcher.start()
        test.addCleanup(patcher.stop)


@unittest.skipUnless(audio_sink_module.HAS_QT_AUDIO, "needs QtMultimedia")
class TestAudioSinkOutput(unittest.TestCase):
    def setUp(self):
        _fake_audio(self)

    def test_the_default_until_one_is_chosen(self):
        sink = AudioSink()
        self.assertTrue(sink.start(48000))
        self.assertEqual(FakeQAudioSink.opened, ["Speakers"])
        self.assertEqual(sink.output_name(), "System default (Speakers)")

    def test_the_chosen_output_is_opened(self):
        sink = AudioSink()
        sink.set_output_device("usb.headset", "USB Headset")
        self.assertTrue(sink.start(48000))
        self.assertEqual(FakeQAudioSink.opened, ["USB Headset"])
        self.assertEqual(sink.output_name(), "USB Headset")
        self.assertFalse(sink.output_missing)

    def test_an_open_stream_moves_to_the_new_output(self):
        sink = AudioSink()
        self.assertTrue(sink.start(48000))
        sink.set_output_device("usb.headset", "USB Headset")
        self.assertEqual(FakeQAudioSink.opened, ["Speakers", "USB Headset"])
        self.assertTrue(sink.is_open)

    def test_an_unplugged_output_falls_back_to_the_default(self):
        sink = AudioSink()
        sink.set_output_device("usb.headset", "USB Headset")
        FakeMediaDevices.outputs = [SPEAKERS]
        self.assertTrue(sink.start(48000))
        self.assertEqual(FakeQAudioSink.opened, ["Speakers"])
        self.assertTrue(sink.output_missing)
        self.assertEqual(sink.output_name(), "USB Headset (not found; using Speakers)")

    def test_a_replugged_output_is_taken_again(self):
        sink = AudioSink()
        sink.set_output_device("usb.headset", "USB Headset")
        FakeMediaDevices.outputs = [SPEAKERS]
        self.assertTrue(sink.start(48000))
        self.assertTrue(sink.output_missing)
        FakeMediaDevices.outputs = [SPEAKERS, HEADSET]  # plugged in again
        sink.set_output_device("usb.headset", "USB Headset")
        self.assertEqual(FakeQAudioSink.opened, ["Speakers", "USB Headset"])
        self.assertFalse(sink.output_missing)

    def test_output_list(self):
        self.assertEqual(
            audio_sink_module.output_devices(),
            [("alsa.speakers", "Speakers"), ("usb.headset", "USB Headset")],
        )


class _PrefsCase(_WindowTestCase):
    def setUp(self):
        super().setUp()
        _fake_audio(self)
        self.folder = tempfile.mkdtemp(prefix="sdr-t11-prefs-")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def run_dialog(self, edit=None, accept=True):
        """Tools > Preferences..., with ``edit(dialog)`` done in it."""
        seen = []

        def run(dialog):
            seen.append(dialog)
            if edit is not None:
                edit(dialog)
            return 1 if accept else 0

        self.win._exec_dialog = run
        self.win._show_preferences()
        return seen[0]

    def folder_dialog(self, *folders):
        """QFileDialog.getExistingDirectory answering ``folders`` in turn."""
        answers = list(folders)
        return mock.patch(
            "sdr_module.gui.file_dialogs.QFileDialog.getExistingDirectory",
            side_effect=lambda *a, **k: answers.pop(0),
        )


class TestPreferences(_PrefsCase):
    def test_in_the_tools_menu_with_ctrl_comma(self):
        actions = [
            a
            for top in self.win.menuBar().actions()
            if top.menu() is not None
            for a in top.menu().actions()
        ]
        action = next(a for a in actions if a.text() == "&Preferences...")
        self.assertEqual(action.shortcut().toString(), "Ctrl+,")

    def test_choose_the_audio_output(self):
        def pick(dialog):
            combo = dialog._output_combo
            items = [combo.itemText(i) for i in range(combo.count())]
            self.assertEqual(
                items, ["System default (Speakers)", "Speakers", "USB Headset"]
            )
            combo.setCurrentIndex(combo.findData("usb.headset"))

        self.run_dialog(pick)
        self.assertEqual(self.store.values["prefs/audio_output"], "usb.headset")
        self.assertEqual(self.store.values["prefs/audio_output_name"], "USB Headset")
        self.assertEqual(self.win._audio.output_id, "usb.headset")
        self.assertIn("Output: USB Headset", self.win._audio_button.toolTip())
        self.assertIn("USB Headset", self.win._message_label.text())

    def test_refresh_finds_an_output_plugged_in_later(self):
        def plug_in(dialog):
            FakeMediaDevices.outputs = [SPEAKERS]
            dialog._fill_outputs(keep=True)
            self.assertEqual(dialog._output_combo.count(), 2)
            FakeMediaDevices.outputs = [SPEAKERS, HEADSET]
            dialog._refresh_btn.click()
            self.assertEqual(dialog._output_combo.count(), 3)

        self.run_dialog(plug_in, accept=False)

    def test_a_chosen_output_not_plugged_in_is_kept_and_named(self):
        self.store.values["prefs/audio_output"] = "usb.headset"
        self.store.values["prefs/audio_output_name"] = "USB Headset"
        FakeMediaDevices.outputs = [SPEAKERS]

        def check(dialog):
            self.assertEqual(
                dialog._output_combo.currentText(), "USB Headset (not connected)"
            )

        self.run_dialog(check, accept=False)

    def test_folders(self):
        recordings = os.path.join(self.folder, "rec")
        shots = os.path.join(self.folder, "shots")
        capture = os.path.join(self.folder, "tmp")
        for path in (recordings, shots, capture):
            os.makedirs(path)

        def change(dialog):
            with self.folder_dialog(recordings, shots, capture):
                dialog._choose_recordings()
                dialog._choose_screenshots()
                dialog._choose_capture()
            self.assertIn("free there", dialog._space_hint.text())
            self.assertIn("a minute", dialog._space_hint.text())
            self.assertTrue(dialog._default_btn.isEnabled())

        self.run_dialog(change)
        from sdr_module.gui import file_dialogs

        self.assertEqual(file_dialogs.recordings_folder(self.store), recordings)
        self.assertEqual(file_dialogs.start_folder(self.store, "screenshot"), shots)
        self.assertEqual(self.store.values["prefs/capture_folder"], capture)
        self.assertEqual(self.win._control_panel.recordings_folder(), recordings)
        self.assertEqual(self.win._capture_folder(), capture)

    def test_cancel_changes_nothing(self):
        other = os.path.join(self.folder, "other")
        os.makedirs(other)
        before = dict(self.store.values)

        def change(dialog):
            with self.folder_dialog(other, other):
                dialog._choose_recordings()
                dialog._choose_screenshots()
            dialog._output_combo.setCurrentIndex(2)
            dialog._confirm_check.setChecked(False)

        self.run_dialog(change, accept=False)
        self.assertEqual(self.store.values, before)
        self.assertEqual(self.win._audio.output_id, "")

    def test_transmit_confirmation(self):
        def untick(dialog):
            self.assertTrue(dialog._confirm_check.isChecked())
            dialog._confirm_check.setChecked(False)

        self.run_dialog(untick)
        self.assertFalse(self.store.values["tx/confirm"])
        self.assertFalse(
            self.win._tx_station._win._settings.get_bool("tx/confirm", True)
        )

    def test_an_unusable_temp_folder_falls_back_to_the_default(self):
        gone = os.path.join(self.folder, "gone")
        self.store.values["prefs/capture_folder"] = gone
        self.win._recording_folder = None
        with self.assertLogs("sdr_module.gui.preferences_dialog", "WARNING"):
            folder = self.win._capture_folder()
        self.assertNotEqual(folder, gone)

        def check(dialog):
            self.assertIn("can't be written to", dialog._space_hint.text())
            dialog._default_btn.click()
            self.assertIn("(default)", dialog._capture_label.text())

        self.run_dialog(check)
        self.assertEqual(self.store.values["prefs/capture_folder"], "")

    def test_every_button_says_what_it_does(self):
        """(Review) The three Change... buttons had no tooltip."""
        from PyQt6.QtWidgets import QPushButton

        def check(dialog):
            standard = set(dialog._buttons.buttons())
            buttons = [b for b in dialog.findChildren(QPushButton) if b not in standard]
            self.assertGreaterEqual(len(buttons), 5)
            for button in buttons:
                self.assertTrue(button.toolTip(), button.text())

        self.run_dialog(check, accept=False)

    def test_save_screenshot_starts_in_the_screenshots_folder(self):
        """(Review) File > Save Screenshot... ignored the Screenshots folder
        chosen here: it opened in the working directory, suggesting
        "sdr.png"."""
        shots = os.path.join(self.folder, "shots")
        os.makedirs(shots)

        def change(dialog):
            with self.folder_dialog(shots):
                dialog._choose_screenshots()

        self.run_dialog(change)
        starts = []

        def save(parent, caption, start, *args, **kwargs):
            starts.append(start)
            return os.path.join(shots, "window"), "PNG (*.png)"

        with mock.patch(
            "sdr_module.gui.file_dialogs.QFileDialog.getSaveFileName",
            side_effect=save,
        ):
            self.win._save_screenshot()
        self.assertEqual(os.path.dirname(starts[0]), shots)
        self.assertRegex(
            os.path.basename(starts[0]), r"^sdr_\d+\.\d{3}MHz_\d{8}_\d{6}\.png$"
        )
        self.assertTrue(os.path.exists(os.path.join(shots, "window.png")))

    def test_a_running_recording_keeps_watching_its_own_drive(self):
        """(Review) A temp folder chosen while an I/Q recording runs is for
        the next one: the running take's free space (it stops below 500 MB)
        is still checked where it is written, not in the new folder."""
        import numpy as np

        first = os.path.join(self.folder, "first")
        second = os.path.join(self.folder, "second")
        for path in (first, second):
            os.makedirs(path)
        self.store.values["prefs/capture_folder"] = first
        self.win._recording_folder = None
        self.win._start_demo_mode()
        self.win._toggle_recording(True)
        self.win._capture(np.zeros(4096, np.complex64))  # the take's file
        self.assertEqual(self.win._samples_buffer.folder, first)

        def change(dialog):
            with self.folder_dialog(second):
                dialog._choose_capture()

        self.run_dialog(change)
        self.assertEqual(self.win._capture_folder(), second)  # the next take
        checked = []
        real = shutil.disk_usage
        with mock.patch(
            "sdr_module.gui.main_window.shutil.disk_usage",
            side_effect=lambda path: checked.append(path) or real(path),
        ):
            self.assertIsNotNone(self.win._free_recording_space())
        self.assertEqual(checked, [first])
        self.win._toggle_recording(False)
        self.win._release_device()

    def test_saved_output_is_used_at_start(self):
        self.store.values["prefs/audio_output"] = "usb.headset"
        self.store.values["prefs/audio_output_name"] = "USB Headset"
        other = self.mw.SDRMainWindow()
        self.addCleanup(other.deleteLater)
        self.addCleanup(other.close)
        self.assertEqual(other._audio.output_id, "usb.headset")
        QApplication.processEvents()


if __name__ == "__main__":
    unittest.main()

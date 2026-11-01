"""Regression tests for the recordings code review (findings REC-xx).

* REC-03: with Save automatically on, a recording stopped because the disk
  was nearly full was copied in full onto that same drive: it filled the
  drive to the last byte, failed, and left broken files. Now the take is
  moved (instantly, needing no space) when it can be, and a copy that
  would leave the drive nearly full is refused; a save that doesn't fit is
  refused before anything is written.
* REC-01 / REC-13: a take whose sample rate (or frequency) is unknown
  couldn't be saved as SigMF, cs16, cu8 or cs8 ("float division by zero"),
  and SigMF wrote an unknown frequency as 0 Hz.
* REC-04: a save that failed (a full disk) left a truncated data file and an
  empty ``.sigmf-meta`` behind.
* REC-05: after the temp drive filled up while recording, the take could
  not be saved anywhere (the unwritable buffered tail failed every save).
* REC-06: a take over 4 GiB failed as WAV only after writing 4 GiB.
* REC-07: ``.cf32``/``.raw`` saves copied the take on the GUI thread (no
  progress or Cancel) and left a partial file when the copy failed.
* REC-08 / REC-09: Convert then Save as SigMF stamped the time of saving as
  the recording's time and dropped its hardware and description; Save As
  suggested (and wrote over) the converted file itself.
* REC-10: Decode Recording placed each message at the end of its 1M-sample
  chunk (up to 22 s late), not where it is in the file.
* REC-12: cu8/cs8/cs16 saves truncated instead of rounding (a DC offset).
* REC-14: the recording buffer kept a Python tuple per display frame.
* REC-15: a take resumed after a pause or REC WAITING was saved as one
  continuous capture, so its times after the gap were wrong.
"""

import contextlib
import errno
import json
import os
import shutil
import signal
import tempfile
import threading
import time
import tracemalloc
import unittest
from collections import namedtuple
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_recording_files import RATE, _FilesCase  # noqa: E402

try:
    import resource
except ImportError:  # pragma: no cover - not on Windows
    resource = None  # type: ignore[assignment]

if HAS_PYQT6:
    from PyQt6.QtWidgets import QMessageBox

Usage = namedtuple("Usage", "total used free")


def _can_limit_file_size():
    return resource is not None and signal.getsignal(signal.SIGXFSZ) in (
        signal.SIG_IGN,
    )


@contextlib.contextmanager
def file_size_limit(nbytes):
    """Writing past ``nbytes`` of any file fails (EFBIG) while this lasts,
    as writing to a full disk does: short writes, then errors."""
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    resource.setrlimit(resource.RLIMIT_FSIZE, (nbytes, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))


def _listing(folder):
    if not os.path.isdir(folder):
        return {}
    return {
        name: os.path.getsize(os.path.join(folder, name)) for name in os.listdir(folder)
    }


def _when(text):
    from datetime import datetime

    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


class _ReviewCase(_FilesCase):
    def record(self, frames=3, stop=True):
        win = self.win
        win._record_button.click()
        for _ in range(frames):
            win._update_display()
        if stop:
            win._record_button.click()

    def recordings_folder(self):
        return self.path("recordings")

    def free(self, megabytes):
        usage = Usage(10**12, 0, int(megabytes * 10**6))
        return mock.patch.object(self.mw.shutil, "disk_usage", return_value=usage)

    def set_format(self, ext):
        combo = self.win._control_panel._format_combo
        index = combo.findData(ext)
        self.assertGreaterEqual(index, 0, ext)
        combo.setCurrentIndex(index)
        self.assertEqual(self.win._panel_recording_extension(), ext)

    def stop_for_disk_space(self, megabytes):
        """The free-space check stops the recording (``megabytes`` left)."""
        win = self.win
        win._next_space_check = 0.0
        with self.free(megabytes):
            win._update_display()
            self.assertFalse(win._recording)
            self.settle(5)  # the automatic save runs right after the frame

    def warnings(self):
        return mock.patch.object(QMessageBox, "warning")


# ---------------------------------------------------------------------------
# REC-03: Save automatically after a disk-nearly-full stop
# ---------------------------------------------------------------------------


class TestAutoSaveOnANearlyFullDisk(_ReviewCase):
    def setUp(self):
        super().setUp()
        self.win._control_panel.set_auto_save(True)

    def test_the_take_is_moved_not_copied_onto_the_full_drive(self):
        self.assertEqual(self.win._panel_recording_extension(), ".sigmf-data")
        self.record(3, stop=False)
        store = self.win._samples_buffer
        temp, count = store.path, store.sample_count
        with mock.patch.object(
            self.mw, "write_recording", side_effect=self.mw.write_recording
        ) as write:
            self.stop_for_disk_space(420)
        write.assert_not_called()  # no second copy on the nearly full drive
        self.assertFalse(os.path.exists(temp))  # moved
        names = sorted(_listing(self.recordings_folder()))
        self.assertEqual(len(names), 2, names)
        data = [n for n in names if n.endswith(".sigmf-data")][0]
        meta = _read_json(os.path.join(self.recordings_folder(), names[1]))
        self.assertEqual(meta["global"]["core:sample_rate"], 2.4e6)
        self.assertEqual(meta["captures"][0]["core:frequency"], 100.1e6)
        self.assertEqual(
            os.path.getsize(os.path.join(self.recordings_folder(), data)), count * 8
        )
        self.assertFalse(self.win._buffer_unsaved)
        self.assertIn("It was saved to", self.message())

    def test_a_copy_that_would_fill_the_drive_is_refused_and_the_take_kept(self):
        self.set_format(".cs16")  # not the take's own layout: a full copy
        self.record(3, stop=False)
        count = self.win._buffer_sample_count()
        with self.assertLogs("sdr_module.gui.main_window", level="ERROR"):
            self.stop_for_disk_space(420)
        self.assertEqual(_listing(self.recordings_folder()), {})  # nothing written
        self.assertTrue(self.win._buffer_unsaved)
        self.assertEqual(self.win._buffer_sample_count(), count)
        text = self.message()
        # (It fits in the 420 MB, but recording keeps 500 MB free.)
        self.assertIn("would be nearly full", text)
        self.assertIn("recording keeps 500 MB free", text)
        self.assertIn("It is kept", text)

    def test_a_take_that_doesnt_fit_at_all_says_so(self):
        # Not "nearly full" when it doesn't fit at all: there isn't room.
        self.set_format(".cs16")
        self.record(3, stop=False)
        with self.assertLogs("sdr_module.gui.main_window", level="ERROR"):
            self.stop_for_disk_space(0.5)
        self.assertEqual(_listing(self.recordings_folder()), {})
        self.assertTrue(self.win._buffer_unsaved)
        text = self.message()
        self.assertIn("there isn't enough space (it needs ", text)
        self.assertIn("has 500 kB free)", text)
        self.assertNotIn("nearly full", text)
        # (Said once: what to do is the auto-save's own advice.)
        self.assertNotIn("free up some space", text)
        self.assertIn("It is kept: save it elsewhere", text)
        # Nobody watches an automatic save: it stays as long as the warning
        # it replaces ("Recording stopped: ..."), not a 6 s blink.
        self.assertEqual(self.win._message_timer.interval(), 20000)

    def test_another_drive_with_room_still_gets_a_copy(self):
        self.set_format(".cs16")
        self.record(3, stop=False)
        free = {self.win._samples_buffer.folder: 420}

        def usage(folder):
            return Usage(10**12, 0, free.get(folder, 10**6) * 1024**2)

        self.win._next_space_check = 0.0
        with mock.patch.object(self.mw.shutil, "disk_usage", side_effect=usage):
            self.win._update_display()
            self.settle(5)
        names = _listing(self.recordings_folder())
        self.assertEqual(len([n for n in names if n.endswith(".cs16")]), 1, names)
        self.assertFalse(self.win._buffer_unsaved)


class TestSaveRoom(_ReviewCase):
    def test_a_save_that_doesnt_fit_is_refused_before_writing(self):
        self.record(3)
        target = self.path("take.cs16")
        with self.save_dialog(target), self.free(0.5), self.warnings() as warn:
            self.assertFalse(self.win._save_recording())
        self.assertIn("isn't enough space", warn.call_args[0][2])
        self.assertFalse(os.path.exists(target))
        self.assertTrue(self.win._buffer_unsaved)

    def test_a_stopped_take_saved_as_sigmf_is_moved_instantly(self):
        self.record(3)
        temp, count = self.win._samples_buffer.path, self.win._buffer_sample_count()
        target = self.path("take.sigmf-data")
        with (
            self.save_dialog(target),
            mock.patch.object(self.mw, "run_job") as job,
            self.free(0.5),  # (no room needed)
        ):
            self.assertTrue(self.win._save_recording())
        job.assert_not_called()
        self.assertFalse(os.path.exists(temp))
        self.assertEqual(os.path.getsize(target), count * 8)
        meta = _read_json(self.path("take.sigmf-meta"))
        self.assertEqual(meta["global"]["core:datatype"], "cf32_le")
        # The saved file is the user's: discarding the take keeps it.
        self.win._discard_recording()
        self.assertTrue(os.path.exists(target))

    def test_never_into_the_take_s_own_temporary_file(self):
        self.record(3)
        temp = self.win._samples_buffer.path
        size = os.path.getsize(temp)
        with self.save_dialog(temp), self.warnings() as warn:
            self.assertFalse(self.win._save_recording())
        self.assertIn("temporary file", warn.call_args[0][2])
        self.assertEqual(os.path.getsize(temp), size)
        self.assertTrue(self.win._samples_buffer.temporary)

    def test_saved_again_to_the_same_file_keeps_the_samples(self):
        self.record(3)
        whole = np.array(self.win._samples_buffer.samples())
        target = self.path("take.sigmf-data")
        for _ in range(2):
            with self.save_dialog(target):
                self.assertTrue(self.win._save_recording())
        np.testing.assert_array_equal(np.fromfile(target, np.complex64), whole)
        self.assertTrue(os.path.exists(self.path("take.sigmf-meta")))


# ---------------------------------------------------------------------------
# REC-01 / REC-13: unknown sample rate and frequency
# ---------------------------------------------------------------------------


class TestUnknownRate(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="sdr-review-rec-")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_every_format_but_wav_saves_with_the_rate_unknown(self):
        from sdr_module.dsp.recording import FileFormat, SampleFormat
        from sdr_module.gui.file_jobs import write_recording

        x = (0.3 * np.exp(2j * np.pi * 0.01 * np.arange(10000))).astype(np.complex64)
        for fmt, sample_fmt, name in (
            (FileFormat.SIGMF, SampleFormat.FLOAT32, "take.sigmf-data"),
            (FileFormat.RAW, SampleFormat.INT16, "take.cs16"),
            (FileFormat.RAW, SampleFormat.UINT8, "take.cu8"),
            (FileFormat.RAW, SampleFormat.INT8, "take.cs8"),
            (FileFormat.RAW, SampleFormat.FLOAT32, "take.raw"),
        ):
            path = os.path.join(self.folder, name)
            written = write_recording(
                iter([x]), len(x), path, fmt, sample_fmt, 0.0, 0.0, lambda _f: None
            )
            self.assertEqual(written, len(x), name)
        meta = _read_json(os.path.join(self.folder, "take.sigmf-meta"))
        self.assertNotIn("core:sample_rate", meta["global"])  # unknown, not 0
        self.assertNotIn("core:frequency", meta["captures"][0])  # REC-13

    def test_recorder_duration_with_an_unknown_rate(self):
        from sdr_module.dsp.recording import AudioRecorder, IQRecorder

        self.assertEqual(IQRecorder(0.0).duration_seconds, 0.0)
        self.assertEqual(AudioRecorder(sample_rate=0).duration_seconds, 0.0)

    def test_an_unknown_frequency_is_left_out_of_sigmf(self):
        # REC-13 alone: the rate is known, the center frequency isn't (it
        # was written as 0.0, a recording centered at 0 Hz to other tools).
        from sdr_module.dsp.recording import (
            FileFormat,
            SampleFormat,
            load_iq_file,
            save_iq_file,
        )

        path = os.path.join(self.folder, "take.sigmf-data")
        save_iq_file(
            path, np.zeros(64, np.complex64), 1e6, 0.0, file_format=FileFormat.SIGMF
        )
        meta = _read_json(os.path.join(self.folder, "take.sigmf-meta"))
        self.assertEqual(meta["global"]["core:sample_rate"], 1e6)
        self.assertNotIn("core:frequency", meta["captures"][0])
        _samples, info = load_iq_file(path)
        self.assertEqual((info.sample_rate, info.center_frequency), (1e6, 0.0))
        self.assertEqual(info.sample_format, SampleFormat.FLOAT32)


class TestUnknownRateInTheWindow(_ReviewCase):
    def test_a_converted_capture_left_unknown_saves_as_sigmf_and_cs16(self):
        from sdr_module.gui import iq_files

        path = self.path("capture.cs16")
        np.zeros(2000, np.int16).tofile(path)

        def fake_exec(dialog):
            if isinstance(
                dialog, (iq_files.ImportOptionsDialog, iq_files.RecordingDetailsDialog)
            ):
                dialog._on_accept()  # rate and center left blank
                return dialog.result()
            if isinstance(dialog, QMessageBox):
                dialog.buttons()[-1].click()
            return 0

        self.win._exec_dialog = fake_exec
        with self.open_dialog(path):
            self.assertTrue(self.win._convert_recording())
        self.assertEqual(self.win._buffer_meta, (None, None))
        for name, meta_name in (
            ("out.sigmf-data", "out.sigmf-meta"),
            ("out.cs16", "out.cs16.sigmf-meta"),
        ):
            with self.save_dialog(self.path(name)), self.warnings() as warn:
                self.assertTrue(self.win._save_recording(), name)
            warn.assert_not_called()
            meta = _read_json(self.path(meta_name))
            self.assertNotIn("core:sample_rate", meta["global"], name)
            self.assertNotIn("core:frequency", meta["captures"][0], name)


# ---------------------------------------------------------------------------
# REC-04: a failed save leaves nothing behind
# ---------------------------------------------------------------------------


class TestFailedSavesLeaveNothing(unittest.TestCase):
    def setUp(self):
        if not _can_limit_file_size():
            self.skipTest("needs RLIMIT_FSIZE (POSIX)")
        self.folder = tempfile.mkdtemp(prefix="sdr-review-rec-")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_a_disk_filling_up_mid_save(self):
        from sdr_module.dsp.recording import FileFormat, SampleFormat
        from sdr_module.gui.file_jobs import write_recording

        x = np.full(1 << 16, 0.25 + 0.5j, np.complex64)  # 512 KB as cf32
        # (A full disk takes nothing more, not even SigMF's small metadata
        # when the write is being closed: 300 bytes is less than it needs.)
        for fmt, sample_fmt, name, room in (
            (FileFormat.SIGMF, SampleFormat.FLOAT32, "take.sigmf-data", 300),
            (FileFormat.SIGMF, SampleFormat.FLOAT32, "take.sigmf-data", 100_000),
            (FileFormat.WAV, SampleFormat.INT16, "take.wav", 100_000),
            (FileFormat.RAW, SampleFormat.INT16, "take.cs16", 100_000),
        ):
            path = os.path.join(self.folder, name)
            with self.assertRaises(OSError), file_size_limit(room):
                write_recording(
                    iter([x[:30000], x[30000:]]),
                    len(x),
                    path,
                    fmt,
                    sample_fmt,
                    RATE,
                    100e6,
                    lambda _f: None,
                )
            self.assertEqual(_listing(self.folder), {}, name)

    def test_sigmf_metadata_that_cant_be_written_leaves_no_empty_file(self):
        from sdr_module.dsp.recording import FileFormat, IQRecorder, SampleFormat

        path = os.path.join(self.folder, "take.sigmf-data")
        meta_path = os.path.join(self.folder, "take.sigmf-meta")
        with open(meta_path, "w") as handle:
            handle.write('{"global": {}, "captures": [], "annotations": []}')
        recorder = IQRecorder(RATE, 100e6, SampleFormat.FLOAT32, FileFormat.SIGMF)
        recorder.set_metadata(description="x" * 4000)
        recorder.start(path)
        recorder.write(np.zeros(16, np.complex64))
        with self.assertRaises(OSError), file_size_limit(1000):
            recorder.stop()
        self.assertFalse(recorder.is_recording)
        # The old metadata is as it was (not emptied), and no temporary
        # file is left beside it.
        self.assertEqual(_read_json(meta_path)["captures"], [])
        self.assertEqual(
            sorted(_listing(self.folder)), ["take.sigmf-data", "take.sigmf-meta"]
        )


# ---------------------------------------------------------------------------
# REC-05: the temp drive fills up while recording
# ---------------------------------------------------------------------------


class TestTempDriveFillsWhileRecording(_ReviewCase):
    def setUp(self):
        super().setUp()
        if not _can_limit_file_size():
            self.skipTest("needs RLIMIT_FSIZE (POSIX)")

    def test_what_was_recorded_can_be_saved_elsewhere(self):
        from sdr_module.dsp.recording import load_iq_file

        win = self.win
        win._record_button.click()
        win._update_display()
        block = win._buffer_sample_count() * 8
        limit = int(2.5 * block) + 12  # (not a whole number of samples)
        with file_size_limit(limit):
            for _ in range(5):
                win._update_display()
                # (Written by the take's writer thread, K-P1: the next frame
                # hears of a write that failed.)
                win._samples_buffer.flush()
            self.assertFalse(win._recording)  # stopped: the file can't grow
            store = win._samples_buffer
            count = store.sample_count
            # Every sample counted is on disk (none stuck in a buffer).
            self.assertEqual(os.path.getsize(store.path), count * 8)
            self.assertEqual(count, limit // 8)
            self.assertIn("on another drive if this one is full", self.message())
            # Saved (to another drive, in a format that fits there).
            target = self.path("elsewhere.cu8")
            with self.save_dialog(target), self.warnings() as warn:
                self.assertTrue(win._save_recording())
            warn.assert_not_called()
        samples, meta = load_iq_file(target)
        self.assertEqual(len(samples), count)
        self.assertEqual(meta.sample_rate, 2.4e6)

    def test_a_failing_save_is_reported_not_raised(self):
        self.record(3)
        with (
            mock.patch.object(
                self.mw._SampleStore, "samples", side_effect=OSError(28, "No space")
            ),
            self.save_dialog(self.path("take.cs16")),
            self.warnings() as warn,
        ):
            self.assertFalse(self.win._save_recording())
        self.assertIn("No space", warn.call_args[0][2])


# ---------------------------------------------------------------------------
# REC-06: WAV files hold 4 GiB at most
# ---------------------------------------------------------------------------


class TestWavSizeLimit(_ReviewCase):
    def test_too_long_for_wav_is_refused_before_writing(self):
        from sdr_module.dsp.recording import FileFormat, SampleFormat
        from sdr_module.gui.file_jobs import write_recording

        path = self.path("long.wav")
        total = (1 << 30) + 1  # (4 GiB and a bit as 16-bit I/Q)
        with self.assertRaises(ValueError) as caught:
            write_recording(
                iter([]),
                total,
                path,
                FileFormat.WAV,
                SampleFormat.INT16,
                2.4e6,
                100e6,
                lambda _f: None,
            )
        self.assertIn("4 GiB", str(caught.exception))
        self.assertIn("about 7 minutes", str(caught.exception))
        self.assertFalse(os.path.exists(path))

    def test_save_refuses_and_auto_save_falls_back_to_sigmf(self):
        from sdr_module.gui import file_jobs

        self.record(3)
        count = self.win._buffer_sample_count()
        small = count * 4 - 1  # (this take is "over 4 GiB")
        with mock.patch.object(file_jobs, "WAV_MAX_DATA_BYTES", small, create=True):
            target = self.path("take.wav")
            with self.save_dialog(target), self.warnings() as warn:
                self.assertFalse(self.win._save_recording())
            self.assertIn("too long for a WAV file", warn.call_args[0][2])
            self.assertFalse(os.path.exists(target))
            self.set_format(".wav")
            self.win._control_panel.set_auto_save(True)
            self.assertTrue(self.win._auto_save_take())
        names = _listing(self.recordings_folder())
        self.assertEqual(
            sorted(os.path.splitext(n)[1] for n in names),
            [".sigmf-data", ".sigmf-meta"],
        )
        self.assertIn("as SigMF", self.message())


# ---------------------------------------------------------------------------
# REC-07: copies run in the background and leave nothing when they fail
# ---------------------------------------------------------------------------


class TestCopiesRunInTheBackground(_ReviewCase):
    def test_saving_cf32_while_recording_copies_in_a_worker(self):
        from sdr_module.dsp.recording import IQRecorder

        self.record(3, stop=False)
        whole = np.array(self.win._samples_buffer.samples())
        threads = []
        real_write = IQRecorder.write

        def spy(recorder, samples):
            threads.append(threading.current_thread())
            return real_write(recorder, samples)

        target = self.path("running.cf32")
        with self.save_dialog(target), mock.patch.object(IQRecorder, "write", spy):
            self.assertTrue(self.win._save_recording())
        self.assertTrue(threads)
        self.assertNotIn(threading.main_thread(), threads)
        np.testing.assert_array_equal(np.fromfile(target, np.complex64), whole)
        self.assertTrue(self.win._recording)  # still going
        self.win._record_button.click()

    def test_a_failed_copy_leaves_no_partial_file(self):
        if not _can_limit_file_size():
            self.skipTest("needs RLIMIT_FSIZE (POSIX)")
        self.record(3, stop=False)  # (running: copied, not moved)
        self.win._samples_buffer.finish()  # (the take itself is all on disk)
        target = self.path("partial.cf32")
        with (
            self.save_dialog(target),
            self.warnings() as warn,
            file_size_limit(100_000),
            self.assertLogs("sdr_module.gui.main_window", level="ERROR"),
        ):
            self.assertFalse(self.win._save_recording())
        self.assertIn("Could not save recording", warn.call_args[0][2])
        self.assertFalse(os.path.exists(target))
        self.assertFalse(os.path.exists(target + ".sigmf-meta"))
        self.win._record_button.click()


# ---------------------------------------------------------------------------
# REC-08 / REC-09: converting keeps the metadata and the original
# ---------------------------------------------------------------------------


class TestConvertKeepsTheRecording(_ReviewCase):
    META = {
        "global": {
            "core:datatype": "cf32_le",
            "core:sample_rate": 250000.0,
            "core:version": "1.0.0",
            "core:hw": "RTL-SDR v3 + QFH",
            "core:description": "NOAA-19 pass",
        },
        "captures": [
            {
                "core:sample_start": 0,
                "core:frequency": 137.1e6,
                "core:datetime": "2020-01-02T03:04:05Z",
            }
        ],
        "annotations": [],
    }

    def _source(self, name="pass"):
        x = (0.3 * np.exp(2j * np.pi * 0.01 * np.arange(50000))).astype(np.complex64)
        x.tofile(self.path(name + ".sigmf-data"))
        with open(self.path(name + ".sigmf-meta"), "w") as handle:
            json.dump(self.META, handle)
        return self.path(name + ".sigmf-data"), x

    def _convert(self, path):
        self.answer("Close")
        with self.open_dialog(path):
            self.assertTrue(self.win._convert_recording())

    def test_time_hardware_and_description_are_kept(self):
        path, _x = self._source()
        self._convert(path)
        os.makedirs(self.path("out"))
        with self.save_dialog(self.path("out/copy.sigmf-data")):
            self.assertTrue(self.win._save_recording())
        meta = _read_json(self.path("out/copy.sigmf-meta"))
        self.assertEqual(meta["captures"][0]["core:datetime"], "2020-01-02T03:04:05Z")
        self.assertEqual(meta["global"]["core:hw"], "RTL-SDR v3 + QFH")
        self.assertEqual(meta["global"]["core:description"], "NOAA-19 pass")
        with self.save_dialog(self.path("out/copy.cs16")):
            self.assertTrue(self.win._save_recording())
        sidecar = _read_json(self.path("out/copy.cs16.sigmf-meta"))
        self.assertEqual(
            sidecar["captures"][0]["core:datetime"], "2020-01-02T03:04:05Z"
        )
        self.assertEqual(sidecar["global"]["core:hw"], "RTL-SDR v3 + QFH")

    def test_an_unknown_time_is_left_out_not_stamped_now(self):
        from sdr_module.gui import iq_files

        path = self.path("capture.cs16")
        np.zeros(2000, np.int16).tofile(path)

        def fake_exec(dialog):
            if isinstance(dialog, iq_files.ImportOptionsDialog):
                dialog._rate_edit.setText("1M")
                dialog._center_edit.setText("433.92")
                dialog._on_accept()
                return dialog.result()
            if isinstance(dialog, QMessageBox):
                dialog.buttons()[-1].click()
            return 0

        self.win._exec_dialog = fake_exec
        with self.open_dialog(path):
            self.assertTrue(self.win._convert_recording())
        with self.save_dialog(self.path("capture.sigmf-data")):
            self.assertTrue(self.win._save_recording())
        meta = _read_json(self.path("capture.sigmf-meta"))
        self.assertNotIn("core:datetime", meta["captures"][0])

    def test_save_as_never_suggests_or_overwrites_the_source(self):
        path, x = self._source()
        self._convert(path)
        self.assertEqual(
            self.win._suggested_iq_name(".sigmf-data"), "pass_converted.sigmf-data"
        )
        self.assertEqual(self.win._suggested_iq_name(".cs16"), "pass.cs16")
        # (Opened by its metadata file: still the same recording.)
        self.win._buffer_source = "pass.sigmf-meta"
        self.assertEqual(
            self.win._suggested_iq_name(".sigmf-data"), "pass_converted.sigmf-data"
        )
        self.win._buffer_source = "pass.sigmf-data"
        before = {
            name: os.path.getmtime(self.path(name)) for name in os.listdir(self.folder)
        }
        for target in ("pass.sigmf-data", "pass.sigmf-meta"):
            with self.save_dialog(self.path(target)), self.warnings() as warn:
                self.assertFalse(self.win._save_recording(), target)
            self.assertIn("the recording you converted", warn.call_args[0][2])
        after = {
            name: os.path.getmtime(self.path(name)) for name in os.listdir(self.folder)
        }
        self.assertEqual(after, before)
        self.assertEqual(_read_json(self.path("pass.sigmf-meta")), self.META)
        np.testing.assert_array_equal(np.fromfile(path, np.complex64), x)

    def test_a_sidecar_source_is_protected_too(self):
        from sdr_module.dsp.recording import SampleFormat, write_sidecar

        x = (0.2 * np.ones(4000)).astype(np.complex64)
        path = self.path("cap.cf32")
        x.tofile(path)
        write_sidecar(path, SampleFormat.FLOAT32, 1e6, 433.92e6)
        self._convert(path)
        with self.save_dialog(path), self.warnings() as warn:
            self.assertFalse(self.win._save_recording())
        self.assertIn("the recording you converted", warn.call_args[0][2])
        np.testing.assert_array_equal(np.fromfile(path, np.complex64), x)

    GAP = [(0, "2020-01-02T03:04:05Z"), (20000, "2020-01-02T03:09:00.250Z")]

    def _captures(self, meta_path):
        return [
            (c["core:sample_start"], c.get("core:datetime"))
            for c in _read_json(meta_path)["captures"]
        ]

    def test_a_recording_with_a_gap_keeps_its_captures(self):
        # A take resumed after a pause is saved with a capture per stretch
        # (REC-15); converted, it was labelled one continuous capture from
        # its first time again.
        path, _x = self._source("gap")
        meta = json.loads(json.dumps(self.META))
        meta["captures"].append(
            {"core:sample_start": 20000, "core:datetime": self.GAP[1][1]}
        )
        with open(self.path("gap.sigmf-meta"), "w") as handle:
            json.dump(meta, handle)
        self._convert(path)
        os.makedirs(self.path("out"))
        for name, meta_name in (
            ("out/copy.sigmf-data", "out/copy.sigmf-meta"),
            ("out/copy.cs16", "out/copy.cs16.sigmf-meta"),
        ):
            with self.save_dialog(self.path(name)):
                self.assertTrue(self.win._save_recording(), name)
            self.assertEqual(self._captures(self.path(meta_name)), self.GAP, name)

    def test_a_sidecar_with_a_gap_is_read_too(self):
        from sdr_module.dsp.recording import SampleFormat, write_sidecar
        from sdr_module.gui.iq_files import describe_iq_file

        path = self.path("cap.cs16")
        np.zeros(2 * 30000, np.int16).tofile(path)
        write_sidecar(
            path,
            SampleFormat.INT16,
            1e6,
            433.92e6,
            start_time=self.GAP[0][1],
            segments=self.GAP[1:] + [(90000, "2020-01-02T04:00:00Z")],
        )
        options, _missing = describe_iq_file(path)
        self.assertEqual(options.start_time, self.GAP[0][1])
        self.assertEqual(options.segments[0], self.GAP[1])
        self._convert(path)
        with self.save_dialog(self.path("copy.sigmf-data")):
            self.assertTrue(self.win._save_recording())
        # (A capture past the end of the samples is left out.)
        self.assertEqual(self._captures(self.path("copy.sigmf-meta")), self.GAP)


# ---------------------------------------------------------------------------
# REC-10: Decode Recording says where in the file each message is
# ---------------------------------------------------------------------------


class TestDecodeTimes(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="sdr-review-rec-")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def _aprs_file(self, rate, seconds, at):
        from sdr_module.dsp import protocol_encoders as enc
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

        burst = enc.aprs_baseband(
            "N0CALL-9", "APZ001", "!4903.50N/07201.75W>Where", rate, ["WIDE1-1"]
        ).astype(np.complex64)
        x = np.zeros(int(seconds * rate), np.complex64)
        start = int(at * rate)
        x[start : start + len(burst)] = burst
        path = os.path.join(self.folder, "aprs.sigmf-data")
        save_iq_file(path, x, rate, 144.39e6, SampleFormat.FLOAT32, FileFormat.SIGMF)
        return path, len(burst) / rate

    def test_a_message_is_placed_where_it_ends_in_the_file(self):
        from sdr_module.gui.file_jobs import decode_recording
        from sdr_module.gui.iq_files import describe_iq_file

        for rate, seconds in ((48e3, 30.0), (240e3, 8.0)):
            path, length = self._aprs_file(rate, seconds, at=2.0)
            options, _missing = describe_iq_file(path)
            found = []
            result = decode_recording(
                options, "AX.25/APRS", 0.0, lambda _f: None, found.extend
            )
            self.assertEqual(result.valid, 1, rate)
            when = [t for _msg, t in result.messages]
            self.assertEqual([t for _msg, t in found], when)
            # It ends at 2.0 s + its length; reported within 0.25 s of that
            # (it was 21.8 s at 48 kS/s: the end of its 1M-sample chunk).
            self.assertAlmostEqual(when[0], 2.0 + length, delta=0.25, msg=rate)


# ---------------------------------------------------------------------------
# REC-12: integer formats are rounded, not truncated
# ---------------------------------------------------------------------------


class TestRounding(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.mkdtemp(prefix="sdr-review-rec-")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_no_dc_offset_or_dead_zone(self):
        from sdr_module.dsp.recording import (
            FileFormat,
            FormatConverter,
            SampleFormat,
            load_iq_file,
            save_iq_file,
        )

        rng = np.random.default_rng(12)
        n = 200_000
        noise = 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        x = noise.astype(np.complex64)
        for fmt in (SampleFormat.UINT8, SampleFormat.INT8, SampleFormat.INT16):
            path = os.path.join(self.folder, f"noise.{fmt.value}")
            save_iq_file(path, x, 1e6, 100e6, fmt, FileFormat.RAW)
            back, _meta = load_iq_file(path, fmt)
            step = 1 / 127.5 if fmt is SampleFormat.UINT8 else 1 / 127
            if fmt is SampleFormat.INT16:
                step = 1 / 32767
            # (Truncating gave a mean of -0.5 step on I and Q for cu8.)
            self.assertLess(abs(back.mean() - x.mean()), 0.05 * step, fmt)
        # A weak tone (under one step of cs8) isn't zeroed by a dead zone.
        t = np.arange(4000)
        tone = (0.76 / 127 * np.exp(2j * np.pi * 0.01 * t)).astype(np.complex64)
        path = os.path.join(self.folder, "tone.cs8")
        save_iq_file(path, tone, 1e6, 100e6, SampleFormat.INT8, FileFormat.RAW)
        back, _meta = load_iq_file(path, SampleFormat.INT8)
        self.assertGreater(np.abs(back).max(), 0.5 / 127)
        # The in-memory converter rounds the same way.
        converted = FormatConverter().convert_sample_format(
            np.array([0.002 + 0j], np.complex64), SampleFormat.UINT8
        )
        self.assertEqual(converted.tolist(), [128, 128])  # 127.755 (was [127, 127])
        converted = FormatConverter().convert_sample_format(
            np.array([0.006 - 0.006j], np.complex64), SampleFormat.INT8
        )
        self.assertEqual(converted.tolist(), [1, -1])  # (was [0, 0])


# ---------------------------------------------------------------------------
# REC-14: the recording buffer's block list
# ---------------------------------------------------------------------------


class TestBlockList(unittest.TestCase):
    def setUp(self):
        from tests.test_gui_chrome import require_pyqt6

        require_pyqt6(self)
        from sdr_module.gui import main_window

        self.mw = main_window
        self.folder = tempfile.mkdtemp(prefix="sdr-review-rec-")
        self.addCleanup(shutil.rmtree, self.folder, True)

    def test_blocks_cost_bytes_not_python_objects(self):
        store = self.mw._SampleStore(self.folder)
        self.addCleanup(store.discard)
        block = np.ones(1, np.complex64)
        store.append(block)
        frames = 30_000  # (over 15 minutes of display frames)
        tracemalloc.start()
        try:
            before = tracemalloc.get_traced_memory()[0]
            for _ in range(frames):
                store.append(block)
            grown = tracemalloc.get_traced_memory()[0] - before
        finally:
            tracemalloc.stop()
        # A tuple per block took about 128 bytes each (3.8 MB here).
        self.assertLess(grown / frames, 24)
        self.assertEqual(len(store), frames + 1)

    def test_the_list_of_blocks_api(self):
        store = self.mw._SampleStore(self.folder)
        self.addCleanup(store.discard)
        self.assertFalse(store)
        blocks = [np.arange(n, dtype=np.complex64) + 1j * n for n in (3, 1, 4, 1, 5)]
        for block in blocks:
            store.append(block)
        self.assertTrue(store)
        self.assertEqual(len(store), 5)
        self.assertEqual(store, blocks)
        np.testing.assert_array_equal(store[2], blocks[2])
        np.testing.assert_array_equal(store[-1], blocks[-1])
        with self.assertRaises(IndexError):
            store[5]
        memory = self.mw._SampleStore.from_samples(blocks[2])
        self.assertEqual(len(memory), 1)
        np.testing.assert_array_equal(memory[0], blocks[2])
        store.discard()
        self.assertEqual((len(store), store.sample_count), (0, 0))


# ---------------------------------------------------------------------------
# REC-15: a take resumed after a gap
# ---------------------------------------------------------------------------


class TestResumedTake(_ReviewCase):
    def _captures(self, name="take.sigmf-data", meta_name="take.sigmf-meta"):
        with self.save_dialog(self.path(name)):
            self.assertTrue(self.win._save_recording())
        return _read_json(self.path(meta_name))["captures"]

    def test_a_pause_starts_a_new_capture_with_its_own_time(self):
        win = self.win
        win._record_button.click()
        win._update_display()
        first = win._buffer_sample_count()
        win._on_panel_record_paused(True)
        win._update_display()  # (not captured)
        time.sleep(0.05)
        win._on_panel_record_paused(False)
        win._update_display()
        win._record_button.click()
        captures = self._captures()
        self.assertEqual([c["core:sample_start"] for c in captures], [0, first])
        self.assertEqual(captures[0]["core:datetime"], win._take_started)
        start, resumed = (_when(c["core:datetime"]) for c in captures)
        self.assertGreaterEqual(resumed, start)
        for capture in captures:
            self.assertEqual(capture["core:frequency"], 100.1e6)
        # The sidecar of a raw file says so too.
        sidecar = self._captures("take.cs16", "take.cs16.sigmf-meta")
        self.assertEqual([c["core:sample_start"] for c in sidecar], [0, first])

    def test_the_receiver_stopped_while_recording(self):
        win = self.win
        win._record_button.click()
        win._update_display()
        first = win._buffer_sample_count()
        win._stop_acquisition()  # REC WAITING
        win._start_acquisition()
        win._update_display()
        win._update_display()
        win._record_button.click()
        captures = self._captures()
        self.assertEqual([c["core:sample_start"] for c in captures], [0, first])

    def test_a_continuous_take_is_one_capture(self):
        self.record(4)
        self.assertEqual(len(self._captures()), 1)

    def test_a_resume_asks_the_time_module_only_what_tests_fake(self):
        # Tests stand in for main_window's ``time`` with a clock that has
        # monotonic, perf_counter, strftime and gmtime (the display loop's
        # and a recording's needs); the stamp of a resumed stretch failed
        # under it ("'_Clock' object has no attribute 'time'").
        from tests.test_gui_round3_main import _FakeClock

        class Clock(_FakeClock):
            gmtime = staticmethod(time.gmtime)

        win = self.win
        with mock.patch.object(self.mw, "time", Clock()):
            win._record_button.click()
            win._update_display()
            first = win._buffer_sample_count()
            win._on_panel_record_paused(True)
            win._on_panel_record_paused(False)
            win._update_display()
            win._record_button.click()
        self.assertGreater(win._buffer_sample_count(), first)
        captures = self._captures()
        self.assertEqual([c["core:sample_start"] for c in captures], [0, first])
        _when(captures[1]["core:datetime"])  # (a valid time)


# ---------------------------------------------------------------------------
# REC-04 (checker): a save that fails on a full disk says so
# ---------------------------------------------------------------------------


class TestAFailedSaveSaysWhy(_ReviewCase):
    """A save's own error is the one reported (REC-04). The samples were
    written with numpy's ``tofile``, whose error says only "131072
    requested and 25000 written": a SigMF or cf32 save onto a disk that
    filled up said that instead of "No space left on device"."""

    def setUp(self):
        super().setUp()
        if not _can_limit_file_size():
            self.skipTest("needs RLIMIT_FSIZE (POSIX)")

    def test_the_writer_names_the_reason(self):
        from sdr_module.dsp.recording import FileFormat, SampleFormat
        from sdr_module.gui.file_jobs import write_recording

        x = np.full(1 << 16, 0.25 + 0.5j, np.complex64)  # 512 KB as cf32
        for fmt, sample_fmt, name in (
            (FileFormat.SIGMF, SampleFormat.FLOAT32, "take.sigmf-data"),
            (FileFormat.RAW, SampleFormat.FLOAT32, "take.cf32"),
            (FileFormat.RAW, SampleFormat.INT16, "take.cs16"),
        ):
            path = self.path(name)
            with self.assertRaises(OSError) as caught, file_size_limit(100_000):
                write_recording(
                    iter([x]),
                    len(x),
                    path,
                    fmt,
                    sample_fmt,
                    RATE,
                    100e6,
                    lambda _f: None,
                )
            # (A size limit fails as a full disk does, with its own errno.)
            self.assertEqual(caught.exception.errno, errno.EFBIG, name)
            self.assertFalse(os.path.exists(path), name)

    def test_the_save_failed_box_says_why(self):
        self.record(3, stop=False)  # (running: written, not moved)
        self.win._samples_buffer.flush()  # (the take itself is all on disk)
        target = self.path("take.sigmf-data")
        with (
            self.save_dialog(target),
            self.warnings() as warn,
            file_size_limit(100_000),
            self.assertLogs("sdr_module.gui.main_window", level="ERROR"),
        ):
            self.assertFalse(self.win._save_recording())
        self.assertIn(os.strerror(errno.EFBIG), warn.call_args[0][2])
        self.assertFalse(os.path.exists(target))
        self.win._record_button.click()

"""Regression tests for the recordings code review (findings REC-xx).

* REC-01 / REC-13: a take whose sample rate (or frequency) is unknown
  couldn't be saved as SigMF, cs16, cu8 or cs8 ("float division by zero"),
  and SigMF wrote an unknown frequency as 0 Hz.
* REC-04: a save that failed (a full disk) left a truncated data file and an
  empty ``.sigmf-meta`` behind.
* REC-12: cu8/cs8/cs16 saves truncated instead of rounding (a DC offset).
"""

import contextlib
import json
import os
import shutil
import signal
import tempfile
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
        usage = Usage(10**12, 0, int(megabytes * 1024**2))
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

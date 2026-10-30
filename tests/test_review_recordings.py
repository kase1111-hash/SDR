"""Regression tests for the recordings code review (findings REC-xx).

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
from tests.test_gui_recording_files import _FilesCase  # noqa: E402

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

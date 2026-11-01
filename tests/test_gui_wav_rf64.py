"""K-R1 in the window: WAV past 4 GiB is RF64, and RF64 recordings open.

Before, File > Save Recording refused a take over 4 GiB as WAV ("it is too
long for a WAV file, which holds 4 GiB at most"), Save automatically saved
it as SigMF instead, and an RF64 file (another program's WAV past 4 GiB)
couldn't be opened, inspected, converted or decoded ("file does not start
with RIFF id"). Now such a take is saved as RF64, the status bar says so,
and RF64 files play, convert, decode and show in Recording Info.

A take "over 4 GiB" is a short one with the classic WAV limit lowered
meanwhile (``wav_file.RIFF_SIZE_LIMIT``); a 5 GiB file is a sparse one
where the file system makes them.
"""

import contextlib
import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_recording_files import RATE  # noqa: E402
from tests.test_review_recordings import _ReviewCase  # noqa: E402
from tests.test_wav_rf64 import _cs16, _rf64, _sparse_files_here, _tone  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import Qt


@contextlib.contextmanager
def _small_limit(nbytes):
    """Classic WAV files hold ``nbytes`` of samples meanwhile (4 GiB
    really): the writer's limit, and file_jobs' (which refused more)."""
    from sdr_module.dsp import wav_file
    from sdr_module.gui import file_jobs

    with (
        mock.patch.object(file_jobs, "WAV_MAX_DATA_BYTES", int(nbytes)),
        mock.patch.object(wav_file, "RIFF_SIZE_LIMIT", 36 + int(nbytes)),
    ):
        yield


def _container(path):
    with open(path, "rb") as handle:
        return handle.read(4)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestSavingPastTheLimit(_ReviewCase):
    def test_save_recording_writes_rf64_and_says_so(self):
        from sdr_module.dsp.recording import IQPlayer, find_sidecar

        self.record(3)
        count = self.win._buffer_sample_count()
        whole = np.array(self.win._samples_buffer.samples()[:count])
        target = self.path("take.wav")
        with (
            _small_limit(count * 4 - 4),  # (this take is "over 4 GiB")
            self.save_dialog(target),
            self.warnings() as warn,
        ):
            self.assertTrue(self.win._save_recording())
        warn.assert_not_called()
        self.assertEqual(_container(target), b"RF64")
        with IQPlayer(target) as player:
            self.assertEqual(player.total_samples, count)
            np.testing.assert_allclose(player.read(count), whole, atol=1 / 32767)
        # Its sidecar leads a SigMF reader past the RF64 header.
        self.assertEqual(find_sidecar(target).header_bytes, 80)
        text = self.message()
        # (RF64 is said before the path, which a narrow status bar cuts.)
        self.assertTrue(
            text.startswith(
                f"Saved {count:,} samples as RF64 (the WAV format for files over "
                "4 GiB) to "
            ),
            text,
        )

    def test_save_automatically_keeps_wav(self):
        # (It saved a take too long for WAV as SigMF instead.)
        self.record(3)
        count = self.win._buffer_sample_count()
        self.set_format(".wav")
        self.win._control_panel.set_auto_save(True)
        with _small_limit(count * 2):
            self.assertTrue(self.win._auto_save_take())
        names = sorted(os.listdir(self.recordings_folder()))
        self.assertEqual(
            [os.path.splitext(n)[1] for n in names], [".wav", ".sigmf-meta"]
        )
        wav = os.path.join(self.recordings_folder(), names[0])
        self.assertEqual(_container(wav), b"RF64")
        text = self.message()
        self.assertTrue(
            text.startswith(
                "Saved the recording as RF64 (the WAV format for files over 4 GiB) "
                "to "
            ),
            text,
        )
        self.assertNotIn("SigMF", text)

    def test_a_take_that_fits_is_a_classic_wav_file(self):
        self.record(3)
        target = self.path("short.wav")
        with self.save_dialog(target), self.warnings() as warn:
            self.assertTrue(self.win._save_recording())
        warn.assert_not_called()
        with open(target, "rb") as handle:
            head = handle.read(44)
        self.assertEqual(head[:4], b"RIFF")
        self.assertEqual(head[36:40], b"data")  # (the classic 44-byte header)
        self.assertNotIn("RF64", self.message())

    def test_the_wav_format_says_what_happens_past_4_gib(self):
        combo = self.win._control_panel._format_combo
        tip = combo.itemData(combo.findData(".wav"), Qt.ItemDataRole.ToolTipRole)
        self.assertIn("RF64", tip)
        self.assertIn("4 GiB", tip)
        self.assertIn("classic WAV", tip)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestOpeningRf64Recordings(_ReviewCase):
    CENTER = 144.44e6

    def _aprs_rf64(self):
        """An APRS packet 50 kHz below the center of an RF64 file, with a
        sidecar giving the center (as this app saves one)."""
        from sdr_module.dsp import protocol_encoders as enc
        from sdr_module.dsp.recording import SampleFormat, write_sidecar

        burst = enc.aprs_baseband(
            "N0CALL-9", "APZ001", "!4903.50N/07201.75W>From RF64", RATE, ["WIDE1-1"]
        ).astype(np.complex64)
        pad = np.zeros(int(0.3 * RATE), np.complex64)
        x = np.concatenate([pad, 0.5 * burst, pad, pad])
        x = (x * np.exp(-2j * np.pi * 50e3 * np.arange(len(x)) / RATE)).astype(
            np.complex64
        )
        path = self.path("aprs.wav")
        offset = _rf64(path, len(x), head=_cs16(x), rate=int(RATE))
        write_sidecar(path, SampleFormat.INT16, RATE, self.CENTER, header_bytes=offset)
        return path, x

    def test_open_recording_plays_it(self):
        path, x = self._aprs_rf64()
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._open_recording())
        warn.assert_not_called()
        self.assertTrue(self.win._playback.active)
        self.assertAlmostEqual(self.win._lo_frequency(), self.CENTER, delta=1.0)
        device = self.win._playback.device
        self.assertEqual(device.total_samples, len(x))
        start = device.position
        for _ in range(3):
            self.win._update_display()
        self.assertGreater(device.position, start)

    def test_recording_info_names_it(self):
        from sdr_module.gui.iq_files import RecordingInfoDialog, describe_iq_file

        path, x = self._aprs_rf64()
        options, missing = describe_iq_file(path)
        self.assertEqual(missing, set())
        dialog = RecordingInfoDialog(options)
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(
            dialog.value("Type"),
            "WAV file (RF64, the WAV format for files over 4 GiB), I/Q as stereo",
        )
        self.assertEqual(dialog.value("Samples"), f"{len(x):,}")
        self.assertIn("look right", dialog._check.text())

    def test_decode_recording_finds_its_packet(self):
        path, _x = self._aprs_rf64()
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("AX.25/APRS")
        self.settle(2)
        self.answer()
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._decode_recording())
        warn.assert_not_called()
        texts = [m["content"] for m in panel._messages]
        self.assertTrue(any("From RF64" in t for t in texts), texts)

    def test_convert_recording_loads_every_sample(self):
        path, x = self._aprs_rf64()
        self.answer("Close")
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._convert_recording())
        warn.assert_not_called()
        count = self.win._buffer_sample_count()
        self.assertEqual(count, len(x))
        loaded = np.array(self.win._samples_buffer.samples()[:count])
        np.testing.assert_allclose(loaded, x, atol=1 / 32767)

    def test_find_signals_in_recording_reads_it(self):
        from sdr_module.dsp.recording import SampleFormat, write_sidecar

        rng = np.random.default_rng(6)
        n = int(2.0 * RATE)
        t = np.arange(n) / RATE
        x = 1e-3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        x = (x + 0.1 * np.exp(2j * np.pi * 30e3 * t)).astype(np.complex64)
        path = self.path("tone.wav")
        offset = _rf64(path, n, head=_cs16(x), rate=int(RATE))
        write_sidecar(path, SampleFormat.INT16, RATE, 145e6, header_bytes=offset)
        tools = self.win._signal_tools
        with self.warnings() as warn:
            self.assertTrue(tools.find_in_recording(path))
        warn.assert_not_called()
        self.assertTrue(tools.panel.showing_recording())
        found = [s.frequency_hz - 145e6 for s in tools.panel.signals()]
        self.assertTrue(any(abs(f - 30e3) < 2e3 for f in found), found)

    def test_a_5_gib_recording_plays_to_its_end(self):
        if not _sparse_files_here(self.folder):
            self.skipTest(
                "this file system makes no sparse files: 5 GiB would be written"
            )
        from sdr_module.dsp.recording import SampleFormat, write_sidecar
        from sdr_module.gui.iq_files import RecordingInfoDialog, describe_iq_file

        frames = (5 << 30) // 4
        tail = _tone(4096, cycles=0.05)
        path = self.path("long.wav")
        offset = _rf64(
            path, frames, head=_cs16(_tone(4096)), tail=_cs16(tail), rate=int(RATE)
        )
        write_sidecar(path, SampleFormat.INT16, RATE, self.CENTER, header_bytes=offset)
        options, _missing = describe_iq_file(path)
        dialog = RecordingInfoDialog(options)
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.value("Samples"), f"{frames:,}")
        self.assertEqual(dialog.value("Size"), "5.4 GB")
        with self.open_dialog(path), self.warnings() as warn:
            self.assertTrue(self.win._open_recording())
        warn.assert_not_called()
        device = self.win._playback.device
        self.assertEqual(device.total_samples, frames)
        device.seek(frames - len(tail))
        np.testing.assert_allclose(device.read_samples(len(tail)), tail, atol=1 / 32767)
        self.assertTrue(device.finished)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 required")
class TestTheOldRefusalIsGone(unittest.TestCase):
    def test_no_wav_problem_but_the_missing_rate(self):
        from sdr_module.dsp.recording import SampleFormat
        from sdr_module.gui.file_jobs import wav_problem

        self.assertEqual(wav_problem((1 << 31) + 1, SampleFormat.INT16, 2.4e6), "")
        self.assertIn("sample rate", wav_problem(1000, SampleFormat.INT16, 0.0))

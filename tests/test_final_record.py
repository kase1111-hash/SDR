"""Recordings: the problems the end-to-end journeys found (final fix wave).

* REC-E2E-01, DEC-E2E-20: quiet little-endian cs16 (and 8-bit samples
  widened to 16 bits) was guessed to be cs8; Import Options offered "Use
  cs8" and called that reading right.
* REC-E2E-08: the sample check said "peak 0.00 of full scale" for quiet
  recordings.
"""

import os

import numpy as np
import pytest

from sdr_module.dsp.recording import (
    SampleFormat,
    check_iq_samples,
    confident_sample_format,
    guess_sample_format,
    write_sidecar,
)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_recording_files import _FilesCase  # noqa: E402

N = 16384  # complex samples: 64 KiB of cs16, what Import Options reads


def _interleave(i, q):
    out = np.empty(2 * len(i))
    out[0::2] = i
    out[1::2] = q
    return out


def _cs16_noise(std, n=N, dc=(0.0, 0.0), seed=1):
    rng = np.random.default_rng(seed)
    iq = _interleave(rng.normal(dc[0], std, n), rng.normal(dc[1], std, n))
    return np.clip(np.round(iq), -32768, 32767).astype("<i2").tobytes()


def _cs8_noise(std, n=2 * N, dc=(0.0, 0.0), seed=2):
    rng = np.random.default_rng(seed)
    iq = _interleave(rng.normal(dc[0], std, n // 2), rng.normal(dc[1], std, n // 2))
    return np.clip(np.round(iq), -128, 127).astype(np.int8).tobytes()


def _cu8_noise(std, n=2 * N, seed=3):
    rng = np.random.default_rng(seed)
    iq = rng.normal(127.5, std, n)
    return np.clip(np.round(iq), 0, 255).astype(np.uint8).tobytes()


class TestQuiet16BitIsNotTakenFor8Bit:
    @pytest.mark.parametrize("std", [1, 5, 20, 34, 50, 60])
    def test_quiet_cs16_noise(self, std):
        # The high byte of a small 16-bit sample only repeats the low
        # byte's sign: it was read as a second 8-bit component.
        data = _cs16_noise(std)
        assert guess_sample_format(data) is SampleFormat.INT16
        assert confident_sample_format(data) is SampleFormat.INT16

    def test_quiet_cs16_with_a_dc_offset(self):
        data = _cs16_noise(10, dc=(40.0, -25.0))
        assert guess_sample_format(data) is SampleFormat.INT16

    @pytest.mark.parametrize("std", [2, 8, 20, 40, 100])
    def test_8_bit_samples_widened_to_16_bits(self, std):
        # SoapySDR's CS16 from an RTL-SDR or HackRF: x << 8, low byte 0.
        rng = np.random.default_rng(4)
        x8 = np.clip(np.round(rng.normal(0, std, 2 * N)), -128, 127)
        data = (x8.astype(np.int16) << 8).astype("<i2").tobytes()
        assert guess_sample_format(data) is SampleFormat.INT16

    def test_quiet_cs16_with_a_strong_burst(self):
        rng = np.random.default_rng(5)
        x = rng.normal(0, 20, 2 * N)
        burst = 1000.0 * np.cos(2 * np.pi * 0.01 * np.arange(N))
        x[N:] += burst  # half of what is read is a strong burst
        data = np.clip(np.round(x), -32768, 32767).astype("<i2").tobytes()
        assert guess_sample_format(data) is SampleFormat.INT16

    def test_the_demo_s_2_m_band_as_cs16(self):
        from sdr_module.gui.device_dialog import MockDevice

        device = MockDevice()
        device.set_frequency(144.39e6)
        device.set_sample_rate(1e6)
        device.start_rx()
        parts = device.read_samples(N).view(np.float32)
        data = np.round(parts * 32767.0).astype("<i2").tobytes()
        assert np.abs(parts).max() < 0.05  # quiet: about 30 LSB of noise
        assert guess_sample_format(data) is SampleFormat.INT16

    @pytest.mark.parametrize(
        "make, fmt",
        [
            (lambda: _cs8_noise(1), SampleFormat.INT8),
            (lambda: _cs8_noise(3), SampleFormat.INT8),
            (lambda: _cs8_noise(10), SampleFormat.INT8),
            (lambda: _cs8_noise(30), SampleFormat.INT8),
            (lambda: _cs8_noise(3, dc=(3.0, -2.0)), SampleFormat.INT8),
            (lambda: _cu8_noise(1), SampleFormat.UINT8),
            (lambda: _cu8_noise(3), SampleFormat.UINT8),
            (lambda: _cu8_noise(30), SampleFormat.UINT8),
            (lambda: bytes(65536), SampleFormat.INT8),
        ],
    )
    def test_8_bit_data_still_reads_as_8_bit(self, make, fmt):
        assert guess_sample_format(make()) is fmt

    def test_loud_noise_is_only_a_guess(self):
        # So loud that the bytes don't tell (cs8 near clipping): the guess
        # is by smoothness, and none is confident ("looks more like" would
        # mislead).
        data = _cs8_noise(70)
        assert confident_sample_format(data) is None
        assert isinstance(guess_sample_format(data), SampleFormat)
        # A strong clean signal is smooth in its own reading only: sure.
        t = np.arange(N)
        tone = _interleave(0.9 * np.cos(0.2 * t), 0.9 * np.sin(0.2 * t))
        data = np.round(tone * 127.5 + 127.5).astype(np.uint8).tobytes()
        assert confident_sample_format(data) is SampleFormat.UINT8


class TestSampleCheckFindsTheWrongReading:
    def test_cs16_read_as_cs8(self, tmp_path):
        path = tmp_path / "capture.iq"
        path.write_bytes(_cs16_noise(34, n=4 * N))
        check = check_iq_samples(path, SampleFormat.INT8)
        assert not check.ok
        assert "I and Q differ by" in check.problem
        assert "probably not cs8 data" in check.problem
        right = check_iq_samples(path, SampleFormat.INT16)
        assert right.ok and not right.note, right.problem or right.note

    def test_a_reading_mostly_at_full_scale_is_not_right(self, tmp_path):
        # cu8 noise read as cs8 sits at +/-full scale (127.5 -> -128).
        path = tmp_path / "rtl.iq"
        path.write_bytes(_cu8_noise(2.2, n=4 * N))
        for fmt in (SampleFormat.INT8, SampleFormat.INT16):
            check = check_iq_samples(path, fmt)
            assert not check.ok, fmt
            assert "at full scale" in check.problem
        assert check_iq_samples(path, SampleFormat.UINT8).ok

    def test_some_clipping_is_noted(self, tmp_path):
        rng = np.random.default_rng(6)
        x = rng.normal(0, 30, 4 * N)
        x[:40] = 127.0  # a few samples at full scale
        path = tmp_path / "hackrf.cs8"
        path.write_bytes(np.clip(np.round(x), -128, 127).astype(np.int8).tobytes())
        check = check_iq_samples(path, SampleFormat.INT8)
        assert check.ok, check.problem
        assert check.full_scale >= 40
        assert "reach full scale" in check.note
        assert "or this isn't cs8 data" in check.note
        # A file that says its format: only the clipping is possible.
        write_sidecar(path, SampleFormat.INT8, 2e6, 100e6)
        check = check_iq_samples(path)
        assert "clipped when it was recorded" in check.note
        assert "isn't cs8" not in check.note

    def test_silence_is_noted(self, tmp_path):
        path = tmp_path / "zeros.cs16"
        path.write_bytes(bytes(4 * N))
        check = check_iq_samples(path, SampleFormat.INT16)
        assert check.ok
        assert "all zero" in check.note
        assert check.peak_dbfs == -np.inf

    def test_peak_in_dbfs(self, tmp_path):
        path = tmp_path / "quiet.cs16"
        tone = 0.01 * np.exp(1j * 0.3 * np.arange(N))
        iq = _interleave(tone.real, tone.imag)
        path.write_bytes(np.round(iq * 32767).astype("<i2").tobytes())
        check = check_iq_samples(path, SampleFormat.INT16)
        assert check.peak_dbfs == pytest.approx(-40.0, abs=0.1)


class TestImportOptionsReadsQuietCs16(_FilesCase):
    def _quiet_capture(self, name):
        path = self.path(name)
        with open(path, "wb") as handle:
            handle.write(_cs16_noise(34, n=8 * N))
        return path

    def test_cs16_is_not_said_to_look_more_like_cs8(self):
        from sdr_module.gui.iq_files import ImportOptionsDialog, describe_iq_file

        path = self._quiet_capture("aprs_144.390MHz_1Msps.cs16")
        options, missing = describe_iq_file(path)
        dialog = ImportOptionsDialog(options, missing, "play")
        self.addCleanup(dialog.deleteLater)
        self.assertIs(dialog.sample_format(), SampleFormat.INT16)
        self.assertFalse(dialog._format_hint_row.isVisibleTo(dialog))
        self.assertIn("The samples look right (peak -", dialog.check_text())
        self.assertIn(" dBFS).", dialog.check_text())
        self.assertNotIn("of full scale", dialog.check_text())
        # Reading it as cs8 is called wrong, not right.
        dialog._format_combo.setCurrentIndex(
            dialog._format_combo.findData(SampleFormat.INT8)
        )
        self.assertIn("probably not cs8 data", dialog.check_text())
        self.assertNotIn("look right", dialog.check_text())

    def test_a_dat_file_is_prefilled_cs16(self):
        from sdr_module.gui.iq_files import describe_iq_file

        path = self._quiet_capture("usrp_samples.dat")
        options, missing = describe_iq_file(path)
        self.assertIn("format", missing)
        self.assertIs(options.sample_format, SampleFormat.INT16)

    def test_no_hint_from_a_wild_guess(self):
        # A loud cs8 capture: the content guess is only by smoothness, so
        # no "looks more like cs16" is offered.
        from sdr_module.gui.iq_files import ImportOptionsDialog, describe_iq_file

        path = self.path("loud.cs8")
        with open(path, "wb") as handle:
            handle.write(_cs8_noise(70, n=8 * N))
        options, missing = describe_iq_file(path)
        dialog = ImportOptionsDialog(options, missing, "play")
        self.addCleanup(dialog.deleteLater)
        self.assertIs(dialog.sample_format(), SampleFormat.INT8)
        self.assertFalse(dialog._format_hint_row.isVisibleTo(dialog))

    def test_recording_info_gives_the_peak_in_dbfs(self):
        # REC-E2E-08: "peak 0.00 of full scale" read like an empty file.
        from sdr_module.dsp.recording import FileFormat, save_iq_file
        from sdr_module.gui.iq_files import RecordingInfoDialog, describe_iq_file

        path = self.path("quiet.sigmf-data")
        tone = (0.009 * np.exp(1j * 0.3 * np.arange(48000))).astype(np.complex64)
        save_iq_file(path, tone, 1e6, 144.39e6, SampleFormat.FLOAT32, FileFormat.SIGMF)
        options, _missing = describe_iq_file(path)
        dialog = RecordingInfoDialog(options)
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(
            dialog._check.text(), "The samples look right (peak -41 dBFS)."
        )


def _save_sigmf(path, center, rate=1e6, seconds=0.2):
    from sdr_module.dsp.recording import FileFormat, save_iq_file

    rng = np.random.default_rng(7)
    n = int(rate * seconds)
    noise = 0.001 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    save_iq_file(
        path,
        noise.astype(np.complex64),
        rate,
        center,
        SampleFormat.FLOAT32,
        FileFormat.SIGMF,
    )
    return path


class TestListeningFollowsTheRecording(_FilesCase):
    """REC-E2E-02 / REC-E2E-03: a recording is heard the way its band
    usually is, and Close goes back to the listening from before."""

    def setUp(self):
        super().setUp()
        win = self.win
        self.assertTrue(win.set_frequency(100.1e6))
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        self.assertTrue(win._band_plan_action.isChecked())

    def listening(self):
        win = self.win
        return (*win._listening_mode(), win._control_panel._bw_combo.currentText())

    def play(self, path):
        with self.open_dialog(path):
            self.assertTrue(self.win._open_recording())

    def test_a_2_m_recording_plays_in_narrow_fm(self):
        self.play(_save_sigmf(self.path("aprs2m.sigmf-data"), 144.39e6))
        self.assertEqual(self.win._current_frequency(), 144.39e6)
        self.assertEqual(self.listening(), ("FM", "5 kHz", "25 kHz"))
        self.assertIn("APRS: listening in FM (narrow", self.message())

    def test_not_when_the_band_plan_is_off(self):
        self.win._band_plan_action.setChecked(False)
        self.play(_save_sigmf(self.path("aprs2m.sigmf-data"), 144.39e6))
        self.assertEqual(self.listening(), ("FM", "75 kHz", "200 kHz"))
        self.assertIn("Playing aprs2m.sigmf-data", self.message())

    def test_close_goes_back_to_the_listening_from_before(self):
        self.play(_save_sigmf(self.path("aprs2m.sigmf-data"), 144.39e6))
        self.win._set_demod_mode("FM", "2.5 kHz", "12.5 kHz")  # changed there
        self.win._playback.close(restore=True)
        self.assertEqual(self.win._current_frequency(), 100.1e6)
        self.assertEqual(self.listening(), ("FM", "75 kHz", "200 kHz"))

    def test_another_file_then_close(self):
        # From one file to another in another band: the band plan from the
        # first file's channel; Close still goes back to the very first.
        self.play(_save_sigmf(self.path("aprs2m.sigmf-data"), 144.39e6))
        self.play(_save_sigmf(self.path("fm.sigmf-data"), 98.5e6))
        self.assertEqual(self.listening(), ("FM", "75 kHz", "200 kHz"))
        self.win._set_demod_mode("AM", "", "10 kHz")
        self.win._playback.close(restore=True)
        self.assertEqual(self.win._current_frequency(), 100.1e6)
        self.assertEqual(self.listening(), ("FM", "75 kHz", "200 kHz"))


class TestImportOptionsFitsItsRows(_FilesCase):
    """REC-E2E-04: the dialog was sized before its check text was set, so
    the Sample rate and Center frequency fields lost their bottom edge."""

    def assert_rows_whole(self, dialog):
        from PyQt6.QtWidgets import QLineEdit

        layout = dialog.layout()
        self.assertGreaterEqual(
            dialog.height(), layout.totalHeightForWidth(dialog.width())
        )
        for edit in dialog.findChildren(QLineEdit):
            row = edit.parentWidget()
            self.assertGreaterEqual(
                row.height(), edit.y() + edit.height(), edit.accessibleName()
            )

    def test_rows_are_whole(self):
        from PyQt6.QtWidgets import QApplication

        from sdr_module.gui import themes
        from sdr_module.gui.iq_files import ImportOptionsDialog, describe_iq_file

        # (A name whose intro takes two lines, and a two-line check.)
        path = self.path("iss_145.75MHz_1Msps.bin")
        rng = np.random.default_rng(2)
        noise = np.round(127.5 + rng.normal(0, 2.2, 2_000_000))
        np.clip(noise, 0, 255).astype(np.uint8).tofile(path)
        options, missing = describe_iq_file(path)
        # Measured the way the app shows it: themed.
        qapp = QApplication.instance()
        old_sheet, old_palette = qapp.styleSheet(), qapp.palette()
        old_theme = themes.current_theme()
        try:
            for theme in themes.THEMES:
                themes.apply_theme(qapp, theme)
                dialog = ImportOptionsDialog(options, missing, "play", parent=self.win)
                self.addCleanup(dialog.deleteLater)
                self.assertFalse(dialog._format_hint_row.isVisibleTo(dialog))
                dialog.show()
                self.settle()
                self.assert_rows_whole(dialog)
                # A longer check (a problem over three lines) grows it too.
                dialog._rate_edit.setText("2.4 MS/s per second, surely")
                self.settle()
                self.assertIn("Can't read the sample rate", dialog.check_text())
                self.assert_rows_whole(dialog)
                dialog.hide()
        finally:
            qapp.setStyleSheet(old_sheet)
            qapp.setPalette(old_palette)
            themes.apply_theme(None, old_theme)


class _TestClock:
    """The ``time`` module of main_window and the demo device, run by hand."""

    def __init__(self, start=1000.0):
        self.now = float(start)

    def monotonic(self):
        return self.now

    def perf_counter(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)

    def __getattr__(self, name):  # strftime, gmtime, time...
        import time

        return getattr(time, name)


class TestDemoTakesWhileOverloaded(_FilesCase):
    """REC-E2E-05: while frames couldn't keep up, the demo device was read
    one display block a frame, fell behind and skipped ahead: the take had
    silent gaps that nothing reported."""

    RATE = 1e6

    def setUp(self):
        super().setUp()
        from unittest import mock

        from sdr_module.gui import device_dialog, main_window

        self.clock = _TestClock()
        for module in (main_window, device_dialog):
            patcher = mock.patch.object(module, "time", self.clock)
            patcher.start()
            self.addCleanup(patcher.stop)
        win = self.win
        win._stop_acquisition()
        self.dev = device_dialog.MockDevice()
        self.dev.set_sample_rate(self.RATE)
        win._device = self.dev
        win._start_acquisition()
        self.load = 1.5  # processing takes 1.5x real time: overloaded
        process = win._process_block

        def slow_process(samples):
            self.clock.advance(self.load * len(samples) / self.RATE)
            process(samples)

        win._process_block = slow_process

    def tearDown(self):
        if self.win._recording:
            self.win._toggle_recording(False)
        self.win._stop_acquisition()
        super().tearDown()

    def run_frames(self, seconds, stall_at=None, stall=0.0):
        end = self.clock.now + seconds
        while self.clock.now < end:
            tick = self.clock.now
            if stall_at is not None and tick >= stall_at:
                self.clock.advance(stall)  # the window froze for a moment
                stall_at = None
            self.win._update_display()
            self.clock.now = max(self.clock.now, tick + 0.033)

    def recorded(self):
        return sum(len(block) for block in self.win._samples_buffer)

    def test_the_take_keeps_every_sample(self):
        win = self.win
        self.run_frames(1.0)
        self.assertTrue(win._pacer.overloaded)
        win._toggle_recording(True)
        started = self.clock.now
        self.run_frames(2.0)
        win._toggle_recording(False)
        seconds = self.clock.now - started
        self.assertGreater(self.recorded(), 0.97 * seconds * self.RATE)
        self.assertNotIn("lost", win._take_summary()[0])
        # The window still processes about a block a frame (it stays
        # responsive), and its audio is paused.
        self.assertTrue(win._pacer.overloaded)

    def test_a_skip_ahead_during_a_take_is_reported(self):
        win = self.win
        win._toggle_recording(True)
        self.run_frames(0.5)
        # Frozen for 4 s: more than a frame can catch up (and the demo
        # follows the clock), so the take has a gap.
        self.run_frames(1.0, stall_at=self.clock.now + 0.2, stall=4.0)
        self.assertIn("The recording has gaps", self.message())
        win._toggle_recording(False)
        text, _state, tip = win._take_summary()
        self.assertIn("lost", text)
        lost = win._take_lost_seconds()
        self.assertGreater(lost, 1.0)
        total = self.recorded() / self.RATE + lost
        self.assertAlmostEqual(total, win._recording_elapsed(), delta=0.2)

    def test_stopping_and_starting_again_is_not_a_gap(self):
        win = self.win
        self.load = 0.3
        win._toggle_recording(True)
        self.run_frames(0.5)
        win._stop_acquisition()  # REC WAITING
        self.clock.advance(3.0)
        win._start_acquisition()
        self.run_frames(0.5)
        win._toggle_recording(False)
        self.assertNotIn("lost", win._take_summary()[0])


class TestGapsAreLoggedOnce(TestDemoTakesWhileOverloaded):
    """REC-E2E-06: the 'recording has gaps' warning was listed twice in
    Error History (logged, then logged again by the status bar)."""

    def test_the_gap_warning_is_logged_once(self):
        win = self.win
        win._toggle_recording(True)
        self.run_frames(0.5)
        with self.assertLogs("sdr_module.gui.main_window", level="WARNING") as logs:
            self.run_frames(1.0, stall_at=self.clock.now + 0.2, stall=4.0)
        gaps = [line for line in logs.output if "The recording has gaps" in line]
        self.assertEqual(len(gaps), 1, logs.output)

    # (The inherited tests run here too: skip them.)
    test_the_take_keeps_every_sample = None
    test_a_skip_ahead_during_a_take_is_reported = None
    test_stopping_and_starting_again_is_not_a_gap = None


class TestStatusFollowsTheState(_FilesCase):
    """REC-E2E-07: stale status-bar messages."""

    def test_back_to_1x_replaces_the_speed_note(self):
        path = _save_sigmf(self.path("fm.sigmf-data"), 100.1e6, seconds=1.0)
        with self.open_dialog(path):
            self.assertTrue(self.win._open_recording())
        bar = self.win._playback_bar
        bar.set_speed(0.5)
        self.assertIn("Playing at 0.5x", self.message())
        bar.set_speed(1.0)
        self.assertEqual(self.message(), "Playing at 1x.")
        self.win._playback.close()

    def test_capturing_replaces_the_armed_note(self):
        win = self.win
        win._stop_acquisition()
        win._toggle_recording(True)
        self.assertTrue(self.message().startswith("Recording armed:"))
        win._start_acquisition()
        win._update_display()
        self.assertFalse(win._capture_pending)
        self.assertEqual(
            self.message(), "Recording I/Q samples. Press Record again to stop."
        )
        win._toggle_recording(False)
        win._stop_acquisition()


class TestPlaybackSaysItsRealSpeed(_FilesCase):
    """REC-E2E-05 (playback): at 4x on a computer that can't keep up the
    recording played at about 0.3x while the status bar only said "Playing
    at 4x: audio plays at 1x only"."""

    def test_a_speed_that_cannot_be_kept_up_is_said(self):
        from unittest import mock

        from sdr_module.gui import playback

        path = _save_sigmf(self.path("aprs2m.sigmf-data"), 144.39e6, seconds=10.0)
        with self.open_dialog(path):
            self.assertTrue(self.win._open_recording())
        controller = self.win._playback
        controller._timer.stop()  # (ticked by hand)
        device = controller.device
        clock = _TestClock()
        with mock.patch.object(playback, "time", clock):
            self.win._playback_bar.set_speed(4.0)
            for _ in range(20):  # 4 s: the file moves 0.3 s a second
                device.seek(device.position + int(0.06 * device.sample_rate))
                clock.advance(0.2)
                controller._tick()
        self.assertIn("Playing at about 0.3x, not 4x", self.message())
        controller.close()

    def test_a_speed_kept_up_says_nothing_more(self):
        from unittest import mock

        from sdr_module.gui import playback

        path = _save_sigmf(self.path("aprs2m.sigmf-data"), 144.39e6, seconds=10.0)
        with self.open_dialog(path):
            self.assertTrue(self.win._open_recording())
        controller = self.win._playback
        controller._timer.stop()
        device = controller.device
        clock = _TestClock()
        with mock.patch.object(playback, "time", clock):
            self.win._playback_bar.set_speed(2.0)
            for _ in range(20):
                device.seek(device.position + int(0.4 * device.sample_rate))
                clock.advance(0.2)
                controller._tick()
        self.assertIn("Playing at 2x", self.message())
        controller.close()


class TestOneSizeUnit(_FilesCase):
    """REC-E2E-09: 'MB' meant MiB in some places (the REC info, the second
    receiver, free space) and decimal MB in others (the take line, Recording
    Info), so the same file showed two sizes."""

    def test_the_same_bytes_read_the_same_everywhere(self):
        from sdr_module.gui.iq_files import RecordingInfoDialog, describe_iq_file

        size = 72_474_856  # (69.1 MiB)
        path = self.path("take.cf32")
        with open(path, "wb") as handle:
            handle.truncate(size)
        options, _missing = describe_iq_file(path)
        options.sample_rate, options.center_hz = 2.4e6, 145.8e6
        info = RecordingInfoDialog(options)
        self.addCleanup(info.deleteLater)
        self.assertEqual(info.value("Size"), "72.5 MB")
        # The main window's recording info, while recording.
        win = self.win
        win._recording_bytes = size
        win._update_recording_status()
        self.assertTrue(win._recording_info_label.text().startswith("72.5 MB"))
        # The second receiver's REC label (shut down with the window).
        second = win._second_rx.show()
        second._status_timer.stop()

        class Recorder:
            file_size_bytes = size
            duration_seconds = 3.0

        second._recorder = Recorder()
        second._update_rec_label()
        self.assertEqual(second._rec_label.text(), "REC 00:03 · 72.5 MB")
        second._recorder = None

    def test_free_space_is_in_the_same_unit(self):
        from collections import namedtuple
        from unittest import mock

        usage = namedtuple("Usage", "total used free")(10**12, 0, 420_000_000)
        with mock.patch.object(self.mw.shutil, "disk_usage", return_value=usage):
            self.assertFalse(self.win._start_recording())
        self.assertIn("only 420 MB of disk space", self.message())

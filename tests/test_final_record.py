"""Recordings: the problems the end-to-end journeys found (final fix wave).

* REC-E2E-01, DEC-E2E-20: quiet little-endian cs16 (and 8-bit samples
  widened to 16 bits) was guessed to be cs8; Import Options offered "Use
  cs8" and called that reading right.
* REC-E2E-02/03: a recording plays the way its band is heard; Close goes
  back to the listening from before.
* REC-E2E-04: Import Options fits its rows.
* REC-E2E-05/06: demo takes keep every sample while frames can't keep up;
  skips are reported, once; playback says when it can't keep a speed.
* REC-E2E-07: status messages follow the state.
* REC-E2E-08: the sample check said "peak 0.00 of full scale" for quiet
  recordings.
* REC-E2E-09 to 16: one size unit, take metadata, the second receiver's
  stop, the Recording group, format names, refusals, the playback strip at
  1024 px, quiet 8-bit saves.
* Leftovers 20, 21, 24, 30 and the flaky input/playback tests.
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
from tests.test_review_recordings import _ReviewCase  # noqa: E402

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


class _DemoTakeCase(_FilesCase):
    """The demo device at 1 MS/s on a fake clock, with processing that
    takes ``load`` times real time."""

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


class TestDemoTakesWhileOverloaded(_DemoTakeCase):
    """REC-E2E-05: while frames couldn't keep up, the demo device was read
    one display block a frame, fell behind and skipped ahead: the take had
    silent gaps that nothing reported."""

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

    def test_a_skip_after_a_disk_full_stop_isnt_the_takes(self):
        from unittest import mock

        win = self.win
        win._toggle_recording(True)
        self.run_frames(0.5)
        check = win._check_recording_space

        def disk_full():
            # (The disk fills up in the frame after a long freeze.)
            if self.clock.now >= stall_end:
                win._stop_recording_with_warning("only 1 MB of disk space is left")
                return False
            return check()

        stall_end = self.clock.now + 0.2 + 4.0
        with mock.patch.object(win, "_check_recording_space", disk_full):
            self.run_frames(1.0, stall_at=self.clock.now + 0.2, stall=4.0)
        self.assertFalse(win._recording)
        self.assertNotIn("lost", win._take_summary()[0])

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


class TestGapsAreLoggedOnce(_DemoTakeCase):
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


class TestSecondReceiverStopSaysWhereTheRecordingWent(_FilesCase):
    """REC-E2E-11: Stop during a second-receiver recording kept the file but
    only said "Stopped; Demo Device (simulated) is closed."; the demo device
    had two names."""

    def test_stop_says_the_recording_was_saved(self):
        win = self.win
        self.store.values["recording/folder"] = self.folder
        rx = win._second_rx.show()
        rx._status_timer.stop()
        self.assertTrue(rx.start())
        rx.tune(145.8e6, recenter=True)
        self.assertTrue(rx._set_recording(True))
        for _ in range(4):
            rx._frame()
        rx.stop()
        text = rx.message()
        self.assertTrue(
            text.startswith(
                "Stopped; Demo Device (simulated signals) is closed. Saved "
            ),
            text,
        )
        self.assertIn("_rx2.sigmf-data", text)
        self.assertIn("File > Open Recording", text)
        self.assertEqual(rx._message_label.property("tone"), "success")

    def test_one_name_for_the_demo_device(self):
        from sdr_module.gui.second_receiver import DEMO_NAME

        self.assertEqual(DEMO_NAME, self.win._device_display_name())


class TestTakeMetadata(_FilesCase):
    """REC-E2E-10: SigMF from main-window takes had an empty author, the
    generic description "SDR Recording", no core:hw and a whole-second first
    timestamp, unlike the second receiver's."""

    def test_a_saved_take_says_what_and_where_it_was(self):
        import json
        import re

        win = self.win
        self.assertTrue(win.set_frequency(145.29e6))
        win._set_demod_mode("FM", "5 kHz", "25 kHz")
        win._toggle_recording(True)
        for _ in range(3):
            win._update_display()
        win._toggle_recording(False)
        path = self.path("rt.sigmf-data")
        with self.save_dialog(path):
            win._save_recording()
        self.settle(5)
        with open(self.path("rt.sigmf-meta")) as handle:
            meta = json.load(handle)
        glob = meta["global"]
        self.assertEqual(glob["core:hw"], "Demo Device (simulated signals)")
        self.assertEqual(glob["core:description"], "Tuned to 145.290 MHz (FM)")
        self.assertNotIn("core:author", glob)  # (unknown: left out)
        stamp = meta["captures"][0]["core:datetime"]
        self.assertRegex(stamp, r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")
        self.assertTrue(re.match(r"^\d{4}", stamp))

    def test_a_take_of_a_recording_playing_keeps_its_hardware(self):
        win = self.win
        path = _save_sigmf(self.path("orig.sigmf-data"), 144.39e6, seconds=1.0)
        from sdr_module.dsp.recording import write_sigmf_meta

        write_sigmf_meta(
            path, SampleFormat.FLOAT32, 1e6, 144.39e6, hardware="RTL-SDR v3"
        )
        with self.open_dialog(path):
            self.assertTrue(win._open_recording())
        win._toggle_recording(True)
        for _ in range(3):
            win._update_display()
        win._toggle_recording(False)
        self.assertEqual(win._take_hardware, "RTL-SDR v3")
        self.assertEqual(
            win._take_description,
            "Tuned to 144.390 MHz (FM), played back from orig.sigmf-data",
        )
        win._playback.close()


class TestRecordingGroupShowsTheOtherKind(_FilesCase):
    """REC-E2E-12: with Record set to I/Q, the Recording group said "Ready"
    while an audio recording ran (and the other way round)."""

    def test_an_audio_recording_shows_while_record_is_iq(self):
        win = self.win
        panel = win._control_panel
        self.store.values["recording/folder"] = self.folder
        self.assertEqual(panel.record_kind(), "iq")
        win._record_audio_action.trigger()
        self.assertIsNotNone(win._audio_take)
        win._update_audio_recording_status()
        text = panel._record_status.text()
        self.assertTrue(text.startswith("● Recording audio 00:0"), text)
        self.assertIn("Ctrl+Shift+A", text)
        self.assertFalse(panel._record_btn.isChecked())  # (I/Q isn't)
        win._record_audio_action.trigger()  # stop
        self.assertEqual(panel._record_status.text(), "Ready")

    def test_an_iq_recording_shows_while_record_is_audio(self):
        win = self.win
        panel = win._control_panel
        panel.set_record_kind("audio")
        win._record_action.trigger()
        win._update_display()
        win._update_recording_status()
        text = panel._record_status.text()
        self.assertTrue(text.startswith("● Recording I/Q 00:0"), text)
        self.assertIn("Ctrl+Shift+R", text)
        win._record_action.trigger()  # stop
        self.assertEqual(panel._record_status.text(), "Ready")
        panel.set_record_kind("iq")

    def test_a_paused_one_is_shown_in_the_paused_tone(self):
        """As the group's own "❚❚ Paused" line (red read as recording)."""
        win = self.win
        panel = win._control_panel
        win._record_action.trigger()
        win._update_display()
        panel._pause_btn.click()
        self.assertTrue(win._recording_paused)
        panel.set_record_kind("audio")
        self.assertEqual(
            panel._record_status.text(),
            "❚❚ I/Q recording paused (Ctrl+Shift+R stops it)",
        )
        self.assertEqual(panel._record_status.property("tone"), "warning")
        panel.set_record_kind("iq")
        panel._pause_btn.click()  # resume
        panel.set_record_kind("audio")
        self.assertEqual(panel._record_status.property("tone"), "danger")
        win._record_action.trigger()  # stop
        self.assertEqual(panel._record_status.text(), "Ready")
        panel.set_record_kind("iq")


class TestOneNameForEachFormat(_FilesCase):
    """REC-E2E-13: the Recording group's Format list and File > Save
    Recording's filters named the same formats differently ("SigMF, keeps
    frequency & rate" / "... and rate", "Raw Float32 I/Q" / "Raw I/Q,
    Float32"), and the group didn't say Save... also writes cu8 and cs8."""

    def test_the_group_and_the_save_dialog_use_the_same_names(self):
        combo = self.win._control_panel._format_combo
        filters = {spec[1]: spec[0] for spec in self.win._SAVE_FORMATS}
        for i in range(combo.count()):
            ext = combo.itemData(i)
            name = combo.itemText(i).removesuffix(f" ({ext})")
            self.assertEqual(filters[ext], f"{name} (*{ext})")
        tip = combo.toolTip()
        self.assertIn("cu8", tip)
        self.assertIn("cs8", tip)


class TestRefusalsAreNotErrors(_FilesCase):
    """REC-E2E-14: saving over the recording you converted was logged as an
    ERROR in Error History, in a box titled "Save Failed"; a mono WAV's
    "it is audio, not I/Q" as a WARNING."""

    def test_saving_over_the_converted_recording_is_refused_quietly(self):
        from unittest import mock

        from PyQt6.QtWidgets import QMessageBox

        path = _save_sigmf(self.path("orig.sigmf-data"), 144.39e6, seconds=0.05)
        self.answer("Close")
        with self.open_dialog(path):
            self.assertTrue(self.win._convert_recording())
        with (
            self.save_dialog(path),
            mock.patch.object(QMessageBox, "warning") as warn,
            self.assertNoLogs("sdr_module.gui.main_window", level="WARNING"),
        ):
            self.assertFalse(self.win._save_recording())
        self.assertEqual(warn.call_args[0][1], "Can't Save There")
        self.assertIn("the recording you converted", warn.call_args[0][2])

    def test_an_audio_wav_is_refused_quietly(self):
        from unittest import mock

        from PyQt6.QtWidgets import QMessageBox

        from sdr_module.dsp.recording import save_audio_file

        path = self.path("mono_audio.wav")
        save_audio_file(path, np.zeros(4800, np.float32), 48000)
        with (
            self.open_dialog(path),
            mock.patch.object(QMessageBox, "warning") as warn,
            self.assertNoLogs("sdr_module.gui.main_window", level="WARNING"),
        ):
            self.win._show_recording_info()
        self.assertIn("it is audio, not I/Q", warn.call_args[0][2])


class TestPlaybackStripAt1024(_FilesCase):
    """REC-E2E-15: at 1024x640 the playback strip squeezed the file's name
    to 'ap…ta' in 40 px and the position slider to 100 px."""

    def test_the_slider_gets_the_room_of_a_name_too_short_to_read(self):
        from PyQt6.QtWidgets import QApplication

        from sdr_module.gui import themes

        path = _save_sigmf(self.path("aprs2m.sigmf-data"), 144.39e6, seconds=1.0)
        with self.open_dialog(path):
            self.assertTrue(self.win._open_recording())
        bar = self.win._playback_bar
        qapp = QApplication.instance()
        old_sheet, old_palette = qapp.styleSheet(), qapp.palette()
        old_theme = themes.current_theme()
        try:
            themes.apply_theme(qapp, "dark")
            self.win.resize(1024, 640)
            self.settle(3)
            # (The status bar names the file.)
            self.assertTrue(bar._file.isHidden())
            self.assertTrue(bar._caption.isHidden())
            self.assertGreater(bar._slider.width(), 170)
            self.win.resize(1366, 768)
            self.settle(3)
            self.assertFalse(bar._file.isHidden())
            self.assertFalse(bar._caption.isHidden())
            self.assertGreaterEqual(bar._file.width(), 120)
            self.assertGreaterEqual(bar._slider.width(), 200)
        finally:
            qapp.setStyleSheet(old_sheet)
            qapp.setPalette(old_palette)
            themes.apply_theme(None, old_theme)
            self.win._playback.close()


class TestQuietEightBitSaves(_FilesCase):
    """REC-E2E-16: a quiet take saved as cu8 or cs8 kept about one bit of
    its signal, with no warning; played back it said "FM audio is
    clipping"."""

    def take(self, freq_hz):
        win = self.win
        self.assertTrue(win.set_frequency(freq_hz))
        win._toggle_recording(True)
        for _ in range(3):
            win._update_display()
        win._toggle_recording(False)

    def test_saving_a_quiet_take_as_8_bit_asks_first(self):
        self.take(145.29e6)  # the demo's 2 m: noise about -36 dBFS
        for ext in (".cu8", ".cs8"):
            path = self.path("rt" + ext)
            self.boxes.clear()
            self.answer("Cancel")
            with self.save_dialog(path):
                self.assertFalse(self.win._save_recording())
            self.assertFalse(os.path.exists(path))
            box = self.boxes[-1]
            self.assertEqual(box.windowTitle(), "Quiet Recording")
            self.assertIn("dBFS", box.informativeText())
            self.assertIn("cs16", box.informativeText())
        self.answer("Save Anyway")
        with self.save_dialog(self.path("rt.cu8")):
            self.assertTrue(self.win._save_recording())
        self.settle(5)
        self.assertTrue(os.path.exists(self.path("rt.cu8")))

    def test_16_bit_and_loud_takes_save_without_asking(self):
        self.take(145.29e6)
        with self.save_dialog(self.path("rt.cs16")):
            self.assertTrue(self.win._save_recording())
        self.take(100.1e6)  # the demo's broadcast station: loud enough
        self.boxes.clear()
        with self.save_dialog(self.path("fm.cu8")):
            self.assertTrue(self.win._save_recording())
        titles = [b.windowTitle() for b in self.boxes if hasattr(b, "windowTitle")]
        self.assertNotIn("Quiet Recording", titles)


class TestNonFiniteMetadataIsMissing:
    """Leftover 20: a recording whose metadata center (or rate) was NaN or
    Infinity played with a NaN LO ("(nan-nan MHz)") instead of asking."""

    @pytest.mark.parametrize(
        "center, rate",
        [
            (float("nan"), 1e6),
            (float("inf"), 1e6),
            (-144.39e6, 1e6),
            (144.39e6, float("nan")),
            (144.39e6, float("-inf")),
            (144.39e6, "fast"),
        ],
    )
    def test_treated_as_unknown(self, tmp_path, center, rate):
        import json

        from sdr_module.dsp.recording import detect_format
        from sdr_module.gui.iq_files import describe_iq_file

        base = tmp_path / "odd"
        np.zeros(1000, np.complex64).tofile(str(base) + ".sigmf-data")
        meta = {
            "global": {"core:datatype": "cf32_le", "core:sample_rate": rate},
            "captures": [{"core:sample_start": 0, "core:frequency": center}],
            "annotations": [],
        }
        with open(str(base) + ".sigmf-meta", "w") as handle:
            json.dump(meta, handle)  # (writes NaN / Infinity as JSON does)
        info = detect_format(str(base) + ".sigmf-data")
        assert info.is_valid, info.error_message
        for value in (info.sample_rate, info.center_frequency):
            assert np.isfinite(value) and value >= 0
        options, missing = describe_iq_file(str(base) + ".sigmf-data")
        good_center = isinstance(center, float) and np.isfinite(center) and center > 0
        good_rate = isinstance(rate, float) and np.isfinite(rate) and rate > 0
        assert ("center" in missing) is not good_center
        assert ("rate" in missing) is not good_rate
        assert np.isfinite(options.center_hz) and np.isfinite(options.sample_rate)


class TestNonFiniteCenterAsksInTheWindow(_FilesCase):
    def test_import_options_asks_for_the_center(self):
        import json

        base = self.path("odd_center")
        np.zeros(48000, np.complex64).tofile(base + ".sigmf-data")
        meta = {
            "global": {"core:datatype": "cf32_le", "core:sample_rate": 240e3},
            "captures": [{"core:sample_start": 0, "core:frequency": float("nan")}],
        }
        with open(base + ".sigmf-meta", "w") as handle:
            json.dump(meta, handle)
        with self.open_dialog(base + ".sigmf-data"):
            self.assertTrue(self.win._open_recording())
        kinds = [type(b).__name__ for b in self.boxes]
        self.assertEqual(kinds, ["ImportOptionsDialog"])
        self.assertTrue(np.isfinite(self.win._lo_frequency()))
        self.assertNotIn("nan", self.win._playback_bar.file_text().lower())
        self.win._playback.close()


class TestCentersAboveTheReceiversRange(_FilesCase):
    """Leftover 21: Import Options refused a typed center above 6 GHz (the
    receiver's range), though recordings there play (a 10 GHz downlink)."""

    def test_a_headerless_10_ghz_capture_can_be_given_its_center(self):
        from sdr_module.gui.iq_files import ImportOptionsDialog, describe_iq_file

        path = self.path("qo100.cs16")
        with open(path, "wb") as handle:
            handle.write(_cs16_noise(30, n=4 * N))
        options, missing = describe_iq_file(path)
        dialog = ImportOptionsDialog(options, missing, "play")
        self.addCleanup(dialog.deleteLater)
        dialog._rate_edit.setText("2.4M")
        dialog._center_edit.setText("10489.75")
        self.assertEqual(dialog.center_hz(), 10489.75e6)
        dialog._on_accept()
        self.assertEqual(dialog.options().center_hz, 10489.75e6)
        # It plays there (the window's range follows the recording).
        self.assertTrue(self.win._playback.open(dialog.options()))
        self.assertEqual(self.win._lo_frequency(), 10489.75e6)
        self.win._playback.close()

    def test_far_too_high_is_still_refused(self):
        from sdr_module.gui.iq_files import ImportOptionsDialog, describe_iq_file

        path = self.path("odd.cs16")
        with open(path, "wb") as handle:
            handle.write(_cs16_noise(30, n=N))
        options, missing = describe_iq_file(path)
        dialog = ImportOptionsDialog(options, missing, "play")
        self.addCleanup(dialog.deleteLater)
        dialog._center_edit.setText("5000G")
        self.assertIsNone(dialog.center_hz())
        self.assertIn("Can't read the center frequency", dialog.check_text())


class TestAutoSaveRefusalFitsTheStatusBar(_ReviewCase):
    """Leftover 30: the 'Save automatically' refusal was ~280 characters;
    at 1366 px the status bar cut off what to do (the tooltip had it)."""

    def test_short_with_the_advice(self):
        self.win._control_panel.set_auto_save(True)
        self.set_format(".cs16")  # (a full copy: refused on the full drive)
        self.record(3, stop=False)
        with self.assertLogs("sdr_module.gui.main_window", level="ERROR"):
            self.stop_for_disk_space(420)
        text = self.message()
        self.assertIn("would be nearly full", text)
        self.assertIn("It is kept: save it elsewhere", text)
        self.assertLessEqual(len(text), 160, text)


class TestWavLimitBeforeRoom(_ReviewCase):
    """Leftover 30: a WAV save over 4 GiB onto a drive with less room was
    first told there isn't room, then (room made) that WAV can't hold it."""

    def test_the_wav_limit_is_said_first(self):
        from unittest import mock

        from PyQt6.QtWidgets import QMessageBox

        from sdr_module.gui import file_jobs

        self.record(3)
        count = self.win._buffer_sample_count()
        small = count * 4 - 1  # (this take is "over 4 GiB")
        target = self.path("take.wav")
        with (
            mock.patch.object(file_jobs, "WAV_MAX_DATA_BYTES", small),
            self.free(0.01),  # (and the drive has no room for it)
            self.save_dialog(target),
            mock.patch.object(QMessageBox, "warning") as warn,
        ):
            self.assertFalse(self.win._save_recording())
        self.assertIn("too long for a WAV file", warn.call_args[0][2])
        self.assertNotIn("enough space", warn.call_args[0][2])
        self.assertFalse(os.path.exists(target))


class TestTextModeIqWriteCleansUp:
    """Leftover 30: text_modes.write_iq_file stopped the recorder in a
    finally: when stop() failed (a full disk), the partial files stayed."""

    def _source(self):
        from sdr_module.protocols import text_modes as tm

        sig = tm.TextSignal("CQ CQ", tm.TextSettings.for_mode(tm.MODE_ASCII))
        return tm, sig.source(48000.0)

    @pytest.mark.parametrize(
        "fmt, names", [("sigmf", ["a.sigmf-data"]), ("cf32", ["a.cf32"])]
    )
    def test_a_failing_stop_leaves_nothing(self, tmp_path, fmt, names):
        from unittest import mock

        from sdr_module.dsp.recording import IQRecorder

        tm, source = self._source()
        path = tmp_path / names[0]
        with (
            mock.patch.object(IQRecorder, "stop", side_effect=OSError(28, "full")),
            pytest.raises(OSError),
        ):
            tm.write_iq_file(str(path), source, fmt)
        assert list(tmp_path.iterdir()) == []

    def test_a_failing_write_reports_its_own_error(self, tmp_path):
        from unittest import mock

        from sdr_module.dsp.recording import IQRecorder

        tm, source = self._source()
        with (
            mock.patch.object(IQRecorder, "write", side_effect=ValueError("bad")),
            mock.patch.object(IQRecorder, "stop", side_effect=OSError(28, "full")),
            pytest.raises(ValueError, match="bad"),
        ):
            tm.write_iq_file(str(tmp_path / "b.sigmf-data"), source, "sigmf")
        assert list(tmp_path.iterdir()) == []


class TestNoRetuneWhileRecordingIq(_FilesCase):
    """Leftover 24: tuning far during an I/Q take retuned the LO: the take
    silently lost the transfers drained after the retune, and its metadata
    kept the first center for the samples recorded after it."""

    def start_take(self):
        win = self.win
        self.assertTrue(win.set_frequency(100.1e6))
        win._toggle_recording(True)
        win._update_display()
        self.assertFalse(win._capture_pending)

    def test_a_far_tune_is_refused_while_recording(self):
        win = self.win
        self.start_take()
        lo = win._lo_frequency()
        self.assertFalse(win.set_frequency(145.8e6))
        self.assertEqual(win._lo_frequency(), lo)
        self.assertEqual(win._current_frequency(), 100.1e6)
        self.assertIn("stop recording to retune", self.message())
        # Inside the span only the tuned frequency moves.
        self.assertTrue(win.set_frequency(100.4e6))
        self.assertEqual(win._lo_frequency(), lo)
        win._update_display()
        self.assertEqual(win._buffer_meta[0], lo)
        win._toggle_recording(False)
        self.assertTrue(win.set_frequency(145.8e6))  # stopped: as before

    def test_keep_centered_doesnt_move_the_lo_while_recording(self):
        win = self.win
        win._keep_centered_action.setChecked(True)
        self.start_take()
        lo = win._lo_frequency()
        self.assertTrue(win.set_frequency(100.3e6))
        self.assertEqual(win._lo_frequency(), lo)
        win._toggle_recording(False)
        win._keep_centered_action.setChecked(False)

    def test_armed_recordings_still_tune(self):
        win = self.win
        win._stop_acquisition()
        win._toggle_recording(True)  # armed: nothing captured yet
        self.assertTrue(win._capture_pending)
        self.assertTrue(win.set_frequency(145.8e6))
        self.assertEqual(win._current_frequency(), 145.8e6)
        win._toggle_recording(False)


class TestInputWaitsWithTheProgressDialogUp(_FilesCase):
    """Flaky test_gui_recording_files::TestInputWhileAJobRuns: on a busy
    computer the progress dialog came up before the test's click, and from
    then on only its window modality held input back, which QTest's
    events (offscreen) bypass: the click started a recording. Input other
    than the dialog's now waits whether it shows or not."""

    def test_a_click_waits_while_the_dialog_shows(self):
        import time

        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtTest import QTest

        from sdr_module.gui.file_jobs import run_job

        win = self.win
        seen = {}

        def press():
            seen["dialog"] = any(
                type(w).__name__ == "QProgressDialog" and w.isVisible()
                for w in win.findChildren(object)
                if hasattr(w, "isVisible")
            )
            QTest.mouseClick(win._record_button, Qt.MouseButton.LeftButton)
            win._spectrum.setFocus()
            QTest.keyClick(win._spectrum, Qt.Key.Key_Space)

        def work(report):
            for i in range(40):
                time.sleep(0.01)
                report(i / 40)
            return "done"

        QTimer.singleShot(150, press)
        result = run_job(win, "Test", "Working…", work, show_after_ms=0)
        self.assertEqual(result.value, "done")
        self.assertTrue(seen["dialog"])  # (the case tested: it was up)
        self.assertFalse(win._recording)
        self.assertTrue(win._is_running)

    def test_the_dialogs_cancel_still_works(self):
        import time

        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtTest import QTest
        from PyQt6.QtWidgets import QProgressDialog, QPushButton

        from sdr_module.gui.file_jobs import run_job

        win = self.win

        def cancel():
            dialog = next(w for w in win.findChildren(QProgressDialog) if w.isVisible())
            button = dialog.findChild(QPushButton)
            QTest.mouseClick(button, Qt.MouseButton.LeftButton)

        def work(report):
            for i in range(300):
                time.sleep(0.01)
                report(i / 300)
            return "done"

        QTimer.singleShot(150, cancel)
        result = run_job(win, "Test", "Working…", work, show_after_ms=0)
        self.assertTrue(result.cancelled)


# ---------------------------------------------------------------------------
# From the check of these fixes (each failed before its fix).


def _quiet_cs8_from_the_demo(n=N):
    """The demo's 1090 MHz band as hackrf_transfer writes it at a low gain:
    cs8 that is almost all zeros (the noise is a fraction of a step)."""
    from sdr_module.gui.device_dialog import MockDevice

    device = MockDevice()
    device.set_frequency(1090e6)
    device.set_sample_rate(2.4e6)
    device.start_rx()
    parts = device.read_samples(n).view(np.float32)
    return np.clip(np.round(parts * 127.0), -128, 127).astype(np.int8).tobytes()


class TestNearlySilentCs8IsNotTakenFor16Bit:
    """A cs8 capture that is almost all zeros has a few nonzero I values more
    than Q ones: the 'low byte spreads wider than the high byte' rule read
    that as 16-bit, sure of it (Import Options offered "Use cs16")."""

    def test_the_demo_s_adsb_band_as_quiet_cs8(self):
        data = _quiet_cs8_from_the_demo()
        assert guess_sample_format(data) is SampleFormat.INT8
        assert confident_sample_format(data) is not SampleFormat.INT16

    @pytest.mark.parametrize("seed", range(6))
    def test_a_few_nonzero_values(self, seed):
        rng = np.random.default_rng(seed)
        x = np.zeros(2 * N, np.int8)
        x[0::2][rng.choice(N, 40, replace=False)] = rng.choice([-1, 1, 2], 40)
        x[1::2][rng.choice(N, 3, replace=False)] = 1
        assert guess_sample_format(x.tobytes()) is SampleFormat.INT8


class TestImportOptionsKeepsQuietCs8(_FilesCase):
    def test_no_use_cs16_for_a_quiet_cs8_capture(self):
        from sdr_module.gui.iq_files import ImportOptionsDialog, describe_iq_file

        for name, prefill in (
            ("adsb_1090MHz_2.4Msps.cs8", SampleFormat.INT8),
            ("adsb_1090MHz_2.4Msps.iq", SampleFormat.INT8),
        ):
            path = self.path(name)
            with open(path, "wb") as handle:
                handle.write(_quiet_cs8_from_the_demo(8 * N))
            options, missing = describe_iq_file(path)
            dialog = ImportOptionsDialog(options, missing, "play")
            self.addCleanup(dialog.deleteLater)
            self.assertIs(dialog.sample_format(), prefill, name)
            self.assertFalse(dialog._format_hint_row.isVisibleTo(dialog), name)


class TestATakeThatStartsBehind(_DemoTakeCase):
    """A take started while the demo lagged (frames couldn't keep up just
    before Record): reading a frame's older samples apart from its newest
    made the demo count the newest as lag too, and a short stall then
    skipped half a second ("The recording has gaps: 550 ms ...")."""

    def lag_s(self):
        dev = self.dev
        return (self.clock.now - dev._stream_epoch) - dev._stream_pos / self.RATE

    def test_a_short_stall_loses_nothing(self):
        win = self.win
        for _ in range(400):  # overloaded: the demo falls behind
            self.run_frames(0.034)
            if 0.30 < self.lag_s() < 0.45:
                break
        self.assertGreater(self.lag_s(), 0.30)
        win._toggle_recording(True)
        self.load = 0.5  # frames keep up again
        self.run_frames(1.5)
        self.assertFalse(win._pacer.overloaded)
        # A 0.3 s freeze: less than the demo's half second of slack.
        self.run_frames(1.0, stall_at=self.clock.now + 0.1, stall=0.3)
        win._toggle_recording(False)
        self.assertNotIn("lost", win._take_summary()[0])
        self.assertNotIn("The recording has gaps", self.message())

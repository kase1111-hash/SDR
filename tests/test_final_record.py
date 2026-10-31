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

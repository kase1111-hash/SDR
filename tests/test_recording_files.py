"""Reading recordings back correctly (``sdr_module.dsp.recording``).

Regression tests for the T5 audit findings on the recording backend:

* rtl_sdr ``.bin`` captures (unsigned 8-bit) read as float32 garbage
  (fg:FG-B03, mc:MC-06, rx:RX-06);
* raw and WAV saves lose their sample rate and centre frequency, now kept
  in a SigMF-style sidecar (rx:RX-07, ux:UX-B12);
* ``detect_format()`` / ``get_file_info()`` picking up another file's
  ``.sigmf-meta`` (rx:RX-16), and cs8 data detected as cu8;
* whole-file reads converting through float64/complex128 (rx:RX-08,
  mc:MC-02): reading now converts in float32, a chunk at a time.
"""

import json
import os
import tracemalloc

import numpy as np
import pytest

from sdr_module.dsp.recording import (
    FileFormat,
    FormatDetector,
    IQPlayer,
    SampleFormat,
    check_iq_samples,
    detect_format,
    find_sidecar,
    get_file_info,
    load_iq_file,
    save_iq_file,
    sidecar_path,
    write_sidecar,
)

RATE = 240e3


def _tone(n=24000, freq=30e3, amplitude=0.5, rate=RATE):
    t = np.arange(n) / rate
    return (amplitude * np.exp(2j * np.pi * freq * t)).astype(np.complex64)


def _write_cu8(path, x):
    raw = np.empty(2 * len(x), np.uint8)
    raw[0::2] = np.clip(np.rint(x.real * 127.5 + 127.5), 0, 255)
    raw[1::2] = np.clip(np.rint(x.imag * 127.5 + 127.5), 0, 255)
    raw.tofile(path)


def _write_cs8(path, x):
    raw = np.empty(2 * len(x), np.int8)
    raw[0::2] = np.rint(x.real * 127)
    raw[1::2] = np.rint(x.imag * 127)
    raw.tofile(path)


def _write_cs16(path, x):
    raw = np.empty(2 * len(x), np.int16)
    raw[0::2] = np.rint(x.real * 32767)
    raw[1::2] = np.rint(x.imag * 32767)
    raw.tofile(path)


class TestRtlSdrBin:
    def test_bin_is_read_as_unsigned_8_bit(self, tmp_path):
        x = _tone()
        path = tmp_path / "rtl.bin"
        _write_cu8(path, x)
        samples, meta = load_iq_file(path)
        assert meta.sample_format is SampleFormat.UINT8
        assert len(samples) == len(x)
        assert np.all(np.isfinite(samples))
        np.testing.assert_allclose(samples, x, atol=1.0 / 127)

    def test_detector_agrees(self, tmp_path):
        path = tmp_path / "rtl.bin"
        _write_cu8(path, _tone())
        info = detect_format(path)
        assert info.sample_format is SampleFormat.UINT8
        assert info.num_samples == 24000


class TestFloat32Reading:
    @pytest.mark.parametrize(
        "fmt,writer,atol",
        [
            (SampleFormat.UINT8, _write_cu8, 1.0 / 127),
            (SampleFormat.INT8, _write_cs8, 1.0 / 127),
            (SampleFormat.INT16, _write_cs16, 1e-4),
        ],
    )
    def test_integer_formats_match_the_original(self, tmp_path, fmt, writer, atol):
        x = _tone()
        path = tmp_path / f"cap.{fmt.value}"
        writer(path, x)
        with IQPlayer(path) as player:
            assert player.sample_format is fmt
            first = player.read(1000)
            rest = player.read(10**9)
        assert first.dtype == np.complex64
        whole = np.concatenate([first, rest])
        np.testing.assert_allclose(whole, x, atol=atol)

    def test_float_formats_round_trip(self, tmp_path):
        x = _tone()
        x.tofile(tmp_path / "cap.cf32")
        x.astype(np.complex128).tofile(tmp_path / "cap.cf64")
        for name in ("cap.cf32", "cap.cf64"):
            samples, _meta = load_iq_file(tmp_path / name)
            np.testing.assert_array_equal(samples, x)

    def test_odd_trailing_bytes_are_ignored(self, tmp_path):
        x = _tone(1000)
        path = tmp_path / "cut.cu8"
        _write_cu8(path, x)
        with open(path, "ab") as handle:
            handle.write(b"\x80")  # half a sample
        samples, _meta = load_iq_file(path)
        assert len(samples) == 1000

    def test_reading_converts_in_float32(self, tmp_path):
        # A 1 M-sample cu8 file must not need complex128 temporaries (16 B
        # per sample each); the result itself is 8 MB.
        x = np.zeros(1_000_000, np.complex64)
        path = tmp_path / "big.cu8"
        _write_cu8(path, x)
        with IQPlayer(path) as player:
            tracemalloc.start()
            samples = player.read(10**7)
            _current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        assert len(samples) == 1_000_000
        assert peak < 20e6, f"peak {peak / 1e6:.0f} MB"

    def test_seek_to_the_end_and_back(self, tmp_path):
        x = _tone(1000)
        path = tmp_path / "s.cf32"
        x.tofile(path)
        with IQPlayer(path) as player:
            player.seek(1000)
            assert len(player.read(10)) == 0
            assert player.eof
            player.seek(990)
            np.testing.assert_array_equal(player.read(100), x[990:])
            player.seek(-5)
            assert player.position == 0


class TestSidecars:
    def test_raw_save_with_sidecar_round_trips_rate_and_centre(self, tmp_path):
        x = _tone()
        path = tmp_path / "take.cs16"
        save_iq_file(path, x, RATE, 433.92e6, SampleFormat.INT16, FileFormat.RAW)
        meta_path = write_sidecar(
            path, SampleFormat.INT16, RATE, 433.92e6, start_time="2026-10-30T12:00:00Z"
        )
        assert meta_path == sidecar_path(path)
        assert meta_path.name == "take.cs16.sigmf-meta"
        with open(meta_path) as handle:
            meta = json.load(handle)
        assert meta["global"]["core:dataset"] == "take.cs16"
        assert meta["global"]["core:datatype"] == "ci16_le"
        samples, info = load_iq_file(path)
        assert info.sample_rate == RATE
        assert info.center_frequency == 433.92e6
        np.testing.assert_allclose(samples, x, atol=1e-4)
        detected = detect_format(path)
        assert detected.file_format is FileFormat.RAW
        assert (detected.sample_rate, detected.center_frequency) == (RATE, 433.92e6)
        assert detected.num_samples == len(x)

    def test_sidecar_names_the_sample_format(self, tmp_path):
        # A .bin written as cs8 (HackRF) with its sidecar is read as cs8.
        x = _tone()
        path = tmp_path / "hackrf.bin"
        _write_cs8(path, x)
        write_sidecar(path, SampleFormat.INT8, 2e6, 100e6)
        samples, info = load_iq_file(path)
        assert info.sample_format is SampleFormat.INT8
        np.testing.assert_allclose(samples, x, atol=1.0 / 127)

    def test_unknown_values_are_left_out(self, tmp_path):
        path = tmp_path / "x.cf32"
        _tone(100).tofile(path)
        meta_path = write_sidecar(path, SampleFormat.FLOAT32, 0.0, 0.0)
        with open(meta_path) as handle:
            meta = json.load(handle)
        assert "core:sample_rate" not in meta["global"]
        assert "core:frequency" not in meta["captures"][0]
        _samples, info = load_iq_file(path)
        assert (info.sample_rate, info.center_frequency) == (0.0, 0.0)

    def test_wav_sidecar_keeps_the_centre_and_is_a_valid_dataset(self, tmp_path):
        x = _tone()
        path = tmp_path / "take.wav"
        save_iq_file(path, x, RATE, 0.0, SampleFormat.INT16, FileFormat.WAV)
        write_sidecar(path, SampleFormat.INT16, RATE, 145.8e6, header_bytes=44)
        _samples, info = load_iq_file(path)
        assert (info.sample_rate, info.center_frequency) == (RATE, 145.8e6)
        # Opening the metadata itself reads the WAV's samples after its header.
        samples, meta = load_iq_file(sidecar_path(path))
        assert meta.center_frequency == 145.8e6
        np.testing.assert_allclose(samples, x, atol=1e-4)

    def test_opening_a_sidecar_reads_its_dataset(self, tmp_path):
        x = _tone()
        path = tmp_path / "take.cf32"
        x.tofile(path)
        write_sidecar(path, SampleFormat.FLOAT32, RATE, 100.1e6)
        samples, meta = load_iq_file(sidecar_path(path))
        np.testing.assert_array_equal(samples, x)
        assert meta.center_frequency == 100.1e6
        info = detect_format(sidecar_path(path))
        assert info.num_samples == len(x)
        assert info.sample_rate == RATE


class TestOtherFilesMetadata:
    """rx:RX-16: a raw file next to a SigMF recording of the same stem."""

    def _pair(self, tmp_path):
        save_iq_file(
            tmp_path / "take.sigmf-data",
            _tone(5000),
            1e6,
            433.92e6,
            SampleFormat.FLOAT32,
            FileFormat.SIGMF,
        )
        _write_cs16(tmp_path / "take.cs16", _tone(1000))
        return tmp_path / "take.cs16"

    def test_detect_format_reads_the_raw_file_itself(self, tmp_path):
        path = self._pair(tmp_path)
        info = detect_format(path)
        assert info.file_format is FileFormat.RAW
        assert info.sample_format is SampleFormat.INT16
        assert info.num_samples == 1000
        assert (info.sample_rate, info.center_frequency) == (0.0, 0.0)
        assert find_sidecar(path) is None

    def test_get_file_info_too(self, tmp_path):
        path = self._pair(tmp_path)
        info = get_file_info(path)
        assert info["sample_format"] == "cs16"
        assert info["num_samples"] == 1000
        assert info["sample_rate"] == "unknown"
        assert info["center_frequency"] == "not specified"

    def test_a_same_stem_meta_naming_the_file_is_used(self, tmp_path):
        path = tmp_path / "cap.cu8"
        _write_cu8(path, _tone(800))
        with open(tmp_path / "cap.sigmf-meta", "w") as handle:
            json.dump(
                {
                    "global": {
                        "core:datatype": "cu8",
                        "core:sample_rate": 2.048e6,
                        "core:dataset": "cap.cu8",
                        "core:version": "1.0.0",
                    },
                    "captures": [{"core:sample_start": 0, "core:frequency": 1090e6}],
                    "annotations": [],
                },
                handle,
            )
        info = detect_format(path)
        assert (info.sample_rate, info.center_frequency) == (2.048e6, 1090e6)
        assert info.num_samples == 800


class TestFileInfoOverrides:
    def test_overrides_for_a_headerless_file(self, tmp_path):
        path = tmp_path / "capture.dat"
        _write_cu8(path, _tone(48000))
        info = get_file_info(
            path,
            sample_format=SampleFormat.UINT8,
            sample_rate=2.4e6,
            center_frequency=433.92e6,
        )
        assert info["sample_format"] == "cu8"
        assert info["num_samples"] == 48000
        assert info["sample_rate_hz"] == 2.4e6
        assert info["center_frequency_hz"] == 433.92e6
        assert info["duration_seconds"] == pytest.approx(0.02)


class TestContentDetection:
    def _guess(self, path):
        return FormatDetector().detect_sample_format_from_data(path)

    def test_8_and_16_bit_and_float(self, tmp_path):
        rng = np.random.default_rng(1)
        noise = (
            0.05 * (rng.standard_normal(20000) + 1j * rng.standard_normal(20000))
        ).astype(np.complex64)
        x = noise + _tone(20000, amplitude=0.3)
        _write_cu8(tmp_path / "a", x)
        _write_cs8(tmp_path / "b", x)
        _write_cs16(tmp_path / "c", x)
        x.tofile(tmp_path / "d")
        assert self._guess(tmp_path / "a") is SampleFormat.UINT8
        assert self._guess(tmp_path / "b") is SampleFormat.INT8
        assert self._guess(tmp_path / "c") is SampleFormat.INT16
        assert self._guess(tmp_path / "d") is SampleFormat.FLOAT32


class TestSampleCheck:
    def test_garbage_is_reported(self, tmp_path):
        path = tmp_path / "rtl.bin"
        _write_cu8(path, _tone(48000))
        check = check_iq_samples(path, SampleFormat.FLOAT32)
        assert not check.ok
        assert "cf32" in check.problem

    def test_good_data_passes(self, tmp_path):
        path = tmp_path / "rtl.bin"
        _write_cu8(path, _tone(48000))
        check = check_iq_samples(path, SampleFormat.UINT8)
        assert check.ok, check.problem
        assert check.samples > 0
        assert check.max_abs == pytest.approx(0.5, abs=0.02)

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.cf32"
        path.write_bytes(b"")
        check = check_iq_samples(path)
        assert not check.ok
        assert check.samples == 0
        assert "no I/Q samples" in check.problem

    def test_a_format_the_file_states_isnt_one_to_choose(self, tmp_path):
        # A SigMF (or WAV, or sidecar) file says its own format: its check
        # used to advise choosing another sample format, which can't be.
        loud = _tone(48000) * 1000
        sigmf = tmp_path / "loud.sigmf-data"
        save_iq_file(sigmf, loud, RATE, 100e6, SampleFormat.FLOAT32, FileFormat.SIGMF)
        check = check_iq_samples(sigmf)
        assert not check.ok
        assert "Choose another sample format" not in check.problem
        assert "another scale" in check.problem
        bad = np.full(48000, np.nan, np.complex64)
        sigmf_nan = tmp_path / "nan.sigmf-data"
        save_iq_file(
            sigmf_nan, bad, RATE, 100e6, SampleFormat.FLOAT32, FileFormat.SIGMF
        )
        check = check_iq_samples(sigmf_nan)
        assert "Choose another sample format" not in check.problem
        assert "damaged" in check.problem
        # The same samples in a headerless file: choosing another format is
        # the advice.
        raw = tmp_path / "loud.cf32"
        loud.tofile(raw)
        check = check_iq_samples(raw, SampleFormat.FLOAT32)
        assert "Choose another sample format" in check.problem


def test_reading_releases_the_file(tmp_path):
    path = tmp_path / "x.cf32"
    _tone(10).tofile(path)
    load_iq_file(path)
    os.remove(path)  # (Windows: fails while open)

"""Tests for the sdr-scan command line interface."""

import numpy as np
import pytest

from sdr_module.cli import create_parser, main
from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file


@pytest.fixture
def iq_capture(tmp_path):
    """Write a short two-tone I/Q capture and return its path + parameters."""
    fs = 2.4e6
    center = 100e6
    t = np.arange(int(fs * 0.02)) / fs
    # Tones at +200 kHz and -500 kHz relative to center, plus a little noise.
    sig = 0.7 * np.exp(2j * np.pi * 200e3 * t) + 0.5 * np.exp(-2j * np.pi * 500e3 * t)
    rng = np.random.default_rng(0)
    sig += 0.01 * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))

    path = tmp_path / "capture.cf32"
    save_iq_file(
        path,
        sig.astype(np.complex64),
        sample_rate=fs,
        center_frequency=center,
        sample_format=SampleFormat.FLOAT32,
        file_format=FileFormat.RAW,
    )
    return path, fs, center


class TestParser:
    def test_known_subcommands_present(self):
        parser = create_parser()
        # argparse stores subcommand names on the subparsers action.
        names = set()
        for action in parser._actions:
            if hasattr(action, "choices") and action.choices:
                names.update(action.choices.keys())
        assert {"info", "devices", "scan", "encode", "decode", "gui"} <= names


class TestInfoAndDevices:
    def test_info_runs(self, capsys):
        assert main(["info"]) == 0
        assert "SDR Module" in capsys.readouterr().out

    def test_no_command_shows_info(self, capsys):
        assert main([]) == 0
        assert "SDR Module" in capsys.readouterr().out


class TestScanOffline:
    def test_detects_both_tones(self, capsys, iq_capture):
        path, fs, center = iq_capture
        rc = main(
            [
                "scan",
                "--input",
                str(path),
                "--sample-rate",
                str(fs),
                "--center",
                str(center),
                "--threshold",
                "-20",
            ]
        )
        out = capsys.readouterr().out
        assert rc == 0
        assert "Detected 2 signal(s)" in out
        # The two tones should appear near 100.2 MHz and 99.5 MHz.
        assert "100.19" in out or "100.20" in out
        assert "99.49" in out or "99.50" in out

    def test_missing_file(self, capsys, tmp_path):
        rc = main(
            ["scan", "--input", str(tmp_path / "nope.cf32"), "--sample-rate", "1"]
        )
        assert rc == 1
        assert "file not found" in capsys.readouterr().out

    def test_raw_without_sample_rate_errors(self, capsys, iq_capture):
        path, _, _ = iq_capture
        rc = main(["scan", "--input", str(path)])
        assert rc == 1
        assert "sample rate unknown" in capsys.readouterr().out

    def test_live_mode_without_input_is_informative(self, capsys):
        rc = main(["scan", "--start", "88", "--end", "108"])
        assert rc == 0
        assert "--input" in capsys.readouterr().out


class TestDecode:
    @pytest.mark.parametrize(
        "protocol", ["pocsag", "flex", "ax25", "aprs", "rds", "adsb", "acars"]
    )
    def test_decode_runs_cleanly(self, capsys, iq_capture, protocol):
        path, fs, _ = iq_capture
        rc = main(["decode", protocol, "--input", str(path), "--sample-rate", str(fs)])
        out = capsys.readouterr().out
        assert rc == 0
        assert f"{protocol.upper()} message(s)" in out

    def test_missing_file(self, capsys, tmp_path):
        rc = main(
            [
                "decode",
                "adsb",
                "--input",
                str(tmp_path / "x.cf32"),
                "--sample-rate",
                "1",
            ]
        )
        assert rc == 1
        assert "file not found" in capsys.readouterr().out


class TestEncode:
    def test_encode_morse_to_file(self, capsys, tmp_path):
        out_file = tmp_path / "morse.cf32"
        rc = main(["encode", "morse", "--text", "SOS", "--output", str(out_file)])
        assert rc == 0
        assert out_file.exists()
        assert "MORSE" in capsys.readouterr().out

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
        assert {
            "info",
            "devices",
            "scan",
            "encode",
            "decode",
            "gui",
            "channels",
        } <= names


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

    @pytest.mark.parametrize(
        "protocol,rate",
        [
            ("adsb", "48000"),  # below the 2 MHz Mode S minimum; used to hang
            ("pocsag", "400"),
            ("flex", "1000"),
            ("ax25", "800"),
            ("acars", "1000"),
        ],
    )
    def test_sample_rate_too_low_errors(self, capsys, iq_capture, protocol, rate):
        path, _, _ = iq_capture
        rc = main(["decode", protocol, "--input", str(path), "--sample-rate", rate])
        out = capsys.readouterr().out
        assert rc == 1
        assert "Error:" in out
        assert "sample rate" in out.lower() or "sample_rate" in out


class TestScanSigMF:
    def test_scan_accepts_meta_or_data_path(self, capsys, tmp_path):
        fs = 48000.0
        t = np.arange(int(fs)) / fs
        sig = (0.5 * np.exp(2j * np.pi * 1000 * t)).astype(np.complex64)
        save_iq_file(
            tmp_path / "cap.sigmf-data",
            sig,
            sample_rate=fs,
            center_frequency=100e6,
            sample_format=SampleFormat.FLOAT32,
            file_format=FileFormat.SIGMF,
        )
        # Rate and center frequency come from the metadata for both paths.
        for name in ("cap.sigmf-data", "cap.sigmf-meta"):
            rc = main(["scan", "--input", str(tmp_path / name)])
            out = capsys.readouterr().out
            assert rc == 0, out
            assert "at 0.048 Msps" in out
            assert "Center: 100.0000 MHz" in out


class TestEncode:
    def test_encode_morse_to_file(self, capsys, tmp_path):
        out_file = tmp_path / "morse.cf32"
        rc = main(["encode", "morse", "--text", "SOS", "--output", str(out_file)])
        assert rc == 0
        assert out_file.exists()
        assert "MORSE" in capsys.readouterr().out

    def test_encode_to_wav_writes_real_wav(self, capsys, tmp_path):
        import wave

        out_file = tmp_path / "morse.wav"
        rc = main(["encode", "morse", "--text", "SOS", "--output", str(out_file)])
        assert rc == 0
        with wave.open(str(out_file)) as w:
            assert w.getnchannels() == 2  # I and Q
            assert w.getframerate() == 48000
            assert w.getnframes() > 0


class TestChannels:
    """CHIRP-compatible saved-channel import/export."""

    def test_export_presets_writes_chirp_csv(self, capsys, tmp_path):
        out_file = tmp_path / "presets.csv"
        rc = main(["channels", "export", str(out_file), "--presets"])
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "CHIRP" in out

        from sdr_module.core.chirp_csv import CHIRP_COLUMNS, read_chirp_csv
        from sdr_module.core.frequency_manager import RX_PRESETS

        header = out_file.read_text(encoding="utf-8").splitlines()[0]
        assert header == ",".join(CHIRP_COLUMNS)
        assert len(read_chirp_csv(out_file).channels) == len(RX_PRESETS)

    def test_list_from_file(self, capsys, tmp_path):
        csv_file = tmp_path / "ch.csv"
        csv_file.write_text(
            "Location,Name,Frequency,Mode\n0,Calling,146.520000,FM\n", encoding="utf-8"
        )
        rc = main(["channels", "list", str(csv_file)])
        out = capsys.readouterr().out
        assert rc == 0, out
        assert "146.5200 MHz" in out
        assert "Calling" in out
        assert "1 channel(s)" in out

    def test_list_shows_repeater_shift(self, capsys, tmp_path):
        csv_file = tmp_path / "rpt.csv"
        csv_file.write_text(
            "Name,Frequency,Duplex,Offset,Tone\nRptr,146.940000,-,0.600000,TSQL\n",
            encoding="utf-8",
        )
        assert main(["channels", "list", str(csv_file)]) == 0
        out = capsys.readouterr().out
        assert "TSQL" in out
        assert "-0.6" in out

    def test_list_missing_file(self, capsys, tmp_path):
        rc = main(["channels", "list", str(tmp_path / "nope.csv")])
        assert rc == 1
        assert "file not found" in capsys.readouterr().out

    def test_import_rejects_non_chirp_file(self, capsys, tmp_path):
        csv_file = tmp_path / "other.csv"
        csv_file.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        rc = main(["channels", "import", str(csv_file)])
        assert rc == 1
        assert "Frequency" in capsys.readouterr().out

    def test_export_round_trips_through_import_format(self, tmp_path):
        """A file we export re-reads with identical channel data."""
        from sdr_module.core.chirp_csv import read_chirp_csv, write_chirp_csv

        out_file = tmp_path / "presets.csv"
        assert main(["channels", "export", str(out_file), "--presets"]) == 0

        channels = read_chirp_csv(out_file).channels
        again = tmp_path / "again.csv"
        write_chirp_csv(again, channels, renumber=True)
        assert again.read_bytes() == out_file.read_bytes()

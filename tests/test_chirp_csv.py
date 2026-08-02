"""Tests for CHIRP-compatible saved-channel CSV import/export."""

import csv
import io

import pytest

from sdr_module.core.chirp_csv import (
    CHIRP_COLUMNS,
    Channel,
    ChirpCsvError,
    chirp_mode_to_demod,
    demod_to_chirp_mode,
    export_bookmarks_csv,
    format_freq,
    import_bookmarks_csv,
    parse_freq,
    presets_to_channels,
    read_chirp_csv,
    write_chirp_csv,
)

# A real CHIRP generic-CSV export: 2 m simplex, a -600 kHz repeater with a
# TSQL tone, an AM airband channel marked skip, and a split-duplex channel.
CHIRP_SAMPLE = (
    "Location,Name,Frequency,Duplex,Offset,Tone,rToneFreq,cToneFreq,DtcsCode,"
    "DtcsPolarity,RxDtcsCode,CrossMode,Mode,TStep,Skip,Power,Comment,URCALL,"
    "RPT1CALL,RPT2CALL,DVCODE\r\n"
    "0,Calling,146.520000,,0.000000,,88.5,88.5,023,NN,023,Tone->Tone,FM,5.00,"
    ",5.0W,2m FM calling,,,,\r\n"
    "1,W1AW Rptr,146.940000,-,0.600000,TSQL,100.0,100.0,023,NN,023,Tone->Tone,"
    "FM,5.00,,50W,Local repeater,,,,\r\n"
    "2,Tower,118.300000,,0.000000,,88.5,88.5,023,NN,023,Tone->Tone,AM,25.00,S,"
    ",Airport tower,,,,\r\n"
    "3,Split,145.100000,split,146.700000,,88.5,88.5,023,NN,023,Tone->Tone,FM,"
    "5.00,,,Odd split,,,,\r\n"
)


class TestFrequencyParsing:
    def test_bare_number_is_mhz(self):
        assert parse_freq("146.520000") == 146_520_000.0

    def test_exact_hz_without_float_error(self):
        # 146.52 * 1e6 in binary float is 146520000.00000003.
        assert parse_freq("146.52") == 146_520_000

    def test_blank_is_zero(self):
        assert parse_freq("") == 0.0
        assert parse_freq(None) == 0.0

    def test_thousands_separator(self):
        assert parse_freq("1,296.000000") == 1_296_000_000.0

    def test_explicit_units(self):
        assert parse_freq("146520 kHz") == 146_520_000.0
        assert parse_freq("1.2 GHz") == 1_200_000_000.0

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_freq("not-a-frequency")

    def test_format_matches_chirp(self):
        assert format_freq(146_520_000.0) == "146.520000"
        assert format_freq(0.0) == "0.000000"


class TestHeader:
    def test_column_order_matches_chirp(self):
        assert list(CHIRP_COLUMNS) == [
            "Location",
            "Name",
            "Frequency",
            "Duplex",
            "Offset",
            "Tone",
            "rToneFreq",
            "cToneFreq",
            "DtcsCode",
            "DtcsPolarity",
            "RxDtcsCode",
            "CrossMode",
            "Mode",
            "TStep",
            "Skip",
            "Power",
            "Comment",
            "URCALL",
            "RPT1CALL",
            "RPT2CALL",
            "DVCODE",
        ]

    def test_written_header_is_exact(self, tmp_path):
        path = tmp_path / "out.csv"
        write_chirp_csv(path, [Channel(name="Test", frequency_hz=100e6)])
        first_line = path.read_text(encoding="utf-8").splitlines()[0]
        assert first_line == ",".join(CHIRP_COLUMNS)


class TestReadChirpFile:
    @pytest.fixture
    def channels(self, tmp_path):
        path = tmp_path / "chirp.csv"
        path.write_text(CHIRP_SAMPLE, encoding="utf-8")
        return read_chirp_csv(path).channels

    def test_reads_every_row(self, channels):
        assert len(channels) == 4

    def test_simplex_channel(self, channels):
        ch = channels[0]
        assert ch.location == 0
        assert ch.name == "Calling"
        assert ch.frequency_hz == 146_520_000.0
        assert ch.mode == "FM"
        assert ch.duplex == ""
        assert ch.tone_mode == ""
        assert ch.power == "5.0W"
        assert ch.comment == "2m FM calling"
        assert ch.tx_frequency_hz == 146_520_000.0

    def test_repeater_channel(self, channels):
        ch = channels[1]
        assert ch.duplex == "-"
        assert ch.offset_hz == 600_000.0
        assert ch.tone_mode == "TSQL"
        assert ch.ctone == 100.0
        assert ch.tx_frequency_hz == 146_340_000.0

    def test_am_channel_with_skip(self, channels):
        ch = channels[2]
        assert ch.mode == "AM"
        assert ch.skip == "S"
        assert ch.tuning_step_khz == 25.0
        assert ch.demod == "AM"

    def test_split_duplex_tx_frequency(self, channels):
        ch = channels[3]
        assert ch.duplex == "split"
        assert ch.tx_frequency_hz == 146_700_000.0

    def test_dtcs_code_with_leading_zeros(self, channels):
        assert channels[0].dtcs == 23
        assert channels[0].rx_dtcs == 23


class TestReadTolerance:
    def test_minimal_file_with_only_required_column(self, tmp_path):
        path = tmp_path / "min.csv"
        path.write_text("Name,Frequency\nWX1,162.550000\n", encoding="utf-8")
        (ch,) = read_chirp_csv(path).channels
        assert ch.name == "WX1"
        assert ch.frequency_hz == 162_550_000.0
        # CHIRP defaults fill in the rest.
        assert ch.mode == "FM"
        assert ch.rtone == 88.5
        assert ch.location is None

    def test_case_insensitive_headers_and_unknown_columns(self, tmp_path):
        path = tmp_path / "odd.csv"
        path.write_text(
            "NAME,frequency,MODE,SomethingElse\nX,446.000000,NFM,ignored\n",
            encoding="utf-8",
        )
        (ch,) = read_chirp_csv(path).channels
        assert ch.name == "X"
        assert ch.mode == "NFM"

    def test_bom_is_stripped(self, tmp_path):
        path = tmp_path / "bom.csv"
        path.write_text("Name,Frequency\nA,100.000000\n", encoding="utf-8-sig")
        (ch,) = read_chirp_csv(path).channels
        assert ch.name == "A"

    def test_blank_rows_skipped(self, tmp_path):
        path = tmp_path / "blanks.csv"
        path.write_text("Name,Frequency\nA,100.000000\n,\n\nB,101.000000\n", "utf-8")
        assert len(read_chirp_csv(path).channels) == 2

    def test_reads_from_open_stream(self):
        report = read_chirp_csv(io.StringIO(CHIRP_SAMPLE))
        assert len(report.channels) == 4

    def test_missing_file(self, tmp_path):
        with pytest.raises(ChirpCsvError, match="file not found"):
            read_chirp_csv(tmp_path / "nope.csv")

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.csv"
        path.write_text("", encoding="utf-8")
        with pytest.raises(ChirpCsvError, match="empty"):
            read_chirp_csv(path)

    def test_not_a_chirp_file(self, tmp_path):
        path = tmp_path / "other.csv"
        path.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
        with pytest.raises(ChirpCsvError, match="Frequency"):
            read_chirp_csv(path)

    def test_bad_row_raises_with_line_number(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("Name,Frequency\nA,100.0\nB,oops\n", encoding="utf-8")
        with pytest.raises(ChirpCsvError, match="line 3"):
            read_chirp_csv(path)

    def test_bad_row_skipped_when_not_strict(self, tmp_path):
        path = tmp_path / "bad.csv"
        path.write_text("Name,Frequency\nA,100.0\nB,oops\nC,\n", encoding="utf-8")
        report = read_chirp_csv(path, strict=False)
        assert [c.name for c in report.channels] == ["A"]
        assert len(report.skipped) == 2

    def test_unknown_mode_is_kept(self, tmp_path):
        path = tmp_path / "mode.csv"
        path.write_text("Name,Frequency,Mode\nA,100.0,LoRa\n", encoding="utf-8")
        (ch,) = read_chirp_csv(path).channels
        assert ch.mode == "LoRa"
        assert ch.demod == "None (I/Q)"


class TestWriteAndRoundTrip:
    def test_round_trip_preserves_fields(self, tmp_path):
        original = read_chirp_csv(io.StringIO(CHIRP_SAMPLE)).channels
        path = tmp_path / "rt.csv"
        write_chirp_csv(path, original)
        reloaded = read_chirp_csv(path).channels
        assert original == reloaded

    def test_written_bytes_match_source(self, tmp_path):
        """A CHIRP file re-exported unchanged is byte-identical."""
        path = tmp_path / "rt.csv"
        write_chirp_csv(path, read_chirp_csv(io.StringIO(CHIRP_SAMPLE)).channels)
        assert path.read_bytes().decode("utf-8") == CHIRP_SAMPLE

    def test_locations_assigned_sequentially(self, tmp_path):
        path = tmp_path / "num.csv"
        write_chirp_csv(
            path,
            [Channel(name=f"C{i}", frequency_hz=100e6 + i) for i in range(3)],
        )
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [r["Location"] for r in rows] == ["0", "1", "2"]

    def test_renumber_overrides_stored_locations(self, tmp_path):
        path = tmp_path / "renum.csv"
        channels = [
            Channel(name="A", frequency_hz=100e6, location=17),
            Channel(name="B", frequency_hz=101e6, location=42),
        ]
        write_chirp_csv(path, channels, renumber=True, start_location=1)
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert [r["Location"] for r in rows] == ["1", "2"]

    def test_writes_to_open_stream(self):
        buffer = io.StringIO()
        count = write_chirp_csv(buffer, [Channel(name="A", frequency_hz=100e6)])
        assert count == 1
        assert "100.000000" in buffer.getvalue()

    def test_field_formatting_matches_chirp(self, tmp_path):
        path = tmp_path / "fmt.csv"
        write_chirp_csv(path, [Channel(name="A", frequency_hz=146.52e6)])
        with open(path, newline="", encoding="utf-8") as handle:
            (row,) = list(csv.DictReader(handle))
        assert row["Frequency"] == "146.520000"
        assert row["DtcsCode"] == "023"
        assert row["rToneFreq"] == "88.5"
        assert row["TStep"] == "5.00"


class TestChannelValidation:
    def test_negative_frequency_rejected(self):
        with pytest.raises(ValueError):
            Channel(name="bad", frequency_hz=-1.0)

    def test_constrained_fields_normalised(self):
        ch = Channel(
            name=" Padded ",
            frequency_hz=100e6,
            duplex="SPLIT",
            tone_mode="tsql",
            skip="s",
            mode="nfm",
            dtcs_polarity="nr",
        )
        assert ch.name == "Padded"
        assert ch.duplex == "split"
        assert ch.tone_mode == "TSQL"
        assert ch.skip == "S"
        assert ch.mode == "NFM"
        assert ch.dtcs_polarity == "NR"

    def test_duplex_off_has_no_tx_frequency(self):
        ch = Channel(name="RX only", frequency_hz=162.55e6, duplex="off")
        assert ch.tx_frequency_hz == 0.0


class TestModeMapping:
    @pytest.mark.parametrize(
        "mode,demod",
        [
            ("FM", "FM"),
            ("NFM", "FM"),
            ("WFM", "FM"),
            ("AM", "AM"),
            ("NAM", "AM"),
            ("USB", "USB"),
            ("LSB", "LSB"),
            ("CW", "CW"),
            ("CWR", "CW"),
            ("P25", "FM"),
            ("Auto", "None (I/Q)"),
        ],
    )
    def test_chirp_mode_to_demod(self, mode, demod):
        assert chirp_mode_to_demod(mode) == demod

    @pytest.mark.parametrize(
        "demod,mode",
        [
            ("FM", "FM"),
            ("AM", "AM"),
            ("USB", "USB"),
            ("LSB", "LSB"),
            ("CW", "CW"),
            ("None (I/Q)", "FM"),
            ("RAW", "FM"),
        ],
    )
    def test_demod_to_chirp_mode(self, demod, mode):
        assert demod_to_chirp_mode(demod) == mode


class TestBookmarkBridge:
    def test_legacy_bookmark_converts(self):
        ch = Channel.from_bookmark({"label": "FM Broadcast", "freq_hz": 100.1e6})
        assert ch.name == "FM Broadcast"
        assert ch.frequency_hz == 100.1e6
        assert ch.mode == "FM"

    def test_bookmark_round_trip_is_lossless(self):
        ch = Channel(
            name="Repeater",
            frequency_hz=147.0e6,
            duplex="+",
            offset_hz=600e3,
            tone_mode="Tone",
            rtone=123.0,
            mode="NFM",
            comment="club machine",
        )
        assert Channel.from_bookmark(ch.to_bookmark()) == ch

    def test_bookmark_keeps_legacy_keys(self):
        bookmark = Channel(name="A", frequency_hz=100e6).to_bookmark()
        assert bookmark["label"] == "A"
        assert bookmark["freq_hz"] == 100e6
        # Default-valued fields are not persisted, keeping settings small.
        assert "rtone" not in bookmark

    def test_export_then_import_bookmarks(self, tmp_path):
        bookmarks = [
            {"label": "Weather", "freq_hz": 162.55e6},
            {"label": "Airband", "freq_hz": 118.3e6, "mode": "AM"},
        ]
        path = tmp_path / "bookmarks.csv"
        assert export_bookmarks_csv(path, bookmarks) == 2

        restored = import_bookmarks_csv(path)
        assert [b["label"] for b in restored] == ["Weather", "Airband"]
        assert restored[0]["freq_hz"] == 162.55e6
        assert restored[1]["mode"] == "AM"

    def test_exported_bookmarks_open_in_chirp_shape(self, tmp_path):
        path = tmp_path / "b.csv"
        export_bookmarks_csv(path, [{"label": "A", "freq_hz": 146.52e6}])
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            assert reader.fieldnames == list(CHIRP_COLUMNS)
            (row,) = list(reader)
        assert row["Name"] == "A"
        assert row["Frequency"] == "146.520000"
        assert row["Location"] == "0"


class TestPresetExport:
    def test_built_in_presets_export(self, tmp_path):
        from sdr_module.core.frequency_manager import RX_PRESETS

        channels = presets_to_channels(RX_PRESETS)
        assert len(channels) == len(RX_PRESETS)

        path = tmp_path / "presets.csv"
        write_chirp_csv(path, channels)
        reloaded = read_chirp_csv(path).channels
        assert [c.name for c in reloaded] == [p.name for p in RX_PRESETS]
        # "RAW" I/Q presets have no CHIRP equivalent and fall back to FM.
        assert all(c.mode in ("FM", "AM", "WFM", "CW", "USB", "LSB") for c in reloaded)

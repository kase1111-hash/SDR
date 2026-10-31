"""How a bookmark is listened to survives a CHIRP CSV round trip (TOOLS-14),
spreadsheet CSVs with ';' or tab separators import (UX-14), and the band
plan names new bookmarks ("FM Broadcast 100.1")."""

import io

import pytest

from sdr_module.core.band_plan import (
    band_for,
    default_name,
    listening_for,
    preset_for,
    short_mhz,
)
from sdr_module.core.chirp_csv import (
    Channel,
    chirp_mode_for,
    export_bookmarks_csv,
    import_bookmarks_csv,
    is_chirp_channel,
    listening_tag,
    presets_to_channels,
    read_chirp_csv,
    split_listening_tag,
    write_chirp_csv,
)


def _round_trip(bookmarks):
    buf = io.StringIO()
    export_bookmarks_csv(buf, bookmarks)
    return import_bookmarks_csv(io.StringIO(buf.getvalue())), buf.getvalue()


class TestListeningRoundTrip:
    def test_audit_case_tower_and_2m_simplex(self):
        """The audit's repro: 'Tower' typed in at 118.1 MHz (now saved in AM)
        and '2m simplex' in FM 12.5 kHz come back listened to as before."""
        before = [
            {"label": "Tower", "freq_hz": 118.1e6, "demod": "AM", "mode": "AM"},
            {
                "label": "2m simplex",
                "freq_hz": 146.52e6,
                "demod": "FM",
                "fm_deviation": "12.5 kHz",
                "mode": "FM",
            },
            {
                "label": "Wx sat",
                "freq_hz": 137.62e6,
                "demod": "FM",
                "fm_deviation": "25 kHz",
                "bandwidth": "50 kHz",
            },
        ]
        after, text = _round_trip(before)
        for old, new in zip(before, after, strict=True):
            for key in ("label", "freq_hz", "demod", "fm_deviation", "bandwidth"):
                assert new.get(key) == old.get(key), (key, old, new)
        # CHIRP's own Mode column says AM for the tower (not FM).
        rows = list(read_chirp_csv(io.StringIO(text)).channels)
        assert [c.mode for c in rows] == ["AM", "FM", "FM"]
        # The comment the application shows has no tag in it.
        assert all("comment" not in b for b in after)

    def test_a_bookmark_without_a_mode_still_keeps_the_mode(self):
        after, text = _round_trip([{"label": "Old", "freq_hz": 100.1e6}])
        assert "[sdr mode=keep]" in text
        assert after == [{"label": "Old", "freq_hz": 100.1e6}]
        assert not is_chirp_channel(after[0])

    def test_iq_group_and_comment(self):
        before = [
            {
                "label": "Remote",
                "freq_hz": 433.92e6,
                "demod": "None (I/Q)",
                "bandwidth": "500 kHz",
                "group": "Sensors",
                "comment": "garage door",
                "skip": "S",
            }
        ]
        after, text = _round_trip(before)
        assert "garage door [sdr mode=I/Q | bw=500 kHz | group=Sensors]" in text
        (b,) = after
        assert b["demod"] == "None (I/Q)"
        assert b["bandwidth"] == "500 kHz"
        assert b["group"] == "Sensors"
        assert b["comment"] == "garage door"
        assert b["skip"] == "S"

    def test_ctcss_tone_rides_in_chirp_fields(self):
        before = [
            {
                "label": "Club",
                "freq_hz": 145.29e6,
                "demod": "FM",
                "fm_deviation": "5 kHz",
                "tone_mode": "TSQL",
                "ctone": 100.0,
            }
        ]
        after, text = _round_trip(before)
        row = read_chirp_csv(io.StringIO(text)).channels[0]
        assert (row.tone_mode, row.ctone) == ("TSQL", 100.0)
        assert (after[0]["tone_mode"], after[0]["ctone"]) == ("TSQL", 100.0)

    def test_chirp_channels_without_a_tag_are_unchanged(self):
        ch = Channel(
            name="Rptr",
            frequency_hz=146.94e6,
            duplex="-",
            offset_hz=600e3,
            tone_mode="TSQL",
            ctone=100.0,
            mode="NFM",
            comment="club",
        )
        assert Channel.from_bookmark(ch.to_bookmark()) == ch
        assert listening_tag(ch.to_bookmark()) == ""

    def test_a_mode_changed_in_chirp_wins_over_the_tag(self):
        _after, text = _round_trip(
            [
                {"label": "A", "freq_hz": 118.1e6, "demod": "AM"},
                {"label": "B", "freq_hz": 100.1e6},
                {
                    "label": "C",
                    "freq_hz": 146.52e6,
                    "demod": "FM",
                    "fm_deviation": "12.5 kHz",
                },
            ]
        )
        rows = text.splitlines()
        # In CHIRP: A becomes USB, B becomes AM; C is left alone.
        rows[1] = rows[1].replace(",AM,", ",USB,")
        rows[2] = rows[2].replace(",FM,", ",AM,")
        after = import_bookmarks_csv(io.StringIO("\n".join(rows)))
        assert after[0].get("demod") is None
        assert after[0]["mode"] == "USB"
        assert after[1].get("mode") == "AM"
        assert "location" in after[1]  # a CHIRP channel now
        assert (after[2]["demod"], after[2]["fm_deviation"]) == ("FM", "12.5 kHz")

    def test_tag_parsing_is_forgiving(self):
        assert split_listening_tag("no tag") == ("no tag", {})
        text, found = split_listening_tag("x [sdr mode=usb; dev=5 kHz; bw=2.7 kHz]")
        assert text == "x"
        assert found == {
            "demod": "USB",
            "fm_deviation": "5 kHz",
            "bandwidth": "2.7 kHz",
        }
        assert split_listening_tag("[sdr mode=RTTY]")[1] == {}  # not a mode here
        assert split_listening_tag("[sdr mode=iq]")[1] == {"demod": "None (I/Q)"}
        # Separators in a group can't break the tag.
        tag = listening_tag({"label": "a", "freq_hz": 1e6, "group": "a;b]c|d"})
        assert split_listening_tag(tag)[1]["group"] == "a,b,c,d"

    def test_chirp_mode_for(self):
        assert chirp_mode_for("FM", "75 kHz") == "WFM"
        assert chirp_mode_for("FM", "2.5 kHz") == "NFM"
        assert chirp_mode_for("FM", "5 kHz") == "FM"
        assert chirp_mode_for("AM") == "AM"
        assert chirp_mode_for("None (I/Q)") == "FM"

    def test_built_in_presets_carry_their_listening(self, tmp_path):
        from sdr_module.core.frequency_manager import RX_PRESETS

        path = tmp_path / "presets.csv"
        write_chirp_csv(path, presets_to_channels(RX_PRESETS), renumber=True)
        imported = {b["label"]: b for b in import_bookmarks_csv(path)}
        assert imported["ACARS"]["demod"] == "AM"
        assert imported["FM Broadcast"]["demod"] == "FM"
        assert imported["FM Broadcast"]["fm_deviation"] == "75 kHz"
        assert imported["ADS-B"]["demod"] == "None (I/Q)"
        assert imported["QRP 20m SSB"]["demod"] == "USB"
        # The description is still the comment.
        assert imported["ACARS"]["comment"] == "Aircraft communications"


class TestSpreadsheetCsv:
    def test_semicolons_and_decimal_commas(self, tmp_path):
        path = tmp_path / "excel.csv"
        path.write_text(
            "Location;Name;Frequency;Mode\n0;Calling;146,520000;FM\n1;Tower;118,1;AM\n",
            encoding="utf-8",
        )
        channels = read_chirp_csv(path).channels
        assert [(c.name, c.frequency_hz, c.mode) for c in channels] == [
            ("Calling", 146.52e6, "FM"),
            ("Tower", 118.1e6, "AM"),
        ]

    def test_tabs(self, tmp_path):
        path = tmp_path / "tabs.csv"
        path.write_text("Name\tFrequency\nWx\t162.550000\n", encoding="utf-8")
        assert read_chirp_csv(path).channels[0].frequency_hz == 162.55e6

    def test_windows_encoding(self, tmp_path):
        path = tmp_path / "latin.csv"
        path.write_bytes("Name;Frequency\nMünchen;145,500\n".encode("cp1252"))
        (ch,) = read_chirp_csv(path).channels
        assert ch.name == "München"
        assert ch.frequency_hz == 145.5e6

    def test_thousands_separators_still_work_with_commas(self, tmp_path):
        path = tmp_path / "big.csv"
        path.write_text('Name,Frequency\nA,"1,296.000000"\n', encoding="utf-8")
        assert read_chirp_csv(path).channels[0].frequency_hz == 1296e6


class TestBandPlan:
    def test_default_names(self):
        assert default_name(100.1e6) == "FM Broadcast 100.1"
        assert default_name(146.52e6) == "2m Calling"  # a preset
        assert default_name(162.55e6) == "NOAA Weather 1"
        assert default_name(118.1e6) == "Airband 118.1"
        assert default_name(145.8125e6) == "ISS SSTV/Voice" or default_name(
            145.8125e6
        ).startswith("2m Ham")
        # (The US 70 cm FM simplex segment, not Europe's PMR446: leftover 14.)
        assert default_name(446.05e6) == "70cm Ham 446.05"
        assert default_name(433.92e6) == "ISM 433 433.92"  # narrower than 70 cm
        assert default_name(7.074e6) == "40m Ham 7.074"
        assert default_name(3e9) == ""

    def test_short_mhz(self):
        assert short_mhz(100.1e6) == "100.1"
        assert short_mhz(125e6) == "125.0"
        assert short_mhz(145.8125e6) == "145.8125"

    def test_listening(self):
        assert listening_for(100.1e6) == ("FM", "75 kHz", "200 kHz")
        assert listening_for(118.1e6) == ("AM", "", "10 kHz")
        assert listening_for(131.55e6) == ("AM", "", "25 kHz")  # ACARS preset
        assert listening_for(433.92e6)[0] == "None (I/Q)"
        assert listening_for(3e9) is None
        assert band_for(433.92e6).name == "ISM 433"
        assert preset_for(100.1e6) is None  # a band, not a channel
        assert preset_for(146.5203e6).name == "2m Calling"

    def test_unreadable_input(self):
        assert band_for("x") is None
        assert preset_for(None) is None


@pytest.mark.parametrize("delimiter", [";", "\t"])
def test_exported_file_resaved_by_a_spreadsheet_imports(delimiter, tmp_path):
    """Export, 'open in a spreadsheet and save' with another separator,
    import: the listening mode survives (the tag rides in Comment)."""
    buf = io.StringIO()
    export_bookmarks_csv(
        buf,
        [{"label": "Tower", "freq_hz": 118.1e6, "demod": "AM", "bandwidth": "10 kHz"}],
    )
    lines = [line.split(",") for line in buf.getvalue().splitlines()]
    path = tmp_path / "resaved.csv"
    path.write_text(
        "\n".join(delimiter.join(cells) for cells in lines), encoding="utf-8"
    )
    (b,) = import_bookmarks_csv(path)
    assert (b["demod"], b["bandwidth"]) == ("AM", "10 kHz")

"""M-D1: Decode Recording's aircraft and Morse text show in the Aircraft
and Text views, apart from what is heard live.

Before, the messages File > Decode Recording found went to the Decoder's
Messages table and Log only: rows from a recording stayed out of the
Aircraft and Text views, so that an aircraft heard in a recording didn't
update the one heard live (DEC-E2E-17). Now each view keeps what each
recording holds in a list of its own, picked under "Showing:" above it
(Live, or the recording's file name; the newest decode is shown when it
starts), timed by the recording's clock: its time of day when the
recording says when it was made (as the Messages table shows it), else
the time into the file. A recording's aircraft never dim or go as the wall
clock runs on, and live aircraft go on in their own list meanwhile. Clear
clears both; decoding the same recording again starts its lists over; the
views' empty-state hints say that recordings show there too.
"""

import json
import os
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_decoder_views import _adsb, _PanelCase, _settle  # noqa: E402
from tests.test_gui_recording_files import _FilesCase  # noqa: E402

REC = "adsb.sigmf-data"
WHEN = datetime(2026, 10, 30, 7, 0, 1, tzinfo=timezone.utc)


def _morse(text, wpm=18.0):
    from sdr_module.dsp.morse import MorseMessage
    from sdr_module.dsp.protocols import ProtocolType

    return MorseMessage(
        protocol=ProtocolType.CW,
        timestamp=0.0,
        raw_bits=b"",
        valid=True,
        text=text,
        wpm=wpm,
    )


def _klm(altitude=38000, lat=52.2568, lon=3.913):
    return _adsb(
        icao_address="4840D6",
        downlink_format=17,
        type_code=11,
        altitude=altitude,
        latitude=lat,
        longitude=lon,
    )


class _ViewsCase(_PanelCase):
    def cells(self, row=0):
        table = self.panel._aircraft_view.table
        return [table.item(row, c).text() for c in range(table.columnCount())]

    def recorded(self, msg, source=REC, **when):
        from sdr_module.dsp.live_decoders import describe_message

        address, content = describe_message(msg)
        name = "CW (Morse)" if type(msg).__name__ == "MorseMessage" else "ADS-B"
        self.panel.add_message(
            name, address, content, message=msg, source=source, **when
        )


# --------------------------------------------------------------------------- #
# Aircraft
# --------------------------------------------------------------------------- #


class TestRecordingAircraft(_ViewsCase):
    def setUp(self):
        super().setUp()
        self.panel._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        self.view = self.panel._aircraft_view

    def test_a_recordings_aircraft_get_a_list_of_their_own(self):
        from sdr_module.gui import decoder_views as dv

        p, view = self.panel, self.view
        p.add_message("ADS-B", "4840D6", "38,000 ft", message=_klm())
        self.assertFalse(view.source_bar().isVisible())  # (live only: as before)
        self.recorded(_klm(12000, 50.0, 4.0), when=WHEN, at_s=0.4)
        # The live aircraft is left alone (DEC-E2E-17 still holds) ...
        live = view.tracker.get("4840D6")
        self.assertEqual((live.messages, live.altitude), (1, 38000))
        # ... and the recording's is in its own list, shown now.
        recorded = view.tracker_for(REC).get("4840D6")
        self.assertEqual((recorded.messages, recorded.altitude), (1, 12000))
        self.assertEqual(view.sources(), [REC])
        self.assertEqual(view.shown_source(), REC)
        bar = view.source_bar()
        self.assertTrue(bar.isVisible())
        self.assertEqual(bar.current_text(), REC)
        self.assertEqual(bar.texts(), ["Live", REC])
        self.assertEqual(view.table.rowCount(), 1)
        self.assertEqual(self.cells()[dv.AC_ALTITUDE], "12,000")
        # Seen: the recording's own time (of day, as the table shows it).
        local = WHEN.astimezone().strftime("%H:%M:%S")
        self.assertEqual(self.cells()[dv.AC_SEEN], local)
        tip = view.table.item(0, dv.AC_NAME).toolTip()
        self.assertIn(REC, tip)
        self.assertIn(f"last heard at {local}", tip)
        # Live again: the live aircraft.
        bar.choose("")
        _settle(1)
        self.assertEqual(view.shown_source(), "")
        self.assertEqual(self.cells()[dv.AC_ALTITUDE], "38,000")
        self.assertEqual(self.cells()[dv.AC_SEEN], "now")

    def test_without_a_start_time_it_is_the_time_into_the_recording(self):
        from sdr_module.gui import decoder_views as dv

        self.recorded(_klm(), source="adsb.cs16", at_s=75.4)
        self.assertEqual(self.cells()[dv.AC_SEEN], "1:15")
        seen = self.view.table.horizontalHeaderItem(dv.AC_SEEN).toolTip()
        self.assertIn("recording", seen)
        self.recorded(_klm(), source="adsb.cs16", at_s=3725.0)
        self.assertEqual(self.cells()[dv.AC_SEEN], "1:02:05")
        self.assertIn("1:02:05 into adsb.cs16", self.view.detail_text("4840D6"))

    def test_they_dont_dim_or_go_as_the_wall_clock_runs_on(self):
        from sdr_module.gui import decoder_views as dv
        from sdr_module.gui.themes import get_palette

        p, view = self.panel, self.view
        p.add_message("ADS-B", "4840D6", "38,000 ft", message=_klm())
        self.recorded(_klm(12000), source="adsb.cs16", at_s=12.0)
        text = get_palette().qcolor("text")
        for minutes in (2, 10):
            self.now += 60.0 * minutes
            view.refresh()
            _settle(1)
            self.assertEqual(view.table.rowCount(), 1)
            self.assertEqual(self.cells()[dv.AC_SEEN], "0:12")
            item = view.table.item(0, dv.AC_NAME)
            self.assertEqual(item.foreground().color(), text, minutes)
        # The live one went after five minutes, in its own list.
        self.assertEqual(len(view.tracker), 0)
        self.assertEqual(len(view.tracker_for("adsb.cs16")), 1)

    def test_a_theme_change_recolours_a_recordings_rows(self):
        from PyQt6.QtWidgets import QApplication

        from sdr_module.gui import decoder_views as dv
        from sdr_module.gui import themes

        qapp = QApplication.instance()
        old_sheet, old_palette = qapp.styleSheet(), qapp.palette()
        old_theme = themes.current_theme()
        self.addCleanup(qapp.setPalette, old_palette)
        self.addCleanup(qapp.setStyleSheet, old_sheet)
        self.addCleanup(themes.apply_theme, None, old_theme)
        self.recorded(_klm(12000), when=WHEN, at_s=0.4)
        for theme in ("light", "dark", "light"):
            themes.apply_theme(qapp, theme)
            _settle(1)
            item = self.view.table.item(0, dv.AC_NAME)
            self.assertEqual(
                item.foreground().color(), themes.get_palette().qcolor("text"), theme
            )

    def test_live_messages_go_on_in_the_live_list_meanwhile(self):
        from sdr_module.gui import decoder_views as dv

        p, view = self.panel, self.view
        self.recorded(_klm(12000), when=WHEN, at_s=0.4)
        p.add_message("ADS-B", "3C6586", "DLH9U", message=_adsb(
            icao_address="3C6586", downlink_format=17, type_code=4,
            callsign="DLH9U"))  # fmt: skip
        self.assertEqual(view.shown_source(), REC)  # (not switched by live)
        self.assertEqual(view.table.rowCount(), 1)
        self.assertEqual(self.cells()[dv.AC_NAME], "4840D6")
        view.source_bar().choose("")
        self.assertEqual(view.table.rowCount(), 1)
        self.assertEqual(self.cells()[dv.AC_NAME], "DLH9U")

    def test_decoding_the_recording_again_starts_its_list_over(self):
        p, view = self.panel, self.view
        self.recorded(_klm(12000), when=WHEN, at_s=0.4)
        p.recording_decode_started(REC)
        self.recorded(_klm(12000), when=WHEN, at_s=0.4)
        self.assertEqual(view.tracker_for(REC).get("4840D6").messages, 1)
        self.assertEqual(view.sources(), [REC])

    def test_picking_a_recordings_aircraft_filters_the_messages(self):
        p, view = self.panel, self.view
        self.recorded(_adsb(icao_address="4840D6", downlink_format=17,
                            type_code=4, callsign="KLM1023"),
                      when=WHEN, at_s=0.1)  # fmt: skip
        self.recorded(_klm(), when=WHEN, at_s=0.4)
        view.table.selectRow(0)
        _settle(1)
        self.assertEqual(p.address_filter(), "4840D6")
        self.assertIn("KLM1023 (4840D6)", p._filter_label.text())

    def test_clear_clears_them(self):
        p, view = self.panel, self.view
        p.add_message("ADS-B", "4840D6", "38,000 ft", message=_klm())
        self.recorded(_klm(12000), when=WHEN, at_s=0.4)
        p.clear()
        _settle(1)
        self.assertEqual(view.sources(), [])
        self.assertEqual(view.shown_source(), "")
        self.assertFalse(view.source_bar().isVisible())
        self.assertEqual(view.table.rowCount(), 0)
        self.assertIsNone(view.tracker_for(REC))
        self.assertTrue(view.placeholder().is_visible())

    def test_the_view_shows_whatever_protocol_is_chosen(self):
        # Decode Recording's own protocol may not be the one chosen for
        # live decoding (POCSAG here): the Aircraft tab comes for the
        # recording's aircraft, in front, and goes with Clear.
        p = self.panel
        p._proto_combo.setCurrentText("POCSAG")
        _settle(1)
        self.assertNotIn("Aircraft", self.tab_titles())
        self.recorded(_klm(12000), when=WHEN, at_s=0.4)
        _settle(1)
        self.assertIn("Aircraft", self.tab_titles())
        self.assertIs(p._tabs.currentWidget(), self.view)
        p.clear()
        _settle(1)
        self.assertNotIn("Aircraft", self.tab_titles())
        self.assertIs(p._tabs.currentWidget(), p._messages_page)

    def test_an_invalid_message_from_a_recording_builds_no_aircraft(self):
        bad = _klm()
        bad.valid = False
        self.panel.add_message("ADS-B", "4840D6", "x", valid=False, message=bad,
                               source=REC, when=WHEN, at_s=0.2)  # fmt: skip
        self.assertEqual(self.view.sources(), [])


# --------------------------------------------------------------------------- #
# Morse text
# --------------------------------------------------------------------------- #


class TestRecordingText(_ViewsCase):
    def setUp(self):
        super().setUp()
        self.panel._proto_combo.setCurrentText("CW (Morse)")
        _settle(1)
        self.view = self.panel._cw_view

    def test_a_recordings_text_comes_with_its_times(self):
        p, view = self.panel, self.view
        p.add_message(
            "CW (Morse)", "18 WPM", "VVV DE SDR", message=_morse("VVV DE SDR")
        )
        self.recorded(_morse("CQ DE N0CALL"), source="cw.wav", at_s=12.3)
        self.recorded(_morse("TU 73", 22.0), source="cw.wav", at_s=75.0)
        self.assertEqual(view.plain_text(), "VVV DE SDR")  # (the live text)
        self.assertEqual(view.plain_text("cw.wav"), "0:12  CQ DE N0CALL\n1:15  TU 73")
        self.assertEqual(view.shown_source(), "cw.wav")
        self.assertEqual(view.shown_text(), view.plain_text("cw.wav"))
        self.assertEqual(view.speed.text(), "22 WPM")  # (its last line's)
        view.source_bar().choose("")
        _settle(1)
        self.assertEqual(view.shown_text(), "VVV DE SDR")

    def test_with_a_start_time_its_time_of_day(self):
        self.recorded(_morse("CQ DE N0CALL"), source="cw.sigmf-data",
                      when=WHEN + timedelta(seconds=2), at_s=2.0)  # fmt: skip
        local = (WHEN + timedelta(seconds=2)).astimezone().strftime("%H:%M:%S")
        self.assertEqual(
            self.view.plain_text("cw.sigmf-data"), f"{local}  CQ DE N0CALL"
        )

    def test_live_text_and_the_line_in_progress_stay_live(self):
        p, view = self.panel, self.view
        self.recorded(_morse("CQ DE N0CALL"), source="cw.wav", at_s=12.3)
        status = type("S", (), {"live_text": "VVV D", "detail": "18 WPM",
                                "signal_present": True, "problem": None})()  # fmt: skip
        p.set_decoder_status(status)
        self.assertEqual(view.shown_text(), "0:12  CQ DE N0CALL")
        view.source_bar().choose("")
        self.assertEqual(view.shown_text(), "VVV D")
        self.assertEqual(view.speed.text(), "18 WPM")

    def test_decoding_it_again_starts_its_text_over(self):
        self.recorded(_morse("CQ DE N0CALL"), source="cw.wav", at_s=12.3)
        self.panel.recording_decode_started("cw.wav")
        self.recorded(_morse("CQ DE N0CALL"), source="cw.wav", at_s=12.3)
        self.assertEqual(self.view.plain_text("cw.wav"), "0:12  CQ DE N0CALL")

    def test_clear_clears_it(self):
        self.recorded(_morse("CQ DE N0CALL"), source="cw.wav", at_s=12.3)
        self.panel.clear()
        _settle(1)
        self.assertEqual(self.view.sources(), [])
        self.assertEqual(self.view.plain_text("cw.wav"), "")
        self.assertFalse(self.view.source_bar().isVisible())
        self.assertTrue(self.view.placeholder().is_visible())


# --------------------------------------------------------------------------- #
# Empty states, and add_message as other callers use it
# --------------------------------------------------------------------------- #


class TestHintsAndCalls(_ViewsCase):
    def test_the_empty_views_say_recordings_show_there_too(self):
        from sdr_module.gui.decoder_views import AircraftView, MorseTextView

        p = self.panel
        p._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        self.assertIn("Decode Recording", p._aircraft_view.placeholder().text())
        p.set_receiving(True)
        p.set_current_frequency(1090e6)
        self.assertIn("Decode Recording", p._aircraft_view.placeholder().text())
        p._proto_combo.setCurrentText("CW (Morse)")
        _settle(1)
        self.assertIn("Decode Recording", p._cw_view.placeholder().text())
        for view in (AircraftView(), MorseTextView()):
            self.addCleanup(view.deleteLater)
            self.assertIn("Decode Recording", view.placeholder().text())

    def test_add_message_is_called_as_before(self):
        # Positional arguments and the keywords other callers pass.
        p = self.panel
        p._proto_combo.setCurrentText("ADS-B")
        p.add_message("ADS-B", "4840D6", "38,000 ft", True, "", message=_klm(),
                      details=["x"], freq_hz=1090e6, sample_rate=2.4e6)  # fmt: skip
        p.add_message("POCSAG", "1 (F0)", "from file", source="pocsag.sigmf-data")
        p.add_message("ADS-B", "4840D6", "12,000 ft", message=_klm(12000),
                      source=REC, when=WHEN)  # fmt: skip
        self.assertEqual(p.get_message_count(), 3)
        self.assertEqual(self.panel._aircraft_view.sources(), [REC])

    def test_a_recordings_row_without_times_is_timed_as_the_table_times_it(self):
        from sdr_module.gui import decoder_views as dv

        p = self.panel
        p._proto_combo.setCurrentText("ADS-B")
        self.recorded(_klm(), source=REC)
        stamp = p._messages[-1]["time"].split(".")[0]
        self.assertEqual(self.cells()[dv.AC_SEEN], stamp)


# --------------------------------------------------------------------------- #
# In the window: File > Decode Recording
# --------------------------------------------------------------------------- #


def _set_start(path, when="2026-10-30T07:00:00Z"):
    meta = path[: -len(".sigmf-data")] + ".sigmf-meta"
    with open(meta) as fh:
        data = json.load(fh)
    data["captures"][0]["core:datetime"] = when
    with open(meta, "w") as fh:
        json.dump(data, fh)


class TestDecodeRecordingInTheViews(_FilesCase):
    def _adsb_file(self, rate=2.4e6):
        """A quarter of a second of the demo airliner's replies (KLM1023),
        recorded at 07:00:00 UTC."""
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file
        from sdr_module.gui import device_dialog as dd

        n = int(rate * 0.25)
        rng = np.random.default_rng(3)
        x = 0.01 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        for k, message in enumerate(("ident", "even", "velocity", "odd", "squawk")):
            t = 0.02 + 0.04 * k
            pulses = dd._modes_pulses(dd._aircraft_frame(message, t))
            first, env = dd._pulse_envelope(pulses, t * rate, rate)
            x[first : first + len(env)] += 0.5 * env
        path = self.path(REC)
        save_iq_file(path, x.astype(np.complex64), rate, 1090e6,
                     SampleFormat.FLOAT32, FileFormat.SIGMF)  # fmt: skip
        _set_start(path)
        return path

    def _cw_file(self, rate=16e3, sigmf=True):
        """ "CQ DE N0CALL", a pause and "TU 73" at 20 WPM: as SigMF recorded
        at 07:00:00 UTC, or as a headerless cf32 file (no start time)."""
        from sdr_module.dsp import protocol_encoders as enc
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

        first = enc.morse_keying("CQ DE N0CALL", rate, 20, lead_s=0.3, tail_s=3.5)
        second = enc.morse_keying("TU 73", rate, 20, lead_s=0.0, tail_s=0.5)
        rng = np.random.default_rng(7)
        key = np.concatenate([first, second])
        x = 0.5 * key + 0.002 * (
            rng.standard_normal(len(key)) + 1j * rng.standard_normal(len(key))
        )
        if not sigmf:
            path = self.path("cw_7.03MHz_16ksps.cf32")
            x.astype(np.complex64).tofile(path)
            return path
        path = self.path("cw.sigmf-data")
        save_iq_file(path, x.astype(np.complex64), rate, 7.030e6,
                     SampleFormat.FLOAT32, FileFormat.SIGMF)  # fmt: skip
        _set_start(path)
        return path

    def test_its_aircraft_show_in_the_aircraft_view(self):
        from sdr_module.gui import decoder_views as dv

        path = self._adsb_file()
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("POCSAG")  # (live: another protocol)
        self.settle(2)
        self.answer()
        answer = self.win._exec_dialog

        def adsb(dialog):
            # (The dialog offers the panel's protocol: ADS-B chosen there.)
            from sdr_module.gui import iq_files

            if isinstance(dialog, iq_files.DecodeOptionsDialog):
                dialog._protocol.setCurrentText("ADS-B")
            return answer(dialog)

        self.win._exec_dialog = adsb
        with self.open_dialog(path):
            self.assertTrue(self.win._decode_recording())
        self.settle(2)
        view = panel._aircraft_view
        self.assertIs(panel._tabs.currentWidget(), view)
        self.assertEqual(view.shown_source(), REC)
        aircraft = view.tracker_for(REC).get("4840D6")
        self.assertIsNotNone(aircraft)
        self.assertEqual(aircraft.callsign, "KLM1023")
        self.assertEqual(aircraft.altitude, 38000)
        self.assertEqual(aircraft.squawk, "3472")
        self.assertIsNotNone(aircraft.latitude)
        row = [view.table.item(0, c).text() for c in range(view.table.columnCount())]
        self.assertEqual(row[dv.AC_NAME], "KLM1023")
        # Seen: when the recording heard it last (07:00:00 UTC + 0.2 s).
        start = datetime(2026, 10, 30, 7, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(row[dv.AC_SEEN], start.astimezone().strftime("%H:%M:%S"))
        self.assertEqual(len(view.tracker), 0)  # (nothing heard live)

    def test_its_morse_text_shows_in_the_text_view(self):
        path = self._cw_file()
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("CW (Morse)")
        self.settle(2)
        self.answer()
        with self.open_dialog(path):
            self.assertTrue(self.win._decode_recording())
        self.settle(2)
        view = panel._cw_view
        self.assertIs(panel._tabs.currentWidget(), view)
        self.assertEqual(view.shown_source(), "cw.sigmf-data")
        lines = view.plain_text("cw.sigmf-data").split("\n")
        self.assertEqual([line.split("  ", 1)[1] for line in lines],
                         ["CQ DE N0CALL", "TU 73"])  # fmt: skip
        start = datetime(2026, 10, 30, 7, 0, 0, tzinfo=timezone.utc).astimezone()
        for line in lines:
            stamp = datetime.strptime(line.split("  ", 1)[0], "%H:%M:%S")
            self.assertEqual((stamp.hour, stamp.minute), (start.hour, start.minute))
        self.assertNotIn("CQ DE N0CALL", view.plain_text())  # (not live)

    def test_without_a_start_time_its_lines_are_timed_into_it(self):
        path = self._cw_file(sigmf=False)
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("CW (Morse)")
        self.settle(2)
        self.answer()
        with self.open_dialog(path):
            self.assertTrue(self.win._decode_recording())
        self.settle(2)
        view = panel._cw_view
        name = os.path.basename(path)
        self.assertEqual(view.shown_source(), name)
        lines = view.plain_text(name).split("\n")
        self.assertEqual([line.split("  ", 1)[1] for line in lines],
                         ["CQ DE N0CALL", "TU 73"])  # fmt: skip
        # Each line when the decoder ended it (2.5 s of quiet after its last
        # word, or the end of the file), as the time into the file.
        length = os.path.getsize(path) / 8 / 16e3
        seconds = []
        for line in lines:
            minutes, secs = line.split("  ", 1)[0].split(":")
            seconds.append(60 * int(minutes) + int(secs))
        self.assertLess(seconds[0], seconds[1])
        self.assertLessEqual(seconds[1], length)
        self.assertGreater(seconds[0], 2)


if __name__ == "__main__":
    unittest.main()

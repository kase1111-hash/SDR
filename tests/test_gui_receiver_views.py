"""The Second Receiver's decoded messages in the Decoder tab's views.

The Second Receiver window (lane rx2) sends what it decodes to the main
window's Decoder tab, marked "2nd RX" (``DecoderPanel.add_message(...,
receiver="2nd RX")``). Until now its rows went to the Messages table and
Log only: its aircraft and Morse text stayed out of the Aircraft and Text
views ("the views are the main receiver's"), so an aircraft it heard was
listed nowhere.

What a radio user wants (post-merge follow-up):

- Aircraft: both receivers hear the same sky, so the Second Receiver's
  aircraft join the live Aircraft list, one row per aircraft whichever
  receiver heard it. Its line under the table (and its row's tooltip) says
  which receiver heard it, and when ("last heard just now (main receiver
  just now, 2nd RX 4 s ago)"); one only the Second Receiver heard says so
  ("... by the 2nd RX"). Its age, dimming and removal go by the latest of
  either.
- Morse: two receivers on two CW frequencies send two texts, which mixed
  line by line would make no sense; the Second Receiver's text is an entry
  of its own in the Text view's "Showing:" list ("2nd RX"), apart from the
  main receiver's (Live). It is shown at once when the main receiver has no
  text, else picked there.
- A view holding the Second Receiver's aircraft or text stays as a tab
  whatever the main receiver decodes (as a recording's does), until Clear.
- Unchanged: the main receiver's lists, words and tabs when the Second
  Receiver sends nothing, Decode Recording's per-recording lists, the
  Messages table's rows, Clear, Export and the spool.
"""

import csv
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6  # noqa: E402
from tests.test_gui_decoder_views import _adsb, _PanelCase, _settle  # noqa: E402

RX2 = "2nd RX"
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


def _klm(**fields):
    base = dict(
        icao_address="4840D6",
        downlink_format=17,
        type_code=11,
        altitude=38000,
        latitude=52.2568,
        longitude=3.913,
    )
    base.update(fields)
    return _adsb(**base)


class _Case(_PanelCase):
    def add(self, msg, receiver="", **extra):
        """As the main window's _push_decoded_message adds a message."""
        from sdr_module.dsp.live_decoders import describe_message, message_details

        address, content = describe_message(msg)
        name = "CW (Morse)" if type(msg).__name__ == "MorseMessage" else "ADS-B"
        self.panel.add_message(
            name,
            address,
            content,
            bool(getattr(msg, "valid", True)),
            message=msg,
            details=message_details(msg),
            receiver=receiver,
            **extra,
        )

    def visible_tabs(self):
        return self.tab_titles()

    def current_tab(self):
        tabs = self.panel._tabs
        return tabs.tabText(tabs.currentIndex())


# --------------------------------------------------------------------------- #
# Aircraft: one live list, whichever receiver heard them
# --------------------------------------------------------------------------- #


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestItsAircraft(_Case):
    def setUp(self):
        super().setUp()
        self.panel._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        self.view = self.panel._aircraft_view

    def test_they_join_the_live_list_one_row_per_aircraft(self):
        from sdr_module.gui import decoder_views as dv

        p, view = self.panel, self.view
        self.add(_klm(callsign="KLM1023", type_code=4, altitude=0))
        self.now += 4.0
        self.add(_klm(altitude=37000), receiver=RX2)
        # One aircraft, whichever receiver heard it: the latest of each.
        self.assertEqual(len(view.tracker), 1)
        aircraft = view.tracker.get("4840D6")
        self.assertEqual(aircraft.messages, 2)
        self.assertEqual(aircraft.callsign, "KLM1023")
        self.assertEqual(aircraft.altitude, 37000)
        self.assertEqual(view.table.rowCount(), 1)
        self.assertEqual(view.table.item(0, dv.AC_NAME).text(), "KLM1023")
        self.assertEqual(view.shown_source(), "")  # (the live list)
        self.assertFalse(view.source_bar().isVisible())  # (no list of its own)
        # Its line says which receiver heard it, and when.
        self.now += 4.0
        view.refresh()  # (as its timer does each second)
        view.select("4840D6")
        _settle(1)
        detail = view.detail.text()
        self.assertIn(
            "last heard 4 s ago (main receiver 8 s ago, 2nd RX 4 s ago)", detail
        )
        self.assertIn(detail, view.table.item(0, dv.AC_NAME).toolTip())
        # Its rows in the Messages table are as before (marked 2nd RX).
        self.assertEqual(
            [m["protocol"] for m in p._messages], ["ADS-B", "2nd RX · ADS-B"]
        )
        self.assertEqual(p.count_for("ADS-B"), 1)

    def test_one_only_the_second_receiver_heard_says_so(self):
        view = self.view
        self.add(_klm(), receiver=RX2)
        self.now += 3.0
        self.assertEqual(view.table.rowCount(), 1)
        self.assertEqual(
            view.detail_text("4840D6").rsplit("  ·  ", 1)[-1],
            "last heard 3 s ago by the 2nd RX",
        )

    def test_heard_again_by_the_main_receiver(self):
        view = self.view
        self.add(_klm(), receiver=RX2)
        self.now += 2.0
        self.add(_klm())
        self.assertEqual(
            view.detail_text("4840D6").rsplit("  ·  ", 1)[-1],
            "last heard just now (main receiver just now, 2nd RX 2 s ago)",
        )

    def test_the_main_receivers_aircraft_read_as_before(self):
        # (Nothing from the Second Receiver: the words are unchanged, even
        # beside an aircraft only it heard.)
        from sdr_module.gui import decoder_views as dv

        view = self.view
        self.add(_klm())
        self.now += 3.0
        self.assertEqual(
            view.detail_text("4840D6").rsplit("  ·  ", 1)[-1], "last heard 3 s ago"
        )
        self.add(_klm(icao_address="3C6586"), receiver=RX2)
        self.assertEqual(
            view.detail_text("4840D6").rsplit("  ·  ", 1)[-1], "last heard 3 s ago"
        )
        aircraft = view.tracker.get("4840D6")
        self.assertEqual(
            dv.describe_aircraft(aircraft, self.now).rsplit("  ·  ", 1)[-1],
            "last heard 3 s ago",
        )

    def test_they_age_by_the_latest_of_either(self):
        from sdr_module.gui import decoder_views as dv

        view = self.view
        self.add(_klm())
        self.now += 50.0
        self.add(_klm(), receiver=RX2)
        self.now += 50.0  # (100 s after the main receiver, 50 s after it)
        view.refresh()
        tip = view.table.item(0, dv.AC_NAME).toolTip()
        self.assertNotIn("Not heard for over a minute", tip)
        self.assertIn("(main receiver 2 min ago, 2nd RX 50 s ago)", tip)
        self.now += 260.0  # (over five minutes after either)
        view.refresh()
        self.assertEqual(len(view.tracker), 0)
        self.assertEqual(view.table.rowCount(), 0)

    def test_the_address_filter_shows_both_receivers_rows(self):
        p, view = self.panel, self.view
        self.add(_klm())
        self.add(_klm(), receiver=RX2)
        self.add(_klm(icao_address="3C6586"), receiver=RX2)
        view.select("4840D6")
        _settle(1)
        self.assertEqual(p.address_filter(), "4840D6")
        shown = [
            p._messages[row]["protocol"]
            for row in range(p._table.rowCount())
            if not p._table.isRowHidden(row)
        ]
        self.assertEqual(shown, ["ADS-B", "2nd RX · ADS-B"])

    def test_a_recordings_list_keeps_to_the_recording(self):
        view = self.view
        self.add(_klm(altitude=12000), source=REC, when=WHEN, at_s=1.0)
        self.add(_klm(icao_address="3C6586"), receiver=RX2)
        recorded = view.tracker_for(REC)
        self.assertEqual([a.icao for a in recorded.aircraft()], ["4840D6"])
        self.assertEqual(recorded.get("4840D6").altitude, 12000)
        self.assertEqual([a.icao for a in view.tracker.aircraft()], ["3C6586"])
        self.assertEqual(view.shown_source(), REC)  # (still the recording)
        self.assertEqual(view.table.rowCount(), 1)
        # A recording's aircraft say when the recording heard them, as before.
        self.assertIn(f"in {REC}", view.detail_text("4840D6"))
        self.assertNotIn(RX2, view.detail_text("4840D6"))


# --------------------------------------------------------------------------- #
# The Aircraft tab while the main receiver decodes something else
# --------------------------------------------------------------------------- #


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestTheTabs(_Case):
    def test_its_aircraft_bring_the_aircraft_tab_not_to_the_front(self):
        p = self.panel
        p._proto_combo.setCurrentText("POCSAG")
        _settle(1)
        self.assertNotIn("Aircraft", self.visible_tabs())
        self.assertEqual(self.current_tab(), "Messages")
        self.add(_klm(), receiver=RX2)
        _settle(1)
        self.assertIn("Aircraft", self.visible_tabs())
        self.assertEqual(self.current_tab(), "Messages")  # (not taken over)
        self.assertEqual(p._aircraft_view.table.rowCount(), 1)
        # Another protocol chosen: it stays (it holds the 2nd RX's aircraft).
        p._proto_combo.setCurrentText("AX.25/APRS")
        _settle(1)
        self.assertIn("Aircraft", self.visible_tabs())
        # Clear: the tabs follow the protocol again.
        p.clear()
        _settle(1)
        self.assertNotIn("Aircraft", self.visible_tabs())

    def test_its_text_brings_the_text_tab_not_to_the_front(self):
        p = self.panel
        p._proto_combo.setCurrentText("POCSAG")
        _settle(1)
        self.add(_morse("CQ DE SECOND"), receiver=RX2)
        _settle(1)
        self.assertIn("Text", self.visible_tabs())
        self.assertEqual(self.current_tab(), "Messages")
        p.clear()
        _settle(1)
        self.assertNotIn("Text", self.visible_tabs())

    def test_the_filter_stays_while_the_view_holds_its_aircraft(self):
        p = self.panel
        p._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        self.add(_klm(), receiver=RX2)
        p._aircraft_view.select("4840D6")
        _settle(1)
        self.assertEqual(p.address_filter(), "4840D6")
        p._proto_combo.setCurrentText("POCSAG")
        _settle(1)
        self.assertEqual(p.address_filter(), "4840D6")


# --------------------------------------------------------------------------- #
# Morse text: an entry of its own under Showing:
# --------------------------------------------------------------------------- #


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestItsText(_Case):
    def setUp(self):
        super().setUp()
        self.view = self.panel._cw_view

    def test_its_text_is_an_entry_of_its_own(self):
        from sdr_module.gui.decoder_views import receiver_source

        p, view = self.panel, self.view
        p._proto_combo.setCurrentText("CW (Morse)")
        _settle(1)
        self.add(_morse("CQ DE MAIN", 20.0))
        self.add(_morse("CQ DE SECOND", 14.0), receiver=RX2)
        # The main receiver's text is left alone, and still shown.
        self.assertEqual(view.plain_text(), "CQ DE MAIN")
        self.assertEqual(view.shown_source(), "")
        bar = view.source_bar()
        self.assertTrue(bar.isVisible())
        self.assertEqual(bar.texts(), ["Live", RX2])
        self.assertEqual(view.other_receivers(), [RX2])
        self.assertEqual(view.sources(), [])  # (no recording)
        # Picked there: its own text and speed.
        bar.choose(receiver_source(RX2))
        _settle(1)
        self.assertEqual(view.shown_source(), receiver_source(RX2))
        self.assertEqual(view.shown_text(), "CQ DE SECOND")
        self.assertEqual(view.speed.text(), "14 WPM")
        self.add(_morse("TU 73", 15.0), receiver=RX2)
        self.assertEqual(view.shown_text(), "CQ DE SECOND\nTU 73")
        self.assertEqual(view.speed.text(), "15 WPM")
        # The main receiver's line meanwhile goes to its own (Live) text.
        self.add(_morse("QRZ"))
        self.assertEqual(view.shown_text(), "CQ DE SECOND\nTU 73")
        self.assertEqual(view.plain_text(), "CQ DE MAIN\nQRZ")
        self.assertEqual(view.plain_text(receiver_source(RX2)), "CQ DE SECOND\nTU 73")
        bar.choose("")
        _settle(1)
        self.assertEqual(view.shown_text(), "CQ DE MAIN\nQRZ")

    def test_shown_at_once_when_the_main_receiver_has_no_text(self):
        from sdr_module.gui.decoder_views import receiver_source

        view = self.view
        self.add(_morse("CQ DE SECOND"), receiver=RX2)
        self.assertEqual(view.shown_source(), receiver_source(RX2))
        self.assertEqual(view.shown_text(), "CQ DE SECOND")
        self.assertEqual(view.source_bar().current_text(), RX2)

    def test_beside_a_recordings_text(self):
        from sdr_module.gui.decoder_views import receiver_source

        view = self.view
        self.add(_morse("CQ DE SECOND"), receiver=RX2)
        self.add(_morse("CQ DE REC"), source="cw.wav", at_s=75.0)
        # The recording asked for is shown (as before); both are listed.
        self.assertEqual(view.shown_source(), "cw.wav")
        self.assertEqual(view.source_bar().texts(), ["Live", RX2, "cw.wav"])
        self.assertEqual(view.shown_text(), "1:15  CQ DE REC")
        self.add(_morse("TU"), receiver=RX2)
        self.assertEqual(view.shown_text(), "1:15  CQ DE REC")
        self.assertEqual(view.plain_text(receiver_source(RX2)), "CQ DE SECOND\nTU")
        # Decoded again: the recording's text starts over, not the 2nd RX's.
        self.panel.recording_decode_started("cw.wav", "CW (Morse)")
        self.assertEqual(view.sources(), [])
        self.assertEqual(view.other_receivers(), [RX2])

    def test_a_recording_named_like_it_is_another_entry(self):
        from sdr_module.gui.decoder_views import receiver_source

        view = self.view
        self.add(_morse("FROM THE RECEIVER"), receiver=RX2)
        self.add(_morse("FROM THE FILE"), source=RX2, at_s=1.0)
        self.assertEqual(view.plain_text(receiver_source(RX2)), "FROM THE RECEIVER")
        self.assertEqual(view.plain_text(RX2), "0:01  FROM THE FILE")

    def test_clear_empties_it(self):
        p, view = self.panel, self.view
        self.add(_morse("CQ DE SECOND"), receiver=RX2)
        p.clear()
        _settle(1)
        self.assertEqual(view.other_receivers(), [])
        self.assertEqual(view.shown_source(), "")
        self.assertFalse(view.source_bar().isVisible())
        self.assertEqual(view.shown_text(), "")


# --------------------------------------------------------------------------- #
# Unchanged without it; Clear, Export, the spool
# --------------------------------------------------------------------------- #


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestUnchangedWithoutIt(_Case):
    def test_the_main_receivers_views_look_as_they_did(self):
        from sdr_module.gui import decoder_views as dv

        p = self.panel
        p._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        view = p._aircraft_view
        self.add(_klm(callsign="KLM1023", type_code=4, altitude=0))
        self.now += 3.0
        view.refresh()
        tip = view.table.item(0, dv.AC_NAME).toolTip()
        self.assertEqual(
            tip, dv.describe_aircraft(view.tracker.get("4840D6"), self.now)
        )
        self.assertTrue(tip.endswith("last heard 3 s ago"))
        self.assertNotIn("receiver", tip)
        self.assertFalse(view.has_other_receivers())
        p._proto_combo.setCurrentText("CW (Morse)")
        _settle(1)
        self.add(_morse("CQ"))
        cw = p._cw_view
        self.assertFalse(cw.source_bar().isVisible())
        self.assertNotIn(RX2, cw.source_bar().texts())
        self.assertFalse(cw.has_other_receivers())
        self.assertNotIn("Aircraft", self.visible_tabs())

    def test_clear_export_and_the_spool(self):
        p = self.panel
        p._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        self.add(_klm())
        self.add(_klm(icao_address="3C6586"), receiver=RX2)
        self.add(_morse("CQ DE SECOND"), receiver=RX2)
        said = []
        box = mock.patch.object(
            self.mod.QMessageBox,
            "information",
            side_effect=lambda *a, **k: said.append(a[2]),
        )
        box.start()
        self.addCleanup(box.stop)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "decoded.csv")
            self.assertEqual(p._export_messages(path), 3)
            self.assertIn("Exported 3 messages", said[-1])
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        protocols = [row[rows[0].index("Protocol")] for row in rows[1:]]
        self.assertEqual(protocols, ["ADS-B", "2nd RX · ADS-B", "2nd RX · CW (Morse)"])
        p.clear()
        _settle(1)
        self.assertEqual(len(p._aircraft_view.tracker), 0)
        self.assertFalse(p._aircraft_view.has_other_receivers())
        self.assertEqual(p._cw_view.other_receivers(), [])
        self.assertEqual(p._spool.count, 0)
        self.add(_klm(), receiver=RX2)
        self.assertEqual(len(p._aircraft_view.tracker), 1)


# --------------------------------------------------------------------------- #
# In the window: the Second Receiver decoding the demo's ADS-B
# --------------------------------------------------------------------------- #

if HAS_PYQT6:
    from tests.test_gui_second_receiver_decoders import _DecodeCase
else:  # pragma: no cover - environment
    _DecodeCase = unittest.TestCase


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestInTheWindow(_DecodeCase):
    def test_its_aircraft_show_in_the_main_windows_aircraft_view(self):
        rx, panel = self.rx, self.panel
        index = rx._rate_combo.findData(2.4e6)
        rx._rate_combo.setCurrentIndex(index)
        rx._rate_combo.activated.emit(index)
        panel._proto_combo.setCurrentText("POCSAG")  # (the main window's)
        self.settle()
        self.start_on(1090e6, ("None (I/Q)", "", "2 MHz"))
        self.choose("ADS-B")
        self.run_seconds(3.0)
        rows = [m for m in self.rows() if m["protocol"] == "2nd RX · ADS-B"]
        self.assertTrue(rows)
        view = panel._aircraft_view
        aircraft = view.tracker.get("4840D6")
        self.assertIsNotNone(aircraft)
        self.assertEqual(aircraft.callsign, "KLM1023")
        self.assertIn("by the 2nd RX", view.detail_text("4840D6"))
        tabs = panel._tabs
        self.assertTrue(tabs.isTabVisible(tabs.indexOf(view)))
        self.assertEqual(panel.count_for("ADS-B"), 0)  # (the main window's own)


if __name__ == "__main__":
    unittest.main()

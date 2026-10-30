"""The Decoder panel's Aircraft and Text views, Auto, and what the window
shows of the decoders: the unread count, the status-bar chip and the RDS
station badge.

fg:FG-G07 / tl:G-02 / ux:UX-G14 / ux:UX-U14: one ADS-B aircraft gave a row
per frame (37 in 12 s); now an Aircraft view has one row per aircraft.
fg:FG-G06: Morse code is read as text. tl:UX-09 / ux:UX-U12: decoding was
invisible from another tab. tl:UX-15: the CSV had no frequency. FG-G05 /
G-06 / UX-G04: RDS names the station next to FREQ and on the radio tuner.
fg:FG-G10 / mc:MC-G13: "Auto" picks the protocol from where it is tuned.
"""

import csv
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QAction
    from PyQt6.QtWidgets import QApplication


def _settle(n=5):
    for _ in range(n):
        QApplication.processEvents()


def _adsb(**fields):
    from sdr_module.dsp.protocols import ADSBMessage, ProtocolType

    base = dict(protocol=ProtocolType.ADSB, timestamp=0.0, raw_bits=b"", valid=True)
    base.update(fields)
    return ADSBMessage(**base)


def _add(panel, msg):
    from sdr_module.dsp.live_decoders import describe_message, message_details

    address, content = describe_message(msg)
    panel.add_message(
        "ADS-B", address, content, True, message=msg, details=message_details(msg)
    )


class _PanelCase(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import decoder_panel

        self.mod = decoder_panel
        self.now = 1000.0
        self.panel = decoder_panel.DecoderPanel()
        self.panel._clock = lambda: self.now
        self.panel.resize(420, 520)
        self.panel.show()
        _settle(2)

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        _settle(1)

    def tab_titles(self):
        tabs = self.panel._tabs
        return [tabs.tabText(i) for i in range(tabs.count()) if tabs.isTabVisible(i)]


# --------------------------------------------------------------------------- #
# Aircraft (ADS-B)
# --------------------------------------------------------------------------- #


class TestAircraftView(_PanelCase):
    def setUp(self):
        super().setUp()
        self.panel._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        self.view = self.panel._aircraft_view

    def feed_klm(self):
        _add(self.panel, _adsb(icao_address="4840D6", downlink_format=17, type_code=4,
                               callsign="KLM1023", category="Heavy"))  # fmt: skip
        _add(self.panel, _adsb(icao_address="4840D6", downlink_format=17, type_code=11,
                               altitude=38000, latitude=52.2568, longitude=3.913))  # fmt: skip
        _add(self.panel, _adsb(icao_address="4840D6", downlink_format=17, type_code=19,
                               velocity=450.0, heading=265.0, vertical_rate=-640))  # fmt: skip
        _add(self.panel, _adsb(icao_address="4840D6", downlink_format=5, squawk="3472"))

    def cells(self, row=0):
        table = self.view.table
        return [table.item(row, c).text() for c in range(table.columnCount())]

    def test_shown_and_in_front_only_for_ads_b(self):
        self.assertEqual(self.panel._tabs.currentWidget(), self.view)
        self.assertEqual(self.tab_titles()[0], "Aircraft")
        self.panel._proto_combo.setCurrentText("POCSAG")
        _settle(1)
        self.assertNotIn("Aircraft", self.tab_titles())
        self.assertIs(self.panel._tabs.currentWidget(), self.panel._messages_page)

    def test_one_row_per_aircraft_with_the_latest_of_everything(self):
        self.feed_klm()
        _add(self.panel, _adsb(icao_address="3C6586", downlink_format=17, type_code=4,
                               callsign="DLH9U"))  # fmt: skip
        self.assertEqual(self.view.table.rowCount(), 2)
        self.assertEqual(
            self.cells(0),
            ["KLM1023", "38,000", "450", "265°", "-640", "3472", "now", "4",
             "52.257, 3.913", "4840D6"],
        )  # fmt: skip
        self.assertEqual(self.cells(1)[0], "DLH9U")
        self.assertEqual(len(self.panel._messages), 5)  # the rows stay too

    def test_unknown_vertical_rate_is_a_dash(self):
        _add(self.panel, _adsb(icao_address="ABC123", downlink_format=17, type_code=19,
                               velocity=300.0, heading=359.7, vertical_rate=0,
                               vertical_rate_known=False))  # fmt: skip
        cells = self.cells(0)
        self.assertEqual((cells[3], cells[4]), ("0°", "–"))

    def test_rows_dim_after_a_minute_and_go_after_five(self):
        self.feed_klm()
        self.now += 61
        self.view.refresh()
        item = self.view.table.item(0, 0)
        from sdr_module.gui.themes import get_palette

        self.assertEqual(item.foreground().color(), get_palette().qcolor("disabled"))
        self.assertFalse(item.font().bold())
        self.assertEqual(self.cells(0)[6], "1 min")
        self.assertIn("Not heard for over a minute", item.toolTip())
        self.now += 240
        self.view.refresh()
        self.assertEqual(self.view.table.rowCount(), 0)
        self.assertTrue(self.view._empty.is_visible())

    def test_another_aircraft_going_keeps_the_selection(self):
        # An aircraft dropped while another is picked: the pick, its
        # filter and the message picked under Messages all stay.
        _add(self.panel, _adsb(icao_address="3C6586", downlink_format=11))
        self.now += 200
        self.feed_klm()
        self.view.select("4840D6")
        self.panel._table.selectRow(1)
        _settle(1)
        seen = []
        self.view.aircraft_selected.connect(seen.append)
        self.now += 150  # 3C6586 is gone, KLM1023 stays
        self.view.refresh()
        _settle(1)
        self.assertEqual(self.view.table.rowCount(), 1)
        self.assertEqual(self.view.selected_icao(), "4840D6")
        self.assertEqual(self.panel.address_filter(), "4840D6")
        self.assertEqual(seen, [])
        self.assertEqual(
            [i.row() for i in self.panel._table.selectionModel().selectedRows()], [1]
        )
        # The picked aircraft going clears the pick and the filter.
        self.now += 400
        self.view.refresh()
        _settle(1)
        self.assertEqual(seen, [""])
        self.assertEqual(self.panel.address_filter(), "")
        self.assertTrue(self.view.detail.isHidden())

    def test_picking_an_aircraft_shows_only_its_messages(self):
        self.feed_klm()
        _add(self.panel, _adsb(icao_address="3C6586", downlink_format=11))
        self.view.select("4840D6")
        _settle(1)
        self.assertEqual(self.panel.address_filter(), "4840D6")
        hidden = [self.panel._table.isRowHidden(r) for r in range(5)]
        self.assertEqual(hidden, [False, False, False, False, True])
        self.assertIn("Messages from KLM1023 (4840D6) only (4 here)",
                      self.panel._filter_label.text())  # fmt: skip
        self.assertIn("Messages*", self.tab_titles())
        self.assertIn("KLM1023 (4840D6)", self.view.detail.text())
        # New messages from others stay hidden while filtered.
        _add(self.panel, _adsb(icao_address="3C6586", downlink_format=11))
        self.assertTrue(self.panel._table.isRowHidden(5))
        # Show All brings them back and drops the selection.
        self.panel._filter_clear.click()
        _settle(1)
        self.assertEqual(self.panel.address_filter(), "")
        self.assertFalse(any(self.panel._table.isRowHidden(r) for r in range(6)))
        self.assertEqual(self.view.selected_icao(), "")
        self.assertIn("Messages", self.tab_titles())

    def test_double_click_goes_to_the_messages(self):
        self.feed_klm()
        self.view.aircraft_activated.emit("4840D6")
        _settle(1)
        self.assertIs(self.panel._tabs.currentWidget(), self.panel._messages_page)
        self.assertEqual(self.panel.address_filter(), "4840D6")

    def test_clear_empties_it(self):
        self.feed_klm()
        self.panel.clear()
        self.assertEqual(self.view.table.rowCount(), 0)
        self.assertEqual(self.view.count(), 0)

    def test_failed_messages_are_not_aircraft(self):
        msg = _adsb(icao_address="ABCDEF", downlink_format=17, valid=False)
        self.panel.add_message("ADS-B", "ABCDEF", "x", False, message=msg)
        self.assertEqual(self.view.table.rowCount(), 0)


# --------------------------------------------------------------------------- #
# Text (Morse)
# --------------------------------------------------------------------------- #


class TestMorseText(_PanelCase):
    def test_running_text_with_the_speed(self):
        from sdr_module.dsp.live_decoders import LiveDecoderStatus
        from sdr_module.dsp.morse import MorseMessage
        from sdr_module.dsp.protocols import ProtocolType

        p = self.panel
        p._proto_combo.setCurrentText("CW (Morse)")
        _settle(1)
        view = p._cw_view
        self.assertIs(p._tabs.currentWidget(), view)
        self.assertEqual(self.tab_titles()[0], "Text")
        p.set_receiving(True)
        status = LiveDecoderStatus(
            name="CW (Morse)", unit="character", valid=3, last_time=self.now,
            detail="18 WPM", live_text="CQ", signal_measured=True, signal_present=True,
        )  # fmt: skip
        p.set_decoder_status(status)
        self.assertEqual(view.plain_text(), "CQ")
        self.assertEqual(view.speed.text(), "18 WPM")
        self.assertEqual(view.state.text(), "Receiving")
        self.assertEqual(
            p.decoder_status_text(),
            "CW (Morse)  ·  18 WPM  ·  3 characters  ·  last just now",
        )
        msg = MorseMessage(protocol=ProtocolType.CW, timestamp=0.0, raw_bits=b"",
                           valid=True, text="CQ DE N0CALL", wpm=18.0)  # fmt: skip
        p.add_message("CW (Morse)", "18 WPM", "CQ DE N0CALL", message=msg)
        p.set_decoder_status(
            LiveDecoderStatus(name="CW (Morse)", detail="18 WPM", live_text="K")
        )
        self.assertEqual(view.plain_text(), "CQ DE N0CALL\nK")
        p.clear()
        self.assertEqual(view.plain_text(), "")
        self.assertEqual(view.speed.text(), "–")


# --------------------------------------------------------------------------- #
# Auto
# --------------------------------------------------------------------------- #


class TestAuto(_PanelCase):
    def test_choices(self):
        auto = self.mod.auto_protocol
        self.assertEqual(auto(1090e6), "ADS-B")
        self.assertEqual(auto(144.39e6), "AX.25/APRS")
        self.assertEqual(auto(144.8e6), "AX.25/APRS")
        self.assertEqual(auto(131.55e6), "ACARS")
        self.assertEqual(auto(152.24e6), "POCSAG")
        self.assertEqual(auto(931.9375e6), "FLEX")
        self.assertEqual(auto(100.1e6, "FM"), "RDS")
        self.assertEqual(auto(7.03e6, "CW"), "CW (Morse)")
        self.assertEqual(auto(99.82e6, "CW"), "CW (Morse)")  # the mode wins
        self.assertIsNone(auto(433.92e6))
        self.assertIsNone(auto(None))

    def test_follows_tuning_and_says_what_it_chose(self):
        p = self.panel
        emitted = []
        p.protocol_changed.connect(emitted.append)
        p.set_current_frequency(1090e6)
        p._proto_combo.setCurrentText("Auto")
        self.assertEqual((p.selection(), p.protocol()), ("Auto", "ADS-B"))
        self.assertEqual(emitted, ["ADS-B"])
        self.assertTrue(p._enabled_check.isChecked())
        p.set_receiving(True)
        self.assertTrue(p.decoder_status_text().startswith("Auto: "))
        self.assertIn("Auto chose ADS-B", p._proto_combo.toolTip())
        self.assertEqual(self.tab_titles()[0], "Aircraft")
        p.set_current_frequency(433.92e6)
        self.assertEqual(emitted, ["ADS-B", "Off"])
        self.assertIsNone(p.protocol())
        self.assertEqual(
            p.decoder_status_text(), "Auto: nothing to decode on 433.920 MHz"
        )
        self.assertIn("Nothing to decode on 433.920 MHz", p._empty.text())
        self.assertTrue(p._enabled_check.isEnabled())
        p.set_demod_mode("CW")
        self.assertEqual(emitted[-1], "CW (Morse)")
        # Tuning within the same protocol's band changes nothing.
        p.set_demod_mode("FM")
        p.set_current_frequency(152.24e6)
        p.set_current_frequency(152.30e6)
        self.assertEqual(emitted[-2:], ["Off", "POCSAG"])


# --------------------------------------------------------------------------- #
# Details, CSV and the tab bar
# --------------------------------------------------------------------------- #


class TestMessageContext(_PanelCase):
    def test_detail_card_has_the_frequency_and_the_details(self):
        p = self.panel
        p.set_current_frequency(1090e6)
        _add(p, _adsb(downlink_format=17, type_code=19, velocity=450.0, heading=265.0))
        p._table.selectRow(0)
        _settle(1)
        self.assertIn("1090.000 MHz", p._detail.meta.text())
        self.assertEqual(
            p._detail.details.text(),
            "DF17 extended squitter, type code 19 (airborne velocity)",
        )
        self.assertFalse(p._detail.details.isHidden())

    def test_csv_has_frequency_and_sample_rate_last(self):
        p = self.panel
        p.set_current_frequency(152.24e6)
        p.set_sample_rate(2.4e6)
        p.add_message("POCSAG", "1 (F0)", "first")
        p.add_message(
            "POCSAG", "2 (F0)", "second", freq_hz=152.3e6, sample_rate=1.024e6
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(self.mod.QMessageBox, "information"),
        ):
            path = os.path.join(tmp, "out.csv")
            self.assertEqual(p._export_messages(path), 2)
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(rows[0][-2:], ["Frequency (MHz)", "Sample rate (MS/s)"])
        self.assertEqual(
            rows[0][:5], ["Date", "Time", "Protocol", "Address", "Content"]
        )
        self.assertEqual(rows[1][-2:], ["152.240000", "2.4"])
        self.assertEqual(rows[2][-2:], ["152.300000", "1.024"])

    def test_status_line_shows_the_decoders_detail(self):
        from sdr_module.dsp.live_decoders import LiveDecoderStatus

        p = self.panel
        p._proto_combo.setCurrentText("RDS")
        p.set_current_frequency(100.1e6)
        p.set_receiving(True)
        p.set_decoder_status(LiveDecoderStatus(
            name="RDS", unit="group", frames=5, valid=5, last_time=self.now,
            detail="SDR DEMO", signal_measured=True,
        ))  # fmt: skip
        self.assertEqual(
            p.decoder_status_text(), "RDS  ·  SDR DEMO  ·  5 groups  ·  last just now"
        )
        p.set_decoder_status(LiveDecoderStatus(
            name="RDS", unit="group", signal_measured=True, signal_present=True,
            undecoded_signal_s=7.0,
        ))  # fmt: skip
        self.assertEqual(
            p.decoder_status_text(),
            "No RDS from this station: not every station sends it",
        )


# --------------------------------------------------------------------------- #
# The window: unread count, chip, RDS station
# --------------------------------------------------------------------------- #


class TestWindowActivity(_WindowTestCase):
    def test_unread_count_on_the_tab_while_hidden(self):
        win = self.win
        win._decoder_panel._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        win._show_panel("bookmarks")
        _settle(1)
        for _ in range(3):
            win._push_decoded_message(_adsb(icao_address="4840D6", downlink_format=11))
        owner = win._panel_owners["decoder"]
        index = owner.indexOf(win._panel_pages["decoder"])
        self.assertEqual(owner.tabText(index), "Decoder (3)")
        self.assertIn("3 new decoded messages", owner.tabToolTip(index))
        win._show_panel("decoder")
        _settle(1)
        self.assertEqual(owner.tabText(index), "Decoder")
        # Seen as they arrive: not counted.
        win._push_decoded_message(_adsb(icao_address="4840D6", downlink_format=11))
        self.assertEqual(owner.tabText(index), "Decoder")

    def test_status_bar_chip(self):
        win = self.win
        chip = win._decoder_chip
        self.assertTrue(chip.isHidden())
        win._decoder_panel._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        self.assertFalse(chip.isHidden())
        self.assertEqual(chip.text(), "ADS-B · 0 msgs")
        win._push_decoded_message(_adsb(icao_address="4840D6", downlink_format=11))
        win._run_decoder(np.zeros(8, dtype=np.complex64))
        self.assertEqual(chip.text(), "ADS-B · 1 msg")
        # Its tooltip names the shortcut View > Panels gives the Decoder.
        shortcut = next(
            a.shortcut().toString()
            for a in win.findChildren(QAction)
            if a.text().replace("&", "") == "Decoder" and not a.shortcut().isEmpty()
        )
        self.assertIn(f"Click to open the Decoder tab ({shortcut}).", chip.toolTip())
        win._decoder_panel._proto_combo.setCurrentText("Auto")
        _settle(1)
        self.assertIn("(chosen by Auto)", chip.toolTip())
        win._decoder_panel._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        win._decoder_panel._enabled_check.setChecked(False)
        win._update_decoder_chip()
        self.assertEqual(chip.text(), "ADS-B · paused")
        win._show_panel("info")
        chip.click()
        _settle(1)
        self.assertIs(win._right_tabs.currentWidget(), win._panel_pages["decoder"])
        win._decoder_panel._proto_combo.setCurrentText("Off")
        _settle(1)
        self.assertTrue(chip.isHidden())

    def test_messages_remember_frequency_and_rate(self):
        win = self.win
        win._decoder_panel._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        win._push_decoded_message(_adsb(icao_address="4840D6", downlink_format=11))
        msg = win._decoder_panel._messages[-1]
        self.assertEqual(msg["freq_hz"], win._current_frequency())
        self.assertEqual(msg["sample_rate"], win._device_sample_rate())
        self.assertEqual(msg["details"], ["DF11 all-call reply"])

    def test_auto_builds_the_decoder_for_where_it_is_tuned(self):
        from sdr_module.dsp.live_decoders import ADSBLiveDecoder, RDSLiveDecoder

        win = self.win
        win.set_frequency(100.1e6)
        win._decoder_panel._proto_combo.setCurrentText("Auto")
        _settle(1)
        self.assertIsInstance(win._decoder, RDSLiveDecoder)
        win.set_frequency(1090e6)
        _settle(1)
        self.assertIsInstance(win._decoder, ADSBLiveDecoder)
        win.set_frequency(433.92e6)
        _settle(1)
        self.assertIsNone(win._decoder)
        # A sample rate change rebuilds Auto's choice, not "Auto" itself.
        win.set_frequency(1090e6)
        _settle(1)
        win._rebuild_decoder()
        self.assertIsInstance(win._decoder, ADSBLiveDecoder)


class TestThemedTabFit(_WindowTestCase):
    """Measured the way the app shows it: themed."""

    @classmethod
    def setUpClass(cls):
        if not HAS_PYQT6:
            return
        from sdr_module.gui import themes

        qapp = QApplication.instance() or QApplication([])
        cls._old_theme = themes.current_theme()
        cls._old_sheet, cls._old_palette = qapp.styleSheet(), qapp.palette()
        themes.apply_theme(qapp, "dark")

    @classmethod
    def tearDownClass(cls):
        if not HAS_PYQT6:
            return
        from sdr_module.gui import themes

        qapp = QApplication.instance()
        themes.apply_theme(None, cls._old_theme)
        qapp.setStyleSheet(cls._old_sheet)
        qapp.setPalette(cls._old_palette)

    def test_every_decoder_tab_title_shows_at_the_minimum_size(self):
        # The Aircraft (or Text) tab makes four: in a narrow panel Clear and
        # Export turn compact so none hides behind scroll arrows.
        win = self.win
        win.resize(1024, 640)
        win._show_panel("decoder")
        panel = win._decoder_panel
        for proto in ("ADS-B", "CW (Morse)"):
            panel._proto_combo.setCurrentText(proto)
            self.settle(10)
            bar = panel._tabs.tabBar()
            visible = [i for i in range(bar.count()) if panel._tabs.isTabVisible(i)]
            self.assertEqual(len(visible), 4, proto)
            corner = panel._tabs.cornerWidget(Qt.Corner.TopRightCorner)
            right = max(bar.tabRect(i).right() for i in visible)
            self.assertLessEqual(right, bar.width(), proto)
            self.assertLessEqual(
                bar.mapTo(panel, bar.rect().topLeft()).x() + right,
                corner.mapTo(panel, corner.rect().topLeft()).x(),
                proto,
            )
        win.resize(1600, 900)
        self.settle(10)
        self.assertEqual(panel._clear_btn.property("role") or "", "")


class TestRdsStation(_WindowTestCase):
    """The demo's 100.1 MHz station names itself next to FREQ."""

    demo = True

    def receive(self, seconds):
        dev = self.win._device
        if not dev.is_streaming:
            dev.start_rx()
        block = int(self.win._device_sample_rate() / 30)
        for _ in range(int(seconds * 30)):
            self.win._process_block(dev.read_samples(block))

    def listen_to_the_station(self):
        win = self.win
        win.set_frequency(100.1e6)
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        win._device._stream_pos = 0
        self.receive(3.0)

    def test_badge_and_tuner(self):
        win = self.win
        self.listen_to_the_station()
        self.assertEqual(win._rds_badge.text(), "SDR DEMO")
        self.assertTrue(win._rds_badge_action.isVisible())
        tip = win._rds_badge.toolTip()
        self.assertIn("RDS: SDR DEMO (PI C0DE)", tip)
        self.assertIn("Program type: Top 40", tip)
        self.assertIn("Now playing: the SDR Module demo tune", tip)
        win._show_radio_tuner()
        _settle(2)
        self.assertEqual(
            win._radio_tuner._freq_display._station,
            "SDR DEMO · Now playing: the SDR Module demo tune (I-vi-IV-V in C)",
        )
        # Another station: the name goes at once (it has to be heard anew);
        # the tuner still names the preset it matches.
        win.set_frequency(99.5e6)
        self.assertFalse(win._rds_badge_action.isVisible())
        self.assertNotIn("SDR DEMO", win._radio_tuner._freq_display._station)
        win._radio_tuner.close()

    def test_only_while_listening_to_broadcast_fm(self):
        win = self.win
        self.listen_to_the_station()
        self.assertTrue(win._rds_badge_action.isVisible())
        win._set_demod_mode("FM", "5 kHz", "25 kHz")  # narrow FM
        self.receive(0.2)
        self.assertFalse(win._rds_badge_action.isVisible())
        self.assertIsNone(win._rds_listener.decoder)

    def test_the_setting_turns_it_off(self):
        win = self.win
        self.assertTrue(win._rds_auto_action.isChecked())  # on by default
        self.listen_to_the_station()
        win._rds_auto_action.setChecked(False)
        self.assertFalse(win._rds_badge_action.isVisible())
        self.assertFalse(self.store.get_bool("decoder/rds_auto", True))
        self.receive(0.3)
        self.assertFalse(win._rds_badge_action.isVisible())

    def test_disconnecting_forgets_the_station(self):
        win = self.win
        self.listen_to_the_station()
        self.assertTrue(win._rds_badge_action.isVisible())
        win._disconnect_device()
        self.assertFalse(win._rds_badge_action.isVisible())
        self.assertIsNone(win._rds_listener.decoder)

    def test_the_decoders_rds_names_it_instead(self):
        win = self.win
        win._decoder_panel._proto_combo.setCurrentText("RDS")
        _settle(1)
        self.listen_to_the_station()
        self.assertEqual(win._rds_badge.text(), "SDR DEMO")
        self.assertIsNone(win._rds_listener.decoder)  # not decoded twice
        rows = [m["content"] for m in win._decoder_panel._messages]
        self.assertEqual(rows[0], "Station name: SDR DEMO")

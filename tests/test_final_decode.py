"""Decoder fixes from the end-to-end journeys (DEC-E2E-*) and the review
leftovers (L27-L29): what users of the finished app ran into in the
Decoder tab, the status-bar chip, File > Decode Recording and the decoders
themselves.
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtWidgets import QApplication


def _settle(n=5):
    for _ in range(n):
        QApplication.processEvents()


def _status(name, unit="message", **fields):
    """A live decoder's status, as the window passes it to the panel."""
    from sdr_module.dsp.live_decoders import LiveDecoderStatus

    base = dict(name=name, unit=unit, signal_measured=True)
    base.update(fields)
    return LiveDecoderStatus(**base)


class _PanelCase(unittest.TestCase):
    """A Decoder panel on its own, shown, with a clock the test sets."""

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

    def front(self):
        return self.panel._tabs.currentWidget()


# --------------------------------------------------------------------------- #
# DEC-E2E-02: a protocol's own view comes to the front
# --------------------------------------------------------------------------- #


class TestProtocolViewInFront(_PanelCase):
    def test_straight_from_ads_b_to_cw_and_back(self):
        # The fallback to Messages looked at the tab that was in front
        # before the switch (now hidden) and covered the view just shown.
        p = self.panel
        p._proto_combo.setCurrentText("ADS-B")
        self.assertIs(self.front(), p._aircraft_view)
        p._proto_combo.setCurrentText("CW (Morse)")
        self.assertIs(self.front(), p._cw_view)
        p._proto_combo.setCurrentText("ADS-B")
        self.assertIs(self.front(), p._aircraft_view)
        # Through another protocol it was right already, and stays so.
        p._proto_combo.setCurrentText("POCSAG")
        self.assertIs(self.front(), p._messages_page)
        p._proto_combo.setCurrentText("CW (Morse)")
        self.assertIs(self.front(), p._cw_view)

    def test_auto_follows_the_mode_to_the_text_view(self):
        p = self.panel
        p.set_current_frequency(1090e6)
        p._proto_combo.setCurrentText("Auto")
        self.assertIs(self.front(), p._aircraft_view)
        p.set_demod_mode("CW")  # Auto: Morse code in CW
        self.assertEqual(p.protocol(), "CW (Morse)")
        self.assertIs(self.front(), p._cw_view)

    def test_a_tab_picked_by_hand_stays_while_the_protocol_does(self):
        p = self.panel
        p._proto_combo.setCurrentText("ADS-B")
        p._tabs.setCurrentWidget(p._raw_output)
        p.set_current_frequency(1090.2e6)  # (nothing changes the protocol)
        self.assertIs(self.front(), p._raw_output)


# --------------------------------------------------------------------------- #
# DEC-E2E-01: "Tune to ..." stays offered once messages are listed
# --------------------------------------------------------------------------- #


class TestTuneToStaysOffered(_PanelCase):
    def setUp(self):
        super().setUp()
        self.tuned = []
        self.panel.tune_requested.connect(lambda *args: self.tuned.append(args))
        self.panel.set_receiving(True)

    def test_offered_beside_the_status_line_with_rows_in_the_table(self):
        # The only button was the empty table's: one row (of any protocol)
        # hid it, and only Clear (losing the messages) brought it back.
        p = self.panel
        p.set_current_frequency(144.39e6)
        p._proto_combo.setCurrentText("AX.25/APRS")
        p.add_message("AX.25/APRS", "N0CALL-9", "49.0583, -72.0292")
        p._proto_combo.setCurrentText("POCSAG")
        p.set_decoder_status(_status("POCSAG", "page"))  # (as the window does)
        _settle(1)
        self.assertFalse(p._empty.is_visible())  # the table lists the packet
        self.assertTrue(
            p.decoder_status_text().startswith("Not tuned to POCSAG"),
            p.decoder_status_text(),
        )
        button = p._tune_btn
        self.assertTrue(button.isVisible())
        self.assertEqual(button.text(), "Tune to 152.240 MHz")
        self.assertIn("pager channel", button.toolTip())
        button.click()
        self.assertEqual(self.tuned[-1][:3], (152.24e6, "POCSAG (152.240 MHz)", "FM"))
        p.set_current_frequency(152.24e6)  # (the window reports the tune)
        self.assertFalse(button.isVisible())

    def test_one_button_while_the_empty_view_in_front_has_its_own(self):
        p = self.panel
        p.set_current_frequency(100.1e6)
        p._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        aircraft = p._aircraft_view.placeholder()
        self.assertTrue(aircraft.button.isVisible())
        self.assertFalse(p._tune_btn.isVisible())
        p._tabs.setCurrentWidget(p._raw_output)  # the Log: no button there
        self.assertTrue(p._tune_btn.isVisible())
        p._tabs.setCurrentWidget(p._messages_page)  # the empty table has one
        self.assertTrue(p._empty.button.isVisible())
        self.assertFalse(p._tune_btn.isVisible())
        p.add_message("ADS-B", "4840D6", "KLM1023")
        self.assertTrue(p._tune_btn.isVisible())

    def test_offered_when_live_morse_fills_the_text_view(self):
        # The Text view's own "Listen in CW" went with its empty state when
        # the line in progress came (CW decodes in any mode), and the one
        # beside the status line stayed hidden.
        p = self.panel
        p.set_source("demo")
        p.set_current_frequency(99.82e6)
        p.set_demod_mode("FM")  # (the band plan's mode for 99.82 MHz)
        p._proto_combo.setCurrentText("CW (Morse)")
        _settle(1)
        text_view = p._cw_view.placeholder()
        self.assertEqual(text_view.action_text(), "Listen in CW")
        self.assertFalse(p._tune_btn.isVisible())
        p.set_decoder_status(_status("CW (Morse)", "character", live_text="VVV DE"))
        self.assertFalse(text_view.is_visible())
        self.assertTrue(p._tune_btn.isVisible())
        self.assertEqual(p._tune_btn.text(), "Listen in CW")

    def test_one_button_when_the_aircraft_list_empties_by_itself(self):
        # Aircraft go five minutes after their last message: the list's
        # own "Tune to" came back beside the status line's.
        from sdr_module.dsp.protocols import ADSBMessage, ProtocolType

        p = self.panel
        view = p._aircraft_view
        view._clock = view.tracker._clock = lambda: self.now
        p.set_current_frequency(100.1e6)
        p._proto_combo.setCurrentText("ADS-B")
        msg = ADSBMessage(
            protocol=ProtocolType.ADSB,
            timestamp=0.0,
            raw_bits=b"",
            valid=True,
            icao_address="4840D6",
            downlink_format=17,
            type_code=4,
            callsign="KLM1023",
        )
        p.add_message("ADS-B", "4840D6", "KLM1023", message=msg)
        self.assertTrue(p._tune_btn.isVisible())
        self.now += 400
        view.refresh()
        self.assertTrue(view.placeholder().button.isVisible())
        self.assertFalse(p._tune_btn.isVisible())

    def test_not_offered_while_paused_or_off(self):
        p = self.panel
        p.set_current_frequency(100.1e6)
        p._proto_combo.setCurrentText("POCSAG")
        p.add_message("POCSAG", "1 (F0)", "page")
        self.assertTrue(p._tune_btn.isVisible())
        p._enabled_check.setChecked(False)
        self.assertFalse(p._tune_btn.isVisible())
        p._enabled_check.setChecked(True)
        p._proto_combo.setCurrentText("Off")
        self.assertFalse(p._tune_btn.isVisible())


# --------------------------------------------------------------------------- #
# DEC-E2E-13: CW (Morse) has a "Tune to" too
# --------------------------------------------------------------------------- #


class TestCwTuneTo(_PanelCase):
    def setUp(self):
        super().setUp()
        self.tuned = []
        self.panel.tune_requested.connect(lambda *args: self.tuned.append(args))
        self.panel.set_receiving(True)

    def test_the_demo_beacon_is_offered(self):
        # The hint named the demo beacon (99.820 MHz) with no button: a
        # typed frequency landed in FM, and Mode had to be changed by hand.
        p = self.panel
        p.set_source("demo")
        p.set_current_frequency(100.1e6)
        p.set_demod_mode("FM")
        p._proto_combo.setCurrentText("CW (Morse)")
        placeholder = p._cw_view.placeholder()
        self.assertIs(self.front(), p._cw_view)
        self.assertEqual(placeholder.action_text(), "Tune to 99.820 MHz")
        placeholder.button.click()
        self.assertEqual(self.tuned[-1], (99.82e6, "CW (Morse) (99.820 MHz)", "CW", ""))
        p.set_current_frequency(99.82e6)
        p.set_demod_mode("CW")
        self.assertEqual(placeholder.action_text(), "")
        self.assertFalse(p._tune_btn.isVisible())

    def test_elsewhere_it_offers_cw_mode(self):
        p = self.panel
        p.set_source("device")
        p.set_current_frequency(7.03e6)
        p.set_demod_mode("USB")
        p._proto_combo.setCurrentText("CW (Morse)")
        placeholder = p._cw_view.placeholder()
        self.assertEqual(placeholder.action_text(), "Listen in CW")
        placeholder.button.click()
        self.assertEqual(self.tuned[-1], (7.03e6, "CW (Morse) (7.030 MHz)", "CW", ""))
        p.set_demod_mode("CW")
        self.assertEqual(placeholder.action_text(), "")

    def test_auto_offers_nothing(self):
        p = self.panel
        p.set_source("demo")
        p.set_current_frequency(7.03e6)
        p._proto_combo.setCurrentText("Auto")
        p.set_demod_mode("CW")  # Auto: Morse code in CW
        self.assertEqual(p.protocol(), "CW (Morse)")
        self.assertEqual(p._cw_view.placeholder().action_text(), "")


class TestCwTuneToInTheWindow(_WindowTestCase):
    demo = True

    def test_the_beacon_in_cw(self):
        win = self.win
        panel = win._decoder_panel
        self.assertEqual(panel._source, "demo")
        panel._proto_combo.setCurrentText("CW (Morse)")
        self.settle(2)
        self.assertEqual(
            panel._cw_view.placeholder().action_text(), "Tune to 99.820 MHz"
        )
        panel._cw_view.placeholder().button.click()
        self.settle(2)
        self.assertAlmostEqual(win._current_frequency(), 99.82e6, delta=1.0)
        self.assertEqual(win._control_panel.get_demod_mode(), "CW")
        self.assertEqual(panel._cw_view.placeholder().action_text(), "")


# --------------------------------------------------------------------------- #
# DEC-E2E-14: "Tune to" listens the way the presets do
# --------------------------------------------------------------------------- #


class TestTuneToListensLikeThePresets(_WindowTestCase):
    def tune(self, name):
        win = self.win
        panel = win._decoder_panel
        panel._proto_combo.setCurrentText(name)
        self.assertTrue(win.set_frequency(100e6))  # away: the button shows
        self.settle(2)
        self.assertTrue(panel._tune_action(name)[0], name)
        panel._tune_to_protocol()
        self.settle(2)
        cp = win._control_panel
        return cp.get_demod_mode(), cp._bw_combo.currentText()

    def test_each_channel_is_heard_as_its_preset_is(self):
        # ACARS came in AM with a 10 kHz filter (the preset, Radio > Band
        # Presets and typing 131.55: 15 kHz), and ADS-B kept the 200 kHz
        # filter (the preset: 2.4 MHz).
        from sdr_module.gui import presets

        for name, freq in (
            ("ACARS", 131.55e6),
            ("ADS-B", 1090e6),
            ("POCSAG", 152.24e6),
            ("AX.25/APRS", 144.39e6),
            ("FLEX", 931.9375e6),
        ):
            mode, _deviation, width = presets.listening(presets.preset_at(freq))
            self.assertEqual(self.tune(name), (mode, width), name)
            self.assertAlmostEqual(self.freq(), freq)


# --------------------------------------------------------------------------- #
# DEC-E2E-19: one name for Decode Recording; Start or Play as it is
# --------------------------------------------------------------------------- #


class TestStartOrPlayWording(_PanelCase):
    def test_the_button_is_named_like_the_menu_item(self):
        self.assertEqual(self.panel._decode_file_btn.text(), "Decode Recording...")

    def test_off_while_receiving_does_not_ask_for_start(self):
        p = self.panel
        p.set_receiving(True)
        self.assertNotIn("Start", p._empty.text())
        self.assertIn("Choose a protocol above", p._empty.text())
        p.set_receiving(False)
        self.assertIn("press Start (Space)", p._empty.text())

    def test_a_recording_says_play(self):
        p = self.panel
        p.set_source("recording")
        p.set_receiving(False)
        p.set_current_frequency(152.24e6)
        p._proto_combo.setCurrentText("POCSAG")
        p.set_decoder_status(_status("POCSAG", "page"))
        self.assertEqual(
            p.decoder_status_text(), "Not playing: press Play (Space) to decode"
        )
        self.assertIn("press Play (Space)", p._empty.text())
        self.assertNotIn("Start", p._empty.text())
        p.set_source("demo")
        self.assertEqual(
            p.decoder_status_text(), "Stopped: press Start (Space) to decode"
        )


class TestProtocolRowFitsTheColumn(_WindowTestCase):
    def test_the_longer_button_leaves_the_column_as_wide_as_before(self):
        # "Decode Recording..." (compact) and the status line's "Tune to"
        # must not make the Decoder wider than the right column the window
        # gives it at 1024x640 (400 px), in the themes the app runs with.
        # (test_gui_panels measures the unthemed panel only against its own
        # narrowest width.)
        from sdr_module.gui import themes

        win = self.win
        panel = win._decoder_panel
        win._show_panel("decoder")
        panel._proto_combo.setCurrentText("POCSAG")
        panel.add_message("AX.25/APRS", "N0CALL-9", "a packet")
        self.assertTrue(win.set_frequency(144.39e6))
        qapp = QApplication.instance()
        old_sheet, old_palette = qapp.styleSheet(), qapp.palette()
        old_theme = themes.current_theme()
        try:
            for theme in themes.THEMES:
                themes.apply_theme(qapp, theme)
                win.resize(1024, 640)
                self.settle(5)
                win._apply_default_splitter_sizes()
                self.settle(5)
                self.assertTrue(panel._tune_btn.isVisible(), theme)
                self.assertLessEqual(win._main_splitter.sizes()[1], 400, theme)
                self.assertLessEqual(
                    panel.minimumSizeHint().width(), panel.width(), theme
                )
                for button in (panel._decode_file_btn, panel._tune_btn):
                    right = button.mapTo(panel, button.rect().topRight()).x()
                    self.assertLess(right, panel.width(), (theme, button.text()))
        finally:
            qapp.setStyleSheet(old_sheet)
            qapp.setPalette(old_palette)
            themes.apply_theme(None, old_theme)


# --------------------------------------------------------------------------- #
# DEC-E2E-17: a recording's messages are told apart from live ones
# --------------------------------------------------------------------------- #


def _adsb(**fields):
    from sdr_module.dsp.protocols import ADSBMessage, ProtocolType

    base = dict(protocol=ProtocolType.ADSB, timestamp=0.0, raw_bits=b"", valid=True)
    base.update(fields)
    return ADSBMessage(**base)


class TestFileRowsApart(_PanelCase):
    def test_marked_with_the_recordings_time_and_kept_out_of_the_live_views(self):
        # Decode File during live ADS-B: the file's rows interleaved with
        # the live ones, both with today's time, and the file's messages
        # updated the live aircraft (position, "Seen now", Msgs).
        from datetime import datetime, timezone

        from sdr_module.dsp.morse import MorseMessage
        from sdr_module.dsp.protocols import ProtocolType

        p = self.panel
        mod = self.mod
        table = p._table
        p.set_receiving(True)
        p.set_current_frequency(1090e6)
        p._proto_combo.setCurrentText("ADS-B")
        live = _adsb(icao_address="4840D6", downlink_format=17, type_code=11,
                     altitude=38000, latitude=52.2568, longitude=3.913)  # fmt: skip
        p.add_message("ADS-B", "4840D6", "38,000 ft", message=live)
        recorded = datetime(2026, 10, 30, 7, 0, 1, tzinfo=timezone.utc)
        old = _adsb(icao_address="4840D6", downlink_format=17, type_code=11,
                    altitude=12000, latitude=50.0, longitude=4.0)  # fmt: skip
        p.add_message("ADS-B", "4840D6", "12,000 ft  (from adsb.cs16 at 0.4 s)",
                      message=old, source="adsb.cs16", when=recorded)  # fmt: skip
        self.assertFalse(table.isColumnHidden(mod.COL_PROTOCOL))
        self.assertEqual(table.item(1, mod.COL_PROTOCOL).text(), "ADS-B (file)")
        local = recorded.astimezone().strftime("%H:%M:%S")
        self.assertEqual(table.item(1, mod.COL_TIME).text(), local)
        self.assertIn("adsb.cs16", table.item(1, mod.COL_ADDRESS).toolTip())
        aircraft = p._aircraft_view.tracker.get("4840D6")
        self.assertEqual((aircraft.messages, aircraft.altitude), (1, 38000))
        self.assertEqual(p.count_for("ADS-B"), 1)  # (the chip's count: live)
        # Morse code from a recording stays out of the live Text view too.
        p._proto_combo.setCurrentText("CW (Morse)")
        line = MorseMessage(protocol=ProtocolType.CW, timestamp=0.0, raw_bits=b"",
                            valid=True, text="CQ DE N0CALL", wpm=18.0)  # fmt: skip
        p.add_message("CW (Morse)", "18 WPM", "CQ DE N0CALL", message=line,
                      source="cw.wav")  # fmt: skip
        self.assertEqual(p._cw_view.plain_text(), "")


# --------------------------------------------------------------------------- #
# DEC-E2E-18: the Aircraft list puts Seen and Msgs early and sorts
# --------------------------------------------------------------------------- #


class TestAircraftList(_PanelCase):
    def setUp(self):
        super().setUp()
        self.panel._proto_combo.setCurrentText("ADS-B")
        _settle(1)
        self.view = self.panel._aircraft_view
        self.table = self.view.table

    def heard(self, icao, altitude, callsign):
        for msg in (
            _adsb(icao_address=icao, downlink_format=17, type_code=4, callsign=callsign),
            _adsb(icao_address=icao, downlink_format=17, type_code=11,
                  altitude=altitude, latitude=50.5, longitude=4.5),
        ):  # fmt: skip
            self.view.update_message(msg)

    def names(self):
        return [self.table.item(r, 0).text() for r in range(self.table.rowCount())]

    def click_header(self, col):
        from PyQt6.QtCore import QPoint, Qt
        from PyQt6.QtTest import QTest

        header = self.table.horizontalHeader()
        x = header.sectionViewportPosition(col) + header.sectionSize(col) // 2
        QTest.mouseClick(header.viewport(), Qt.MouseButton.LeftButton,
                         pos=QPoint(x, header.height() // 2))  # fmt: skip
        _settle(1)

    def test_seen_and_msgs_come_before_track_and_vertical_speed(self):
        # Seen and Msgs were behind the sideways scroll at every size.
        titles = [
            self.table.horizontalHeaderItem(c).text()
            for c in range(self.table.columnCount())
        ]
        self.assertEqual(
            titles,
            ["Flight", "Altitude", "Speed", "Seen", "Msgs", "Track", "V/S",
             "Squawk", "Position", "ICAO"],
        )  # fmt: skip

    def test_a_click_on_a_title_sorts_by_it_as_numbers(self):
        from sdr_module.gui import decoder_views

        self.heard("400001", 5000, "LOW1")
        self.heard("400002", 38000, "HIGH1")
        self.heard("400003", 12000, "MID1")
        self.assertEqual(self.names(), ["LOW1", "HIGH1", "MID1"])  # as heard
        widths = [self.table.columnWidth(c) for c in range(self.table.columnCount())]
        self.click_header(decoder_views.AC_ALTITUDE)
        self.assertEqual(self.names(), ["LOW1", "MID1", "HIGH1"])
        title = self.table.horizontalHeaderItem(decoder_views.AC_ALTITUDE)
        self.assertEqual(title.text(), "Altitude ▲")
        # (Qt's arrow is hidden: it took room in every title, and then two
        # columns fewer fitted.)
        self.assertFalse(self.table.horizontalHeader().isSortIndicatorShown())
        self.assertEqual(
            self.table.columnWidth(decoder_views.AC_SPEED),
            widths[decoder_views.AC_SPEED],
        )
        self.click_header(decoder_views.AC_ALTITUDE)
        self.assertEqual(self.names(), ["HIGH1", "MID1", "LOW1"])
        self.assertEqual(title.text(), "Altitude ▼")
        # A new aircraft goes to its place.
        self.heard("400004", 20000, "NEW1")
        self.assertEqual(self.names(), ["HIGH1", "NEW1", "MID1", "LOW1"])
        # An update moves its row; picking and copying still find it.
        self.view.select("400003")
        self.view.update_message(
            _adsb(icao_address="400003", downlink_format=17, type_code=11,
                  altitude=41000, latitude=50.5, longitude=4.5)
        )  # fmt: skip
        self.assertEqual(self.names(), ["MID1", "HIGH1", "NEW1", "LOW1"])
        self.assertEqual(self.view.selected_icao(), "400003")
        self.assertEqual(self.panel.address_filter(), "400003")
        self.assertIn("MID1 (400003)", self.view.detail.text())


# --------------------------------------------------------------------------- #
# DEC-E2E-21: F1 lists the Aircraft tab's keys
# --------------------------------------------------------------------------- #


class TestHelpListsTheAircraftKeys(unittest.TestCase):
    def test_copy_and_double_click_are_listed(self):
        # Only "Return: show the selected aircraft's messages" was there,
        # while Ctrl+C copies the aircraft's line and a double-click works
        # like Return.
        from sdr_module.gui.help_dialog import PANEL_KEYS

        rows = {rid: (section, keys, desc) for rid, section, keys, desc in PANEL_KEYS}
        self.assertEqual(
            rows["air_copy"],
            ("Decoder messages", ("Ctrl+C",), "Aircraft tab: copy the selected aircraft's line"),
        )  # fmt: skip
        self.assertIn("double-click", rows["air_enter"][2])


# --------------------------------------------------------------------------- #
# DEC-E2E-04: Decode Recording's frequency follows the protocol chosen
# --------------------------------------------------------------------------- #


def _options(center=144.09e6, rate=1e6, seconds=10.0, name="aprs_1M_offset.cs16"):
    from sdr_module.gui.iq_files import IqFileOptions

    return IqFileOptions(
        path=os.path.join(os.sep, "recordings", name),
        sample_rate=rate,
        center_hz=center,
        total_samples=int(seconds * rate),
        source="a sidecar file",
    )


class TestDecodeDialogFrequency(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)

    def dialog(self, options, protocol="POCSAG", defaults=None):
        from sdr_module.gui.iq_files import DecodeOptionsDialog

        protocols = ["POCSAG", "FLEX", "AX.25/APRS", "ADS-B", "ACARS", "RDS"]
        start = (defaults or {}).get(protocol, options.center_hz)
        dialog = DecodeOptionsDialog(
            options, protocols, protocol, start, None, defaults=defaults
        )
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_choosing_a_protocol_moves_to_its_channel(self):
        # It stayed on the frequency of the protocol it opened with (the
        # file's center, for POCSAG), and APRS at 144.390 MHz was missed.
        defaults = {"POCSAG": 144.09e6, "AX.25/APRS": 144.39e6, "RDS": 144.09e6}
        dialog = self.dialog(_options(), "POCSAG", defaults)
        self.assertEqual(dialog._freq.text(), "144.090 MHz")
        dialog._protocol.setCurrentText("AX.25/APRS")
        self.assertEqual(dialog._freq.text(), "144.390 MHz")
        dialog._on_accept()
        self.assertEqual(dialog.choice()[0], "AX.25/APRS")
        self.assertAlmostEqual(dialog.choice()[1], 300e3, delta=1.0)

    def test_a_frequency_typed_by_hand_stays(self):
        from PyQt6.QtTest import QTest

        defaults = {"POCSAG": 144.09e6, "AX.25/APRS": 144.39e6}
        dialog = self.dialog(_options(), "POCSAG", defaults)
        dialog._freq.selectAll()
        QTest.keyClicks(dialog._freq, "144.2")
        dialog._protocol.setCurrentText("AX.25/APRS")
        self.assertEqual(dialog._freq.text(), "144.2")

    def test_without_defaults_nothing_changes(self):
        dialog = self.dialog(_options(), "POCSAG")
        dialog._protocol.setCurrentText("AX.25/APRS")
        self.assertEqual(dialog._freq.text(), "144.090 MHz")

    # DEC-E2E-09: a protocol the recording's rate is too low for

    def test_a_protocol_too_fast_for_the_recording_is_greyed_out(self):
        from PyQt6.QtCore import Qt

        dialog = self.dialog(_options(rate=1e6), "POCSAG")
        combo = dialog._protocol
        index = combo.findText("ADS-B")
        self.assertFalse(combo.model().item(index).isEnabled())
        self.assertEqual(
            combo.itemData(index, Qt.ItemDataRole.ToolTipRole),
            "ADS-B needs at least 2 MS/s; this recording is 1 MS/s",
        )
        self.assertTrue(combo.model().item(combo.findText("POCSAG")).isEnabled())

    def test_chosen_in_the_decoder_it_gives_way_and_says_why(self):
        dialog = self.dialog(_options(rate=1e6), "ADS-B")
        self.assertNotEqual(dialog._protocol.currentText(), "ADS-B")
        self.assertIn(
            "ADS-B needs at least 2 MS/s; this recording is 1 MS/s",
            dialog._note.text(),
        )
        self.assertEqual(dialog._note.property("tone"), "warning")

    def test_the_decode_job_names_the_recording_not_the_device(self):
        # "ADS-B needs at least 2 MS/s; the device runs at 1 MS/s" while
        # the receiver ran at 2.4 MS/s.
        from sdr_module.gui.file_jobs import decode_recording

        with self.assertRaises(ValueError) as caught:
            decode_recording(_options(rate=1e6), "ADS-B", 0.0, lambda _f: None)
        self.assertEqual(
            str(caught.exception),
            "ADS-B needs at least 2 MS/s; this recording is 1 MS/s",
        )


class TestDecodeRecordingDefaults(_WindowTestCase):
    def test_the_window_gives_each_protocol_its_channel(self):
        from sdr_module.gui import iq_files

        win = self.win
        win._decoder_panel._proto_combo.setCurrentText("Off")
        self.settle(2)
        self.assertTrue(win.set_frequency(100.1e6))
        options = _options()
        seen = {}

        def fake_exec(dialog):
            if isinstance(dialog, iq_files.DecodeOptionsDialog):
                seen["start"] = dialog._freq.text()
                dialog._protocol.setCurrentText("AX.25/APRS")
                seen["aprs"] = dialog._freq.text()
                dialog._protocol.setCurrentText("ACARS")  # (not in the file)
                seen["acars"] = dialog._freq.text()
            return 0  # cancelled: nothing is decoded

        win._exec_dialog = fake_exec
        self.assertFalse(win._decode_recording(options.path, options))
        self.assertEqual(seen["start"], "144.090 MHz")  # POCSAG: the center
        self.assertEqual(seen["aprs"], "144.390 MHz")
        self.assertEqual(seen["acars"], "144.090 MHz")


# --------------------------------------------------------------------------- #
# DEC-E2E-05: Export starts in the recordings folder and remembers its own
# --------------------------------------------------------------------------- #


class TestExportFolder(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        import tempfile

        from sdr_module.gui import decoder_panel, file_dialogs
        from tests.test_gui_chrome import _FakeSettings

        self.mod = decoder_panel
        self.dialogs = file_dialogs
        self.store = _FakeSettings()
        self.panel = decoder_panel.DecoderPanel(settings=self.store)
        self.addCleanup(self.panel.deleteLater)
        self.panel.add_message("POCSAG", "1 (F0)", "first page")
        self.folder = tempfile.mkdtemp(prefix="sdr-export-")
        self.addCleanup(__import__("shutil").rmtree, self.folder, True)

    def export(self, answer):
        from unittest import mock

        with (
            mock.patch.object(
                self.dialogs.QFileDialog, "getSaveFileName", return_value=(answer, "")
            ) as dialog,
            mock.patch.object(self.mod.QMessageBox, "information"),
        ):
            count = self.panel._export_messages()
        return count, dialog.call_args[0][2]

    def test_starts_in_the_recordings_folder_not_the_working_directory(self):
        # The suggested name was bare ("decoded-....csv"): the dialog opened
        # in the folder the app was started from ("/", or System32 when
        # started from a Windows shortcut).
        count, suggested = self.export("")
        self.assertEqual(count, 0)  # (cancelled)
        self.assertEqual(
            os.path.dirname(suggested), self.dialogs.default_folder("decode")
        )
        self.assertRegex(os.path.basename(suggested), r"^decoded-\d{8}-\d{6}\.csv$")

    def test_remembers_the_folder_used(self):
        target = os.path.join(self.folder, "mine.csv")
        count, _suggested = self.export(target)
        self.assertEqual(count, 1)
        self.assertTrue(os.path.exists(target))
        _count, suggested = self.export("")
        self.assertEqual(os.path.dirname(suggested), self.folder)


# --------------------------------------------------------------------------- #
# DEC-E2E-06: only the newest messages are kept, and it says so
# --------------------------------------------------------------------------- #


class TestOnlyTheNewestAreKept(_PanelCase):
    def test_the_table_stats_and_export_say_so(self):
        import csv
        import tempfile
        from unittest import mock

        p = self.panel
        p._max_messages = 5  # (1,000 in the app)
        base_tip = p._table.toolTip()
        for i in range(3):
            p.add_message("POCSAG", f"{i} (F0)", f"page {i}")
        self.assertEqual(p._table.toolTip(), base_tip)  # nothing dropped yet
        self.assertEqual(p._kept_label.text(), "3")
        for i in range(3, 8):
            p.add_message("POCSAG", f"{i} (F0)", f"page {i}")
        self.assertEqual(p._table.rowCount(), 5)
        # The oldest went without a word; now the table says it...
        self.assertIn("newest 5 of 8 messages", p._table.toolTip())
        self.assertEqual(p._msg_count_label.text(), "8")
        self.assertEqual(p._kept_label.text(), "5 (the newest)")
        self.assertIn("newest 5 of the 8", p._export_btn.toolTip())
        # ... and so does Export.
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(self.mod.QMessageBox, "information") as box,
        ):
            path = os.path.join(tmp, "out.csv")
            self.assertEqual(p._export_messages(path), 5)
            with open(path, newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(len(rows), 6)  # the header and the newest 5
        text = box.call_args[0][2]
        self.assertIn("Exported the newest 5 of 8 messages", text)
        self.assertIn("older ones", text)
        p.clear()
        self.assertEqual(p._table.toolTip(), base_tip)
        self.assertEqual(p._kept_label.text(), "0")


# --------------------------------------------------------------------------- #
# DEC-E2E-11: short columns show their text in full
# --------------------------------------------------------------------------- #


class TestColumnsShowTheirText(_PanelCase):
    def widest_time(self, font):
        """The time of day widest in ``font`` (its digits are proportional,
        with fractional advances)."""
        from PyQt6.QtGui import QFontMetricsF

        fm = QFontMetricsF(font)

        def best(digits):
            return max(digits, key=fm.horizontalAdvance)

        def width(digit):
            return fm.horizontalAdvance(digit)

        h1 = max(
            "012",
            key=lambda d: width(d) + width(best("0123" if d == "2" else "0123456789")),
        )
        h2 = best("0123" if h1 == "2" else "0123456789")
        return f"{h1}{h2}:{best('012345')}{best('0123456789')}:{best('012345')}{best('0123456789')}"  # fmt: skip

    def test_protocol_address_and_time_are_not_elided(self):
        # 'CW (Mors…' (83 px wanted, 82 given: the theme pads cells by 15
        # px, not 14) and '20:44:…' (sized from "00:00:00", while 4 is
        # wider than 0 and 1 narrower). Measured in the app's themes.
        from datetime import datetime
        from unittest import mock

        from sdr_module.gui import themes

        qapp = QApplication.instance()
        old_sheet, old_palette = qapp.styleSheet(), qapp.palette()
        old_theme = themes.current_theme()
        mod = self.mod
        try:
            for theme in themes.THEMES:
                themes.apply_theme(qapp, theme)
                p = mod.DecoderPanel()
                self.addCleanup(p.deleteLater)
                p.resize(420, 520)
                p.show()
                _settle(2)
                table = p._table
                widest = self.widest_time(table.font())
                fake = mock.Mock(wraps=datetime)
                fake.now.return_value = datetime.strptime(widest, "%H:%M:%S")
                with mock.patch.object(mod, "datetime", fake):
                    p.add_message("RDS", "C0DE", "Station name: SDR DEMO")
                    p.add_message("CW (Morse)", "15 WPM", "VVV DE SDR")
                _settle(2)
                self.assertFalse(table.isColumnHidden(mod.COL_PROTOCOL))
                for col in (mod.COL_TIME, mod.COL_PROTOCOL, mod.COL_ADDRESS):
                    self.assertGreaterEqual(
                        table.columnWidth(col),
                        table.sizeHintForColumn(col),
                        (theme, col, widest),
                    )
                p.close()
        finally:
            qapp.setStyleSheet(old_sheet)
            qapp.setPalette(old_palette)
            themes.apply_theme(None, old_theme)


# --------------------------------------------------------------------------- #
# DEC-E2E-08: the status line and the chip count the same messages
# DEC-E2E-10: tuned away, the status line says so (not the green count)
# --------------------------------------------------------------------------- #


class TestOneCount(_PanelCase):
    def setUp(self):
        super().setUp()
        self.panel.tune_requested.connect(lambda *_args: None)  # (the window)
        self.panel.set_receiving(True)

    def test_rds_counts_its_messages_and_names_the_groups(self):
        # The line said "802 groups" beside the chip's "3 msgs".
        p = self.panel
        p.set_current_frequency(100.1e6)
        p._proto_combo.setCurrentText("RDS")
        p.set_decoder_status(_status("RDS", "group", frames=40, valid=40,
                                     last_time=self.now, detail="SDR DEMO"))  # fmt: skip
        self.assertEqual(
            p.decoder_status_text(),
            "RDS  ·  SDR DEMO  ·  0 messages (40 groups)  ·  last just now",
        )
        for text in ("Station name: SDR DEMO", "Program type: Top 40", "Radio text"):
            p.add_message("RDS", "C0DE", text)
        self.assertEqual(p.count_for("RDS"), 3)
        self.assertEqual(
            p.decoder_status_text(),
            "RDS  ·  SDR DEMO  ·  3 messages (40 groups)  ·  last just now",
        )

    def test_a_rebuilt_decoder_does_not_start_the_count_again(self):
        # After a sample-rate change the line counted from zero ("ADS-B ·
        # 4 messages") while the chip went on ("ADS-B · 32 msgs").
        p = self.panel
        p.set_current_frequency(1090e6)
        p._proto_combo.setCurrentText("ADS-B")
        for _ in range(5):
            p.add_message("ADS-B", "4840D6", "KLM1023")
        p.set_decoder_status(_status("ADS-B", frames=5, valid=5, last_time=self.now))
        self.assertEqual(
            p.decoder_status_text(), "ADS-B  ·  5 messages  ·  last just now"
        )
        # Rebuilt (another rate): its own counts start again.
        p.set_decoder_status(_status("ADS-B", frames=1, valid=1, last_time=self.now))
        p.add_message("ADS-B", "4840D6", "KLM1023")
        self.assertEqual(
            p.decoder_status_text(), "ADS-B  ·  6 messages  ·  last just now"
        )
        self.assertEqual(p.count_for("ADS-B"), 6)

    def test_failed_messages_are_not_counted_as_decoded(self):
        p = self.panel
        p.set_current_frequency(152.24e6)
        p._proto_combo.setCurrentText("POCSAG")
        p.add_message("POCSAG", "1 (F0)", "good")
        p.add_message("POCSAG", "2 (F0)", "bad", valid=False)
        p.set_decoder_status(_status("POCSAG", "page", frames=2, valid=1, failed=1,
                                     last_time=self.now))  # fmt: skip
        self.assertEqual(p.count_for("POCSAG"), 1)
        self.assertEqual(
            p.decoder_status_text(),
            "POCSAG  ·  1 page  ·  last just now  ·  1 failed check",
        )

    def test_tuned_away_says_not_tuned_with_what_came_before(self):
        p = self.panel
        p.set_current_frequency(100.1e6)
        p._proto_combo.setCurrentText("RDS")
        for text in ("Station name: SDR DEMO", "Program type: Top 40", "Radio text"):
            p.add_message("RDS", "C0DE", text)
        p.set_decoder_status(_status("RDS", "group", frames=56, valid=56,
                                     last_time=self.now, detail="SDR DEMO"))  # fmt: skip
        self.assertEqual(p._status_label.property("tone"), "success")
        self.now += 1.0
        p.set_current_frequency(145e6)  # Ctrl+L 145.0
        self.now += 12.0
        p.set_decoder_status(_status("RDS", "group", frames=56, valid=56,
                                     last_time=self.now - 13.0))  # fmt: skip
        self.assertEqual(
            p.decoder_status_text(),
            "Not tuned to RDS: listening on 145.000 MHz (3 messages earlier)",
        )
        self.assertFalse(p._status_label.property("tone"))  # not the green
        self.assertEqual(p._tune_btn.text(), "Tune to 100.100 MHz")

    def test_still_decoding_off_the_band_shows_the_count(self):
        # ADS-B decodes from the whole span: with the VFO moved off 1090 MHz
        # (the LO still there) the messages go on, and so does the count.
        p = self.panel
        p.set_current_frequency(1090e6)
        p._proto_combo.setCurrentText("ADS-B")
        p.add_message("ADS-B", "4840D6", "KLM1023")
        self.now += 1.0
        p.set_current_frequency(1091.5e6)
        self.now += 1.0
        p.add_message("ADS-B", "4840D6", "KLM1023")
        p.set_decoder_status(_status("ADS-B", frames=2, valid=2, last_time=self.now))
        self.assertEqual(
            p.decoder_status_text(), "ADS-B  ·  2 messages  ·  last just now"
        )


class TestOneCountInTheWindow(_WindowTestCase):
    def test_the_chip_and_the_status_line_agree(self):
        import numpy as np

        from tests.test_gui_decoder_views import _adsb

        win = self.win
        panel = win._decoder_panel
        self.assertTrue(win.set_frequency(1090e6))
        panel._proto_combo.setCurrentText("ADS-B")
        self.settle(2)
        panel.set_receiving(True)  # (no device runs in this test)
        for _ in range(3):
            win._push_decoded_message(_adsb(icao_address="4840D6", downlink_format=11))
        win._run_decoder(np.zeros(64, dtype=np.complex64))
        self.assertEqual(win._decoder_chip.text(), "ADS-B · 3 msgs")
        self.assertTrue(
            panel.decoder_status_text().startswith("ADS-B  ·  3 messages"),
            panel.decoder_status_text(),
        )
        # A new decoder (as after a sample-rate change) keeps both at 3.
        win._rebuild_decoder()
        win._run_decoder(np.zeros(64, dtype=np.complex64))
        self.assertEqual(win._decoder_chip.text(), "ADS-B · 3 msgs")
        self.assertTrue(
            panel.decoder_status_text().startswith("ADS-B  ·  3 messages"),
            panel.decoder_status_text(),
        )


class TestExportFolderInTheWindow(_WindowTestCase):
    def test_the_panel_remembers_in_the_windows_settings(self):
        self.assertIs(self.win._decoder_panel._settings, self.win._settings)


# --------------------------------------------------------------------------- #
# DEC-E2E-07: the status-bar chip says "paused" and the decoder's problem
# --------------------------------------------------------------------------- #


class TestChipPausedAndProblems(_WindowTestCase):
    demo = True  # (its sample rate can be changed)

    def select(self, name):
        self.win._decoder_panel._proto_combo.setCurrentText(name)
        self.settle(2)

    def test_unticking_decode_says_paused_at_once(self):
        # The chip went on saying "AX.25/APRS · 0 msgs" while the status
        # line said "Paused" (only an unrelated refresh updated it).
        win = self.win
        chip = win._decoder_chip
        self.select("AX.25/APRS")
        self.assertEqual(chip.text(), "AX.25/APRS · 0 msgs")
        win._decoder_panel._enabled_check.click()  # untick, as a user does
        self.assertEqual(chip.text(), "AX.25/APRS · paused")
        win._decoder_panel._enabled_check.click()
        self.assertEqual(chip.text(), "AX.25/APRS · 0 msgs")

    def test_a_decoder_that_cannot_decode_says_so(self):
        # ADS-B at 1.8 MS/s: "ADS-B · 3 msgs" with no hint that nothing
        # can decode.
        win = self.win
        chip = win._decoder_chip
        self.assertTrue(win.set_sample_rate(1.8e6))
        self.select("ADS-B")
        self.assertEqual(chip.text(), "ADS-B · can't decode")
        self.assertEqual(chip.property("tone"), "warning")
        self.assertIn("ADS-B needs at least 2 MS/s", chip.toolTip())
        self.assertIn("Click to open the Decoder tab", chip.toolTip())
        self.assertTrue(win.set_sample_rate(2.4e6))
        self.assertEqual(chip.text(), "ADS-B · 0 msgs")
        self.assertFalse(chip.property("tone"))


# --------------------------------------------------------------------------- #
# DEC-E2E-16: one Error History entry per sample-rate step with ADS-B
# --------------------------------------------------------------------------- #


class TestOneWarningPerRateStep(_WindowTestCase):
    demo = True

    def test_each_step_too_low_for_ads_b_is_listed_once(self):
        # Each step gave two warnings: the rebuilt decoder's ("ADS-B
        # decoder: ADS-B needs at least 2 MS/s; the device runs at 1.8
        # MS/s") and the rate's status message saying it again.
        import logging

        win = self.win
        win._decoder_panel._proto_combo.setCurrentText("ADS-B")
        self.settle(2)
        with self.assertLogs(level=logging.WARNING) as logs:
            self.assertTrue(win._device_controls.choose_rate(1.8e6))
            self.assertTrue(win._device_controls.choose_rate(1.4e6))
        records = [r.getMessage() for r in logs.records]
        self.assertEqual(len(records), 2, records)
        self.assertTrue(all("ADS-B" in text for text in records), records)
        # The status bar still says it.
        self.assertIn("ADS-B needs", win._message_label.text())
        self.assertEqual(win._message_label.property("tone"), "warning")


# --------------------------------------------------------------------------- #
# L28: the sample-rate tips recommend 2.4 MS/s for ADS-B
# --------------------------------------------------------------------------- #


class TestAdsbRateTips(_WindowTestCase):
    demo = True

    def test_the_tips_say_2_4_ms_s_or_more(self):
        # They said "at least 2 MS/s", but 2.0-2.1 MS/s decode poorly.
        from sdr_module.dsp.live_decoders import create_live_decoder
        from sdr_module.gui.device_settings import ADSB_RATE_TIP

        self.assertIn("needs 2.4 MS/s or more", ADSB_RATE_TIP)
        remedy = create_live_decoder("ADS-B", 1.8e6).status.remedy
        self.assertIn("Choose 2.4 MS/s or more", remedy)

    def test_a_rate_that_decodes_poorly_says_so(self):
        win = self.win
        win._decoder_panel._proto_combo.setCurrentText("ADS-B")
        self.settle(2)
        self.assertTrue(win._device_controls.choose_rate(2.0e6))
        self.assertIn(
            "ADS-B decodes best at 2.4 MS/s or more", win._message_label.text()
        )
        self.assertTrue(win._device_controls.choose_rate(1.8e6))
        self.assertIn("ADS-B needs 2.4 MS/s or more", win._message_label.text())
        self.assertTrue(win._device_controls.choose_rate(2.4e6))
        self.assertNotIn("ADS-B", win._message_label.text())


# --------------------------------------------------------------------------- #
# L29: Clear clears FLEX's "sends FLEX at 3200 bps"
# --------------------------------------------------------------------------- #


class TestClearForgetsTheFlexSpeedNote(_WindowTestCase):
    def test_clear_takes_the_note_away(self):
        win = self.win
        panel = win._decoder_panel
        self.assertTrue(win.set_frequency(931.9375e6))
        panel._proto_combo.setCurrentText("FLEX")
        self.settle(2)
        decoder = win._decoder
        decoder._decoder.last_unsupported = "3200 bps 2-level"
        decoder._other_speed_s = decoder._stream_s  # (one was just heard)
        panel.set_receiving(True)
        panel.set_decoder_status(decoder.status)
        self.assertIn("sends FLEX at 3200 bps", panel.decoder_status_text())
        panel.add_message("FLEX", "1234567", "page")  # (Clear needs one)
        panel._clear_btn.click()
        self.assertIsNone(decoder.status.problem)
        self.assertNotIn("3200 bps", panel.decoder_status_text())


# (The Decode Recording window test answers the dialogs as the recording
# tests do.)
from tests.test_gui_recording_files import RATE, _FilesCase  # noqa: E402


class TestFileRowsApartInTheWindow(_FilesCase):
    def test_decode_recording_marks_its_rows(self):
        import numpy as np

        from sdr_module.dsp import protocol_encoders as enc
        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file
        from sdr_module.gui.iq_files import describe_iq_file

        burst = enc.aprs_baseband(
            "N0CALL-9", "APZ001", "!4903.50N/07201.75W>From a file", RATE, ["WIDE1-1"]
        ).astype(np.complex64)
        pad = np.zeros(int(0.3 * RATE), np.complex64)
        path = self.path("aprs.sigmf-data")
        save_iq_file(
            path,
            np.concatenate([pad, burst, pad, pad]),
            RATE,
            144.39e6,
            SampleFormat.FLOAT32,
            FileFormat.SIGMF,
        )
        panel = self.win._decoder_panel
        panel._proto_combo.setCurrentText("AX.25/APRS")
        self.settle(2)
        self.answer()
        with self.open_dialog(path):
            self.assertTrue(self.win._decode_recording())
        self.assertEqual(
            [m["protocol"] for m in panel._messages], ["AX.25/APRS (file)"]
        )
        self.assertEqual(panel._messages[0]["source"], "aprs.sigmf-data")
        self.assertEqual(panel.count_for("AX.25/APRS"), 0)  # (the live count)
        # Stamped with the recording's time (its start plus where in it).
        from datetime import datetime, timedelta

        options, _missing = describe_iq_file(path)
        start = datetime.fromisoformat(options.start_time.replace("Z", "+00:00"))
        stamped = datetime.strptime(
            f"{panel._messages[0]['date']} {panel._messages[0]['time']}",
            "%Y-%m-%d %H:%M:%S.%f",
        )
        expected = (start + timedelta(seconds=0.3)).astimezone().replace(tzinfo=None)
        self.assertLess(abs((stamped - expected).total_seconds()), 1.0)


if __name__ == "__main__":
    unittest.main()

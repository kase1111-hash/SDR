#!/usr/bin/env python3
"""
T12: one preset table, and listening the way a band is heard.

mc:MC-U06, rx:UX-08, ux:UX-U07: Radio > Band Presets, the control panel's
Presets group, the welcome screen and the Scanner tab offered different
preset lists under different names; the ADS-B preset left the ADS-B
decoder off; the Presets group showed an unrelated preset as if it were
what the receiver was tuned to. Wave-1 review: a far tune kept the old
band's mode profile (broadcast FM settings on a 2 m repeater, about 23 dB
too quiet). ux:UX-U08: the presets' transmit line and the License Profile
took room in the receive panel for RX-only users.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_one_preset_table.py``
"""

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, _WindowTestCase  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest


def _require_qt(case):
    if not HAS_PYQT6:
        case.skipTest("PyQt6 not available")


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


class TestOneTable(unittest.TestCase):
    def setUp(self):
        _require_qt(self)
        from sdr_module.gui import presets

        self.presets = presets

    def test_favourites_are_frequency_manager_presets(self):
        from sdr_module.core.frequency_manager import get_frequency_manager

        fm = get_frequency_manager()
        for favourite in self.presets.FAVOURITES:
            self.assertIsNotNone(fm.get_preset_by_name(favourite.preset), favourite)
            self.assertIsNotNone(self.presets.band_of(favourite), favourite)

    def test_the_menu_is_built_from_the_table(self):
        from sdr_module.gui import main_window

        names = [main_window._plain(p.label) for p in main_window.BAND_PRESETS]
        self.assertEqual(names, [f.preset for f in self.presets.FAVOURITES])
        for item in main_window.BAND_PRESETS:
            preset = self.presets.preset_by_name(main_window._plain(item.label))
            self.assertEqual(item.frequency_hz, preset.frequency_hz)
            mode, deviation, width = self.presets.listening(preset)
            self.assertEqual((item.mode, item.fm_deviation or "", item.bandwidth),
                             (mode, deviation, width))  # fmt: skip
        self.assertEqual(
            list(main_window._PRESET_CATEGORY_LABELS), self.presets.categories()
        )

    def test_the_scanner_offers_the_favourites_bands(self):
        from sdr_module.gui.scanner_dialog import SCAN_PRESETS

        self.assertEqual(
            [preset.label for preset in SCAN_PRESETS],
            [self.presets.band_label(f) for f in self.presets.FAVOURITES if f.scanned],
        )
        airband = next(p for p in SCAN_PRESETS if p.label.startswith("Airband"))
        self.assertEqual(airband.mode, "AM")

    def test_presets_name_their_decoder(self):
        decoder = self.presets.decoder_for
        by_name = self.presets.preset_by_name
        self.assertEqual(decoder(by_name("ADS-B")), "ADS-B")
        self.assertEqual(decoder(by_name("APRS")), "AX.25/APRS")
        self.assertEqual(decoder(by_name("ACARS")), "ACARS")
        self.assertEqual(decoder(by_name("POCSAG")), "POCSAG")
        self.assertEqual(decoder(by_name("FLEX")), "FLEX")
        self.assertEqual(decoder(by_name("ISS SSTV/Voice")), "SSTV")
        self.assertEqual(decoder(by_name("FM Broadcast")), "")
        self.assertIn("ADS-B decoder", self.presets.describe(by_name("ADS-B")))

    def test_every_decoder_named_is_one_the_decoder_tab_offers(self):
        from sdr_module.gui.decoder_panel import DecoderPanel

        panel = DecoderPanel()
        self.addCleanup(panel.deleteLater)
        offered = {
            panel._proto_combo.itemText(i) for i in range(panel._proto_combo.count())
        }
        for preset in self.presets.presets():
            name = self.presets.decoder_for(preset)
            if name and name != "SSTV":  # (SSTV has a tab of its own)
                self.assertIn(name, offered, preset.name)

    def test_describe_uses_the_mode_list_words_and_the_filter_apply_picks(self):
        from sdr_module.gui.control_panel import bandwidth_options

        by_name = self.presets.preset_by_name
        self.assertEqual(
            self.presets.describe(by_name("FM Broadcast")),
            "100.100 MHz · FM (broadcast), 200 kHz filter",
        )
        self.assertEqual(
            self.presets.describe(by_name("2m Calling")),
            "146.520 MHz · FM, 5 kHz deviation, 25 kHz filter",
        )
        for preset in self.presets.presets():
            mode, _dev, width = self.presets.listening(preset)
            if width:
                self.assertIn(width, bandwidth_options(mode), preset.name)

    def test_preset_at_finds_the_tuned_preset_only(self):
        at = self.presets.preset_at
        self.assertEqual(at(162.55e6).name, "NOAA Weather 1")
        self.assertEqual(at(162.5502e6).name, "NOAA Weather 1")  # within 500 Hz
        self.assertIsNone(at(162.5e6))
        # Looked for in the category shown first (2m Calling is in both).
        self.assertEqual(at(146.52e6, "Amateur").category, "Amateur")


# --------------------------------------------------------------------------- #
# In the window
# --------------------------------------------------------------------------- #


def _band_presets_menu(win):
    from PyQt6.QtGui import QAction

    return next(
        a.menu()
        for a in win.findChildren(QAction)
        if a.menu() is not None and a.text() == "Band &Presets"
    )


def _plain(text):
    return text.split("\t")[0].replace("&", "")


class TestPresetsInTheWindow(_WindowTestCase):
    def menu_item(self, name, submenu=None):
        menu = _band_presets_menu(self.win)
        if submenu is not None:
            menu = next(
                a.menu()
                for a in menu.actions()
                if a.menu() is not None and _plain(a.text()) == submenu
            )
        return next(a for a in menu.actions() if _plain(a.text()) == name)

    def test_favourites_on_top_then_a_submenu_per_category(self):
        from sdr_module.gui import presets

        menu = _band_presets_menu(self.win)
        actions = menu.actions()
        count = len(presets.FAVOURITES)
        self.assertEqual(
            [_plain(a.text()) for a in actions[:count]],
            [f.preset for f in presets.FAVOURITES],
        )
        self.assertTrue(actions[count].isSeparator())
        submenus = {
            _plain(a.text()): a.menu() for a in actions[count + 1 :] if a.menu()
        }
        self.assertEqual(list(submenus), presets.categories())
        for category, submenu in submenus.items():
            self.assertEqual(
                [_plain(a.text()) for a in submenu.actions()],
                [p.name for p in presets.presets(category)],
            )
        # Each item shows its frequency and says how it is listened to.
        item = self.menu_item("2m Calling")
        self.assertTrue(item.text().endswith("\t146.520 MHz"))
        self.assertIn("5 kHz deviation", item.toolTip())

    def test_ads_b_turns_its_decoder_on_and_shows_the_decoder_tab(self):
        win = self.win
        decoder = win._decoder_panel
        self.assertFalse(decoder._enabled_check.isChecked())
        self.menu_item("ADS-B").trigger()
        self.assertEqual(win._current_frequency(), 1090e6)
        self.assertEqual(win._control_panel.get_demod_mode(), "None (I/Q)")
        self.assertEqual(decoder._proto_combo.currentText(), "ADS-B")
        self.assertTrue(decoder._enabled_check.isChecked())
        self.assertEqual(win._current_panel_key(), "decoder")
        self.assertIn("ADS-B decoder on", win._message_label.text())

    def test_aprs_and_pocsag_select_their_decoders(self):
        win = self.win
        self.menu_item("APRS", "Amateur").trigger()
        self.assertEqual(win._decoder_panel._proto_combo.currentText(), "AX.25/APRS")
        self.assertEqual(win._control_panel.get_fm_deviation(), 5e3)
        self.menu_item("POCSAG", "Paging").trigger()
        self.assertEqual(win._decoder_panel._proto_combo.currentText(), "POCSAG")
        self.assertEqual(win._current_frequency(), 152.24e6)

    def test_the_presets_group_applies_the_same_way(self):
        win = self.win
        panel = win._control_panel
        self.assertTrue(panel.select_preset("ADS-B"))
        panel._apply_preset_btn.click()
        self.assertEqual(win._decoder_panel._proto_combo.currentText(), "ADS-B")
        self.assertEqual(win._current_panel_key(), "decoder")
        menu_text = win._message_label.text()
        win._clear_status_message()
        self.menu_item("ADS-B").trigger()
        self.assertEqual(win._message_label.text(), menu_text)

    def test_menu_and_group_say_the_same_thing(self):
        win = self.win
        self.menu_item("NOAA Weather 1").trigger()
        self.assertEqual(
            win._message_label.text(),
            "NOAA Weather 1: 162.550 MHz, FM, 25 kHz filter",
        )
        # The group shows the preset tuned now.
        panel = win._control_panel
        self.assertEqual(panel._preset_combo.currentText(), "NOAA Weather 1")

    def test_the_group_never_shows_an_unrelated_preset(self):
        win = self.win
        panel = win._control_panel
        win.set_frequency(162.55e6)
        self.assertEqual(panel._preset_combo.currentText(), "NOAA Weather 1")
        win.set_frequency(162.5e6)  # between presets
        self.assertEqual(panel._preset_combo.currentIndex(), -1)
        self.assertIn("No preset at 162.500 MHz", panel._preset_info.text())
        self.assertFalse(panel._apply_preset_btn.isEnabled())
        win.set_frequency(146.52e6)
        self.assertEqual(panel._preset_combo.currentText(), "2m Calling")

    def test_the_transmit_line_needs_a_transmitter(self):
        # ux:UX-U08: "Listen ✓ · Transmit ..." only with a HackRF connected.
        win = self.win
        panel = win._control_panel
        win._start_demo_mode()
        panel.select_preset("2m Calling")
        self.assertTrue(panel._preset_tx.isHidden())
        with mock.patch.object(type(win), "_device_can_transmit", return_value=True):
            win._refresh_state_ui()
        self.assertFalse(panel._preset_tx.isHidden())
        self.assertTrue(panel._preset_tx.text())
        win._refresh_state_ui()
        self.assertTrue(panel._preset_tx.isHidden())

    def test_the_license_profile_lives_in_the_ham_id_tab(self):
        from sdr_module.gui.main_window import HAS_HAM_RADIO

        if not HAS_HAM_RADIO:
            self.skipTest("no ham radio tools")
        win = self.win
        group = win._control_panel._license_group
        self.assertFalse(win._control_panel.isAncestorOf(group))
        self.assertTrue(win._callsign_panel.isAncestorOf(group))

    def test_the_sstv_preset_shows_the_sstv_tab(self):
        from sdr_module.gui.main_window import HAS_HAM_RADIO

        if not HAS_HAM_RADIO:
            self.skipTest("no ham radio tools")
        self.menu_item("ISS SSTV/Voice", "Space").trigger()
        self.assertEqual(self.win._current_panel_key(), "sstv")
        self.assertIn("SSTV", self.win._message_label.text())


# --------------------------------------------------------------------------- #
# Listening the way a band is heard
# --------------------------------------------------------------------------- #


class TestBandPlanListening(_WindowTestCase):
    def setUp(self):
        super().setUp()
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")  # broadcast FM
        self.win.set_frequency(100.1e6)
        self.panel = self.win._control_panel

    def type_frequency(self, text):
        spin = self.win._frequency_field()
        spin.setFocus()
        self.settle()
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin, text)
        QTest.keyClick(spin, Qt.Key.Key_Return)

    def listening(self):
        return (
            self.panel.get_demod_mode(),
            self.panel._fm_dev_combo.currentText(),
            self.panel._bw_combo.currentText(),
        )

    def test_a_typed_frequency_in_another_band_listens_its_way(self):
        self.type_frequency("145.29")
        self.assertEqual(self.win._current_frequency(), 145.29e6)
        self.assertEqual(self.listening(), ("FM", "5 kHz", "25 kHz"))
        text = self.win._message_label.text()
        self.assertTrue(text.startswith("2m Ham: listening in FM (narrow"), text)
        self.assertIn("Radio > Choose Mode From Band Plan", text)

    def test_it_can_be_turned_off_and_that_is_remembered(self):
        action = self.win._band_plan_action
        self.assertTrue(action.isChecked())  # on by default
        action.trigger()
        self.assertFalse(action.isChecked())
        self.assertIs(self.store.values.get("tuning/band_plan_mode"), False)
        self.type_frequency("145.29")
        self.assertEqual(self.listening(), ("FM", "75 kHz", "200 kHz"))

    def test_within_the_band_your_mode_stays(self):
        self.type_frequency("145.29")
        self.win._set_demod_mode("AM")  # picked by hand
        self.type_frequency("146.94")
        self.assertEqual(self.panel.get_demod_mode(), "AM")

    def test_a_bookmark_with_a_mode_is_never_overridden(self):
        self.win._on_bookmark_tune(145.29e6, "Repeater", "AM", "")
        self.assertEqual(self.panel.get_demod_mode(), "AM")

    def test_a_bookmark_without_a_mode_gets_the_band_plan(self):
        self.win._on_bookmark_tune(162.55e6, "NOAA", "", "")
        self.assertEqual(self.listening(), ("FM", "5 kHz", "25 kHz"))
        self.assertIn("NOAA Weather", self.win._message_label.text())

    def test_a_recent_frequency_without_a_mode_gets_the_band_plan(self):
        from sdr_module.gui.tuning import TunedEntry

        self.assertTrue(self.win._tune_entry(TunedEntry(118.3e6, "", "")))
        self.assertEqual(self.panel.get_demod_mode(), "AM")

    def test_morse_code_stays_cw_in_a_cw_segment(self):
        self.win.set_frequency(14.06e6)
        self.win._set_demod_mode("CW")
        self.type_frequency("7.03")
        self.assertEqual(self.panel.get_demod_mode(), "CW")

    def test_your_sideband_stays(self):
        self.win.set_frequency(14.2e6)
        self.win._set_demod_mode("USB")
        self.type_frequency("7.074")  # FT8: USB, though the band plan says LSB
        self.assertEqual(self.panel.get_demod_mode(), "USB")

    def test_lsb_switches_to_usb_where_the_band_is_usb(self):
        self.win.set_frequency(7.2e6)
        self.win._set_demod_mode("LSB")
        self.type_frequency("14.2")  # 20 m voice and data are USB
        self.assertEqual(self.panel.get_demod_mode(), "USB")
        self.assertIn("20m Ham: listening in USB", self.win._message_label.text())

    def test_presets_keep_their_own_mode(self):
        self.win.set_frequency(14.2e6)
        self.panel.select_preset("ISM 433 MHz")
        self.panel._apply_preset_btn.click()
        # The 70 cm band plan says FM; the preset says raw I/Q.
        self.assertEqual(self.panel.get_demod_mode(), "None (I/Q)")


class TestOnlyFarTunesChooseTheMode(_WindowTestCase):
    """T12 review: a step (the Frequency field's arrows, a quick-tune
    button) is not a far tune, and a preset channel typed in is heard the
    preset's way, not its band's."""

    def setUp(self):
        super().setUp()
        self.win._set_demod_mode("FM", "75 kHz", "200 kHz")  # broadcast FM
        self.win.set_frequency(100.1e6)
        self.panel = self.win._control_panel

    type_frequency = TestBandPlanListening.type_frequency
    listening = TestBandPlanListening.listening

    def quick_tune(self, text):
        button = next(b for b in self.panel._quick_tune_buttons if b.text() == text)
        button.click()

    def test_stepping_the_field_across_a_band_edge_keeps_the_mode(self):
        # The field's arrows step like Left/Right on the plots: from 107.9
        # MHz (broadcast FM) into the airband, FM stays.
        self.win.set_frequency(107.9e6)
        spin = self.win._frequency_field()
        spin.setFocus()
        self.settle()
        for _ in range(3):
            QTest.keyClick(spin, Qt.Key.Key_Up)
        self.assertEqual(self.win._current_frequency(), 108.2e6)
        self.assertEqual(self.listening(), ("FM", "75 kHz", "200 kHz"))
        self.assertNotIn("Choose Mode From Band Plan", self.win._message_label.text())

    def test_quick_tune_out_of_the_band_and_back_keeps_your_mode(self):
        self.type_frequency("147.995")  # 2 m: narrow FM
        self.win._set_demod_mode("AM")  # picked by hand
        self.quick_tune("+10k")  # 148.005 MHz: above the 2 m band
        self.quick_tune("−10k")  # back in 2 m
        self.assertEqual(self.win._current_frequency(), 147.995e6)
        self.assertEqual(self.panel.get_demod_mode(), "AM")

    def test_a_typed_cw_calling_frequency_is_heard_in_cw(self):
        # 7.030 MHz is the QRP 40m CW preset: CW, though 40 m is an LSB band.
        self.type_frequency("7.03")
        self.assertEqual(self.win._current_frequency(), 7.03e6)
        self.assertEqual(self.panel.get_demod_mode(), "CW")
        text = self.win._message_label.text()
        self.assertTrue(text.startswith("QRP 40m CW: listening in CW"), text)

    def test_a_typed_preset_channel_gets_the_presets_deviation(self):
        # 446.000 MHz is 70cm Calling (5 kHz deviation): heard the preset's
        # way, and named after it. (The band plan has no PMR446 band there
        # any more: it is the US 70 cm FM simplex segment.)
        self.type_frequency("446")
        self.assertEqual(self.listening(), ("FM", "5 kHz", "25 kHz"))
        self.assertIn("70cm Calling", self.win._message_label.text())

    def test_a_preset_channel_outside_the_bands_is_heard_its_way(self):
        # POCSAG (152.240 MHz) lies in none of the band plan's bands. (At
        # 12.5 kHz: a pager's FSK swings beyond 5 kHz, W-01.)
        self.type_frequency("152.24")
        self.assertEqual(self.listening(), ("FM", "12.5 kHz", "25 kHz"))
        self.assertIn("POCSAG", self.win._message_label.text())


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
Review fixes for the Control, Bookmarks and Decoder panels: recording state
(armed / red only while recording), frequency display and limits, copy that
matches the rest of the app, bookmark modes, keyboard focus, and the
decoder's "Tune to ..." empty state.

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_integration_panels.py``
"""

import csv
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QPoint, Qt, QTimer
    from PyQt6.QtGui import QContextMenuEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import (
        QApplication,
        QMenu,
        QMessageBox,
        QPushButton,
        QScrollArea,
    )

    HAS_PYQT6 = True
    app = QApplication.instance() or QApplication([])
except ImportError:  # pragma: no cover - environment
    HAS_PYQT6 = False


def settle(n: int = 3) -> None:
    for _ in range(n):
        QApplication.processEvents()


class _FakeSettings:
    """In-memory stand-in for GuiSettings (no QSettings on disk)."""

    def __init__(self):
        self.bookmarks = []

    def get_bookmarks(self):
        return [dict(b) for b in self.bookmarks]

    def set_bookmarks(self, bookmarks):
        self.bookmarks = [dict(b) for b in bookmarks]


def _answer_message_box(key=None, button_text=None, seen=None):
    """Answer the next visible QMessageBox with a key press or a button."""

    def answer():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, QMessageBox) and widget.isVisible():
                if seen is not None:
                    seen["default"] = widget.defaultButton().text()
                if button_text is not None:
                    for button in widget.buttons():
                        if button.text().replace("&", "") == button_text:
                            button.click()
                            return
                QTest.keyClick(widget, key)
                return
        QTimer.singleShot(20, answer)

    QTimer.singleShot(20, answer)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestControlPanelFixes(unittest.TestCase):
    def setUp(self):
        from sdr_module.core.frequency_manager import get_frequency_manager
        from sdr_module.gui.control_panel import ControlPanel

        self._fm = get_frequency_manager()
        self._license = self._fm.get_license_class()
        self.panel = ControlPanel()
        self.panel.resize(360, 900)
        self.panel.show()
        settle()

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        self._fm.set_license_class(self._license)
        settle(1)

    def select(self, category, preset):
        self.panel._category_combo.setCurrentText(category)
        self.panel._preset_combo.setCurrentText(preset)

    # -- Recording ------------------------------------------------------

    def test_record_button_is_red_only_while_recording(self):
        button = self.panel._record_btn
        self.assertFalse(button.property("role"))
        button.setChecked(True)
        self.assertEqual(button.property("role"), "danger")
        button.setChecked(False)
        self.assertFalse(button.property("role"))

    def test_armed_recording_is_shown_as_armed_not_recording(self):
        p = self.panel
        p.set_recording_state(True)
        p.set_recording_armed(True)
        self.assertTrue(p.is_recording_armed())
        self.assertIn("Armed", p._record_status.text())
        self.assertEqual(p._record_status.property("tone"), "warning")
        self.assertEqual(p._record_time.property("tone"), "muted")
        self.assertIn("receiver starts", p._record_status.toolTip())
        # The receiver starts: now it really records.
        p.set_recording_armed(False)
        self.assertIn("Recording", p._record_status.text())
        self.assertEqual(p._record_status.property("tone"), "danger")
        self.assertEqual(p._record_time.property("tone"), "danger")
        # Stopping the recording clears the armed state.
        p.set_recording_armed(True)
        p.set_recording_state(False)
        self.assertFalse(p.is_recording_armed())
        self.assertEqual(p._record_status.text(), "Ready")
        p.set_recording_state(True)
        self.assertNotIn("Armed", p._record_status.text())

    def test_armed_text_does_not_widen_the_panel(self):
        content = self.panel._content
        before = content.minimumSizeHint().width()
        self.panel.set_recording_state(True)
        self.panel.set_recording_armed(True)
        self.assertLessEqual(content.minimumSizeHint().width(), before)

    def test_recording_formats_use_the_save_dialog_names(self):
        from sdr_module.gui.control_panel import RECORDING_FORMATS

        combo = self.panel._format_combo
        names = [combo.itemText(i) for i in range(combo.count())]
        self.assertEqual(names, list(RECORDING_FORMATS))
        self.assertTrue(all("IQ" not in n.replace("I/Q", "") for n in names))
        # The main window maps the label to a file type by these tests.
        self.assertIn(".cf32", names[0])
        self.assertTrue(names[1].lower().startswith("wav"))
        self.assertIn("sigmf", names[2].lower())
        self.assertIn("not audio", combo.toolTip())

    # -- Frequency ------------------------------------------------------

    def test_frequency_shows_three_to_six_decimals_in_mhz(self):
        entry = self.panel._freq_input
        spin = entry._freq_input
        self.panel.set_frequency(100e6)
        self.assertEqual(spin.text(), "100.000")
        self.panel.set_frequency(137.9125e6)
        self.assertEqual(spin.text(), "137.9125")
        self.panel.set_frequency(145.800001e6)
        self.assertEqual(spin.text(), "145.800001")
        self.assertEqual(entry.get_frequency(), 145.800001e6)
        # A 10 kHz step keeps the 1 Hz part.
        spin.stepBy(1)
        self.assertEqual(entry.get_frequency(), 145.810001e6)
        entry._unit_combo.setCurrentText("GHz")
        self.assertEqual(spin.text(), "0.145810001")
        self.panel.set_frequency(1.09e9)
        self.assertEqual(spin.text(), "1.090")
        entry._unit_combo.setCurrentText("kHz")
        self.assertEqual(
            spin.text().replace(spin.locale().groupSeparator(), ""), "1090000.000"
        )

    def test_typed_frequency_with_few_decimals_is_accepted(self):
        entry = self.panel._freq_input
        spin = entry._freq_input
        events = []
        self.panel.frequency_changed.connect(events.append)
        spin.lineEdit().selectAll()
        QTest.keyClicks(spin, "146.52")
        QTest.keyClick(spin, Qt.Key.Key_Return)
        self.assertEqual(events[-1], 146.52e6)
        self.assertEqual(spin.text(), "146.520")

    def test_non_finite_frequencies_are_ignored(self):
        self.panel.set_frequency(145.8e6)
        for bad in (float("nan"), float("inf"), float("-inf"), "junk", None):
            self.panel.set_frequency(bad)
            self.assertEqual(self.panel._freq_input.get_frequency(), 145.8e6)
        self.assertEqual(self.panel._freq_input._freq_input.text(), "145.800")

    def test_tuning_range_matches_the_widest_supported_device(self):
        from sdr_module.gui.control_panel import MAX_FREQUENCY_HZ

        self.assertEqual(MAX_FREQUENCY_HZ, 6e9)
        self.panel.set_frequency(8e9)
        self.assertEqual(self.panel._freq_input.get_frequency(), 6e9)

    def test_quick_tune_buttons_have_spoken_names(self):
        names = [b.accessibleName() for b in self.panel._quick_tune_buttons]
        self.assertEqual(
            names,
            [
                "Tune down 1 MHz",
                "Tune down 100 kHz",
                "Tune down 10 kHz",
                "Tune up 10 kHz",
                "Tune up 100 kHz",
                "Tune up 1 MHz",
            ],
        )

    # -- Presets and license -------------------------------------------

    def test_presets_sit_directly_under_the_frequency(self):
        from PyQt6.QtWidgets import QGroupBox

        titles = [g.title() for g in self.panel.findChildren(QGroupBox)]
        self.assertEqual(titles[:3], ["Frequency", "Presets", "Demodulation"])

    def test_preset_details_use_the_mode_list_names(self):
        self.select("Broadcast", "FM Broadcast")
        info = self.panel._preset_info.text()
        self.assertIn("FM (broadcast)", info)
        self.assertNotIn("WFM", info)
        self.select("Aviation", "ADS-B")
        info = self.panel._preset_info.text()
        self.assertIn("None (I/Q)", info)
        self.assertNotIn("RAW", info)
        # Applying it selects exactly that entry.
        self.panel._apply_preset()
        self.assertEqual(self.panel.get_demod_mode(), "None (I/Q)")

    def test_tx_line_always_says_listening_works(self):
        from sdr_module.core.frequency_manager import LicenseClass

        self.panel.set_license_class(LicenseClass.NONE)
        for category, preset in (
            ("Broadcast", "FM Broadcast"),
            ("Amateur", "2m Calling"),
            ("GNSS", "GPS L1 (C/A)"),
            ("Aviation", "Air Traffic Control"),
        ):
            self.select(category, preset)
            text = self.panel._preset_tx.text()
            self.assertTrue(text.startswith("Listen ✓"), text)
            self.assertNotIn("✕", text)
            self.assertNotEqual(self.panel._preset_tx.property("tone"), "danger")
        self.select("Broadcast", "FM Broadcast")
        self.assertIn("not allowed here", self.panel._preset_tx.text())
        self.select("Amateur", "2m Calling")
        self.assertIn("needs an amateur license", self.panel._preset_tx.text())

    def test_power_limit_is_the_legal_limit_not_150_percent(self):
        from sdr_module.core.frequency_manager import LicenseClass
        from sdr_module.gui.control_panel import _power_limit_tip

        self.panel.set_license_class(LicenseClass.AMATEUR_EXTRA)
        self.select("Amateur", "2m Calling")
        text = self.panel._preset_tx.text()
        self.assertIn("Transmit ✓", text)
        self.assertNotIn("headroom", text)
        if "legal limit" in text:
            self.assertIn("transmitter output", self.panel._preset_tx.toolTip())
        license_text = self.panel._license_info.text()
        self.assertNotIn("headroom", license_text)
        self.assertNotIn("%", license_text)
        # Sub-watt limits are not rounded down to "0 W".
        tip = _power_limit_tip(0.5, None)
        self.assertIn("0.5 W", tip)
        self.assertIn("0.75 W", tip)

    def test_demod_settings_getters(self):
        self.panel._demod_combo.setCurrentText("FM")
        self.panel._fm_dev_combo.setCurrentText("75 kHz")
        self.assertEqual(self.panel.get_demod_mode(), "FM")
        self.assertEqual(self.panel.get_fm_deviation_text(), "75 kHz")


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestBookmarksPanelFixes(unittest.TestCase):
    def setUp(self):
        from sdr_module.gui import bookmarks_panel

        self.mod = bookmarks_panel
        self.store = _FakeSettings()
        self._real_settings = bookmarks_panel.GuiSettings
        bookmarks_panel.GuiSettings = lambda: self.store
        self.panel = bookmarks_panel.BookmarksPanel()
        self.panel.resize(380, 360)
        self.panel.show()
        self.tuned = []
        self.panel.tune_requested.connect(lambda *args: self.tuned.append(args))
        settle()

    def tearDown(self):
        self.mod.GuiSettings = self._real_settings
        self.panel.close()
        self.panel.deleteLater()
        settle(1)

    def test_csv_menu_uses_the_file_menu_names(self):
        texts = [a.text() for a in self.panel._csv_menu.actions()]
        self.assertEqual(
            texts,
            ["Import Channels (CHIRP CSV)...", "Export Channels (CHIRP CSV)..."],
        )

    def test_add_field_covers_the_whole_tuning_range(self):
        from sdr_module.gui.control_panel import MAX_FREQUENCY_HZ, MIN_FREQUENCY_HZ

        spin = self.panel._freq_input
        self.assertAlmostEqual(spin.minimum() * 1e6, MIN_FREQUENCY_HZ)
        self.assertAlmostEqual(spin.maximum() * 1e6, MAX_FREQUENCY_HZ)
        self.panel.set_current_frequency(5.8e9)
        self.panel._add_btn.click()
        self.assertEqual(self.store.bookmarks[-1]["freq_hz"], 5.8e9)
        self.assertEqual(self.store.bookmarks[-1]["label"], "5800.000 MHz")

    def test_non_finite_current_frequency_is_ignored(self):
        self.panel.set_current_frequency(146.52e6)
        self.panel.set_current_frequency(float("nan"))
        self.panel.set_current_frequency(float("inf"))
        self.assertAlmostEqual(self.panel._freq_input.value(), 146.52)

    def test_bookmark_remembers_and_restores_its_mode(self):
        self.panel.add_bookmark("Tower", 118.3e6, "AM")
        self.panel.add_bookmark("Broadcast", 100.1e6, "FM", "75 kHz")
        self.panel.add_bookmark("Raw", 433.92e6, "None (I/Q)")
        self.panel.add_bookmark("Old style", 162.55e6)
        stored = self.store.bookmarks
        self.assertEqual(
            stored[0],
            {"label": "Tower", "freq_hz": 118.3e6, "demod": "AM", "mode": "AM"},
        )
        self.assertEqual(stored[1]["mode"], "WFM")
        self.assertEqual(stored[1]["fm_deviation"], "75 kHz")
        self.assertNotIn("mode", stored[2])
        self.assertEqual(stored[3], {"label": "Old style", "freq_hz": 162.55e6})
        column = [
            self.panel._list.topLevelItem(i).text(self.mod.COL_MODE) for i in range(4)
        ]
        self.assertEqual(column, ["AM", "WFM", "I/Q", ""])
        for i in range(4):
            self.panel._tune_index(i)
        self.assertEqual(
            self.tuned,
            [
                (118.3e6, "Tower", "AM", ""),
                (100.1e6, "Broadcast", "FM", "75 kHz"),
                (433.92e6, "Raw", "None (I/Q)", ""),
                (162.55e6, "Old style", "", ""),
            ],
        )

    def test_chirp_modes_map_to_the_mode_list(self):
        tune_mode = self.mod.tune_mode
        chirp = {"label": "x", "freq_hz": 146.52e6, "location": 1}
        self.assertEqual(tune_mode(chirp), ("FM", "5 kHz"))  # CHIRP default FM
        cases = {
            "NFM": ("FM", "5 kHz"),
            "WFM": ("FM", "75 kHz"),
            "AM": ("AM", ""),
            "USB": ("USB", ""),
            "LSB": ("LSB", ""),
            "CWR": ("CW", ""),
            "RTTY": ("", ""),
            "Auto": ("", ""),
        }
        for mode, expected in cases.items():
            self.assertEqual(tune_mode(dict(chirp, mode=mode)), expected, mode)

    def test_imported_channel_tunes_in_its_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "air.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("Name,Frequency,Mode\nTower,118.300000,AM\n")
            self.assertEqual(self.panel.import_csv(path), 1)
        self.panel._list.setCurrentItem(self.panel._list.topLevelItem(0))
        QTest.keyClick(self.panel._list, Qt.Key.Key_Return)
        self.assertEqual(self.tuned[-1], (118.3e6, "Tower", "AM", ""))

    def test_add_saves_the_mode_only_for_the_tuned_frequency(self):
        self.panel.set_mode_source(lambda: ("USB", ""))
        self.panel.set_current_frequency(14.2e6)
        self.panel._add_btn.click()
        self.assertEqual(self.store.bookmarks[-1]["demod"], "USB")
        # A typed-in frequency: how it is listened to is unknown.
        self.panel._freq_input.setValue(7.1)
        self.panel._add_btn.click()
        self.assertNotIn("demod", self.store.bookmarks[-1])

    def test_exported_bookmark_keeps_its_mode(self):
        self.panel.add_bookmark("Tower", 118.3e6, "AM")
        self.panel.add_bookmark("Raw", 433.92e6, "None (I/Q)")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            real_info = self.mod.QMessageBox.information
            self.mod.QMessageBox.information = staticmethod(lambda *a, **k: None)
            try:
                self.assertEqual(self.panel.export_csv(path), 2)
            finally:
                self.mod.QMessageBox.information = real_info
            with open(path, encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["Mode"], "AM")
        self.assertEqual(rows[1]["Mode"], "FM")  # CHIRP has no I/Q mode

    def test_two_argument_slots_still_work(self):
        seen = []
        self.panel.tune_requested.connect(lambda f, label: seen.append((f, label)))
        self.panel.add_bookmark("Tower", 118.3e6, "AM")
        self.panel._tune_index(0)
        self.assertEqual(seen, [(118.3e6, "Tower")])

    def test_enter_in_the_remove_confirmation_cancels(self):
        self.panel.add_bookmark("Keep me", 146.52e6)
        seen = {}
        _answer_message_box(key=Qt.Key.Key_Return, seen=seen)
        self.panel._list.setCurrentItem(self.panel._list.topLevelItem(0))
        QTest.keyClick(self.panel._list, Qt.Key.Key_Backspace)
        self.assertEqual(seen["default"], "Cancel")
        self.assertEqual(self.panel.bookmark_count(), 1)
        # The Remove button still removes, and no dialog is left behind.
        _answer_message_box(button_text="Remove")
        QTest.keyClick(self.panel._list, Qt.Key.Key_Delete)
        self.assertEqual(self.panel.bookmark_count(), 0)
        settle()
        from PyQt6.QtCore import QCoreApplication, QEvent

        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        self.assertEqual(self.panel.findChildren(QMessageBox), [])


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestDecoderPanelFixes(unittest.TestCase):
    def setUp(self):
        from sdr_module.gui import decoder_panel

        self.mod = decoder_panel
        self.panel = decoder_panel.DecoderPanel()
        self.panel.resize(420, 420)
        self.panel.show()
        settle()

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        settle(1)

    def connect_tuning(self):
        tuned = []
        self.panel.tune_requested.connect(lambda *args: tuned.append(args))
        return tuned

    def test_empty_states_use_the_apps_own_verbs(self):
        p = self.panel
        texts = [p._empty.text()]
        for name in ("POCSAG", "ADS-B", "ACARS", "AX.25/APRS"):
            p._proto_combo.setCurrentText(name)
            texts.append(p._empty.text())
        for text in texts:
            self.assertNotIn("acquisition", text.lower())
            self.assertIn("Start", text)
        self.assertNotIn("Hover", p._proto_combo.toolTip())

    def test_decode_box_is_unticked_while_off_and_keeps_the_choice(self):
        p = self.panel
        check = p._enabled_check
        self.assertFalse(check.isChecked())
        self.assertFalse(check.isEnabled())
        p._proto_combo.setCurrentText("POCSAG")
        self.assertTrue(check.isChecked())
        check.setChecked(False)
        p._proto_combo.setCurrentText("Off")
        self.assertFalse(check.isChecked())
        p._proto_combo.setCurrentText("FLEX")
        self.assertFalse(check.isChecked())  # the user's "paused" is kept
        self.assertFalse(p.is_decoding())
        check.setChecked(True)
        p._proto_combo.setCurrentText("Off")
        p._proto_combo.setCurrentText("ADS-B")
        self.assertTrue(check.isChecked())
        self.assertTrue(p.is_decoding())

    def test_adsb_reply_with_only_an_address_says_so(self):
        self.panel.add_adsb_message("47BBDC")
        item = self.panel._table.item(0, self.mod.COL_MESSAGE)
        self.assertEqual(item.text(), "(aircraft address only)")

    def test_tune_button_needs_a_listener(self):
        self.panel._proto_combo.setCurrentText("ADS-B")
        self.assertEqual(self.panel._empty.action_text(), "")
        self.assertTrue(self.panel._empty.button.isHidden())

    def test_tune_button_tunes_to_the_protocol_channel(self):
        tuned = self.connect_tuning()
        p = self.panel
        p._proto_combo.setCurrentText("ADS-B")
        button = p._empty.button
        self.assertEqual(p._empty.action_text(), "Tune to 1090.000 MHz")
        self.assertTrue(button.isVisible())
        # Laid out under the message, inside the table.
        self.assertGreaterEqual(
            button.geometry().top(), p._empty.label.geometry().bottom()
        )
        self.assertTrue(p._table.viewport().rect().contains(button.geometry()))
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        self.assertEqual(tuned, [(1090e6, "ADS-B (1090.000 MHz)", "", "")])
        p._proto_combo.setCurrentText("AX.25/APRS")
        p._empty.button.click()
        self.assertEqual(tuned[-1][0], 144.39e6)
        self.assertEqual(tuned[-1][2:], ("FM", "5 kHz"))
        # Pagers use many channels, so there is no single one to offer.
        p._proto_combo.setCurrentText("POCSAG")
        self.assertEqual(p._empty.action_text(), "")

    def test_hint_follows_the_receiver(self):
        self.connect_tuning()
        p = self.panel
        p._proto_combo.setCurrentText("ADS-B")
        p.set_current_frequency(1090e6)
        p.set_receiving(False)
        self.assertIn("Tuned to 1090.000 MHz", p._empty.text())
        self.assertIn("press Start", p._empty.text())
        self.assertEqual(p._empty.action_text(), "")
        p.set_receiving(True)
        self.assertIn("Listening for ADS-B on 1090.000 MHz", p._empty.text())
        p.set_current_frequency(100e6)
        self.assertIn("Tune to 1090 MHz to decode", p._empty.text())
        self.assertEqual(p._empty.action_text(), "Tune to 1090.000 MHz")
        p.set_current_frequency(float("nan"))
        self.assertEqual(p._tuned_hz, 100e6)
        # The button disappears with the empty state.
        p.add_message("ADS-B", "ABC123", "hello")
        self.assertTrue(p._empty.button.isHidden())
        p.clear()
        self.assertTrue(p._empty.button.isVisible())

    def test_rds_is_listed_but_not_selectable(self):
        combo = self.panel._proto_combo
        index = combo.findText("RDS")
        self.assertGreater(index, 0)
        self.assertFalse(combo.model().item(index).isEnabled())
        self.assertIn(
            "isn't available", combo.itemData(index, Qt.ItemDataRole.ToolTipRole)
        )

    def test_tab_leaves_the_message_table(self):
        p = self.panel
        table = p._table
        self.assertFalse(table.tabKeyNavigation())
        self.assertEqual(table.accessibleName(), "Decoded messages")
        stats = [a for a in p.findChildren(QScrollArea)]
        self.assertTrue(stats)
        for area in stats:
            self.assertEqual(area.focusPolicy(), Qt.FocusPolicy.NoFocus)
        p._proto_combo.setCurrentText("POCSAG")
        for i in range(3):
            p.add_message("POCSAG", str(i), f"msg {i}")
        p.activateWindow()
        table.setFocus()
        table.setCurrentCell(0, 0)
        settle()
        if QApplication.focusWidget() is not table:
            self.skipTest("no keyboard focus on this platform")
        QTest.keyClick(table, Qt.Key.Key_Tab)
        settle()
        self.assertIsNot(QApplication.focusWidget(), table)

    def test_clear_and_export_are_regular_push_buttons(self):
        self.assertIsInstance(self.panel._clear_btn, QPushButton)
        self.assertIsInstance(self.panel._export_btn, QPushButton)
        self.assertFalse(self.panel._tabs.tabBar().drawBase())

    def test_context_menu_does_not_leak(self):
        from PyQt6.QtCore import QCoreApplication, QEvent

        table = self.panel._table
        for _ in range(3):

            def close_menu():
                for widget in QApplication.topLevelWidgets():
                    if isinstance(widget, QMenu) and widget.isVisible():
                        widget.close()
                        return
                QTimer.singleShot(10, close_menu)

            QTimer.singleShot(10, close_menu)
            event = QContextMenuEvent(
                QContextMenuEvent.Reason.Mouse,
                QPoint(5, 5),
                table.mapToGlobal(QPoint(5, 5)),
            )
            table.contextMenuEvent(event)
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        self.assertEqual(table.findChildren(QMenu), [])


if __name__ == "__main__":
    unittest.main()

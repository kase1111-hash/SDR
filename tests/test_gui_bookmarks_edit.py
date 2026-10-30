#!/usr/bin/env python3
"""
Bookmarks you can manage (T8): a mode on Add, the Edit dialog, the
right-click menu, the filter, groups, no duplicates (Ctrl+B, imports), names
from the band ("FM Broadcast 100.1"), the built-in presets as CHIRP CSV, and
a bookmark's filter and CTCSS tone squelch when it is tuned.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_bookmarks_edit.py``
"""

import csv
import os
import re
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _FakeSettings,
    _WindowTestCase,
    require_pyqt6,
)

if HAS_PYQT6:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication


def _settle(n=3):
    for _ in range(n):
        QApplication.processEvents()


class _PanelCase(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import bookmarks_panel

        self.mod = bookmarks_panel
        self.store = _FakeSettings()
        self._real_settings = bookmarks_panel.GuiSettings
        bookmarks_panel.GuiSettings = lambda: self.store
        self._real_info = bookmarks_panel.QMessageBox.information
        self.infos = []
        bookmarks_panel.QMessageBox.information = staticmethod(
            lambda *args, **_k: self.infos.append(args[2] if len(args) > 2 else "")
        )
        self._panels = []

    def tearDown(self):
        self.mod.GuiSettings = self._real_settings
        self.mod.QMessageBox.information = self._real_info
        for panel in self._panels:
            panel.close()
            panel.deleteLater()
        _settle(1)

    def make(self, bookmarks=()):
        self.store.bookmarks = [dict(b) for b in bookmarks]
        panel = self.mod.BookmarksPanel()
        panel.resize(380, 420)
        panel.show()
        _settle()
        self._panels.append(panel)
        self.messages = []
        panel.message.connect(lambda text, tone: self.messages.append((text, tone)))
        return panel

    def rows(self, panel):
        """Visible rows as (depth, name), in display order."""
        out = []

        def walk(item, depth):
            if not item.isHidden():
                out.append((depth, item.text(0)))
            for i in range(item.childCount()):
                walk(item.child(i), depth + 1)

        for i in range(panel._list.topLevelItemCount()):
            walk(panel._list.topLevelItem(i), 0)
        return out


class TestAddRow(_PanelCase):
    def test_mode_follows_the_receiver_and_the_band(self):
        panel = self.make()
        mode = ["FM", "75 kHz"]
        panel.set_mode_source(lambda: tuple(mode))
        panel.set_width_source(lambda: "200 kHz")
        panel.set_current_frequency(100.1e6)
        self.assertEqual(panel._mode_input.currentData(), "FM")
        mode[:] = ["AM", ""]
        panel.refresh_mode()
        self.assertEqual(panel._mode_input.currentData(), "AM")
        # A frequency typed in: its band's usual mode (airband: AM).
        mode[:] = ["FM", "75 kHz"]
        panel._freq_input.setValue(118.1)
        self.assertEqual(panel._mode_input.currentData(), "AM")
        panel._add_btn.click()
        saved = self.store.bookmarks[-1]
        self.assertEqual(saved["label"], "Airband 118.1")
        self.assertEqual((saved["demod"], saved["bandwidth"]), ("AM", "10 kHz"))
        # The tuned frequency: the receiver's mode, deviation and filter.
        panel.set_current_frequency(100.1e6)
        panel._add_btn.click()
        saved = self.store.bookmarks[-1]
        self.assertEqual(saved["label"], "FM Broadcast 100.1")
        self.assertEqual(
            (saved["demod"], saved["fm_deviation"], saved["bandwidth"]),
            ("FM", "75 kHz", "200 kHz"),
        )

    def test_a_frequency_typed_in_the_same_band_keeps_the_current_mode(self):
        panel = self.make()
        panel.set_mode_source(lambda: ("USB", ""))
        panel.set_current_frequency(144.3e6)  # SSB on 2 m
        panel._freq_input.setValue(144.25)
        self.assertEqual(panel._mode_input.currentData(), "USB")
        panel._freq_input.setValue(125.0)  # airband: its own mode
        self.assertEqual(panel._mode_input.currentData(), "AM")

    def test_context_menu_rename_and_remove_act_on_the_clicked_bookmark(self):
        panel = self.make(
            [{"label": "A", "freq_hz": 100e6}, {"label": "B", "freq_hz": 101e6}]
        )
        panel._list.setCurrentItem(panel._list.topLevelItem(0))  # A selected
        menu = panel._context_menu(1)  # ...but B right-clicked
        with mock.patch.object(
            self.mod.QInputDialog, "getText", return_value=("Bee", True)
        ):
            next(a for a in menu.actions() if a.text().startswith("Rename")).trigger()
        self.assertEqual([b["label"] for b in self.store.bookmarks], ["A", "Bee"])
        panel._confirm_remove = lambda _b: True
        next(a for a in menu.actions() if a.text().startswith("Remove")).trigger()
        self.assertEqual([b["label"] for b in self.store.bookmarks], ["A"])
        menu.deleteLater()

    def test_a_mode_picked_by_hand_is_saved(self):
        panel = self.make()
        panel.set_mode_source(lambda: ("FM", "75 kHz"))
        panel.set_current_frequency(100.1e6)
        index = panel._mode_input.findData("USB")
        panel._mode_input.setCurrentIndex(index)
        panel._mode_input.activated.emit(index)
        panel._label_input.setText("Oddity")
        panel._add_btn.click()
        saved = self.store.bookmarks[-1]
        self.assertEqual((saved["label"], saved["demod"]), ("Oddity", "USB"))
        self.assertNotIn("fm_deviation", saved)
        # The Mode follows the receiver again after Add.
        self.assertEqual(panel._mode_input.currentData(), "FM")

    def test_add_skips_a_duplicate_and_says_so(self):
        panel = self.make([{"label": "Tower", "freq_hz": 118.1e6, "demod": "AM"}])
        panel._freq_input.setValue(118.1)
        panel._add_btn.click()
        self.assertEqual(panel.bookmark_count(), 1)
        self.assertEqual(self.messages[-1], ("Already bookmarked: Tower", "info"))
        self.assertEqual(panel._selected_index(), 0)

    def test_same_channel_rules(self):
        same = self.mod.same_channel
        tower = {"label": "T", "freq_hz": 118.1e6, "demod": "AM"}
        self.assertTrue(same(tower, 118.1e6, "AM"))
        self.assertTrue(same(tower, 118.100005e6, ""))
        self.assertFalse(same(tower, 118.1e6, "FM"))
        self.assertFalse(same(tower, 118.125e6, "AM"))
        keeps = {"label": "Old", "freq_hz": 100.1e6}
        self.assertTrue(same(keeps, 100.1e6, "FM", "75 kHz"))
        fm = {"label": "S", "freq_hz": 146.52e6, "demod": "FM", "fm_deviation": "5 kHz"}
        self.assertFalse(same(fm, 146.52e6, "FM", "12.5 kHz"))


class TestEditDialog(_PanelCase):
    def test_every_field_is_saved(self):
        from sdr_module.gui.bookmark_editor import BookmarkEditDialog

        original = {
            "label": "Rptr",
            "freq_hz": 146.94e6,
            "demod": "FM",
            "fm_deviation": "5 kHz",
        }
        dialog = BookmarkEditDialog(original, ("FM", "5 kHz"), ["Club"])
        dialog._name.setText("Club repeater")
        dialog._freq.setValue(145.29)
        dialog._deviation.setCurrentText("2.5 kHz")
        dialog._width.setCurrentIndex(dialog._width.findData("12.5 kHz"))
        dialog._tone.setCurrentIndex(dialog._tone.findData(100.0))
        dialog._group.setCurrentText("Club")
        dialog._skip.setChecked(True)
        dialog._comment.setText("Tuesday net")
        edited = dialog.bookmark()
        self.assertEqual(
            edited,
            {
                "label": "Club repeater",
                "freq_hz": 145.29e6,
                "demod": "FM",
                "fm_deviation": "2.5 kHz",
                "mode": "NFM",
                "bandwidth": "12.5 kHz",
                "tone_mode": "TSQL",
                "ctone": 100.0,
                "group": "Club",
                "skip": "S",
                "comment": "Tuesday net",
            },
        )
        dialog.deleteLater()

    def test_modes_shape_the_other_fields(self):
        from sdr_module.gui.bookmark_editor import BookmarkEditDialog

        dialog = BookmarkEditDialog({"label": "x", "freq_hz": 118.1e6}, ("", ""))
        self.assertEqual(dialog.demod(), "")
        self.assertFalse(dialog._width.isEnabled())
        self.assertIn("keeps the mode", dialog._hint.text())
        dialog._mode.setCurrentIndex(dialog._mode.findData("AM"))
        self.assertFalse(dialog._deviation.isEnabled())
        self.assertFalse(dialog._tone.isEnabled())
        widths = [dialog._width.itemText(i) for i in range(dialog._width.count())]
        self.assertIn("10 kHz", widths)
        self.assertNotIn("200 kHz", widths)
        edited = dialog.bookmark()
        self.assertEqual((edited["demod"], edited["mode"]), ("AM", "AM"))
        self.assertNotIn("bandwidth", edited)  # the mode's usual width
        # Keep the receiver's mode: a CHIRP channel gets CHIRP's "Auto".
        chirp = BookmarkEditDialog(
            {"label": "C", "freq_hz": 146.52e6, "location": 3, "mode": "NFM"},
            ("FM", "5 kHz"),
        )
        chirp._mode.setCurrentIndex(0)
        self.assertEqual(chirp.bookmark()["mode"], "Auto")
        self.assertEqual(self.mod.tune_mode(chirp.bookmark()), ("", ""))
        dialog.deleteLater()
        chirp.deleteLater()

    def test_repeater_shift(self):
        from sdr_module.gui.bookmark_editor import BookmarkEditDialog

        dialog = BookmarkEditDialog(
            {"label": "R", "freq_hz": 146.94e6, "duplex": "-", "offset_hz": 600e3},
            ("FM", "5 kHz"),
        )
        self.assertEqual(dialog._duplex.currentData(), "-")
        self.assertAlmostEqual(dialog._offset.value(), 0.6)
        self.assertTrue(dialog._offset.isEnabled())
        dialog._duplex.setCurrentIndex(dialog._duplex.findData("+"))
        dialog._offset.setValue(0.5)
        edited = dialog.bookmark()
        self.assertEqual((edited["duplex"], edited["offset_hz"]), ("+", 500e3))
        dialog._duplex.setCurrentIndex(0)
        self.assertFalse(dialog._offset.isEnabled())
        # Every label's mnemonic works and none is used twice.
        from PyQt6.QtWidgets import QLabel

        keys = {}
        for label in dialog.findChildren(QLabel):
            match = re.search(r"&([^&])", label.text())
            if match:
                self.assertIsNotNone(label.buddy(), label.text())
                keys.setdefault(match.group(1).lower(), []).append(label.text())
        self.assertEqual({k: v for k, v in keys.items() if len(v) > 1}, {})
        self.assertNotIn("duplex", dialog.bookmark())
        # A split channel (TX elsewhere) is kept as it is.
        split = BookmarkEditDialog(
            {"label": "S", "freq_hz": 145.1e6, "duplex": "split", "offset_hz": 146.7e6},
            ("FM", "5 kHz"),
        )
        self.assertEqual(split.bookmark()["duplex"], "split")
        self.assertEqual(split.bookmark()["offset_hz"], 146.7e6)
        dialog.deleteLater()
        split.deleteLater()

    def test_clearing_the_tone_and_other_fields(self):
        from sdr_module.gui.bookmark_editor import BookmarkEditDialog

        original = {
            "label": "R",
            "freq_hz": 145.29e6,
            "demod": "FM",
            "fm_deviation": "5 kHz",
            "tone_mode": "TSQL",
            "ctone": 100.0,
            "group": "G",
            "skip": "S",
            "comment": "c",
        }
        dialog = BookmarkEditDialog(original, ("FM", "5 kHz"), ["G"])
        self.assertEqual(dialog._tone.currentData(), 100.0)
        dialog._tone.setCurrentIndex(0)
        dialog._group.setCurrentText("")
        dialog._skip.setChecked(False)
        dialog._comment.clear()
        dialog._name.clear()
        edited = dialog.bookmark()
        for key in ("tone_mode", "ctone", "group", "skip", "comment"):
            self.assertNotIn(key, edited)
        self.assertEqual(edited["label"], "145.290 MHz")
        dialog.deleteLater()

    def test_edit_button_saves_through_the_dialog(self):
        panel = self.make([{"label": "A", "freq_hz": 146.52e6, "demod": "FM"}])
        panel._list.setCurrentItem(panel._list.topLevelItem(0))
        self.assertTrue(panel._edit_btn.isEnabled())
        self.assertIs(panel._rename_btn, panel._edit_btn)
        edited = {"label": "B", "freq_hz": 146.55e6, "demod": "FM", "group": "Ham"}
        with mock.patch.object(self.mod, "_edit_dialog", return_value=edited) as dialog:
            panel._edit_btn.click()
        args = dialog.call_args[0]
        self.assertEqual(args[2], ("FM", ""))  # how it is listened to now
        self.assertEqual(self.store.bookmarks, [edited])
        self.assertEqual(self.rows(panel), [(0, "Ham"), (1, "B")])
        with mock.patch.object(self.mod, "_edit_dialog", return_value=None):
            self.assertFalse(panel.edit_bookmark(0))  # cancelled
        self.assertEqual(self.store.bookmarks, [edited])


class TestListTools(_PanelCase):
    SAMPLE = [
        {"label": "Tower", "freq_hz": 118.1e6, "demod": "AM", "group": "Airport"},
        {"label": "Calling", "freq_hz": 146.52e6, "demod": "FM"},
        {"label": "Ground", "freq_hz": 121.9e6, "demod": "AM", "group": "Airport"},
        {"label": "Wx", "freq_hz": 162.55e6, "demod": "FM", "comment": "weather"},
    ]

    def test_groups_are_folders(self):
        panel = self.make(self.SAMPLE)
        self.assertEqual(
            self.rows(panel),
            [(0, "Airport"), (1, "Tower"), (1, "Ground"), (0, "Calling"), (0, "Wx")],
        )
        folder = panel._list.topLevelItem(0)
        self.assertEqual(folder.text(1), "2 bookmarks")
        self.assertFalse(folder.flags() & Qt.ItemFlag.ItemIsSelectable)
        # A folder is not a bookmark: nothing to tune or remove.
        panel._list.setCurrentItem(folder)
        self.assertEqual(panel._selected_index(), -1)
        self.assertFalse(panel._tune_btn.isEnabled())
        # Rows map to their bookmark wherever they are shown.
        panel._list.setCurrentItem(folder.child(1))
        self.assertEqual(panel._selected_index(), 2)
        self.assertEqual(panel.groups(), ["Airport"])
        # Collapsed groups stay collapsed when the list is rebuilt.
        folder.setExpanded(False)
        panel.add_bookmark("New", 100e6)
        self.assertFalse(panel._list.topLevelItem(0).isExpanded())

    def test_filter(self):
        panel = self.make(self.SAMPLE)
        panel._filter.setText("air")  # the group
        self.assertEqual(
            self.rows(panel), [(0, "Airport"), (1, "Tower"), (1, "Ground")]
        )
        panel._filter.setText("146.52")
        self.assertEqual(self.rows(panel), [(0, "Calling")])
        panel._filter.setText("weather")  # the comment
        self.assertEqual(self.rows(panel), [(0, "Wx")])
        panel._filter.setText("am tow")  # every word
        self.assertEqual(self.rows(panel), [(0, "Airport"), (1, "Tower")])
        panel._filter.setText("nothing like it")
        self.assertEqual(self.rows(panel), [])
        self.assertTrue(panel._empty.is_visible())
        self.assertIn("No bookmarks match", panel._empty.text())
        self.assertIn("all 4 bookmarks", panel._empty.text())
        panel._filter.setFocus()
        QTest.keyClick(panel._filter, Qt.Key.Key_Escape)
        self.assertEqual(panel._filter.text(), "")
        self.assertEqual(len(self.rows(panel)), 5)

    def test_selecting_a_hidden_bookmark_clears_the_filter(self):
        panel = self.make(self.SAMPLE)
        panel._filter.setText("Tower")
        panel.select_bookmark(3)
        self.assertEqual(panel._filter.text(), "")
        self.assertEqual(panel._selected_index(), 3)

    def test_context_menu(self):
        panel = self.make(self.SAMPLE)
        scans = []
        panel.scan_requested.connect(scans.append)
        menu = panel._context_menu(1)
        texts = [a.text() for a in menu.actions() if not a.isSeparator()]
        self.assertEqual(
            texts,
            [
                "Tune to 146.520 MHz",
                "Edit...",
                "Rename...\tF2",
                "Copy Frequency",
                "Move Up",
                "Move Down",
                "Scan Bookmarks from Here",
                "Remove...\tDelete",
            ],
        )
        tuned = []
        panel.tune_requested.connect(lambda *args: tuned.append(args))
        menu.actions()[0].trigger()
        self.assertEqual(tuned, [(146.52e6, "Calling", "FM", "")])
        copy = next(a for a in menu.actions() if a.text() == "Copy Frequency")
        copy.trigger()
        self.assertEqual(QGuiApplication.clipboard().text(), "146.520 MHz")
        self.assertEqual(self.messages[-1], ("Copied 146.520 MHz", "info"))
        scan = next(a for a in menu.actions() if a.text() == "Scan Bookmarks from Here")
        scan.trigger()
        self.assertEqual(scans, [1])
        menu.deleteLater()

    def test_move_up_and_down_within_the_group(self):
        panel = self.make(self.SAMPLE)
        menu = panel._context_menu(0)  # Tower: first of Airport
        up = next(a for a in menu.actions() if a.text() == "Move Up")
        self.assertFalse(up.isEnabled())
        menu.deleteLater()
        self.assertTrue(panel.move_bookmark(0, 1))  # past Calling, to Ground
        self.assertEqual(
            [b["label"] for b in self.store.bookmarks],
            ["Ground", "Calling", "Tower", "Wx"],
        )
        self.assertEqual(panel._selected_index(), 2)
        self.assertEqual(
            self.rows(panel),
            [(0, "Airport"), (1, "Ground"), (1, "Tower"), (0, "Calling"), (0, "Wx")],
        )
        self.assertFalse(panel.move_bookmark(3, 1))  # last of the list
        self.assertTrue(panel.move_bookmark(3, -1))
        self.assertEqual(self.store.bookmarks[1]["label"], "Wx")

    def test_scan_from_here_needs_the_scanner(self):
        panel = self.make(self.SAMPLE)
        menu = panel._context_menu(0)
        scan = next(a for a in menu.actions() if a.text() == "Scan Bookmarks from Here")
        self.assertFalse(scan.isEnabled())  # nothing connected
        menu.deleteLater()

    def test_tuning_sends_the_whole_bookmark_too(self):
        panel = self.make(
            [
                {
                    "label": "R",
                    "freq_hz": 145.29e6,
                    "demod": "FM",
                    "fm_deviation": "5 kHz",
                    "bandwidth": "12.5 kHz",
                    "tone_mode": "TSQL",
                    "ctone": 100.0,
                }
            ]
        )
        order = []
        panel.tune_requested.connect(lambda *a: order.append(("tune", a)))
        panel.bookmark_tuned.connect(lambda b: order.append(("details", b)))
        panel._tune_index(0)
        self.assertEqual(order[0], ("tune", (145.29e6, "R", "FM", "5 kHz")))
        self.assertEqual(order[1][0], "details")
        self.assertEqual(
            self.mod.listening(order[1][1]), ("FM", "5 kHz", "12.5 kHz", 100.0)
        )

    def test_skipped_bookmarks_say_so_in_the_list(self):
        panel = self.make(
            [
                {"label": "A", "freq_hz": 118.1e6, "demod": "AM", "skip": "S"},
                {"label": "B", "freq_hz": 125.0e6, "skip": "S"},
            ]
        )
        self.assertEqual(
            panel._list.topLevelItem(0).text(self.mod.COL_MODE), "AM · Skip"
        )
        self.assertEqual(panel._list.topLevelItem(1).text(self.mod.COL_MODE), "Skip")
        panel._filter.setText("skip")
        self.assertEqual(len(self.rows(panel)), 2)

    def test_list_shows_filter_tone_group_and_skip_in_the_tooltip(self):
        panel = self.make(
            [
                {
                    "label": "R",
                    "freq_hz": 145.29e6,
                    "demod": "FM",
                    "bandwidth": "12.5 kHz",
                    "tone_mode": "TSQL",
                    "ctone": 100.0,
                    "group": "Club",
                    "skip": "S",
                }
            ]
        )
        tip = panel._list.topLevelItem(0).child(0).toolTip(0)
        for text in (
            "Filter: 12.5 kHz",
            "TSQL: 100 Hz",
            "Group: Club",
            "Skipped by Scan",
        ):
            self.assertIn(text, tip)


class TestCsvTools(_PanelCase):
    def test_import_append_skips_bookmarks_already_there(self):
        panel = self.make([{"label": "Tower", "freq_hz": 118.3e6, "demod": "AM"}])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "air.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(
                    "Name,Frequency,Mode\nTWR,118.300000,AM\nGND,121.900000,AM\n"
                )
            with mock.patch.object(
                panel, "_ask_replace_or_append", return_value="append"
            ):
                self.assertEqual(panel.import_csv(path), 1)
                self.assertEqual(panel.import_csv(path), 0)  # again: nothing new
        self.assertEqual([b["label"] for b in self.store.bookmarks], ["Tower", "GND"])
        self.assertIn("Already bookmarked, so skipped: TWR", self.infos[0])
        self.assertIn("Nothing new to import", self.infos[1])

    def test_semicolon_files_import(self):
        panel = self.make()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "excel.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("Location;Name;Frequency;Mode\n0;Calling;146,520000;FM\n")
            self.assertEqual(panel.import_csv(path), 1)
        self.assertEqual(self.store.bookmarks[0]["freq_hz"], 146.52e6)

    def test_round_trip_keeps_how_each_is_listened_to(self):
        panel = self.make()
        panel.add_bookmark("Tower", 118.1e6, "AM", "", "10 kHz")
        panel.add_bookmark("2m simplex", 146.52e6, "FM", "12.5 kHz")
        panel.add_bookmark("Typed long ago", 433.92e6)
        before = [self.mod.listening(b) for b in self.store.bookmarks]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.csv")
            self.assertEqual(panel.export_csv(path), 3)
            with mock.patch.object(
                panel, "_ask_replace_or_append", return_value="replace"
            ):
                self.assertEqual(panel.import_csv(path), 3)
        after = [self.mod.listening(b) for b in self.store.bookmarks]
        self.assertEqual(after, before)

    def test_export_built_in_presets(self):
        from sdr_module.core.chirp_csv import CHIRP_COLUMNS
        from sdr_module.core.frequency_manager import RX_PRESETS

        panel = self.make()
        self.assertTrue(panel._presets_action.isEnabled())  # even with none saved
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "presets")
            self.assertEqual(panel.export_presets_csv(path), len(RX_PRESETS))
            with open(path + ".csv", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(CHIRP_COLUMNS))
                rows = list(reader)
        acars = next(r for r in rows if r["Name"] == "ACARS")
        self.assertEqual(acars["Mode"], "AM")
        self.assertIn("[sdr mode=AM", acars["Comment"])
        self.assertIn(f"Exported {len(RX_PRESETS)} presets", self.infos[-1])


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #
class TestBookmarksInTheWindow(_WindowTestCase):
    def test_ctrl_b_names_after_the_band_and_skips_duplicates(self):
        """MC-U14 / UX-U16: one bookmark per channel, named after its band."""
        win = self.win
        win.set_frequency(100.1e6)
        win._set_demod_mode("FM", "75 kHz", "200 kHz")
        QTest.keyClick(win._spectrum, Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)
        saved = self.store.bookmarks[-1]
        self.assertEqual(saved["label"], "FM Broadcast 100.1")
        self.assertEqual(
            (saved["demod"], saved["fm_deviation"], saved["bandwidth"]),
            ("FM", "75 kHz", "200 kHz"),
        )
        self.assertIn("Bookmarked FM Broadcast 100.1 (FM)", win._message_label.text())
        self.assertIn("F2", win._message_label.text())
        QTest.keyClick(win._spectrum, Qt.Key.Key_B, Qt.KeyboardModifier.ControlModifier)
        self.assertEqual(len(self.store.bookmarks), 1)
        self.assertEqual(
            win._message_label.text(), "Already bookmarked: FM Broadcast 100.1"
        )
        self.assertEqual(win._message_label.property("tone"), "info")
        self.assertEqual(win._bookmarks_panel._selected_index(), 0)
        # In another mode it is another channel.
        win._set_demod_mode("AM")
        win._bookmark_current_frequency()
        self.assertEqual(len(self.store.bookmarks), 2)

    def test_ctrl_b_keeps_the_tone_squelch(self):
        win = self.win
        win.set_frequency(145.29e6)
        win._set_demod_mode("FM", "5 kHz", "25 kHz")
        win._control_panel.set_tone_squelch_hz(100.0)
        win._bookmark_current_frequency()
        saved = self.store.bookmarks[-1]
        self.assertEqual((saved["tone_mode"], saved["ctone"]), ("TSQL", 100.0))

    def test_tuning_a_bookmark_sets_its_filter_and_tone_squelch(self):
        win, panel = self.win, self.win._bookmarks_panel
        panel.add_bookmark("Club", 145.29e6, "FM", "5 kHz", "12.5 kHz")
        panel.set_tone(0, 100.0)
        panel.add_bookmark("Simplex", 146.52e6, "FM", "5 kHz")
        win._set_demod_mode("AM")
        panel._tune_index(0)
        self.assertAlmostEqual(win._current_frequency(), 145.29e6)
        self.assertEqual(win._control_panel._bw_combo.currentText(), "12.5 kHz")
        self.assertEqual(win._control_panel.tone_squelch_hz(), 100.0)
        self.assertEqual(win._tone_squelch.tone_hz, 100.0)  # T4's ToneSquelch
        self.assertIn("tone squelch 100.0 Hz", win._message_label.text())
        # A bookmark without a tone turns the tone squelch off.
        panel._tune_index(1)
        self.assertEqual(win._control_panel.tone_squelch_hz(), 0.0)
        self.assertIsNone(win._tone_squelch.tone_hz)
        self.assertEqual(win._control_panel._bw_combo.currentText(), "25 kHz")

    def test_a_waterfall_bookmark_elsewhere_gets_no_tone(self):
        win = self.win
        win.set_frequency(145.29e6)
        win._set_demod_mode("FM", "5 kHz", "25 kHz")
        win._control_panel.set_tone_squelch_hz(100.0)
        win._waterfall.bookmark_requested.emit(145.5e6)
        saved = self.store.bookmarks[-1]
        self.assertEqual(saved["freq_hz"], 145.5e6)
        self.assertNotIn("tone_mode", saved)

    def test_a_bookmark_without_a_mode_keeps_the_tone_squelch(self):
        win, panel = self.win, self.win._bookmarks_panel
        win._control_panel.set_tone_squelch_hz(100.0)
        panel.add_bookmark("Old", 145.29e6)
        panel._tune_index(0)
        self.assertEqual(win._control_panel.tone_squelch_hz(), 100.0)

    def test_the_add_row_follows_the_mode(self):
        win, panel = self.win, self.win._bookmarks_panel
        win.set_frequency(125e6)
        win._set_demod_mode("AM")
        self.assertEqual(panel._mode_input.currentData(), "AM")
        win._set_demod_mode("USB")
        self.assertEqual(panel._mode_input.currentData(), "USB")

    def test_messages_reach_the_status_bar(self):
        win, panel = self.win, self.win._bookmarks_panel
        panel.add_bookmark("A", 146.52e6, "FM", "5 kHz")
        panel._copy_frequency(0)
        self.assertEqual(win._message_label.text(), "Copied 146.520 MHz")


class TestNoHardCodedColors(unittest.TestCase):
    """The new panels take every colour from the theme."""

    def test_no_literal_colors(self):
        root = os.path.join(os.path.dirname(__file__), "..", "src", "sdr_module", "gui")
        pattern = re.compile(
            r"#[0-9a-fA-F]{3,8}\b|QColor\(\s*\d|rgba?\(|"
            r"\b(gray|grey|green|red|orange|blue|yellow|white|black)\b\s*[;\"']"
        )
        for name in (
            "scanner_panel.py",
            "scan_engine.py",
            "scanner_host.py",
            "bookmark_editor.py",
            "bookmarks_panel.py",
        ):
            with open(os.path.join(root, name), encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    self.assertIsNone(
                        pattern.search(line), f"{name}:{number}: {line.strip()}"
                    )


if __name__ == "__main__":
    unittest.main()

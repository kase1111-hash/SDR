"""Behavior tests for the Decoder and Bookmarks tool panels."""

from __future__ import annotations

import csv
import os
import re
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QGuiApplication
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QAbstractItemView, QApplication, QHeaderView

    HAS_PYQT6 = True
except ImportError:  # pragma: no cover - PyQt6 missing
    HAS_PYQT6 = False

_APP = None


def _app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def _settle(frames: int = 5) -> None:
    for _ in range(frames):
        _app().processEvents()


class _FakeSettings:
    """In-memory stand-in for GuiSettings so tests never touch QSettings."""

    def __init__(self, bookmarks=None):
        self.bookmarks = list(bookmarks or [])

    def get_bookmarks(self):
        return list(self.bookmarks)

    def set_bookmarks(self, bookmarks):
        self.bookmarks = list(bookmarks)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestDecoderPanel(unittest.TestCase):
    def setUp(self):
        _app()
        from sdr_module.gui import decoder_panel

        self.mod = decoder_panel
        self._real_info = decoder_panel.QMessageBox.information
        decoder_panel.QMessageBox.information = staticmethod(lambda *a, **k: None)
        self.panel = decoder_panel.DecoderPanel()
        self.panel.resize(360, 420)

    def tearDown(self):
        self.mod.QMessageBox.information = self._real_info
        self.panel.close()
        self.panel.deleteLater()
        _settle(1)

    def test_starts_off_with_decode_disabled_and_hint(self):
        p = self.panel
        self.assertIsNone(p.protocol())
        self.assertFalse(p.is_decoding())
        self.assertFalse(p._enabled_check.isEnabled())
        self.assertEqual(p._enabled_check.text(), "Decode")
        self.assertTrue(p._empty.is_visible())
        self.assertIn("No decoder selected", p._empty.text())

    def test_protocol_selection_emits_and_updates_hint(self):
        p = self.panel
        seen = []
        p.protocol_changed.connect(seen.append)
        p._proto_combo.setCurrentText("POCSAG")
        self.assertEqual(seen, ["POCSAG"])
        self.assertTrue(p._enabled_check.isEnabled())
        self.assertTrue(p.is_decoding())
        self.assertIn("Waiting for POCSAG", p._empty.text())
        p._enabled_check.setChecked(False)
        self.assertFalse(p._enabled_check.isChecked())
        self.assertIn("paused", p._empty.text())

    def test_protocol_names_match_main_window_mapping(self):
        from sdr_module.gui.main_window import SDRMainWindow

        names = [
            self.panel._proto_combo.itemText(i)
            for i in range(1, self.panel._proto_combo.count())
        ]
        self.assertEqual(sorted(names), sorted(SDRMainWindow._DECODER_PROTOCOLS))

    def test_buttons_disabled_until_there_are_messages(self):
        p = self.panel
        self.assertFalse(p._clear_btn.isEnabled())
        self.assertFalse(p._export_btn.isEnabled())
        self.assertTrue(p._clear_btn.toolTip())
        p.add_pocsag_message(1234567, "HELLO", 2)
        self.assertTrue(p._clear_btn.isEnabled())
        self.assertTrue(p._export_btn.isEnabled())
        self.assertFalse(p._empty.is_visible())
        p.clear()
        self.assertEqual(p.get_message_count(), 0)
        self.assertEqual(p._table.rowCount(), 0)
        self.assertFalse(p._clear_btn.isEnabled())
        self.assertTrue(p._empty.is_visible())
        self.assertEqual(p._msg_count_label.text(), "0")

    def test_table_is_read_only_row_selecting_and_message_stretches(self):
        t = self.panel._table
        self.assertEqual(t.editTriggers(), QAbstractItemView.EditTrigger.NoEditTriggers)
        self.assertEqual(
            t.selectionBehavior(), QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.assertFalse(t.verticalHeader().isVisible())
        self.assertTrue(t.alternatingRowColors())
        self.assertEqual(
            t.horizontalHeader().sectionResizeMode(self.mod.COL_MESSAGE),
            QHeaderView.ResizeMode.Stretch,
        )

    def test_invalid_rows_use_theme_role_not_fixed_color(self):
        p = self.panel
        p.add_message("FLEX", "42", "BAD", valid=False)
        item = p._table.item(0, self.mod.COL_MESSAGE)
        self.assertTrue(item.data(self.mod._INVALID_ROLE))
        self.assertEqual(item.background().style(), Qt.BrushStyle.NoBrush)
        self.assertEqual(p._invalid_count_label.text(), "1")
        self.assertEqual(p._invalid_count_label.property("tone"), "danger")

    def test_table_and_store_are_trimmed_together(self):
        p = self.panel
        p._max_messages = 5
        for i in range(8):
            p.add_message("POCSAG", str(i), f"msg {i}")
        self.assertEqual(p.get_message_count(), 5)
        self.assertEqual(p._table.rowCount(), 5)
        self.assertEqual(p._table.item(0, self.mod.COL_ADDRESS).text(), "3")
        # Totals count everything received since the last Clear.
        self.assertEqual(p._msg_count_label.text(), "8")

    def test_protocol_column_only_when_several_protocols(self):
        p = self.panel
        p.add_message("POCSAG", "1", "a")
        self.assertTrue(p._table.isColumnHidden(self.mod.COL_PROTOCOL))
        p.add_message("FLEX", "2", "b")
        self.assertFalse(p._table.isColumnHidden(self.mod.COL_PROTOCOL))
        self.assertGreater(p._table.columnWidth(self.mod.COL_PROTOCOL), 50)

    def test_follows_newest_message_until_user_scrolls_up(self):
        p = self.panel
        p.show()
        _settle()
        bar = p._table.verticalScrollBar()
        for i in range(60):
            p.add_message("POCSAG", str(i), "x" * 20)
        _settle()
        self.assertGreater(bar.maximum(), 0)
        self.assertEqual(bar.value(), bar.maximum())
        bar.setValue(0)
        for i in range(10):
            p.add_message("POCSAG", str(i), "y")
        _settle()
        self.assertEqual(bar.value(), 0)

    def test_log_is_terminal_and_lists_every_message(self):
        p = self.panel
        self.assertEqual(p._raw_output.property("role"), "terminal")
        self.assertTrue(p._log_empty.is_visible())
        p.add_message("RDS", "PI 1234", "Radio text")
        p.add_message("FLEX", "7", "Bad", valid=False, raw="a5 5a")
        self.assertFalse(p._log_empty.is_visible())
        text = p._raw_output.toPlainText()
        self.assertIn("RDS  PI 1234  Radio text", text)
        self.assertIn("[invalid]", text)
        self.assertIn("raw: a5 5a", text)
        p.clear()
        self.assertEqual(p._raw_output.toPlainText(), "")
        self.assertTrue(p._log_empty.is_visible())

    def test_export_writes_proper_csv(self):
        p = self.panel
        p.add_message("POCSAG", "1 (F0)", 'He said "hi", then left')
        p.add_message("FLEX", "2", "x", valid=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out")
            self.assertEqual(p._export_messages(path), 2)
            with open(path + ".csv", newline="", encoding="utf-8") as handle:
                rows = list(csv.reader(handle))
        self.assertEqual(
            rows[0][:5], ["Date", "Time", "Protocol", "Address", "Content"]
        )
        self.assertEqual(rows[1][4], 'He said "hi", then left')
        self.assertEqual(rows[2][5], "False")

    def test_copy_selected_rows(self):
        p = self.panel
        p.add_message("POCSAG", "1", "first")
        p.add_message("POCSAG", "2", "second")
        p._table.selectAll()
        self.assertEqual(p.copy_selected(), 2)
        text = QGuiApplication.clipboard().text()
        self.assertIn("first", text)
        self.assertIn("second", text)

    def test_formatted_helpers(self):
        p = self.panel
        p.add_adsb_message("A1B2C3", "UAL123", 35000, 37.5, -122.25, 450)
        content = p._messages[-1]["content"]
        self.assertIn("35,000 ft", content)
        self.assertIn("450 kt", content)
        p.add_aprs_message("W1AW-9", "APRS", 0, 0, "hello")
        self.assertEqual(p._messages[-1]["content"], "hello")
        self.assertEqual(p._messages[-1]["address"], "W1AW-9>APRS")


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestBookmarksPanel(unittest.TestCase):
    SAMPLE = [
        {"label": "2m Calling", "freq_hz": 146.52e6},
        {"label": "Tower", "freq_hz": 118.3e6},
    ]

    def setUp(self):
        _app()
        from sdr_module.gui import bookmarks_panel

        self.mod = bookmarks_panel
        self._real_settings = bookmarks_panel.GuiSettings
        self.store = _FakeSettings()
        bookmarks_panel.GuiSettings = lambda: self.store
        self._real_info = bookmarks_panel.QMessageBox.information
        bookmarks_panel.QMessageBox.information = staticmethod(lambda *a, **k: None)
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
        panel.resize(360, 360)
        self._panels.append(panel)
        return panel

    def test_format_mhz(self):
        self.assertEqual(self.mod.format_mhz(146.52e6), "146.520 MHz")
        self.assertEqual(self.mod.format_mhz(145.8125e6), "145.8125 MHz")
        self.assertEqual(self.mod.format_mhz(7.074e6), "7.074 MHz")
        self.assertEqual(self.mod.format_mhz(1_090_000_001), "1090.000001 MHz")

    def test_empty_state_and_disabled_actions(self):
        panel = self.make()
        self.assertTrue(panel._empty.is_visible())
        self.assertIn("Ctrl+B", panel._empty.text())
        for btn in (panel._tune_btn, panel._rename_btn, panel._remove_btn):
            self.assertFalse(btn.isEnabled())
            self.assertIn("Select a bookmark", btn.toolTip())
        self.assertFalse(panel._export_action.isEnabled())

    def test_columns_show_name_and_right_aligned_frequency(self):
        panel = self.make(self.SAMPLE)
        item = panel._list.topLevelItem(0)
        self.assertEqual(item.text(self.mod.COL_NAME), "2m Calling")
        self.assertEqual(item.text(self.mod.COL_FREQ), "146.520 MHz")
        self.assertTrue(
            item.textAlignment(self.mod.COL_FREQ) & Qt.AlignmentFlag.AlignRight.value
        )
        # No bookmark has mode/tone detail, so that column stays hidden.
        self.assertTrue(panel._list.isColumnHidden(self.mod.COL_MODE))
        self.assertFalse(panel._empty.is_visible())

    def test_selection_enables_actions(self):
        panel = self.make(self.SAMPLE)
        panel._list.setCurrentItem(panel._list.topLevelItem(1))
        for btn in (panel._tune_btn, panel._rename_btn, panel._remove_btn):
            self.assertTrue(btn.isEnabled())

    def test_enter_in_name_field_adds_and_selects(self):
        panel = self.make()
        panel.show()
        panel._label_input.setText("NOAA")
        panel._freq_input.setValue(162.55)
        QTest.keyClick(panel._label_input, Qt.Key.Key_Return)
        self.assertEqual(self.store.bookmarks[-1]["label"], "NOAA")
        self.assertAlmostEqual(self.store.bookmarks[-1]["freq_hz"], 162.55e6)
        self.assertEqual(panel._label_input.text(), "")
        self.assertEqual(panel._selected_index(), 0)

    def test_blank_name_uses_frequency(self):
        panel = self.make()
        panel._freq_input.setValue(145.8125)
        panel._add_btn.click()
        self.assertEqual(self.store.bookmarks[-1]["label"], "145.8125 MHz")

    def test_double_click_and_enter_tune(self):
        panel = self.make(self.SAMPLE)
        panel.show()
        tuned = []
        panel.tune_requested.connect(lambda f, label: tuned.append((f, label)))
        item = panel._list.topLevelItem(1)
        panel._list.itemDoubleClicked.emit(item, 0)
        self.assertEqual(tuned[-1], (118.3e6, "Tower"))
        panel._list.setCurrentItem(panel._list.topLevelItem(0))
        QTest.keyClick(panel._list, Qt.Key.Key_Return)
        self.assertEqual(tuned[-1], (146.52e6, "2m Calling"))

    def test_delete_key_removes_after_confirmation(self):
        panel = self.make(self.SAMPLE)
        panel.show()
        answers = [False, True]
        panel._confirm_remove = lambda _b: answers.pop(0)
        panel._list.setCurrentItem(panel._list.topLevelItem(0))
        QTest.keyClick(panel._list, Qt.Key.Key_Delete)
        self.assertEqual(len(self.store.bookmarks), 2)  # cancelled
        QTest.keyClick(panel._list, Qt.Key.Key_Delete)
        self.assertEqual([b["label"] for b in self.store.bookmarks], ["Tower"])
        # The neighbour is selected so Delete can be pressed again.
        self.assertEqual(panel._selected_index(), 0)

    def test_import_asks_replace_or_append(self):
        panel = self.make(self.SAMPLE)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "in.csv")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("Name,Frequency,Mode\nAir,121.500000,AM\n")
            panel._ask_replace_or_append = lambda n: None
            self.assertEqual(panel.import_csv(path), 0)
            self.assertEqual(len(self.store.bookmarks), 2)
            panel._ask_replace_or_append = lambda n: "append"
            self.assertEqual(panel.import_csv(path), 1)
            self.assertEqual(len(self.store.bookmarks), 3)
            self.assertFalse(panel._list.isColumnHidden(self.mod.COL_MODE))
            panel._ask_replace_or_append = lambda n: "replace"
            self.assertEqual(panel.import_csv(path), 1)
            self.assertEqual([b["label"] for b in self.store.bookmarks], ["Air"])

    def test_frequency_spinbox_has_mhz_suffix(self):
        panel = self.make()
        self.assertEqual(panel._freq_input.suffix(), " MHz")
        panel.set_current_frequency(145.8125e6)
        self.assertAlmostEqual(panel._freq_input.value(), 145.8125)


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestDecoderReadability(unittest.TestCase):
    """Review fixes: the Message column stays readable at 360 px and the
    selected message can be read in full."""

    LONG = "WEATHER KSFO 261756Z 28015KT 10SM FEW015 SCT200 18/12 A2992 " * 4

    def setUp(self):
        _app()
        from sdr_module.gui import decoder_panel

        self.mod = decoder_panel
        self.panel = decoder_panel.DecoderPanel()
        self.panel.resize(360, 460)
        self.panel.show()
        _settle()

    def tearDown(self):
        self.panel.close()
        self.panel.deleteLater()
        _settle(1)

    def _feed_mixed(self):
        p = self.panel
        p.add_pocsag_message(1234567, "Page " * 12, 3)
        p.add_aprs_message("W1AW-9", "APRS-LONG", 41.7, -72.7, "Mobile")
        p.add_message("FLEX", "002001234", "Corrupted", valid=False)
        _settle()

    def test_message_column_keeps_a_readable_share_at_360px(self):
        self._feed_mixed()
        t = self.panel._table
        self.assertFalse(t.isColumnHidden(self.mod.COL_PROTOCOL))
        avail = t.viewport().width()
        fixed = sum(
            t.columnWidth(c)
            for c in (self.mod.COL_TIME, self.mod.COL_PROTOCOL, self.mod.COL_ADDRESS)
        )
        message = avail - fixed
        self.assertGreaterEqual(message, self.mod._MIN_MESSAGE_WIDTH - 10)
        # The squeezed Address column still shows its whole header.
        self.assertGreaterEqual(
            t.columnWidth(self.mod.COL_ADDRESS),
            self.panel._header_width(self.mod.COL_ADDRESS),
        )

    def test_wide_panel_gives_address_its_full_width(self):
        self._feed_mixed()
        self.panel.resize(720, 460)
        _settle()
        t = self.panel._table
        wanted = self.panel._col_content[self.mod.COL_ADDRESS]
        self.assertEqual(t.columnWidth(self.mod.COL_ADDRESS), wanted)

    def test_selecting_a_row_shows_the_whole_message(self):
        p = self.panel
        p.add_message("ACARS", ".N12345", "Short text")
        detail = p._detail
        self.assertTrue(detail.isHidden())
        p._table.selectRow(0)
        _settle()
        self.assertFalse(detail.isHidden())
        self.assertEqual(detail.full_text(), "Short text")
        self.assertIn(".N12345", detail.meta.text())
        # Esc clears the selection and hides the card again.
        p._table.setFocus()
        QTest.keyClick(p._table, Qt.Key.Key_Escape)
        self.assertTrue(detail.isHidden())

    def test_long_message_is_cut_to_a_few_lines_without_widening_panel(self):
        p = self.panel
        p.add_message("ACARS", ".N12345", self.LONG, valid=False)
        p._table.selectRow(0)
        _settle(10)
        body = p._detail.body.text()
        self.assertLessEqual(body.count("\n") + 1, self.mod._DETAIL_LINES)
        self.assertTrue(body.endswith("…"))
        self.assertEqual(p._detail.body.toolTip(), self.LONG)
        self.assertEqual(p._detail.meta.property("tone"), "danger")
        self.assertLessEqual(p.minimumSizeHint().width(), 360)
        self.assertEqual(p.width(), 360)

    def test_multi_selection_summarises_and_clear_hides(self):
        p = self.panel
        for i in range(3):
            p.add_message("POCSAG", str(i), f"m{i}")
        p._table.selectAll()
        _settle()
        self.assertIn("3 messages selected", p._detail.meta.text())
        self.assertIn("Ctrl+C", p._detail.body.text())
        p.clear()
        self.assertTrue(p._detail.isHidden())

    def test_decoded_text_is_never_rendered_as_markup(self):
        p = self.panel
        p.add_message("POCSAG", "1", "<b>not bold</b>")
        p._table.selectRow(0)
        _settle()
        self.assertEqual(p._detail.body.textFormat(), Qt.TextFormat.PlainText)
        self.assertIn("<b>", p._detail.body.text())

    def test_elide_lines(self):
        font = self.panel.font()
        self.assertEqual(self.mod.elide_lines("short", font, 300, 3), "short")
        cut = self.mod.elide_lines(self.LONG, font, 200, 2)
        self.assertEqual(cut.count("\n"), 1)
        self.assertTrue(cut.endswith("…"))

    def test_log_placeholder_does_not_promise_raw_bytes(self):
        self.assertNotIn("raw", self.panel._log_empty.text().lower())


@unittest.skipUnless(HAS_PYQT6, "PyQt6 not available")
class TestBookmarksReview(unittest.TestCase):
    """Review fixes for the Bookmarks panel."""

    def setUp(self):
        _app()
        from sdr_module.gui import bookmarks_panel

        self.mod = bookmarks_panel
        self._real_settings = bookmarks_panel.GuiSettings
        self.store = _FakeSettings()
        bookmarks_panel.GuiSettings = lambda: self.store
        self._panels = []

    def tearDown(self):
        self.mod.GuiSettings = self._real_settings
        for panel in self._panels:
            panel.close()
            panel.deleteLater()
        _settle(1)

    def make(self, bookmarks=()):
        self.store.bookmarks = [dict(b) for b in bookmarks]
        panel = self.mod.BookmarksPanel()
        panel.resize(360, 360)
        self._panels.append(panel)
        return panel

    def test_spinbox_shows_three_to_six_decimals_but_keeps_hz(self):
        spin = self.make()._freq_input
        self.assertEqual(spin.text(), "100.100 MHz")  # DEFAULT_FREQUENCY_HZ
        spin.setValue(145.8125)
        self.assertEqual(spin.text(), "145.8125 MHz")
        spin.setValue(1090.000001)
        self.assertEqual(spin.text(), "1090.000001 MHz")
        self.assertAlmostEqual(spin.value(), 1090.000001, places=6)

    def test_name_field_says_it_is_optional(self):
        self.assertIn("optional", self.make()._label_input.placeholderText())

    def test_imported_channel_with_default_mode_shows_fm(self):
        panel = self.make(
            [
                {"label": "Plain", "freq_hz": 146.52e6},
                {"label": "Rptr", "freq_hz": 146.94e6, "location": 2},
            ]
        )
        self.assertFalse(panel._list.isColumnHidden(self.mod.COL_MODE))
        self.assertEqual(panel._list.topLevelItem(0).text(self.mod.COL_MODE), "")
        self.assertEqual(panel._list.topLevelItem(1).text(self.mod.COL_MODE), "FM")
        self.assertIn("Mode: FM", panel._list.topLevelItem(1).toolTip(0))

    def test_malformed_stored_values_do_not_crash(self):
        panel = self.make(
            [{"label": "Odd", "freq_hz": None, "tone_mode": "Tone", "rtone": None}]
        )
        item = panel._list.topLevelItem(0)
        self.assertEqual(item.text(self.mod.COL_FREQ), "0.000 MHz")
        self.assertIn("Tone: 88.5 Hz", item.toolTip(0))

    def test_plurals(self):
        self.assertEqual(self.mod._plural(1, "channel"), "1 channel")
        self.assertEqual(self.mod._plural(1200, "channel"), "1,200 channels")


class TestPanelsHaveNoHardCodedColors(unittest.TestCase):
    def test_no_literal_colors(self):
        root = os.path.join(os.path.dirname(__file__), "..", "src", "sdr_module", "gui")
        pattern = re.compile(
            r"#[0-9a-fA-F]{3,8}\b|QColor\(\s*\d|"
            r"\b(gray|grey|green|red|orange|blue|yellow|white|black)\b\s*[;\"']"
        )
        for name in ("decoder_panel.py", "bookmarks_panel.py"):
            with open(os.path.join(root, name), encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    self.assertIsNone(
                        pattern.search(line), f"{name}:{number}: {line.strip()}"
                    )


if __name__ == "__main__":
    unittest.main()

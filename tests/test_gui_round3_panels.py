#!/usr/bin/env python3
"""
Round 3 review fixes for the control, decoder and bookmarks panels:

* the preset details show the bandwidth Apply Preset really sets;
* the panels start on ``settings_store.DEFAULT_FREQUENCY_HZ``;
* the squelch tooltip describes the channel level it compares against;
* empty-state titles are short phrases without a period, shown as a title
  over the hint (like the plots' painted placeholders);
* the decoder's Clear / Export buttons are no longer cut off by the
  underline sub-tabs, in either theme.

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_round3_panels.py``
"""

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication

    HAS_PYQT6 = True
    app = QApplication.instance() or QApplication([])
except ImportError:  # pragma: no cover - environment
    HAS_PYQT6 = False


def settle(frames: int = 3) -> None:
    for _ in range(frames):
        QApplication.processEvents()


class _FakeSettings:
    """In-memory stand-in for GuiSettings so tests never touch QSettings."""

    def __init__(self):
        self.bookmarks = []

    def get_bookmarks(self):
        return list(self.bookmarks)

    def set_bookmarks(self, bookmarks):
        self.bookmarks = list(bookmarks)


class _Themed(unittest.TestCase):
    """Applies the dark theme for the class and restores the old look."""

    @classmethod
    def setUpClass(cls):
        if not HAS_PYQT6:
            raise unittest.SkipTest("PyQt6 not available")
        from sdr_module.gui import themes

        qapp = QApplication.instance()
        cls._old_theme = themes.current_theme()
        cls._old_sheet, cls._old_palette = qapp.styleSheet(), qapp.palette()
        themes.apply_theme(qapp, "dark")

    @classmethod
    def tearDownClass(cls):
        from sdr_module.gui import themes

        qapp = QApplication.instance()
        themes.apply_theme(None, cls._old_theme)
        qapp.setStyleSheet(cls._old_sheet)
        qapp.setPalette(cls._old_palette)

    def setUp(self):
        self._widgets = []

    def tearDown(self):
        for widget in self._widgets:
            widget.close()
            widget.deleteLater()
        settle(1)

    def keep(self, widget, width=360, height=420):
        widget.resize(width, height)
        widget.show()
        settle()
        self._widgets.append(widget)
        return widget


# ---------------------------------------------------------------------------
# Control panel
# ---------------------------------------------------------------------------


class TestControlPanel(_Themed):
    def setUp(self):
        super().setUp()
        from sdr_module.core.frequency_manager import get_frequency_manager
        from sdr_module.gui.control_panel import ControlPanel

        self._fm = get_frequency_manager()
        self._license = self._fm.get_license_class()
        self.panel = self.keep(ControlPanel(), height=700)

    def tearDown(self):
        super().tearDown()
        self._fm.set_license_class(self._license)

    def select(self, category, name):
        self.panel._category_combo.setCurrentText(category)
        self.panel._preset_combo.setCurrentText(name)
        self.assertEqual(self.panel._preset_combo.currentText(), name)

    def test_details_show_the_bandwidth_apply_sets(self):
        p = self.panel
        for category in ("Amateur", "Broadcast", "Aviation", "QRP", "ISM"):
            p._category_combo.setCurrentText(category)
            for i in range(p._preset_combo.count()):
                p._preset_combo.setCurrentIndex(i)
                name = p._preset_combo.currentText()
                first_line = p._preset_info.text().split("\n")[0]
                p._apply_preset()
                applied = p._bw_combo.currentText()
                self.assertIn(f" · {applied} · ", first_line, name)

    def test_2m_calling_promises_25_khz_and_explains_the_channel(self):
        self.select("Amateur", "2m Calling")
        info = self.panel._preset_info
        self.assertIn("· 25 kHz ·", info.text())
        self.assertNotIn("15 kHz", info.text())
        # The nominal channel width is still available, in the tooltip.
        self.assertIn("15 kHz", info.toolTip())
        self.assertIn("25 kHz", info.toolTip())
        self.panel._apply_preset()
        self.assertEqual(self.panel._bw_combo.currentText(), "25 kHz")

    def test_no_tooltip_when_the_channel_matches_an_option(self):
        self.select("Broadcast", "FM Broadcast")
        self.assertIn("· 200 kHz ·", self.panel._preset_info.text())
        self.assertEqual(self.panel._preset_info.toolTip(), "")

    def test_starts_on_the_default_frequency(self):
        from sdr_module.gui.control_panel import FrequencyInput
        from sdr_module.gui.settings_store import DEFAULT_FREQUENCY_HZ

        self.assertEqual(self.panel._freq_input.get_frequency(), DEFAULT_FREQUENCY_HZ)
        self.assertEqual(self.panel._freq_input._freq_input.text(), "100.100")
        self.assertEqual(FrequencyInput().get_frequency(), DEFAULT_FREQUENCY_HZ)

    def test_squelch_tooltip_names_the_channel_level(self):
        tip = self.panel._squelch_slider.toolTip()
        self.assertIn("tuned channel", tip)
        self.assertIn("LEVEL", tip)
        self.assertNotIn("peak", tip.lower())

    def test_step_buttons_use_a_true_minus_sign(self):
        labels = [b.text() for b in self.panel._quick_tune_buttons]
        self.assertEqual(labels, ["−1M", "−100k", "−10k", "+10k", "+100k", "+1M"])
        # The helper itself keeps plain ASCII.
        self.assertEqual(self.panel._format_offset(-10e3), "-10k")
        self.assertLessEqual(
            self.panel._content.minimumSizeHint().width(),
            self.panel._scroll.viewport().width(),
        )


# ---------------------------------------------------------------------------
# Empty states
# ---------------------------------------------------------------------------


def _title(text: str) -> str:
    return text.split("\n", 1)[0]


class TestPlaceholderMarkup(unittest.TestCase):
    def setUp(self):
        if not HAS_PYQT6:
            self.skipTest("PyQt6 not available")
        from sdr_module.gui.decoder_panel import placeholder_html

        self.html = placeholder_html

    def test_title_is_bold_over_the_hint(self):
        markup = self.html("No bookmarks yet\nPress Ctrl+B.")
        self.assertIn("font-weight: 600", markup)
        self.assertIn(">No bookmarks yet</p>", markup)
        self.assertIn("Press Ctrl+B.", markup)

    def test_text_is_escaped(self):
        markup = self.html("<b>x</b>\na & b")
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", markup)
        self.assertIn("a &amp; b", markup)
        self.assertNotIn("<b>", markup)

    def test_single_line_is_shown_as_is(self):
        self.assertEqual(self.html("Scanning & more"), "Scanning &amp; more")

    def test_hint_lines_keep_their_breaks(self):
        self.assertIn("one<br>two", self.html("T\none\ntwo"))


class TestDecoderEmptyStates(_Themed):
    def setUp(self):
        super().setUp()
        from sdr_module.gui.decoder_panel import DecoderPanel

        self.panel = self.keep(DecoderPanel())
        self.panel.tune_requested.connect(lambda *args: None)

    def all_texts(self):
        p = self.panel
        texts = [p._empty.text(), p._log_empty.text()]
        for name in ("POCSAG", "FLEX", "AX.25/APRS", "ADS-B", "ACARS", "RDS"):
            p._proto_combo.setCurrentText(name)
            texts.append(p._empty.text())
        p._proto_combo.setCurrentText("ADS-B")
        p.set_current_frequency(1090e6)
        texts.append(p._empty.text())  # tuned, not running
        p.set_receiving(True)
        texts.append(p._empty.text())  # listening
        p.set_current_frequency(100e6)
        texts.append(p._empty.text())  # running, elsewhere
        p._enabled_check.setChecked(False)
        texts.append(p._empty.text())  # paused
        return texts

    def test_titles_have_no_final_period(self):
        texts = self.all_texts()
        self.assertIn("No decoder selected", [_title(t) for t in texts])
        self.assertIn("Decoding paused", [_title(t) for t in texts])
        for text in texts:
            title = _title(text)
            self.assertIn("\n", text, text)
            self.assertFalse(title.endswith((".", "…")), title)
            # Hint sentences keep their periods.
            self.assertTrue(text.rstrip().endswith("."), text)

    def test_title_renders_bold_and_text_stays_plain(self):
        p = self.panel
        self.assertEqual(p._empty.label.textFormat(), Qt.TextFormat.RichText)
        self.assertIn("font-weight: 600", p._empty.label.text())
        self.assertTrue(p._empty.text().startswith("No decoder selected\n"))
        self.assertNotIn("<", p._empty.text())

    def test_stats_empty_line_matches(self):
        self.assertEqual(self.panel._proto_none.text(), "No messages yet")

    def test_message_without_text_shows_a_dash(self):
        p = self.panel
        p.add_message("ADS-B", "A1B2C3", "")
        item = p._table.item(0, 3)
        self.assertEqual(item.text(), "–")
        self.assertEqual(item.toolTip(), "No message text")
        self.assertEqual(p._last_msg_label.text(), p._messages[-1]["time"])
        p.clear()
        self.assertEqual(p._last_msg_label.text(), "–")


class TestBookmarksEmptyState(_Themed):
    def setUp(self):
        super().setUp()
        from sdr_module.gui import bookmarks_panel

        self.mod = bookmarks_panel
        self._real_settings = bookmarks_panel.GuiSettings
        self.store = _FakeSettings()
        bookmarks_panel.GuiSettings = lambda: self.store
        self.panel = self.keep(bookmarks_panel.BookmarksPanel(), height=360)

    def tearDown(self):
        super().tearDown()
        self.mod.GuiSettings = self._real_settings

    def test_title_has_no_period(self):
        text = self.panel._empty.text()
        self.assertEqual(_title(text), "No bookmarks yet")
        self.assertIn("Ctrl+B", text)
        self.assertIn("font-weight: 600", self.panel._empty.label.text())

    def test_add_field_starts_on_the_default_frequency(self):
        from sdr_module.gui.settings_store import DEFAULT_FREQUENCY_HZ

        spin = self.panel._freq_input
        self.assertAlmostEqual(spin.value() * 1e6, DEFAULT_FREQUENCY_HZ, places=0)
        self.assertEqual(spin.text(), "100.100 MHz")


# ---------------------------------------------------------------------------
# Decoder corner buttons
# ---------------------------------------------------------------------------


class TestDecoderCornerButtons(_Themed):
    def check_whole(self):
        from sdr_module.gui.decoder_panel import DecoderPanel

        panel = self.keep(DecoderPanel())
        corner = panel._tabs.cornerWidget(Qt.Corner.TopRightCorner)
        self.assertIsNotNone(corner)
        self.assertGreaterEqual(corner.height(), corner.sizeHint().height())
        for button in (panel._clear_btn, panel._export_btn):
            self.assertEqual(button.property("role"), None)  # regular buttons
            self.assertGreaterEqual(button.height(), button.sizeHint().height())
            bottom = button.mapTo(corner, button.rect().bottomLeft()).y()
            self.assertLess(bottom, corner.height(), button.text())
        # The tab titles and the buttons share one row.
        bar = panel._tabs.tabBar()
        self.assertEqual(bar.height(), bar.sizeHint().height())
        self.assertGreaterEqual(bar.height(), corner.sizeHint().height())

    def test_buttons_are_not_clipped_dark(self):
        self.check_whole()

    def test_buttons_are_not_clipped_light(self):
        from sdr_module.gui import themes

        themes.apply_theme(QApplication.instance(), "light")
        try:
            self.check_whole()
        finally:
            themes.apply_theme(QApplication.instance(), "dark")


if __name__ == "__main__":
    unittest.main()

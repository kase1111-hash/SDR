#!/usr/bin/env python3
"""
Tests for the spectrum and waterfall plots: layout (header strip above the
plot, shared margins), click-to-tune and hover, theme-driven colors, the
waterfall scroll direction and the header controls.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_plots.py``
"""

import os
import re
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QFontMetrics
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

    HAS_PYQT6 = True
    PYQT6_IMPORT_ERROR = ""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
except ImportError as _import_error:  # pragma: no cover - environment
    HAS_PYQT6 = False
    PYQT6_IMPORT_ERROR = str(_import_error)

GUI_DIR = Path(__file__).resolve().parent.parent / "src" / "sdr_module" / "gui"


def require_pyqt6(test_case: unittest.TestCase) -> None:
    """Skip when PyQt6 is unavailable (fail instead if SDR_REQUIRE_GUI is set)."""
    if HAS_PYQT6:
        return
    if os.environ.get("SDR_REQUIRE_GUI"):
        test_case.fail(
            f"PyQt6 is required but could not be imported: {PYQT6_IMPORT_ERROR}"
        )
    test_case.skipTest("PyQt6 not available")


def _settle():
    for _ in range(3):
        QApplication.processEvents()


def _signal_frame(n=2048):
    rng = np.random.default_rng(0)
    x = rng.normal(-85, 2, n)
    x[1000:1050] = -20.0
    return x


class TestAxisHelpers(unittest.TestCase):
    """Nice tick steps and compact MHz labels."""

    def setUp(self):
        require_pyqt6(self)

    def test_nice_step_is_1_2_5_decade(self):
        from sdr_module.gui.spectrum_widget import nice_step

        self.assertEqual(nice_step(0.13), 0.2)
        self.assertEqual(nice_step(0.2), 0.2)
        self.assertEqual(nice_step(3.0), 5.0)
        self.assertEqual(nice_step(7.0), 10.0)
        self.assertEqual(nice_step(180e3), 200e3)
        self.assertEqual(nice_step(0), 1.0)

    def test_nice_ticks_are_inclusive_and_float_safe(self):
        from sdr_module.gui.spectrum_widget import nice_ticks

        ticks = nice_ticks(98.8, 101.2, 0.2)
        self.assertEqual(len(ticks), 13)
        self.assertAlmostEqual(ticks[0], 98.8)
        self.assertAlmostEqual(ticks[-1], 101.2)

    def test_freq_ticks_labels_fit_and_are_compact(self):
        from sdr_module.gui.spectrum_widget import axis_font, format_mhz, freq_ticks

        fm = QFontMetrics(axis_font())
        ticks, decimals = freq_ticks(98.8e6, 101.2e6, 800, fm)
        self.assertGreaterEqual(len(ticks), 4)
        self.assertEqual(decimals, 1)
        self.assertEqual(format_mhz(100e6, decimals), "100.0")
        step_px = (ticks[1] - ticks[0]) / 2.4e6 * 800
        widest = max(fm.horizontalAdvance(format_mhz(f, decimals)) for f in ticks)
        self.assertGreater(step_px, widest)

    def test_format_mhz_never_shows_negative_zero(self):
        from sdr_module.gui.spectrum_widget import format_mhz

        self.assertEqual(format_mhz(-1.0, 1), "0.0")


class TestPlotLayout(unittest.TestCase):
    """The header strip sits above the plot; the plot never overlaps it."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui.spectrum_widget import SpectrumWidget
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        self.spectrum = SpectrumWidget()
        self.waterfall = WaterfallWidget()
        for w in (self.spectrum, self.waterfall):
            w.resize(820, 360)
            w.show()
        _settle()

    def tearDown(self):
        for w in (self.spectrum, self.waterfall):
            w.close()
            w.deleteLater()
        _settle()

    def test_header_strip_on_top_and_canvas_below(self):
        for w in (self.spectrum, self.waterfall):
            header, canvas = w._header, w._canvas
            self.assertEqual(header.property("role"), "header-strip")
            self.assertEqual(header.geometry().top(), 0)
            self.assertGreaterEqual(canvas.geometry().top(), header.geometry().bottom())
            self.assertEqual(canvas.geometry().bottom(), w.height() - 1)
            self.assertGreater(canvas.height(), w.height() / 2)

    def test_plots_share_horizontal_margins(self):
        s_rect = self.spectrum.plot_rect()
        w_rect = self.waterfall.plot_rect()
        self.assertGreater(s_rect.width(), 0)
        self.assertEqual(s_rect.left(), w_rect.left())
        self.assertEqual(s_rect.right(), w_rect.right())

    def test_combos_are_not_truncated(self):
        combos = (
            self.spectrum._avg_combo,
            self.waterfall._history_combo,
        )
        for combo in combos:
            self.assertNotEqual(combo.minimumWidth(), combo.maximumWidth())
            self.assertGreaterEqual(combo.width(), combo.sizeHint().width())

    def test_header_labels_use_roles_not_inline_styles(self):
        from PyQt6.QtWidgets import QLabel

        for w in (self.spectrum, self.waterfall):
            for label in w._header.findChildren(QLabel):
                self.assertEqual(label.styleSheet(), "")
                self.assertIn(label.property("role"), ("caption", "hint"))

    def test_hint_hides_when_header_is_too_narrow(self):
        self.spectrum.resize(820, 360)
        _settle()
        self.assertFalse(self.spectrum._hint_label.isHidden())
        needed = self.spectrum._header.layout().sizeHint().width()
        self.spectrum.resize(needed - 10, 360)
        _settle()
        self.assertTrue(self.spectrum._hint_label.isHidden())
        self.spectrum.resize(820, 360)
        _settle()
        self.assertFalse(self.spectrum._hint_label.isHidden())


class TestClickToTuneAndHover(unittest.TestCase):
    """Clicks inside the plot tune; clicks in the margins do not."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui.spectrum_widget import SpectrumWidget
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        self.spectrum = SpectrumWidget()
        self.waterfall = WaterfallWidget()
        for w in (self.spectrum, self.waterfall):
            w.resize(820, 360)
            w.show()
        _settle()

    def tearDown(self):
        for w in (self.spectrum, self.waterfall):
            w.close()
            w.deleteLater()
        _settle()

    def _click(self, widget, x, y):
        received = []
        widget.frequency_clicked.connect(received.append)
        QTest.mouseClick(
            widget._canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(int(x), int(y)),
        )
        return received

    def test_click_at_center_tunes_to_center(self):
        for w in (self.spectrum, self.waterfall):
            rect = w.plot_rect()
            got = self._click(w, rect.center().x(), rect.center().y())
            self.assertEqual(len(got), 1)
            hz_per_px = 2.4e6 / rect.width()
            self.assertLessEqual(abs(got[0] - 100e6), hz_per_px)
            # Snapped to the display resolution (1 kHz at ~3 kHz/px)
            self.assertEqual(got[0] % 1000, 0)

    def test_click_near_right_edge_is_above_center(self):
        rect = self.spectrum.plot_rect()
        got = self._click(self.spectrum, rect.right() - 2, rect.center().y())
        self.assertEqual(len(got), 1)
        self.assertGreater(got[0], 101.1e6)
        self.assertLessEqual(got[0], 101.2e6)

    def test_click_in_axis_margin_does_nothing(self):
        for w in (self.spectrum, self.waterfall):
            rect = w.plot_rect()
            got = self._click(w, rect.left() - 10, rect.center().y())
            self.assertEqual(got, [])

    def test_hover_tracks_and_clears(self):
        self.spectrum.update_spectrum(_signal_frame())
        canvas = self.spectrum._canvas
        self.assertTrue(canvas.hasMouseTracking())
        self.assertEqual(canvas.cursor().shape(), Qt.CursorShape.CrossCursor)
        self.spectrum._set_hover(QPointF(200, 80))
        canvas.grab()  # paints the readout without raising
        self.assertIsNotNone(self.spectrum._hover)
        self.spectrum._set_hover(None)
        self.assertIsNone(self.spectrum._hover)

    def test_waterfall_hover_paints_readout(self):
        for _ in range(5):
            self.waterfall.add_line(_signal_frame())
        self.waterfall._set_hover(QPointF(300, 10))
        self.waterfall._canvas.grab()  # must not raise


class TestThemeColors(unittest.TestCase):
    """Plots paint with the active palette and follow theme changes."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import themes

        self.themes = themes
        self._previous = themes.current_theme()

    def tearDown(self):
        self.themes.apply_theme(QApplication.instance(), self._previous)

    def _corner_color(self, widget):
        image = widget._canvas.grab().toImage()
        return image.pixelColor(2, image.height() - 2).name()

    def test_plot_background_follows_theme(self):
        from sdr_module.gui.spectrum_widget import SpectrumWidget
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        app_ = QApplication.instance()
        spectrum, waterfall = SpectrumWidget(), WaterfallWidget()
        for w in (spectrum, waterfall):
            w.resize(600, 300)
            w.show()
        _settle()
        try:
            for theme in ("light", "dark"):
                self.themes.apply_theme(app_, theme)
                _settle()
                expected = self.themes.get_palette(theme).plot_bg
                self.assertEqual(self._corner_color(spectrum), expected)
                self.assertEqual(self._corner_color(waterfall), expected)
        finally:
            for w in (spectrum, waterfall):
                w.close()
                w.deleteLater()

    def test_waterfall_empty_rows_rerender_on_theme_change(self):
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        app_ = QApplication.instance()
        self.themes.apply_theme(app_, "dark")
        widget = WaterfallWidget(history_size=10)
        widget.add_line(np.zeros(2048))
        self.themes.apply_theme(app_, "light")
        bottom = widget._image.pixelColor(5, 9).name()
        self.assertEqual(bottom, self.themes.get_palette("light").plot_bg)
        widget.deleteLater()

    def test_no_hard_coded_colors_outside_colormaps(self):
        pattern = re.compile(
            r"#[0-9a-fA-F]{3,8}\b|\b(gray|grey|green|red|orange|blue|yellow|"
            r"white|black)\b\s*[;\"']|QColor\(\s*\d|rgb\("
        )
        for name in ("spectrum_widget.py", "waterfall_widget.py"):
            text = (GUI_DIR / name).read_text()
            # Data-visualization colormaps are the one allowed exception.
            text = re.sub(r"COLORMAPS = \{.*?\n    \}\n", "", text, flags=re.S)
            hits = [m.group(0) for m in pattern.finditer(text)]
            self.assertEqual(hits, [], f"{name} hard-codes colors: {hits}")
            self.assertNotIn("setStyleSheet", text, name)


class TestWaterfallBehavior(unittest.TestCase):
    """Scroll direction, incremental rendering and the Clear button."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        self.widget = WaterfallWidget(history_size=8)

    def tearDown(self):
        self.widget.deleteLater()

    def test_newest_line_at_top_older_scroll_down(self):
        w = self.widget
        w.add_line(np.full(2048, -100.0))  # bottom of range
        w.add_line(np.zeros(2048))  # full scale
        top = w._image.pixelColor(100, 0)
        second = w._image.pixelColor(100, 1)
        full = tuple(int(v) for v in w._colormap[255])
        low = tuple(int(v) for v in w._colormap[0])
        self.assertEqual((top.red(), top.green(), top.blue()), full)
        self.assertEqual((second.red(), second.green(), second.blue()), low)

    def test_incremental_scroll_matches_full_render(self):
        w = self.widget
        rng = np.random.default_rng(3)
        for _ in range(12):  # more than history_size: exercises the scroll
            w.add_line(rng.uniform(-100, 0, 2048))
        incremental = w._image_rgb.copy()
        w._render_image()
        np.testing.assert_array_equal(incremental, w._image_rgb)

    def test_clear_action_state_and_tooltip(self):
        w = self.widget
        self.assertFalse(w._clear_action.isEnabled())
        self.assertIn("empty", w._clear_action.toolTip())
        w.add_line(np.zeros(2048))
        self.assertTrue(w._clear_action.isEnabled())
        w._clear_action.trigger()
        self.assertEqual(len(w._history), 0)
        self.assertFalse(w._clear_action.isEnabled())
        self.assertIsNone(w._image)

    def test_colors_menu_shows_title_case_names(self):
        from PyQt6.QtWidgets import QMenu

        w = self.widget
        self.assertEqual(w.colormap_name(), "turbo")
        menu = QMenu()
        w._display_menu(menu)
        colors = next(
            a.menu()
            for a in menu.actions()
            if a.menu() and a.text().replace("&", "") == "Colors"
        )
        names = [a.text() for a in colors.actions()]
        self.assertIn("Turbo", names)
        self.assertIn("Viridis", names)
        self.assertEqual(
            [a.text() for a in colors.actions() if a.isChecked()], ["Turbo"]
        )
        next(a for a in colors.actions() if a.text() == "Viridis").trigger()
        self.assertEqual(w.colormap_name(), "viridis")
        menu.deleteLater()

    def test_time_labels_fit_the_shared_left_margin(self):
        from sdr_module.gui.spectrum_widget import axis_font, plot_side_margins
        from sdr_module.gui.waterfall_widget import format_age, time_step

        fm = QFontMetrics(axis_font())
        left, _right = plot_side_margins(fm)
        for raw in (0.01, 0.07, 0.3, 1.5, 4, 12, 45, 100, 250, 900, 5000):
            step = time_step(raw)
            self.assertGreaterEqual(step, raw if raw <= 3600 else 3600)
            for k in range(0, 12):
                label = format_age(k * step, step)
                self.assertLessEqual(fm.horizontalAdvance(label), left - 8, label)
        self.assertEqual(format_age(0, 0.1), "0s")
        self.assertEqual(format_age(0.5, 0.5), "0.5s")
        self.assertEqual(format_age(120, 60), "2m")

    def test_seconds_per_row_from_arrival_times(self):
        w = self.widget
        for _ in range(3):
            w.add_line(np.zeros(2048))
        w._times.clear()
        w._times.extend([10.0, 10.5, 11.0])
        self.assertAlmostEqual(w._seconds_per_row(), 0.5)


class TestSpectrumBehavior(unittest.TestCase):
    """Peak hold controls and retune behavior."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui.spectrum_widget import SpectrumWidget

        self.widget = SpectrumWidget()

    def tearDown(self):
        self.widget.deleteLater()

    def test_peak_hold_clears_on_retune_only(self):
        w = self.widget
        w.update_spectrum(np.full(2048, -40.0))
        w.set_center_freq(w._center_freq)  # unchanged: keep the peak
        self.assertTrue(np.all(w._peak_hold == -40.0))
        w.set_center_freq(145e6)
        self.assertTrue(np.all(w._peak_hold == -120.0))
        w.update_spectrum(np.full(2048, -40.0))
        w.set_frequency_range(145e6, 1.0e6)
        self.assertTrue(np.all(w._peak_hold == -120.0))

    def test_peak_checkbox_toggles_trace_and_reset_button(self):
        w = self.widget
        self.assertTrue(w._peak_check.isChecked())
        self.assertTrue(w._peak_reset_btn.isEnabled())
        w._peak_check.setChecked(False)
        self.assertFalse(w._show_peak)
        self.assertFalse(w._peak_reset_btn.isEnabled())
        self.assertIn("Peak hold", w._peak_reset_btn.toolTip())
        w._peak_check.setChecked(True)
        w.update_spectrum(np.full(2048, -30.0))
        w._peak_reset_btn.click()
        self.assertTrue(np.all(w._peak_hold == -120.0))

    def test_nan_bins_do_not_latch_peak_hold(self):
        w = self.widget
        data = np.full(2048, -50.0)
        data[10] = np.nan
        w.update_spectrum(data)
        self.assertFalse(np.any(np.isnan(w._peak_hold)))

    def test_empty_state_until_first_frame(self):
        w = self.widget
        w.resize(600, 300)
        self.assertFalse(w._has_data)
        w._canvas.grab()  # placeholder paints without raising
        w.update_spectrum(_signal_frame())
        self.assertTrue(w._has_data)
        w._canvas.grab()


class TestReviewFixes(unittest.TestCase):
    """Regressions found in review: carrier visibility, pause-aware time axis,
    empty input, empty-state hover, placeholder backdrop, header min width."""

    def setUp(self):
        require_pyqt6(self)
        self._widgets = []

    def tearDown(self):
        for w in self._widgets:
            w.close()
            w.deleteLater()
        _settle()

    def _show(self, widget, size):
        widget.resize(*size)
        widget.show()
        _settle()
        self._widgets.append(widget)
        return widget

    def test_columns_max_2d_matches_reduceat(self):
        from sdr_module.gui.spectrum_widget import columns_max

        rng = np.random.default_rng(5)
        data = rng.integers(0, 255, (7, 2048), dtype=np.uint8)
        for cols in (850, 333, 100, 2047):
            edges = (np.arange(cols) * 2048 // cols).astype(np.intp)
            expected = np.maximum.reduceat(data, edges, axis=1)
            np.testing.assert_array_equal(columns_max(data, cols, axis=1), expected)
        line = rng.normal(-80, 5, 2048)
        self.assertEqual(len(columns_max(line, 900)), 900)
        self.assertEqual(columns_max(line, 900).max(), line.max())

    def test_narrow_carrier_keeps_full_colormap_color(self):
        """A 1-bin carrier must not be blurred away when 2048 bins share
        ~850 pixels (it used to render at a fraction of its brightness)."""
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        w = self._show(WaterfallWidget(history_size=100), (900, 300))
        rng = np.random.default_rng(1)
        carriers = (211, 1402, 1777)
        for _ in range(100):
            line = rng.normal(-90, 2, 2048)
            for c in carriers:
                line[c] = -20.0
            w.add_line(line)
        _settle()
        image = w._canvas.grab().toImage()
        rect = w.plot_rect()
        expected = tuple(
            int(v) for v in w._colormap[w._level_index(np.array([[-20.0]]))[0, 0]]
        )
        y = int(rect.top() + 20)
        for c in carriers:
            xc = int(rect.left() + (c + 0.5) / 2048 * rect.width())
            reds = [image.pixelColor(xc + dx, y).red() for dx in range(-2, 3)]
            self.assertGreaterEqual(max(reds), expected[0] - 3, f"carrier {c}: {reds}")

    def test_display_image_scroll_matches_rebuild(self):
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        w = self._show(WaterfallWidget(history_size=16), (700, 300))
        rng = np.random.default_rng(2)
        w.add_line(rng.uniform(-100, 0, 2048))
        cols = w._display_columns(w.plot_rect())
        self.assertLess(cols, 2048)
        w._display_image(cols)  # build the decimated image once
        for _ in range(20):  # then scroll it incrementally past history_size
            w.add_line(rng.uniform(-100, 0, 2048))
        scrolled = w._disp_rgb.copy()
        w._disp_rgb = None
        w._display_image(cols)
        np.testing.assert_array_equal(scrolled, w._disp_rgb)

    def test_time_axis_survives_a_pause(self):
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        w = WaterfallWidget(history_size=200)
        self._widgets.append(w)
        for _ in range(150):
            w.add_line(np.full(2048, -60.0))
        # 75 lines at 30 Hz, a 60 s pause, 75 more lines.
        times = [1000 + i / 30 for i in range(75)] + [1060 + i / 30 for i in range(75)]
        w._times.clear()
        w._times.extend(times)
        self.assertAlmostEqual(w._seconds_per_row(), 1 / 30, places=6)
        ages = w._row_ages()
        self.assertEqual(ages[0], 0.0)
        np.testing.assert_array_equal(w._gap_rows(ages, w._seconds_per_row()), [75])
        w.resize(700, 400)
        w._canvas.grab()  # paints the separator and labels without raising

    def test_elapsed_labels_fit_the_left_margin(self):
        from sdr_module.gui.spectrum_widget import axis_font, plot_side_margins
        from sdr_module.gui.waterfall_widget import format_elapsed

        fm = QFontMetrics(axis_font())
        left, _right = plot_side_margins(fm)
        for age in (0.0, 0.7, 9.94, 10, 65, 99.4, 99.6, 3000, 5969, 5971, 86400, 3.6e6):
            label = format_elapsed(age)
            self.assertLessEqual(len(label), 5, label)
            self.assertLessEqual(fm.horizontalAdvance(label), left - 8, label)

    def test_spectrum_ignores_empty_frames(self):
        from sdr_module.gui.spectrum_widget import SpectrumWidget

        w = SpectrumWidget()
        self._widgets.append(w)
        w.update_spectrum(np.array([]))  # used to raise ValueError
        self.assertFalse(w._has_data)
        w.update_spectrum(_signal_frame())
        before = w._spectrum.copy()
        w.update_spectrum([])
        np.testing.assert_array_equal(w._spectrum, before)

    def test_hover_readout_shown_before_any_data(self):
        """Click-to-tune works before Start, so hovering must say so."""
        import sdr_module.gui.spectrum_widget as sw_mod
        import sdr_module.gui.waterfall_widget as wf_mod

        seen = []

        def record(_painter, lines, *args, **kwargs):
            seen.append(list(lines))

        for mod, cls in (
            (sw_mod, sw_mod.SpectrumWidget),
            (wf_mod, wf_mod.WaterfallWidget),
        ):
            w = self._show(cls(), (700, 320))
            original = mod.draw_readout
            mod.draw_readout = record
            try:
                seen.clear()
                w._set_hover(w.plot_rect().center())
                w._canvas.grab()
            finally:
                mod.draw_readout = original
            self.assertTrue(seen, cls.__name__)
            self.assertTrue(seen[-1][0].endswith("MHz"), seen[-1])
            self.assertTrue(seen[-1][-1].startswith("Click to tune"), seen[-1])
            self.assertFalse(any("dBFS" in s for s in seen[-1]), seen[-1])

    def test_placeholder_masks_marker_and_grid(self):
        """The empty-state text sits on a backdrop, not across the marker."""
        from sdr_module.gui.spectrum_widget import SpectrumWidget

        w = self._show(SpectrumWidget(), (700, 320))
        image = w._canvas.grab().toImage()
        rect = w.plot_rect()
        x = int(round(rect.center().x()))

        def marker_pixels(y0, y1):
            # The tuned-frequency marker is warm (plot_marker is yellow/orange
            # in both themes); grid lines, text and background are not.
            count = 0
            for y in range(int(y0), int(y1)):
                c = image.pixelColor(x, y)
                if c.red() > 120 and c.red() - c.blue() > 30:
                    count += 1
            return count

        # The backdrop's top padding (above the title glyphs, whose subpixel
        # antialiasing can look warm too) must be free of the marker.
        from PyQt6.QtGui import QFont

        font = QFont(w._canvas.font())
        title_font = QFont(font)
        title_font.setPointSizeF(max(8.0, font.pointSizeF() + 1))
        title_font.setBold(True)
        total = QFontMetrics(title_font).height() + 4 + QFontMetrics(font).height()
        text_top = rect.center().y() - total / 2
        self.assertGreater(marker_pixels(rect.top() + 10, rect.top() + 40), 0)
        self.assertEqual(marker_pixels(text_top - 7, text_top - 1), 0)

    def test_header_hint_never_raises_minimum_width(self):
        from sdr_module.gui.spectrum_widget import SpectrumWidget
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        for cls in (SpectrumWidget, WaterfallWidget):
            w = self._show(cls(), (1000, 320))
            self.assertFalse(w._hint_label.isHidden())
            with_hint = w.minimumSizeHint().width()
            w._hint_label.hide()
            without_hint = w.minimumSizeHint().width()
            self.assertLessEqual(with_hint - without_hint, 10, cls.__name__)


if __name__ == "__main__":
    unittest.main()

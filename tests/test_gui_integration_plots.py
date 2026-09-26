#!/usr/bin/env python3
"""
Integration tests for the spectrum and waterfall plots: the keyboard focus
frame, and waterfall time labels that always fit the shared left margin.

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_integration_plots.py``
"""

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QFontMetrics
    from PyQt6.QtWidgets import QApplication, QLineEdit, QVBoxLayout, QWidget

    HAS_PYQT6 = True
    PYQT6_IMPORT_ERROR = ""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
except ImportError as _import_error:  # pragma: no cover - environment
    HAS_PYQT6 = False
    PYQT6_IMPORT_ERROR = str(_import_error)


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


def _frame(level=None, n=2048):
    if level is not None:
        return np.full(n, float(level))
    rng = np.random.default_rng(0)
    x = rng.normal(-85, 2, n)
    x[1000:1050] = -20.0
    return x


class TestElapsedLabels(unittest.TestCase):
    """format_elapsed() keeps its four-character promise at every age."""

    def setUp(self):
        require_pyqt6(self)

    def test_just_below_ten_seconds_rounds_to_whole_seconds(self):
        from sdr_module.gui.waterfall_widget import format_elapsed

        self.assertEqual(format_elapsed(9.94), "9.9s")
        self.assertEqual(format_elapsed(9.96), "10s")
        self.assertEqual(format_elapsed(9.999), "10s")
        self.assertEqual(format_elapsed(10.0), "10s")

    def test_every_age_fits_four_characters_and_the_margin(self):
        from sdr_module.gui.spectrum_widget import axis_font, plot_side_margins
        from sdr_module.gui.waterfall_widget import format_elapsed

        fm = QFontMetrics(axis_font())
        left, _right = plot_side_margins(fm)
        ages = np.concatenate(
            [
                np.linspace(0, 12, 12001),  # every 1 ms around the 10 s edge
                np.linspace(90, 110, 2001),
                np.geomspace(100, 1e12, 4000),
                [-5.0, 99.5 * 60 - 1e-6, 99.5 * 3600 - 1e-6, 999.5 * 86400 - 1],
            ]
        )
        for age in ages.tolist():
            label = format_elapsed(age)
            self.assertLessEqual(len(label), 4, (age, label))
            self.assertLessEqual(fm.horizontalAdvance(label), left - 8, (age, label))

    def test_non_finite_age_does_not_raise(self):
        from sdr_module.gui.waterfall_widget import format_elapsed

        for age in (float("nan"), float("inf"), -float("inf")):
            self.assertEqual(format_elapsed(age), "--")


class TestTimeAxisLabels(unittest.TestCase):
    """Tick labels on a tall waterfall never spill past the canvas edge."""

    def setUp(self):
        require_pyqt6(self)

    def test_step_is_coarse_enough_for_four_character_labels(self):
        from sdr_module.gui.spectrum_widget import nice_ticks
        from sdr_module.gui.waterfall_widget import format_age, time_step

        # 0.5 s steps would need "10.5s" once the history reaches 10 s.
        self.assertEqual(time_step(0.45), 0.5)
        self.assertEqual(time_step(0.45, max_age=9.9), 0.5)
        self.assertEqual(time_step(0.45, max_age=16.6), 1.0)
        # 30 s steps would need "1200s": minutes instead.
        self.assertEqual(time_step(28, max_age=1200), 60.0)
        for raw in (0.05, 0.3, 0.45, 2, 12, 28, 45, 100, 900):
            for max_age in (5, 9.9, 16.6, 120, 999, 1200, 5000, 40000):
                step = time_step(raw, max_age=max_age)
                self.assertGreaterEqual(step, raw)
                for age in nice_ticks(0.0, max_age, step):
                    label = format_age(age, step)
                    self.assertLessEqual(len(label), 4, (raw, max_age, step, label))

    def test_tall_waterfall_labels_stay_inside_the_canvas(self):
        from sdr_module.gui import waterfall_widget as wf
        from sdr_module.gui.themes import get_palette

        w = wf.WaterfallWidget()
        try:
            for _ in range(500):
                w.add_line(_frame(-60))
            # 30 lines per second: ~16.6 s of history over a very tall plot.
            w._times.clear()
            w._times.extend(1000 + i / 30 for i in range(500))
            w.resize(700, 1150)
            w.show()
            _settle()
            calls = []
            original = wf.time_step

            def record(raw, **kwargs):
                step = original(raw, **kwargs)
                calls.append((raw, kwargs.get("max_age"), step))
                return step

            with mock.patch.object(wf, "time_step", side_effect=record):
                img = w._canvas.grab().toImage()
            # The plot is tall enough for 0.5 s steps, which would need
            # "10.0s" ... "16.5s"; whole seconds are used instead.
            [(raw, max_age, step)] = calls
            self.assertEqual(original(raw), 0.5)
            self.assertGreater(max_age, 16)
            self.assertEqual(step, 1.0)
            # Nothing is painted in the two leftmost columns (a too-wide
            # right-aligned label would start there, clipped).
            bg = get_palette().qcolor("plot_bg").rgb()
            for x in (0, 1):
                column = {img.pixel(x, y) for y in range(img.height())}
                self.assertEqual(column, {bg}, f"text clipped at x={x}")
        finally:
            w.close()
            w.deleteLater()


class TestPlotFocusFrame(unittest.TestCase):
    """A focused plot (Space and Left/Right act on it) shows an accent frame."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import themes
        from sdr_module.gui.spectrum_widget import SpectrumWidget
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        self.themes = themes
        self._previous = themes.current_theme()
        self.window = QWidget()
        layout = QVBoxLayout(self.window)
        self.other = QLineEdit()
        self.spectrum = SpectrumWidget()
        self.waterfall = WaterfallWidget()
        for plot in (self.spectrum, self.waterfall):
            # As in the main window: a click, Tab or F6 focuses a plot.
            plot.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        layout.addWidget(self.other)
        layout.addWidget(self.spectrum)
        layout.addWidget(self.waterfall)
        self.window.resize(800, 700)
        self.window.show()
        self.window.activateWindow()
        _settle()
        self.spectrum.update_spectrum(_frame())
        for _ in range(20):
            self.waterfall.add_line(_frame(-40))

    def tearDown(self):
        self.themes.apply_theme(QApplication.instance(), self._previous)
        self.window.close()
        self.window.deleteLater()
        _settle()

    def _focus(self, widget):
        widget.setFocus(Qt.FocusReason.TabFocusReason)
        _settle()
        self.assertTrue(widget.hasFocus())

    def test_frame_is_accent_only_while_focused_in_both_themes(self):
        for theme in ("dark", "light"):
            self.themes.apply_theme(QApplication.instance(), theme)
            _settle()
            accent = self.themes.get_palette(theme).qcolor("accent").rgb()
            for plot in (self.spectrum, self.waterfall):
                with self.subTest(theme=theme, plot=type(plot).__name__):
                    rect = plot.plot_rect().toRect()
                    y = rect.center().y()
                    x = rect.center().x() + 7  # clear of the center marker
                    edges = [
                        (rect.left() - 1, y),
                        (rect.left() - 2, y),
                        (rect.right() + 1, y),
                        (rect.right() + 2, y),
                        (x, rect.top() - 1),
                        (x, rect.top() - 2),
                        (x, rect.bottom() + 1),
                        (x, rect.bottom() + 2),
                    ]
                    self._focus(self.other)
                    idle = plot._canvas.grab().toImage()
                    self._focus(plot)
                    focused = plot._canvas.grab().toImage()
                    for px, py in edges:
                        self.assertNotEqual(idle.pixel(px, py), accent, (px, py))
                        self.assertEqual(focused.pixel(px, py), accent, (px, py))
                    # The frame sits outside the plot: no data is covered.
                    for py in (rect.top() + 2, y, rect.bottom() - 2):
                        for px in (rect.left(), rect.right()):
                            self.assertEqual(
                                focused.pixel(px, py), idle.pixel(px, py), (px, py)
                            )

    def test_focus_change_repaints_the_canvas(self):
        for plot in (self.spectrum, self.waterfall):
            self._focus(self.other)
            with mock.patch.object(plot._canvas, "update") as update:
                self._focus(plot)
                self.assertTrue(update.called, type(plot).__name__)
            with mock.patch.object(plot._canvas, "update") as update:
                self._focus(self.other)
                self.assertTrue(update.called, type(plot).__name__)

    def test_waterfall_ticks_do_not_mark_the_image(self):
        """Time-axis ticks end on the frame, not on the first image column."""
        w = self.waterfall
        while len(w._history) < w._history_size:
            w.add_line(_frame(-40))
        w._times.clear()
        w._times.extend(1000 + i / 30 for i in range(len(w._history)))
        img = w._canvas.grab().toImage()
        rect = w.plot_rect().toRect()
        inner = img.pixel(rect.left() + 3, rect.center().y())
        first_column = {
            img.pixel(rect.left(), y) for y in range(rect.top(), rect.bottom())
        }
        self.assertEqual(first_column, {inner})


if __name__ == "__main__":
    unittest.main()

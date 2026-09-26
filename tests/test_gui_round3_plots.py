#!/usr/bin/env python3
"""
Spectrum and waterfall plots: the third review round.

The demodulated channel (passband) shaded on the spectrum and bracketed on
the waterfall, the empty-state title sized above its hint for pixel-sized
fonts, waterfall time-axis labels that never overlap after Stop -> Start,
and header strips whose controls line up with the plot below.

Run offscreen:
``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_round3_plots.py``
"""

import math
import os
import unittest
from itertools import combinations
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt6.QtCore import QRectF
    from PyQt6.QtGui import QColor, QFont, QFontMetrics
    from PyQt6.QtWidgets import QApplication, QComboBox, QPushButton

    HAS_PYQT6 = True
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
except ImportError:  # pragma: no cover - environment
    HAS_PYQT6 = False


def _settle():
    for _ in range(3):
        QApplication.processEvents()


def _flat_frame(level=-95.0, n=2048):
    return np.full(n, level)


def _dist(a: "QColor", b: "QColor") -> int:
    return (
        abs(a.red() - b.red()) + abs(a.green() - b.green()) + abs(a.blue() - b.blue())
    )


class _PlotCase(unittest.TestCase):
    """Creates plots at a fixed size and cleans them up."""

    size = (900, 300)

    def setUp(self):
        if not HAS_PYQT6:
            self.skipTest("PyQt6 not available")
        self._widgets = []

    def tearDown(self):
        for w in self._widgets:
            w.close()
            w.deleteLater()
        _settle()

    def _show(self, widget):
        self._widgets.append(widget)
        widget.resize(*self.size)
        widget.show()
        _settle()
        return widget

    def spectrum(self, passband=None, data=True):
        from sdr_module.gui.spectrum_widget import SpectrumWidget

        w = SpectrumWidget()
        w.set_frequency_range(100.0e6, 2.4e6)
        if passband is not None:
            w.set_passband(*passband)
        if data:
            w.update_spectrum(_flat_frame())
        return self._show(w)

    def waterfall(self, passband=None):
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        w = WaterfallWidget()
        w.set_frequency_range(100.0e6, 2.4e6)
        if passband is not None:
            w.set_passband(*passband)
        return self._show(w)


# ---------------------------------------------------------------------------
# Passband
# ---------------------------------------------------------------------------


class TestPassbandGeometry(unittest.TestCase):
    """passband_px(): where the band goes, in whole pixels."""

    def setUp(self):
        if not HAS_PYQT6:
            self.skipTest("PyQt6 not available")
        # 1000 px for 1 MHz: 1 px per kHz, tuned frequency at x = 500.
        self.plot = QRectF(0, 0, 1000, 100)

    def px(self, band, span=1e6):
        from sdr_module.gui.spectrum_widget import passband_px

        return passband_px(band, self.plot, span)

    def test_wide_band_maps_to_its_offsets(self):
        self.assertEqual(self.px((-100e3, 100e3)), (400.0, 600.0))
        self.assertEqual(self.px((-12.5e3, 12.5e3)), (488.0, 513.0))

    def test_narrow_upper_sideband_grows_right_of_the_marker(self):
        x0, x1 = self.px((0.0, 2800.0))
        self.assertEqual(x0, 500.0)
        self.assertGreaterEqual(x1 - x0, 3.0)

    def test_narrow_lower_sideband_grows_left_of_the_marker(self):
        x0, x1 = self.px((-2800.0, 0.0))
        self.assertEqual(x1, 500.0)
        self.assertGreaterEqual(x1 - x0, 3.0)

    def test_narrow_cw_band_stays_centered(self):
        x0, x1 = self.px((-250.0, 250.0))
        self.assertGreaterEqual(x1 - x0, 3.0)
        self.assertAlmostEqual((x0 + x1) / 2, 500.5, delta=1.0)

    def test_nothing_to_draw(self):
        from sdr_module.gui.spectrum_widget import passband_px

        self.assertIsNone(self.px(None))
        self.assertIsNone(self.px((-1e3, 1e3), span=0))
        self.assertIsNone(passband_px((-1e3, 1e3), QRectF(0, 0, 0, 100), 1e6))


class TestSpectrumPassband(_PlotCase):
    def test_set_passband_normalizes_and_hides(self):
        w = self.spectrum()
        self.assertIsNone(w.passband())
        w.set_passband(-12500, 12500)
        self.assertEqual(w.passband(), (-12500.0, 12500.0))
        for low, high in (
            (None, None),
            (None, 1e3),
            (1e3, None),
            (2800, 2800),
            (1e3, -1e3),
            (math.nan, 1e3),
            (-math.inf, 1e3),
            ("wide", 1e3),
        ):
            w.set_passband(-12500, 12500)
            w.set_passband(low, high)
            self.assertIsNone(w.passband(), (low, high))

    def test_band_is_shaded_with_the_accent_inside_the_plot_only(self):
        from sdr_module.gui.themes import get_palette

        base = self.spectrum()
        shaded = self.spectrum(passband=(-100e3, 100e3))
        before = base._canvas.grab().toImage()
        after = shaded._canvas.grab().toImage()
        plot = shaded.plot_rect()
        cx = plot.center().x()
        px_per_hz = plot.width() / 2.4e6
        y = int(plot.top() + plot.height() * 0.3)  # above the trace, off the grid
        accent = get_palette().qcolor("accent")

        inside = int(cx + 50e3 * px_per_hz)
        self.assertNotEqual(after.pixelColor(inside, y), before.pixelColor(inside, y))
        self.assertLess(
            _dist(after.pixelColor(inside, y), accent),
            _dist(before.pixelColor(inside, y), accent),
        )
        # A light tint, not a solid block: the plot background still dominates.
        self.assertLess(
            _dist(after.pixelColor(inside, y), before.pixelColor(inside, y)),
            _dist(accent, before.pixelColor(inside, y)) // 3,
        )
        outside = int(cx + 150e3 * px_per_hz)
        self.assertEqual(after.pixelColor(outside, y), before.pixelColor(outside, y))

    def test_edges_are_marked_more_strongly_than_the_fill(self):
        from sdr_module.gui.spectrum_widget import passband_px
        from sdr_module.gui.themes import get_palette

        w = self.spectrum(passband=(-100e3, 100e3))
        image = w._canvas.grab().toImage()
        plot = w.plot_rect()
        x0, x1 = passband_px(w.passband(), plot, 2.4e6)
        y = int(plot.top() + plot.height() * 0.3)
        accent = get_palette().qcolor("accent")
        interior = image.pixelColor(int((x0 + x1) / 2) + 5, y)
        for edge_x in (int(x0), int(x1) - 1):
            self.assertLess(
                _dist(image.pixelColor(edge_x, y), accent), _dist(interior, accent)
            )

    def test_band_wider_than_the_span_is_clipped_to_the_plot(self):
        base = self.spectrum()
        wide = self.spectrum(passband=(-5e6, 5e6))
        before = base._canvas.grab().toImage()
        after = wide._canvas.grab().toImage()
        plot = wide.plot_rect()
        y = int(plot.top() + plot.height() * 0.3)
        # Margins (axis labels) are untouched, the plot is shaded edge to edge.
        for x in (int(plot.left()) - 3, int(plot.right()) + 3):
            self.assertEqual(after.pixelColor(x, y), before.pixelColor(x, y), x)
        for x in (int(plot.left()) + 2, int(plot.right()) - 3):
            self.assertNotEqual(after.pixelColor(x, y), before.pixelColor(x, y), x)
        # Above and below the plot (legend, frequency labels) too.
        cx = int(plot.center().x()) + 20
        for yy in (int(plot.top()) - 4, int(plot.bottom()) + 4):
            self.assertEqual(after.pixelColor(cx, yy), before.pixelColor(cx, yy), yy)

    def test_tuned_marker_stays_on_top(self):
        base = self.spectrum()
        shaded = self.spectrum(passband=(-100e3, 100e3))
        plot = shaded.plot_rect()
        x = round(plot.center().x())
        y = int(plot.top()) + 2  # inside the marker's flag
        self.assertEqual(
            shaded._canvas.grab().toImage().pixelColor(x, y),
            base._canvas.grab().toImage().pixelColor(x, y),
        )

    def test_hiding_restores_the_plain_plot(self):
        base = self.spectrum()
        w = self.spectrum(passband=(0, 2800))
        w.set_passband(None, None)
        _settle()
        self.assertEqual(w._canvas.grab().toImage(), base._canvas.grab().toImage())

    def test_shaded_before_any_data(self):
        base = self.spectrum(data=False)
        w = self.spectrum(passband=(-100e3, 100e3), data=False)
        plot = w.plot_rect()
        x = int(plot.center().x() + 60e3 * plot.width() / 2.4e6)
        y = int(plot.top() + 4)  # above the "No spectrum yet" backdrop
        self.assertNotEqual(
            w._canvas.grab().toImage().pixelColor(x, y),
            base._canvas.grab().toImage().pixelColor(x, y),
        )

    def test_main_window_contract(self):
        """main_window._update_passband passes _channel_edges() straight in."""
        from sdr_module.gui import main_window

        w = self.spectrum()
        stub = mock.Mock()
        stub._spectrum = w
        stub._control_panel._demod_combo.currentText.return_value = "USB"
        stub._channel_bandwidth.return_value = 25e3
        stub._device_sample_rate.return_value = 2.4e6
        main_window.SDRMainWindow._update_passband(stub)
        self.assertEqual(w.passband(), (0.0, 2800.0))
        stub._control_panel._demod_combo.currentText.return_value = "FM"
        main_window.SDRMainWindow._update_passband(stub)
        self.assertEqual(w.passband(), (-12500.0, 12500.0))


class TestWaterfallPassband(_PlotCase):
    def test_bracket_on_the_top_edge_only(self):
        from sdr_module.gui.themes import get_palette

        base = self.waterfall()
        w = self.waterfall(passband=(-100e3, 100e3))
        self.assertEqual(w.passband(), (-100e3, 100e3))
        before = base._canvas.grab().toImage()
        after = w._canvas.grab().toImage()
        plot = w.plot_rect()
        x = int(plot.center().x() + 60e3 * plot.width() / 2.4e6)
        top = int(plot.top())
        self.assertEqual(after.pixelColor(x, top), get_palette().qcolor("accent"))
        # Everything below the bracket's end ticks is untouched.
        for y in range(top + 10, int(plot.bottom()), 7):
            for xx in range(int(plot.left()), int(plot.right()), 23):
                self.assertEqual(after.pixelColor(xx, y), before.pixelColor(xx, y))

    def test_hidden_by_default_and_on_request(self):
        base = self.waterfall()
        w = self.waterfall(passband=(-100e3, 100e3))
        w.set_passband(5, 5)
        self.assertIsNone(w.passband())
        _settle()
        self.assertEqual(w._canvas.grab().toImage(), base._canvas.grab().toImage())


# ---------------------------------------------------------------------------
# Placeholder title
# ---------------------------------------------------------------------------


class TestPlaceholderTitle(unittest.TestCase):
    def setUp(self):
        if not HAS_PYQT6:
            self.skipTest("PyQt6 not available")

    def _fonts(self, font):
        from sdr_module.gui.spectrum_widget import draw_placeholder
        from sdr_module.gui.themes import get_palette

        painter = mock.MagicMock()
        draw_placeholder(
            painter, QRectF(0, 0, 600, 200), "Title", "A hint", font, get_palette()
        )
        title_font, hint_font = (c.args[0] for c in painter.setFont.call_args_list)
        return title_font, hint_font

    def test_pixel_sized_font(self):
        """The stylesheet sizes widget fonts in pixels (pointSizeF() is -1)."""
        font = QFont()
        font.setPixelSize(12)
        title, hint = self._fonts(font)
        self.assertEqual(title.pixelSize(), 14)
        self.assertTrue(title.bold())
        self.assertEqual(hint.pixelSize(), 12)
        self.assertGreater(QFontMetrics(title).height(), QFontMetrics(hint).height())

    def test_point_sized_font(self):
        font = QFont()
        font.setPointSizeF(9.0)
        title, hint = self._fonts(font)
        self.assertEqual(title.pointSizeF(), 10.0)
        self.assertTrue(title.bold())
        self.assertEqual(hint.pointSizeF(), 9.0)


# ---------------------------------------------------------------------------
# Waterfall time axis
# ---------------------------------------------------------------------------


class TestTimeAxisLabels(_PlotCase):
    size = (700, 300)

    def layout(self, w, times):
        w._times.clear()
        w._times.extend(times)
        return w._time_axis_layout(w.plot_rect())

    def assert_clean(self, w, labels):
        plot = w.plot_rect()
        fh = w._fm.height()
        texts = [text for _t, _y, text in labels]
        self.assertEqual(texts[0], "0s", texts)
        for (_t1, y1, s1), (_t2, y2, s2) in combinations(labels, 2):
            self.assertGreaterEqual(abs(y1 - y2), fh + 2, f"{s1!r} and {s2!r} overlap")
        for _t, y, text in labels:
            self.assertGreaterEqual(y, plot.top() + fh / 2 - 1e-9, text)
            self.assertLessEqual(y, plot.bottom() - fh / 2 + 1e-9, text)

    def test_no_overlap_while_the_newest_run_grows_after_a_pause(self):
        """Stop -> Start: the "0s" label must never run into the pause label."""
        w = self.waterfall()
        old = [1000 + i / 30 for i in range(300)]
        pause_labels = 0
        for fresh in range(1, 120):
            new = [1030 + i / 30 for i in range(fresh)]
            separators, labels = self.layout(w, old + new)
            self.assertEqual(len(separators), 1)
            self.assert_clean(w, labels)
            pause_labels += any(t == separators[0] for t, _y, _s in labels)
        # Once the newest run is tall enough, the pause is labelled too.
        self.assertGreater(pause_labels, 50)

    def test_pause_label_sits_on_its_separator(self):
        w = self.waterfall()
        times = [1000 + i / 30 for i in range(300)] + [
            1060 + i / 30 for i in range(150)
        ]
        separators, labels = self.layout(w, times)
        (sep,) = separators
        gap = [(t, y, s) for t, y, s in labels if t == sep]
        self.assertEqual(len(gap), 1)
        self.assertEqual(gap[0][1], sep)  # not clamped: well inside the plot
        self.assertTrue(gap[0][2].endswith("s"))
        self.assert_clean(w, labels)

    def test_several_pauses_and_one_at_the_bottom(self):
        w = self.waterfall()
        times = []
        t = 1000.0
        for run in (200, 3, 3, 200, 60, 34):
            times += [t + i / 30 for i in range(run)]
            t = times[-1] + 20
        separators, labels = self.layout(w, times)
        self.assertEqual(len(separators), 5)
        self.assert_clean(w, labels)

    def test_no_labels_without_timing(self):
        w = self.waterfall()
        self.assertEqual(self.layout(w, []), ([], []))
        self.assertEqual(self.layout(w, [1000.0]), ([], []))

    def test_paints_with_a_pause(self):
        w = self.waterfall()
        for _ in range(40):
            w.add_line(_flat_frame(-60.0))
        w._times.clear()
        w._times.extend([1000 + i / 30 for i in range(30)] + [1050, 1050.03])
        w._canvas.grab()  # draws separators and labels without raising


# ---------------------------------------------------------------------------
# Header strips
# ---------------------------------------------------------------------------


class TestThemedPlots(_PlotCase):
    """Header strips and placeholders under the real application theme."""

    @classmethod
    def setUpClass(cls):
        if not HAS_PYQT6:
            return
        from sdr_module.gui import themes

        cls._theme = themes.current_theme()
        themes.apply_theme(QApplication.instance(), "dark")

    @classmethod
    def tearDownClass(cls):
        if not HAS_PYQT6:
            return
        from sdr_module.gui import themes

        themes.apply_theme(QApplication.instance(), cls._theme)

    def _controls(self, w):
        return [
            c for c in w._header.findChildren((QComboBox, QPushButton)) if c.isVisible()
        ]

    def test_last_control_ends_on_the_plot_frame(self):
        for w in (self.spectrum(), self.waterfall()):
            last = max(self._controls(w), key=lambda c: c.geometry().right())
            # The plot frame is drawn in the pixel column at plot.right().
            self.assertEqual(last.geometry().right(), int(w.plot_rect().right()))

    def test_compact_buttons_match_the_combos(self):
        for w in (self.spectrum(), self.waterfall()):
            controls = self._controls(w)
            heights = {c.height() for c in controls}
            centers = {c.geometry().center().y() for c in controls}
            self.assertEqual(len(heights), 1, heights)
            self.assertEqual(len(centers), 1, centers)
            buttons = [c for c in controls if isinstance(c, QPushButton)]
            self.assertTrue(buttons)
            for b in buttons:
                self.assertEqual(b.property("role"), "compact")

    def test_placeholder_title_is_larger_than_its_hint(self):
        """The stylesheet gives the canvases a pixel-sized font."""
        from sdr_module.gui import spectrum_widget, waterfall_widget

        for w, module in (
            (self.spectrum(data=False), spectrum_widget),
            (self.waterfall(), waterfall_widget),
        ):
            font = w._canvas.font()
            self.assertGreater(font.pixelSize(), 0)
            with mock.patch.object(module, "draw_placeholder") as draw:
                w._canvas.grab()
            self.assertTrue(draw.called, w)
            self.assertEqual(draw.call_args.args[4], font)
            painter = mock.MagicMock()
            spectrum_widget.draw_placeholder(
                painter, w.plot_rect(), "Title", "Hint", font, draw.call_args.args[5]
            )
            title_font = painter.setFont.call_args_list[0].args[0]
            self.assertEqual(title_font.pixelSize(), font.pixelSize() + 2)
            self.assertGreater(
                QFontMetrics(title_font).height(), QFontMetrics(font).height()
            )


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
Zoom (rx:G-02, ux:UX-U09): x1 to x16 around the tuned frequency (View >
Zoom, Ctrl+= / Ctrl+- / Ctrl+0) or the pointer (Ctrl+wheel), on the
spectrum and the waterfall together, with their axes, hover, click-to-tune,
passband and measurement in step.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_plot_zoom.py``
"""

import json
import os
import types
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
    from PyQt6.QtCore import QPoint, QPointF, Qt
    from PyQt6.QtGui import QKeySequence, QWheelEvent
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QMenu

    from sdr_module.gui import spectrum_widget as swm
    from sdr_module.gui import waterfall_widget as wfm
    from sdr_module.gui.plot_view import PlotView

RATE = 2.4e6
CENTER = 100e6
N = 2048


def _settle(n=3):
    for _ in range(n):
        QApplication.processEvents()


def _line(carriers=(), seed=0, floor=-90.0):
    rng = np.random.default_rng(seed)
    line = rng.normal(floor, 1.0, N)
    for freq, level in carriers:
        b = int(round(N / 2 + (freq - CENTER) / (RATE / N)))
        line[b] = level
    return line


def _wheel(widget, pos, notches=1, mods=None):
    mods = mods if mods is not None else Qt.KeyboardModifier.NoModifier
    event = QWheelEvent(
        QPointF(pos),
        QPointF(widget.mapToGlobal(QPointF(pos).toPoint())),
        QPoint(0, 0),
        QPoint(0, 120 * notches),
        Qt.MouseButton.NoButton,
        mods,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t


class _PlotPair(unittest.TestCase):
    """A spectrum and a waterfall sharing one view, as in the window."""

    def setUp(self):
        require_pyqt6(self)
        self.clock = _Clock()
        patcher = mock.patch.object(
            wfm, "time", types.SimpleNamespace(monotonic=self.clock.monotonic)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.view = PlotView()
        self.sp = swm.SpectrumWidget()
        self.wf = wfm.WaterfallWidget(history_size=100)
        for w in (self.sp, self.wf):
            w.set_view_model(self.view)
            w.set_frequency_range(CENTER, RATE)
            w.resize(1000, 360)
            w.show()
            self.addCleanup(w.deleteLater)
        _settle()

    def feed(self, n=60, carriers=((100.32e6, -30.0),)):
        for i in range(n):
            self.clock.t += 1 / 30
            line = _line(carriers, seed=i)
            self.sp.update_spectrum(line)
            self.wf.add_line(line)


class TestZoomedPlots(_PlotPair):
    def test_both_plots_zoom_together_around_the_tuned_frequency(self):
        for w in (self.sp, self.wf):
            w.set_tuned_frequency(100.3e6)
        self.sp.set_zoom(4)
        for w in (self.sp, self.wf):
            plot = w.plot_rect()
            self.assertAlmostEqual(
                w.frequency_at(plot.center().x()), 100.3e6, delta=1e3
            )
            self.assertAlmostEqual(
                w.frequency_at(plot.right()) - w.frequency_at(plot.left()),
                600e3,
                delta=1,
            )
        self.assertEqual(self.wf.view_model().zoom, 4)

    def test_the_trace_and_the_waterfall_show_a_carrier_where_it_is(self):
        self.feed()
        self.sp.set_tuned_frequency(100.3e6)
        self.sp.set_zoom(8)
        plot = self.sp.plot_rect()
        xs, ys = self.sp._trace_xy(self.sp._spectrum, plot, int(plot.width()))
        peak_x = float(xs[int(np.argmin(ys))])
        self.assertAlmostEqual(self.sp.frequency_at(peak_x), 100.32e6, delta=RATE / N)
        # The waterfall's newest row, drawn in the same place.
        wplot = self.wf.plot_rect()
        cols = self.wf._display_columns(wplot)
        self.wf._display_image(cols)
        codes = self.wf._disp_codes[0]
        target = self.wf._display_target(wplot)
        col = int(np.argmax(codes))
        x = target.left() + (col + 0.5) * target.width() / codes.shape[0]
        self.assertAlmostEqual(self.wf.frequency_at(x), 100.32e6, delta=RATE / N)
        self.assertLess(abs(x - peak_x), 3.0)  # lined up with the spectrum

    def test_clicks_hover_and_measure_use_the_zoomed_axis(self):
        self.feed()
        self.sp.set_tuned_frequency(100.3e6)
        self.sp.set_zoom(16)  # 150 kHz across 1000 px: ~150 Hz per pixel
        tuned = []
        self.sp.frequency_clicked.connect(tuned.append)
        x = self.sp._freq_x(100.31e6, self.sp.plot_rect())
        QTest.mouseClick(
            self.sp._canvas,
            Qt.MouseButton.LeftButton,
            pos=QPointF(x, self.sp.plot_rect().center().y()).toPoint(),
        )
        self.assertAlmostEqual(tuned[0], 100.31e6, delta=200)
        # The waterfall measures the zoomed carrier.
        plot = self.wf.plot_rect()
        x0 = self.wf._freq_to_x(100.315e6, plot)
        x1 = self.wf._freq_to_x(100.325e6, plot)
        self.wf._drag_measure(QPointF(x0, plot.top() + 5), QPointF(x1, plot.top() + 80))
        self.wf._finish_measure()
        stats = self.wf.measurement()
        self.assertAlmostEqual(stats["bandwidth"], 10e3, delta=300)
        self.assertAlmostEqual(stats["peak_freq"], 100.32e6, delta=RATE / N)
        self.assertGreater(stats["peak_db"], -31.0)

    def test_the_passband_follows_the_zoom(self):
        for w in (self.sp, self.wf):
            w.set_tuned_frequency(100.3e6)
            w.set_passband(300e3 - 12.5e3, 300e3 + 12.5e3)
        self.sp.set_zoom(4)
        plot = self.sp.plot_rect()
        extent = swm.passband_px(
            self.sp.passband(),
            plot,
            self.view.span,
            300e3,
            view_offset_hz=CENTER - self.view.center,
        )
        mid = (extent[0] + extent[1]) / 2
        self.assertAlmostEqual(self.sp.frequency_at(mid), 100.3e6, delta=2e3)
        self.assertAlmostEqual(
            extent[1] - extent[0], 25e3 / 600e3 * plot.width(), delta=2
        )

    def test_ctrl_wheel_zooms_around_the_pointer(self):
        plot = self.sp.plot_rect()
        x = plot.left() + plot.width() * 0.8
        under = self.sp.frequency_at(x)
        _wheel(
            self.sp._canvas,
            QPointF(x, plot.center().y()),
            1,
            Qt.KeyboardModifier.ControlModifier,
        )
        self.assertEqual(self.view.zoom, 2)
        self.assertAlmostEqual(self.sp.frequency_at(x), under, delta=RATE / 1000)
        # The waterfall's Ctrl+wheel zooms the same view.
        _wheel(
            self.wf._canvas,
            QPointF(x, plot.center().y()),
            1,
            Qt.KeyboardModifier.ControlModifier,
        )
        self.assertEqual(self.view.zoom, 4)
        _wheel(
            self.wf._canvas,
            QPointF(x, plot.center().y()),
            -2,
            Qt.KeyboardModifier.ControlModifier,
        )
        self.assertEqual(self.view.zoom, 1)

    def test_a_zoomed_view_follows_the_tuned_frequency(self):
        self.sp.set_tuned_frequency(100.0e6)
        self.sp.set_zoom(8)
        self.assertTrue(self.view.contains(100.0e6))
        self.sp.set_tuned_frequency(100.9e6)  # tuned out of sight
        self.assertTrue(self.view.contains(100.9e6))

    def test_zoom_in_here_from_either_context_menu(self):
        def actions(widget, pos):
            captured = {}

            def fake_exec(menu, *_args):
                for act in menu.actions():
                    captured[act.text().replace("&", "")] = act

            with mock.patch.object(QMenu, "exec", fake_exec):
                widget._show_context_menu(pos, QPoint(0, 0))
            return captured

        plot = self.sp.plot_rect()
        pos = QPointF(plot.left() + plot.width() * 0.25, plot.center().y())
        freq = self.sp._snapped_frequency_at(pos.x())
        actions(self.sp, pos)["Zoom In Here"].trigger()
        self.assertEqual(self.view.zoom, 2)
        self.assertAlmostEqual(self.view.center, freq, delta=1)
        self.feed(5)
        wplot = self.wf.plot_rect()
        wpos = QPointF(wplot.center().x(), wplot.top() + 10)
        actions(self.wf, wpos)["Zoom In Here"].trigger()
        self.assertEqual(self.view.zoom, 4)

    def test_frequency_ticks_cover_the_zoomed_span(self):
        self.sp.set_tuned_frequency(100.3e6)
        self.sp.set_zoom(16)
        fm = self.sp._fm
        ticks, decimals = swm.freq_ticks(
            self.view.start, self.view.end, self.sp.plot_rect().width(), fm
        )
        self.assertTrue(all(self.view.start <= t <= self.view.end for t in ticks))
        self.assertGreaterEqual(len(ticks), 3)
        labels = [swm.format_mhz(t, decimals) for t in ticks]
        self.assertEqual(len(set(labels)), len(labels), labels)  # tell apart
        self.sp._canvas.grab()  # paints the zoomed axes
        self.wf.set_frequency_axis_visible(True)
        self.wf._canvas.grab()


class TestWindowZoom(_WindowTestCase):
    demo = True

    def test_view_menu_zoom_actions_and_keys(self):
        win = self.win
        shortcuts = {
            act.text().replace("&", ""): [s.toString() for s in act.shortcuts()]
            for act in (
                win._zoom_in_action,
                win._zoom_out_action,
                win._zoom_full_action,
            )
        }
        self.assertIn("Ctrl+=", shortcuts["Zoom In"])
        self.assertEqual(shortcuts["Zoom Out"], ["Ctrl+-"])
        self.assertEqual(shortcuts["Full Span"], ["Ctrl+0"])
        win.set_frequency(100.3e6)
        win._zoom_in_action.trigger()
        win._zoom_in_action.trigger()
        view = win._plot_view
        self.assertEqual(view.zoom, 4)
        self.assertAlmostEqual(view.center, 100.3e6, delta=1)
        self.assertIs(win._spectrum.view_model(), win._waterfall.view_model())
        self.assertIn("Zoom ×4", win._message_label.text())
        win._zoom_out_action.trigger()
        self.assertEqual(view.zoom, 2)
        win._zoom_full_action.trigger()
        self.assertEqual(view.zoom, 1)
        win._zoom_out_action.trigger()
        self.assertIn("full span", win._message_label.text())

    def test_ctrl_plus_minus_and_zero_from_the_keyboard(self):
        win = self.win
        win._spectrum.setFocus()
        self.settle()
        QTest.keySequence(win, QKeySequence("Ctrl+="))
        self.assertEqual(win._plot_view.zoom, 2)
        QTest.keySequence(win, QKeySequence("Ctrl+-"))
        self.assertEqual(win._plot_view.zoom, 1)
        QTest.keySequence(win, QKeySequence("Ctrl+="))
        QTest.keySequence(win, QKeySequence("Ctrl+="))
        QTest.keySequence(win, QKeySequence("Ctrl+0"))
        self.assertEqual(win._plot_view.zoom, 1)

    def test_zoom_is_remembered(self):
        win = self.win
        win._zoom_in_action.trigger()
        win._zoom_in_action.trigger()
        win._zoom_in_action.trigger()
        saved = json.loads(self.store.values["spectrum_display"])
        self.assertEqual(saved["zoom"], 8)
        other = self.mw.SDRMainWindow()
        try:
            for timer in (
                other._display_timer,
                other._status_timer,
                other._hotplug_timer,
            ):
                timer.stop()
            self.assertEqual(other._plot_view.zoom, 8)
            self.assertTrue(other._plot_view.contains(other._current_frequency()))
        finally:
            other.close()
            other.deleteLater()

    def test_tuning_while_zoomed_keeps_the_vfo_in_view(self):
        win = self.win
        win.set_frequency(100.1e6)
        for _ in range(4):
            win._zoom_in_action.trigger()  # x16: 150 kHz
        view = win._plot_view
        for _ in range(5):
            win._tune_steps(1)  # 100 kHz steps
            self.assertTrue(view.contains(win._current_frequency()))

    def test_hover_and_click_tune_on_a_zoomed_waterfall_in_the_window(self):
        win = self.win
        win.set_frequency(100.1e6)
        win._waterfall.set_history_seconds(0)
        for _ in range(20):
            win._process_block(win._device.read_samples(80000))
        win._zoom_in_action.trigger()
        win._zoom_in_action.trigger()
        wf = win._waterfall
        plot = wf.plot_rect()
        x = wf._freq_to_x(100.15e6, plot)
        QTest.mouseClick(
            wf._canvas,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.AltModifier,
            QPointF(x, plot.top() + 10).toPoint(),
        )
        self.assertAlmostEqual(win._current_frequency(), 100.15e6, delta=1e3)


if __name__ == "__main__":
    unittest.main()

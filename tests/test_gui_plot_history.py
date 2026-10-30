#!/usr/bin/env python3
"""
Honest plot history: the waterfall's rows keep the frequency they were
received at (a retune or a span change moves them, it never relabels them),
the frozen spectrum after Stop is not redrawn under a new frequency, the
plots say when they are stopped, and the hover readout never hides the
measurement readout.

Regression tests for rx:RX-02, ux:UX-B03, mc:MC-04, ux:UX-B14 and rx:RX-09.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_gui_plot_history.py``
"""

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
    from PyQt6.QtCore import QPointF, QRectF, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication, QMenu

    from sdr_module.gui import spectrum_widget as swm
    from sdr_module.gui import waterfall_widget as wfm

N = 2048
RATE = 2.4e6
BIN = RATE / N


def _settle(n=3):
    for _ in range(n):
        QApplication.processEvents()


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def monotonic(self):
        return self.t


def _line(rng, carriers=(), center=100.1e6, rate=RATE, floor=-85.0):
    """A noise line with carriers at absolute frequencies (Hz)."""
    line = rng.normal(floor, 1.5, N)
    for freq, level in carriers:
        b = int(round(N / 2 + (freq - center) / (rate / N)))
        line[b - 2 : b + 3] = level
    return line


class _ReadoutRecorder:
    """Records what the plots' hover readouts say (and where they are)."""

    def __init__(self, module):
        self.module = module
        self.calls = []
        self._original = module.draw_readout

    def __enter__(self):
        def record(painter, lines, anchor_x, plot, fm, palette, *args, **kwargs):
            box = self._original(
                painter, lines, anchor_x, plot, fm, palette, *args, **kwargs
            )
            self.calls.append((list(lines), box))
            return box

        self.module.draw_readout = record
        return self

    def __exit__(self, *exc):
        self.module.draw_readout = self._original
        return False

    @property
    def last(self):
        return self.calls[-1][0] if self.calls else None


def _level(lines):
    """The dBFS value in a readout, or None when it shows no level."""
    for line in lines or ():
        if "dBFS" in line:
            text = line.split("dBFS")[0].strip()
            try:
                return float(text)
            except ValueError:
                return None
    return None


class _WaterfallHistoryCase(unittest.TestCase):
    history_size = 100

    def setUp(self):
        require_pyqt6(self)
        self.clock = _Clock()
        patcher = mock.patch.object(
            wfm, "time", types.SimpleNamespace(monotonic=self.clock.monotonic)
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.w = wfm.WaterfallWidget(history_size=self.history_size)
        self.w.set_frequency_range(100.1e6, RATE)
        self.w.resize(900, 420)
        self.w.show()
        _settle()
        self.rng = np.random.default_rng(11)

    def tearDown(self):
        self.w.close()
        self.w.deleteLater()

    def feed(self, n, carriers=(), center=None, rate=RATE):
        center = self.w._center_freq if center is None else center
        for _ in range(n):
            self.clock.t += 1 / 30
            self.w.add_line(_line(self.rng, carriers, center, rate))

    def row_y(self, row):
        """Canvas y of the middle of image row ``row`` (0 = newest)."""
        plot = self.w.plot_rect()
        rows = (
            self.w._rows_shown()
            if hasattr(self.w, "_rows_shown")
            else self.w._history_size
        )
        return plot.top() + (row + 0.5) * plot.height() / rows

    def drawn_peak_x(self, row):
        """Canvas x where row ``row``'s strongest signal is drawn."""
        plot = self.w.plot_rect()
        cols = self.w._display_columns(plot)
        self.w._display_image(cols)
        codes = self.w._disp_codes[row]
        col = int(np.argmax(codes))
        target = (
            self.w._display_target(plot) if hasattr(self.w, "_display_target") else plot
        )
        return target.left() + (col + 0.5) * target.width() / codes.shape[0]

    def hover(self, x, y):
        with _ReadoutRecorder(wfm) as rec:
            self.w._set_hover(QPointF(x, y))
            self.w._canvas.grab()
        self.w._set_hover(None)
        return rec.last


class TestWaterfallRowsKeepTheirFrequency(_WaterfallHistoryCase):
    """rx:RX-02 / ux:UX-B03: a retune or a span change moves the old rows."""

    def test_old_rows_are_drawn_at_their_own_frequency_after_a_retune(self):
        self.feed(40, carriers=[(100.3e6, -30.0)])
        self.w.set_center_freq(100.6e6)  # the LO moves up 500 kHz
        self.feed(10)  # noise only at the new frequency
        old_row = 20
        x = self.drawn_peak_x(old_row)
        freq = self.w.frequency_at(x)
        self.assertAlmostEqual(freq, 100.3e6, delta=3 * BIN)

    def test_hover_click_bookmark_and_measure_an_old_row(self):
        self.feed(40, carriers=[(100.3e6, -30.0)])
        self.w.set_center_freq(100.6e6)
        self.feed(10)
        old_row = 20
        y = self.row_y(old_row)
        x_true = self.w._freq_to_x(100.3e6, self.w.plot_rect())
        x_wrong = self.w._freq_to_x(100.8e6, self.w.plot_rect())
        # Hover: the station is read where it really was...
        lines = self.hover(x_true, y)
        self.assertIsNotNone(_level(lines), lines)
        self.assertGreater(_level(lines), -40.0, lines)
        # ...and nothing is invented 500 kHz higher.
        lines = self.hover(x_wrong, y)
        level = _level(lines)
        self.assertTrue(level is None or level < -60.0, lines)
        # A click on the station tunes it.
        tuned = []
        self.w.frequency_clicked.connect(tuned.append)
        self.w.set_click_snap(100e3, "fm")
        x = self.drawn_peak_x(old_row)
        QTest.mouseClick(
            self.w._canvas, Qt.MouseButton.LeftButton, pos=QPointF(x, y).toPoint()
        )
        self.assertEqual(len(tuned), 1)
        self.assertAlmostEqual(tuned[0], 100.3e6, delta=2e3)
        # The context menu's Bookmark names the station's frequency.
        marked = []
        self.w.bookmark_requested.connect(marked.append)
        captured = {}

        def fake_exec(menu, *_args):
            for act in menu.actions():
                captured[act.text().replace("&", "")] = act

        with mock.patch.object(QMenu, "exec", fake_exec):
            self.w._show_context_menu(QPointF(x, y), QPointF(0, 0).toPoint())
        next(a for t, a in captured.items() if t.startswith("Bookmark")).trigger()
        self.assertAlmostEqual(marked[0], 100.3e6, delta=3 * BIN)
        # A measurement over the old rows finds the station where it was.
        plot = self.w.plot_rect()
        a = QPointF(x - 40, self.row_y(old_row - 5))
        b = QPointF(x + 40, self.row_y(old_row + 5))
        self.w._drag_measure(a, b)
        self.w._finish_measure()
        stats = self.w.measurement()
        self.assertGreater(stats["peak_db"], -35.0)
        self.assertAlmostEqual(stats["peak_freq"], 100.3e6, delta=3 * BIN)
        self.assertTrue(plot.contains(a))

    def test_a_span_change_remaps_old_rows(self):
        self.feed(40, carriers=[(100.6e6, -30.0)])
        self.w.set_sample_rate(1.2e6)  # half the span, same center
        self.feed(5, rate=1.2e6)
        x = self.drawn_peak_x(20)
        self.assertAlmostEqual(self.w.frequency_at(x), 100.6e6, delta=6 * BIN)

    def test_hover_says_when_an_old_row_was_not_received_there(self):
        self.feed(40, carriers=[(100.3e6, -30.0)])
        self.w.set_center_freq(101.3e6)  # 1 MHz up: half the old span left
        self.feed(5)
        plot = self.w.plot_rect()
        x = self.w._freq_to_x(102.2e6, plot)  # above anything received before
        lines = self.hover(x, self.row_y(20))
        self.assertIn("Not received then", lines)
        lines = self.hover(x, self.row_y(1))  # a new row: a level
        self.assertIsNotNone(_level(lines), lines)

    def test_a_retune_beyond_the_span_leaves_no_ghost(self):
        self.feed(40, carriers=[(100.3e6, -30.0)])
        self.w.set_center_freq(433.92e6)
        self.feed(5)
        for row in (10, 20, 30):
            y = self.row_y(row)
            for frac in (0.2, 0.5, 0.583, 0.8):
                plot = self.w.plot_rect()
                lines = self.hover(plot.left() + frac * plot.width(), y)
                level = _level(lines)
                self.assertTrue(level is None or level < -60.0, (row, frac, lines))


class TestRemapReusesCodes(_WaterfallHistoryCase):
    """Rows still on the span the codes were on are moved, not quantized
    again: the result is exactly what a full rebuild gives."""

    def rebuilt(self):
        reused = self.w._codes.copy()
        self.w._ring = None  # no old codes to reuse: quantize every row
        fresh = self.w._codes.copy()
        return reused, fresh

    def test_a_retune_shift_matches_a_full_rebuild(self):
        self.feed(30, carriers=[(100.3e6, -30.0)])
        self.w.set_center_freq(100.6e6)
        self.feed(20, carriers=[(100.3e6, -30.0)])
        self.w.set_center_freq(100.2e6)  # rows from two spans now
        reused, fresh = self.rebuilt()
        np.testing.assert_array_equal(reused, fresh)

    def test_a_span_and_size_change_match_a_full_rebuild(self):
        self.feed(30, carriers=[(100.3e6, -30.0)])
        self.w.set_sample_rate(1.2e6)
        reused, fresh = self.rebuilt()
        np.testing.assert_array_equal(reused, fresh)
        self.feed(10, rate=1.2e6)
        self.w.set_fft_size(4096)
        reused, fresh = self.rebuilt()
        np.testing.assert_array_equal(reused, fresh)

    def test_quantize_works_on_a_copy(self):
        line = np.array([-50.0, np.nan, np.inf, -np.inf], dtype=np.float32)
        before = line.copy()
        codes = wfm.quantize_db(line)
        np.testing.assert_array_equal(line, before)  # the input is untouched
        self.assertEqual(codes.dtype, np.uint16)
        self.assertTrue((codes > 0).all())


class TestMeasurementReadoutStaysVisible(_WaterfallHistoryCase):
    """rx:RX-09: after a drag-to-measure the hover readout never covers the
    measurement's readout."""

    def _rects(self):
        rects = {}
        original_box = self.w._draw_box_readout

        def record_box(painter, lines, box, plot, p):
            frame = original_box(painter, lines, box, plot, p)
            if frame is None:  # (before draw_box_readout returned its frame)
                fm = self.w._fm
                width = max(fm.horizontalAdvance(s) for s in lines) + 12
                height = fm.height() * len(lines) + 6
                x = box.right() + 8
                if x + width > plot.right() - 2:
                    x = box.left() - 8 - width
                x = min(max(plot.left() + 2, x), plot.right() - 2 - width)
                y = min(max(plot.top() + 4, box.top()), plot.bottom() - 2 - height)
                frame = QRectF(x, y, width, height)
            rects["measure"] = frame
            return frame

        original_readout = wfm.draw_readout

        def record_readout(painter, lines, anchor_x, plot, fm, palette, *a, **k):
            box = original_readout(painter, lines, anchor_x, plot, fm, palette, *a, **k)
            if box is None:
                width = max(fm.horizontalAdvance(s) for s in lines) + 12
                height = fm.height() * len(lines) + 6
                x = anchor_x + 10
                if x + width > plot.right() - 2:
                    x = anchor_x - 10 - width
                x = max(plot.left() + 2, x)
                box = QRectF(x, plot.top() + 6, width, height)
            rects["hover"] = box
            return box

        return rects, record_box, record_readout

    def test_hover_readout_moves_off_the_measurement(self):
        self.feed(90, carriers=[(100.1e6, -30.0)])
        plot = self.w.plot_rect()
        cx = plot.center().x()
        canvas = self.w._canvas
        cases = (
            (QPointF(cx - 60, plot.top() + 8), QPointF(cx + 60, plot.top() + 120)),
            (QPointF(cx + 60, plot.top() + 8), QPointF(cx - 60, plot.top() + 120)),
            (
                QPointF(plot.right() - 90, plot.top() + 4),
                QPointF(plot.right() - 5, plot.top() + 60),
            ),
        )
        for a, b in cases:
            rects, record_box, record_readout = self._rects()
            QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=a.toPoint())
            for i in range(1, 6):
                QTest.mouseMove(canvas, (a + (b - a) * (i / 5)).toPoint())
            QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=b.toPoint())
            self.w._set_hover(b)  # the pointer rests where the drag ended
            with (
                mock.patch.object(self.w, "_draw_box_readout", record_box),
                mock.patch.object(wfm, "draw_readout", record_readout),
            ):
                canvas.grab()
            self.assertIn("measure", rects, (a, b))
            if "hover" in rects:
                overlap = rects["hover"].intersected(rects["measure"])
                self.assertTrue(overlap.isEmpty(), (a, b, rects))
            self.w.clear_measurement()


class TestStoppedPlots(_WindowTestCase):
    """mc:MC-04 / ux:UX-B14: after Stop (or Disconnect) the plots say so and
    retuning never relabels the frozen data."""

    demo = True

    def run_frames(self, n=6):
        win = self.win
        if not win._is_running:
            win._start_acquisition()
        for _ in range(n):
            win._process_block(win._device.read_samples(80000))
        self.settle()

    def strongest_x(self):
        """Canvas x of the strongest bin of the spectrum trace."""
        sp = self.win._spectrum
        data = sp._spectrum
        plot = sp.plot_rect()
        b = int(np.argmax(data))
        return plot.left() + (b + 0.5) / len(data) * plot.width(), float(data[b])

    def spectrum_hover(self, x):
        sp = self.win._spectrum
        with _ReadoutRecorder(swm) as rec:
            sp._set_hover(QPointF(x, sp.plot_rect().center().y()))
            sp._canvas.grab()
        sp._set_hover(None)
        return rec.last

    def test_retuning_while_stopped_does_not_move_the_frozen_trace(self):
        win = self.win
        win.set_frequency(100.1e6)
        self.run_frames()
        x, peak = self.strongest_x()
        self.assertGreater(peak, -40.0)
        win._stop_acquisition()
        win.set_frequency(433.92e6)
        self.settle()
        self.assertAlmostEqual(win._lo_frequency(), 433.92e6, delta=1.0)
        lines = self.spectrum_hover(x)
        level = _level(lines)
        # No broadcast station is invented at 433.9 MHz.
        self.assertTrue(level is None or level < peak - 20.0, lines)

    def test_waterfall_does_not_relabel_rows_after_stop_and_retune(self):
        win = self.win
        win.set_frequency(100.1e6)
        win._waterfall.set_history_seconds(0)
        self.run_frames(20)
        wf = win._waterfall
        row = 5
        plot = wf.plot_rect()
        codes = wf._codes[row].copy()
        win._stop_acquisition()
        win.set_frequency(433.92e6)
        self.settle()
        # Every old row lies outside 432.7-435.1 MHz now: nothing of it is
        # drawn there.
        wf._display_image(wf._display_columns(plot))
        self.assertGreater(int(codes.max()), 0)
        self.assertEqual(int(wf._codes[row].max()), 0)

    def _legend_texts(self, plot_widget):
        texts = []

        class Painter:
            def setPen(self, *_a):
                pass

            def drawLine(self, *_a):
                pass

            def drawText(self, *args):
                texts.append(args[-1])

        from sdr_module.gui.themes import get_palette

        plot_widget._draw_legend(Painter(), plot_widget.plot_rect(), get_palette())
        return texts

    def test_stop_dims_the_plots_and_drops_live(self):
        win = self.win
        self.run_frames()
        self.assertIn("Live", self._legend_texts(win._spectrum))
        win._stop_acquisition()
        self.settle()
        self.assertNotIn("Live", self._legend_texts(win._spectrum))
        self.assertIn("Stopped", win._spectrum.stale_text() or "")
        self.assertIn("Space", win._spectrum.stale_text() or "")
        self.assertIn("Stopped", win._waterfall.stale_text() or "")
        # Receiving again clears it with the first frame.
        self.run_frames(1)
        self.assertIsNone(win._spectrum.stale_text())
        self.assertIsNone(win._waterfall.stale_text())
        self.assertIn("Live", self._legend_texts(win._spectrum))

    def test_the_stopped_text_follows_the_state(self):
        win = self.win
        self.run_frames(2)
        playback = mock.Mock()
        playback.finished = False
        with (
            mock.patch.object(type(win._playback), "active", True),
            mock.patch.object(type(win._playback), "device", playback),
        ):
            win._sync_plot_state(False, True)
            self.assertEqual(win._spectrum.stale_text(), "Paused - press Space to play")
            playback.finished = True
            win._sync_plot_state(False, True)
            self.assertIn("End of the recording", win._spectrum.stale_text())
        with mock.patch.object(win, "_tx_busy", return_value=True):
            win._sync_plot_state(False, True)
            self.assertIn("Transmitting", win._waterfall.stale_text())
        win._sync_plot_state(True, True)
        self.assertIsNone(win._spectrum.stale_text())

    def test_disconnect_says_so_and_clears_peak_hold(self):
        win = self.win
        self.run_frames()
        win._disconnect_device()
        self.settle()
        self.assertTrue(np.all(win._spectrum._peak_hold <= -120.0))
        self.assertIn("No device", win._spectrum.stale_text() or "")


if __name__ == "__main__":
    unittest.main()

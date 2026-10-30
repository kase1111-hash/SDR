#!/usr/bin/env python3
"""
Regression tests for the plots code review (PLOT-1 ... PLOT-10): the
spectrum, the waterfall, their shared zoom and the main window's plot feed.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest tests/test_review_plots.py``
"""

import os
import types
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, require_pyqt6  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import QPointF
    from PyQt6.QtWidgets import QApplication

    from sdr_module.dsp.frame_spectrum import FrameSpectrumAnalyzer
    from sdr_module.gui import spectrum_widget as swm
    from sdr_module.gui import waterfall_widget as wfm
    from sdr_module.gui.plot_view import PlotView

CENTER = 100e6


def _settle(n=3):
    for _ in range(n):
        QApplication.processEvents()


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def monotonic(self):
        return self.t


def _on_bin_frame(rate, size, k_off=37, frames=1):
    """A frame with a tone exactly on bin ``size // 2 + k_off`` (fftshift
    order: ``CENTER + k_off * rate / size``), through the real analyzer."""
    bin_hz = rate / size
    n = max(size, int(rate / 30))
    t = np.arange(n) / rate
    rng = np.random.default_rng(1)
    x = 0.1 * np.exp(2j * np.pi * k_off * bin_hz * t)
    x = x + (rng.normal(size=n) + 1j * rng.normal(size=n)) * 1e-4
    frame = FrameSpectrumAnalyzer(size).process(x.astype(np.complex64))
    assert int(np.argmax(frame.mean_db)) == size // 2 + k_off
    return frame, CENTER + k_off * bin_hz


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
        self.wf = wfm.WaterfallWidget(history_size=50)
        for w in (self.sp, self.wf):
            w.set_view_model(self.view)
            w.resize(1000, 360)
            w.show()
            self.addCleanup(w.deleteLater)
        _settle()

    def axis(self, rate, size):
        for w in (self.sp, self.wf):
            w.set_frequency_range(CENTER, rate)
            w.set_fft_size(size)
            # The tuned marker out of the way of the carrier.
            w.set_tuned_frequency(CENTER - 0.45 * rate)

    def feed(self, frame, n=3):
        for _ in range(n):
            self.clock.t += 1 / 30
            self.sp.update_spectrum(frame.mean_db)
            self.wf.add_line(frame.max_db)


class TestBinsDrawnAtTheirFrequency(_PlotPair):
    """PLOT-1: FFT bin k (fftshift order) is at start + k * bin, and both
    plots draw, read and measure it there (not half a bin higher)."""

    def check(self, rate, size):
        frame, true_f = _on_bin_frame(rate, size)
        self.axis(rate, size)
        self.feed(frame)
        self.sp.zoom_to(true_f, 4)  # x16
        self.assertEqual(self.view.zoom, 16)
        _settle()
        bin_hz = rate / size
        plot = self.sp.plot_rect()
        px_per_bin = bin_hz / self.view.span * plot.width()
        self.assertGreater(px_per_bin, 7.0)  # a half-bin error would show
        x_true = self.sp._freq_x(true_f, plot)

        # The spectrum trace's apex is drawn at the carrier's frequency.
        xs, ys = self.sp._trace_xy(self.sp._main_trace(), plot, int(plot.width()))
        x_apex = float(xs[int(np.argmin(ys))])
        self.assertLess(abs(x_apex - x_true), 0.5)
        # Right-click Tune to / Bookmark / Copy at the drawn apex.
        menu_f = self.sp._snapped_frequency_at(x_apex)
        self.assertLess(abs(menu_f - true_f), 0.1 * bin_hz)
        # The hover's level reads the carrier's bin there.
        level = self.sp._level_at(x_true, plot)
        self.assertAlmostEqual(level, float(frame.mean_db.max()), places=3)

        # The waterfall's brightest column is centered on it too.
        wplot = self.wf.plot_rect()
        self.assertAlmostEqual(wplot.left(), plot.left())
        cols = self.wf._display_columns(wplot)
        self.wf._display_image(cols)
        codes = self.wf._disp_codes[0]
        target = self.wf._display_target(wplot)
        col = int(np.argmax(codes))
        x_col = target.left() + (col + 0.5) * target.width() / codes.shape[0]
        self.assertLess(abs(x_col - self.wf._freq_to_x(true_f, wplot)), 0.5)
        # Its level readout at the carrier, and a measurement over it.
        y = wplot.top() + 2
        self.assertAlmostEqual(
            self.wf.level_at(self.wf._freq_to_x(true_f, wplot), y),
            float(frame.max_db.max()),
            delta=0.125,
        )
        x0 = self.wf._freq_to_x(true_f - 3 * bin_hz, wplot)
        x1 = self.wf._freq_to_x(true_f + 3 * bin_hz, wplot)
        self.wf._drag_measure(QPointF(x0, wplot.top() + 1), QPointF(x1, y + 20))
        self.wf._finish_measure()
        stats = self.wf.measurement()
        self.assertLess(abs(stats["peak_freq"] - true_f), 0.1 * bin_hz)

    def test_rtl_rate_1024_points(self):
        self.check(2.4e6, 1024)

    def test_rtl_rate_2048_points(self):
        self.check(2.4e6, 2048)

    def test_hackrf_rate_1024_points(self):
        self.check(20e6, 1024)

    def test_a_measurement_counts_the_bins_it_covers(self):
        # Only the bins whose span overlaps the measured band: a band
        # around one bin's center reads that bin, not its upper neighbour.
        rate, size = 2.4e6, 1024
        bin_hz = rate / size
        line = np.full(size, -90.0)
        line[size // 2 + 10] = -20.0  # CENTER + 10 bins
        line[size // 2 + 11] = -10.0  # its upper neighbour is stronger
        self.axis(rate, size)
        for _ in range(3):
            self.clock.t += 1 / 30
            self.wf.add_line(line)
        self.sp.zoom_to(CENTER + 10 * bin_hz, 4)
        wplot = self.wf.plot_rect()
        f = CENTER + 10 * bin_hz
        x0 = self.wf._freq_to_x(f - 0.3 * bin_hz, wplot)
        x1 = self.wf._freq_to_x(f + 0.3 * bin_hz, wplot)
        self.wf._drag_measure(
            QPointF(x0, wplot.top() + 1), QPointF(x1, wplot.top() + 20)
        )
        self.wf._finish_measure()
        stats = self.wf.measurement()
        self.assertAlmostEqual(stats["peak_db"], -20.0, delta=0.125)
        self.assertAlmostEqual(stats["peak_freq"], f, delta=1.0)

    def newest_row(self):
        ring = self.wf._current_ring()
        return ring[self.wf._ring_rows(0)]

    def test_rows_re_mapped_onto_wider_bins_stay_where_they_were(self):
        # 2.4 -> 20 MS/s: the carrier (22 old bins up) is 0.36 of a new bin
        # below new bin +3's center, so it belongs in that bin (it fell in
        # the bin below when each bin was taken to start at its center).
        size = 1024
        frame, true_f = _on_bin_frame(2.4e6, size, k_off=22)
        self.axis(2.4e6, size)
        self.feed(frame, n=1)
        self.wf.set_frequency_range(CENTER, 20e6)
        new_bin = 20e6 / size
        row = self.newest_row()
        self.assertEqual(int(np.argmax(row)), size // 2 + 3)
        found = CENTER + (int(np.argmax(row)) - size // 2) * new_bin
        self.assertLess(abs(found - true_f), 0.5 * new_bin)

    def test_rows_re_mapped_onto_narrower_bins_stay_where_they_were(self):
        # 1024 -> 32768 points: one old bin lights a run of new bins,
        # centered on the carrier (not half an old bin above it).
        rate = 2.4e6
        frame, true_f = _on_bin_frame(rate, 1024)
        self.axis(rate, 1024)
        self.feed(frame, n=1)
        self.wf.set_fft_size(32768)
        row = self.newest_row()
        lit = np.flatnonzero(row == row.max())
        middle = CENTER + (lit.mean() - 32768 // 2) * rate / 32768
        self.assertLess(abs(middle - true_f), rate / 32768)


if __name__ == "__main__":
    unittest.main()

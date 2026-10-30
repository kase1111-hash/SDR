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

from tests.test_gui_chrome import (  # noqa: E402
    HAS_PYQT6,
    _WindowTestCase,
    require_pyqt6,
)

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


class _Hardware:
    """A hardware-like SDRDevice: transfers of 256k samples (an RTL-SDR's)
    queued by a driver, each holding one carrier as heard at the LO the
    transfer was taken at."""

    RATE = 2.4e6
    CARRIER = 100.3e6
    TRANSFER = 256 * 1024

    @classmethod
    def make(cls):
        import queue

        from sdr_module.devices.base import SDRDevice

        rate, carrier, size = cls.RATE, cls.CARRIER, cls.TRANSFER

        class Device(SDRDevice):
            def __init__(self):
                super().__init__()
                self._state.sample_rate = rate
                self._state.frequency = 100.1e6
                self.rng = np.random.default_rng(3)
                self.clock = 0

            def transfer(self, lo=None):
                """Queue one transfer taken at ``lo`` (the tuner's LO)."""
                lo = self._state.frequency if lo is None else lo
                k = self.clock + np.arange(size)
                self.clock += size
                noise = self.rng.standard_normal(size) + 1j * self.rng.standard_normal(
                    size
                )
                x = 1e-3 * noise
                if abs(carrier - lo) < rate / 2:
                    x = x + 0.3 * np.exp(2j * np.pi * (carrier - lo) * k / rate)
                try:
                    self._sample_queue.put_nowait(x.astype(np.complex64))
                except queue.Full:
                    pass

            def set_frequency(self, freq_hz):
                self._state.frequency = float(freq_hz)
                return True

            def open(self, index=0):
                return True

            def close(self):
                pass

            def set_sample_rate(self, rate_hz):
                return True

            def set_bandwidth(self, bw_hz):
                return True

            def set_gain(self, gain_db):
                return True

            def set_gain_mode(self, auto):
                return True

            def start_rx(self, callback=None):
                with self._state_lock:
                    self._state.is_streaming = True
                return True

            def stop_rx(self):
                with self._state_lock:
                    self._state.is_streaming = False
                return True

        return Device()


class TestRetuneDropsStaleSamples(_WindowTestCase):
    """PLOT-2: after the LO is retuned on hardware, the transfers it queued
    and the one in flight hold the old frequency: they are not drawn at
    the new one (no ghost row, no false peak in the just-reset hold)."""

    demo = True

    def setUp(self):
        super().setUp()
        from sdr_module.gui import main_window

        win = self.win
        win._stop_acquisition()
        self.dev = _Hardware.make()
        win._device = self.dev
        win._demo_mode = False
        win._tune_to(100.1e6, recenter=True)
        win._waterfall.set_history_seconds(0)  # Live: one row per frame
        win._start_acquisition()
        for timer in (win._display_timer, win._status_timer, win._hotplug_timer):
            timer.stop()
        # The tuner's settling time is not what is tested here.
        patcher = mock.patch.object(main_window, "_RETUNE_SETTLE_S", 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._release)

    def _release(self):
        self.win._stop_acquisition()
        self.win._device = None

    def frame(self):
        self.win._pacer.reset()
        self.win._update_display()

    def ghost(self, line, axis):
        """Level and row median at where the old frequency's carrier would
        be drawn on ``axis`` (the new LO)."""
        center, rate = axis
        n = len(line)
        ghost_f = 101.5e6 + (_Hardware.CARRIER - 100.1e6)
        k = int(round((ghost_f - center) / (rate / n) + n / 2))
        return float(np.max(line[k - 2 : k + 3])), float(np.median(line))

    def test_no_ghost_row_or_peak_after_a_retune(self):
        win, dev = self.win, self.dev
        sp, wf = win._spectrum, win._waterfall
        sp.set_peak_hold(True)
        for _ in range(4):
            dev.transfer()
            self.frame()
        rows = len(wf._history)
        self.assertGreater(rows, 0)
        # Two transfers wait in the queue, one is in flight: all three
        # were taken at 100.1 MHz.
        dev.transfer()
        dev.transfer()
        self.assertTrue(win._tune_to(101.5e6, recenter=True))
        self.assertEqual(dev.state.frequency, 101.5e6)
        dev.transfer(lo=100.1e6)  # the one in flight arrives late
        for _ in range(3):
            self.frame()
            dev.transfer()
        self.frame()
        new_rows = [
            line
            for line, axis in zip(wf._history, wf._row_axes, strict=True)
            if axis[0] == 101.5e6
        ]
        self.assertGreater(len(new_rows), 0)  # the new frequency is drawn
        for line in new_rows:
            level, median = self.ghost(line, (101.5e6, _Hardware.RATE))
            self.assertLess(level - median, 15.0)
        hold, median = self.ghost(sp._peak_hold, (sp._center_freq, sp._sample_rate))
        self.assertLess(hold - float(np.median(sp._spectrum)), 15.0)

    def test_dropped_transfers_count_as_received(self):
        win, dev = self.win, self.dev
        dev.transfer()
        self.frame()
        before = sum(count for _t, count in win._rx_history)
        for _ in range(3):
            dev.transfer()
        self.assertTrue(win._tune_to(101.5e6, recenter=True))
        self.assertIsNone(dev.read_samples(1, timeout=0.0))  # dropped
        after = sum(count for _t, count in win._rx_history)
        self.assertEqual(after - before, 3 * _Hardware.TRANSFER)

    def test_the_plots_wait_for_the_tuner_to_settle(self):
        import time

        from sdr_module.gui import main_window

        win, dev = self.win, self.dev
        dev.transfer()
        self.frame()
        rows = len(win._waterfall._history)
        with (
            mock.patch.object(main_window, "_RETUNE_STALE_SAMPLES", 0),
            mock.patch.object(main_window, "_RETUNE_SETTLE_S", 0.15),
        ):
            self.assertTrue(win._tune_to(101.5e6, recenter=True))
        dev.transfer()
        self.frame()  # within the settling time: not drawn
        self.assertEqual(len(win._waterfall._history), rows)
        time.sleep(0.16)
        dev.transfer()
        self.frame()
        self.assertEqual(len(win._waterfall._history), rows + 1)
        self.assertEqual(win._waterfall._row_axes[-1][0], 101.5e6)

    def test_tuning_within_the_span_keeps_the_samples(self):
        # Only the VFO moves: the samples are still where they were taken.
        win, dev = self.win, self.dev
        dev.transfer()
        self.frame()
        dev.transfer()
        self.assertTrue(win._tune_to(100.2e6))
        self.assertEqual(win._lo_frequency(), 100.1e6)
        rows = len(win._waterfall._history)
        self.frame()
        self.assertEqual(len(win._waterfall._history), rows + 1)


class TestSecondReceiverPlots(unittest.TestCase):
    """PLOT-3: the Second Receiver window's spectrum menus work there as in
    the main window: both plots zoom together, Bookmark adds a bookmark,
    FFT Size and Window change what is computed (for both plots), and the
    plots' messages show in its status line."""

    def setUp(self):
        require_pyqt6(self)
        from tests.test_gui_second_receiver import _SecondCase

        class Case(_SecondCase):
            def runTest(self):  # pragma: no cover - not run
                pass

        self.case = Case()
        self.case.setUp()
        self.addCleanup(self.case.tearDown)
        self.win, self.rx = self.case.win, self.case.rx
        self.assertTrue(self.rx.start())
        self.rx._frame_timer.stop()
        self.rx.tune(145.8e6)
        self.feed()

    def feed(self, count=6):
        for _ in range(count):
            self.rx._process(self.rx.device.read_samples(80_000))

    def menu_pick(self, plot, freq, prefix):
        from PyQt6.QtCore import QPoint
        from PyQt6.QtWidgets import QMenu

        def fake_exec(menu, *_args, **_kwargs):
            for action in menu.actions():
                if action.text().replace("&", "").startswith(prefix):
                    action.trigger()
                    return action
            self.fail(f"no {prefix!r} in the menu")

        rect = plot.plot_rect()
        x = (
            plot._freq_x(freq, rect)
            if hasattr(plot, "_freq_x")
            else plot._freq_to_x(freq, rect)
        )
        with mock.patch.object(QMenu, "exec", fake_exec):
            plot._show_context_menu(QPointF(x, rect.center().y()), QPoint(0, 0))
        _settle()

    def test_both_plots_share_one_view(self):
        sp, wf = self.rx._spectrum, self.rx._waterfall
        self.assertIs(sp.view_model(), wf.view_model())
        self.menu_pick(sp, 145.8e6, "Zoom In Here")
        self.assertEqual(wf.view_model().zoom, 2)
        self.assertEqual(
            (sp.view_model().start, sp.view_model().end),
            (wf.view_model().start, wf.view_model().end),
        )

    def test_the_spectrums_bookmark_item_adds_a_bookmark(self):
        panel = self.win._bookmarks_panel
        before = panel.bookmark_count()
        self.menu_pick(self.rx._spectrum, 145.8e6, "Bookmark")
        self.assertEqual(panel.bookmark_count(), before + 1)
        self.assertIn("Bookmarked", self.rx.message())

    def test_fft_size_and_window_change_what_both_plots_show(self):
        rx = self.rx
        sp, wf = rx._spectrum, rx._waterfall
        sp._choose_resolution(16384, "blackman-harris")
        self.feed(4)
        self.assertEqual(wf.fft_size(), 16384)
        self.assertEqual(len(wf._history[-1]), 16384)
        self.assertEqual(len(sp._spectrum), 16384)
        # Real 16384-point bins (not 2048 interpolated): the demo's
        # strongest carrier is a few bins wide, not ~3 kHz.
        s = np.asarray(sp._spectrum)
        k = int(np.argmax(s))
        lo = hi = k
        while lo > 0 and s[lo - 1] >= s[k] - 6:
            lo -= 1
        while hi < len(s) - 1 and s[hi + 1] >= s[k] - 6:
            hi += 1
        width = (hi - lo + 1) * sp._sample_rate / len(s)
        self.assertLess(width, 1500.0)
        self.assertIn("16384-point", rx.message())

    def test_the_plots_messages_show_in_the_status_line(self):
        self.rx._spectrum.message.emit("Copied 145.800 MHz", "info")
        self.assertEqual(self.rx.message(), "Copied 145.800 MHz")
        self.rx._waterfall.message.emit("Copied the measurement", "info")
        self.assertEqual(self.rx.message(), "Copied the measurement")


class TestSecondReceiverRetuneDropsStaleSamples(unittest.TestCase):
    """PLOT-2 in the Second Receiver window (its own hardware and plot
    feed): no ghost of the old frequency after its LO is retuned."""

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui import second_receiver
        from tests.test_gui_second_receiver import _SecondCase

        class Case(_SecondCase):
            def runTest(self):  # pragma: no cover - not run
                pass

        self.case = Case()
        self.case.setUp()
        self.addCleanup(self.case.tearDown)
        rx = self.rx = self.case.rx
        self.dev = _Hardware.make()
        rx._device = self.dev
        rx._running = True
        self.addCleanup(setattr, rx, "_device", None)
        self.addCleanup(setattr, rx, "_running", False)
        self.dev.start_rx()
        rx._waterfall.set_history_seconds(0)
        patcher = mock.patch.object(second_receiver, "_RETUNE_SETTLE_S", 0.0)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.assertTrue(rx.tune(100.1e6, recenter=True))

    def frame(self):
        self.rx._pacer.reset()
        self.rx._frame()

    def test_no_ghost_row_after_a_retune(self):
        rx, dev = self.rx, self.dev
        for _ in range(3):
            dev.transfer()
            self.frame()
        dev.transfer()  # queued at 100.1 MHz
        self.assertTrue(rx.tune(101.5e6, recenter=True))
        self.assertEqual(dev.state.frequency, 101.5e6)
        dev.transfer(lo=100.1e6)  # in flight at the retune
        for _ in range(3):
            self.frame()
            dev.transfer()
        self.frame()
        wf = rx._waterfall
        ghost_f = 101.5e6 + (_Hardware.CARRIER - 100.1e6)
        rows = 0
        for line, (center, rate) in zip(wf._history, wf._row_axes, strict=True):
            if center != 101.5e6:
                continue
            rows += 1
            n = len(line)
            k = int(round((ghost_f - center) / (rate / n) + n / 2))
            self.assertLess(
                float(np.max(line[k - 2 : k + 3])) - float(np.median(line)), 15.0
            )
        self.assertGreater(rows, 0)


class TestWaterfallExport(_PlotPair):
    """PLOT-4: Save Image keeps the rows, not the status overlays of the
    live view (the dimmed Stopped veil and its badge, PAUSED, the focus
    frame); the view itself keeps them."""

    def image(self, path):
        from PyQt6.QtGui import QImage

        image = QImage(path)
        self.assertFalse(image.isNull())
        return image

    def test_a_stopped_or_paused_waterfall_saves_like_a_live_one(self):
        import tempfile

        rng = np.random.default_rng(4)
        self.wf.set_frequency_range(CENTER, 2.4e6)
        for _ in range(40):
            self.clock.t += 1 / 30
            line = rng.normal(-90.0, 2.0, 2048)
            line[1200:1210] = -30.0
            self.wf.add_line(line)
        _settle()
        folder = tempfile.mkdtemp(prefix="wf-export-")
        self.addCleanup(lambda: __import__("shutil").rmtree(folder, True))
        live = os.path.join(folder, "live.png")
        self.assertTrue(self.wf.export_image(live))
        self.wf.set_stale("Stopped - press Space")
        self.wf.set_paused(True)
        self.wf.setFocus()
        _settle()
        frozen = os.path.join(folder, "frozen.png")
        self.assertTrue(self.wf.export_image(frozen))
        self.assertEqual(self.image(frozen), self.image(live))
        # The view still says why it is frozen.
        self.assertEqual(self.wf.stale_text(), "Stopped - press Space")
        self.assertTrue(self.wf.is_paused())
        shown = self.wf._canvas.grab().toImage()
        self.assertNotEqual(shown, self.image(live).convertToFormat(shown.format()))


class TestPlaybackSpeedTimesRows(_WindowTestCase):
    """PLOT-5: while a recording plays at 0.5x-4x, the waterfall's time
    axis, hover age and measured duration are in the recording's time (a
    1 s burst measures 1 s at any speed), not wall-clock time."""

    demo = True
    RATE = 240e3

    def setUp(self):
        super().setUp()
        import shutil
        import tempfile

        from sdr_module.dsp.recording import FileFormat, SampleFormat, save_iq_file

        folder = tempfile.mkdtemp(prefix="plot5-")
        self.addCleanup(shutil.rmtree, folder, True)
        self.win._recording_folder = folder
        rate = self.RATE
        n = int(6 * rate)
        rng = np.random.default_rng(5)
        x = 1e-3 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        t = np.arange(n) / rate
        burst = (t >= 2.0) & (t < 3.0)  # exactly 1 s, 30 kHz up
        x[burst] += 0.3 * np.exp(2j * np.pi * 30e3 * t[burst])
        path = os.path.join(folder, "burst.sigmf-data")
        save_iq_file(
            path,
            x.astype(np.complex64),
            rate,
            145e6,
            SampleFormat.FLOAT32,
            FileFormat.SIGMF,
        )
        with mock.patch(
            "sdr_module.gui.main_window.QFileDialog.getOpenFileName",
            return_value=(path, ""),
        ):
            self.win._open_recording_action.trigger()
        self.settle()
        self.assertTrue(self.win._playback.active)
        self.win._waterfall.set_history_seconds(0)  # a row per frame
        # The waterfall's own clock (wall time) advances a frame a frame.
        self.wall = _Clock()
        patcher = mock.patch.object(
            wfm, "time", types.SimpleNamespace(monotonic=self.wall.monotonic)
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def play(self, speed, seconds=5.5):
        """Frames at 30 a second of wall time, each with ``speed`` / 30 s
        of the recording (as the display loop reads it at that speed)."""
        win = self.win
        win._playback_bar.set_speed(speed)
        dev = win._device
        per_frame = int(round(self.RATE * speed / 30))
        for _ in range(int(seconds * 30 / speed)):
            self.wall.t += 1 / 30
            samples = dev.read_samples(per_frame)
            win._plot_frame(samples)

    def burst_rows(self):
        wf = self.win._waterfall
        rows = []
        for i, (line, (center, rate)) in enumerate(
            zip(wf._history, wf._row_axes, strict=True)
        ):
            n = len(line)
            k = int(round(30e3 / (rate / n) + n / 2))
            if float(np.max(line[k - 3 : k + 4])) - float(np.median(line)) > 30:
                rows.append(i)
        return rows

    def measure_burst(self):
        wf = self.win._waterfall
        rows = self.burst_rows()
        self.assertGreater(len(rows), 3)
        count = len(wf._history)
        newest, oldest = count - 1 - rows[-1], count - 1 - rows[0]
        plot = wf.plot_rect()
        per_row = plot.height() / wf._rows_shown()
        x0 = wf._freq_to_x(145e6 + 20e3, plot)
        x1 = wf._freq_to_x(145e6 + 40e3, plot)
        y0 = plot.top() + (newest + 0.5) * per_row
        y1 = plot.top() + (oldest + 0.5) * per_row
        wf._drag_measure(QPointF(x0, y0), QPointF(x1, y1))
        wf._finish_measure()
        return wf.measurement()["duration"]

    def test_a_one_second_burst_measures_one_second_at_4x(self):
        self.play(4.0)
        self.assertAlmostEqual(self.measure_burst(), 1.0, delta=0.15)
        # The rows span the recording's 5.5 s, not 1.4 s of wall time.
        ages = self.win._waterfall._row_ages()
        self.assertAlmostEqual(float(ages[-1]), 5.5, delta=0.3)

    def test_and_at_half_speed(self):
        self.play(0.5, seconds=4.0)
        self.assertAlmostEqual(self.measure_burst(), 1.0, delta=0.15)


if __name__ == "__main__":
    unittest.main()

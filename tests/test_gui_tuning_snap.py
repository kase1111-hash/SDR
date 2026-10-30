#!/usr/bin/env python3
"""
Click-to-tune snaps onto the signal clicked (rx:UX-03, mc:MC-U18): the
spectrum and the waterfall search a few pixels around the click for a
signal, tune its center and move that onto the tuning step's raster when
it is close; Alt+click tunes exactly where clicked; the hover readout shows
where a click will land.

Run offscreen: ``QT_QPA_PLATFORM=offscreen python -m pytest
tests/test_gui_tuning_snap.py``
"""

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from tests.test_gui_chrome import HAS_PYQT6, require_pyqt6  # noqa: E402

if HAS_PYQT6:
    from PyQt6.QtCore import QPointF, Qt
    from PyQt6.QtTest import QTest
    from PyQt6.QtWidgets import QApplication

RATE = 2.4e6
CENTER = 100e6
N = 2048


def _spectrum(signals, seed=5):
    """Noise at -90 dB plus (offset, width, level) signals (width 0: a
    carrier's Hann main lobe), in fftshift order."""
    rng = np.random.default_rng(seed)
    power = -90.0 + 3.0 * rng.standard_normal(N)
    freqs = CENTER - RATE / 2 + np.arange(N) * RATE / N
    for offset, width, level in signals:
        f0 = CENTER + offset
        if width <= 0:
            x = (freqs - f0) / (RATE / N)
            lobe = np.where(
                np.abs(x) < 2, 20 * np.log10(np.abs(np.sinc(x / 2)) + 1e-9), -200
            )
            power = np.maximum(power, level + lobe)
        else:
            inside = np.abs(freqs - f0) <= width / 2
            power = np.where(inside, np.maximum(power, level), power)
    return power


class _PlotCase(unittest.TestCase):
    def setUp(self):
        require_pyqt6(self)
        self.clicked = []

    def show(self, widget):
        widget.resize(1100, 320)
        widget.show()
        for _ in range(3):
            QApplication.processEvents()
        widget.frequency_clicked.connect(self.clicked.append)
        self.addCleanup(widget.deleteLater)
        return widget

    @staticmethod
    def x_of(widget, freq):
        plot = widget.plot_rect()
        return plot.left() + (freq - (CENTER - RATE / 2)) / RATE * plot.width()


class TestSpectrumSnap(_PlotCase):
    def make(self, signals, step=100e3, mode="wfm"):
        from sdr_module.gui.spectrum_widget import SpectrumWidget

        w = self.show(SpectrumWidget())
        w.set_frequency_range(CENTER, RATE)
        w.update_spectrum(_spectrum(signals))
        w.set_click_snap(step, mode)
        return w

    def click(self, w, freq, alt=False):
        y = int(w.plot_rect().center().y())
        mods = (
            Qt.KeyboardModifier.AltModifier if alt else Qt.KeyboardModifier.NoModifier
        )
        QTest.mouseClick(
            w._canvas,
            Qt.MouseButton.LeftButton,
            mods,
            QPointF(self.x_of(w, freq), y).toPoint(),
        )
        return self.clicked[-1]

    def test_a_click_on_a_station_tunes_its_channel(self):
        w = self.make([(500e3, 180e3, -40.0)])
        self.assertEqual(self.click(w, 100.512e6), 100.5e6)
        self.assertEqual(self.click(w, 100.489e6), 100.5e6)
        # The hover readout shows where the click lands.
        self.assertEqual(w.landing_frequency(self.x_of(w, 100.512e6)), 100.5e6)

    def test_alt_click_tunes_exactly_where_clicked(self):
        w = self.make([(500e3, 180e3, -40.0)])
        landed = self.click(w, 100.512e6, alt=True)
        self.assertNotEqual(landed, 100.5e6)
        self.assertLess(abs(landed - 100.512e6), 3e3)  # one pixel

    def test_a_cw_carrier_lands_inside_the_cw_filter(self):
        carrier = CENTER - 280_123.0
        w = self.make([(carrier - CENTER, 0, -45.0)], step=10.0, mode="cw")
        landed = self.click(w, carrier + 2.5e3)  # a pixel or so off
        self.assertLess(abs(landed - carrier), 150.0)  # filter is +/-250 Hz

    def test_without_a_step_clicks_tune_to_the_pixel(self):
        from sdr_module.gui.spectrum_widget import SpectrumWidget

        w = self.show(SpectrumWidget())
        w.set_frequency_range(CENTER, RATE)
        w.update_spectrum(_spectrum([(500e3, 180e3, -40.0)]))
        landed = self.click(w, 100.512e6)
        self.assertLess(abs(landed - 100.512e6), 3e3)

    def test_a_keyed_carrier_is_found_with_the_key_up(self):
        """The demo's keyed CW beacon: in a frame with the key up a click
        landed on the pixel grid (a kHz or two off: silence in CW)."""
        carrier = CENTER - 280_123.0
        w = self.make([(carrier - CENTER, 0, -45.0)], step=10.0, mode="cw")
        for seed in range(6, 10):  # key up for the last few frames
            w.update_spectrum(_spectrum([], seed=seed))
        landed = self.click(w, carrier + 2e3)
        self.assertLess(abs(landed - carrier), 150.0)
        self.assertEqual(w.landing_frequency(self.x_of(w, carrier + 2e3)), landed)
        # Another LO: what was seen at the old one no longer counts.
        w.set_frequency_range(CENTER + 1e3, RATE)
        w.update_spectrum(_spectrum([], seed=11))
        w.update_spectrum(_spectrum([], seed=12))
        self.assertGreater(abs(self.click(w, carrier + 2e3) - carrier), 1e3)

    def test_the_marker_follows_the_tuned_frequency(self):
        w = self.make([])
        self.assertEqual(w.tuned_frequency(), CENTER)
        w.set_tuned_frequency(100.3e6)
        self.assertEqual(w.tuned_frequency(), 100.3e6)
        w.set_tuned_frequency(float("nan"))
        self.assertEqual(w.tuned_frequency(), CENTER)


class TestPassbandAtTheVfo(unittest.TestCase):
    """A sideband narrower than a pixel grows away from the tuned
    frequency, wherever in the span the VFO is."""

    def test_sub_pixel_sidebands_grow_away_from_the_marker(self):
        require_pyqt6(self)
        from PyQt6.QtCore import QRectF

        from sdr_module.gui.spectrum_widget import PASSBAND_MIN_PX, passband_px

        plot = QRectF(0, 0, 1000, 100)
        span, vfo = 2.4e6, -300e3
        marker_x = plot.center().x() + vfo * plot.width() / span
        usb = passband_px((vfo, vfo + 2.8e3), plot, span, anchor_hz=vfo)
        lsb = passband_px((vfo - 2.8e3, vfo), plot, span, anchor_hz=vfo)
        self.assertAlmostEqual(usb[0], round(marker_x), delta=1)
        self.assertGreaterEqual(usb[1] - usb[0], PASSBAND_MIN_PX - 1)
        self.assertAlmostEqual(lsb[1], round(marker_x), delta=1)
        self.assertLessEqual(lsb[0], lsb[1] - PASSBAND_MIN_PX + 1)


class TestWaterfallSnap(_PlotCase):
    def make(self, rows, step=100e3, mode="wfm"):
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        w = self.show(WaterfallWidget())
        w.set_frequency_range(CENTER, RATE)
        for row in rows:
            w.add_line(row)
        w.set_click_snap(step, mode)
        return w

    def click(self, w, freq, y, alt=False):
        mods = (
            Qt.KeyboardModifier.AltModifier if alt else Qt.KeyboardModifier.NoModifier
        )
        QTest.mouseClick(
            w._canvas,
            Qt.MouseButton.LeftButton,
            mods,
            QPointF(self.x_of(w, freq), y).toPoint(),
        )
        return self.clicked[-1]

    def test_a_click_on_a_trace_tunes_the_signal_in_those_rows(self):
        # An old burst at +300 kHz, then empty rows on top of it.
        burst = _spectrum([(300e3, 25e3, -45.0)])
        empty = _spectrum([], seed=9)
        w = self.make([burst] * 40 + [empty] * 40, step=12.5e3, mode="nfm")
        plot = w.plot_rect()
        row_px = plot.height() / w._history_size
        burst_y = plot.top() + 60 * row_px  # among the burst's rows
        empty_y = plot.top() + 5 * row_px
        self.assertEqual(self.click(w, 100.304e6, burst_y), 100.3e6)
        # Over empty rows the click goes to the step's raster only.
        self.assertEqual(self.click(w, 100.304e6, empty_y), 100.3e6)
        self.assertEqual(self.click(w, 100.309e6, empty_y), 100.3125e6)
        landed = self.click(w, 100.304e6, burst_y, alt=True)
        self.assertLess(abs(landed - 100.304e6), 3e3)


class TestWindowSnapsWithTheModesStep(unittest.TestCase):
    def test_the_window_gives_the_plots_its_step_and_mode(self):
        require_pyqt6(self)
        from sdr_module.gui import bookmarks_panel, main_window
        from tests.test_gui_chrome import _FakeSettings

        store = _FakeSettings()
        saved = (main_window.GuiSettings, bookmarks_panel.GuiSettings)
        main_window.GuiSettings = bookmarks_panel.GuiSettings = lambda: store
        try:
            win = main_window.SDRMainWindow()
        finally:
            main_window.GuiSettings, bookmarks_panel.GuiSettings = saved
        self.addCleanup(win.deleteLater)
        for timer in (win._display_timer, win._status_timer, win._hotplug_timer):
            timer.stop()
        for plot in (win._spectrum, win._waterfall):
            self.assertEqual((plot._snap_step, plot._snap_mode), (100e3, "wfm"))
        win._set_demod_mode("USB")
        for plot in (win._spectrum, win._waterfall):
            self.assertEqual((plot._snap_step, plot._snap_mode), (100.0, "usb"))


if __name__ == "__main__":
    unittest.main()

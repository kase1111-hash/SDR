"""The waterfall's Color Levels > Auto fits what is in view when zoomed.

The spectrum's Range > Auto follows the zoom (K-L2); the waterfall's Auto
levels kept fitting the whole span: zoomed x16 on the demo's -62 dBFS
beacon its colours still went up to -16 dBFS (a station out of view), so
the beacon and the noise around it shared a few colours at the bottom of
the scale. Now Auto (and Fit to Current Signals) fit the noise and the
strongest signal of the bins in view, refit at once when the zoom changes
or the zoomed view moves, and fit the whole span again at x1.

The rows' noise statistic the readouts use (the hover's SNR for rows
without their averaged spectrum, the measurement box's Peak, the public
``noise_floor_db``) stays the whole span's: only the colours follow the
zoom.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from tests.test_gui_chrome import _WindowTestCase
from tests.test_plot_readouts import CENTER, HAS_QT, RATE, _app, _settle, frames

if HAS_QT:
    from PyQt6.QtCore import QPointF

SIZE = 2048
#: A strong noise-like station +720 to +864 kHz from the center (out of
#: view at x16 around the center) 50 dB over the noise in a bin, and a weak
#: carrier 9.4 kHz from the center (in view) 20 dB over it.
STRONG = (0.30, 0.36, 50.0)
WEAK = ((8, 20.0),)


def _spectra(n=12, seed=7, strong=STRONG, weak=WEAK):
    return frames(SIZE, n=n, carriers=weak, wide=strong, seed=seed)


@unittest.skipUnless(HAS_QT, "PyQt6 is required")
class TestColorLevelsAutoFollowTheZoom(unittest.TestCase):
    def setUp(self):
        _app()
        from sdr_module.gui import waterfall_widget as wfm
        from sdr_module.gui.plot_view import PlotView

        self.wfm = wfm
        self.PlotView = PlotView
        self.t = 0

    def waterfall(self):
        wf = self.wfm.WaterfallWidget(history_size=100)
        self.addCleanup(wf.deleteLater)
        self.addCleanup(wf.close)
        wf.set_view_model(self.PlotView(CENTER, RATE))
        wf.set_frequency_range(CENTER, RATE)
        wf.set_fft_size(SIZE)
        wf.set_history_seconds(0.0)
        wf.resize(1000, 360)
        wf.show()
        _settle()
        return wf

    def feed(self, wf, spectra):
        for frame in spectra:
            wf.add_line(
                frame.max_db,
                timestamp=self.t / 30,
                noise_db=frame.noise_floor_db,
                mean_db=frame.mean_db,
            )
            self.t += 1

    def test_auto_fits_what_is_in_view_when_zoomed(self):
        """Zoomed x16 on the weak carrier, Auto kept the ceiling for the
        strong station out of view (about 50 dB over the noise): the weak
        carrier and the noise shared the bottom of the colours."""
        wf = self.waterfall()
        wf.set_auto_levels(True)
        spectra = _spectra()
        self.feed(wf, spectra)
        noise = wf.noise_floor_db()
        floor, ceiling = wf.levels()
        self.assertGreater(ceiling, noise + 40.0)  # the whole span: the strong one
        wf.set_zoom(16)  # around the center: the weak carrier in view
        _settle()
        floor16, ceiling16 = wf.levels()
        self.assertLess(ceiling16, noise + 30.0, (noise, wf.levels()))
        self.assertGreater(ceiling16, noise + 10.0, (noise, wf.levels()))
        self.assertLessEqual(floor16, noise, (noise, wf.levels()))
        self.assertTrue(wf.auto_levels())
        # New rows keep it on what is in view, even as the strong station
        # out of view gets stronger still...
        self.feed(wf, _spectra(n=6, seed=8, strong=(0.30, 0.36, 60.0)))
        self.assertLess(wf.levels()[1], noise + 30.0, wf.levels())
        # ...and follow a signal in view that gets stronger (20 dB more).
        self.feed(wf, _spectra(n=6, seed=9, weak=((8, 40.0),)))
        self.assertGreaterEqual(wf.levels()[1], ceiling16 + 8.0, wf.levels())

    def test_it_follows_a_pan_and_fits_the_whole_span_at_x1(self):
        wf = self.waterfall()
        wf.set_auto_levels(True)
        spectra = _spectra()
        self.feed(wf, spectra)
        noise = wf.noise_floor_db()
        wf.set_zoom(16)
        _settle()
        self.assertLess(wf.levels()[1], noise + 30.0)
        # The zoomed view moves to the strong station: refit at once.
        wf.zoom_to(CENTER + 790e3, 0)
        _settle()
        self.assertGreater(wf.levels()[1], noise + 40.0, wf.levels())
        wf.zoom_to(CENTER, 0)
        _settle()
        self.assertLess(wf.levels()[1], noise + 30.0, wf.levels())
        # Back at x1: the whole span's levels, from the whole span's noise
        # and strongest signal (what Auto shows when never zoomed).
        wf.set_zoom(1)
        _settle()
        self.assertEqual(
            wf.levels(),
            self.wfm.auto_level_targets(wf.noise_floor_db(), wf._peak_db),
        )

    def test_the_readouts_noise_statistic_ignores_the_zoom(self):
        """(Guard) The rows' noise floor the readouts judge against, and
        what they read, are the whole span's whatever the zoom."""
        zoomed, whole = self.waterfall(), self.waterfall()
        for wf in (zoomed, whole):
            wf.set_auto_levels(True)
        zoomed.set_zoom(16)
        _settle()
        spectra = _spectra()
        self.feed(zoomed, spectra)
        self.t = 0
        self.feed(whole, spectra)
        self.assertEqual(zoomed.noise_floor_db(), whole.noise_floor_db())
        self.assertEqual(zoomed._snr_noise(), whole._snr_noise())
        self.assertNotEqual(zoomed.levels(), whole.levels())  # (only the colours)
        # The measurement box's Peak over the weak carrier reads the same.
        texts = []
        for wf in (zoomed, whole):
            plot = wf.plot_rect()
            x0 = wf._freq_to_x(CENTER + 4e3, plot)
            x1 = wf._freq_to_x(CENTER + 15e3, plot)
            wf._drag_measure(QPointF(x0, plot.top() + 1), QPointF(x1, plot.top() + 40))
            wf._finish_measure()
            texts.append([line for line in wf._measure_lines() if "Peak" in line])
        self.assertTrue(texts[0], texts)
        self.assertEqual(texts[0], texts[1])

    def test_fixed_levels_stay_when_zoomed(self):
        """(Guard) Levels set by hand don't move with the zoom."""
        wf = self.waterfall()
        self.feed(wf, _spectra())
        wf.set_db_range(-120.0, -10.0)
        wf.set_zoom(16)
        _settle()
        self.feed(wf, _spectra(n=4, seed=8))
        self.assertEqual(wf.levels(), (-120.0, -10.0))
        self.assertFalse(wf.auto_levels())

    def test_fit_to_current_signals_fits_what_is_in_view(self):
        """Color Levels > Fit to Current Signals (manual levels, once)
        frames what is in view too."""
        wf = self.waterfall()
        self.feed(wf, _spectra())
        noise = wf.noise_floor_db()
        wf.set_zoom(16)
        _settle()
        wf._fit_levels_once()
        self.assertFalse(wf.auto_levels())
        self.assertLess(wf.levels()[1], noise + 30.0, wf.levels())
        wf.set_zoom(1)
        _settle()
        wf._fit_levels_once()
        self.assertGreater(wf.levels()[1], noise + 40.0, wf.levels())

    def test_rows_before_a_retune_set_nothing_in_view(self):
        """(Guard) Zoomed, after a retune the newest row is on the old
        frequency until a line arrives: the levels wait for one (no fit
        to bins of another frequency), then follow it."""
        wf = self.waterfall()
        wf.set_auto_levels(True)
        self.feed(wf, _spectra())
        noise = wf.noise_floor_db()
        wf.set_zoom(16)
        _settle()
        before = wf.levels()
        # Retuned 790 kHz up, the view stays on the same frequencies; the
        # newest row is still on the old ones.
        wf.set_frequency_range(CENTER + 790e3, RATE)
        _settle()
        self.assertEqual(wf.levels(), before)
        # Rows on the new frequency, with a strong station where the view
        # is (790 kHz below the new center): the levels follow it.
        self.feed(wf, _spectra(n=4, seed=10, strong=(-0.36, -0.30, 50.0)))
        self.assertGreater(wf.levels()[1], noise + 40.0, wf.levels())


@unittest.skipUnless(HAS_QT, "PyQt6 is required")
class TestTheWindowsWaterfallZoomed(_WindowTestCase):
    """The demo device as the window shows it: zoomed x16 on the -62 dBFS
    beacon at 99.82 MHz, the waterfall's colours went up to -16 dBFS (the
    100.1 MHz station, out of view) while the spectrum's range followed the
    zoom."""

    demo = True

    def setUp(self):
        super().setUp()
        from sdr_module.gui import device_dialog

        patcher = mock.patch.object(device_dialog, "_MAX_LAG_S", 1e9)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.win._device._rng = np.random.default_rng(3)

    def frames(self, count):
        win = self.win
        rate = win._device_sample_rate()
        for _ in range(count):
            win._pacer.reset()
            win._process_block(win._device.read_samples(int(rate / 30)))

    def test_zoomed_on_the_beacon(self):
        win = self.win
        win.set_frequency(100.1e6)
        if not win._is_running:
            win._start_acquisition()
        wf, sp = win._waterfall, win._spectrum
        wf.set_auto_levels(True)
        self.frames(20)
        whole = wf.levels()
        self.assertGreater(whole[1], -30.0, whole)  # the station at 100.1 MHz
        sp.zoom_to(99.82e6, 4)  # x16 (both plots share the view)
        self.settle()
        self.frames(10)
        floor, ceiling = wf.levels()
        self.assertLess(ceiling, -45.0, (whole, wf.levels()))
        self.assertGreater(ceiling, -62.0, (whole, wf.levels()))  # the beacon
        sp.set_zoom(1)
        self.settle()
        self.assertGreater(wf.levels()[1], -30.0, wf.levels())


if __name__ == "__main__":
    unittest.main()

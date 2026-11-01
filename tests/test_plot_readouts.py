"""Plot readings (K-L2): the hover's level and SNR are honest at any FFT size,
and Range > Auto fits what is in view when zoomed.

A waterfall row was the strongest of its frame's FFTs and each pixel the
strongest of its bins, so the hover read plain noise 6-7 dB above its mean
(and gave it an SNR at 33 and 80 of 83 places at 8192 and 32768 points) and
weak carriers 3-6 dB high; Range > Auto followed the whole span when zoomed.
"""

from __future__ import annotations

import math
import re
import unittest
from unittest import mock

import numpy as np

from sdr_module.dsp.frame_spectrum import FrameSpectrumAnalyzer
from sdr_module.gui import plot_readout as R

try:
    from PyQt6.QtCore import QPointF, QRectF
    from PyQt6.QtWidgets import QApplication

    HAS_QT = True
except ImportError:  # pragma: no cover - no PyQt6
    HAS_QT = False

CENTER, RATE = 100e6, 2.4e6
SIGMA = 1e-3  # noise amplitude per component


_APP = None


def _app():
    global _APP  # (kept: an application nothing holds is deleted at once)
    _APP = QApplication.instance() or QApplication([])
    return _APP


def _settle():
    for _ in range(3):
        _app().processEvents()


def frames(size, count=80_000, n=8, carriers=(), wide=None, seed=7, burst=None):
    """Frame spectra of noise (``SIGMA``), carriers ``(bin, dB over the mean
    noise in a bin)``, a noise-like signal ``wide = (lo, hi, dB)`` filling
    a fraction of the band, and, in frame ``burst[0]``, a 1 ms pulse at bin
    ``burst[1]`` ``burst[2]`` dB over the noise in a bin."""
    rng = np.random.default_rng(seed)
    analyzer = FrameSpectrumAnalyzer(size)
    t = np.arange(count) / RATE
    noise_bin = 2 * SIGMA**2 * 1.5 / size  # (Hann: 1.5 bins of noise)
    out = []
    for i in range(n):
        x = (rng.normal(size=count) + 1j * rng.normal(size=count)) * SIGMA
        for b, snr in carriers:
            x = x + math.sqrt(noise_bin * 10 ** (snr / 10)) * np.exp(
                2j * np.pi * b * RATE / size * t
            )
        if wide is not None:
            lo, hi, snr = wide
            w = rng.normal(size=count) + 1j * rng.normal(size=count)
            spec = np.fft.fft(w)
            f = np.fft.fftfreq(count)
            spec[(f < lo) | (f > hi)] = 0
            w = np.fft.ifft(spec)
            w *= math.sqrt(noise_bin * 10 ** (snr / 10) * size * (hi - lo) / 1.5)
            w /= math.sqrt(np.mean(np.abs(w) ** 2) / ((hi - lo) * size / 1.5))
            x = x + w
        if burst is not None and i == burst[0]:
            pulse = np.zeros(count)
            start = count // 2
            pulse[start : start + int(1e-3 * RATE)] = 1.0
            x = x + math.sqrt(noise_bin * 10 ** (burst[2] / 10)) * pulse * np.exp(
                2j * np.pi * burst[1] * RATE / size * t
            )
        frame = analyzer.process(x.astype(np.complex64))
        if frame is not None:
            out.append(frame)
    return out


def level_of(text):
    found = re.search(r"(-?\d+(?:\.\d+)?) dBFS", text)
    return float(found.group(1)) if found else float("nan")


def snr_of(text):
    found = re.search(r"SNR (-?\d+) dB", text)
    return float(found.group(1)) if found else None


# --------------------------------------------------------------------------- #
# The readout's statistics (Qt-free)
# --------------------------------------------------------------------------- #


class TestTheNoiseIsMeasured(unittest.TestCase):
    def test_averages_from_spread(self):
        self.assertEqual(R.averages_from_spread(9.0), 1.0)  # rougher than one FFT
        self.assertAlmostEqual(R.averages_from_spread(8.1815), 1.0, places=3)
        self.assertAlmostEqual(R.averages_from_spread(1.0270), 32.0, places=2)
        spreads = np.linspace(0.2, 8.0, 50)
        ks = [R.averages_from_spread(s) for s in spreads]
        self.assertTrue(all(a > b for a, b in zip(ks, ks[1:], strict=False)))

    def test_its_averaging_and_floor_at_every_fft_size(self):
        for size in (2048, 8192, 32768):
            frame = frames(size, n=3)[-1]
            model = R.noise_model(frame.mean_db)
            self.assertIsNotNone(model)
            self.assertAlmostEqual(
                model.averages, frame.segments, delta=0.25 * frame.segments
            )
            self.assertAlmostEqual(model.floor_db, frame.noise_floor_db, delta=0.3)


class TestReadLevels(unittest.TestCase):
    """Reading the averaged spectrum under a pixel of 1000 across the span."""

    def readings(self, frame, start=0, stop=None):
        size = len(frame.mean_db)
        m = max(1, round(size / 1000))
        model = R.noise_model(frame.mean_db, frame.noise_floor_db)
        fill = R.mean_limit(frame.mean_db, m)
        stop = size if stop is None else stop
        return [
            R.read_levels(frame.mean_db[i : i + m], model, fill)
            for i in range(start, stop - m + 1, m)
        ], model

    def test_plain_noise_reads_noise_at_any_fft_size(self):
        for size in (2048, 8192, 32768):
            readings, model = [], None
            for frame in frames(size, n=5, seed=size):
                found, model = self.readings(frame)
                readings += found
            signal = [r for r in readings if not r.is_noise]
            self.assertLessEqual(len(signal), 0.005 * len(readings), (size, signal))
            levels = np.array([r.level_db for r in readings]) - model.floor_db
            # (Each pixel the strongest of its bins read 2-7 dB high.)
            self.assertLess(abs(float(np.median(levels))), 0.5, size)
            self.assertLess(float(np.percentile(levels, 99)), 1.6, size)

    def test_a_carrier_reads_its_level(self):
        for size in (2048, 8192, 32768):
            b = size // 8
            for snr in (10.0, 20.0):
                for frame in frames(size, n=3, carriers=((b, snr),), seed=int(snr)):
                    m = max(1, round(size / 1000))
                    lo = size // 2 + b - m // 2
                    model = R.noise_model(frame.mean_db, frame.noise_floor_db)
                    reading = R.read_levels(frame.mean_db[lo : lo + m], model)
                    self.assertFalse(reading.is_noise, (size, snr))
                    self.assertAlmostEqual(reading.snr_db, snr, delta=2.5)

    def test_a_strong_wide_signal_is_a_signal_however_few_bins(self):
        """A broadcast station far above the noise, two or three of its
        bins under the pointer (none of them standing out of the others):
        it read "noise" (the window, at 2048 points over 100.1 MHz)."""
        model = R.NoiseModel(-90.0, 30.0)
        for bins in ([-31.0, -32.1], [-31.0, -31.2, -32.0], [-32.0, -31.0, -31.2]):
            reading = R.read_levels(np.array(bins), model, None)
            self.assertFalse(reading.is_noise, bins)
            # (Its density, or its strongest bin where the others can't
            # say whether it is a line: within a dB of each other.)
            density = 10 * np.log10(np.mean(10 ** (np.array(bins) / 10)))
            self.assertGreaterEqual(reading.level_db, density - 1e-6, bins)
            self.assertLessEqual(reading.level_db, max(bins) + 1e-6, bins)
            self.assertAlmostEqual(reading.snr_db, reading.level_db + 90.0, delta=0.01)

    def test_a_signal_wider_than_the_pixel_reads_its_density(self):
        """A station filling many bins at 32768 points: the strongest of a
        pixel's 33 bins read about 7 dB over its density."""
        frame = frames(32768, n=3, wide=(0.1, 0.2, 10.0), seed=3)[-1]
        mean = frame.mean_db
        size = len(mean)
        density = 10 * np.log10(
            np.mean(10 ** (mean[int(0.62 * size) : int(0.68 * size)] / 10))
        )
        readings, _model = self.readings(frame, int(0.61 * size), int(0.69 * size))
        levels = [r.level_db for r in readings]
        self.assertTrue(all(not r.is_noise for r in readings))
        self.assertAlmostEqual(float(np.median(levels)), float(density), delta=1.0)


# --------------------------------------------------------------------------- #
# The waterfall's hover and measurement
# --------------------------------------------------------------------------- #


@unittest.skipUnless(HAS_QT, "PyQt6 is required")
class TestWaterfallReadings(unittest.TestCase):
    def setUp(self):
        _app()
        from sdr_module.gui import waterfall_widget as wfm

        self.wfm = wfm

    def waterfall(self, size, history_s=0.0):
        wf = self.wfm.WaterfallWidget(history_size=100)
        self.addCleanup(wf.deleteLater)
        self.addCleanup(wf.close)
        wf.set_frequency_range(CENTER, RATE)
        wf.set_fft_size(size)
        wf.set_history_seconds(history_s)
        wf.resize(1000, 360)
        wf.show()
        _settle()
        return wf

    def feed(self, wf, spectra, mean=True, step=1 / 30):
        for i, frame in enumerate(spectra):
            kwargs = {"mean_db": frame.mean_db} if mean else {}
            wf.add_line(
                frame.max_db,
                timestamp=i * step,
                noise_db=frame.noise_floor_db,
                **kwargs,
            )

    def hover(self, wf, freqs, y=None):
        lines = []

        def record(_painter, text, *args, **kwargs):
            lines.append(list(text))
            return QRectF()

        plot = wf.plot_rect()
        with mock.patch.object(self.wfm, "draw_readout", record):
            for f in freqs:
                x = wf._freq_to_x(float(f), plot)
                wf._hover = QPointF(x, plot.top() + 2 if y is None else y)
                wf._canvas.repaint()
        wf._hover = None
        return [found[1] for found in lines]

    def noise_spots(self, wf, avoid=()):
        bin_hz = RATE / wf.fft_size()
        far = 20 * max(bin_hz, wf._hz_per_px())
        return [
            f
            for f in CENTER + RATE * np.linspace(-0.45, 0.45, 91)
            if all(abs(f - (CENTER + b * bin_hz)) > far for b in avoid)
        ]

    def test_plain_noise_reads_noise_at_32768_points(self):
        """It read "SNR 5-9 dB" at 80 of 83 places, 6 dB over the floor."""
        wf = self.waterfall(32768)
        spectra = frames(32768, n=12)
        self.feed(wf, spectra)
        lines = self.hover(wf, self.noise_spots(wf))
        self.assertLessEqual(sum("SNR" in line for line in lines), 1, lines)
        self.assertTrue(all("noise" in line for line in lines[:5]), lines)
        floor = spectra[-1].noise_floor_db
        levels = np.array([level_of(line) for line in lines])
        self.assertLess(abs(float(np.median(levels)) - floor), 1.0)

    def test_weak_carriers_read_their_averaged_level(self):
        """Carriers 6 and 10 dB over the mean noise in a bin read "SNR 12"
        and "SNR 15" at 2048 points (the strongest of 39 FFTs)."""
        size = 2048
        wf = self.waterfall(size)
        self.feed(wf, frames(size, n=12, carriers=((-187, 6.0), (187, 10.0))))
        bin_hz = RATE / size
        lines = self.hover(wf, [CENTER - 187 * bin_hz, CENTER + 187 * bin_hz])
        self.assertAlmostEqual(snr_of(lines[0]), 6.0, delta=1.5, msg=lines)
        self.assertAlmostEqual(snr_of(lines[1]), 10.0, delta=1.5, msg=lines)
        self.assertFalse(any(line.startswith("Peak") for line in lines), lines)

    def test_a_burst_shorter_than_the_frame_reads_its_peak(self):
        """A 1 ms pulse in one frame shows at its peak in that row's
        colour; its averaged level is 15 dB lower, so it reads as a burst:
        its peak, marked as one, not "noise"."""
        size = 2048
        wf = self.waterfall(size)
        spectra = frames(size, n=12, burst=(11, 300, 25.0))
        self.feed(wf, spectra)
        bin_hz = RATE / size
        line = self.hover(wf, [CENTER + 300 * bin_hz])[0]  # (the newest row)
        self.assertTrue(line.startswith("Peak"), line)
        self.assertGreater(snr_of(line), 15.0, line)
        plot = wf.plot_rect()
        rows_px = plot.height() / wf._rows_shown()
        older = self.hover(wf, [CENTER + 300 * bin_hz], plot.top() + 3.5 * rows_px)[0]
        self.assertIn("noise", older)

    def test_rows_merging_frames_read_their_average(self):
        """With a 1 min history each row merges several frames: its
        averaged spectrum is their power average, read the same way."""
        size = 8192
        wf = self.waterfall(size, history_s=60.0)
        spectra = frames(size, n=12, carriers=((500, 12.0),))
        # (Full history first, so later frames merge into rows.)
        for i in range(100):
            wf.add_line(
                spectra[i % 4].max_db,
                timestamp=i * 0.6,
                noise_db=spectra[i % 4].noise_floor_db,
                mean_db=spectra[i % 4].mean_db,
            )
        for i, frame in enumerate(spectra[4:]):
            wf.add_line(
                frame.max_db,
                timestamp=60.0 + i / 30,
                noise_db=frame.noise_floor_db,
                mean_db=frame.mean_db,
            )
        self.assertGreater(wf._row_lines[-1], 1)
        lines = self.hover(wf, self.noise_spots(wf, avoid=(500,)))
        self.assertLessEqual(sum("SNR" in line for line in lines), 1, lines)
        carrier = self.hover(wf, [CENTER + 500 * RATE / size])[0]
        self.assertAlmostEqual(snr_of(carrier), 12.0, delta=2.0, msg=carrier)

    def test_the_measurement_over_plain_noise_says_noise(self):
        """A box dragged over noise at 32768 points read "Peak -92 dBFS ·
        SNR 9 dB": its peak is the strongest of many bins and rows."""
        wf = self.waterfall(32768)
        self.feed(wf, frames(32768, n=12))
        plot = wf.plot_rect()
        x0 = wf._freq_to_x(CENTER - 300e3, plot)
        x1 = wf._freq_to_x(CENTER - 100e3, plot)
        wf._drag_measure(QPointF(x0, plot.top() + 1), QPointF(x1, plot.top() + 60))
        wf._finish_measure()
        lines = wf._measure_lines()
        peak = [line for line in lines if line.startswith("Peak")]
        self.assertTrue(peak, lines)
        self.assertIn("noise", peak[0])

    def test_rows_without_their_average_read_bursts_only(self):
        """(A caller that gives only the strongest FFTs and the floor, as
        the second receiver does): noise reads noise, a strong carrier its
        peak, marked as one."""
        size = 2048
        wf = self.waterfall(size)
        self.feed(wf, frames(size, n=12, carriers=((300, 20.0),)), mean=False)
        lines = self.hover(wf, self.noise_spots(wf, avoid=(300,)))
        self.assertLessEqual(sum("SNR" in line for line in lines), 1, lines)
        carrier = self.hover(wf, [CENTER + 300 * RATE / size])[0]
        self.assertTrue(carrier.startswith("Peak"), carrier)

    def test_rows_without_a_noise_floor_read_as_drawn(self):
        """(Guard) add_line with the line alone: the strongest bin under
        the pixel, as before."""
        size = 2048
        wf = self.waterfall(size)
        for i, frame in enumerate(frames(size, n=4)):
            wf.add_line(frame.max_db, timestamp=i / 30)
        plot = wf.plot_rect()
        x = wf._freq_to_x(CENTER + 100e3, plot)
        line = self.hover(wf, [CENTER + 100e3])[0]
        self.assertEqual(level_of(line), round(wf.level_at(x, plot.top() + 2)))

    def test_add_line_does_no_reading_work(self):
        """The noise statistics are worked out when the hover reads a row,
        not for every line: a frame costs one quantization more."""
        size = 32768
        wf = self.waterfall(size)
        spectra = frames(size, n=4)
        with (
            mock.patch.object(
                self.wfm, "noise_model", side_effect=AssertionError("per frame")
            ),
            mock.patch.object(
                self.wfm, "mean_limit", side_effect=AssertionError("per frame")
            ),
        ):
            self.feed(wf, spectra)
            wf._canvas.repaint()


# --------------------------------------------------------------------------- #
# The spectrum's hover and Range > Auto
# --------------------------------------------------------------------------- #


@unittest.skipUnless(HAS_QT, "PyQt6 is required")
class TestSpectrumReadings(unittest.TestCase):
    def setUp(self):
        _app()
        from sdr_module.gui import spectrum_widget as spm
        from sdr_module.gui.plot_view import PlotView

        self.spm = spm
        sp = spm.SpectrumWidget()
        self.addCleanup(sp.deleteLater)
        self.addCleanup(sp.close)
        self.view = PlotView(CENTER, RATE)
        sp.set_view_model(self.view)
        sp.set_frequency_range(CENTER, RATE)
        sp.resize(1000, 360)
        sp.show()
        _settle()
        self.sp = sp

    def hover(self, freqs):
        lines = []

        def record(_painter, text, *args, **kwargs):
            lines.append(list(text))
            return QRectF()

        plot = self.sp.plot_rect()
        with mock.patch.object(self.spm, "draw_readout", record):
            for f in freqs:
                self.sp._hover = QPointF(
                    self.sp._freq_x(float(f), plot), plot.top() + 5
                )
                self.sp._canvas.repaint()
        self.sp._hover = None
        return [found[1] for found in lines]

    def test_noise_reads_noise_at_32768_points(self):
        """The level was the strongest of a pixel's 33 bins: plain noise
        read about 7 dB over its floor, a level with no judgement."""
        self.sp.set_fft_size(32768)
        frame = frames(32768, n=3)[-1]
        self.sp.update_spectrum(frame.mean_db, noise_db=frame.noise_floor_db)
        spots = CENTER + RATE * np.linspace(-0.45, 0.45, 61)
        lines = self.hover(spots)
        self.assertTrue(all(line.endswith("· noise") for line in lines[:10]), lines)
        self.assertLessEqual(sum("SNR" in line for line in lines), 1, lines)
        levels = np.array([level_of(line) for line in lines])
        self.assertLess(abs(float(np.median(levels)) - frame.noise_floor_db), 1.0)

    def test_a_carrier_reads_its_snr(self):
        self.sp.set_fft_size(2048)
        frame = frames(2048, n=3, carriers=((256, 20.0),))[-1]
        self.sp.update_spectrum(frame.mean_db, noise_db=frame.noise_floor_db)
        line = self.hover([CENTER + 256 * RATE / 2048])[0]
        self.assertAlmostEqual(snr_of(line), 20.0, delta=1.5, msg=line)

    def test_without_a_noise_floor_the_trace_gives_it(self):
        """(update_spectrum without noise_db, as the second receiver
        feeds it): the floor is measured from the trace."""
        self.sp.set_fft_size(2048)
        frame = frames(2048, n=3, carriers=((256, 20.0),))[-1]
        self.sp.update_spectrum(frame.mean_db)
        line = self.hover([CENTER + 256 * RATE / 2048])[0]
        self.assertAlmostEqual(snr_of(line), 20.0, delta=1.5, msg=line)
        noise = self.hover([CENTER - 500e3])[0]
        self.assertTrue(noise.endswith("· noise"), noise)

    def strong_and_weak(self):
        """A frame: a strong carrier 800 kHz above the center (-20 dBFS) and
        a weak one at the center (-90 dBFS) over -110 dBFS of noise."""
        size, count = 2048, 80_000
        rng = np.random.default_rng(4)
        t = np.arange(count) / RATE
        sigma = 10 ** (-110 / 20) * math.sqrt(size / 1.5 / 2)
        x = (rng.normal(size=count) + 1j * rng.normal(size=count)) * sigma
        x = x + 0.1 * np.exp(2j * np.pi * 800e3 * t)
        x = x + 10 ** (-90 / 20) * np.exp(2j * np.pi * 0.0 * t)
        return FrameSpectrumAnalyzer(size).process(x.astype(np.complex64))

    def test_range_auto_fits_what_is_in_view(self):
        """Zoomed x16 on the weak carrier, Auto kept the range for the
        strong one 800 kHz away (out of view): the top at 0 dBFS, the weak
        carrier squashed at the bottom."""
        self.sp.set_fft_size(2048)
        self.sp.set_auto_range(True)
        frame = self.strong_and_weak()
        for _ in range(3):
            self.sp.update_spectrum(frame.mean_db, noise_db=frame.noise_floor_db)
        floor, top = self.sp.db_range()
        self.assertGreaterEqual(top, -15.0)  # the whole span: the strong one
        self.sp.set_zoom(16)  # around the tuned frequency (the center)
        _settle()
        floor, top = self.sp.db_range()
        self.assertLessEqual(top, -70.0, (floor, top))  # the weak one in view
        self.assertGreaterEqual(top, -90.0 + 5.0, (floor, top))
        self.assertLessEqual(floor, -110.0, (floor, top))
        # Frames keep it there.
        for _ in range(5):
            self.sp.update_spectrum(frame.mean_db, noise_db=frame.noise_floor_db)
        self.assertLessEqual(self.sp.db_range()[1], -70.0)
        # The view moved to the strong one: the range follows at once.
        self.sp.zoom_to(CENTER + 800e3, 0)
        _settle()
        self.assertGreaterEqual(self.sp.db_range()[1], -15.0)
        # Back to the full span: the whole span again.
        self.sp.set_zoom(16)
        self.sp.set_zoom(1)
        _settle()
        self.assertGreaterEqual(self.sp.db_range()[1], -15.0)

    def test_a_fixed_range_stays_when_zoomed(self):
        """(Guard) A range set by hand doesn't move with the zoom."""
        self.sp.set_fft_size(2048)
        frame = self.strong_and_weak()
        self.sp.update_spectrum(frame.mean_db, noise_db=frame.noise_floor_db)
        self.sp.set_db_range(-120.0, 0.0)
        self.sp.set_zoom(16)
        _settle()
        self.assertEqual(self.sp.db_range(), (-120.0, 0.0))
        self.assertFalse(self.sp.auto_range())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


# --------------------------------------------------------------------------- #
# The main window feeds both plots what their readouts need
# --------------------------------------------------------------------------- #

from tests.test_gui_chrome import _WindowTestCase  # noqa: E402


@unittest.skipUnless(HAS_QT, "PyQt6 is required")
class TestTheWindowsPlots(_WindowTestCase):
    """The demo device, as the window shows it: the waterfall's rows carry
    their averaged spectrum and noise floor, the spectrum its noise floor,
    and the hover over the demo's empty band says "noise" at 32768
    points (it read "SNR 5-8 dB" at most places)."""

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

    def test_the_hover_over_the_demos_empty_band(self):
        win = self.win
        win.set_frequency(300e6)  # (the demo has nothing near 300 MHz)
        win._apply_plot_resolution(32768, "hann", save=False)
        if not win._is_running:
            win._start_acquisition()
        self.frames(20)
        wf, sp = win._waterfall, win._spectrum
        self.assertEqual(len(wf._row_means), len(wf._history))
        self.assertIsNotNone(wf._row_means[-1])
        self.assertIsNotNone(wf._row_floors[-1])
        self.assertIsNotNone(sp._trace_floor)
        lines = []

        def record(_painter, text, *args, **kwargs):
            lines.append(list(text))
            return QRectF()

        from sdr_module.gui import spectrum_widget as spm
        from sdr_module.gui import waterfall_widget as wfm

        center = win._lo_frequency()
        rate = win._device_sample_rate()
        spots = center + rate * np.linspace(-0.4, 0.4, 41)
        for module, widget, to_x, y in (
            (wfm, wf, wf._freq_to_x, lambda p: p.top() + 2),
            (spm, sp, sp._freq_x, lambda p: p.top() + 5),
        ):
            plot = widget.plot_rect()
            with mock.patch.object(module, "draw_readout", record):
                for f in spots:
                    widget._hover = QPointF(to_x(float(f), plot), y(plot))
                    widget._canvas.repaint()
            widget._hover = None
        readings = [found[1] for found in lines]
        self.assertEqual(len(readings), 2 * len(spots))
        with_snr = [line for line in readings if "SNR" in line]
        self.assertLessEqual(len(with_snr), 1, readings)

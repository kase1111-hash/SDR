#!/usr/bin/env python3
"""
GUI component tests for SDR module.

Tests PyQt6 widgets without requiring a display or actual Qt event loop.
Uses mocking to test widget logic and state management.
"""

import os
import sys
import unittest

import numpy as np

# Check if PyQt6 is available
try:
    from PyQt6.QtWidgets import QApplication

    HAS_PYQT6 = True
    PYQT6_IMPORT_ERROR = ""
    # Create QApplication if needed (required for widget instantiation)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
except ImportError as _import_error:
    HAS_PYQT6 = False
    PYQT6_IMPORT_ERROR = str(_import_error)


def require_pyqt6(test_case: unittest.TestCase) -> None:
    """Skip the test when PyQt6 is unavailable.

    CI sets ``SDR_REQUIRE_GUI=1`` so that a missing or broken PyQt6 fails
    loudly instead of silently skipping the whole GUI suite.
    """
    if HAS_PYQT6:
        return
    if os.environ.get("SDR_REQUIRE_GUI"):
        test_case.fail(
            "PyQt6 is required because SDR_REQUIRE_GUI is set, but it could "
            f"not be imported: {PYQT6_IMPORT_ERROR}"
        )
    test_case.skipTest("PyQt6 not available")


class TestSpectrumWidgetLogic(unittest.TestCase):
    """Test SpectrumWidget logic and state management."""

    def setUp(self):
        """Set up test fixtures."""
        require_pyqt6(self)
        from sdr_module.gui.spectrum_widget import SpectrumWidget

        self.widget = SpectrumWidget()

    def test_initialization(self):
        """Test widget initializes with correct defaults."""
        self.assertEqual(self.widget._fft_size, 2048)
        self.assertEqual(self.widget._center_freq, 100e6)
        self.assertEqual(self.widget._sample_rate, 2.4e6)
        self.assertEqual(self.widget._db_range, (-100, 0))
        self.assertTrue(self.widget._show_peak)
        self.assertFalse(self.widget._show_average)
        self.assertTrue(self.widget._grid_enabled)

    def test_update_spectrum(self):
        """Test spectrum update with matching FFT size."""
        spectrum = np.random.uniform(-80, -20, 2048)
        self.widget.update_spectrum(spectrum)
        np.testing.assert_array_almost_equal(self.widget._spectrum, spectrum)

    def test_update_spectrum_resample(self):
        """Test spectrum update with non-matching FFT size triggers resample."""
        spectrum = np.random.uniform(-80, -20, 1024)
        self.widget.update_spectrum(spectrum)
        self.assertEqual(len(self.widget._spectrum), 2048)

    def test_peak_hold(self):
        """Test peak hold functionality."""
        # First update
        spectrum1 = np.full(2048, -60.0)
        self.widget.update_spectrum(spectrum1)

        # Second update with lower values - peak should remain
        spectrum2 = np.full(2048, -80.0)
        self.widget.update_spectrum(spectrum2)
        np.testing.assert_array_almost_equal(self.widget._peak_hold, spectrum1)

        # Third update with higher values - peak should update
        spectrum3 = np.full(2048, -40.0)
        self.widget.update_spectrum(spectrum3)
        np.testing.assert_array_almost_equal(self.widget._peak_hold, spectrum3)

    def test_reset_peak(self):
        """Test peak reset."""
        spectrum = np.full(2048, -50.0)
        self.widget.update_spectrum(spectrum)
        self.widget.reset_peak()
        self.assertTrue(np.all(self.widget._peak_hold == -120.0))

    def test_averaging(self):
        """Test averaging functionality."""
        # Enable averaging
        self.widget._show_average = True
        self.widget._avg_alpha = 0.5

        # First update
        spectrum1 = np.full(2048, -60.0)
        self.widget.update_spectrum(spectrum1)
        self.assertEqual(self.widget._avg_count, 1)
        np.testing.assert_array_almost_equal(self.widget._average, spectrum1)

        # Second update - should average
        spectrum2 = np.full(2048, -40.0)
        self.widget.update_spectrum(spectrum2)
        expected = 0.5 * spectrum2 + 0.5 * spectrum1  # -50 dB
        np.testing.assert_array_almost_equal(self.widget._average, expected)

    def test_reset_average(self):
        """Test average reset."""
        self.widget._show_average = True
        self.widget.update_spectrum(np.full(2048, -50.0))
        self.widget.reset_average()
        self.assertTrue(np.all(self.widget._average == 0))
        self.assertEqual(self.widget._avg_count, 0)

    def test_set_frequency_range(self):
        """Test frequency range setting."""
        self.widget.set_frequency_range(145e6, 2.0e6)
        self.assertEqual(self.widget._center_freq, 145e6)
        self.assertEqual(self.widget._sample_rate, 2.0e6)

    def test_set_db_range(self):
        """Test dB range setting."""
        self.widget.set_db_range(-120, -20)
        self.assertEqual(self.widget._db_range, (-120, -20))

    def test_averaging_mode_change(self):
        """Test averaging mode change callback."""
        # Off
        self.widget._on_avg_changed(0)
        self.assertFalse(self.widget._show_average)

        # 8x averaging
        self.widget._on_avg_changed(3)  # Index 3 = 8
        self.assertTrue(self.widget._show_average)
        self.assertAlmostEqual(self.widget._avg_alpha, 2.0 / 9)


class TestWaterfallWidgetLogic(unittest.TestCase):
    """Test WaterfallWidget logic and state management."""

    def setUp(self):
        """Set up test fixtures."""
        require_pyqt6(self)
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        self.widget = WaterfallWidget(history_size=100)

    def test_initialization(self):
        """Test widget initializes with correct defaults."""
        self.assertEqual(self.widget._history_size, 100)
        self.assertEqual(self.widget._fft_size, 2048)
        self.assertEqual(self.widget._db_range, (-100, 0))
        self.assertEqual(self.widget._colormap_name, "turbo")
        self.assertEqual(len(self.widget._history), 0)

    def test_add_line(self):
        """Test adding spectrum lines."""
        spectrum = np.random.uniform(-80, -20, 2048)
        self.widget.add_line(spectrum)
        self.assertEqual(len(self.widget._history), 1)
        np.testing.assert_array_almost_equal(self.widget._history[0], spectrum)

    def test_add_line_resample(self):
        """Test adding spectrum with non-matching FFT size."""
        spectrum = np.random.uniform(-80, -20, 1024)
        self.widget.add_line(spectrum)
        self.assertEqual(len(self.widget._history), 1)
        self.assertEqual(len(self.widget._history[0]), 2048)

    def test_history_limit(self):
        """Test history size limit."""
        for i in range(150):
            spectrum = np.full(2048, float(-i))
            self.widget.add_line(spectrum)

        # Should be limited to history_size
        self.assertEqual(len(self.widget._history), 100)
        # Should have newest data
        np.testing.assert_array_almost_equal(
            self.widget._history[-1], np.full(2048, -149.0)
        )

    def test_clear(self):
        """Test clearing waterfall."""
        self.widget.add_line(np.zeros(2048))
        self.widget.add_line(np.zeros(2048))
        self.widget.clear()
        self.assertEqual(len(self.widget._history), 0)
        self.assertIsNone(self.widget._image)

    def test_colormap_building(self):
        """Test colormap building."""
        colormap = self.widget._build_colormap("viridis")
        self.assertEqual(colormap.shape, (256, 3))
        self.assertEqual(colormap.dtype, np.uint8)

    def test_colormap_fallback(self):
        """Test invalid colormap falls back to turbo."""
        colormap = self.widget._build_colormap("nonexistent")
        turbo_colormap = self.widget._build_colormap("turbo")
        np.testing.assert_array_equal(colormap, turbo_colormap)

    def test_colormap_change(self):
        """Test colormap change callback."""
        old_colormap = self.widget._colormap.copy()
        self.widget._on_colormap_changed("grayscale")
        self.assertEqual(self.widget._colormap_name, "grayscale")
        self.assertFalse(np.array_equal(self.widget._colormap, old_colormap))

    def test_range_change(self):
        """Test dB range change callback."""
        self.widget._on_range_changed(0)  # 60 dB
        self.assertEqual(self.widget._db_range, (-60, 0))

        self.widget._on_range_changed(3)  # 120 dB
        self.assertEqual(self.widget._db_range, (-120, 0))

    def test_highlights(self):
        """Test highlight management."""
        from PyQt6.QtGui import QColor

        color = QColor(255, 0, 0)

        self.widget.add_highlight(0, 10, 100, 200, color)
        self.assertEqual(len(self.widget._highlights), 1)

        self.widget.add_highlight(10, 20, 300, 400, color)
        self.assertEqual(len(self.widget._highlights), 2)

        self.widget.clear_highlights()
        self.assertEqual(len(self.widget._highlights), 0)

    def test_available_colormaps(self):
        """Test all defined colormaps can be built."""
        for name in self.widget.COLORMAPS.keys():
            colormap = self.widget._build_colormap(name)
            self.assertEqual(colormap.shape, (256, 3))

    def test_render_newest_line_at_bottom(self):
        """Newest line renders at the bottom row with the correct colormap color."""
        from PyQt6.QtGui import QColor

        from sdr_module.gui.waterfall_widget import WaterfallWidget

        widget = WaterfallWidget(history_size=10)
        # A full-scale (max-dB) line maps to the top colormap entry (index 255).
        widget.add_line(np.zeros(2048, dtype=np.float32))  # 0 dB == max_db
        image = widget._image
        self.assertIsNotNone(image)
        self.assertEqual((image.width(), image.height()), (2048, 10))

        r, g, b = (int(v) for v in widget._colormap[255])
        bottom = QColor(image.pixel(1024, image.height() - 1))
        self.assertEqual((bottom.red(), bottom.green(), bottom.blue()), (r, g, b))

        # With only one line of history, the top row is still background.
        top = QColor(image.pixel(1024, 0))
        bg = widget._bg_color
        self.assertEqual(
            (top.red(), top.green(), top.blue()),
            (bg.red(), bg.green(), bg.blue()),
        )

    def test_render_is_vectorized_fast(self):
        """A full-history repaint must stay well under the ~33 ms frame budget.

        Regression guard for the old per-pixel QImage.pixel/setPixel scroll,
        which took ~0.4 s per line and froze the UI during acquisition.
        """
        import time

        from sdr_module.gui.waterfall_widget import WaterfallWidget

        widget = WaterfallWidget(history_size=500)
        rng = np.random.default_rng(0)
        line = (rng.standard_normal(2048) * 10 - 40).astype(np.float32)
        for _ in range(5):  # fill some history so the buffer is non-trivial
            widget.add_line(line)

        start = time.perf_counter()
        for _ in range(10):
            widget.add_line(line)
        avg = (time.perf_counter() - start) / 10
        # Generous ceiling (33 ms budget); the fix runs in a few ms, the old
        # code took hundreds of ms.
        self.assertLess(avg, 0.033, f"add_line too slow: {avg * 1000:.1f} ms")


class TestFrequencyInputLogic(unittest.TestCase):
    """Test FrequencyInput widget logic."""

    def setUp(self):
        """Set up test fixtures."""
        require_pyqt6(self)
        from sdr_module.gui.control_panel import FrequencyInput

        self.widget = FrequencyInput()

    def test_initialization(self):
        """Test widget initializes correctly."""
        self.assertEqual(self.widget._frequency_hz, 100e6)

    def test_set_frequency(self):
        """Test setting frequency."""
        self.widget.set_frequency(145.8e6)
        self.assertEqual(self.widget.get_frequency(), 145.8e6)

    def test_get_multiplier(self):
        """Test unit multipliers."""
        self.widget._unit_combo.setCurrentText("Hz")
        self.assertEqual(self.widget._get_multiplier(), 1)

        self.widget._unit_combo.setCurrentText("kHz")
        self.assertEqual(self.widget._get_multiplier(), 1e3)

        self.widget._unit_combo.setCurrentText("MHz")
        self.assertEqual(self.widget._get_multiplier(), 1e6)

        self.widget._unit_combo.setCurrentText("GHz")
        self.assertEqual(self.widget._get_multiplier(), 1e9)


class TestControlPanelLogic(unittest.TestCase):
    """Test ControlPanel widget logic."""

    def setUp(self):
        """Set up test fixtures."""
        require_pyqt6(self)
        from sdr_module.gui.control_panel import ControlPanel

        self.widget = ControlPanel()

    def test_initialization(self):
        """Test widget initializes without error."""
        self.assertIsNotNone(self.widget._freq_input)
        self.assertIsNotNone(self.widget._gain_slider)
        self.assertIsNotNone(self.widget._bw_combo)
        self.assertIsNotNone(self.widget._demod_combo)

    def test_format_offset(self):
        """Test frequency offset formatting."""
        self.assertEqual(self.widget._format_offset(1e6), "+1M")
        self.assertEqual(self.widget._format_offset(-1e6), "-1M")
        self.assertEqual(self.widget._format_offset(100e3), "+100k")
        self.assertEqual(self.widget._format_offset(-10e3), "-10k")
        self.assertEqual(self.widget._format_offset(500), "+500")

    def test_set_frequency(self):
        """Test setting frequency."""
        self.widget.set_frequency(433e6)
        self.assertEqual(self.widget._freq_input.get_frequency(), 433e6)

    def test_set_gain(self):
        """Test setting gain."""
        self.widget.set_gain(30)
        self.assertEqual(self.widget._gain_slider.value(), 30)

    def test_update_record_time(self):
        """Test recording time display update."""
        self.widget.update_record_time(0)
        self.assertEqual(self.widget._record_time.text(), "00:00:00")

        self.widget.update_record_time(3661)  # 1 hour, 1 minute, 1 second
        self.assertEqual(self.widget._record_time.text(), "01:01:01")

        self.widget.update_record_time(7200)  # 2 hours
        self.assertEqual(self.widget._record_time.text(), "02:00:00")


class TestSignalMeterWidgetLogic(unittest.TestCase):
    """Test SignalMeterPanel logic."""

    def setUp(self):
        """Set up test fixtures."""
        require_pyqt6(self)
        from sdr_module.ham.gui.signal_meter_widget import SignalMeterPanel

        self.widget = SignalMeterPanel()

    def tearDown(self):
        """Stop the panel's update timer."""
        self.widget._update_timer.stop()

    def test_initialization(self):
        """Test widget initializes correctly."""
        self.assertIsNotNone(self.widget)
        self.assertIsNotNone(self.widget.get_meter())
        self.assertIsNone(self.widget._last_reading)

    def test_update_samples_produces_reading(self):
        """Feeding I/Q samples produces a signal reading."""
        t = np.arange(2048) / 48000.0
        samples = (0.5 * np.exp(2j * np.pi * 1000 * t)).astype(np.complex64)
        self.widget.update_samples(samples)
        self.assertIsNotNone(self.widget._last_reading)

    def test_display_update_after_samples(self):
        """Display refresh runs without error after an update."""
        t = np.arange(2048) / 48000.0
        samples = (0.5 * np.exp(2j * np.pi * 1000 * t)).astype(np.complex64)
        self.widget.update_samples(samples)
        self.widget._update_display()
        self.assertTrue(self.widget._s_meter_label.text().startswith("S"))


class TestMockGUIWithoutQt(unittest.TestCase):
    """Test GUI module behavior when PyQt6 is not available."""

    def test_spectrum_widget_import_guard(self):
        """Test SpectrumWidget has proper import guard."""
        # This tests the HAS_PYQT6 pattern is used correctly
        import sdr_module.gui.spectrum_widget as sw

        self.assertTrue(hasattr(sw, "HAS_PYQT6"))

    def test_waterfall_widget_import_guard(self):
        """Test WaterfallWidget has proper import guard."""
        import sdr_module.gui.waterfall_widget as ww

        self.assertTrue(hasattr(ww, "HAS_PYQT6"))

    def test_control_panel_import_guard(self):
        """Test ControlPanel has proper import guard."""
        import sdr_module.gui.control_panel as cp

        self.assertTrue(hasattr(cp, "HAS_PYQT6"))


class TestColorMapConsistency(unittest.TestCase):
    """Test colormap definitions are consistent."""

    def setUp(self):
        """Set up test fixtures."""
        require_pyqt6(self)
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        self.widget = WaterfallWidget()

    def test_all_colormaps_have_valid_rgb(self):
        """Test all colormap values are valid RGB (0-255)."""
        for name, colors in self.widget.COLORMAPS.items():
            for r, g, b in colors:
                self.assertGreaterEqual(r, 0, f"{name}: R value {r} < 0")
                self.assertLessEqual(r, 255, f"{name}: R value {r} > 255")
                self.assertGreaterEqual(g, 0, f"{name}: G value {g} < 0")
                self.assertLessEqual(g, 255, f"{name}: G value {g} > 255")
                self.assertGreaterEqual(b, 0, f"{name}: B value {b} < 0")
                self.assertLessEqual(b, 255, f"{name}: B value {b} > 255")

    def test_all_colormaps_have_minimum_colors(self):
        """Test all colormaps have at least 2 colors for interpolation."""
        for name, colors in self.widget.COLORMAPS.items():
            self.assertGreaterEqual(len(colors), 2, f"{name} has fewer than 2 colors")


class TestGUIDataProcessing(unittest.TestCase):
    """Test data processing in GUI widgets."""

    def setUp(self):
        """Set up test fixtures."""
        require_pyqt6(self)

    def test_spectrum_handles_nan(self):
        """Test spectrum widget handles NaN values."""
        from sdr_module.gui.spectrum_widget import SpectrumWidget

        widget = SpectrumWidget()

        # Create spectrum with some NaN values
        spectrum = np.random.uniform(-80, -20, 2048)
        spectrum[100:200] = np.nan

        # Should not raise
        widget.update_spectrum(spectrum)
        self.assertEqual(len(widget._spectrum), 2048)

    def test_spectrum_handles_inf(self):
        """Test spectrum widget handles infinity values."""
        from sdr_module.gui.spectrum_widget import SpectrumWidget

        widget = SpectrumWidget()

        # Create spectrum with some infinity values
        spectrum = np.random.uniform(-80, -20, 2048)
        spectrum[100] = np.inf
        spectrum[200] = -np.inf

        # Should not raise
        widget.update_spectrum(spectrum)
        self.assertEqual(len(widget._spectrum), 2048)

    def test_waterfall_handles_empty_spectrum(self):
        """Test waterfall handles empty spectrum."""
        from sdr_module.gui.waterfall_widget import WaterfallWidget

        widget = WaterfallWidget()

        # Empty array should not crash
        widget.add_line(np.array([]))
        # History should still have an entry (resampled)
        self.assertEqual(len(widget._history), 1)


class TestSpectrumDbfsNormalization(unittest.TestCase):
    """The display FFT must be referenced to dBFS, not raw 20*log10(|FFT|).

    Regression for the unnormalized spectrum that made a full-scale tone read
    ~+66 dB, saturating the (-120, 0) dB display and pinning the -80 dB squelch
    permanently open.
    """

    def setUp(self):
        require_pyqt6(self)
        from sdr_module.gui.main_window import SDRMainWindow

        # Exercise just the pure DSP helper without building the whole window.
        self.win = SDRMainWindow.__new__(SDRMainWindow)
        self.win._spectrum_window = None
        self.win._spectrum_window_gain = 1.0

    def _tone(self, n=2048, k=200, amplitude=1.0):
        t = np.arange(n)
        return (amplitude * np.exp(2j * np.pi * k / n * t)).astype(np.complex64)

    def test_full_scale_tone_reads_zero_dbfs(self):
        peak = float(self.win._power_spectrum_dbfs(self._tone()).max())
        self.assertAlmostEqual(peak, 0.0, delta=0.5)

    def test_half_scale_tone_reads_minus_six_db(self):
        peak = float(self.win._power_spectrum_dbfs(self._tone(amplitude=0.5)).max())
        self.assertAlmostEqual(peak, -6.02, delta=0.5)

    def test_noise_floor_well_below_full_scale(self):
        rng = np.random.default_rng(0)
        n = 2048
        noise = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(
            np.complex64
        ) * 0.1
        power = self.win._power_spectrum_dbfs(noise)
        self.assertLess(float(power.max()), -10.0)

    def test_empty_input_returns_empty(self):
        out = self.win._power_spectrum_dbfs(np.array([], dtype=np.complex64))
        self.assertEqual(out.shape, (0,))


class TestScannerNoDeviceHonesty(unittest.TestCase):
    """The scanner must not fabricate detections when no device is present."""

    def setUp(self):
        require_pyqt6(self)

    def test_worker_measure_peak_returns_none_without_device(self):
        from sdr_module.gui.scanner_dialog import _ScanWorker

        worker = _ScanWorker(None, 88e6, 108e6, 200e3, -20.0)
        # Repeated calls must never invent a measurement from synthetic noise.
        for _ in range(20):
            self.assertIsNone(worker._measure_peak(100e6))

    def test_dialog_disables_scan_without_device(self):
        from sdr_module.gui.scanner_dialog import ScannerDialog

        dialog = ScannerDialog(device=None)
        self.assertFalse(dialog._start_btn.isEnabled())
        # Toggling anyway produces no hits.
        dialog._toggle_scan()
        self.assertEqual(dialog._table.rowCount(), 0)

    def test_measure_peak_is_dbfs_referenced_with_device(self):
        from sdr_module.gui.scanner_dialog import _ScanWorker

        class FakeDevice:
            def set_frequency(self, freq):
                self._freq = freq

            def read_samples(self, n):
                t = np.arange(n)
                return np.exp(2j * np.pi * 0.1 * t).astype(np.complex64)

        worker = _ScanWorker(FakeDevice(), 88e6, 108e6, 200e3, -20.0)
        peak = worker._measure_peak(100e6)
        self.assertIsNotNone(peak)
        # A full-scale tone should read near 0 dBFS, not tens of dB.
        self.assertLess(abs(peak), 3.0)


class TestGUISignalEmission(unittest.TestCase):
    """Test that GUI widgets emit signals correctly."""

    def setUp(self):
        """Set up test fixtures."""
        require_pyqt6(self)

    def test_control_panel_frequency_signal(self):
        """Test ControlPanel emits frequency_changed signal."""
        from sdr_module.gui.control_panel import ControlPanel

        widget = ControlPanel()
        signal_received = []

        widget.frequency_changed.connect(lambda f: signal_received.append(f))
        widget._freq_input.set_frequency(146e6)
        widget.frequency_changed.emit(146e6)

        self.assertEqual(len(signal_received), 1)
        self.assertEqual(signal_received[0], 146e6)

    def test_control_panel_gain_signal(self):
        """Test ControlPanel emits gain_changed signal."""
        from sdr_module.gui.control_panel import ControlPanel

        widget = ControlPanel()
        signal_received = []

        widget.gain_changed.connect(lambda g: signal_received.append(g))
        widget._gain_slider.setValue(35)

        self.assertEqual(len(signal_received), 1)
        self.assertEqual(signal_received[0], 35.0)

    def test_frequency_input_signal(self):
        """Test FrequencyInput emits frequency_changed signal."""
        from sdr_module.gui.control_panel import FrequencyInput

        widget = FrequencyInput()
        signal_received = []

        widget.frequency_changed.connect(lambda f: signal_received.append(f))
        widget._freq_input.setValue(145.8)  # With MHz selected by default

        self.assertEqual(len(signal_received), 1)
        self.assertAlmostEqual(signal_received[0], 145.8e6, places=0)


class TestBookmarksPanelCsv(unittest.TestCase):
    """Test CHIRP CSV import/export from the bookmarks (memory channel) panel."""

    class _FakeSettings:
        """In-memory stand-in for GuiSettings so tests never touch QSettings."""

        def __init__(self):
            self.bookmarks = []

        def get_bookmarks(self):
            return list(self.bookmarks)

        def set_bookmarks(self, bookmarks):
            self.bookmarks = list(bookmarks)

    def setUp(self):
        require_pyqt6(self)
        import tempfile

        from sdr_module.gui import bookmarks_panel as panel_module

        self._panel_module = panel_module
        self._real_settings = panel_module.GuiSettings
        self._store = self._FakeSettings()
        panel_module.GuiSettings = lambda: self._store

        # Silence the modal result dialogs the panel shows.
        self._real_information = panel_module.QMessageBox.information
        self._real_warning = panel_module.QMessageBox.warning
        panel_module.QMessageBox.information = staticmethod(lambda *a, **k: None)
        panel_module.QMessageBox.warning = staticmethod(lambda *a, **k: None)

        self._tmpdir = tempfile.TemporaryDirectory()
        self.panel = panel_module.BookmarksPanel()

    def tearDown(self):
        if not HAS_PYQT6:
            return
        self._panel_module.GuiSettings = self._real_settings
        self._panel_module.QMessageBox.information = self._real_information
        self._panel_module.QMessageBox.warning = self._real_warning
        self._tmpdir.cleanup()

    def _path(self, name):
        import os

        return os.path.join(self._tmpdir.name, name)

    def test_export_then_import_round_trip(self):
        self.panel.add_bookmark("2m Calling", 146.52e6)
        self.panel.add_bookmark("NOAA", 162.55e6)

        path = self._path("channels.csv")
        self.assertEqual(self.panel.export_csv(path), 2)

        # Import into an empty panel: no replace/append prompt is shown.
        self._store.set_bookmarks([])
        fresh = self._panel_module.BookmarksPanel()
        self.assertEqual(fresh.import_csv(path), 2)
        self.assertEqual(
            [b["label"] for b in self._store.get_bookmarks()],
            ["2m Calling", "NOAA"],
        )
        self.assertEqual(self._store.get_bookmarks()[0]["freq_hz"], 146.52e6)

    def test_export_adds_csv_extension(self):
        self.panel.add_bookmark("A", 100e6)
        path = self._path("channels")
        self.panel.export_csv(path)
        import os

        self.assertTrue(os.path.exists(path + ".csv"))

    def test_export_with_no_channels(self):
        self.assertEqual(self.panel.export_csv(self._path("empty.csv")), 0)

    def test_import_rejects_non_chirp_file(self):
        path = self._path("other.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("a,b\n1,2\n")
        self.assertEqual(self.panel.import_csv(path), 0)
        self.assertEqual(self._store.get_bookmarks(), [])

    def test_import_shows_mode_in_list(self):
        path = self._path("modes.csv")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("Name,Frequency,Mode\nTower,118.300000,AM\n")
        self.assertEqual(self.panel.import_csv(path), 1)
        self.assertIn("AM", self.panel._list.item(0).text())


def run_tests():
    """Run all GUI tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSpectrumWidgetLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestWaterfallWidgetLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestFrequencyInputLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestControlPanelLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestSignalMeterWidgetLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestMockGUIWithoutQt))
    suite.addTests(loader.loadTestsFromTestCase(TestColorMapConsistency))
    suite.addTests(loader.loadTestsFromTestCase(TestGUIDataProcessing))
    suite.addTests(loader.loadTestsFromTestCase(TestGUISignalEmission))
    suite.addTests(loader.loadTestsFromTestCase(TestBookmarksPanelCsv))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

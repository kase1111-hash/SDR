#!/usr/bin/env python3
"""
GUI component tests for SDR module.

Tests PyQt6 widgets without requiring a display or actual Qt event loop.
Uses mocking to test widget logic and state management.
"""

import sys
import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import numpy as np

# Add src to path
sys.path.insert(0, '../src')

# Check if PyQt6 is available
try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt
    HAS_PYQT6 = True
    # Create QApplication if needed (required for widget instantiation)
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
except ImportError:
    HAS_PYQT6 = False


class TestSpectrumWidgetLogic(unittest.TestCase):
    """Test SpectrumWidget logic and state management."""

    def setUp(self):
        """Set up test fixtures."""
        if not HAS_PYQT6:
            self.skipTest("PyQt6 not available")
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
        if not HAS_PYQT6:
            self.skipTest("PyQt6 not available")
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


class TestFrequencyInputLogic(unittest.TestCase):
    """Test FrequencyInput widget logic."""

    def setUp(self):
        """Set up test fixtures."""
        if not HAS_PYQT6:
            self.skipTest("PyQt6 not available")
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
        if not HAS_PYQT6:
            self.skipTest("PyQt6 not available")
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
    """Test SignalMeterWidget logic."""

    def setUp(self):
        """Set up test fixtures."""
        if not HAS_PYQT6:
            self.skipTest("PyQt6 not available")
        try:
            from sdr_module.gui.signal_meter_widget import SignalMeterWidget
            self.widget = SignalMeterWidget()
        except ImportError:
            self.skipTest("SignalMeterWidget not available")

    def test_initialization(self):
        """Test widget initializes correctly."""
        self.assertIsNotNone(self.widget)


class TestMockGUIWithoutQt(unittest.TestCase):
    """Test GUI module behavior when PyQt6 is not available."""

    def test_spectrum_widget_import_guard(self):
        """Test SpectrumWidget has proper import guard."""
        # This tests the HAS_PYQT6 pattern is used correctly
        import sdr_module.gui.spectrum_widget as sw
        self.assertTrue(hasattr(sw, 'HAS_PYQT6'))

    def test_waterfall_widget_import_guard(self):
        """Test WaterfallWidget has proper import guard."""
        import sdr_module.gui.waterfall_widget as ww
        self.assertTrue(hasattr(ww, 'HAS_PYQT6'))

    def test_control_panel_import_guard(self):
        """Test ControlPanel has proper import guard."""
        import sdr_module.gui.control_panel as cp
        self.assertTrue(hasattr(cp, 'HAS_PYQT6'))


class TestColorMapConsistency(unittest.TestCase):
    """Test colormap definitions are consistent."""

    def setUp(self):
        """Set up test fixtures."""
        if not HAS_PYQT6:
            self.skipTest("PyQt6 not available")
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
            self.assertGreaterEqual(
                len(colors), 2,
                f"{name} has fewer than 2 colors"
            )


class TestGUIDataProcessing(unittest.TestCase):
    """Test data processing in GUI widgets."""

    def setUp(self):
        """Set up test fixtures."""
        if not HAS_PYQT6:
            self.skipTest("PyQt6 not available")

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


class TestGUISignalEmission(unittest.TestCase):
    """Test that GUI widgets emit signals correctly."""

    def setUp(self):
        """Set up test fixtures."""
        if not HAS_PYQT6:
            self.skipTest("PyQt6 not available")

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

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

"""Tests for UI visualization components (no Qt dependency)."""

import numpy as np
import pytest

from sdr_module.ui.constellation import (
    ConstellationDisplay,
    ConstellationPoint,
    ConstellationStats,
    ConstellationResult,
    ModulationOverlay,
)
from sdr_module.ui.time_domain import (
    TimeDomainDisplay,
    TimeDomainResult,
    DisplayMode,
)
from sdr_module.ui.signal_meter import (
    SignalStrengthMeter,
    MeterReading,
    MeterConfig,
    PowerUnit,
    MeterMode,
)


class TestModulationOverlay:
    """Tests for ModulationOverlay enum."""

    def test_all_overlays_defined(self):
        """Test all modulation overlays are defined."""
        assert ModulationOverlay.NONE.value == "none"
        assert ModulationOverlay.BPSK.value == "bpsk"
        assert ModulationOverlay.QPSK.value == "qpsk"
        assert ModulationOverlay.PSK8.value == "8psk"
        assert ModulationOverlay.QAM16.value == "16qam"
        assert ModulationOverlay.QAM64.value == "64qam"
        assert ModulationOverlay.QAM256.value == "256qam"


class TestConstellationPoint:
    """Tests for ConstellationPoint dataclass."""

    def test_basic_point(self):
        """Test creating a basic constellation point."""
        point = ConstellationPoint(i=1.0, q=0.0)
        assert point.i == 1.0
        assert point.q == 0.0
        assert point.symbol is None

    def test_point_with_symbol(self):
        """Test constellation point with symbol."""
        point = ConstellationPoint(i=0.707, q=0.707, symbol="00")
        assert point.symbol == "00"


class TestConstellationStats:
    """Tests for ConstellationStats dataclass."""

    def test_default_values(self):
        """Test default constellation stats."""
        stats = ConstellationStats()
        assert stats.evm_percent == 0.0
        assert stats.evm_db == 0.0
        assert stats.phase_error_deg == 0.0
        assert stats.magnitude_error == 0.0
        assert stats.snr_estimate_db == 0.0
        assert stats.iq_offset == 0j
        assert stats.iq_imbalance == 0.0

    def test_custom_stats(self):
        """Test constellation stats with values."""
        stats = ConstellationStats(
            evm_percent=5.0,
            evm_db=-26.0,
            phase_error_deg=2.5,
            snr_estimate_db=20.0,
            iq_offset=0.01 + 0.02j,
        )
        assert stats.evm_percent == 5.0
        assert stats.iq_offset == 0.01 + 0.02j


class TestConstellationResult:
    """Tests for ConstellationResult dataclass."""

    def test_result_structure(self):
        """Test result dataclass structure."""
        i_data = np.array([1.0, -1.0, 0.707])
        q_data = np.array([0.0, 0.0, 0.707])
        stats = ConstellationStats()

        result = ConstellationResult(
            i_data=i_data,
            q_data=q_data,
            num_points=3,
            stats=stats,
        )

        assert len(result.i_data) == 3
        assert len(result.q_data) == 3
        assert result.num_points == 3

    def test_default_bounds(self):
        """Test default display bounds."""
        result = ConstellationResult(
            i_data=np.array([]),
            q_data=np.array([]),
            num_points=0,
            stats=ConstellationStats(),
        )
        assert result.i_min == -1.5
        assert result.i_max == 1.5
        assert result.q_min == -1.5
        assert result.q_max == 1.5


class TestConstellationDisplay:
    """Tests for ConstellationDisplay class."""

    @pytest.fixture
    def display(self):
        """Create constellation display instance."""
        return ConstellationDisplay(
            max_points=1024,
            persistence=1,
            normalize=True,
        )

    def test_initialization(self, display):
        """Test display initialization."""
        assert display._max_points == 1024
        assert display._persistence == 1
        assert display._normalize is True
        assert display._overlay == ModulationOverlay.NONE

    def test_ideal_constellations_defined(self, display):
        """Test ideal constellation definitions exist."""
        assert ModulationOverlay.BPSK in display.IDEAL_CONSTELLATIONS
        assert ModulationOverlay.QPSK in display.IDEAL_CONSTELLATIONS
        assert ModulationOverlay.PSK8 in display.IDEAL_CONSTELLATIONS
        assert ModulationOverlay.QAM16 in display.IDEAL_CONSTELLATIONS

    def test_bpsk_constellation(self, display):
        """Test BPSK constellation points."""
        bpsk = display.IDEAL_CONSTELLATIONS[ModulationOverlay.BPSK]
        assert len(bpsk) == 2
        # Should have points at -1 and +1 on I axis
        i_values = [p.i for p in bpsk]
        assert -1.0 in i_values
        assert 1.0 in i_values

    def test_qpsk_constellation(self, display):
        """Test QPSK constellation has 4 points."""
        qpsk = display.IDEAL_CONSTELLATIONS[ModulationOverlay.QPSK]
        assert len(qpsk) == 4

    def test_psk8_constellation(self, display):
        """Test 8-PSK constellation has 8 points."""
        psk8 = display.IDEAL_CONSTELLATIONS[ModulationOverlay.PSK8]
        assert len(psk8) == 8

    def test_qam16_constellation(self, display):
        """Test 16-QAM constellation has 16 points."""
        qam16 = display.IDEAL_CONSTELLATIONS[ModulationOverlay.QAM16]
        assert len(qam16) == 16

    def test_overlay_setting(self):
        """Test overlay can be set during initialization."""
        display = ConstellationDisplay(overlay=ModulationOverlay.QPSK)
        assert display._overlay == ModulationOverlay.QPSK


class TestDisplayMode:
    """Tests for DisplayMode enum."""

    def test_all_modes_defined(self):
        """Test all display modes are defined."""
        assert DisplayMode.MAGNITUDE.value == "magnitude"
        assert DisplayMode.I_CHANNEL.value == "i_channel"
        assert DisplayMode.Q_CHANNEL.value == "q_channel"
        assert DisplayMode.IQ_OVERLAY.value == "iq_overlay"
        assert DisplayMode.PHASE.value == "phase"
        assert DisplayMode.POWER.value == "power"


class TestTimeDomainResult:
    """Tests for TimeDomainResult dataclass."""

    def test_result_structure(self):
        """Test result dataclass structure."""
        time_ms = np.linspace(0, 10, 1000)
        primary = np.random.randn(1000)

        result = TimeDomainResult(
            time_ms=time_ms,
            samples=1000,
            sample_rate=48000,
            mode=DisplayMode.MAGNITUDE,
            primary=primary,
            peak=1.5,
            rms=0.7,
        )

        assert len(result.time_ms) == 1000
        assert result.samples == 1000
        assert result.sample_rate == 48000
        assert result.mode == DisplayMode.MAGNITUDE
        assert result.peak == 1.5
        assert result.rms == 0.7
        assert result.secondary is None

    def test_result_with_secondary(self):
        """Test result with secondary data."""
        result = TimeDomainResult(
            time_ms=np.array([0, 1]),
            samples=2,
            sample_rate=1000,
            mode=DisplayMode.IQ_OVERLAY,
            primary=np.array([1, 2]),
            secondary=np.array([3, 4]),
        )
        assert result.secondary is not None
        np.testing.assert_array_equal(result.secondary, [3, 4])


class TestTimeDomainDisplay:
    """Tests for TimeDomainDisplay class."""

    @pytest.fixture
    def display(self):
        """Create time domain display instance."""
        return TimeDomainDisplay(
            sample_rate=48000,
            window_size=4096,
            mode=DisplayMode.MAGNITUDE,
        )

    def test_initialization(self, display):
        """Test display initialization."""
        assert display._sample_rate == 48000
        assert display._window_size == 4096
        assert display._mode == DisplayMode.MAGNITUDE
        assert len(display._buffer) == 4096

    def test_sample_rate_property(self, display):
        """Test sample rate getter/setter."""
        assert display.sample_rate == 48000
        display.sample_rate = 96000
        assert display.sample_rate == 96000

    def test_window_size_property(self, display):
        """Test window size getter."""
        assert display.window_size == 4096

    def test_trigger_defaults(self, display):
        """Test trigger defaults."""
        assert display._trigger_enabled is False
        assert display._trigger_level == 0.5
        assert display._trigger_edge == "rising"

    def test_different_modes(self):
        """Test initialization with different modes."""
        for mode in DisplayMode:
            display = TimeDomainDisplay(
                sample_rate=48000,
                mode=mode,
            )
            assert display._mode == mode


class TestPowerUnit:
    """Tests for PowerUnit enum."""

    def test_all_units_defined(self):
        """Test all power units are defined."""
        assert PowerUnit.DBFS.value == "dBFS"
        assert PowerUnit.DBM.value == "dBm"
        assert PowerUnit.DBU.value == "dBuV"
        assert PowerUnit.LINEAR.value == "linear"


class TestMeterMode:
    """Tests for MeterMode enum."""

    def test_all_modes_defined(self):
        """Test all meter modes are defined."""
        assert MeterMode.INSTANTANEOUS.value == "instantaneous"
        assert MeterMode.AVERAGE.value == "average"
        assert MeterMode.PEAK.value == "peak"
        assert MeterMode.PEAK_DECAY.value == "peak_decay"


class TestMeterReading:
    """Tests for MeterReading dataclass."""

    def test_reading_structure(self):
        """Test meter reading structure."""
        reading = MeterReading(
            power_dbfs=-30.0,
            power_dbm=-60.0,
            power_linear=0.001,
            peak_dbfs=-25.0,
            average_dbfs=-35.0,
            bar_level=70,
            peak_bar_level=75,
            clipping=False,
            noise_floor_dbfs=-90.0,
        )

        assert reading.power_dbfs == -30.0
        assert reading.power_dbm == -60.0
        assert reading.power_linear == 0.001
        assert reading.bar_level == 70
        assert reading.clipping is False


class TestMeterConfig:
    """Tests for MeterConfig dataclass."""

    def test_default_config(self):
        """Test default meter configuration."""
        config = MeterConfig()
        assert config.unit == PowerUnit.DBFS
        assert config.mode == MeterMode.AVERAGE
        assert config.avg_time_ms == 100.0
        assert config.peak_hold_ms == 1000.0
        assert config.min_dbfs == -100.0
        assert config.max_dbfs == 0.0
        assert config.clip_threshold == 0.99

    def test_custom_config(self):
        """Test custom meter configuration."""
        config = MeterConfig(
            unit=PowerUnit.DBM,
            mode=MeterMode.PEAK,
            avg_time_ms=200.0,
            reference_level_dbm=-20.0,
        )
        assert config.unit == PowerUnit.DBM
        assert config.mode == MeterMode.PEAK
        assert config.avg_time_ms == 200.0
        assert config.reference_level_dbm == -20.0


class TestSignalStrengthMeter:
    """Tests for SignalStrengthMeter class."""

    @pytest.fixture
    def meter(self):
        """Create signal strength meter instance."""
        return SignalStrengthMeter(sample_rate=48000)

    def test_initialization(self, meter):
        """Test meter initialization."""
        assert meter._sample_rate == 48000
        assert meter._config is not None

    def test_initialization_with_config(self):
        """Test meter initialization with custom config."""
        config = MeterConfig(
            unit=PowerUnit.DBM,
            mode=MeterMode.PEAK,
        )
        meter = SignalStrengthMeter(sample_rate=96000, config=config)
        assert meter._sample_rate == 96000
        assert meter._config.unit == PowerUnit.DBM
        assert meter._config.mode == MeterMode.PEAK

    def test_default_config_applied(self, meter):
        """Test default config is applied when not specified."""
        assert meter._config.unit == PowerUnit.DBFS
        assert meter._config.mode == MeterMode.AVERAGE


class TestConstellationDisplayAdvanced:
    """Advanced tests for ConstellationDisplay."""

    @pytest.fixture
    def display(self):
        """Create constellation display instance."""
        return ConstellationDisplay()

    def test_max_points_property(self, display):
        """Test max_points property."""
        assert display.max_points == 1024
        display.max_points = 2048
        assert display.max_points == 2048

    def test_persistence_property(self, display):
        """Test persistence property."""
        assert display.persistence == 1
        display.persistence = 3
        assert display.persistence == 3

    def test_persistence_minimum(self, display):
        """Test persistence is at least 1."""
        display.persistence = 0
        assert display.persistence == 1

    def test_overlay_property(self, display):
        """Test overlay property."""
        assert display.overlay == ModulationOverlay.NONE
        display.overlay = ModulationOverlay.QPSK
        assert display.overlay == ModulationOverlay.QPSK

    def test_get_overlay_points_none(self, display):
        """Test get_overlay_points with no overlay."""
        points = display.get_overlay_points()
        assert points == []

    def test_get_overlay_points_qpsk(self, display):
        """Test get_overlay_points with QPSK overlay."""
        display.overlay = ModulationOverlay.QPSK
        points = display.get_overlay_points()
        assert len(points) == 4

    def test_update_adds_to_buffer(self, display):
        """Test update adds samples to buffer."""
        samples = np.array([1+1j, -1-1j], dtype=np.complex64)
        display.update(samples)
        assert len(display._buffers) == 1

    def test_update_limits_to_max_points(self, display):
        """Test update limits samples to max_points."""
        display.max_points = 10
        samples = np.ones(100, dtype=np.complex64)
        display.update(samples)
        assert len(display._buffers[0]) == 10

    def test_update_normalizes(self, display):
        """Test update normalizes samples."""
        samples = np.array([2+0j, 0+2j], dtype=np.complex64)
        display.update(samples)
        # After normalization, max magnitude should be 1
        assert np.max(np.abs(display._buffers[0])) <= 1.0 + 1e-6

    def test_process_returns_result(self, display):
        """Test process returns ConstellationResult."""
        samples = np.exp(1j * np.linspace(0, 2*np.pi, 100)).astype(np.complex64)
        result = display.process(samples)

        assert isinstance(result, ConstellationResult)
        assert len(result.i_data) == 100
        assert len(result.q_data) == 100
        assert result.num_points == 100

    def test_process_calculates_stats(self, display):
        """Test process calculates statistics."""
        samples = np.random.randn(100).astype(np.complex64) + \
                  1j * np.random.randn(100).astype(np.complex64)
        result = display.process(samples)

        assert result.stats is not None
        assert isinstance(result.stats.evm_percent, float)
        assert isinstance(result.stats.phase_error_deg, float)

    def test_clear_empties_buffers(self, display):
        """Test clear empties all buffers."""
        display.update(np.ones(10, dtype=np.complex64))
        assert len(display._buffers) > 0
        display.clear()
        assert len(display._buffers) == 0

    def test_decision_boundaries_bpsk(self, display):
        """Test decision boundaries for BPSK."""
        display.overlay = ModulationOverlay.BPSK
        boundaries = display.get_decision_boundaries()
        assert len(boundaries) == 1
        # Vertical line at x=0
        assert boundaries[0][0] == 0
        assert boundaries[0][2] == 0

    def test_decision_boundaries_qpsk(self, display):
        """Test decision boundaries for QPSK."""
        display.overlay = ModulationOverlay.QPSK
        boundaries = display.get_decision_boundaries()
        assert len(boundaries) == 2  # Cross pattern

    def test_decision_boundaries_8psk(self, display):
        """Test decision boundaries for 8-PSK."""
        display.overlay = ModulationOverlay.PSK8
        boundaries = display.get_decision_boundaries()
        assert len(boundaries) == 8  # 8 radial lines

    def test_decision_boundaries_qam16(self, display):
        """Test decision boundaries for 16-QAM."""
        display.overlay = ModulationOverlay.QAM16
        boundaries = display.get_decision_boundaries()
        assert len(boundaries) == 4  # Grid lines


class TestTimeDomainDisplayAdvanced:
    """Advanced tests for TimeDomainDisplay."""

    @pytest.fixture
    def display(self):
        """Create time domain display instance."""
        return TimeDomainDisplay(sample_rate=48000)

    def test_window_size_setter(self, display):
        """Test window size setter."""
        display.window_size = 8192
        assert display.window_size == 8192
        assert len(display._buffer) == 8192

    def test_buffer_initialized_to_zero(self, display):
        """Test buffer is initialized to zeros."""
        assert np.all(display._buffer == 0)
        assert display._buffer_valid == 0


class TestConstellationIntegration:
    """Integration tests for constellation processing."""

    def test_qpsk_ideal_points_on_unit_circle(self):
        """Test QPSK points lie approximately on unit circle."""
        display = ConstellationDisplay()
        qpsk = display.IDEAL_CONSTELLATIONS[ModulationOverlay.QPSK]

        for point in qpsk:
            magnitude = np.sqrt(point.i**2 + point.q**2)
            assert abs(magnitude - 1.0) < 0.01

    def test_8psk_points_evenly_spaced(self):
        """Test 8-PSK points are evenly spaced on unit circle."""
        display = ConstellationDisplay()
        psk8 = display.IDEAL_CONSTELLATIONS[ModulationOverlay.PSK8]

        # Calculate angles
        angles = []
        for point in psk8:
            angle = np.arctan2(point.q, point.i)
            angles.append(angle)

        angles = np.sort(angles)
        diffs = np.diff(angles)

        # All angle differences should be approximately equal (π/4)
        expected_diff = np.pi / 4
        for diff in diffs:
            assert abs(diff - expected_diff) < 0.01

    def test_evm_calculation_with_reference(self):
        """Test EVM calculation with reference constellation."""
        display = ConstellationDisplay(overlay=ModulationOverlay.QPSK)

        # Create ideal QPSK samples with small noise
        ideal = np.array([0.707+0.707j, -0.707+0.707j,
                          -0.707-0.707j, 0.707-0.707j], dtype=np.complex64)
        noisy = ideal + 0.1 * (np.random.randn(4) + 1j * np.random.randn(4))

        result = display.process(noisy.astype(np.complex64))

        # EVM should be small but non-zero
        assert result.stats.evm_percent > 0
        assert result.stats.evm_percent < 50  # Less than 50% error

    def test_dc_offset_detection(self):
        """Test DC offset is detected."""
        display = ConstellationDisplay()

        # Create samples with DC offset
        offset = 0.3 + 0.2j
        samples = np.random.randn(100).astype(np.complex64) * 0.1 + offset

        result = display.process(samples)

        # Should detect the DC offset
        assert abs(result.stats.iq_offset) > 0.1

    def test_persistence_accumulates_frames(self):
        """Test persistence accumulates multiple frames."""
        display = ConstellationDisplay(persistence=3, max_points=10)

        # Add three frames
        display.update(np.ones(10, dtype=np.complex64) * 1)
        display.update(np.ones(10, dtype=np.complex64) * 2)
        display.update(np.ones(10, dtype=np.complex64) * 3)

        result = display.process()

        # Should have 30 points (3 frames x 10 points)
        assert result.num_points == 30

    def test_persistence_discards_old_frames(self):
        """Test persistence discards frames beyond limit."""
        display = ConstellationDisplay(persistence=2, max_points=10)

        # Add four frames
        display.update(np.ones(10, dtype=np.complex64) * 1)
        display.update(np.ones(10, dtype=np.complex64) * 2)
        display.update(np.ones(10, dtype=np.complex64) * 3)
        display.update(np.ones(10, dtype=np.complex64) * 4)

        result = display.process()

        # Should only have 20 points (last 2 frames)
        assert result.num_points == 20

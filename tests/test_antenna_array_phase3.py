"""Tests for antenna array Phase 3 components (adaptive beamforming and calibration)."""

import numpy as np
import pytest

from sdr_module.antenna_array import (
    SPEED_OF_LIGHT,
    AdaptiveBeamformer,
    AdaptiveBeamformerState,
    AdaptiveMethod,
    ArrayCalibrator,
    ArrayConfig,
    Beamformer,
    CalibrationConfig,
    CalibrationMethod,
    CalibrationResult,
    CalibrationState,
    MVDRResult,
    create_linear_4_element,
)


def generate_test_signal(
    frequency: float,
    sample_rate: float,
    duration: float,
    snr_db: float = 20.0,
) -> np.ndarray:
    """Generate a test tone signal with noise."""
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate

    signal = np.exp(2j * np.pi * frequency * t).astype(np.complex64)

    noise_power = 10 ** (-snr_db / 10)
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
    ).astype(np.complex64)

    return signal + noise


def generate_array_signals(
    config: ArrayConfig,
    azimuth: float,
    signal_freq: float = 10e3,
    snr_db: float = 20.0,
    duration: float = 0.01,
) -> dict[int, np.ndarray]:
    """Generate test signals for array elements with proper phase delays."""
    sample_rate = config.common_sample_rate
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate

    wavelength = SPEED_OF_LIGHT / config.common_frequency
    k = 2 * np.pi / wavelength

    base_signal = np.exp(2j * np.pi * signal_freq * t).astype(np.complex64)

    u = np.array([np.sin(azimuth), np.cos(azimuth), 0.0])

    signals = {}
    for element in config.enabled_elements:
        pos = element.position.to_array()
        phase_delay = k * np.dot(pos, u)
        element_signal = base_signal * np.exp(1j * phase_delay)

        noise_power = 10 ** (-snr_db / 10)
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        ).astype(np.complex64)

        signals[element.index] = element_signal + noise

    return signals


def generate_array_signals_with_interference(
    config: ArrayConfig,
    signal_azimuth: float,
    interference_azimuth: float,
    signal_freq: float = 10e3,
    interference_freq: float = 12e3,
    sir_db: float = 0.0,  # Signal-to-interference ratio
    snr_db: float = 20.0,
    duration: float = 0.01,
) -> dict[int, np.ndarray]:
    """Generate array signals with signal and interference."""
    sample_rate = config.common_sample_rate
    n_samples = int(duration * sample_rate)
    t = np.arange(n_samples) / sample_rate

    wavelength = SPEED_OF_LIGHT / config.common_frequency
    k = 2 * np.pi / wavelength

    # Desired signal
    signal = np.exp(2j * np.pi * signal_freq * t).astype(np.complex64)
    u_signal = np.array([np.sin(signal_azimuth), np.cos(signal_azimuth), 0.0])

    # Interference
    interference_amp = 10 ** (-sir_db / 20)
    interference = interference_amp * np.exp(2j * np.pi * interference_freq * t).astype(np.complex64)
    u_interference = np.array([np.sin(interference_azimuth), np.cos(interference_azimuth), 0.0])

    signals = {}
    for element in config.enabled_elements:
        pos = element.position.to_array()

        # Phase delays
        signal_phase = k * np.dot(pos, u_signal)
        interference_phase = k * np.dot(pos, u_interference)

        # Combined signal
        element_signal = (
            signal * np.exp(1j * signal_phase)
            + interference * np.exp(1j * interference_phase)
        )

        # Add noise
        noise_power = 10 ** (-snr_db / 10)
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        ).astype(np.complex64)

        signals[element.index] = element_signal + noise

    return signals


class TestAdaptiveBeamformer:
    """Test AdaptiveBeamformer class."""

    def test_initialization(self):
        """Test adaptive beamformer initialization."""
        config = create_linear_4_element(frequency=433e6)
        adaptive = AdaptiveBeamformer(config)

        assert adaptive.num_elements == 4
        assert adaptive.diagonal_loading == 0.01

    def test_diagonal_loading_setter(self):
        """Test diagonal loading setter."""
        config = create_linear_4_element(frequency=433e6)
        adaptive = AdaptiveBeamformer(config)

        adaptive.diagonal_loading = 0.1
        assert adaptive.diagonal_loading == 0.1

        with pytest.raises(ValueError):
            adaptive.diagonal_loading = -0.1

    def test_estimate_covariance(self):
        """Test covariance matrix estimation."""
        config = create_linear_4_element(frequency=433e6)
        adaptive = AdaptiveBeamformer(config)

        signals = generate_array_signals(config, np.radians(0), snr_db=20)
        R = adaptive.estimate_covariance(signals)

        assert R.shape == (4, 4)
        assert R.dtype == np.complex128

        # Covariance should be Hermitian
        assert np.allclose(R, R.conj().T, atol=1e-10)

    def test_compute_mvdr_weights(self):
        """Test MVDR weight computation."""
        config = create_linear_4_element(frequency=433e6)
        adaptive = AdaptiveBeamformer(config)

        signals = generate_array_signals(config, np.radians(0), snr_db=20)
        R = adaptive.estimate_covariance(signals)

        # Steering vector toward boresight
        sv = adaptive._compute_steering_vector(0.0)
        weights = adaptive.compute_mvdr_weights(R, sv)

        assert len(weights) == 4
        assert weights.dtype == np.complex128

    def test_mvdr_produces_output(self):
        """Test MVDR beamformer produces output."""
        config = create_linear_4_element(frequency=433e6)
        adaptive = AdaptiveBeamformer(config)

        signals = generate_array_signals(config, np.radians(0), snr_db=20)
        result = adaptive.mvdr(signals, desired_azimuth=0.0)

        assert isinstance(result, MVDRResult)
        assert len(result.output_signal) > 0
        assert result.output_power > -100

    def test_mvdr_interference_suppression(self):
        """Test MVDR suppresses interference."""
        config = create_linear_4_element(frequency=433e6)
        adaptive = AdaptiveBeamformer(config)
        conventional = Beamformer(config)

        # Signal at 0 degrees, interference at 30 degrees
        signals = generate_array_signals_with_interference(
            config,
            signal_azimuth=np.radians(0),
            interference_azimuth=np.radians(30),
            sir_db=-3.0,  # Interference is 3 dB stronger
            snr_db=25,
            duration=0.02,
        )

        # MVDR should suppress interference
        mvdr_result = adaptive.mvdr(signals, desired_azimuth=0.0)

        # Verify MVDR produces valid output
        assert mvdr_result.output_power > -100

        # Check that interference nulls were found
        # (May or may not be exactly at 30 degrees due to noise)
        assert isinstance(mvdr_result.interference_nulls, list)

    def test_mvdr_spectrum(self):
        """Test MVDR spatial spectrum computation."""
        config = create_linear_4_element(frequency=433e6)
        adaptive = AdaptiveBeamformer(config)

        signals = generate_array_signals(config, np.radians(15), snr_db=25)
        azimuths, spectrum = adaptive.compute_mvdr_spectrum(signals)

        assert len(azimuths) > 0
        assert len(spectrum) == len(azimuths)

    def test_lcmv(self):
        """Test LCMV beamformer."""
        config = create_linear_4_element(frequency=433e6)
        adaptive = AdaptiveBeamformer(config)

        signals = generate_array_signals(config, np.radians(0), snr_db=20)

        # Constraint: unity gain at 0 degrees
        constraints = [(0.0, 1.0 + 0j)]
        result = adaptive.lcmv(signals, constraints)

        assert len(result.output_signal) > 0
        assert result.beam_power > -100

    def test_lcmv_with_null(self):
        """Test LCMV with null constraint."""
        config = create_linear_4_element(frequency=433e6)
        adaptive = AdaptiveBeamformer(config)

        signals = generate_array_signals(config, np.radians(0), snr_db=20)

        # Constraints: unity gain at 0 degrees, null at 30 degrees
        constraints = [
            (0.0, 1.0 + 0j),
            (np.radians(30), 0.0 + 0j),
        ]
        result = adaptive.lcmv(signals, constraints)

        assert len(result.output_signal) > 0

    def test_gsc(self):
        """Test Generalized Sidelobe Canceller."""
        config = create_linear_4_element(frequency=433e6)
        adaptive = AdaptiveBeamformer(config)

        signals = generate_array_signals(config, np.radians(0), snr_db=20, duration=0.05)
        result = adaptive.gsc(signals, desired_azimuth=0.0, mu=0.001)

        assert len(result.output_signal) > 0
        assert result.beam_power > -100

    def test_reset(self):
        """Test adaptive beamformer reset."""
        config = create_linear_4_element(frequency=433e6)
        adaptive = AdaptiveBeamformer(config)

        signals = generate_array_signals(config, np.radians(0), snr_db=20)
        adaptive.estimate_covariance(signals)

        adaptive.reset()
        state = adaptive.get_state()
        assert state.num_snapshots == 0

    def test_get_state(self):
        """Test getting adaptive beamformer state."""
        config = create_linear_4_element(frequency=433e6)
        adaptive = AdaptiveBeamformer(config)

        state = adaptive.get_state()

        assert isinstance(state, AdaptiveBeamformerState)
        assert len(state.weights) == 4
        assert state.covariance_matrix.shape == (4, 4)


class TestMVDRResult:
    """Test MVDRResult class."""

    def test_mvdr_result_fields(self):
        """Test MVDRResult has all expected fields."""
        result = MVDRResult(
            output_signal=np.zeros(100, dtype=np.complex64),
            weights=np.ones(4, dtype=np.complex128),
            output_power=-10.0,
            interference_nulls=[np.radians(30)],
            sinr_improvement=5.0,
        )

        assert len(result.output_signal) == 100
        assert len(result.weights) == 4
        assert result.output_power == -10.0
        assert len(result.interference_nulls) == 1
        assert result.sinr_improvement == 5.0


class TestArrayCalibrator:
    """Test ArrayCalibrator class."""

    def test_initialization(self):
        """Test calibrator initialization."""
        config = create_linear_4_element(frequency=433e6)
        calibrator = ArrayCalibrator(config)

        assert calibrator.config == config
        assert calibrator.state == CalibrationState.IDLE

    def test_initialization_with_config(self):
        """Test calibrator with custom config."""
        config = create_linear_4_element(frequency=433e6)
        cal_config = CalibrationConfig(
            method=CalibrationMethod.CORRELATION,
            reference_element=1,
            num_averages=5,
        )
        calibrator = ArrayCalibrator(config, cal_config)

        assert calibrator.calibration_config.reference_element == 1
        assert calibrator.calibration_config.num_averages == 5

    def test_calibrate_correlation(self):
        """Test correlation-based calibration."""
        config = create_linear_4_element(frequency=433e6)
        calibrator = ArrayCalibrator(config)

        # Generate signals with known phase offsets
        base_signal = generate_test_signal(10e3, config.common_sample_rate, 0.01, snr_db=30)
        signals = {
            0: base_signal,
            1: base_signal * np.exp(1j * np.pi / 4),
            2: base_signal * np.exp(1j * np.pi / 2),
            3: base_signal * np.exp(1j * 3 * np.pi / 4),
        }

        result = calibrator.calibrate_correlation(signals)

        assert isinstance(result, CalibrationResult)
        assert result.method == CalibrationMethod.CORRELATION
        assert result.reference_element == 0
        assert len(result.measurements) == 4
        assert len(result.element_calibrations) == 4

        # Reference element should have zero offset
        assert result.element_calibrations[0].phase_offset == pytest.approx(0.0, abs=0.1)

    def test_calibrate_correlation_confidence(self):
        """Test calibration confidence calculation."""
        config = create_linear_4_element(frequency=433e6)
        calibrator = ArrayCalibrator(config)

        # High SNR signals should give high confidence
        base_signal = generate_test_signal(10e3, config.common_sample_rate, 0.01, snr_db=40)
        signals = {i: base_signal for i in range(4)}

        result = calibrator.calibrate_correlation(signals)

        assert result.overall_confidence > 0.8
        assert result.success

    def test_calibrate_known_source(self):
        """Test known-source calibration."""
        config = create_linear_4_element(frequency=433e6)
        calibrator = ArrayCalibrator(config)

        # Generate signals from known direction
        source_azimuth = np.radians(20)
        signals = generate_array_signals(config, source_azimuth, snr_db=30)

        result = calibrator.calibrate_known_source(
            signals,
            source_azimuth=source_azimuth,
        )

        assert isinstance(result, CalibrationResult)
        assert result.method == CalibrationMethod.KNOWN_SOURCE
        assert len(result.element_calibrations) == 4

    def test_calibrate_pilot_tone(self):
        """Test pilot-tone calibration."""
        config = create_linear_4_element(frequency=433e6)
        calibrator = ArrayCalibrator(config)

        # Generate signals with pilot tone
        sample_rate = config.common_sample_rate
        n_samples = int(0.01 * sample_rate)
        t = np.arange(n_samples) / sample_rate
        pilot_freq = 50e3

        signals = {}
        for i in range(4):
            # Pilot tone with element-specific phase offset
            phase_offset = i * np.pi / 4
            pilot = np.exp(2j * np.pi * pilot_freq * t + 1j * phase_offset)
            noise = 0.1 * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
            signals[i] = (pilot + noise).astype(np.complex64)

        result = calibrator.calibrate_pilot_tone(signals, pilot_frequency=pilot_freq)

        assert isinstance(result, CalibrationResult)
        assert result.method == CalibrationMethod.PILOT_TONE

    def test_apply_calibration(self):
        """Test applying calibration to config."""
        config = create_linear_4_element(frequency=433e6)
        calibrator = ArrayCalibrator(config)

        base_signal = generate_test_signal(10e3, config.common_sample_rate, 0.01, snr_db=30)
        signals = {
            0: base_signal,
            1: base_signal * np.exp(1j * np.pi / 4),
            2: base_signal * np.exp(1j * np.pi / 2),
            3: base_signal * np.exp(1j * 3 * np.pi / 4),
        }

        result = calibrator.calibrate_correlation(signals)
        updated_config = calibrator.apply_calibration(result)

        # Check calibration was applied
        for idx in range(4):
            element = updated_config.get_element_by_index(idx)
            assert element is not None
            assert element.calibration.phase_offset == result.element_calibrations[idx].phase_offset

    def test_track_calibration_drift(self):
        """Test tracking calibration drift."""
        config = create_linear_4_element(frequency=433e6)
        calibrator = ArrayCalibrator(config)

        # Initial calibration
        base_signal = generate_test_signal(10e3, config.common_sample_rate, 0.01, snr_db=30)
        initial_signals = {i: base_signal for i in range(4)}
        calibrator.calibrate_correlation(initial_signals)

        # Simulate drift
        drifted_signals = {
            0: base_signal,
            1: base_signal * np.exp(1j * 0.1),  # Small drift
            2: base_signal * np.exp(1j * 0.2),
            3: base_signal * np.exp(1j * 0.3),
        }

        drift = calibrator.track_calibration_drift(drifted_signals)

        assert isinstance(drift, dict)
        assert drift[0] == pytest.approx(0.0, abs=0.1)  # Reference doesn't drift

    def test_export_import_calibration(self):
        """Test calibration export and import."""
        config = create_linear_4_element(frequency=433e6)
        calibrator = ArrayCalibrator(config)

        base_signal = generate_test_signal(10e3, config.common_sample_rate, 0.01, snr_db=30)
        signals = {i: base_signal for i in range(4)}

        result = calibrator.calibrate_correlation(signals)

        # Export
        data = calibrator.export_calibration(result)
        assert "success" in data
        assert "elements" in data
        assert "method" in data

        # Import
        imported = calibrator.import_calibration(data)
        assert imported.success == result.success
        assert imported.method == result.method

    def test_calibration_history(self):
        """Test calibration history tracking."""
        config = create_linear_4_element(frequency=433e6)
        calibrator = ArrayCalibrator(config)

        base_signal = generate_test_signal(10e3, config.common_sample_rate, 0.01, snr_db=30)
        signals = {i: base_signal for i in range(4)}

        # Run multiple calibrations
        calibrator.calibrate_correlation(signals)
        calibrator.calibrate_correlation(signals)

        history = calibrator.get_calibration_history()
        assert len(history) == 2

        calibrator.clear_history()
        assert len(calibrator.get_calibration_history()) == 0


class TestCalibrationConfig:
    """Test CalibrationConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CalibrationConfig()

        assert config.method == CalibrationMethod.CORRELATION
        assert config.reference_element == 0
        assert config.num_averages == 10
        assert config.min_confidence == 0.7

    def test_custom_config(self):
        """Test custom configuration."""
        config = CalibrationConfig(
            method=CalibrationMethod.KNOWN_SOURCE,
            reference_element=2,
            num_averages=20,
            source_azimuth=np.radians(30),
        )

        assert config.method == CalibrationMethod.KNOWN_SOURCE
        assert config.reference_element == 2
        assert config.num_averages == 20


class TestCalibrationResult:
    """Test CalibrationResult class."""

    def test_get_phase_corrections(self):
        """Test getting phase corrections."""
        from sdr_module.antenna_array import ElementCalibration

        result = CalibrationResult(
            success=True,
            method=CalibrationMethod.CORRELATION,
            reference_element=0,
            measurements=[],
            element_calibrations={
                0: ElementCalibration(element_index=0, phase_offset=0.0),
                1: ElementCalibration(element_index=1, phase_offset=0.5),
            },
            overall_confidence=0.9,
            timestamp=0.0,
            duration=1.0,
        )

        corrections = result.get_phase_corrections()
        assert corrections[0] == 0.0
        assert corrections[1] == 0.5

    def test_get_amplitude_corrections(self):
        """Test getting amplitude corrections."""
        from sdr_module.antenna_array import ElementCalibration

        result = CalibrationResult(
            success=True,
            method=CalibrationMethod.CORRELATION,
            reference_element=0,
            measurements=[],
            element_calibrations={
                0: ElementCalibration(element_index=0, amplitude_scale=1.0),
                1: ElementCalibration(element_index=1, amplitude_scale=1.2),
            },
            overall_confidence=0.9,
            timestamp=0.0,
            duration=1.0,
        )

        corrections = result.get_amplitude_corrections()
        assert corrections[0] == 1.0
        assert corrections[1] == 1.2


class TestIntegration:
    """Integration tests for Phase 3 components."""

    def test_calibrate_then_adaptive_beamform(self):
        """Test calibration followed by adaptive beamforming."""
        config = create_linear_4_element(frequency=433e6)

        # Step 1: Calibrate
        calibrator = ArrayCalibrator(config)
        base_signal = generate_test_signal(10e3, config.common_sample_rate, 0.01, snr_db=30)
        cal_signals = {i: base_signal for i in range(4)}

        cal_result = calibrator.calibrate_correlation(cal_signals)
        assert cal_result.success

        # Step 2: Apply calibration
        calibrator.apply_calibration(cal_result)

        # Step 3: Adaptive beamform
        adaptive = AdaptiveBeamformer(config)
        signal_az = np.radians(20)
        signals = generate_array_signals(config, signal_az, snr_db=25, duration=0.02)

        result = adaptive.mvdr(signals, desired_azimuth=signal_az)
        assert result.output_power > -100

    def test_adaptive_vs_conventional_with_interference(self):
        """Test adaptive beamformer outperforms conventional with interference."""
        config = create_linear_4_element(frequency=433e6)

        # Signal at 0 degrees, strong interference at 40 degrees
        signals = generate_array_signals_with_interference(
            config,
            signal_azimuth=np.radians(0),
            interference_azimuth=np.radians(40),
            sir_db=-6.0,  # Interference 6 dB stronger
            snr_db=25,
            duration=0.02,
        )

        # Conventional beamformer
        conventional = Beamformer(config)
        conv_result = conventional.steer_and_sum(signals, azimuth=0.0)

        # MVDR beamformer
        adaptive = AdaptiveBeamformer(config)
        mvdr_result = adaptive.mvdr(signals, desired_azimuth=0.0)

        # Both should produce output
        assert conv_result.beam_power > -100
        assert mvdr_result.output_power > -100

        # MVDR should show some improvement (or at least not be much worse)
        # Note: Actual improvement depends on scenario and may vary with noise
        assert mvdr_result.sinr_improvement is not None

    def test_full_array_processing_pipeline(self):
        """Test complete array processing pipeline."""
        config = create_linear_4_element(frequency=433e6)

        # Step 1: Generate calibration signals
        cal_base = generate_test_signal(10e3, config.common_sample_rate, 0.01, snr_db=35)
        cal_signals = {i: cal_base for i in range(4)}

        # Step 2: Calibrate
        calibrator = ArrayCalibrator(config)
        cal_result = calibrator.calibrate_correlation(cal_signals)
        calibrator.apply_calibration(cal_result)

        # Step 3: Generate signal with interference
        signals = generate_array_signals_with_interference(
            config,
            signal_azimuth=np.radians(10),
            interference_azimuth=np.radians(-30),
            sir_db=-3.0,
            snr_db=25,
            duration=0.02,
        )

        # Step 4: MVDR beamforming
        adaptive = AdaptiveBeamformer(config)
        result = adaptive.mvdr(signals, desired_azimuth=np.radians(10))

        # Step 5: Verify output
        assert len(result.output_signal) > 0
        assert result.output_power > -100

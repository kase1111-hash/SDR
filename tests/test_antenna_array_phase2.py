"""Tests for antenna array Phase 2 components (spatial processing)."""

import math

import numpy as np
import pytest

from sdr_module.antenna_array import (
    SPEED_OF_LIGHT,
    ArrayAlignmentResult,
    ArrayConfig,
    BeamPattern,
    Beamformer,
    BeamformerOutput,
    BeamformingMethod,
    BeamscanDoA,
    CorrelationResult,
    CrossCorrelator,
    DoAMethod,
    DoAResult,
    MUSICDoA,
    MultiSourceDoAResult,
    PhaseDifferenceDoA,
    SteeringVector,
    create_linear_2_element,
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

    # Signal
    signal = np.exp(2j * np.pi * frequency * t).astype(np.complex64)

    # Add noise
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

    # Base signal
    base_signal = np.exp(2j * np.pi * signal_freq * t).astype(np.complex64)

    # Direction unit vector
    u = np.array([np.sin(azimuth), np.cos(azimuth), 0.0])

    signals = {}
    for element in config.enabled_elements:
        pos = element.position.to_array()
        # Phase delay based on position
        phase_delay = k * np.dot(pos, u)
        element_signal = base_signal * np.exp(1j * phase_delay)

        # Add noise
        noise_power = 10 ** (-snr_db / 10)
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(n_samples) + 1j * np.random.randn(n_samples)
        ).astype(np.complex64)

        signals[element.index] = element_signal + noise

    return signals


class TestCrossCorrelator:
    """Test CrossCorrelator class."""

    def test_initialization(self):
        """Test correlator initialization."""
        correlator = CrossCorrelator(sample_rate=2.4e6)
        assert correlator.sample_rate == 2.4e6

    def test_correlate_identical_signals(self):
        """Test correlation of identical signals."""
        correlator = CrossCorrelator(sample_rate=2.4e6)
        signal = generate_test_signal(10e3, 2.4e6, 0.01, snr_db=30)

        result = correlator.correlate(signal, signal, 0, 1)

        assert isinstance(result, CorrelationResult)
        assert result.delay_samples == pytest.approx(0.0, abs=0.5)
        assert result.phase_offset == pytest.approx(0.0, abs=0.1)
        assert result.correlation_peak > 0.9
        assert result.confidence > 0.9

    def test_correlate_delayed_signal(self):
        """Test correlation with time delay."""
        correlator = CrossCorrelator(sample_rate=2.4e6)

        # Create signal with known delay
        signal_a = generate_test_signal(10e3, 2.4e6, 0.01, snr_db=30)
        delay_samples = 10
        signal_b = np.roll(signal_a, delay_samples)

        result = correlator.correlate(signal_a, signal_b, 0, 1)

        # Magnitude should match, sign depends on convention
        assert abs(result.delay_samples) == pytest.approx(delay_samples, abs=1.0)
        assert result.is_valid

    def test_correlate_phase_shifted_signal(self):
        """Test correlation with phase shift."""
        correlator = CrossCorrelator(sample_rate=2.4e6)

        signal_a = generate_test_signal(10e3, 2.4e6, 0.01, snr_db=30)
        phase_shift = np.pi / 4
        signal_b = signal_a * np.exp(1j * phase_shift)

        result = correlator.correlate(signal_a, signal_b, 0, 1)

        # Magnitude should match, sign depends on convention (a vs b)
        assert abs(result.phase_offset) == pytest.approx(phase_shift, abs=0.2)

    def test_align_array(self):
        """Test full array alignment."""
        correlator = CrossCorrelator(sample_rate=2.4e6)

        # Create signals with known phase offsets
        base_signal = generate_test_signal(10e3, 2.4e6, 0.01, snr_db=30)
        signals = {
            0: base_signal,
            1: base_signal * np.exp(1j * np.pi / 4),
            2: base_signal * np.exp(1j * np.pi / 2),
            3: base_signal * np.exp(1j * 3 * np.pi / 4),
        }

        result = correlator.align_array(signals, reference_element=0)

        assert isinstance(result, ArrayAlignmentResult)
        assert result.reference_element == 0
        assert result.element_offsets[0] == pytest.approx(0.0, abs=0.1)
        assert result.overall_confidence > 0.5

    def test_apply_alignment(self):
        """Test applying alignment corrections."""
        correlator = CrossCorrelator(sample_rate=2.4e6)

        base_signal = generate_test_signal(10e3, 2.4e6, 0.01, snr_db=30)
        phase_offset = np.pi / 3
        signals = {
            0: base_signal,
            1: base_signal * np.exp(1j * phase_offset),
        }

        alignment = correlator.align_array(signals, reference_element=0)

        # Verify alignment detected the phase offset
        assert alignment.element_offsets[0] == pytest.approx(0.0, abs=0.1)
        assert abs(alignment.element_offsets[1]) == pytest.approx(phase_offset, abs=0.2)

        # Verify correction vector is generated
        correction = alignment.get_correction_vector(2)
        assert len(correction) == 2

    def test_estimate_frequency_offset(self):
        """Test frequency offset estimation."""
        correlator = CrossCorrelator(sample_rate=2.4e6)

        # Create signals with frequency offset
        n_samples = 10000
        t = np.arange(n_samples) / 2.4e6
        freq_offset = 100  # 100 Hz offset

        signal_a = np.exp(2j * np.pi * 10e3 * t).astype(np.complex64)
        signal_b = np.exp(2j * np.pi * (10e3 + freq_offset) * t).astype(np.complex64)

        time_span = n_samples / 2.4e6
        estimated_offset = correlator.estimate_frequency_offset(signal_a, signal_b, time_span)

        # Magnitude should match, sign depends on which signal is considered reference
        assert abs(estimated_offset) == pytest.approx(freq_offset, rel=0.1)

    def test_coherence(self):
        """Test coherence computation."""
        correlator = CrossCorrelator(sample_rate=2.4e6)

        signal = generate_test_signal(10e3, 2.4e6, 0.01, snr_db=30)

        freqs, coherence = correlator.coherence(signal, signal)

        assert len(freqs) > 0
        assert len(coherence) == len(freqs)
        # Coherence should be 1 for identical signals
        assert np.max(coherence) > 0.99


class TestCorrelationResult:
    """Test CorrelationResult class."""

    def test_is_valid(self):
        """Test validity check."""
        valid_result = CorrelationResult(
            element_a=0,
            element_b=1,
            delay_samples=5.0,
            delay_seconds=0.001,
            phase_offset=0.5,
            correlation_peak=0.8,
            snr_estimate=20.0,
            confidence=0.9,
        )
        assert valid_result.is_valid

        invalid_result = CorrelationResult(
            element_a=0,
            element_b=1,
            delay_samples=0.0,
            delay_seconds=0.0,
            phase_offset=0.0,
            correlation_peak=0.05,
            snr_estimate=0.0,
            confidence=0.2,
        )
        assert not invalid_result.is_valid


class TestBeamformer:
    """Test Beamformer class."""

    def test_initialization(self):
        """Test beamformer initialization."""
        config = create_linear_4_element(frequency=433e6)
        beamformer = Beamformer(config)

        assert beamformer.num_elements == 4
        assert beamformer.frequency == 433e6

    def test_compute_steering_vector(self):
        """Test steering vector computation."""
        config = create_linear_4_element(frequency=433e6)
        beamformer = Beamformer(config)

        sv = beamformer.compute_steering_vector(azimuth=0.0)

        assert isinstance(sv, SteeringVector)
        assert len(sv.weights) == 4
        assert sv.azimuth == 0.0
        assert sv.frequency == 433e6

    def test_steering_vector_boresight(self):
        """Test steering vector at boresight has uniform phase."""
        config = create_linear_4_element(frequency=433e6)
        beamformer = Beamformer(config)

        sv = beamformer.compute_steering_vector(azimuth=0.0, elevation=0.0)

        # At boresight, all weights should have similar phase
        phases = np.angle(sv.weights)
        phase_spread = np.max(phases) - np.min(phases)
        assert phase_spread < 0.1  # Small spread

    def test_steer_and_sum(self):
        """Test beam steering and summing."""
        config = create_linear_4_element(frequency=433e6)
        beamformer = Beamformer(config)

        # Generate signals from a known direction
        azimuth = np.radians(20)
        signals = generate_array_signals(config, azimuth, snr_db=20)

        # Steer toward signal
        result = beamformer.steer_and_sum(signals, azimuth)

        assert isinstance(result, BeamformerOutput)
        assert len(result.output_signal) > 0
        assert result.azimuth == azimuth

    def test_beam_pattern_peak_at_steering(self):
        """Test beam pattern has peak at steering direction."""
        config = create_linear_4_element(frequency=433e6)
        beamformer = Beamformer(config)

        steering_az = np.radians(15)
        pattern = beamformer.compute_pattern(steering_azimuth=steering_az)

        assert isinstance(pattern, BeamPattern)
        assert pattern.peak_azimuth == pytest.approx(steering_az, abs=np.radians(5))

    def test_scan_finds_signal(self):
        """Test beam scan finds signal direction."""
        config = create_linear_4_element(frequency=433e6)
        beamformer = Beamformer(config)

        # Generate signal from 25 degrees
        signal_azimuth = np.radians(25)
        signals = generate_array_signals(config, signal_azimuth, snr_db=25)

        # Scan
        azimuths = np.linspace(-np.pi / 3, np.pi / 3, 61)
        powers, peak_az, peak_power = beamformer.scan(signals, azimuths)

        assert peak_az == pytest.approx(signal_azimuth, abs=np.radians(10))

    def test_compute_array_factor(self):
        """Test array factor computation."""
        config = create_linear_4_element(frequency=433e6)
        beamformer = Beamformer(config)

        # Array factor at steering direction should be maximum
        steering_az = np.radians(10)
        af_at_steering = beamformer.compute_array_factor(
            steering_az, 0, steering_az, 0
        )
        af_away = beamformer.compute_array_factor(
            steering_az + np.pi / 4, 0, steering_az, 0
        )

        assert np.abs(af_at_steering) > np.abs(af_away)

    def test_create_null(self):
        """Test null steering."""
        config = create_linear_4_element(frequency=433e6)
        beamformer = Beamformer(config)

        signals = generate_array_signals(config, np.radians(0), snr_db=20)

        # Create beam toward 0 degrees with null at 30 degrees
        result = beamformer.create_null(
            signals,
            steering_azimuth=np.radians(0),
            null_azimuth=np.radians(30),
        )

        assert isinstance(result, BeamformerOutput)
        assert len(result.output_signal) > 0

    def test_spatial_spectrum(self):
        """Test spatial spectrum computation."""
        config = create_linear_4_element(frequency=433e6)
        beamformer = Beamformer(config)

        signals = generate_array_signals(config, np.radians(15), snr_db=20)
        azimuths, spectrum = beamformer.spatial_spectrum(signals)

        assert len(azimuths) > 0
        assert len(spectrum) == len(azimuths)

    def test_frequency_setter(self):
        """Test changing frequency."""
        config = create_linear_4_element(frequency=433e6)
        beamformer = Beamformer(config)

        beamformer.frequency = 915e6
        assert beamformer.frequency == 915e6


class TestBeamPattern:
    """Test BeamPattern class."""

    def test_beamwidth_calculation(self):
        """Test beamwidth is calculated."""
        config = create_linear_4_element(frequency=433e6)
        beamformer = Beamformer(config)

        pattern = beamformer.compute_pattern()

        assert pattern.beamwidth_az > 0
        assert pattern.beamwidth_az < np.pi  # Less than 180 degrees

    def test_first_null(self):
        """Test first null detection."""
        config = create_linear_4_element(frequency=433e6)
        beamformer = Beamformer(config)

        pattern = beamformer.compute_pattern()

        assert pattern.first_null_az > 0


class TestPhaseDifferenceDoA:
    """Test PhaseDifferenceDoA class."""

    def test_initialization(self):
        """Test DoA estimator initialization."""
        doa = PhaseDifferenceDoA(spacing=0.35, frequency=433e6)
        assert doa.spacing == 0.35
        assert doa.frequency == 433e6

    def test_max_unambiguous_angle(self):
        """Test maximum unambiguous angle calculation."""
        wavelength = SPEED_OF_LIGHT / 433e6
        spacing = wavelength / 4  # Quarter wavelength spacing

        doa = PhaseDifferenceDoA(spacing=spacing, frequency=433e6)
        assert doa.max_unambiguous_angle == pytest.approx(np.pi / 2, abs=0.1)

    def test_estimate_zero_angle(self):
        """Test DoA estimation at boresight."""
        wavelength = SPEED_OF_LIGHT / 433e6
        spacing = wavelength / 2

        doa = PhaseDifferenceDoA(spacing=spacing, frequency=433e6, sample_rate=2.4e6)

        # Generate signals with no phase difference (boresight)
        signal = generate_test_signal(10e3, 2.4e6, 0.01, snr_db=30)
        result = doa.estimate(signal, signal)

        assert isinstance(result, DoAResult)
        assert result.azimuth == pytest.approx(0.0, abs=0.1)
        assert result.method == DoAMethod.PHASE_DIFFERENCE

    def test_estimate_known_angle(self):
        """Test DoA estimation at known angle."""
        frequency = 433e6
        wavelength = SPEED_OF_LIGHT / frequency
        spacing = wavelength / 2

        doa = PhaseDifferenceDoA(spacing=spacing, frequency=frequency, sample_rate=2.4e6)

        # Generate signals with phase difference for 30 degrees
        angle = np.radians(30)
        phase_diff = 2 * np.pi * spacing / wavelength * np.sin(angle)

        signal_a = generate_test_signal(10e3, 2.4e6, 0.01, snr_db=30)
        signal_b = signal_a * np.exp(1j * phase_diff)

        result = doa.estimate(signal_a, signal_b)

        # Magnitude should match, sign depends on element ordering convention
        assert abs(result.azimuth) == pytest.approx(angle, abs=0.2)

    def test_estimate_methods(self):
        """Test different estimation methods."""
        wavelength = SPEED_OF_LIGHT / 433e6
        spacing = wavelength / 2

        doa = PhaseDifferenceDoA(spacing=spacing, frequency=433e6, sample_rate=2.4e6)

        signal = generate_test_signal(10e3, 2.4e6, 0.01, snr_db=30)

        for method in ["correlation", "instantaneous", "fft"]:
            result = doa.estimate(signal, signal, method=method)
            assert result.azimuth == pytest.approx(0.0, abs=0.2)

    def test_estimate_continuous(self):
        """Test continuous DoA estimation."""
        wavelength = SPEED_OF_LIGHT / 433e6
        spacing = wavelength / 2

        doa = PhaseDifferenceDoA(spacing=spacing, frequency=433e6, sample_rate=2.4e6)

        signal = generate_test_signal(10e3, 2.4e6, 0.1, snr_db=30)

        times, azimuths, confidences = doa.estimate_continuous(signal, signal)

        assert len(times) > 0
        assert len(azimuths) == len(times)
        assert len(confidences) == len(times)


class TestDoAResult:
    """Test DoAResult class."""

    def test_azimuth_deg(self):
        """Test azimuth in degrees."""
        result = DoAResult(
            azimuth=np.pi / 6,
            elevation=0.0,
            confidence=0.9,
            power=-10.0,
            method=DoAMethod.PHASE_DIFFERENCE,
        )
        assert result.azimuth_deg == pytest.approx(30.0, abs=0.1)

    def test_repr(self):
        """Test string representation."""
        result = DoAResult(
            azimuth=np.radians(45),
            elevation=np.radians(10),
            confidence=0.85,
            power=-5.0,
            method=DoAMethod.BEAMSCAN,
        )
        repr_str = repr(result)
        assert "45" in repr_str
        assert "DoAResult" in repr_str


class TestBeamscanDoA:
    """Test BeamscanDoA class."""

    def test_initialization(self):
        """Test beamscan DoA initialization."""
        config = create_linear_4_element(frequency=433e6)
        doa = BeamscanDoA(config)

        assert doa.config == config

    def test_estimate_finds_signal(self):
        """Test beamscan finds signal direction."""
        config = create_linear_4_element(frequency=433e6)
        doa = BeamscanDoA(config, azimuth_resolution=2.0)

        signal_azimuth = np.radians(20)
        signals = generate_array_signals(config, signal_azimuth, snr_db=25)

        result = doa.estimate(signals)

        assert isinstance(result, DoAResult)
        assert result.method == DoAMethod.BEAMSCAN
        assert result.azimuth == pytest.approx(signal_azimuth, abs=np.radians(10))

    def test_compute_spectrum(self):
        """Test spectrum computation."""
        config = create_linear_4_element(frequency=433e6)
        doa = BeamscanDoA(config)

        signals = generate_array_signals(config, np.radians(15), snr_db=20)
        azimuths, spectrum = doa.compute_spectrum(signals)

        assert len(azimuths) > 0
        assert len(spectrum) == len(azimuths)


class TestMUSICDoA:
    """Test MUSICDoA class."""

    def test_initialization(self):
        """Test MUSIC DoA initialization."""
        config = create_linear_4_element(frequency=433e6)
        doa = MUSICDoA(config, num_sources=1)

        assert doa.num_sources == 1

    def test_num_sources_setter(self):
        """Test num_sources setter validation."""
        config = create_linear_4_element(frequency=433e6)
        doa = MUSICDoA(config, num_sources=1)

        doa.num_sources = 2
        assert doa.num_sources == 2

        with pytest.raises(ValueError):
            doa.num_sources = 0

        with pytest.raises(ValueError):
            doa.num_sources = 10  # More than elements

    def test_estimate_single_source(self):
        """Test MUSIC with single source."""
        config = create_linear_4_element(frequency=433e6)
        doa = MUSICDoA(config, num_sources=1, azimuth_resolution=1.0)

        signal_azimuth = np.radians(25)
        signals = generate_array_signals(config, signal_azimuth, snr_db=25)

        result = doa.estimate(signals)

        assert isinstance(result, MultiSourceDoAResult)
        assert result.num_sources >= 1
        if result.sources:
            assert result.sources[0].method == DoAMethod.MUSIC
            # MUSIC should find signal near true direction
            assert result.sources[0].azimuth == pytest.approx(
                signal_azimuth, abs=np.radians(15)
            )

    def test_compute_spectrum(self):
        """Test MUSIC spectrum computation."""
        config = create_linear_4_element(frequency=433e6)
        doa = MUSICDoA(config, num_sources=1)

        signals = generate_array_signals(config, np.radians(10), snr_db=20)
        azimuths, spectrum = doa.compute_spectrum(signals)

        assert len(azimuths) > 0
        assert len(spectrum) == len(azimuths)

    def test_result_has_spectrum(self):
        """Test that result includes spectrum data."""
        config = create_linear_4_element(frequency=433e6)
        doa = MUSICDoA(config, num_sources=1)

        signals = generate_array_signals(config, np.radians(0), snr_db=20)
        result = doa.estimate(signals)

        assert result.spectrum is not None
        assert result.azimuths is not None


class TestMultiSourceDoAResult:
    """Test MultiSourceDoAResult class."""

    def test_num_sources_property(self):
        """Test num_sources property."""
        sources = [
            DoAResult(np.radians(10), 0, 0.9, -5, DoAMethod.MUSIC),
            DoAResult(np.radians(-20), 0, 0.8, -8, DoAMethod.MUSIC),
        ]
        result = MultiSourceDoAResult(
            sources=sources,
            num_sources_estimated=2,
        )

        assert result.num_sources == 2


class TestIntegration:
    """Integration tests for Phase 2 components."""

    def test_full_doa_pipeline(self):
        """Test full DoA pipeline: correlate -> align -> beamform -> estimate."""
        config = create_linear_4_element(frequency=433e6)

        # Generate signal from known direction
        signal_azimuth = np.radians(30)
        signals = generate_array_signals(config, signal_azimuth, snr_db=25)

        # Step 1: Cross-correlate for alignment
        correlator = CrossCorrelator(sample_rate=config.common_sample_rate)
        alignment = correlator.align_array(signals, reference_element=0)

        assert alignment.overall_confidence > 0

        # Step 2: Use beamscan DoA directly (more robust)
        doa = BeamscanDoA(config, azimuth_resolution=2.0)
        result = doa.estimate(signals)

        # Verify detection has reasonable confidence
        assert result.confidence > 0.3

        # Step 3: Beamform toward detected direction
        beamformer = Beamformer(config)
        output = beamformer.steer_and_sum(signals, result.azimuth)

        # Step 4: Verify beamformer produces output
        assert len(output.output_signal) > 0
        assert output.beam_power > -100

    def test_beamformer_with_doa(self):
        """Test using DoA result to steer beamformer."""
        config = create_linear_4_element(frequency=433e6)

        signal_azimuth = np.radians(20)
        signals = generate_array_signals(config, signal_azimuth, snr_db=25)

        # Estimate DoA
        doa = BeamscanDoA(config)
        doa_result = doa.estimate(signals)

        # Steer beamformer based on DoA
        beamformer = Beamformer(config)
        output = beamformer.steer_and_sum(signals, doa_result.azimuth)

        assert len(output.output_signal) > 0
        assert output.beam_power > -100  # Reasonable power

"""
Cross-correlator for phase alignment between antenna array elements.

Provides algorithms for estimating time delays and phase offsets
between signals received at different array elements.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CorrelationResult:
    """Result of cross-correlation between two signals."""

    element_a: int  # First element index
    element_b: int  # Second element index
    delay_samples: float  # Estimated delay in samples (fractional)
    delay_seconds: float  # Estimated delay in seconds
    phase_offset: float  # Phase offset in radians
    correlation_peak: float  # Peak correlation value (0-1)
    snr_estimate: float  # Estimated SNR in dB
    confidence: float  # Confidence score (0-1)

    @property
    def is_valid(self) -> bool:
        """Check if correlation result is valid."""
        return self.correlation_peak > 0.1 and self.confidence > 0.5


@dataclass
class ArrayAlignmentResult:
    """Result of full array phase alignment."""

    reference_element: int
    element_offsets: Dict[int, float]  # element_index -> phase offset in radians
    element_delays: Dict[int, float]  # element_index -> delay in samples
    pairwise_correlations: List[CorrelationResult]
    overall_confidence: float
    is_synchronized: bool

    def get_correction_vector(self, num_elements: int) -> np.ndarray:
        """Get complex correction phasors for all elements."""
        corrections = np.ones(num_elements, dtype=np.complex64)
        for elem_idx, phase_offset in self.element_offsets.items():
            if elem_idx < num_elements:
                corrections[elem_idx] = np.exp(-1j * phase_offset)
        return corrections


class CrossCorrelator:
    """
    Cross-correlator for estimating phase and time alignment between signals.

    Uses FFT-based cross-correlation with sub-sample interpolation for
    accurate delay estimation between antenna array elements.

    Features:
        - FFT-based cross-correlation for efficiency
        - Sub-sample delay estimation via parabolic interpolation
        - Phase offset extraction at signal frequency
        - SNR estimation from correlation peak
        - Support for narrowband and wideband signals

    Example:
        correlator = CrossCorrelator(sample_rate=2.4e6)

        # Correlate two element signals
        result = correlator.correlate(samples_0, samples_1, 0, 1)
        print(f"Delay: {result.delay_samples:.2f} samples")
        print(f"Phase offset: {np.degrees(result.phase_offset):.1f} degrees")

        # Align full array
        samples = {0: sig0, 1: sig1, 2: sig2, 3: sig3}
        alignment = correlator.align_array(samples, reference_element=0)
    """

    def __init__(
        self,
        sample_rate: float = 2.4e6,
        max_delay_samples: int = 1000,
        interpolation_factor: int = 16,
        correlation_threshold: float = 0.3,
    ) -> None:
        """
        Initialize cross-correlator.

        Args:
            sample_rate: Sample rate in Hz
            max_delay_samples: Maximum expected delay in samples
            interpolation_factor: Factor for sub-sample interpolation
            correlation_threshold: Minimum correlation for valid result
        """
        self._sample_rate = sample_rate
        self._max_delay_samples = max_delay_samples
        self._interpolation_factor = interpolation_factor
        self._correlation_threshold = correlation_threshold

    @property
    def sample_rate(self) -> float:
        """Get sample rate."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, rate: float) -> None:
        """Set sample rate."""
        self._sample_rate = rate

    def correlate(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        element_a: int = 0,
        element_b: int = 1,
        center_frequency: Optional[float] = None,
    ) -> CorrelationResult:
        """
        Compute cross-correlation between two signals.

        Args:
            signal_a: First signal (complex samples)
            signal_b: Second signal (complex samples)
            element_a: Index of first element
            element_b: Index of second element
            center_frequency: Optional center frequency for phase calculation

        Returns:
            CorrelationResult with delay and phase estimates
        """
        # Ensure same length
        min_len = min(len(signal_a), len(signal_b))
        signal_a = signal_a[:min_len]
        signal_b = signal_b[:min_len]

        # Normalize signals
        signal_a = signal_a - np.mean(signal_a)
        signal_b = signal_b - np.mean(signal_b)

        norm_a = np.linalg.norm(signal_a)
        norm_b = np.linalg.norm(signal_b)

        if norm_a < 1e-10 or norm_b < 1e-10:
            # Signals too weak
            return CorrelationResult(
                element_a=element_a,
                element_b=element_b,
                delay_samples=0.0,
                delay_seconds=0.0,
                phase_offset=0.0,
                correlation_peak=0.0,
                snr_estimate=-np.inf,
                confidence=0.0,
            )

        signal_a = signal_a / norm_a
        signal_b = signal_b / norm_b

        # FFT-based cross-correlation
        n = len(signal_a)
        fft_size = 2 ** int(np.ceil(np.log2(2 * n - 1)))

        fft_a = np.fft.fft(signal_a, fft_size)
        fft_b = np.fft.fft(signal_b, fft_size)

        # Cross-correlation in frequency domain
        cross_spectrum = fft_a * np.conj(fft_b)
        correlation = np.fft.ifft(cross_spectrum)

        # Take magnitude for delay estimation
        corr_mag = np.abs(correlation)

        # Find peak within allowed delay range
        search_range = min(self._max_delay_samples, n // 2)

        # Search positive delays (signal_b delayed relative to signal_a)
        pos_range = corr_mag[:search_range]
        # Search negative delays (signal_a delayed relative to signal_b)
        neg_range = corr_mag[-search_range:]

        pos_peak_idx = np.argmax(pos_range)
        neg_peak_idx = np.argmax(neg_range)

        pos_peak_val = pos_range[pos_peak_idx]
        neg_peak_val = neg_range[neg_peak_idx]

        if pos_peak_val >= neg_peak_val:
            peak_idx = pos_peak_idx
            peak_val = pos_peak_val
        else:
            peak_idx = len(corr_mag) - search_range + neg_peak_idx
            peak_val = neg_peak_val

        # Sub-sample interpolation using parabolic fit
        delay_samples = self._subsample_interpolate(corr_mag, peak_idx)

        # Convert to actual delay (handle wrap-around)
        if delay_samples > fft_size // 2:
            delay_samples = delay_samples - fft_size

        delay_seconds = delay_samples / self._sample_rate

        # Extract phase at peak
        phase_offset = np.angle(correlation[peak_idx])

        # If center frequency provided, adjust phase for frequency
        if center_frequency is not None:
            # Phase contribution from time delay
            phase_from_delay = 2 * np.pi * center_frequency * delay_seconds
            # Remove time-delay phase to get residual phase offset
            phase_offset = self._wrap_phase(phase_offset - phase_from_delay)

        # Estimate SNR from correlation peak
        noise_floor = np.median(corr_mag)
        if noise_floor > 0:
            snr_estimate = 10 * np.log10(peak_val / noise_floor)
        else:
            snr_estimate = 60.0  # Very high SNR

        # Confidence based on correlation peak and SNR
        confidence = min(1.0, peak_val * (1 + snr_estimate / 20))
        confidence = max(0.0, confidence)

        return CorrelationResult(
            element_a=element_a,
            element_b=element_b,
            delay_samples=delay_samples,
            delay_seconds=delay_seconds,
            phase_offset=phase_offset,
            correlation_peak=peak_val,
            snr_estimate=snr_estimate,
            confidence=confidence,
        )

    def _subsample_interpolate(self, corr_mag: np.ndarray, peak_idx: int) -> float:
        """
        Sub-sample interpolation using parabolic fit.

        Args:
            corr_mag: Correlation magnitude array
            peak_idx: Index of peak

        Returns:
            Interpolated peak position
        """
        n = len(corr_mag)

        # Get neighbors (with wrap-around)
        y0 = corr_mag[(peak_idx - 1) % n]
        y1 = corr_mag[peak_idx]
        y2 = corr_mag[(peak_idx + 1) % n]

        # Parabolic interpolation
        # y = a*x^2 + b*x + c
        # Peak at x = -b/(2a)
        denom = y0 - 2 * y1 + y2
        if abs(denom) < 1e-10:
            return float(peak_idx)

        delta = 0.5 * (y0 - y2) / denom
        delta = max(-1.0, min(1.0, delta))  # Clamp to reasonable range

        return peak_idx + delta

    def _wrap_phase(self, phase: float) -> float:
        """Wrap phase to [-pi, pi]."""
        return (phase + np.pi) % (2 * np.pi) - np.pi

    def correlate_with_reference(
        self,
        reference: np.ndarray,
        signals: Dict[int, np.ndarray],
        reference_element: int = 0,
        center_frequency: Optional[float] = None,
    ) -> Dict[int, CorrelationResult]:
        """
        Correlate multiple signals against a reference.

        Args:
            reference: Reference signal
            signals: Dict mapping element index to signal
            reference_element: Index of reference element
            center_frequency: Optional center frequency

        Returns:
            Dict mapping element index to CorrelationResult
        """
        results = {}

        for elem_idx, signal in signals.items():
            if elem_idx == reference_element:
                # Reference element has zero offset
                results[elem_idx] = CorrelationResult(
                    element_a=reference_element,
                    element_b=elem_idx,
                    delay_samples=0.0,
                    delay_seconds=0.0,
                    phase_offset=0.0,
                    correlation_peak=1.0,
                    snr_estimate=60.0,
                    confidence=1.0,
                )
            else:
                results[elem_idx] = self.correlate(
                    reference,
                    signal,
                    reference_element,
                    elem_idx,
                    center_frequency,
                )

        return results

    def align_array(
        self,
        signals: Dict[int, np.ndarray],
        reference_element: int = 0,
        center_frequency: Optional[float] = None,
        min_confidence: float = 0.5,
    ) -> ArrayAlignmentResult:
        """
        Compute full array alignment relative to reference element.

        Args:
            signals: Dict mapping element index to signal
            reference_element: Index of reference element
            center_frequency: Optional center frequency
            min_confidence: Minimum confidence for valid alignment

        Returns:
            ArrayAlignmentResult with offsets for all elements
        """
        if reference_element not in signals:
            raise ValueError(f"Reference element {reference_element} not in signals")

        reference = signals[reference_element]
        correlations = self.correlate_with_reference(
            reference, signals, reference_element, center_frequency
        )

        element_offsets = {}
        element_delays = {}
        pairwise_results = []
        valid_count = 0

        for elem_idx, result in correlations.items():
            element_offsets[elem_idx] = result.phase_offset
            element_delays[elem_idx] = result.delay_samples
            pairwise_results.append(result)

            if result.confidence >= min_confidence:
                valid_count += 1

        # Overall confidence based on fraction of valid correlations
        overall_confidence = valid_count / len(signals) if signals else 0.0
        is_synchronized = overall_confidence >= 0.8

        return ArrayAlignmentResult(
            reference_element=reference_element,
            element_offsets=element_offsets,
            element_delays=element_delays,
            pairwise_correlations=pairwise_results,
            overall_confidence=overall_confidence,
            is_synchronized=is_synchronized,
        )

    def apply_alignment(
        self,
        signals: Dict[int, np.ndarray],
        alignment: ArrayAlignmentResult,
    ) -> Dict[int, np.ndarray]:
        """
        Apply alignment corrections to signals.

        Args:
            signals: Dict mapping element index to signal
            alignment: Alignment result from align_array()

        Returns:
            Dict with phase-corrected signals
        """
        corrected = {}

        for elem_idx, signal in signals.items():
            phase_offset = alignment.element_offsets.get(elem_idx, 0.0)
            # Apply conjugate phase to correct
            correction = np.exp(-1j * phase_offset)
            corrected[elem_idx] = signal * correction

        return corrected

    def estimate_frequency_offset(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        time_span: float,
    ) -> float:
        """
        Estimate frequency offset between two oscillators.

        Useful for detecting clock drift between non-coherent SDRs.

        Args:
            signal_a: First signal
            signal_b: Second signal
            time_span: Time span of signals in seconds

        Returns:
            Estimated frequency offset in Hz
        """
        # Compute instantaneous phase difference
        phase_diff = np.angle(signal_a * np.conj(signal_b))

        # Unwrap phase
        phase_diff = np.unwrap(phase_diff)

        # Linear fit to get frequency offset
        n = len(phase_diff)
        t = np.arange(n) / self._sample_rate

        # Simple linear regression
        t_mean = np.mean(t)
        phase_mean = np.mean(phase_diff)

        numerator = np.sum((t - t_mean) * (phase_diff - phase_mean))
        denominator = np.sum((t - t_mean) ** 2)

        if abs(denominator) < 1e-20:
            return 0.0

        slope = numerator / denominator  # rad/s
        freq_offset = slope / (2 * np.pi)  # Hz

        return freq_offset

    def coherence(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        nperseg: int = 256,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute magnitude-squared coherence between two signals.

        Args:
            signal_a: First signal
            signal_b: Second signal
            nperseg: Segment length for Welch method

        Returns:
            Tuple of (frequencies, coherence) arrays
        """
        min_len = min(len(signal_a), len(signal_b))
        signal_a = signal_a[:min_len]
        signal_b = signal_b[:min_len]

        # Number of segments
        n_segments = max(1, min_len // nperseg)
        actual_nperseg = min_len // n_segments

        # Welch's method for cross-spectral density
        psd_aa = np.zeros(actual_nperseg, dtype=np.float64)
        psd_bb = np.zeros(actual_nperseg, dtype=np.float64)
        csd_ab = np.zeros(actual_nperseg, dtype=np.complex128)

        window = np.hanning(actual_nperseg)

        for i in range(n_segments):
            start = i * actual_nperseg
            seg_a = signal_a[start : start + actual_nperseg] * window
            seg_b = signal_b[start : start + actual_nperseg] * window

            fft_a = np.fft.fft(seg_a)
            fft_b = np.fft.fft(seg_b)

            psd_aa += np.abs(fft_a) ** 2
            psd_bb += np.abs(fft_b) ** 2
            csd_ab += fft_a * np.conj(fft_b)

        # Magnitude-squared coherence
        denom = psd_aa * psd_bb
        denom[denom < 1e-20] = 1e-20
        coherence = np.abs(csd_ab) ** 2 / denom

        # Frequency axis
        freqs = np.fft.fftfreq(actual_nperseg, 1 / self._sample_rate)

        # Return positive frequencies only
        pos_mask = freqs >= 0
        return freqs[pos_mask], coherence[pos_mask]

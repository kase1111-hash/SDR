"""
Direction of Arrival (DoA) estimation algorithms.

Provides algorithms for estimating the direction from which
signals arrive at an antenna array.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .array_config import SPEED_OF_LIGHT, ArrayConfig

logger = logging.getLogger(__name__)


class DoAMethod(Enum):
    """Direction of Arrival estimation methods."""

    PHASE_DIFFERENCE = "phase_difference"  # Simple 2-element phase difference
    BEAMSCAN = "beamscan"  # Conventional beamformer scan
    CAPON = "capon"  # Capon/MVDR
    MUSIC = "music"  # MUSIC algorithm
    ESPRIT = "esprit"  # ESPRIT algorithm


@dataclass
class DoAResult:
    """Result of direction of arrival estimation."""

    azimuth: float  # Estimated azimuth in radians
    elevation: float  # Estimated elevation in radians
    confidence: float  # Confidence score (0-1)
    power: float  # Signal power in dB
    method: DoAMethod  # Method used

    @property
    def azimuth_deg(self) -> float:
        """Azimuth in degrees."""
        return np.degrees(self.azimuth)

    @property
    def elevation_deg(self) -> float:
        """Elevation in degrees."""
        return np.degrees(self.elevation)

    def __repr__(self) -> str:
        return (
            f"DoAResult(az={self.azimuth_deg:.1f}°, el={self.elevation_deg:.1f}°, "
            f"conf={self.confidence:.2f}, power={self.power:.1f}dB)"
        )


@dataclass
class MultiSourceDoAResult:
    """Result of multi-source DoA estimation."""

    sources: List[DoAResult]  # Detected sources
    num_sources_estimated: int  # Estimated number of sources
    spectrum: Optional[np.ndarray] = None  # Spatial spectrum
    azimuths: Optional[np.ndarray] = None  # Azimuth values for spectrum

    @property
    def num_sources(self) -> int:
        """Number of detected sources."""
        return len(self.sources)


class PhaseDifferenceDoA:
    """
    Simple 2-element phase difference direction finder.

    Uses the phase difference between two antenna elements to
    estimate the direction of arrival. Best suited for single-source
    scenarios with a 2-element array.

    The relationship between phase difference and angle is:
        phi = (2*pi*d/lambda) * sin(theta)

    where:
        phi = phase difference (radians)
        d = element spacing (meters)
        lambda = wavelength (meters)
        theta = angle of arrival (radians, 0 = boresight)

    Limitations:
        - Assumes single source
        - Ambiguity when d > lambda/2 (grating lobes)
        - Only provides azimuth (2D)

    Example:
        doa = PhaseDifferenceDoA(spacing=0.35, frequency=433e6)
        result = doa.estimate(signal_0, signal_1)
        print(f"Direction: {result.azimuth_deg:.1f} degrees")
    """

    def __init__(
        self,
        spacing: float,
        frequency: float,
        sample_rate: float = 2.4e6,
    ) -> None:
        """
        Initialize phase difference DoA estimator.

        Args:
            spacing: Element spacing in meters
            frequency: Operating frequency in Hz
            sample_rate: Sample rate in Hz
        """
        self._spacing = spacing
        self._frequency = frequency
        self._sample_rate = sample_rate
        self._wavelength = SPEED_OF_LIGHT / frequency

        # Check for ambiguity
        if spacing > self._wavelength / 2:
            logger.warning(
                f"Element spacing ({spacing:.3f}m) > lambda/2 ({self._wavelength/2:.3f}m). "
                "Grating lobe ambiguity may occur."
            )

    @property
    def spacing(self) -> float:
        """Get element spacing."""
        return self._spacing

    @property
    def frequency(self) -> float:
        """Get operating frequency."""
        return self._frequency

    @frequency.setter
    def frequency(self, freq: float) -> None:
        """Set operating frequency."""
        self._frequency = freq
        self._wavelength = SPEED_OF_LIGHT / freq

    @property
    def wavelength(self) -> float:
        """Get wavelength."""
        return self._wavelength

    @property
    def max_unambiguous_angle(self) -> float:
        """Maximum unambiguous angle (radians)."""
        ratio = self._wavelength / (2 * self._spacing)
        if ratio >= 1:
            return np.pi / 2
        return np.arcsin(ratio)

    def estimate(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        method: str = "correlation",
    ) -> DoAResult:
        """
        Estimate direction of arrival from two signals.

        Args:
            signal_a: Signal from first element
            signal_b: Signal from second element
            method: Phase estimation method ("correlation", "instantaneous", "fft")

        Returns:
            DoAResult with estimated direction
        """
        if method == "correlation":
            phase_diff, confidence = self._phase_from_correlation(signal_a, signal_b)
        elif method == "instantaneous":
            phase_diff, confidence = self._phase_from_instantaneous(signal_a, signal_b)
        elif method == "fft":
            phase_diff, confidence = self._phase_from_fft(signal_a, signal_b)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Convert phase difference to angle
        # phi = (2*pi*d/lambda) * sin(theta)
        # sin(theta) = phi * lambda / (2*pi*d)
        k = 2 * np.pi / self._wavelength
        sin_theta = phase_diff / (k * self._spacing)

        # Clamp to valid range
        sin_theta = np.clip(sin_theta, -1.0, 1.0)
        azimuth = np.arcsin(sin_theta)

        # Estimate power
        power = 10 * np.log10(
            np.mean(np.abs(signal_a) ** 2 + np.abs(signal_b) ** 2) / 2 + 1e-20
        )

        return DoAResult(
            azimuth=azimuth,
            elevation=0.0,
            confidence=confidence,
            power=power,
            method=DoAMethod.PHASE_DIFFERENCE,
        )

    def _phase_from_correlation(
        self, signal_a: np.ndarray, signal_b: np.ndarray
    ) -> Tuple[float, float]:
        """Estimate phase using cross-correlation at zero lag."""
        # Normalize
        signal_a = signal_a - np.mean(signal_a)
        signal_b = signal_b - np.mean(signal_b)

        # Cross-correlation at zero lag
        correlation = np.sum(signal_a * np.conj(signal_b))
        phase_diff = np.angle(correlation)

        # Confidence from correlation magnitude
        norm = np.sqrt(np.sum(np.abs(signal_a) ** 2) * np.sum(np.abs(signal_b) ** 2))
        if norm > 0:
            confidence = np.abs(correlation) / norm
        else:
            confidence = 0.0

        return phase_diff, confidence

    def _phase_from_instantaneous(
        self, signal_a: np.ndarray, signal_b: np.ndarray
    ) -> Tuple[float, float]:
        """Estimate phase from instantaneous phase difference."""
        # Compute instantaneous phase difference
        phase_diff_inst = np.angle(signal_a * np.conj(signal_b))

        # Average (circular mean)
        mean_phase = np.angle(np.mean(np.exp(1j * phase_diff_inst)))

        # Confidence from phase consistency
        phase_variance = 1 - np.abs(np.mean(np.exp(1j * phase_diff_inst)))
        confidence = 1 - phase_variance

        return mean_phase, confidence

    def _phase_from_fft(
        self, signal_a: np.ndarray, signal_b: np.ndarray
    ) -> Tuple[float, float]:
        """Estimate phase at peak frequency using FFT."""
        n = len(signal_a)

        fft_a = np.fft.fft(signal_a)
        fft_b = np.fft.fft(signal_b)

        # Find peak frequency
        power = np.abs(fft_a) ** 2 + np.abs(fft_b) ** 2
        peak_idx = np.argmax(power[: n // 2])  # Positive frequencies only

        # Phase difference at peak
        phase_diff = np.angle(fft_a[peak_idx]) - np.angle(fft_b[peak_idx])
        phase_diff = self._wrap_phase(phase_diff)

        # Confidence from peak prominence
        noise_floor = np.median(power)
        if noise_floor > 0:
            snr = power[peak_idx] / noise_floor
            confidence = min(1.0, snr / 100)
        else:
            confidence = 1.0

        return phase_diff, confidence

    def _wrap_phase(self, phase: float) -> float:
        """Wrap phase to [-pi, pi]."""
        return (phase + np.pi) % (2 * np.pi) - np.pi

    def estimate_continuous(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        window_size: int = 1024,
        overlap: float = 0.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate DoA continuously over time.

        Args:
            signal_a: Signal from first element
            signal_b: Signal from second element
            window_size: Samples per estimation window
            overlap: Overlap fraction between windows

        Returns:
            Tuple of (times, azimuths, confidences)
        """
        step = int(window_size * (1 - overlap))
        n_windows = (len(signal_a) - window_size) // step + 1

        times = []
        azimuths = []
        confidences = []

        for i in range(n_windows):
            start = i * step
            end = start + window_size

            result = self.estimate(signal_a[start:end], signal_b[start:end])

            times.append((start + window_size // 2) / self._sample_rate)
            azimuths.append(result.azimuth)
            confidences.append(result.confidence)

        return np.array(times), np.array(azimuths), np.array(confidences)


class BeamscanDoA:
    """
    Conventional beamscan direction of arrival estimator.

    Scans the beam across all angles and finds the peak response.
    Works with any array geometry.

    Example:
        config = create_linear_4_element(frequency=433e6)
        doa = BeamscanDoA(config)
        result = doa.estimate(signals)
    """

    def __init__(
        self,
        config: ArrayConfig,
        azimuth_resolution: float = 1.0,  # degrees
    ) -> None:
        """
        Initialize beamscan DoA estimator.

        Args:
            config: Array configuration
            azimuth_resolution: Scan resolution in degrees
        """
        self._config = config
        self._positions = config.get_position_matrix()
        self._num_elements = len(self._positions)
        self._wavelength = SPEED_OF_LIGHT / config.common_frequency
        self._frequency = config.common_frequency

        # Scan angles
        n_angles = int(180 / azimuth_resolution) + 1
        self._scan_azimuths = np.linspace(-np.pi / 2, np.pi / 2, n_angles)

    @property
    def config(self) -> ArrayConfig:
        """Get array configuration."""
        return self._config

    def _compute_steering_vector(self, azimuth: float, elevation: float = 0.0) -> np.ndarray:
        """Compute steering vector for given direction."""
        k = 2 * np.pi / self._wavelength

        cos_el = np.cos(elevation)
        u = np.array([
            cos_el * np.sin(azimuth),
            cos_el * np.cos(azimuth),
            np.sin(elevation),
        ])

        phases = k * (self._positions @ u)
        return np.exp(1j * phases).astype(np.complex64)

    def estimate(
        self,
        signals: Dict[int, np.ndarray],
        elevation: float = 0.0,
    ) -> DoAResult:
        """
        Estimate direction of arrival.

        Args:
            signals: Dict mapping element index to signal
            elevation: Elevation angle (radians), default 0

        Returns:
            DoAResult with estimated direction
        """
        # Build data matrix
        lengths = [len(sig) for sig in signals.values()]
        if not lengths:
            return DoAResult(
                azimuth=0.0,
                elevation=elevation,
                confidence=0.0,
                power=-np.inf,
                method=DoAMethod.BEAMSCAN,
            )

        min_len = min(lengths)
        data = np.zeros((self._num_elements, min_len), dtype=np.complex64)

        for elem in self._config.enabled_elements:
            idx = elem.index
            if idx in signals and idx < self._num_elements:
                data[idx, :] = signals[idx][:min_len]

        # Scan all azimuths
        powers = np.zeros(len(self._scan_azimuths))

        for i, az in enumerate(self._scan_azimuths):
            sv = self._compute_steering_vector(az, elevation)
            # Beamformer output power
            output = sv.conj() @ data
            powers[i] = np.mean(np.abs(output) ** 2)

        # Find peak
        peak_idx = np.argmax(powers)
        peak_azimuth = self._scan_azimuths[peak_idx]
        peak_power = 10 * np.log10(powers[peak_idx] + 1e-20)

        # Confidence from peak sharpness
        if len(powers) > 2:
            noise_floor = np.median(powers)
            if noise_floor > 0:
                peak_to_median = powers[peak_idx] / noise_floor
                confidence = min(1.0, peak_to_median / 10)
            else:
                confidence = 1.0
        else:
            confidence = 0.5

        return DoAResult(
            azimuth=peak_azimuth,
            elevation=elevation,
            confidence=confidence,
            power=peak_power,
            method=DoAMethod.BEAMSCAN,
        )

    def compute_spectrum(
        self,
        signals: Dict[int, np.ndarray],
        azimuths: Optional[np.ndarray] = None,
        elevation: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute spatial spectrum.

        Args:
            signals: Dict mapping element index to signal
            azimuths: Azimuth angles (radians), default scan angles
            elevation: Elevation angle (radians)

        Returns:
            Tuple of (azimuths, spectrum_db)
        """
        if azimuths is None:
            azimuths = self._scan_azimuths

        # Build data matrix
        lengths = [len(sig) for sig in signals.values()]
        if not lengths:
            return azimuths, np.full(len(azimuths), -np.inf)

        min_len = min(lengths)
        data = np.zeros((self._num_elements, min_len), dtype=np.complex64)

        for elem in self._config.enabled_elements:
            idx = elem.index
            if idx in signals and idx < self._num_elements:
                data[idx, :] = signals[idx][:min_len]

        # Compute spectrum
        spectrum = np.zeros(len(azimuths))

        for i, az in enumerate(azimuths):
            sv = self._compute_steering_vector(az, elevation)
            output = sv.conj() @ data
            spectrum[i] = np.mean(np.abs(output) ** 2)

        spectrum_db = 10 * np.log10(spectrum + 1e-20)

        return azimuths, spectrum_db


class MUSICDoA:
    """
    MUSIC (MUltiple SIgnal Classification) direction of arrival estimator.

    Uses subspace decomposition to achieve super-resolution DoA estimation.
    Can resolve multiple sources and works well in moderate SNR conditions.

    Algorithm:
        1. Compute sample covariance matrix
        2. Eigendecompose to find signal and noise subspaces
        3. Compute MUSIC pseudo-spectrum
        4. Find peaks in spectrum

    Example:
        config = create_linear_4_element(frequency=433e6)
        doa = MUSICDoA(config, num_sources=2)
        result = doa.estimate(signals)
        for source in result.sources:
            print(f"Source at {source.azimuth_deg:.1f} degrees")
    """

    def __init__(
        self,
        config: ArrayConfig,
        num_sources: int = 1,
        azimuth_resolution: float = 0.5,  # degrees
    ) -> None:
        """
        Initialize MUSIC DoA estimator.

        Args:
            config: Array configuration
            num_sources: Expected number of sources
            azimuth_resolution: Scan resolution in degrees
        """
        self._config = config
        self._positions = config.get_position_matrix()
        self._num_elements = len(self._positions)
        self._num_sources = num_sources
        self._wavelength = SPEED_OF_LIGHT / config.common_frequency

        # Scan angles
        n_angles = int(180 / azimuth_resolution) + 1
        self._scan_azimuths = np.linspace(-np.pi / 2, np.pi / 2, n_angles)

        # Precompute steering vectors
        self._steering_vectors = np.zeros(
            (self._num_elements, len(self._scan_azimuths)), dtype=np.complex64
        )
        for i, az in enumerate(self._scan_azimuths):
            self._steering_vectors[:, i] = self._compute_steering_vector(az)

    @property
    def num_sources(self) -> int:
        """Get expected number of sources."""
        return self._num_sources

    @num_sources.setter
    def num_sources(self, n: int) -> None:
        """Set expected number of sources."""
        if n < 1:
            raise ValueError("num_sources must be >= 1")
        if n >= self._num_elements:
            raise ValueError(f"num_sources must be < num_elements ({self._num_elements})")
        self._num_sources = n

    def _compute_steering_vector(self, azimuth: float, elevation: float = 0.0) -> np.ndarray:
        """Compute steering vector for given direction."""
        k = 2 * np.pi / self._wavelength

        cos_el = np.cos(elevation)
        u = np.array([
            cos_el * np.sin(azimuth),
            cos_el * np.cos(azimuth),
            np.sin(elevation),
        ])

        phases = k * (self._positions @ u)
        return np.exp(1j * phases).astype(np.complex64)

    def estimate(
        self,
        signals: Dict[int, np.ndarray],
        num_sources: Optional[int] = None,
        elevation: float = 0.0,
    ) -> MultiSourceDoAResult:
        """
        Estimate direction of arrival using MUSIC algorithm.

        Args:
            signals: Dict mapping element index to signal
            num_sources: Number of sources (overrides constructor)
            elevation: Elevation angle (radians)

        Returns:
            MultiSourceDoAResult with detected sources
        """
        n_sources = num_sources or self._num_sources

        # Build data matrix
        lengths = [len(sig) for sig in signals.values()]
        if not lengths:
            return MultiSourceDoAResult(
                sources=[],
                num_sources_estimated=0,
            )

        min_len = min(lengths)
        data = np.zeros((self._num_elements, min_len), dtype=np.complex64)

        for elem in self._config.enabled_elements:
            idx = elem.index
            if idx in signals and idx < self._num_elements:
                data[idx, :] = signals[idx][:min_len]

        # Compute sample covariance matrix
        R = (data @ data.conj().T) / min_len

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(R)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Noise subspace (smallest eigenvalues)
        noise_subspace = eigenvectors[:, n_sources:]

        # Compute MUSIC pseudo-spectrum
        spectrum = np.zeros(len(self._scan_azimuths))

        for i in range(len(self._scan_azimuths)):
            sv = self._steering_vectors[:, i]
            # MUSIC spectrum = 1 / |a^H * En * En^H * a|
            projection = noise_subspace.conj().T @ sv
            denominator = np.sum(np.abs(projection) ** 2)
            if denominator > 1e-20:
                spectrum[i] = 1.0 / denominator
            else:
                spectrum[i] = 1e20

        spectrum_db = 10 * np.log10(spectrum + 1e-20)

        # Find peaks
        sources = self._find_peaks(spectrum_db, n_sources, elevation)

        # Estimate number of sources from eigenvalues
        n_estimated = self._estimate_num_sources(eigenvalues)

        return MultiSourceDoAResult(
            sources=sources,
            num_sources_estimated=n_estimated,
            spectrum=spectrum_db,
            azimuths=self._scan_azimuths,
        )

    def _find_peaks(
        self, spectrum_db: np.ndarray, n_sources: int, elevation: float
    ) -> List[DoAResult]:
        """Find peaks in MUSIC spectrum."""
        sources = []

        # Simple peak finding
        spectrum_copy = spectrum_db.copy()

        for _ in range(n_sources):
            peak_idx = np.argmax(spectrum_copy)
            peak_power = spectrum_copy[peak_idx]

            # Check if peak is significant
            noise_floor = np.median(spectrum_db)
            if peak_power - noise_floor < 3.0:  # 3 dB threshold
                break

            azimuth = self._scan_azimuths[peak_idx]

            # Confidence from peak prominence
            prominence = peak_power - noise_floor
            confidence = min(1.0, prominence / 20)

            sources.append(
                DoAResult(
                    azimuth=azimuth,
                    elevation=elevation,
                    confidence=confidence,
                    power=peak_power,
                    method=DoAMethod.MUSIC,
                )
            )

            # Null out this peak for next iteration
            peak_width = max(3, len(spectrum_copy) // 20)
            start = max(0, peak_idx - peak_width)
            end = min(len(spectrum_copy), peak_idx + peak_width + 1)
            spectrum_copy[start:end] = -np.inf

        return sources

    def _estimate_num_sources(self, eigenvalues: np.ndarray) -> int:
        """Estimate number of sources from eigenvalues using MDL criterion."""
        n = len(eigenvalues)

        # Simple approach: count eigenvalues significantly above noise floor
        noise_floor = np.median(eigenvalues[n // 2 :])
        threshold = noise_floor * 10  # 10 dB above noise

        n_sources = 0
        for ev in eigenvalues:
            if ev > threshold:
                n_sources += 1
            else:
                break

        return max(1, min(n_sources, n - 1))

    def compute_spectrum(
        self,
        signals: Dict[int, np.ndarray],
        azimuths: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute MUSIC pseudo-spectrum.

        Args:
            signals: Dict mapping element index to signal
            azimuths: Optional custom azimuth angles

        Returns:
            Tuple of (azimuths, spectrum_db)
        """
        result = self.estimate(signals)
        if result.spectrum is not None and result.azimuths is not None:
            return result.azimuths, result.spectrum
        return self._scan_azimuths, np.zeros(len(self._scan_azimuths))

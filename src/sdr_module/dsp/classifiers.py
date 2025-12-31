"""
Signal classification for automatic modulation recognition.

Provides automatic detection of:
- Analog vs. digital signals
- Modulation type estimation
- Signal bandwidth estimation
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .demodulators import ModulationType


class SignalType(Enum):
    """High-level signal type."""

    UNKNOWN = "unknown"
    NOISE = "noise"
    ANALOG = "analog"
    DIGITAL = "digital"
    PULSED = "pulsed"


@dataclass
class ClassificationResult:
    """Result of signal classification."""

    signal_type: SignalType
    modulation: Optional[ModulationType]
    confidence: float  # 0.0 to 1.0
    bandwidth_hz: float
    center_offset_hz: float
    snr_db: float
    features: dict


class SignalClassifier:
    """
    Automatic signal classifier.

    Uses statistical features to identify signal characteristics
    and estimate modulation type.
    """

    def __init__(self, sample_rate: float):
        """
        Initialize classifier.

        Args:
            sample_rate: Sample rate in Hz
        """
        self._sample_rate = sample_rate

    def classify(self, samples: np.ndarray) -> ClassificationResult:
        """
        Classify signal from I/Q samples.

        Args:
            samples: Complex I/Q samples

        Returns:
            ClassificationResult with signal type and features
        """
        # Extract features
        features = self._extract_features(samples)

        # Estimate signal type
        signal_type = self._detect_signal_type(features)

        # Estimate modulation
        modulation = None
        if signal_type == SignalType.ANALOG:
            modulation = self._classify_analog(samples, features)
        elif signal_type == SignalType.DIGITAL:
            modulation = self._classify_digital(samples, features)

        # Estimate bandwidth
        bandwidth = self._estimate_bandwidth(samples)

        return ClassificationResult(
            signal_type=signal_type,
            modulation=modulation,
            confidence=features.get("confidence", 0.5),
            bandwidth_hz=bandwidth,
            center_offset_hz=features.get("center_offset", 0.0),
            snr_db=features.get("snr_db", 0.0),
            features=features,
        )

    def _extract_features(self, samples: np.ndarray) -> dict:
        """Extract statistical features from samples."""
        features = {}

        # Basic statistics
        magnitude = np.abs(samples)
        phase = np.angle(samples)

        features["mean_magnitude"] = np.mean(magnitude)
        features["std_magnitude"] = np.std(magnitude)
        features["max_magnitude"] = np.max(magnitude)

        # Normalized statistics
        norm_mag = magnitude / (np.max(magnitude) + 1e-10)
        features["crest_factor"] = np.max(norm_mag) / (
            np.sqrt(np.mean(norm_mag**2)) + 1e-10
        )

        # Phase statistics (unwrapped)
        phase_unwrapped = np.unwrap(phase)
        phase_diff = np.diff(phase_unwrapped)
        features["mean_phase_diff"] = np.mean(phase_diff)
        features["std_phase_diff"] = np.std(phase_diff)

        # Instantaneous frequency
        inst_freq = phase_diff * self._sample_rate / (2 * np.pi)
        features["mean_inst_freq"] = np.mean(inst_freq)
        features["std_inst_freq"] = np.std(inst_freq)

        # Spectral features
        spectrum = np.abs(np.fft.fft(samples))
        20 * np.log10(spectrum + 1e-10)

        features["spectral_flatness"] = self._spectral_flatness(spectrum)
        features["spectral_centroid"] = self._spectral_centroid(spectrum)

        # SNR estimation
        features["snr_db"] = self._estimate_snr(samples)

        # Kurtosis (measure of "peakedness")
        features["kurtosis"] = self._kurtosis(magnitude)

        return features

    def _detect_signal_type(self, features: dict) -> SignalType:
        """Detect high-level signal type from features."""
        snr = features.get("snr_db", 0)
        flatness = features.get("spectral_flatness", 1.0)
        kurtosis = features.get("kurtosis", 3.0)
        std_phase = features.get("std_phase_diff", 0)

        # Noise detection
        if snr < 3.0 and flatness > 0.8:
            return SignalType.NOISE

        # Pulsed signal detection
        if kurtosis > 10:
            return SignalType.PULSED

        # Digital vs analog
        # Digital signals tend to have discrete phase values
        if std_phase < 0.5:
            return SignalType.DIGITAL
        elif std_phase > 1.5:
            return SignalType.ANALOG

        # Default to unknown
        return SignalType.UNKNOWN

    def _classify_analog(
        self, samples: np.ndarray, features: dict
    ) -> Optional[ModulationType]:
        """Classify analog modulation type."""
        std_mag = features.get("std_magnitude", 0)
        std_freq = features.get("std_inst_freq", 0)
        mean_mag = features.get("mean_magnitude", 0)

        # AM: varying amplitude, stable frequency
        if std_mag / (mean_mag + 1e-10) > 0.3 and std_freq < 1000:
            return ModulationType.AM

        # FM: stable amplitude, varying frequency
        if std_mag / (mean_mag + 1e-10) < 0.1 and std_freq > 1000:
            return ModulationType.FM

        # SSB: check for asymmetric spectrum
        spectrum = np.abs(np.fft.fft(samples))
        n = len(spectrum)
        lower_power: float = float(np.sum(spectrum[: n // 2] ** 2))
        upper_power: float = float(np.sum(spectrum[n // 2 :] ** 2))

        if upper_power > 2 * lower_power:
            return ModulationType.USB
        elif lower_power > 2 * upper_power:
            return ModulationType.LSB

        return None

    def _classify_digital(
        self, samples: np.ndarray, features: dict
    ) -> Optional[ModulationType]:
        """Classify digital modulation type."""
        # Constellation analysis
        magnitude = np.abs(samples)
        phase = np.angle(samples)

        # Normalize magnitude
        norm_mag = magnitude / (np.max(magnitude) + 1e-10)

        # Check for OOK/ASK: amplitude levels
        # Use fixed range to handle constant-magnitude signals
        mag_hist, _ = np.histogram(norm_mag, bins=20, range=(0, 1))
        n_peaks: int = int(np.sum(mag_hist > len(samples) * 0.05))

        if n_peaks <= 2:
            # Binary amplitude -> OOK
            return ModulationType.OOK

        # Check for FSK: frequency levels
        phase_diff = np.diff(np.unwrap(phase))
        freq_hist, _ = np.histogram(phase_diff, bins=20)
        n_freq_peaks: int = int(np.sum(freq_hist > len(samples) * 0.05))

        if n_freq_peaks == 2:
            return ModulationType.FSK

        # Check for PSK: phase levels
        phase_normalized = np.mod(phase + np.pi, 2 * np.pi) - np.pi
        phase_hist, _ = np.histogram(phase_normalized, bins=16)
        n_phase_peaks: int = int(np.sum(phase_hist > len(samples) * 0.03))

        if n_phase_peaks == 2:
            return ModulationType.BPSK
        elif n_phase_peaks == 4:
            return ModulationType.QPSK

        return None

    def _spectral_flatness(self, spectrum: np.ndarray) -> float:
        """Calculate spectral flatness (Wiener entropy)."""
        spectrum = spectrum + 1e-10  # Avoid log(0)
        geometric_mean = np.exp(np.mean(np.log(spectrum)))
        arithmetic_mean = np.mean(spectrum)
        return geometric_mean / (arithmetic_mean + 1e-10)

    def _spectral_centroid(self, spectrum: np.ndarray) -> float:
        """Calculate spectral centroid."""
        freqs = np.arange(len(spectrum))
        return np.sum(freqs * spectrum) / (np.sum(spectrum) + 1e-10)

    def _estimate_snr(self, samples: np.ndarray) -> float:
        """Estimate SNR from samples."""
        spectrum = np.abs(np.fft.fft(samples)) ** 2

        # Find signal and noise regions
        sorted_power = np.sort(spectrum)
        noise_floor = np.mean(sorted_power[: len(sorted_power) // 4])
        signal_power = np.mean(sorted_power[-len(sorted_power) // 4 :])

        snr_linear = signal_power / (noise_floor + 1e-10)
        return 10 * np.log10(snr_linear + 1e-10)

    def _estimate_bandwidth(self, samples: np.ndarray) -> float:
        """Estimate occupied bandwidth."""
        spectrum = np.abs(np.fft.fft(samples)) ** 2
        spectrum = np.fft.fftshift(spectrum)

        total_power: float = float(np.sum(spectrum))
        cumsum = np.cumsum(spectrum) / total_power

        # Find 99% power bandwidth
        lower_idx = np.searchsorted(cumsum, 0.005)
        upper_idx = np.searchsorted(cumsum, 0.995)

        bandwidth_bins = upper_idx - lower_idx
        bandwidth_hz = bandwidth_bins * self._sample_rate / len(spectrum)

        return bandwidth_hz

    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std < 1e-10:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

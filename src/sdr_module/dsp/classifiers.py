"""
Signal classification for automatic modulation recognition.

Provides automatic detection of:
- Analog vs. digital signals
- Modulation type estimation
- Signal bandwidth estimation
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

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
    features: Dict[str, Any]


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

        confidence = self._compute_confidence(signal_type, modulation, features)

        return ClassificationResult(
            signal_type=signal_type,
            modulation=modulation,
            confidence=confidence,
            bandwidth_hz=bandwidth,
            center_offset_hz=features.get("center_offset", 0.0),
            snr_db=features.get("snr_db", 0.0),
            features=features,
        )

    def _compute_confidence(
        self,
        signal_type: SignalType,
        modulation: Optional[ModulationType],
        features: Dict[str, Any],
    ) -> float:
        """Estimate a classification confidence in [0, 1].

        This is a heuristic, not a calibrated probability: it grows with SNR
        and with how specific the result is (a concrete modulation is more
        trustworthy than a bare DIGITAL/ANALOG label, which beats UNKNOWN).
        For NOISE it reports confidence that the input *is* noise — a flat
        spectrum with low SNR. The previous code returned a constant 0.5 for
        every input.
        """
        snr = float(features.get("snr_db", 0.0))
        # 3 dB -> 0, 30 dB -> 1.
        snr_quality = float(np.clip((snr - 3.0) / 27.0, 0.0, 1.0))
        flatness = float(np.clip(features.get("spectral_flatness", 1.0), 0.0, 1.0))

        if signal_type == SignalType.NOISE:
            return float(np.clip(0.5 * flatness + 0.5 * (1.0 - snr_quality), 0.0, 1.0))

        if modulation is not None:
            decisiveness = 1.0
        elif signal_type in (
            SignalType.DIGITAL,
            SignalType.ANALOG,
            SignalType.PULSED,
        ):
            decisiveness = 0.6
        else:  # UNKNOWN
            decisiveness = 0.3

        return float(np.clip(snr_quality * decisiveness, 0.0, 1.0))

    def _extract_features(self, samples: np.ndarray) -> Dict[str, Any]:
        """Extract statistical features from samples."""
        features: Dict[str, Any] = {}

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

        # Amplitude coefficient of variation: ~0 for constant-envelope signals
        # (FM/FSK/PSK), moderate for noise (Rayleigh ~0.52) and AM, ~1 for OOK.
        mean_mag = float(np.mean(magnitude))
        features["amp_cv"] = float(np.std(magnitude) / (mean_mag + 1e-12))

        # Fraction of samples near zero amplitude (on/off keying signature).
        norm_mag_full = magnitude / (np.max(magnitude) + 1e-12)
        features["frac_low"] = float(np.mean(norm_mag_full < 0.25))

        # Spectral features
        spectrum = np.abs(np.fft.fft(samples))
        psd = spectrum**2

        features["spectral_flatness"] = self._spectral_flatness(spectrum)
        features["spectral_centroid"] = self._spectral_centroid(spectrum)
        # Peak-to-median PSD ratio: high when a dominant carrier is present
        # (AM/FM/OOK-on), low for noise and constant-envelope wideband PSK.
        features["peak_ratio"] = float(np.max(psd) / (np.median(psd) + 1e-12))

        # Bimodality of the amplitude-weighted instantaneous frequency: high
        # for 2-FSK (two discrete tones), lower for continuous FM.
        features["freq_bimodality"] = self._freq_bimodality(samples)

        # SNR estimation
        features["snr_db"] = self._estimate_snr(samples)

        # Kurtosis (measure of "peakedness")
        features["kurtosis"] = self._kurtosis(magnitude)

        return features

    def _freq_bimodality(self, samples: np.ndarray) -> float:
        """Bimodality coefficient of the amplitude-weighted instantaneous
        frequency. Values above ~0.8 indicate two discrete tones (FSK) rather
        than a continuously varying frequency (FM)."""
        magnitude = np.abs(samples)
        if len(samples) < 4:
            return 0.0
        inst = np.diff(np.unwrap(np.angle(samples)))
        weight = (magnitude[1:] / (np.max(magnitude) + 1e-12)) ** 2
        wsum = float(np.sum(weight)) + 1e-12
        mu = float(np.sum(inst * weight) / wsum)
        centered = inst - mu
        m2 = float(np.sum(weight * centered**2) / wsum)
        if m2 < 1e-18:
            return 0.0
        m3 = float(np.sum(weight * centered**3) / wsum)
        m4 = float(np.sum(weight * centered**4) / wsum)
        skew = m3 / (m2**1.5)
        kurt = m4 / (m2**2)
        return float((skew * skew + 1.0) / (kurt + 1e-12))

    def _detect_signal_type(self, features: Dict[str, Any]) -> SignalType:
        """Detect the high-level signal type from features.

        The previous version keyed off ``std_phase_diff`` alone, which is
        backwards for these signals (AM has near-constant phase and read as
        "digital"; BPSK's pi phase jumps read as "analog"). This uses the
        amplitude envelope, spectral flatness and carrier strength instead:

        - constant envelope (``amp_cv`` ~ 0): FM/FSK (narrowband) or PSK
          (wideband) -> ANALOG for continuous FM, otherwise DIGITAL;
        - amplitude varies: on/off keying -> DIGITAL, else AM/SSB -> ANALOG;
        - broadband with no dominant carrier and Rayleigh amplitude -> NOISE.
        """
        flatness = float(features.get("spectral_flatness", 1.0))
        kurtosis = float(features.get("kurtosis", 3.0))
        amp_cv = float(features.get("amp_cv", 0.0))
        peak_ratio = float(features.get("peak_ratio", 0.0))
        frac_low = float(features.get("frac_low", 0.0))
        bimodality = float(features.get("freq_bimodality", 0.0))

        # Impulsive / pulsed: a very peaky amplitude distribution.
        if kurtosis > 10:
            return SignalType.PULSED

        # Broadband noise: flat spectrum, no dominant carrier, and an
        # amplitude spread characteristic of complex Gaussian noise (Rayleigh
        # envelope has std/mean ~ 0.52), which excludes both constant-envelope
        # signals (cv ~ 0) and on/off keying (cv ~ 1).
        if flatness > 0.6 and peak_ratio < 50.0 and 0.3 < amp_cv < 0.8:
            return SignalType.NOISE

        if amp_cv < 0.2:
            # Constant envelope. Wideband -> phase-shift keying (digital);
            # narrowband -> frequency modulation, digital (FSK, two discrete
            # tones) if bimodal, else analog FM.
            if flatness >= 0.6:
                return SignalType.DIGITAL
            return SignalType.DIGITAL if bimodality > 0.8 else SignalType.ANALOG

        # Amplitude varies: on/off keying is digital, continuous AM is analog.
        if frac_low > 0.2:
            return SignalType.DIGITAL
        return SignalType.ANALOG

    def _classify_analog(
        self, samples: np.ndarray, features: Dict[str, Any]
    ) -> Optional[ModulationType]:
        """Classify analog modulation type (reached only for ANALOG signals)."""
        amp_cv = float(features.get("amp_cv", 0.0))

        # Constant envelope analog -> FM.
        if amp_cv < 0.2:
            return ModulationType.FM

        # Amplitude-modulated analog: AM (double sideband + carrier, symmetric)
        # vs SSB (one sideband). Compare the two halves of the shifted power
        # spectrum, excluding the central carrier bins.
        psd = np.fft.fftshift(np.abs(np.fft.fft(samples)) ** 2)
        n = len(psd)
        center = n // 2
        guard = max(1, n // 64)
        lower_power = float(np.sum(psd[: center - guard]))
        upper_power = float(np.sum(psd[center + guard :]))

        if upper_power > 3 * lower_power:
            return ModulationType.USB
        if lower_power > 3 * upper_power:
            return ModulationType.LSB
        return ModulationType.AM

    def _classify_digital(
        self, samples: np.ndarray, features: Dict[str, Any]
    ) -> Optional[ModulationType]:
        """Classify digital modulation type (reached only for DIGITAL signals)."""
        amp_cv = float(features.get("amp_cv", 0.0))
        flatness = float(features.get("spectral_flatness", 1.0))

        # Amplitude varies -> on/off keying.
        if amp_cv >= 0.2:
            return ModulationType.OOK

        # Constant envelope, narrowband -> frequency-shift keying.
        if flatness < 0.6:
            return ModulationType.FSK

        # Constant envelope, wideband -> phase-shift keying. Identify the order
        # with the M-th-power nonlinearity, which is robust to noise: for M-PSK
        # the M-th power of the unit-magnitude signal collapses onto a single
        # point, so |mean((x/|x|)^M)| ~ 1. BPSK collapses at M=2, QPSK at M=4.
        unit = samples / (np.abs(samples) + 1e-12)
        c2 = float(np.abs(np.mean(unit**2)))
        c4 = float(np.abs(np.mean(unit**4)))
        if c2 > 0.5:
            return ModulationType.BPSK
        if c4 > 0.5:
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

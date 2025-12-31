"""
Spectrum analyzer for FFT-based frequency analysis.

Provides real-time spectrum display capabilities including:
- Configurable FFT size and windowing
- Multiple averaging modes
- Power spectral density calculation
- Peak detection
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


class WindowType(Enum):
    """FFT window functions."""

    RECTANGULAR = "rectangular"
    HANN = "hann"
    HAMMING = "hamming"
    BLACKMAN = "blackman"
    BLACKMAN_HARRIS = "blackman-harris"
    FLAT_TOP = "flat-top"


class AveragingMode(Enum):
    """Spectrum averaging modes."""

    NONE = "none"
    RMS = "rms"
    PEAK_HOLD = "peak_hold"
    MIN_HOLD = "min_hold"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


@dataclass
class SpectrumResult:
    """Result of spectrum analysis."""

    frequencies: np.ndarray  # Frequency bins in Hz
    power_db: np.ndarray  # Power in dB
    center_freq: float  # Center frequency in Hz
    sample_rate: float  # Sample rate in Hz
    fft_size: int  # FFT size used
    rbw: float  # Resolution bandwidth in Hz


@dataclass
class Peak:
    """Detected spectral peak."""

    frequency: float  # Frequency in Hz
    power_db: float  # Power in dB
    bin_index: int  # FFT bin index


class SpectrumAnalyzer:
    """
    FFT-based spectrum analyzer.

    Computes power spectral density from I/Q samples with
    configurable windowing, averaging, and peak detection.
    """

    def __init__(
        self,
        fft_size: int = 4096,
        window: WindowType = WindowType.HANN,
        averaging: AveragingMode = AveragingMode.RMS,
        avg_count: int = 10,
        overlap: float = 0.5,
    ):
        """
        Initialize spectrum analyzer.

        Args:
            fft_size: FFT size (power of 2)
            window: Window function type
            averaging: Averaging mode
            avg_count: Number of FFTs to average
            overlap: Overlap ratio (0.0 to 0.9)
        """
        self._fft_size = fft_size
        self._window_type = window
        self._averaging_mode = averaging
        self._avg_count = avg_count
        self._overlap = min(0.9, max(0.0, overlap))

        # Precompute window
        self._window = self._create_window(fft_size, window)
        self._window_gain = np.sum(self._window) ** 2

        # Averaging state
        self._avg_buffer: Optional[np.ndarray] = None
        self._avg_index = 0
        self._peak_hold: Optional[np.ndarray] = None
        self._min_hold: Optional[np.ndarray] = None
        self._exp_avg: Optional[np.ndarray] = None
        self._exp_alpha = 0.1  # Exponential averaging factor

        # Last result cache
        self._center_freq = 0.0
        self._sample_rate = 1.0

    @property
    def fft_size(self) -> int:
        """Get FFT size."""
        return self._fft_size

    @fft_size.setter
    def fft_size(self, size: int) -> None:
        """Set FFT size."""
        self._fft_size = size
        self._window = self._create_window(size, self._window_type)
        self._window_gain = np.sum(self._window) ** 2
        self._reset_averaging()

    @property
    def resolution_bandwidth(self) -> float:
        """Get resolution bandwidth in Hz."""
        return self._sample_rate / self._fft_size

    def _create_window(self, size: int, window_type: WindowType) -> np.ndarray:
        """Create window function."""
        if window_type == WindowType.RECTANGULAR:
            return np.ones(size)
        elif window_type == WindowType.HANN:
            return np.hanning(size)
        elif window_type == WindowType.HAMMING:
            return np.hamming(size)
        elif window_type == WindowType.BLACKMAN:
            return np.blackman(size)
        elif window_type == WindowType.BLACKMAN_HARRIS:
            # 4-term Blackman-Harris
            n = np.arange(size)
            a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
            return (
                a0
                - a1 * np.cos(2 * np.pi * n / size)
                + a2 * np.cos(4 * np.pi * n / size)
                - a3 * np.cos(6 * np.pi * n / size)
            )
        elif window_type == WindowType.FLAT_TOP:
            n = np.arange(size)
            a0, a1, a2, a3, a4 = (
                0.21557895,
                0.41663158,
                0.277263158,
                0.083578947,
                0.006947368,
            )
            return (
                a0
                - a1 * np.cos(2 * np.pi * n / size)
                + a2 * np.cos(4 * np.pi * n / size)
                - a3 * np.cos(6 * np.pi * n / size)
                + a4 * np.cos(8 * np.pi * n / size)
            )
        else:
            return np.hanning(size)

    def _reset_averaging(self) -> None:
        """Reset averaging buffers."""
        self._avg_buffer = None
        self._avg_index = 0
        self._peak_hold = None
        self._min_hold = None
        self._exp_avg = None

    def set_averaging(self, mode: AveragingMode, count: int = 10) -> None:
        """Configure averaging mode."""
        self._averaging_mode = mode
        self._avg_count = count
        self._reset_averaging()

    def compute_spectrum(
        self, samples: np.ndarray, center_freq: float = 0.0, sample_rate: float = 1.0
    ) -> SpectrumResult:
        """
        Compute power spectrum from I/Q samples.

        Args:
            samples: Complex I/Q samples
            center_freq: Center frequency in Hz
            sample_rate: Sample rate in Hz

        Returns:
            SpectrumResult with frequency and power arrays
        """
        self._center_freq = center_freq
        self._sample_rate = sample_rate

        # Process in FFT-sized chunks
        n_samples = len(samples)
        hop_size = int(self._fft_size * (1 - self._overlap))
        n_ffts = max(1, (n_samples - self._fft_size) // hop_size + 1)

        # Accumulate power spectra
        power_sum = np.zeros(self._fft_size)

        for i in range(n_ffts):
            start = i * hop_size
            end = start + self._fft_size

            if end > n_samples:
                break

            chunk = samples[start:end]

            # Apply window and compute FFT
            windowed = chunk * self._window
            spectrum = np.fft.fftshift(np.fft.fft(windowed))

            # Compute power (magnitude squared)
            power = np.abs(spectrum) ** 2

            power_sum += power

        # Average
        if n_ffts > 0:
            power_avg = power_sum / n_ffts
        else:
            power_avg = power_sum

        # Normalize by window gain
        power_avg /= self._window_gain

        # Apply averaging mode
        power_db = self._apply_averaging(power_avg)

        # Generate frequency axis
        frequencies = (
            np.fft.fftshift(np.fft.fftfreq(self._fft_size, 1.0 / sample_rate))
            + center_freq
        )

        return SpectrumResult(
            frequencies=frequencies,
            power_db=power_db,
            center_freq=center_freq,
            sample_rate=sample_rate,
            fft_size=self._fft_size,
            rbw=sample_rate / self._fft_size,
        )

    def _apply_averaging(self, power_linear: np.ndarray) -> np.ndarray:
        """Apply averaging mode and convert to dB."""
        if self._averaging_mode == AveragingMode.NONE:
            return 10 * np.log10(power_linear + 1e-20)

        elif self._averaging_mode == AveragingMode.RMS:
            if self._avg_buffer is None:
                self._avg_buffer = np.zeros((self._avg_count, self._fft_size))

            self._avg_buffer[self._avg_index % self._avg_count] = power_linear
            self._avg_index += 1

            count = min(self._avg_index, self._avg_count)
            avg_power = np.mean(self._avg_buffer[:count], axis=0)
            return 10 * np.log10(avg_power + 1e-20)

        elif self._averaging_mode == AveragingMode.PEAK_HOLD:
            if self._peak_hold is None:
                self._peak_hold = power_linear.copy()
            else:
                self._peak_hold = np.maximum(self._peak_hold, power_linear)
            return 10 * np.log10(self._peak_hold + 1e-20)

        elif self._averaging_mode == AveragingMode.MIN_HOLD:
            if self._min_hold is None:
                self._min_hold = power_linear.copy()
            else:
                self._min_hold = np.minimum(self._min_hold, power_linear)
            return 10 * np.log10(self._min_hold + 1e-20)

        elif self._averaging_mode == AveragingMode.LINEAR:
            if self._avg_buffer is None:
                self._avg_buffer = np.zeros((self._avg_count, self._fft_size))

            self._avg_buffer[self._avg_index % self._avg_count] = power_linear
            self._avg_index += 1

            count = min(self._avg_index, self._avg_count)
            avg_power = np.mean(self._avg_buffer[:count], axis=0)
            return 10 * np.log10(avg_power + 1e-20)

        elif self._averaging_mode == AveragingMode.EXPONENTIAL:
            if self._exp_avg is None:
                self._exp_avg = power_linear.copy()
            else:
                self._exp_avg = (
                    self._exp_alpha * power_linear
                    + (1 - self._exp_alpha) * self._exp_avg
                )
            return 10 * np.log10(self._exp_avg + 1e-20)

        else:
            return 10 * np.log10(power_linear + 1e-20)

    def find_peaks(
        self,
        result: SpectrumResult,
        threshold_db: float = -60,
        min_distance_hz: float = 1000,
    ) -> List[Peak]:
        """
        Find peaks in spectrum.

        Args:
            result: Spectrum analysis result
            threshold_db: Minimum power threshold
            min_distance_hz: Minimum distance between peaks

        Returns:
            List of detected peaks
        """
        peaks = []
        power_db = result.power_db
        frequencies = result.frequencies

        min_distance_bins = int(min_distance_hz / result.rbw)

        # Simple peak detection
        for i in range(1, len(power_db) - 1):
            if power_db[i] > threshold_db:
                if power_db[i] > power_db[i - 1] and power_db[i] > power_db[i + 1]:
                    # Check minimum distance from existing peaks
                    too_close = False
                    for p in peaks:
                        if abs(i - p.bin_index) < min_distance_bins:
                            if power_db[i] > p.power_db:
                                peaks.remove(p)
                            else:
                                too_close = True
                            break

                    if not too_close:
                        peaks.append(
                            Peak(
                                frequency=frequencies[i],
                                power_db=power_db[i],
                                bin_index=i,
                            )
                        )

        # Sort by power
        peaks.sort(key=lambda p: p.power_db, reverse=True)
        return peaks

    def reset(self) -> None:
        """Reset analyzer state and averaging."""
        self._reset_averaging()

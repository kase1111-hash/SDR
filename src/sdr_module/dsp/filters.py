"""
Digital filter implementations for SDR signal processing.

Provides various filter types for signal conditioning:
- Low-pass, high-pass, band-pass, band-stop
- FIR filter design using windowed-sinc method
- Real-time filtering with overlap-save
"""

import numpy as np
from typing import Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class FilterType(Enum):
    """Filter response types."""
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"
    BANDPASS = "bandpass"
    BANDSTOP = "bandstop"


@dataclass
class FilterSpec:
    """Filter specification."""
    filter_type: FilterType
    cutoff_low: float      # Lower cutoff frequency (Hz)
    cutoff_high: float     # Upper cutoff frequency (Hz)
    sample_rate: float     # Sample rate (Hz)
    num_taps: int          # Number of filter taps
    window: str = "hamming"  # Window function


class FIRFilter:
    """
    FIR filter implementation.

    Uses windowed-sinc design for filter coefficient generation
    and overlap-save for efficient convolution.
    """

    def __init__(self, spec: FilterSpec):
        """
        Initialize FIR filter.

        Args:
            spec: Filter specification
        """
        self._spec = spec
        self._taps = self._design_filter(spec)
        self._buffer: Optional[np.ndarray] = None

    @property
    def taps(self) -> np.ndarray:
        """Get filter coefficients."""
        return self._taps.copy()

    @property
    def num_taps(self) -> int:
        """Get number of filter taps."""
        return len(self._taps)

    @property
    def delay_samples(self) -> int:
        """Get filter group delay in samples."""
        return (len(self._taps) - 1) // 2

    def _design_filter(self, spec: FilterSpec) -> np.ndarray:
        """Design filter using windowed-sinc method."""
        n = spec.num_taps
        fs = spec.sample_rate

        # Normalize frequencies
        fc_low = spec.cutoff_low / fs
        fc_high = spec.cutoff_high / fs

        # Time axis
        t = np.arange(n) - (n - 1) / 2
        t[t == 0] = 1e-10  # Avoid division by zero

        if spec.filter_type == FilterType.LOWPASS:
            h = 2 * fc_high * np.sinc(2 * fc_high * t)

        elif spec.filter_type == FilterType.HIGHPASS:
            h = np.sinc(t) - 2 * fc_low * np.sinc(2 * fc_low * t)

        elif spec.filter_type == FilterType.BANDPASS:
            h = (2 * fc_high * np.sinc(2 * fc_high * t) -
                 2 * fc_low * np.sinc(2 * fc_low * t))

        elif spec.filter_type == FilterType.BANDSTOP:
            h = (np.sinc(t) -
                 2 * fc_high * np.sinc(2 * fc_high * t) +
                 2 * fc_low * np.sinc(2 * fc_low * t))

        else:
            h = np.zeros(n)
            h[(n - 1) // 2] = 1.0  # Pass-through

        # Apply window
        window = self._get_window(spec.window, n)
        h *= window

        # Normalize
        h /= np.sum(h)

        return h

    def _get_window(self, window_type: str, length: int) -> np.ndarray:
        """Get window function."""
        window_type = window_type.lower()
        if window_type == "hamming":
            return np.hamming(length)
        elif window_type == "hann" or window_type == "hanning":
            return np.hanning(length)
        elif window_type == "blackman":
            return np.blackman(length)
        elif window_type == "kaiser":
            return np.kaiser(length, 8.0)
        else:
            return np.ones(length)

    def filter(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply filter to samples.

        Args:
            samples: Input samples (real or complex)

        Returns:
            Filtered samples
        """
        return np.convolve(samples, self._taps, mode='same')

    def filter_stream(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply filter with state preservation for streaming.

        Uses overlap-save method for efficient streaming.

        Args:
            samples: Input samples

        Returns:
            Filtered samples
        """
        n_taps = len(self._taps)

        # Initialize buffer if needed
        if self._buffer is None:
            self._buffer = np.zeros(n_taps - 1, dtype=samples.dtype)

        # Prepend buffer
        padded = np.concatenate([self._buffer, samples])

        # Convolve
        filtered = np.convolve(padded, self._taps, mode='valid')

        # Update buffer
        self._buffer = samples[-(n_taps - 1):]

        return filtered

    def reset(self) -> None:
        """Reset filter state."""
        self._buffer = None

    def frequency_response(
        self,
        n_points: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute frequency response.

        Args:
            n_points: Number of frequency points

        Returns:
            Tuple of (frequencies, magnitude_db)
        """
        freqs = np.fft.rfftfreq(n_points, 1 / self._spec.sample_rate)
        response = np.fft.rfft(self._taps, n_points)
        magnitude_db = 20 * np.log10(np.abs(response) + 1e-20)
        return freqs, magnitude_db


class FilterBank:
    """
    Collection of filters for multi-band processing.
    """

    def __init__(self, sample_rate: float):
        """
        Initialize filter bank.

        Args:
            sample_rate: Sample rate in Hz
        """
        self._sample_rate = sample_rate
        self._filters = {}

    def add_filter(self, name: str, filter_obj: FIRFilter) -> None:
        """Add a filter to the bank."""
        self._filters[name] = filter_obj

    def create_lowpass(
        self,
        name: str,
        cutoff: float,
        num_taps: int = 101
    ) -> FIRFilter:
        """Create and add a lowpass filter."""
        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=cutoff,
            sample_rate=self._sample_rate,
            num_taps=num_taps
        )
        filt = FIRFilter(spec)
        self._filters[name] = filt
        return filt

    def create_highpass(
        self,
        name: str,
        cutoff: float,
        num_taps: int = 101
    ) -> FIRFilter:
        """Create and add a highpass filter."""
        spec = FilterSpec(
            filter_type=FilterType.HIGHPASS,
            cutoff_low=cutoff,
            cutoff_high=self._sample_rate / 2,
            sample_rate=self._sample_rate,
            num_taps=num_taps
        )
        filt = FIRFilter(spec)
        self._filters[name] = filt
        return filt

    def create_bandpass(
        self,
        name: str,
        low_cutoff: float,
        high_cutoff: float,
        num_taps: int = 101
    ) -> FIRFilter:
        """Create and add a bandpass filter."""
        spec = FilterSpec(
            filter_type=FilterType.BANDPASS,
            cutoff_low=low_cutoff,
            cutoff_high=high_cutoff,
            sample_rate=self._sample_rate,
            num_taps=num_taps
        )
        filt = FIRFilter(spec)
        self._filters[name] = filt
        return filt

    def get_filter(self, name: str) -> Optional[FIRFilter]:
        """Get filter by name."""
        return self._filters.get(name)

    def apply(self, name: str, samples: np.ndarray) -> np.ndarray:
        """Apply named filter to samples."""
        filt = self._filters.get(name)
        if filt is None:
            raise ValueError(f"Filter '{name}' not found")
        return filt.filter(samples)

    def reset_all(self) -> None:
        """Reset all filter states."""
        for filt in self._filters.values():
            filt.reset()

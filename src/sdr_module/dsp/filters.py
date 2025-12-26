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


class Decimator:
    """
    Sample rate decimation with anti-aliasing filtering.

    Reduces sample rate by an integer factor while preventing
    aliasing through lowpass filtering.

    Features:
    - Configurable decimation factor
    - Built-in anti-aliasing lowpass filter
    - Polyphase implementation for efficiency
    - Multi-stage decimation for large factors
    - Supports real and complex signals
    """

    def __init__(
        self,
        input_rate: float,
        decimation_factor: int,
        num_taps: int = 0,
        cutoff_ratio: float = 0.8,
        window: str = "kaiser"
    ):
        """
        Initialize decimator.

        Args:
            input_rate: Input sample rate in Hz
            decimation_factor: Decimation factor (integer >= 1)
            num_taps: Number of filter taps (0 = auto-calculate)
            cutoff_ratio: Filter cutoff as ratio of output Nyquist (0.0-1.0)
            window: Window function for filter design
        """
        if decimation_factor < 1:
            raise ValueError("Decimation factor must be >= 1")

        self._input_rate = input_rate
        self._factor = decimation_factor
        self._output_rate = input_rate / decimation_factor
        self._cutoff_ratio = cutoff_ratio
        self._window = window

        # Calculate filter parameters
        nyquist_out = self._output_rate / 2
        self._cutoff = nyquist_out * cutoff_ratio

        # Auto-calculate taps if not specified
        if num_taps <= 0:
            # Rule of thumb: more taps for larger decimation
            # At least 4 taps per decimation factor
            num_taps = max(31, decimation_factor * 8 + 1)
            # Make odd for symmetric filter
            if num_taps % 2 == 0:
                num_taps += 1
        self._num_taps = num_taps

        # Design anti-aliasing filter
        self._filter = self._design_filter()

        # State buffer for streaming
        self._buffer: Optional[np.ndarray] = None
        self._phase = 0  # Current decimation phase

    def _design_filter(self) -> np.ndarray:
        """Design anti-aliasing lowpass filter."""
        n = self._num_taps
        fc = self._cutoff / self._input_rate  # Normalized cutoff

        # Time axis centered at 0
        t = np.arange(n) - (n - 1) / 2
        t[t == 0] = 1e-10

        # Sinc function
        h = 2 * fc * np.sinc(2 * fc * t)

        # Apply window
        if self._window.lower() == "kaiser":
            # Kaiser with beta=8 for good stopband
            window = np.kaiser(n, 8.0)
        elif self._window.lower() == "blackman":
            window = np.blackman(n)
        elif self._window.lower() == "hamming":
            window = np.hamming(n)
        else:
            window = np.hanning(n)

        h *= window

        # Normalize for unity gain at DC
        h /= np.sum(h)

        return h.astype(np.float32)

    @property
    def input_rate(self) -> float:
        """Get input sample rate."""
        return self._input_rate

    @property
    def output_rate(self) -> float:
        """Get output sample rate."""
        return self._output_rate

    @property
    def factor(self) -> int:
        """Get decimation factor."""
        return self._factor

    @property
    def num_taps(self) -> int:
        """Get number of filter taps."""
        return self._num_taps

    @property
    def filter_coefficients(self) -> np.ndarray:
        """Get filter coefficients."""
        return self._filter.copy()

    @property
    def group_delay(self) -> float:
        """Get group delay in seconds."""
        return (self._num_taps - 1) / 2 / self._input_rate

    def decimate(self, samples: np.ndarray) -> np.ndarray:
        """
        Decimate samples (one-shot, no state).

        Args:
            samples: Input samples (real or complex)

        Returns:
            Decimated samples at output rate
        """
        if self._factor == 1:
            return samples.copy()

        # Apply anti-aliasing filter
        filtered = np.convolve(samples, self._filter, mode='same')

        # Downsample by taking every Nth sample
        decimated = filtered[::self._factor]

        return decimated

    def decimate_stream(self, samples: np.ndarray) -> np.ndarray:
        """
        Decimate with state preservation for streaming.

        Args:
            samples: Input samples

        Returns:
            Decimated samples
        """
        if self._factor == 1:
            return samples.copy()

        n_taps = self._num_taps

        # Initialize buffer if needed
        if self._buffer is None:
            self._buffer = np.zeros(n_taps - 1, dtype=samples.dtype)

        # Prepend buffer
        padded = np.concatenate([self._buffer, samples])

        # Filter
        filtered = np.convolve(padded, self._filter, mode='valid')

        # Update buffer
        self._buffer = samples[-(n_taps - 1):]

        # Decimate with phase tracking
        start_idx = (self._factor - self._phase) % self._factor
        decimated = filtered[start_idx::self._factor]

        # Update phase
        self._phase = (self._phase + len(filtered)) % self._factor

        return decimated

    def decimate_polyphase(self, samples: np.ndarray) -> np.ndarray:
        """
        Polyphase decimation (more efficient for large factors).

        Splits filter into polyphase components and only
        computes outputs that will be kept.

        Args:
            samples: Input samples

        Returns:
            Decimated samples
        """
        if self._factor == 1:
            return samples.copy()

        # Create polyphase filter bank
        n_phases = self._factor
        n_taps_per_phase = (len(self._filter) + n_phases - 1) // n_phases

        # Pad filter to multiple of factor
        filter_padded = np.zeros(n_taps_per_phase * n_phases, dtype=self._filter.dtype)
        filter_padded[:len(self._filter)] = self._filter

        # Reshape into polyphase components (time-reversed for convolution)
        polyphase = filter_padded.reshape(n_taps_per_phase, n_phases).T

        # Pad samples to multiple of decimation factor
        n_samples = len(samples)
        pad_len = (n_phases - (n_samples % n_phases)) % n_phases
        if pad_len > 0:
            samples = np.concatenate([samples, np.zeros(pad_len, dtype=samples.dtype)])

        # Reshape samples into polyphase components
        n_output = len(samples) // n_phases
        samples_poly = samples.reshape(n_output, n_phases).T

        # Compute each polyphase output
        output = np.zeros(n_output, dtype=samples.dtype)
        for phase in range(n_phases):
            phase_filter = polyphase[phase]
            phase_signal = samples_poly[phase]

            # Convolve and add
            if len(phase_filter) > 0 and len(phase_signal) > 0:
                contribution = np.convolve(phase_signal, phase_filter, mode='full')
                output += contribution[:n_output]

        return output

    def get_frequency_response(
        self,
        n_points: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get filter frequency response.

        Args:
            n_points: Number of frequency points

        Returns:
            (frequencies_hz, magnitude_db)
        """
        freqs = np.fft.rfftfreq(n_points, 1 / self._input_rate)
        response = np.fft.rfft(self._filter, n_points)
        magnitude_db = 20 * np.log10(np.abs(response) + 1e-20)
        return freqs, magnitude_db

    def reset(self) -> None:
        """Reset decimator state."""
        self._buffer = None
        self._phase = 0


class Interpolator:
    """
    Sample rate interpolation with anti-imaging filtering.

    Increases sample rate by an integer factor with proper
    anti-imaging filtering.

    Features:
    - Configurable interpolation factor
    - Built-in anti-imaging lowpass filter
    - Zero-stuffing with filtering
    - Supports real and complex signals
    """

    def __init__(
        self,
        input_rate: float,
        interpolation_factor: int,
        num_taps: int = 0,
        cutoff_ratio: float = 0.8,
        window: str = "kaiser"
    ):
        """
        Initialize interpolator.

        Args:
            input_rate: Input sample rate in Hz
            interpolation_factor: Interpolation factor (integer >= 1)
            num_taps: Number of filter taps (0 = auto)
            cutoff_ratio: Filter cutoff as ratio of input Nyquist (0.0-1.0)
            window: Window function for filter design
        """
        if interpolation_factor < 1:
            raise ValueError("Interpolation factor must be >= 1")

        self._input_rate = input_rate
        self._factor = interpolation_factor
        self._output_rate = input_rate * interpolation_factor
        self._cutoff_ratio = cutoff_ratio
        self._window = window

        # Cutoff at input Nyquist
        nyquist_in = input_rate / 2
        self._cutoff = nyquist_in * cutoff_ratio

        # Auto-calculate taps
        if num_taps <= 0:
            num_taps = max(31, interpolation_factor * 8 + 1)
            if num_taps % 2 == 0:
                num_taps += 1
        self._num_taps = num_taps

        # Design anti-imaging filter (at output rate)
        self._filter = self._design_filter()

        # State buffer
        self._buffer: Optional[np.ndarray] = None

    def _design_filter(self) -> np.ndarray:
        """Design anti-imaging lowpass filter."""
        n = self._num_taps
        fc = self._cutoff / self._output_rate

        t = np.arange(n) - (n - 1) / 2
        t[t == 0] = 1e-10

        h = 2 * fc * np.sinc(2 * fc * t)

        if self._window.lower() == "kaiser":
            window = np.kaiser(n, 8.0)
        elif self._window.lower() == "blackman":
            window = np.blackman(n)
        elif self._window.lower() == "hamming":
            window = np.hamming(n)
        else:
            window = np.hanning(n)

        h *= window

        # Scale for interpolation gain
        h *= self._factor
        h /= np.sum(h) / self._factor

        return h.astype(np.float32)

    @property
    def input_rate(self) -> float:
        """Get input sample rate."""
        return self._input_rate

    @property
    def output_rate(self) -> float:
        """Get output sample rate."""
        return self._output_rate

    @property
    def factor(self) -> int:
        """Get interpolation factor."""
        return self._factor

    @property
    def num_taps(self) -> int:
        """Get number of filter taps."""
        return self._num_taps

    def interpolate(self, samples: np.ndarray) -> np.ndarray:
        """
        Interpolate samples (one-shot).

        Args:
            samples: Input samples

        Returns:
            Interpolated samples at output rate
        """
        if self._factor == 1:
            return samples.copy()

        # Zero-stuff
        upsampled = np.zeros(len(samples) * self._factor, dtype=samples.dtype)
        upsampled[::self._factor] = samples

        # Apply anti-imaging filter
        filtered = np.convolve(upsampled, self._filter, mode='same')

        return filtered

    def interpolate_stream(self, samples: np.ndarray) -> np.ndarray:
        """
        Interpolate with state for streaming.

        Args:
            samples: Input samples

        Returns:
            Interpolated samples
        """
        if self._factor == 1:
            return samples.copy()

        n_taps = self._num_taps

        # Zero-stuff
        upsampled = np.zeros(len(samples) * self._factor, dtype=samples.dtype)
        upsampled[::self._factor] = samples

        # Initialize buffer
        if self._buffer is None:
            self._buffer = np.zeros(n_taps - 1, dtype=upsampled.dtype)

        # Prepend buffer
        padded = np.concatenate([self._buffer, upsampled])

        # Filter
        filtered = np.convolve(padded, self._filter, mode='valid')

        # Update buffer
        self._buffer = upsampled[-(n_taps - 1):]

        return filtered

    def reset(self) -> None:
        """Reset interpolator state."""
        self._buffer = None


class Resampler:
    """
    Rational resampling (P/Q rate change).

    Combines interpolation and decimation for arbitrary
    rational rate changes.

    Features:
    - Arbitrary P/Q rate conversion
    - Efficient combined filtering
    - Automatic factor calculation
    """

    def __init__(
        self,
        input_rate: float,
        output_rate: float,
        num_taps: int = 0,
        window: str = "kaiser"
    ):
        """
        Initialize resampler.

        Args:
            input_rate: Input sample rate in Hz
            output_rate: Desired output sample rate in Hz
            num_taps: Number of filter taps (0 = auto)
            window: Window function
        """
        self._input_rate = input_rate
        self._output_rate = output_rate

        # Find rational approximation P/Q
        self._interp_factor, self._decim_factor = self._find_rational(
            output_rate, input_rate
        )

        # Intermediate rate
        self._intermediate_rate = input_rate * self._interp_factor

        # Design combined filter at intermediate rate
        min_nyquist = min(input_rate, output_rate) / 2
        self._cutoff = min_nyquist * 0.8

        if num_taps <= 0:
            num_taps = max(31, (self._interp_factor + self._decim_factor) * 4 + 1)
            if num_taps % 2 == 0:
                num_taps += 1
        self._num_taps = num_taps

        self._filter = self._design_filter()

        # State
        self._buffer: Optional[np.ndarray] = None
        self._phase = 0

    def _find_rational(self, num: float, den: float, max_factor: int = 1000) -> Tuple[int, int]:
        """Find rational approximation P/Q."""
        from math import gcd

        # Try to find integer ratio
        ratio = num / den

        best_p, best_q = 1, 1
        best_error = abs(ratio - 1)

        for q in range(1, max_factor + 1):
            p = round(ratio * q)
            if p > 0 and p <= max_factor:
                error = abs(ratio - p / q)
                if error < best_error:
                    best_error = error
                    best_p, best_q = p, q
                    if error < 1e-9:
                        break

        # Simplify
        g = gcd(best_p, best_q)
        return best_p // g, best_q // g

    def _design_filter(self) -> np.ndarray:
        """Design resampling filter."""
        n = self._num_taps
        fc = self._cutoff / self._intermediate_rate

        t = np.arange(n) - (n - 1) / 2
        t[t == 0] = 1e-10

        h = 2 * fc * np.sinc(2 * fc * t)
        window = np.kaiser(n, 8.0)
        h *= window

        # Scale for interpolation
        h *= self._interp_factor
        h /= np.sum(h) / self._interp_factor

        return h.astype(np.float32)

    @property
    def input_rate(self) -> float:
        """Get input rate."""
        return self._input_rate

    @property
    def output_rate(self) -> float:
        """Get output rate."""
        return self._output_rate

    @property
    def interpolation_factor(self) -> int:
        """Get interpolation factor."""
        return self._interp_factor

    @property
    def decimation_factor(self) -> int:
        """Get decimation factor."""
        return self._decim_factor

    @property
    def actual_output_rate(self) -> float:
        """Get actual output rate (may differ slightly from requested)."""
        return self._input_rate * self._interp_factor / self._decim_factor

    def resample(self, samples: np.ndarray) -> np.ndarray:
        """
        Resample signal.

        Args:
            samples: Input samples

        Returns:
            Resampled output
        """
        if self._interp_factor == 1 and self._decim_factor == 1:
            return samples.copy()

        # Interpolate
        if self._interp_factor > 1:
            upsampled = np.zeros(len(samples) * self._interp_factor, dtype=samples.dtype)
            upsampled[::self._interp_factor] = samples
        else:
            upsampled = samples

        # Filter
        filtered = np.convolve(upsampled, self._filter, mode='same')

        # Decimate
        if self._decim_factor > 1:
            output = filtered[::self._decim_factor]
        else:
            output = filtered

        return output

    def resample_stream(self, samples: np.ndarray) -> np.ndarray:
        """
        Resample with state for streaming.

        Args:
            samples: Input samples

        Returns:
            Resampled output
        """
        if self._interp_factor == 1 and self._decim_factor == 1:
            return samples.copy()

        n_taps = self._num_taps

        # Interpolate
        if self._interp_factor > 1:
            upsampled = np.zeros(len(samples) * self._interp_factor, dtype=samples.dtype)
            upsampled[::self._interp_factor] = samples
        else:
            upsampled = samples

        # Buffer management
        if self._buffer is None:
            self._buffer = np.zeros(n_taps - 1, dtype=upsampled.dtype)

        padded = np.concatenate([self._buffer, upsampled])
        filtered = np.convolve(padded, self._filter, mode='valid')
        self._buffer = upsampled[-(n_taps - 1):]

        # Decimate with phase tracking
        if self._decim_factor > 1:
            start_idx = (self._decim_factor - self._phase) % self._decim_factor
            output = filtered[start_idx::self._decim_factor]
            self._phase = (self._phase + len(filtered)) % self._decim_factor
        else:
            output = filtered

        return output

    def reset(self) -> None:
        """Reset resampler state."""
        self._buffer = None
        self._phase = 0


"""
Digital filter implementations for SDR signal processing.

Provides various filter types for signal conditioning:
- Low-pass, high-pass, band-pass, band-stop
- FIR filter design using windowed-sinc method
- Real-time filtering with overlap-save
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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
    cutoff_low: float  # Lower cutoff frequency (Hz)
    cutoff_high: float  # Upper cutoff frequency (Hz)
    sample_rate: float  # Sample rate (Hz)
    num_taps: int  # Number of filter taps
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
            h = 2 * fc_high * np.sinc(2 * fc_high * t) - 2 * fc_low * np.sinc(
                2 * fc_low * t
            )

        elif spec.filter_type == FilterType.BANDSTOP:
            h = (
                np.sinc(t)
                - 2 * fc_high * np.sinc(2 * fc_high * t)
                + 2 * fc_low * np.sinc(2 * fc_low * t)
            )

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
        return np.convolve(samples, self._taps, mode="same")

    def filter_stream(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply filter with state preservation for streaming.

        Uses overlap-save method for efficient streaming.

        Args:
            samples: Input samples (must have at least 1 sample)

        Returns:
            Filtered samples

        Raises:
            ValueError: If samples array is empty
        """
        if len(samples) == 0:
            raise ValueError("Input samples array cannot be empty")

        n_taps = len(self._taps)

        # Initialize buffer if needed
        if self._buffer is None:
            self._buffer = np.zeros(n_taps - 1, dtype=samples.dtype)

        # Prepend buffer
        padded = np.concatenate([self._buffer, samples])

        # Convolve
        filtered = np.convolve(padded, self._taps, mode="valid")

        # Update buffer
        self._buffer = samples[-(n_taps - 1) :]

        return filtered

    def reset(self) -> None:
        """Reset filter state."""
        self._buffer = None

    def frequency_response(self, n_points: int = 512) -> Tuple[np.ndarray, np.ndarray]:
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
        self._filters: Dict[str, FIRFilter] = {}

    def add_filter(self, name: str, filter_obj: FIRFilter) -> None:
        """Add a filter to the bank."""
        self._filters[name] = filter_obj

    def create_lowpass(
        self, name: str, cutoff: float, num_taps: int = 101
    ) -> FIRFilter:
        """Create and add a lowpass filter."""
        spec = FilterSpec(
            filter_type=FilterType.LOWPASS,
            cutoff_low=0,
            cutoff_high=cutoff,
            sample_rate=self._sample_rate,
            num_taps=num_taps,
        )
        filt = FIRFilter(spec)
        self._filters[name] = filt
        return filt

    def create_highpass(
        self, name: str, cutoff: float, num_taps: int = 101
    ) -> FIRFilter:
        """Create and add a highpass filter."""
        spec = FilterSpec(
            filter_type=FilterType.HIGHPASS,
            cutoff_low=cutoff,
            cutoff_high=self._sample_rate / 2,
            sample_rate=self._sample_rate,
            num_taps=num_taps,
        )
        filt = FIRFilter(spec)
        self._filters[name] = filt
        return filt

    def create_bandpass(
        self, name: str, low_cutoff: float, high_cutoff: float, num_taps: int = 101
    ) -> FIRFilter:
        """Create and add a bandpass filter."""
        spec = FilterSpec(
            filter_type=FilterType.BANDPASS,
            cutoff_low=low_cutoff,
            cutoff_high=high_cutoff,
            sample_rate=self._sample_rate,
            num_taps=num_taps,
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
        window: str = "kaiser",
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
        filtered = np.convolve(samples, self._filter, mode="same")

        # Downsample by taking every Nth sample
        decimated = filtered[:: self._factor]

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
        filtered = np.convolve(padded, self._filter, mode="valid")

        # Update buffer
        self._buffer = samples[-(n_taps - 1) :]

        # Decimate with phase tracking
        start_idx = (self._factor - self._phase) % self._factor
        decimated = filtered[start_idx :: self._factor]

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
        filter_padded[: len(self._filter)] = self._filter

        # Reshape into polyphase components and time-reverse for convolution
        # The time reversal ensures correct polyphase decimation output
        polyphase = filter_padded.reshape(n_taps_per_phase, n_phases).T[:, ::-1]

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
                contribution = np.convolve(phase_signal, phase_filter, mode="full")
                output += contribution[:n_output]

        return output

    def get_frequency_response(
        self, n_points: int = 512
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
        window: str = "kaiser",
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

        # Scale for interpolation gain: normalize to unity then scale by factor
        # This ensures the interpolated signal has correct amplitude
        h *= self._factor / np.sum(h)

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
        upsampled[:: self._factor] = samples

        # Apply anti-imaging filter
        filtered = np.convolve(upsampled, self._filter, mode="same")

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
        upsampled[:: self._factor] = samples

        # Initialize buffer
        if self._buffer is None:
            self._buffer = np.zeros(n_taps - 1, dtype=upsampled.dtype)

        # Prepend buffer
        padded = np.concatenate([self._buffer, upsampled])

        # Filter
        filtered = np.convolve(padded, self._filter, mode="valid")

        # Update buffer
        self._buffer = upsampled[-(n_taps - 1) :]

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
        window: str = "kaiser",
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

    def _find_rational(
        self, num: float, den: float, max_factor: int = 1000
    ) -> Tuple[int, int]:
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

        # Scale for interpolation: normalize to unity then scale by factor
        # This ensures the interpolated signal has correct amplitude
        h *= self._interp_factor / np.sum(h)

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
            upsampled = np.zeros(
                len(samples) * self._interp_factor, dtype=samples.dtype
            )
            upsampled[:: self._interp_factor] = samples
        else:
            upsampled = samples

        # Filter
        filtered = np.convolve(upsampled, self._filter, mode="same")

        # Decimate
        if self._decim_factor > 1:
            output = filtered[:: self._decim_factor]
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
            upsampled = np.zeros(
                len(samples) * self._interp_factor, dtype=samples.dtype
            )
            upsampled[:: self._interp_factor] = samples
        else:
            upsampled = samples

        # Buffer management
        if self._buffer is None:
            self._buffer = np.zeros(n_taps - 1, dtype=upsampled.dtype)

        padded = np.concatenate([self._buffer, upsampled])
        filtered = np.convolve(padded, self._filter, mode="valid")
        self._buffer = upsampled[-(n_taps - 1) :]

        # Decimate with phase tracking
        if self._decim_factor > 1:
            start_idx = (self._decim_factor - self._phase) % self._decim_factor
            output = filtered[start_idx :: self._decim_factor]
            self._phase = (self._phase + len(filtered)) % self._decim_factor
        else:
            output = filtered

        return output

    def reset(self) -> None:
        """Reset resampler state."""
        self._buffer = None
        self._phase = 0


class AGCMode(Enum):
    """AGC detection modes."""

    RMS = "rms"  # Root Mean Square level detection
    PEAK = "peak"  # Peak level detection
    LOG = "log"  # Logarithmic domain processing
    MAGNITUDE = "magnitude"  # Magnitude-based (good for complex signals)


@dataclass
class AGCConfig:
    """AGC configuration parameters."""

    target_level: float = 1.0  # Target output amplitude
    attack_time: float = 0.001  # Attack time in seconds (fast response)
    decay_time: float = 0.1  # Decay/release time in seconds
    hang_time: float = 0.0  # Hold time before decay starts
    max_gain: float = 100.0  # Maximum gain (linear)
    min_gain: float = 0.001  # Minimum gain (linear)
    mode: AGCMode = AGCMode.RMS  # Detection mode
    reference_level: float = 0.0  # Reference level for log mode (dB)


class AGC:
    """
    Automatic Gain Control for SDR signal conditioning.

    Maintains consistent signal amplitude by dynamically adjusting gain
    based on input signal level. Essential for handling varying signal
    strengths in real-world radio reception.

    Features:
    - Multiple detection modes: RMS, peak, log-domain, magnitude
    - Configurable attack/decay time constants
    - Hang time to prevent pumping on speech
    - Min/max gain limiting
    - Real and complex signal support
    - Streaming with state preservation

    Typical use cases:
    - Normalizing signal levels before demodulation
    - Preventing ADC saturation in receive chains
    - Maintaining consistent audio output levels
    """

    def __init__(self, sample_rate: float, config: Optional[AGCConfig] = None):
        """
        Initialize AGC.

        Args:
            sample_rate: Sample rate in Hz
            config: AGC configuration (uses defaults if None)
        """
        self._sample_rate = sample_rate
        self._config = config or AGCConfig()

        # Calculate time constants as sample-domain coefficients
        self._update_coefficients()

        # State variables
        self._gain = 1.0  # Current gain
        self._level = 0.0  # Detected signal level
        self._hang_counter = 0  # Hang time counter
        self._peak_hold = 0.0  # Peak hold for peak detector

        # For RMS calculation (exponential moving average)
        self._rms_squared = 0.0

    def _update_coefficients(self) -> None:
        """Calculate filter coefficients from time constants."""
        # Attack coefficient (fast, gain decreases)
        if self._config.attack_time > 0:
            self._attack_coeff = 1.0 - np.exp(
                -1.0 / (self._config.attack_time * self._sample_rate)
            )
        else:
            self._attack_coeff = 1.0  # Instant attack

        # Decay coefficient (slow, gain increases)
        if self._config.decay_time > 0:
            self._decay_coeff = 1.0 - np.exp(
                -1.0 / (self._config.decay_time * self._sample_rate)
            )
        else:
            self._decay_coeff = 1.0  # Instant decay

        # Hang time in samples
        self._hang_samples = int(self._config.hang_time * self._sample_rate)

    @property
    def sample_rate(self) -> float:
        """Get sample rate."""
        return self._sample_rate

    @property
    def config(self) -> AGCConfig:
        """Get current configuration."""
        return self._config

    @property
    def current_gain(self) -> float:
        """Get current gain value."""
        return self._gain

    @property
    def current_gain_db(self) -> float:
        """Get current gain in dB."""
        return 20 * np.log10(self._gain + 1e-20)

    @property
    def current_level(self) -> float:
        """Get detected signal level."""
        return self._level

    def set_config(self, config: AGCConfig) -> None:
        """Update AGC configuration."""
        self._config = config
        self._update_coefficients()

    def set_attack_time(self, attack_time: float) -> None:
        """Set attack time in seconds."""
        self._config.attack_time = attack_time
        self._update_coefficients()

    def set_decay_time(self, decay_time: float) -> None:
        """Set decay time in seconds."""
        self._config.decay_time = decay_time
        self._update_coefficients()

    def set_target_level(self, level: float) -> None:
        """Set target output level."""
        self._config.target_level = level

    def _detect_level_rms(self, sample: complex) -> float:
        """RMS level detection with exponential averaging."""
        power = np.abs(sample) ** 2
        # Use attack/decay asymmetry
        if power > self._rms_squared:
            self._rms_squared += self._attack_coeff * (power - self._rms_squared)
        else:
            self._rms_squared += self._decay_coeff * (power - self._rms_squared)
        return np.sqrt(self._rms_squared)

    def _detect_level_peak(self, sample: complex) -> float:
        """Peak level detection with decay."""
        magnitude = np.abs(sample)
        if magnitude > self._peak_hold:
            self._peak_hold = magnitude
            self._hang_counter = self._hang_samples
        elif self._hang_counter > 0:
            self._hang_counter -= 1
        else:
            self._peak_hold *= 1.0 - self._decay_coeff
        return self._peak_hold

    def _detect_level_magnitude(self, sample: complex) -> float:
        """Simple magnitude-based detection with smoothing."""
        magnitude = np.abs(sample)
        if magnitude > self._level:
            self._level += self._attack_coeff * (magnitude - self._level)
        else:
            if self._hang_counter > 0:
                self._hang_counter -= 1
            else:
                self._level += self._decay_coeff * (magnitude - self._level)
        return self._level

    def _detect_level(self, sample: complex) -> float:
        """Detect signal level based on configured mode."""
        if self._config.mode == AGCMode.RMS:
            return self._detect_level_rms(sample)
        elif self._config.mode == AGCMode.PEAK:
            return self._detect_level_peak(sample)
        elif self._config.mode == AGCMode.MAGNITUDE:
            return self._detect_level_magnitude(sample)
        elif self._config.mode == AGCMode.LOG:
            # Log-domain: work in dB
            magnitude = np.abs(sample)
            if magnitude > 0:
                level_db = 20 * np.log10(magnitude)
                if level_db > self._config.reference_level:
                    return 10 ** (level_db / 20)
            return self._level
        return np.abs(sample)

    def _compute_gain(self, level: float) -> float:
        """Compute required gain from detected level."""
        if level > 1e-10:
            desired_gain = self._config.target_level / level
        else:
            desired_gain = self._config.max_gain

        # Apply gain limits
        gain = np.clip(desired_gain, self._config.min_gain, self._config.max_gain)
        return gain

    def process_sample(self, sample: complex) -> complex:
        """
        Process a single sample through AGC.

        Args:
            sample: Input sample (real or complex)

        Returns:
            Gain-adjusted sample
        """
        # Detect level
        self._level = self._detect_level(sample)

        # Compute desired gain
        target_gain = self._compute_gain(self._level)

        # Smooth gain changes (attack/decay on gain itself)
        if target_gain < self._gain:
            # Attacking (gain decreasing = signal increasing)
            self._gain += self._attack_coeff * (target_gain - self._gain)
        else:
            # Decaying (gain increasing = signal decreasing)
            if self._hang_counter > 0:
                pass  # Hold gain during hang time
            else:
                self._gain += self._decay_coeff * (target_gain - self._gain)

        # Apply gain
        return sample * self._gain

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Process array of samples through AGC.

        Args:
            samples: Input samples (real or complex array)

        Returns:
            Gain-adjusted samples
        """
        output = np.zeros_like(samples)
        for i, sample in enumerate(samples):
            output[i] = self.process_sample(sample)
        return output

    def process_block(self, samples: np.ndarray) -> np.ndarray:
        """
        Efficient block-based AGC processing.

        Uses block-level statistics for faster processing at the
        cost of slightly less precise gain control.

        Args:
            samples: Input samples

        Returns:
            Gain-adjusted samples
        """
        # Detect block level
        if self._config.mode == AGCMode.RMS:
            block_level = np.sqrt(np.mean(np.abs(samples) ** 2))
        elif self._config.mode == AGCMode.PEAK:
            block_level = np.max(np.abs(samples))
        else:
            block_level = np.mean(np.abs(samples))

        # Smooth level tracking
        if block_level > self._level:
            self._level += self._attack_coeff * (block_level - self._level)
        else:
            self._level += self._decay_coeff * (block_level - self._level)

        # Compute and apply gain
        target_gain = self._compute_gain(self._level)

        if target_gain < self._gain:
            self._gain += self._attack_coeff * (target_gain - self._gain)
        else:
            self._gain += self._decay_coeff * (target_gain - self._gain)

        return samples * self._gain

    def process_with_ramp(self, samples: np.ndarray) -> np.ndarray:
        """
        Process with per-sample gain ramping for smooth transitions.

        Interpolates gain across the block for artifact-free processing.

        Args:
            samples: Input samples

        Returns:
            Gain-adjusted samples with smooth transitions
        """
        n_samples = len(samples)
        output = np.zeros_like(samples)
        gains = np.zeros(n_samples)

        # Process each sample and record gain
        for i, sample in enumerate(samples):
            self._level = self._detect_level(sample)
            target_gain = self._compute_gain(self._level)

            if target_gain < self._gain:
                self._gain += self._attack_coeff * (target_gain - self._gain)
            else:
                self._gain += self._decay_coeff * (target_gain - self._gain)

            gains[i] = self._gain

        # Apply gain ramp
        output = samples * gains
        return output

    def get_gain_history(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process samples and return gain history for analysis.

        Args:
            samples: Input samples

        Returns:
            Tuple of (processed_samples, gain_values)
        """
        n_samples = len(samples)
        output = np.zeros_like(samples)
        gains = np.zeros(n_samples)

        for i, sample in enumerate(samples):
            output[i] = self.process_sample(sample)
            gains[i] = self._gain

        return output, gains

    def reset(self) -> None:
        """Reset AGC state."""
        self._gain = 1.0
        self._level = 0.0
        self._hang_counter = 0
        self._peak_hold = 0.0
        self._rms_squared = 0.0

    def freeze(self) -> None:
        """Freeze AGC at current gain (disable adaptation)."""
        self._config.attack_time = float("inf")
        self._config.decay_time = float("inf")
        self._update_coefficients()

    def unfreeze(self, attack_time: float = 0.001, decay_time: float = 0.1) -> None:
        """Unfreeze AGC and restore time constants."""
        self._config.attack_time = attack_time
        self._config.decay_time = decay_time
        self._update_coefficients()


class FastAGC:
    """
    Optimized AGC for high-performance applications.

    Uses vectorized operations for block processing with
    minimal per-sample overhead. Suitable for high sample rates.
    """

    def __init__(
        self,
        sample_rate: float,
        target_level: float = 1.0,
        attack_time: float = 0.001,
        decay_time: float = 0.1,
        max_gain: float = 100.0,
        min_gain: float = 0.001,
    ):
        """
        Initialize FastAGC.

        Args:
            sample_rate: Sample rate in Hz
            target_level: Target output amplitude
            attack_time: Attack time in seconds
            decay_time: Decay time in seconds
            max_gain: Maximum gain
            min_gain: Minimum gain
        """
        self._sample_rate = sample_rate
        self._target = target_level
        self._max_gain = max_gain
        self._min_gain = min_gain

        # Time constants
        self._attack = 1.0 - np.exp(-1.0 / (attack_time * sample_rate))
        self._decay = 1.0 - np.exp(-1.0 / (decay_time * sample_rate))

        # State
        self._gain = 1.0
        self._envelope = 0.0

    @property
    def current_gain(self) -> float:
        """Get current gain."""
        return self._gain

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Fast block processing.

        Args:
            samples: Input samples

        Returns:
            Gain-adjusted samples
        """
        # Envelope detection (vectorized)
        magnitudes = np.abs(samples)

        # Block envelope (RMS-like)
        block_env = np.sqrt(np.mean(magnitudes**2))

        # Update envelope with attack/decay
        if block_env > self._envelope:
            self._envelope += self._attack * (block_env - self._envelope)
        else:
            self._envelope += self._decay * (block_env - self._envelope)

        # Compute gain
        if self._envelope > 1e-10:
            target_gain = self._target / self._envelope
            target_gain = np.clip(target_gain, self._min_gain, self._max_gain)
        else:
            target_gain = self._gain

        # Smooth gain
        if target_gain < self._gain:
            self._gain += self._attack * (target_gain - self._gain)
        else:
            self._gain += self._decay * (target_gain - self._gain)

        return samples * self._gain

    def reset(self) -> None:
        """Reset state."""
        self._gain = 1.0
        self._envelope = 0.0


class SquelchMode(Enum):
    """Squelch detection modes."""

    CARRIER = "carrier"  # Signal amplitude based
    NOISE = "noise"  # Noise level based (opens on low noise)
    CTCSS = "ctcss"  # Continuous Tone-Coded Squelch System
    DCS = "dcs"  # Digital-Coded Squelch
    VOX = "vox"  # Voice-operated (audio activity)


class SquelchState(Enum):
    """Squelch gate state."""

    CLOSED = "closed"  # Muted - no signal
    OPENING = "opening"  # Transitioning to open
    OPEN = "open"  # Unmuted - signal present
    CLOSING = "closing"  # Transitioning to closed (tail time)


@dataclass
class SquelchConfig:
    """Squelch configuration parameters."""

    threshold: float = 0.1  # Open threshold (linear amplitude)
    hysteresis: float = 0.02  # Close offset below threshold
    attack_time: float = 0.005  # Time to fully open (seconds)
    release_time: float = 0.02  # Time to fully close (seconds)
    tail_time: float = 0.3  # Hold open after signal drops (seconds)
    mode: SquelchMode = SquelchMode.CARRIER
    ctcss_freq: float = 0.0  # CTCSS tone frequency (67-254.1 Hz)
    ctcss_bandwidth: float = 10.0  # CTCSS detection bandwidth


class Squelch:
    """
    Signal-level based squelch for SDR audio muting.

    Mutes output when signal level drops below threshold, preventing
    the user from hearing noise during periods of no signal activity.

    Features:
    - Carrier-based squelch (amplitude threshold)
    - Noise squelch (opens when noise floor drops)
    - CTCSS tone squelch (sub-audible tone detection)
    - Configurable threshold with hysteresis
    - Smooth attack/release transitions
    - Tail time to prevent choppy audio
    - VOX mode for voice-operated switching

    Typical use cases:
    - FM receiver squelch
    - Repeater access with CTCSS
    - Scanner squelch for channel monitoring
    - VOX for hands-free operation
    """

    def __init__(self, sample_rate: float, config: Optional[SquelchConfig] = None):
        """
        Initialize Squelch.

        Args:
            sample_rate: Sample rate in Hz
            config: Squelch configuration (uses defaults if None)
        """
        self._sample_rate = sample_rate
        self._config = config or SquelchConfig()

        # State
        self._state = SquelchState.CLOSED
        self._gate = 0.0  # Current gate level (0=closed, 1=open)
        self._signal_level = 0.0  # Detected signal level
        self._noise_level = 1.0  # Estimated noise floor
        self._tail_counter = 0  # Samples remaining in tail time
        self._ctcss_detected = False  # CTCSS tone present

        # Calculate coefficients
        self._update_coefficients()

        # CTCSS detector state
        self._ctcss_filter_state = 0.0 + 0.0j
        self._ctcss_magnitude = 0.0

    def _update_coefficients(self) -> None:
        """Calculate time-domain coefficients."""
        # Attack rate (gate opens)
        if self._config.attack_time > 0:
            self._attack_rate = 1.0 / (self._config.attack_time * self._sample_rate)
        else:
            self._attack_rate = 1.0

        # Release rate (gate closes)
        if self._config.release_time > 0:
            self._release_rate = 1.0 / (self._config.release_time * self._sample_rate)
        else:
            self._release_rate = 1.0

        # Tail time in samples
        self._tail_samples = int(self._config.tail_time * self._sample_rate)

        # Level smoothing coefficient (fast tracking)
        self._level_alpha = 1.0 - np.exp(-1.0 / (0.01 * self._sample_rate))

        # Noise estimation coefficient (slow tracking)
        self._noise_alpha = 1.0 - np.exp(-1.0 / (0.5 * self._sample_rate))

        # CTCSS Goertzel coefficient
        if self._config.ctcss_freq > 0:
            k = int(
                0.5
                + (self._sample_rate * 0.02)
                * self._config.ctcss_freq
                / self._sample_rate
            )
            self._ctcss_coeff = 2.0 * np.cos(
                2.0 * np.pi * k / (self._sample_rate * 0.02)
            )

    @property
    def sample_rate(self) -> float:
        """Get sample rate."""
        return self._sample_rate

    @property
    def config(self) -> SquelchConfig:
        """Get current configuration."""
        return self._config

    @property
    def state(self) -> SquelchState:
        """Get current squelch state."""
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if squelch is open (audio passing)."""
        return self._state in (
            SquelchState.OPEN,
            SquelchState.OPENING,
            SquelchState.CLOSING,
        )

    @property
    def gate_level(self) -> float:
        """Get current gate level (0.0 to 1.0)."""
        return self._gate

    @property
    def signal_level(self) -> float:
        """Get detected signal level."""
        return self._signal_level

    @property
    def signal_level_db(self) -> float:
        """Get signal level in dB."""
        return 20 * np.log10(self._signal_level + 1e-20)

    @property
    def noise_level(self) -> float:
        """Get estimated noise floor."""
        return self._noise_level

    def set_threshold(self, threshold: float) -> None:
        """Set squelch threshold."""
        self._config.threshold = threshold

    def set_threshold_db(self, threshold_db: float) -> None:
        """Set squelch threshold in dB."""
        self._config.threshold = 10 ** (threshold_db / 20)

    def set_config(self, config: SquelchConfig) -> None:
        """Update configuration."""
        self._config = config
        self._update_coefficients()

    def _detect_carrier_level(self, sample: complex) -> float:
        """Detect signal level for carrier squelch."""
        magnitude = np.abs(sample)
        self._signal_level += self._level_alpha * (magnitude - self._signal_level)
        return self._signal_level

    def _detect_noise_level(self, sample: complex) -> float:
        """
        Detect noise level for noise squelch.

        Noise squelch opens when noise floor drops (signal present).
        """
        magnitude = np.abs(sample)

        # Track signal level
        self._signal_level += self._level_alpha * (magnitude - self._signal_level)

        # Slowly track noise floor (only when it drops)
        if magnitude < self._noise_level:
            self._noise_level += self._noise_alpha * (magnitude - self._noise_level)
        else:
            # Very slow rise for noise floor
            self._noise_level += (self._noise_alpha * 0.1) * (
                magnitude - self._noise_level
            )

        # Signal-to-noise ratio indicator
        if self._noise_level > 1e-10:
            return self._signal_level / self._noise_level
        return self._signal_level

    def _detect_ctcss(self, samples: np.ndarray) -> bool:
        """
        Detect CTCSS sub-audible tone.

        Uses Goertzel algorithm for efficient single-frequency detection.
        """
        if self._config.ctcss_freq <= 0:
            return False

        # Simple CTCSS detection using bandpass energy
        # Generate CTCSS reference
        n = len(samples)
        t = np.arange(n) / self._sample_rate
        ref_i = np.cos(2 * np.pi * self._config.ctcss_freq * t)
        ref_q = np.sin(2 * np.pi * self._config.ctcss_freq * t)

        # Correlate (assumes real audio input)
        if np.iscomplexobj(samples):
            audio = samples.real
        else:
            audio = samples

        # Integrate over block
        corr_i: float = float(np.sum(audio * ref_i))
        corr_q: float = float(np.sum(audio * ref_q))
        magnitude = np.sqrt(corr_i**2 + corr_q**2) / n

        # Smooth magnitude
        self._ctcss_magnitude += 0.1 * (magnitude - self._ctcss_magnitude)

        # Compare to threshold (CTCSS is typically 10-15% modulation)
        # Threshold relative to signal level
        ctcss_threshold = self._signal_level * 0.05
        self._ctcss_detected = self._ctcss_magnitude > ctcss_threshold

        return self._ctcss_detected

    def _detect_vox(self, samples: np.ndarray) -> float:
        """
        Voice-operated squelch detection.

        Detects audio activity based on energy and variation.
        """
        if np.iscomplexobj(samples):
            audio = np.abs(samples)
        else:
            audio = np.abs(samples)

        # RMS level
        rms = np.sqrt(np.mean(audio**2))
        self._signal_level += self._level_alpha * (rms - self._signal_level)

        return self._signal_level

    def _should_open(self, level: float) -> bool:
        """Determine if squelch should open."""
        if self._config.mode == SquelchMode.CARRIER:
            return level > self._config.threshold

        elif self._config.mode == SquelchMode.NOISE:
            # Noise squelch: open when SNR is high (noise drops)
            return level > (1.0 / self._config.threshold)  # Invert threshold meaning

        elif self._config.mode == SquelchMode.CTCSS:
            # Must have both signal AND correct tone
            return (level > self._config.threshold) and self._ctcss_detected

        elif self._config.mode == SquelchMode.VOX:
            return level > self._config.threshold

        return level > self._config.threshold

    def _should_close(self, level: float) -> bool:
        """Determine if squelch should close (with hysteresis)."""
        close_threshold = self._config.threshold - self._config.hysteresis

        if self._config.mode == SquelchMode.CARRIER:
            return level < close_threshold

        elif self._config.mode == SquelchMode.NOISE:
            return level < (1.0 / close_threshold)

        elif self._config.mode == SquelchMode.CTCSS:
            # Close if signal drops OR tone disappears
            return (level < close_threshold) or not self._ctcss_detected

        elif self._config.mode == SquelchMode.VOX:
            return level < close_threshold

        return level < close_threshold

    def _update_gate(self) -> None:
        """Update gate level based on state."""
        if self._state == SquelchState.OPENING:
            self._gate += self._attack_rate
            if self._gate >= 1.0:
                self._gate = 1.0
                self._state = SquelchState.OPEN

        elif self._state == SquelchState.CLOSING:
            self._gate -= self._release_rate
            if self._gate <= 0.0:
                self._gate = 0.0
                self._state = SquelchState.CLOSED

    def process_sample(self, sample: complex) -> complex:
        """
        Process a single sample through squelch.

        Args:
            sample: Input sample

        Returns:
            Gated sample (may be attenuated or muted)
        """
        # Detect level
        if self._config.mode == SquelchMode.NOISE:
            level = self._detect_noise_level(sample)
        else:
            level = self._detect_carrier_level(sample)

        # State machine
        if self._state == SquelchState.CLOSED:
            if self._should_open(level):
                self._state = SquelchState.OPENING
                self._tail_counter = self._tail_samples

        elif self._state == SquelchState.OPEN:
            if self._should_close(level):
                if self._tail_counter > 0:
                    self._tail_counter -= 1
                else:
                    self._state = SquelchState.CLOSING
            else:
                self._tail_counter = self._tail_samples

        elif self._state == SquelchState.OPENING:
            if self._should_close(level):
                self._state = SquelchState.CLOSING

        elif self._state == SquelchState.CLOSING:
            if self._should_open(level):
                self._state = SquelchState.OPENING
                self._tail_counter = self._tail_samples

        # Update gate
        self._update_gate()

        # Apply gate
        return sample * self._gate

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Process array of samples through squelch.

        Args:
            samples: Input samples

        Returns:
            Gated samples
        """
        # For CTCSS mode, detect tone first
        if self._config.mode == SquelchMode.CTCSS:
            self._detect_ctcss(samples)

        # For VOX mode, use block detection
        if self._config.mode == SquelchMode.VOX:
            self._detect_vox(samples)

        # Process samples
        output = np.zeros_like(samples)
        for i, sample in enumerate(samples):
            output[i] = self.process_sample(sample)

        return output

    def process_block(self, samples: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Block-based squelch processing.

        More efficient for large blocks, returns squelch state.

        Args:
            samples: Input samples

        Returns:
            Tuple of (gated_samples, is_open)
        """
        # Detect CTCSS if needed
        if self._config.mode == SquelchMode.CTCSS:
            self._detect_ctcss(samples)

        # Block level detection
        if self._config.mode == SquelchMode.NOISE:
            magnitudes = np.abs(samples)
            block_level = float(np.mean(magnitudes))
            self._signal_level = block_level
            # Simple noise estimate
            noise_est: float = float(np.percentile(magnitudes, 10))
            if noise_est > 1e-10:
                level = block_level / noise_est
            else:
                level = block_level
        else:
            level = np.sqrt(np.mean(np.abs(samples) ** 2))
            self._signal_level = level

        # Determine state

        if self._state == SquelchState.CLOSED and self._should_open(level):
            self._state = SquelchState.OPENING
        elif self._state == SquelchState.OPEN and self._should_close(level):
            if self._tail_counter > len(samples):
                self._tail_counter -= len(samples)
            else:
                self._state = SquelchState.CLOSING

        # Generate gate ramp for smooth transitions
        n = len(samples)
        if self._state == SquelchState.OPENING:
            # Ramp from current gate to 1.0
            end_gate = min(1.0, self._gate + self._attack_rate * n)
            gate = np.linspace(self._gate, end_gate, n)
            self._gate = end_gate
            if self._gate >= 1.0:
                self._state = SquelchState.OPEN

        elif self._state == SquelchState.CLOSING:
            # Ramp from current gate to 0.0
            end_gate = max(0.0, self._gate - self._release_rate * n)
            gate = np.linspace(self._gate, end_gate, n)
            self._gate = end_gate
            if self._gate <= 0.0:
                self._state = SquelchState.CLOSED

        elif self._state == SquelchState.OPEN:
            gate = np.ones(n)
            self._gate = 1.0
            self._tail_counter = self._tail_samples

        else:  # CLOSED
            gate = np.zeros(n)
            self._gate = 0.0

        return samples * gate, self.is_open

    def get_status(self) -> Dict[str, Any]:
        """Get squelch status information."""
        return {
            "state": self._state.value,
            "is_open": self.is_open,
            "gate_level": self._gate,
            "signal_level": self._signal_level,
            "signal_level_db": self.signal_level_db,
            "noise_level": self._noise_level,
            "threshold": self._config.threshold,
            "mode": self._config.mode.value,
            "ctcss_detected": (
                self._ctcss_detected if self._config.mode == SquelchMode.CTCSS else None
            ),
        }

    def reset(self) -> None:
        """Reset squelch state."""
        self._state = SquelchState.CLOSED
        self._gate = 0.0
        self._signal_level = 0.0
        self._noise_level = 1.0
        self._tail_counter = 0
        self._ctcss_detected = False
        self._ctcss_magnitude = 0.0

    def force_open(self) -> None:
        """Force squelch open (bypass)."""
        self._state = SquelchState.OPEN
        self._gate = 1.0

    def force_close(self) -> None:
        """Force squelch closed (mute)."""
        self._state = SquelchState.CLOSED
        self._gate = 0.0


# Common CTCSS tones (EIA standard)
CTCSS_TONES = {
    "XZ": 67.0,
    "WZ": 69.3,
    "XA": 71.9,
    "WA": 74.4,
    "XB": 77.0,
    "WB": 79.7,
    "YZ": 82.5,
    "YA": 85.4,
    "YB": 88.5,
    "ZZ": 91.5,
    "ZA": 94.8,
    "ZB": 97.4,
    "1Z": 100.0,
    "1A": 103.5,
    "1B": 107.2,
    "2Z": 110.9,
    "2A": 114.8,
    "2B": 118.8,
    "3Z": 123.0,
    "3A": 127.3,
    "3B": 131.8,
    "4Z": 136.5,
    "4A": 141.3,
    "4B": 146.2,
    "5Z": 151.4,
    "5A": 156.7,
    "5B": 162.2,
    "6Z": 167.9,
    "6A": 173.8,
    "6B": 179.9,
    "7Z": 186.2,
    "7A": 192.8,
    "M1": 203.5,
    "M2": 210.7,
    "M3": 218.1,
    "M4": 225.7,
    "M5": 233.6,
    "M6": 241.8,
    "M7": 250.3,
}


def get_ctcss_tone(name: str) -> Optional[float]:
    """Get CTCSS tone frequency by name."""
    return CTCSS_TONES.get(name.upper())


def find_ctcss_tone(frequency: float, tolerance: float = 1.0) -> Optional[str]:
    """Find CTCSS tone name by frequency."""
    for name, freq in CTCSS_TONES.items():
        if abs(freq - frequency) <= tolerance:
            return name
    return None


class NoiseReductionMethod(Enum):
    """Noise reduction algorithms."""

    SPECTRAL_SUBTRACTION = "spectral_subtraction"  # FFT-based spectral subtraction
    WIENER = "wiener"  # Wiener filter
    LMS = "lms"  # Least Mean Squares adaptive
    NLMS = "nlms"  # Normalized LMS
    MOVING_AVERAGE = "moving_average"  # Simple smoothing
    MEDIAN = "median"  # Median filter for impulse noise
    GATE = "gate"  # Noise gate (mute below threshold)


@dataclass
class NoiseReductionConfig:
    """Noise reduction configuration."""

    method: NoiseReductionMethod = NoiseReductionMethod.SPECTRAL_SUBTRACTION
    # Spectral subtraction parameters
    fft_size: int = 1024  # FFT size for spectral analysis
    overlap: float = 0.5  # FFT overlap (0.0-0.9)
    noise_estimation_frames: int = 10  # Frames for initial noise estimation
    subtraction_factor: float = 1.0  # Over-subtraction factor (1.0-3.0)
    floor_factor: float = 0.01  # Spectral floor to prevent musical noise
    # Wiener filter parameters
    wiener_alpha: float = 0.98  # Noise estimate smoothing
    # LMS/NLMS parameters
    lms_step_size: float = 0.01  # Adaptation step size (mu)
    lms_filter_length: int = 32  # Adaptive filter length
    # Gate parameters
    gate_threshold: float = 0.02  # Noise gate threshold
    gate_attack: float = 0.001  # Gate attack time (seconds)
    gate_release: float = 0.05  # Gate release time (seconds)
    # General
    smoothing_alpha: float = 0.1  # Output smoothing


class NoiseReduction:
    """
    DSP-based noise reduction for SDR signals.

    Provides multiple algorithms for reducing noise in received signals,
    improving intelligibility and signal quality.

    Features:
    - Spectral subtraction: FFT-based noise removal
    - Wiener filtering: Optimal linear filtering
    - LMS/NLMS: Adaptive noise cancellation
    - Noise gate: Simple threshold-based muting
    - Median filter: Impulse noise removal
    - Real and complex signal support

    Typical use cases:
    - Improving weak signal reception
    - Reducing background noise in voice
    - Cleaning up noisy data signals
    - Pre-processing before demodulation
    """

    def __init__(
        self, sample_rate: float, config: Optional[NoiseReductionConfig] = None
    ):
        """
        Initialize noise reduction.

        Args:
            sample_rate: Sample rate in Hz
            config: Configuration (uses defaults if None)
        """
        self._sample_rate = sample_rate
        self._config = config or NoiseReductionConfig()

        # FFT parameters
        self._fft_size = self._config.fft_size
        self._hop_size = int(self._fft_size * (1 - self._config.overlap))
        self._window = np.hanning(self._fft_size)

        # Noise estimation state
        self._noise_spectrum = None
        self._noise_frames_collected = 0
        self._noise_estimated = False

        # Processing buffers
        self._input_buffer = np.array([], dtype=np.complex128)
        self._output_buffer = np.array([], dtype=np.complex128)
        self._overlap_buffer = np.zeros(self._fft_size - self._hop_size)

        # LMS state
        self._lms_weights = np.zeros(self._config.lms_filter_length)
        self._lms_buffer = np.zeros(self._config.lms_filter_length)

        # Gate state
        self._gate_level = 0.0
        self._gate_attack_coeff = 1.0 - np.exp(
            -1.0 / (self._config.gate_attack * sample_rate)
        )
        self._gate_release_coeff = 1.0 - np.exp(
            -1.0 / (self._config.gate_release * sample_rate)
        )

        # Wiener state
        self._wiener_noise_psd = None

        # Statistics
        self._input_power = 0.0
        self._output_power = 0.0
        self._noise_reduction_db = 0.0

    @property
    def sample_rate(self) -> float:
        """Get sample rate."""
        return self._sample_rate

    @property
    def config(self) -> NoiseReductionConfig:
        """Get configuration."""
        return self._config

    @property
    def noise_estimated(self) -> bool:
        """Check if noise has been estimated."""
        return self._noise_estimated

    @property
    def noise_reduction_db(self) -> float:
        """Get estimated noise reduction in dB."""
        return self._noise_reduction_db

    def set_config(self, config: NoiseReductionConfig) -> None:
        """Update configuration."""
        self._config = config
        self._fft_size = config.fft_size
        self._hop_size = int(self._fft_size * (1 - config.overlap))
        self._window = np.hanning(self._fft_size)

    def estimate_noise(self, noise_samples: np.ndarray) -> None:
        """
        Estimate noise spectrum from a noise-only segment.

        Args:
            noise_samples: Samples containing only noise (no signal)
        """
        # Compute average power spectrum of noise
        n_frames = len(noise_samples) // self._hop_size
        if n_frames < 1:
            return

        noise_psd = np.zeros(self._fft_size // 2 + 1)

        for i in range(n_frames):
            start = i * self._hop_size
            end = start + self._fft_size
            if end > len(noise_samples):
                break

            frame = noise_samples[start:end]
            if len(frame) < self._fft_size:
                frame = np.pad(frame, (0, self._fft_size - len(frame)))

            windowed = frame * self._window
            spectrum = np.fft.rfft(windowed)
            noise_psd += np.abs(spectrum) ** 2

        noise_psd /= n_frames
        self._noise_spectrum = np.sqrt(noise_psd)
        self._wiener_noise_psd = noise_psd
        self._noise_estimated = True

    def _spectral_subtraction(self, samples: np.ndarray) -> np.ndarray:
        """
        Spectral subtraction noise reduction.

        Estimates noise spectrum and subtracts it from signal spectrum.
        """
        output: list[float] = []
        is_complex = np.iscomplexobj(samples)

        # Process in overlapping frames
        for i in range(0, len(samples) - self._fft_size + 1, self._hop_size):
            frame = samples[i : i + self._fft_size]

            # Apply window
            if is_complex:
                windowed = frame * self._window
            else:
                windowed = frame.astype(np.float64) * self._window

            # FFT
            spectrum = np.fft.rfft(windowed)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)

            # Estimate noise if not done
            if not self._noise_estimated:
                if self._noise_spectrum is None:
                    self._noise_spectrum = magnitude.copy()
                else:
                    # Running average for noise estimation
                    alpha = 1.0 / (self._noise_frames_collected + 1)
                    self._noise_spectrum = (
                        1 - alpha
                    ) * self._noise_spectrum + alpha * magnitude

                self._noise_frames_collected += 1
                if self._noise_frames_collected >= self._config.noise_estimation_frames:
                    self._noise_estimated = True

            # Spectral subtraction
            if self._noise_estimated and self._noise_spectrum is not None:
                # Over-subtraction
                subtracted = (
                    magnitude - self._config.subtraction_factor * self._noise_spectrum
                )

                # Spectral floor to prevent musical noise
                floor = self._config.floor_factor * magnitude
                subtracted = np.maximum(subtracted, floor)

                # Reconstruct
                spectrum_clean = subtracted * np.exp(1j * phase)
            else:
                spectrum_clean = spectrum

            # IFFT
            frame_clean = np.fft.irfft(spectrum_clean, n=self._fft_size)

            # Overlap-add
            if len(output) == 0:
                output = list(frame_clean[: self._hop_size])
            else:
                # Add overlap from previous frame to beginning of current frame
                overlap_len = min(len(self._overlap_buffer), len(frame_clean))
                for j in range(overlap_len):
                    # Add overlap samples to the end of current output
                    output_idx = len(output) - len(self._overlap_buffer) + j
                    if 0 <= output_idx < len(output):
                        output[output_idx] += frame_clean[j]
                output.extend(frame_clean[overlap_len : self._hop_size + overlap_len])

            self._overlap_buffer = frame_clean[self._hop_size :]

        result = np.array(output[: len(samples)])

        if is_complex:
            return result.astype(np.complex128)
        return result

    def _wiener_filter(self, samples: np.ndarray) -> np.ndarray:
        """
        Wiener filter noise reduction.

        Optimal linear filter that minimizes mean square error.
        """
        output: List[float] = []
        is_complex = np.iscomplexobj(samples)

        for i in range(0, len(samples) - self._fft_size + 1, self._hop_size):
            frame = samples[i : i + self._fft_size]
            windowed = frame * self._window

            # FFT
            spectrum = np.fft.rfft(windowed)
            power = np.abs(spectrum) ** 2

            # Initialize noise PSD if needed
            if self._wiener_noise_psd is None:
                self._wiener_noise_psd = power.copy()
            assert self._wiener_noise_psd is not None
            wiener_noise: np.ndarray = self._wiener_noise_psd

            # Update noise estimate (during silence)
            signal_power = np.mean(power)
            noise_power = np.mean(self._wiener_noise_psd)

            if signal_power < noise_power * 1.5:  # Likely noise-only
                alpha = self._config.wiener_alpha
                self._wiener_noise_psd = alpha * wiener_noise + (1 - alpha) * power
                wiener_noise = self._wiener_noise_psd

            # Wiener filter gain
            # H(f) = max(0, 1 - noise_psd / signal_psd)
            snr = power / (wiener_noise + 1e-10)
            gain = np.maximum(0, 1 - 1 / (snr + 1e-10))
            gain = np.sqrt(gain)  # Amplitude domain

            # Apply gain
            spectrum_clean = spectrum * gain

            # IFFT
            frame_clean = np.fft.irfft(spectrum_clean, n=self._fft_size)
            output.extend(frame_clean[: self._hop_size])

        result = np.array(output[: len(samples)])
        if is_complex:
            return result.astype(np.complex128)
        return result

    def _lms_filter(self, samples: np.ndarray, normalized: bool = False) -> np.ndarray:
        """
        LMS/NLMS adaptive noise cancellation.

        Learns to predict and cancel noise from the signal.
        """
        output = np.zeros_like(samples)
        n = len(samples)
        mu = self._config.lms_step_size

        for i in range(n):
            # Shift buffer
            self._lms_buffer = np.roll(self._lms_buffer, 1)
            self._lms_buffer[0] = samples[i]

            # Filter output (noise estimate)
            noise_est = np.dot(self._lms_weights, self._lms_buffer)

            # Error (desired signal)
            error = samples[i] - noise_est

            # Update weights
            if normalized:
                # NLMS normalization
                norm = np.dot(self._lms_buffer, self._lms_buffer) + 1e-10
                self._lms_weights += (mu / norm) * error * self._lms_buffer
            else:
                # Standard LMS
                self._lms_weights += mu * error * self._lms_buffer

            output[i] = error

        return output

    def _moving_average(self, samples: np.ndarray) -> np.ndarray:
        """Simple moving average smoothing."""
        window_size = max(3, self._config.lms_filter_length // 4)
        kernel = np.ones(window_size) / window_size
        return np.convolve(samples, kernel, mode="same")

    def _median_filter(self, samples: np.ndarray) -> np.ndarray:
        """Median filter for impulse noise removal."""
        window_size = 5
        output = np.zeros_like(samples)
        half = window_size // 2

        for i in range(len(samples)):
            start = max(0, i - half)
            end = min(len(samples), i + half + 1)
            if np.iscomplexobj(samples):
                # For complex, filter magnitude and preserve phase
                window = samples[start:end]
                mags = np.abs(window)
                median_idx = np.argsort(mags)[len(mags) // 2]
                output[i] = window[median_idx]
            else:
                output[i] = np.median(samples[start:end])

        return output

    def _noise_gate(self, samples: np.ndarray) -> np.ndarray:
        """Noise gate - mute below threshold."""
        output = np.zeros_like(samples)

        for i, sample in enumerate(samples):
            magnitude = np.abs(sample)

            # Update gate level
            if magnitude > self._config.gate_threshold:
                self._gate_level += self._gate_attack_coeff * (1.0 - self._gate_level)
            else:
                self._gate_level += self._gate_release_coeff * (0.0 - self._gate_level)

            output[i] = sample * self._gate_level

        return output

    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction to samples.

        Args:
            samples: Input samples (real or complex)

        Returns:
            Noise-reduced samples
        """
        # Track input power
        self._input_power = np.mean(np.abs(samples) ** 2)

        # Apply selected method
        method = self._config.method

        if method == NoiseReductionMethod.SPECTRAL_SUBTRACTION:
            output = self._spectral_subtraction(samples)
        elif method == NoiseReductionMethod.WIENER:
            output = self._wiener_filter(samples)
        elif method == NoiseReductionMethod.LMS:
            output = self._lms_filter(samples, normalized=False)
        elif method == NoiseReductionMethod.NLMS:
            output = self._lms_filter(samples, normalized=True)
        elif method == NoiseReductionMethod.MOVING_AVERAGE:
            output = self._moving_average(samples)
        elif method == NoiseReductionMethod.MEDIAN:
            output = self._median_filter(samples)
        elif method == NoiseReductionMethod.GATE:
            output = self._noise_gate(samples)
        else:
            output = samples.copy()

        # Track output power and compute reduction
        self._output_power = np.mean(np.abs(output) ** 2)
        if self._input_power > 1e-20:
            ratio = self._output_power / self._input_power
            if ratio > 0:
                self._noise_reduction_db = -10 * np.log10(ratio)

        return output

    def process_with_reference(
        self, signal: np.ndarray, noise_reference: np.ndarray
    ) -> np.ndarray:
        """
        Adaptive noise cancellation with reference signal.

        Uses a separate noise reference (e.g., from another microphone)
        to cancel correlated noise from the primary signal.

        Args:
            signal: Primary signal with noise
            noise_reference: Reference noise signal

        Returns:
            Noise-cancelled signal
        """
        output = np.zeros_like(signal)
        L = self._config.lms_filter_length
        mu = self._config.lms_step_size
        ref_buffer = np.zeros(L)

        for i in range(len(signal)):
            # Shift reference buffer
            ref_buffer = np.roll(ref_buffer, 1)
            ref_buffer[0] = noise_reference[i] if i < len(noise_reference) else 0

            # Filter reference to estimate noise in primary
            noise_est = np.dot(self._lms_weights, ref_buffer)

            # Subtract noise estimate from primary
            error = signal[i] - noise_est

            # Update weights (NLMS)
            norm = np.dot(ref_buffer, ref_buffer) + 1e-10
            self._lms_weights += (mu / norm) * error * ref_buffer

            output[i] = error

        return output

    def get_noise_spectrum(self) -> Optional[np.ndarray]:
        """Get estimated noise spectrum."""
        return self._noise_spectrum.copy() if self._noise_spectrum is not None else None

    def get_stats(self) -> Dict[str, Any]:
        """Get noise reduction statistics."""
        return {
            "method": self._config.method.value,
            "noise_estimated": self._noise_estimated,
            "input_power": self._input_power,
            "output_power": self._output_power,
            "noise_reduction_db": self._noise_reduction_db,
            "fft_size": self._fft_size,
        }

    def reset(self) -> None:
        """Reset noise reduction state."""
        self._noise_spectrum = None
        self._noise_frames_collected = 0
        self._noise_estimated = False
        self._input_buffer = np.array([], dtype=np.complex128)
        self._output_buffer = np.array([], dtype=np.complex128)
        self._overlap_buffer = np.zeros(self._fft_size - self._hop_size)
        self._lms_weights = np.zeros(self._config.lms_filter_length)
        self._lms_buffer = np.zeros(self._config.lms_filter_length)
        self._gate_level = 0.0
        self._wiener_noise_psd = None


class SpectralSubtraction(NoiseReduction):
    """
    Convenience class for spectral subtraction noise reduction.

    Pre-configured for spectral subtraction method.
    """

    def __init__(
        self,
        sample_rate: float,
        fft_size: int = 1024,
        subtraction_factor: float = 1.5,
        floor_factor: float = 0.02,
    ):
        """
        Initialize spectral subtraction.

        Args:
            sample_rate: Sample rate in Hz
            fft_size: FFT size (power of 2)
            subtraction_factor: Over-subtraction (1.0-3.0)
            floor_factor: Spectral floor (0.01-0.1)
        """
        config = NoiseReductionConfig(
            method=NoiseReductionMethod.SPECTRAL_SUBTRACTION,
            fft_size=fft_size,
            subtraction_factor=subtraction_factor,
            floor_factor=floor_factor,
        )
        super().__init__(sample_rate, config)


class AdaptiveNoiseCancel(NoiseReduction):
    """
    Convenience class for adaptive noise cancellation.

    Uses NLMS algorithm for noise cancellation.
    """

    def __init__(
        self, sample_rate: float, filter_length: int = 64, step_size: float = 0.01
    ):
        """
        Initialize adaptive noise canceller.

        Args:
            sample_rate: Sample rate in Hz
            filter_length: Adaptive filter length
            step_size: LMS step size (mu)
        """
        config = NoiseReductionConfig(
            method=NoiseReductionMethod.NLMS,
            lms_filter_length=filter_length,
            lms_step_size=step_size,
        )
        super().__init__(sample_rate, config)

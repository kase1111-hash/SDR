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


class AGCMode(Enum):
    """AGC detection modes."""
    RMS = "rms"           # Root Mean Square level detection
    PEAK = "peak"         # Peak level detection
    LOG = "log"           # Logarithmic domain processing
    MAGNITUDE = "magnitude"  # Magnitude-based (good for complex signals)


@dataclass
class AGCConfig:
    """AGC configuration parameters."""
    target_level: float = 1.0      # Target output amplitude
    attack_time: float = 0.001     # Attack time in seconds (fast response)
    decay_time: float = 0.1        # Decay/release time in seconds
    hang_time: float = 0.0         # Hold time before decay starts
    max_gain: float = 100.0        # Maximum gain (linear)
    min_gain: float = 0.001        # Minimum gain (linear)
    mode: AGCMode = AGCMode.RMS    # Detection mode
    reference_level: float = 0.0   # Reference level for log mode (dB)


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

    def __init__(
        self,
        sample_rate: float,
        config: Optional[AGCConfig] = None
    ):
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
        self._gain = 1.0           # Current gain
        self._level = 0.0          # Detected signal level
        self._hang_counter = 0     # Hang time counter
        self._peak_hold = 0.0      # Peak hold for peak detector

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
            self._peak_hold *= (1.0 - self._decay_coeff)
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

    def get_gain_history(
        self,
        samples: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        self._config.attack_time = float('inf')
        self._config.decay_time = float('inf')
        self._update_coefficients()

    def unfreeze(
        self,
        attack_time: float = 0.001,
        decay_time: float = 0.1
    ) -> None:
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
        min_gain: float = 0.001
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
        block_env = np.sqrt(np.mean(magnitudes ** 2))

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


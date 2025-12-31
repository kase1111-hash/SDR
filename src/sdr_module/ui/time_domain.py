"""
Time Domain Display - Amplitude vs. time waveform visualization.

Provides real-time time-domain visualization of I/Q samples with
multiple display modes and configurable parameters.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np


class DisplayMode(Enum):
    """Time domain display modes."""

    MAGNITUDE = "magnitude"  # |I + jQ| - envelope
    I_CHANNEL = "i_channel"  # In-phase component only
    Q_CHANNEL = "q_channel"  # Quadrature component only
    IQ_OVERLAY = "iq_overlay"  # Both I and Q overlaid
    PHASE = "phase"  # Phase angle (unwrapped)
    POWER = "power"  # Power (magnitude squared) in dB


@dataclass
class TimeDomainResult:
    """Result from time domain processing."""

    time_ms: np.ndarray  # Time axis in milliseconds
    samples: int  # Number of samples
    sample_rate: float  # Sample rate in Hz
    mode: DisplayMode  # Display mode used

    # Data arrays (depending on mode)
    primary: np.ndarray  # Primary data (I, Q, mag, phase, or power)
    secondary: Optional[np.ndarray] = None  # Secondary data (Q for IQ_OVERLAY)

    # Statistics
    peak: float = 0.0  # Peak value
    rms: float = 0.0  # RMS value
    min_val: float = 0.0  # Minimum value
    max_val: float = 0.0  # Maximum value


class TimeDomainDisplay:
    """
    Time domain display for I/Q samples.

    Visualizes amplitude vs. time with multiple display modes:
    - Magnitude (envelope)
    - I channel only
    - Q channel only
    - I and Q overlaid
    - Phase
    - Power (dB)
    """

    def __init__(
        self,
        sample_rate: float,
        window_size: int = 4096,
        mode: DisplayMode = DisplayMode.MAGNITUDE,
    ):
        """
        Initialize time domain display.

        Args:
            sample_rate: Sample rate in Hz
            window_size: Number of samples to display
            mode: Initial display mode
        """
        self._sample_rate = sample_rate
        self._window_size = window_size
        self._mode = mode

        # Sample buffer
        self._buffer = np.zeros(window_size, dtype=np.complex64)
        self._buffer_valid = 0  # Number of valid samples in buffer

        # Triggering
        self._trigger_enabled = False
        self._trigger_level = 0.5
        self._trigger_edge = "rising"  # "rising" or "falling"

    @property
    def sample_rate(self) -> float:
        """Get current sample rate."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, rate: float) -> None:
        """Set sample rate."""
        self._sample_rate = rate

    @property
    def window_size(self) -> int:
        """Get current window size."""
        return self._window_size

    @window_size.setter
    def window_size(self, size: int) -> None:
        """Set window size, reallocating buffer if needed."""
        if size != self._window_size:
            self._window_size = size
            self._buffer = np.zeros(size, dtype=np.complex64)
            self._buffer_valid = 0

    @property
    def mode(self) -> DisplayMode:
        """Get current display mode."""
        return self._mode

    @mode.setter
    def mode(self, new_mode: DisplayMode) -> None:
        """Set display mode."""
        self._mode = new_mode

    @property
    def time_span_ms(self) -> float:
        """Get time span of window in milliseconds."""
        return (self._window_size / self._sample_rate) * 1000

    def set_trigger(
        self, enabled: bool = True, level: float = 0.5, edge: str = "rising"
    ) -> None:
        """
        Configure triggering for stable waveform display.

        Args:
            enabled: Enable/disable triggering
            level: Trigger level (0.0 to 1.0, normalized)
            edge: Trigger edge ("rising" or "falling")
        """
        self._trigger_enabled = enabled
        self._trigger_level = level
        self._trigger_edge = edge

    def update(self, samples: np.ndarray) -> None:
        """
        Update display with new samples.

        Args:
            samples: Complex I/Q samples
        """
        n_samples = len(samples)

        if n_samples >= self._window_size:
            # More samples than window - keep most recent
            self._buffer[:] = samples[-self._window_size :]
            self._buffer_valid = self._window_size
        else:
            # Shift buffer and append new samples
            shift = self._window_size - n_samples
            self._buffer[:shift] = self._buffer[n_samples:]
            self._buffer[shift:] = samples
            self._buffer_valid = min(self._buffer_valid + n_samples, self._window_size)

    def process(self, samples: Optional[np.ndarray] = None) -> TimeDomainResult:
        """
        Process samples and return display data.

        Args:
            samples: Optional new samples to process. If None, uses buffer.

        Returns:
            TimeDomainResult with time axis and amplitude data
        """
        if samples is not None:
            self.update(samples)

        # Get valid portion of buffer
        if self._buffer_valid < self._window_size:
            data = (
                self._buffer[-self._buffer_valid :]
                if self._buffer_valid > 0
                else self._buffer[:1]
            )
        else:
            data = self._buffer

        # Apply triggering if enabled
        if self._trigger_enabled and len(data) > 1:
            data = self._apply_trigger(data)

        # Generate time axis
        n_samples = len(data)
        time_ms = np.arange(n_samples) / self._sample_rate * 1000

        # Process based on mode
        primary, secondary = self._compute_display_data(data)

        # Compute statistics
        peak = float(np.max(np.abs(primary)))
        rms = float(np.sqrt(np.mean(primary**2)))
        min_val = float(np.min(primary))
        max_val = float(np.max(primary))

        return TimeDomainResult(
            time_ms=time_ms,
            samples=n_samples,
            sample_rate=self._sample_rate,
            mode=self._mode,
            primary=primary,
            secondary=secondary,
            peak=peak,
            rms=rms,
            min_val=min_val,
            max_val=max_val,
        )

    def _compute_display_data(
        self, data: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Compute display data based on current mode."""

        if self._mode == DisplayMode.MAGNITUDE:
            return np.abs(data).astype(np.float32), None

        elif self._mode == DisplayMode.I_CHANNEL:
            return np.real(data).astype(np.float32), None

        elif self._mode == DisplayMode.Q_CHANNEL:
            return np.imag(data).astype(np.float32), None

        elif self._mode == DisplayMode.IQ_OVERLAY:
            return (np.real(data).astype(np.float32), np.imag(data).astype(np.float32))

        elif self._mode == DisplayMode.PHASE:
            return np.unwrap(np.angle(data)).astype(np.float32), None

        elif self._mode == DisplayMode.POWER:
            power = np.abs(data) ** 2
            # Convert to dB with floor to avoid log(0)
            power_db = 10 * np.log10(power + 1e-12)
            return power_db.astype(np.float32), None

        else:
            return np.abs(data).astype(np.float32), None

    def _apply_trigger(self, data: np.ndarray) -> np.ndarray:
        """Apply triggering to find stable waveform start point."""
        # Use magnitude for triggering
        mag = np.abs(data)

        # Normalize to 0-1 range
        mag_min = np.min(mag)
        mag_max = np.max(mag)
        if mag_max - mag_min > 1e-10:
            mag_norm = (mag - mag_min) / (mag_max - mag_min)
        else:
            return data  # No variation, can't trigger

        # Find trigger point
        trigger_idx = 0

        if self._trigger_edge == "rising":
            # Find rising edge crossing trigger level
            below = mag_norm[:-1] < self._trigger_level
            above = mag_norm[1:] >= self._trigger_level
            crossings = np.where(below & above)[0]
        else:
            # Find falling edge crossing trigger level
            above = mag_norm[:-1] >= self._trigger_level
            below = mag_norm[1:] < self._trigger_level
            crossings = np.where(above & below)[0]

        if len(crossings) > 0:
            trigger_idx = crossings[0]

        # Return data starting from trigger point
        return data[trigger_idx:]

    def get_display_range(self) -> Tuple[float, float]:
        """Get appropriate Y-axis range for current mode."""
        if self._mode == DisplayMode.POWER:
            return (-80.0, 0.0)  # dB range
        elif self._mode == DisplayMode.PHASE:
            return (-np.pi * 4, np.pi * 4)  # Allow for unwrapped phase
        else:
            return (-1.5, 1.5)  # Normalized amplitude

    def clear(self) -> None:
        """Clear the display buffer."""
        self._buffer.fill(0)
        self._buffer_valid = 0

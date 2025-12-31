"""
Signal Strength Meter - Real-time RSSI/signal level indicator.

Provides signal power measurement with configurable units,
averaging, and peak hold functionality.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


class PowerUnit(Enum):
    """Power measurement units."""

    DBFS = "dBFS"  # dB relative to full scale
    DBM = "dBm"  # dB relative to 1 milliwatt
    DBU = "dBuV"  # dB relative to 1 microvolt
    LINEAR = "linear"  # Linear power (0-1 normalized)


class MeterMode(Enum):
    """Meter response mode."""

    INSTANTANEOUS = "instantaneous"  # No averaging
    AVERAGE = "average"  # RMS average
    PEAK = "peak"  # Peak hold
    PEAK_DECAY = "peak_decay"  # Peak with decay


@dataclass
class MeterReading:
    """Single meter reading with all measurements."""

    power_dbfs: float  # Power in dBFS
    power_dbm: float  # Power in dBm (requires calibration)
    power_linear: float  # Linear power (0-1)

    peak_dbfs: float  # Peak hold value
    average_dbfs: float  # Running average

    # For bar display (0-100 scale)
    bar_level: int  # Current level (0-100)
    peak_bar_level: int  # Peak level (0-100)

    # Signal quality indicators
    clipping: bool  # True if signal is clipping
    noise_floor_dbfs: float  # Estimated noise floor


@dataclass
class MeterConfig:
    """Configuration for signal strength meter."""

    unit: PowerUnit = PowerUnit.DBFS
    mode: MeterMode = MeterMode.AVERAGE

    # Averaging
    avg_time_ms: float = 100.0  # Averaging time constant
    peak_hold_ms: float = 1000.0  # Peak hold time
    peak_decay_db_per_sec: float = 20.0  # Peak decay rate

    # Calibration (for dBm conversion)
    reference_level_dbm: float = -30.0  # dBm at 0 dBFS

    # Display range
    min_dbfs: float = -100.0  # Minimum display level
    max_dbfs: float = 0.0  # Maximum display level (0 = full scale)

    # Clipping threshold
    clip_threshold: float = 0.99  # Linear level above which clipping is detected


class SignalStrengthMeter:
    """
    Real-time signal strength meter.

    Measures signal power from I/Q samples with configurable
    units, averaging, and peak hold functionality.

    Features:
    - Multiple power units (dBFS, dBm, linear)
    - Averaging with configurable time constant
    - Peak hold with decay
    - Clipping detection
    - Noise floor estimation
    - Bar meter output (0-100 scale)
    """

    def __init__(self, sample_rate: float, config: Optional[MeterConfig] = None):
        """
        Initialize signal strength meter.

        Args:
            sample_rate: Sample rate in Hz
            config: Meter configuration (uses defaults if None)
        """
        self._sample_rate = sample_rate
        self._config = config or MeterConfig()

        # State
        self._current_power = -100.0  # dBFS
        self._average_power = -100.0  # dBFS
        self._peak_power = -100.0  # dBFS
        self._noise_floor = -100.0  # dBFS

        # Timing
        self._last_update_time = 0.0
        self._peak_time = 0.0
        self._total_samples = 0

        # History for noise floor estimation
        self._power_history: List[float] = []
        self._history_max_len = 100

    @property
    def sample_rate(self) -> float:
        """Get sample rate."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, rate: float) -> None:
        """Set sample rate."""
        self._sample_rate = rate

    @property
    def config(self) -> MeterConfig:
        """Get current configuration."""
        return self._config

    @config.setter
    def config(self, cfg: MeterConfig) -> None:
        """Set configuration."""
        self._config = cfg

    def update(self, samples: np.ndarray) -> MeterReading:
        """
        Update meter with new samples and return reading.

        Args:
            samples: Complex I/Q samples

        Returns:
            MeterReading with current measurements
        """
        # Calculate instantaneous power
        power_linear = np.mean(np.abs(samples) ** 2)
        power_dbfs = self._linear_to_dbfs(power_linear)

        # Check for clipping
        peak_amplitude = np.max(np.abs(samples))
        clipping = peak_amplitude >= self._config.clip_threshold

        # Update current power
        self._current_power = power_dbfs

        # Calculate time delta
        n_samples = len(samples)
        time_delta_ms = (n_samples / self._sample_rate) * 1000
        self._total_samples += n_samples

        # Update average (exponential moving average)
        alpha = self._calculate_alpha(time_delta_ms, self._config.avg_time_ms)
        if self._average_power <= -99.0:
            self._average_power = power_dbfs
        else:
            self._average_power = alpha * power_dbfs + (1 - alpha) * self._average_power

        # Update peak
        self._update_peak(power_dbfs, time_delta_ms)

        # Update noise floor estimate
        self._update_noise_floor(power_dbfs)

        # Select output value based on mode
        if self._config.mode == MeterMode.INSTANTANEOUS:
            output_dbfs = power_dbfs
        elif self._config.mode == MeterMode.AVERAGE:
            output_dbfs = self._average_power
        elif self._config.mode in (MeterMode.PEAK, MeterMode.PEAK_DECAY):
            output_dbfs = self._peak_power
        else:
            output_dbfs = power_dbfs

        # Convert to other units
        power_dbm = self._dbfs_to_dbm(output_dbfs)
        power_linear_out = self._dbfs_to_linear(output_dbfs)

        # Calculate bar levels (0-100)
        bar_level = self._dbfs_to_bar(output_dbfs)
        peak_bar_level = self._dbfs_to_bar(self._peak_power)

        return MeterReading(
            power_dbfs=output_dbfs,
            power_dbm=power_dbm,
            power_linear=power_linear_out,
            peak_dbfs=self._peak_power,
            average_dbfs=self._average_power,
            bar_level=bar_level,
            peak_bar_level=peak_bar_level,
            clipping=clipping,
            noise_floor_dbfs=self._noise_floor,
        )

    def _calculate_alpha(self, time_delta_ms: float, time_constant_ms: float) -> float:
        """Calculate exponential moving average alpha."""
        if time_constant_ms <= 0:
            return 1.0
        return min(1.0, time_delta_ms / time_constant_ms)

    def _update_peak(self, power_dbfs: float, time_delta_ms: float) -> None:
        """Update peak hold value."""
        if power_dbfs > self._peak_power:
            self._peak_power = power_dbfs
            self._peak_time = 0.0
        else:
            self._peak_time += time_delta_ms

            if self._config.mode == MeterMode.PEAK_DECAY:
                # Decay peak over time
                decay_db = (self._config.peak_decay_db_per_sec * time_delta_ms) / 1000.0
                self._peak_power = max(power_dbfs, self._peak_power - decay_db)
            elif self._config.mode == MeterMode.PEAK:
                # Reset peak after hold time
                if self._peak_time >= self._config.peak_hold_ms:
                    self._peak_power = power_dbfs
                    self._peak_time = 0.0

    def _update_noise_floor(self, power_dbfs: float) -> None:
        """Update noise floor estimate using minimum of recent history."""
        self._power_history.append(power_dbfs)
        if len(self._power_history) > self._history_max_len:
            self._power_history.pop(0)

        # Noise floor is the 10th percentile of recent readings
        if len(self._power_history) >= 10:
            sorted_powers = sorted(self._power_history)
            idx = len(sorted_powers) // 10
            self._noise_floor = sorted_powers[idx]

    def _linear_to_dbfs(self, linear: float) -> float:
        """Convert linear power to dBFS."""
        if linear <= 0:
            return self._config.min_dbfs
        dbfs = 10 * np.log10(linear)
        return max(self._config.min_dbfs, min(self._config.max_dbfs, dbfs))

    def _dbfs_to_linear(self, dbfs: float) -> float:
        """Convert dBFS to linear power."""
        return 10 ** (dbfs / 10)

    def _dbfs_to_dbm(self, dbfs: float) -> float:
        """Convert dBFS to dBm using calibration."""
        return dbfs + self._config.reference_level_dbm

    def _dbfs_to_bar(self, dbfs: float) -> int:
        """Convert dBFS to bar level (0-100)."""
        range_db = self._config.max_dbfs - self._config.min_dbfs
        if range_db <= 0:
            return 0
        normalized = (dbfs - self._config.min_dbfs) / range_db
        return int(max(0, min(100, normalized * 100)))

    def reset(self) -> None:
        """Reset meter state."""
        self._current_power = -100.0
        self._average_power = -100.0
        self._peak_power = -100.0
        self._noise_floor = -100.0
        self._peak_time = 0.0
        self._total_samples = 0
        self._power_history.clear()

    def reset_peak(self) -> None:
        """Reset only peak hold."""
        self._peak_power = self._current_power
        self._peak_time = 0.0

    def get_snr_db(self) -> float:
        """
        Get estimated SNR based on current signal and noise floor.

        Returns:
            Estimated SNR in dB
        """
        return self._average_power - self._noise_floor

    def format_reading(self, reading: MeterReading) -> str:
        """
        Format reading as string for display.

        Args:
            reading: Meter reading to format

        Returns:
            Formatted string
        """
        unit = self._config.unit

        if unit == PowerUnit.DBFS:
            return f"{reading.power_dbfs:.1f} dBFS"
        elif unit == PowerUnit.DBM:
            return f"{reading.power_dbm:.1f} dBm"
        elif unit == PowerUnit.DBU:
            # dBuV = dBm + 107 (in 50 ohm system)
            dbuv = reading.power_dbm + 107
            return f"{dbuv:.1f} dBµV"
        elif unit == PowerUnit.LINEAR:
            return f"{reading.power_linear:.4f}"
        else:
            return f"{reading.power_dbfs:.1f} dBFS"

    def render_bar(self, reading: MeterReading, width: int = 40) -> str:
        """
        Render ASCII bar meter.

        Args:
            reading: Meter reading to render
            width: Bar width in characters

        Returns:
            ASCII bar string
        """
        filled = int(reading.bar_level * width / 100)
        peak_pos = int(reading.peak_bar_level * width / 100)

        bar = ""
        for i in range(width):
            if i < filled:
                bar += "█"
            elif i == peak_pos:
                bar += "│"
            else:
                bar += "░"

        # Add clipping indicator
        clip_ind = "!" if reading.clipping else " "

        return f"[{bar}]{clip_ind}"

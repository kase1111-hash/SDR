"""
Frequency Locking - Automatic signal detection and tracking.

Provides frequency lock functionality to "zero in" on detected signals,
track signal drift, and maintain lock on moving signals.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


class LockState(Enum):
    """Frequency lock state."""

    UNLOCKED = "unlocked"  # No lock, searching
    ACQUIRING = "acquiring"  # Signal detected, acquiring lock
    LOCKED = "locked"  # Locked onto signal
    LOST = "lost"  # Lock was lost


class LockMode(Enum):
    """Frequency lock mode."""

    PEAK = "peak"  # Lock to strongest peak
    NEAREST = "nearest"  # Lock to nearest signal to current freq
    MANUAL = "manual"  # Lock to manually specified frequency


@dataclass
class LockTarget:
    """Information about a lock target signal."""

    frequency_hz: float  # Center frequency in Hz
    power_db: float  # Signal power in dB
    bandwidth_hz: float  # Estimated signal bandwidth
    snr_db: float  # Signal-to-noise ratio


@dataclass
class LockStatus:
    """Current frequency lock status."""

    state: LockState  # Lock state
    target_freq_hz: float  # Target frequency
    offset_hz: float  # Offset from center frequency
    error_hz: float  # Frequency error (for tracking)
    power_db: float  # Current signal power
    snr_db: float  # Current SNR
    lock_quality: float  # Lock quality (0-1)
    time_locked_ms: float  # Time locked in milliseconds


@dataclass
class LockConfig:
    """Configuration for frequency locker."""

    mode: LockMode = LockMode.PEAK

    # Detection thresholds
    min_snr_db: float = 10.0  # Minimum SNR to acquire lock
    min_power_db: float = -80.0  # Minimum power to consider

    # Lock behavior
    lock_bandwidth_hz: float = 10000.0  # Bandwidth around target to consider locked
    acquire_time_ms: float = 100.0  # Time to hold signal before lock
    lost_time_ms: float = 500.0  # Time without signal before lost

    # Tracking
    max_drift_hz_per_sec: float = 1000.0  # Maximum expected drift rate
    tracking_bandwidth_hz: float = 5000.0  # Search bandwidth when tracking

    # Update timing
    update_interval_ms: float = 10.0  # Expected interval between updates (for drift calculation)


class FrequencyLocker:
    """
    Frequency locker for automatic signal detection and tracking.

    Features:
    - Automatic signal detection from spectrum
    - Lock to strongest or nearest signal
    - Drift tracking to follow moving signals
    - Lock quality estimation
    - Hysteresis to prevent lock chatter
    """

    def __init__(
        self,
        sample_rate: float,
        fft_size: int = 4096,
        config: Optional[LockConfig] = None,
    ):
        """
        Initialize frequency locker.

        Args:
            sample_rate: Sample rate in Hz
            fft_size: FFT size for frequency resolution
            config: Lock configuration (uses defaults if None)
        """
        self._sample_rate = sample_rate
        self._fft_size = fft_size
        self._config = config or LockConfig()

        # Frequency resolution
        self._freq_resolution = sample_rate / fft_size

        # State
        self._state = LockState.UNLOCKED
        self._target_freq = 0.0
        self._lock_start_time = 0.0
        self._time_in_state_ms = 0.0
        self._last_power_db = -100.0
        self._last_snr_db = 0.0

        # Tracking history
        self._freq_history: List[float] = []
        self._history_max_len = 50

        # Noise floor estimation
        self._noise_floor_db = -100.0

    @property
    def sample_rate(self) -> float:
        """Get sample rate."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, rate: float) -> None:
        """Set sample rate."""
        self._sample_rate = rate
        self._freq_resolution = rate / self._fft_size

    @property
    def config(self) -> LockConfig:
        """Get current configuration."""
        return self._config

    @config.setter
    def config(self, cfg: LockConfig) -> None:
        """Set configuration."""
        self._config = cfg

    @property
    def state(self) -> LockState:
        """Get current lock state."""
        return self._state

    @property
    def target_frequency(self) -> float:
        """Get current target frequency."""
        return self._target_freq

    def update(
        self,
        spectrum_db: np.ndarray,
        center_freq: float = 0.0,
        time_delta_ms: float = 10.0,
    ) -> LockStatus:
        """
        Update lock state with new spectrum data.

        Args:
            spectrum_db: Power spectrum in dB (FFT bins)
            center_freq: Center frequency of spectrum in Hz
            time_delta_ms: Time since last update in milliseconds

        Returns:
            LockStatus with current lock information
        """
        # Update noise floor estimate
        self._update_noise_floor(spectrum_db)

        # Find signals in spectrum
        signals = self._detect_signals(spectrum_db, center_freq)

        # Update state machine
        self._time_in_state_ms += time_delta_ms

        if self._state == LockState.UNLOCKED:
            self._handle_unlocked(signals, time_delta_ms)
        elif self._state == LockState.ACQUIRING:
            self._handle_acquiring(signals, time_delta_ms)
        elif self._state == LockState.LOCKED:
            self._handle_locked(signals, center_freq, time_delta_ms)
        elif self._state == LockState.LOST:
            self._handle_lost(signals, time_delta_ms)

        # Calculate offset from center
        offset_hz = self._target_freq - center_freq

        # Calculate frequency error (difference from expected)
        error_hz = 0.0
        if len(self._freq_history) >= 2:
            expected = self._freq_history[-1]
            error_hz = self._target_freq - expected

        # Calculate lock quality
        lock_quality = self._calculate_lock_quality()

        return LockStatus(
            state=self._state,
            target_freq_hz=self._target_freq,
            offset_hz=offset_hz,
            error_hz=error_hz,
            power_db=self._last_power_db,
            snr_db=self._last_snr_db,
            lock_quality=lock_quality,
            time_locked_ms=(
                self._time_in_state_ms if self._state == LockState.LOCKED else 0.0
            ),
        )

    def _detect_signals(
        self, spectrum_db: np.ndarray, center_freq: float
    ) -> List[LockTarget]:
        """Detect signals in spectrum above threshold."""
        signals = []

        n_bins = len(spectrum_db)
        freq_per_bin = self._sample_rate / n_bins

        # Threshold is noise floor + minimum SNR
        threshold_db = self._noise_floor_db + self._config.min_snr_db

        # Also check absolute minimum power
        threshold_db = max(threshold_db, self._config.min_power_db)

        # Find peaks above threshold
        above_threshold = spectrum_db > threshold_db

        # Find contiguous regions above threshold
        regions = self._find_regions(above_threshold)

        for start, end in regions:
            # Find peak within region
            region_spectrum = spectrum_db[start:end]
            peak_idx_local = np.argmax(region_spectrum)
            peak_idx = start + peak_idx_local
            peak_power = spectrum_db[peak_idx]

            # Calculate frequency (centered FFT)
            if peak_idx < n_bins // 2:
                freq_offset = peak_idx * freq_per_bin
            else:
                freq_offset = (peak_idx - n_bins) * freq_per_bin

            signal_freq = center_freq + freq_offset

            # Estimate bandwidth (3 dB down from peak)
            bandwidth = self._estimate_bandwidth(spectrum_db, peak_idx, freq_per_bin)

            # Calculate SNR
            snr_db = peak_power - self._noise_floor_db

            if snr_db >= self._config.min_snr_db:
                signals.append(
                    LockTarget(
                        frequency_hz=signal_freq,
                        power_db=peak_power,
                        bandwidth_hz=bandwidth,
                        snr_db=snr_db,
                    )
                )

        return signals

    def _find_regions(self, mask: np.ndarray) -> List[Tuple[int, int]]:
        """Find contiguous regions in boolean mask."""
        regions = []
        in_region = False
        start = 0

        for i, val in enumerate(mask):
            if val and not in_region:
                start = i
                in_region = True
            elif not val and in_region:
                regions.append((start, i))
                in_region = False

        if in_region:
            regions.append((start, len(mask)))

        return regions

    def _estimate_bandwidth(
        self, spectrum_db: np.ndarray, peak_idx: int, freq_per_bin: float
    ) -> float:
        """Estimate signal bandwidth using 3 dB method."""
        peak_power = spectrum_db[peak_idx]
        threshold = peak_power - 3.0  # 3 dB down

        # Search left
        left = peak_idx
        while left > 0 and spectrum_db[left] > threshold:
            left -= 1

        # Search right
        right = peak_idx
        while right < len(spectrum_db) - 1 and spectrum_db[right] > threshold:
            right += 1

        bandwidth_bins = right - left
        return bandwidth_bins * freq_per_bin

    def _update_noise_floor(self, spectrum_db: np.ndarray) -> None:
        """Update noise floor estimate."""
        # Use 10th percentile as noise floor
        self._noise_floor_db = float(np.percentile(spectrum_db, 10))

    def _handle_unlocked(self, signals: List[LockTarget], time_delta_ms: float) -> None:
        """Handle unlocked state."""
        target = self._select_target(signals)

        if target is not None:
            self._target_freq = target.frequency_hz
            self._last_power_db = target.power_db
            self._last_snr_db = target.snr_db
            self._state = LockState.ACQUIRING
            self._time_in_state_ms = 0.0
            self._freq_history.clear()
            self._freq_history.append(target.frequency_hz)

    def _handle_acquiring(
        self, signals: List[LockTarget], time_delta_ms: float
    ) -> None:
        """Handle acquiring state."""
        # Look for signal near target
        target = self._find_signal_near(signals, self._target_freq)

        if target is not None:
            # Update target frequency (track during acquisition)
            self._target_freq = target.frequency_hz
            self._last_power_db = target.power_db
            self._last_snr_db = target.snr_db
            self._freq_history.append(target.frequency_hz)

            # Check if we've held lock long enough
            if self._time_in_state_ms >= self._config.acquire_time_ms:
                self._state = LockState.LOCKED
                self._time_in_state_ms = 0.0
        else:
            # Lost signal during acquisition
            self._state = LockState.UNLOCKED
            self._time_in_state_ms = 0.0

    def _handle_locked(
        self, signals: List[LockTarget], center_freq: float, time_delta_ms: float
    ) -> None:
        """Handle locked state."""
        # Look for signal near target (with tracking bandwidth)
        target = self._find_signal_near(signals, self._target_freq)

        if target is not None:
            # Update target with tracking
            self._target_freq = target.frequency_hz
            self._last_power_db = target.power_db
            self._last_snr_db = target.snr_db

            # Update history for drift tracking
            self._freq_history.append(target.frequency_hz)
            if len(self._freq_history) > self._history_max_len:
                self._freq_history.pop(0)
        else:
            # Signal lost
            self._state = LockState.LOST
            self._time_in_state_ms = 0.0

    def _handle_lost(self, signals: List[LockTarget], time_delta_ms: float) -> None:
        """Handle lost state."""
        # Try to reacquire near last known frequency
        target = self._find_signal_near(signals, self._target_freq)

        if target is not None:
            # Reacquired
            self._target_freq = target.frequency_hz
            self._last_power_db = target.power_db
            self._last_snr_db = target.snr_db
            self._state = LockState.LOCKED
            self._time_in_state_ms = 0.0
        elif self._time_in_state_ms >= self._config.lost_time_ms:
            # Lost for too long, go back to unlocked
            self._state = LockState.UNLOCKED
            self._time_in_state_ms = 0.0

    def _select_target(self, signals: List[LockTarget]) -> Optional[LockTarget]:
        """Select target signal based on lock mode."""
        if not signals:
            return None

        if self._config.mode == LockMode.PEAK:
            # Select strongest signal
            return max(signals, key=lambda s: s.power_db)

        elif self._config.mode == LockMode.NEAREST:
            # Select nearest to current target
            if self._target_freq == 0:
                return max(signals, key=lambda s: s.power_db)
            return min(signals, key=lambda s: abs(s.frequency_hz - self._target_freq))

        elif self._config.mode == LockMode.MANUAL:
            # In manual mode, only lock if signal is near target
            return self._find_signal_near(signals, self._target_freq)

        return None

    def _find_signal_near(
        self, signals: List[LockTarget], target_freq: float
    ) -> Optional[LockTarget]:
        """Find signal within tracking bandwidth of target."""
        for signal in signals:
            if (
                abs(signal.frequency_hz - target_freq)
                <= self._config.tracking_bandwidth_hz
            ):
                return signal
        return None

    def _calculate_lock_quality(self) -> float:
        """Calculate lock quality (0-1) based on signal metrics."""
        if self._state not in (LockState.LOCKED, LockState.ACQUIRING):
            return 0.0

        # Quality based on SNR (10 dB = 0.5, 30 dB = 1.0)
        snr_quality = min(1.0, max(0.0, (self._last_snr_db - 10) / 20))

        # Quality based on frequency stability
        if len(self._freq_history) >= 3:
            freq_std = np.std(self._freq_history[-10:])
            max_std = self._config.tracking_bandwidth_hz / 2
            stability_quality = max(0.0, 1.0 - freq_std / max_std)
        else:
            stability_quality = 0.5

        # Combined quality
        return 0.6 * snr_quality + 0.4 * stability_quality

    def lock_to_frequency(self, frequency_hz: float) -> None:
        """
        Manually lock to a specific frequency.

        Args:
            frequency_hz: Target frequency in Hz
        """
        self._target_freq = frequency_hz
        self._state = LockState.ACQUIRING
        self._time_in_state_ms = 0.0
        self._freq_history.clear()
        self._config.mode = LockMode.MANUAL

    def unlock(self) -> None:
        """Release frequency lock."""
        self._state = LockState.UNLOCKED
        self._time_in_state_ms = 0.0
        self._target_freq = 0.0
        self._freq_history.clear()

    def get_drift_rate(self) -> float:
        """
        Get estimated frequency drift rate.

        Returns:
            Drift rate in Hz/second
        """
        if len(self._freq_history) < 2:
            return 0.0

        # Simple linear regression over history
        n = len(self._freq_history)
        x = np.arange(n)
        y = np.array(self._freq_history)

        # Least squares fit
        slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
            n * np.sum(x**2) - np.sum(x) ** 2
        )

        # Convert to Hz/second using configured update interval
        samples_per_second = 1000.0 / self._config.update_interval_ms
        return slope * samples_per_second

"""
Automatic Frequency Control (AFC) - Drift compensation.

Measures and corrects frequency drift in received signals,
keeping the signal centered in the passband.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Callable
import numpy as np


class AFCMode(Enum):
    """AFC operating mode."""
    OFF = "off"                 # AFC disabled
    HOLD = "hold"               # Hold current correction, don't update
    SLOW = "slow"               # Slow tracking for stable signals
    MEDIUM = "medium"           # Medium tracking speed
    FAST = "fast"               # Fast tracking for drifting signals


class AFCMethod(Enum):
    """Frequency error detection method."""
    FFT_PEAK = "fft_peak"       # FFT peak detection
    PHASE_DIFF = "phase_diff"   # Phase difference method
    CORRELATION = "correlation"  # Cross-correlation method


@dataclass
class AFCStatus:
    """AFC status information."""
    mode: AFCMode               # Current mode
    enabled: bool               # AFC is active
    locked: bool                # AFC is locked to signal
    frequency_error_hz: float   # Measured frequency error
    correction_hz: float        # Applied correction
    drift_rate_hz_per_sec: float  # Estimated drift rate
    signal_present: bool        # Signal detected


@dataclass
class AFCConfig:
    """AFC configuration."""
    mode: AFCMode = AFCMode.MEDIUM
    method: AFCMethod = AFCMethod.FFT_PEAK

    # Loop parameters
    loop_bandwidth_hz: float = 100.0    # Control loop bandwidth
    damping_factor: float = 0.707       # Loop damping (0.707 = critically damped)

    # Limits
    max_correction_hz: float = 50000.0  # Maximum correction range
    max_rate_hz_per_sec: float = 1000.0  # Maximum correction rate

    # Detection
    min_signal_db: float = -60.0        # Minimum signal level for AFC
    lock_threshold_hz: float = 100.0    # Error threshold for "locked" state

    # FFT parameters (for FFT_PEAK method)
    fft_size: int = 1024


class AutomaticFrequencyControl:
    """
    Automatic Frequency Control for drift compensation.

    Measures frequency error and generates correction signal
    to keep the received signal centered in the passband.

    Features:
    - Multiple detection methods (FFT, phase difference)
    - Configurable loop bandwidth and damping
    - Drift rate estimation
    - Lock detection
    - Correction limiting
    """

    def __init__(
        self,
        sample_rate: float,
        config: Optional[AFCConfig] = None
    ):
        """
        Initialize AFC.

        Args:
            sample_rate: Sample rate in Hz
            config: AFC configuration (uses defaults if None)
        """
        self._sample_rate = sample_rate
        self._config = config or AFCConfig()

        # State
        self._correction_hz = 0.0
        self._error_hz = 0.0
        self._integrator = 0.0
        self._locked = False
        self._signal_present = False

        # History for drift estimation
        self._error_history: list = []
        self._time_history: list = []
        self._total_time_ms = 0.0

        # Previous samples for phase method
        self._prev_samples: Optional[np.ndarray] = None

        # Callback for frequency updates
        self._on_correction: Optional[Callable[[float], None]] = None

        # Calculate loop coefficients
        self._update_loop_coefficients()

    def _update_loop_coefficients(self) -> None:
        """Calculate PI loop coefficients from bandwidth and damping."""
        # Second-order loop coefficients
        omega_n = 2 * np.pi * self._config.loop_bandwidth_hz
        zeta = self._config.damping_factor

        # Proportional and integral gains
        self._kp = 2 * zeta * omega_n
        self._ki = omega_n ** 2

        # Timing factor (normalized to 1 second)
        self._loop_gain = 1.0

    @property
    def sample_rate(self) -> float:
        """Get sample rate."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, rate: float) -> None:
        """Set sample rate."""
        self._sample_rate = rate

    @property
    def config(self) -> AFCConfig:
        """Get current configuration."""
        return self._config

    @config.setter
    def config(self, cfg: AFCConfig) -> None:
        """Set configuration."""
        self._config = cfg
        self._update_loop_coefficients()

    @property
    def correction(self) -> float:
        """Get current frequency correction in Hz."""
        return self._correction_hz

    @property
    def is_locked(self) -> bool:
        """Check if AFC is locked."""
        return self._locked

    def set_correction_callback(self, callback: Callable[[float], None]) -> None:
        """
        Set callback for frequency correction updates.

        Args:
            callback: Function called with correction value in Hz
        """
        self._on_correction = callback

    def update(
        self,
        samples: np.ndarray,
        time_delta_ms: float = 10.0
    ) -> AFCStatus:
        """
        Update AFC with new samples.

        Args:
            samples: Complex I/Q samples
            time_delta_ms: Time since last update in milliseconds

        Returns:
            AFCStatus with current AFC state
        """
        self._total_time_ms += time_delta_ms

        # Check if AFC is enabled
        if self._config.mode == AFCMode.OFF:
            return self._make_status()

        # Detect frequency error
        error_hz, signal_present = self._detect_error(samples)
        self._signal_present = signal_present

        if not signal_present:
            # No signal - hold current correction
            return self._make_status()

        self._error_hz = error_hz

        # Update error history for drift estimation
        self._error_history.append(error_hz)
        self._time_history.append(self._total_time_ms)
        if len(self._error_history) > 100:
            self._error_history.pop(0)
            self._time_history.pop(0)

        # Check if in HOLD mode
        if self._config.mode == AFCMode.HOLD:
            return self._make_status()

        # Apply loop filter and update correction
        self._update_correction(error_hz, time_delta_ms)

        # Check lock status
        self._locked = abs(error_hz) < self._config.lock_threshold_hz

        # Call callback if set
        if self._on_correction is not None:
            self._on_correction(self._correction_hz)

        return self._make_status()

    def _detect_error(self, samples: np.ndarray) -> tuple:
        """
        Detect frequency error using configured method.

        Returns:
            (error_hz, signal_present) tuple
        """
        if self._config.method == AFCMethod.FFT_PEAK:
            return self._detect_error_fft(samples)
        elif self._config.method == AFCMethod.PHASE_DIFF:
            return self._detect_error_phase(samples)
        elif self._config.method == AFCMethod.CORRELATION:
            return self._detect_error_correlation(samples)
        else:
            return self._detect_error_fft(samples)

    def _detect_error_fft(self, samples: np.ndarray) -> tuple:
        """Detect frequency error using FFT peak."""
        # Compute FFT
        n = min(len(samples), self._config.fft_size)
        windowed = samples[:n] * np.hanning(n)
        spectrum = np.fft.fft(windowed, self._config.fft_size)
        power = np.abs(spectrum) ** 2

        # Check signal level
        peak_power_db = 10 * np.log10(np.max(power) + 1e-12)
        10 * np.log10(np.median(power) + 1e-12)

        if peak_power_db < self._config.min_signal_db:
            return 0.0, False

        # Find peak bin
        peak_bin = np.argmax(power)

        # Parabolic interpolation for sub-bin accuracy
        if 0 < peak_bin < len(power) - 1:
            alpha = np.log(power[peak_bin - 1] + 1e-12)
            beta = np.log(power[peak_bin] + 1e-12)
            gamma = np.log(power[peak_bin + 1] + 1e-12)
            delta = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
            peak_bin_interp = peak_bin + delta
        else:
            peak_bin_interp = peak_bin

        # Convert bin to frequency
        freq_per_bin = self._sample_rate / self._config.fft_size

        if peak_bin_interp < self._config.fft_size / 2:
            error_hz = peak_bin_interp * freq_per_bin
        else:
            error_hz = (peak_bin_interp - self._config.fft_size) * freq_per_bin

        return error_hz, True

    def _detect_error_phase(self, samples: np.ndarray) -> tuple:
        """Detect frequency error using phase difference method."""
        # Check signal level
        power = np.mean(np.abs(samples) ** 2)
        power_db = 10 * np.log10(power + 1e-12)

        if power_db < self._config.min_signal_db:
            self._prev_samples = samples[-100:].copy()
            return 0.0, False

        if self._prev_samples is None:
            self._prev_samples = samples[-100:].copy()
            return 0.0, True

        # Compute average phase difference between consecutive samples
        phase_diff = np.angle(samples[1:] * np.conj(samples[:-1]))
        avg_phase_diff = np.mean(phase_diff)

        # Convert phase difference to frequency
        # f = (delta_phase / 2π) * sample_rate
        error_hz = (avg_phase_diff / (2 * np.pi)) * self._sample_rate

        self._prev_samples = samples[-100:].copy()
        return error_hz, True

    def _detect_error_correlation(self, samples: np.ndarray) -> tuple:
        """Detect frequency error using cross-correlation."""
        if self._prev_samples is None or len(self._prev_samples) < 100:
            self._prev_samples = samples.copy()
            return 0.0, False

        # Check signal level
        power = np.mean(np.abs(samples) ** 2)
        power_db = 10 * np.log10(power + 1e-12)

        if power_db < self._config.min_signal_db:
            self._prev_samples = samples.copy()
            return 0.0, False

        # Cross-correlate with previous samples
        n = min(len(samples), len(self._prev_samples), 256)
        corr = np.correlate(samples[:n], self._prev_samples[:n], mode='full')

        # Find peak of correlation
        peak_idx = np.argmax(np.abs(corr))
        center = len(corr) // 2

        # Time offset in samples
        time_offset = peak_idx - center

        # Frequency error from time offset (approximate)
        if abs(time_offset) > 0:
            error_hz = time_offset * (self._sample_rate / n)
        else:
            # Use phase method for fine estimation
            phase_diff = np.angle(np.sum(samples[:n] * np.conj(self._prev_samples[:n])))
            error_hz = (phase_diff / (2 * np.pi)) * (self._sample_rate / n)

        self._prev_samples = samples.copy()
        return error_hz, True

    def _update_correction(self, error_hz: float, time_delta_ms: float) -> None:
        """Update correction using PI loop filter."""
        # Get loop speed factor based on mode
        speed_factor = self._get_speed_factor()

        # Time in seconds
        dt = time_delta_ms / 1000.0

        # PI controller
        # Proportional term
        p_term = self._kp * error_hz * speed_factor

        # Integral term
        self._integrator += self._ki * error_hz * dt * speed_factor

        # Limit integrator (anti-windup)
        max_int = self._config.max_correction_hz * 0.8
        self._integrator = np.clip(self._integrator, -max_int, max_int)

        # Calculate correction
        correction = p_term + self._integrator

        # Rate limit
        max_change = self._config.max_rate_hz_per_sec * dt
        correction_change = correction - self._correction_hz
        correction_change = np.clip(correction_change, -max_change, max_change)

        # Apply correction
        self._correction_hz += correction_change

        # Limit total correction
        self._correction_hz = np.clip(
            self._correction_hz,
            -self._config.max_correction_hz,
            self._config.max_correction_hz
        )

    def _get_speed_factor(self) -> float:
        """Get speed factor based on AFC mode."""
        factors = {
            AFCMode.OFF: 0.0,
            AFCMode.HOLD: 0.0,
            AFCMode.SLOW: 0.1,
            AFCMode.MEDIUM: 1.0,
            AFCMode.FAST: 5.0,
        }
        return factors.get(self._config.mode, 1.0)

    def _make_status(self) -> AFCStatus:
        """Create current status object."""
        return AFCStatus(
            mode=self._config.mode,
            enabled=self._config.mode not in (AFCMode.OFF, AFCMode.HOLD),
            locked=self._locked,
            frequency_error_hz=self._error_hz,
            correction_hz=self._correction_hz,
            drift_rate_hz_per_sec=self.get_drift_rate(),
            signal_present=self._signal_present,
        )

    def get_drift_rate(self) -> float:
        """
        Get estimated frequency drift rate.

        Returns:
            Drift rate in Hz/second
        """
        if len(self._error_history) < 10:
            return 0.0

        # Linear regression on error history
        n = len(self._error_history)
        t = np.array(self._time_history[-n:]) / 1000.0  # Convert to seconds
        e = np.array(self._error_history[-n:])

        # Least squares fit
        t_mean = np.mean(t)
        e_mean = np.mean(e)
        numerator = np.sum((t - t_mean) * (e - e_mean))
        denominator = np.sum((t - t_mean) ** 2)

        if abs(denominator) < 1e-10:
            return 0.0

        return numerator / denominator

    def reset(self) -> None:
        """Reset AFC state."""
        self._correction_hz = 0.0
        self._error_hz = 0.0
        self._integrator = 0.0
        self._locked = False
        self._signal_present = False
        self._error_history.clear()
        self._time_history.clear()
        self._prev_samples = None

    def set_correction(self, correction_hz: float) -> None:
        """
        Manually set correction value.

        Args:
            correction_hz: Correction in Hz
        """
        self._correction_hz = np.clip(
            correction_hz,
            -self._config.max_correction_hz,
            self._config.max_correction_hz
        )
        self._integrator = correction_hz  # Preset integrator

    def apply_correction(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply frequency correction to samples.

        Args:
            samples: Complex I/Q samples

        Returns:
            Frequency-corrected samples
        """
        if abs(self._correction_hz) < 0.01:
            return samples

        # Generate correction tone
        # Positive correction shifts signal down in frequency
        n = len(samples)
        t = np.arange(n) / self._sample_rate
        correction_tone = np.exp(-2j * np.pi * self._correction_hz * t)

        # Apply correction
        return samples * correction_tone.astype(np.complex64)

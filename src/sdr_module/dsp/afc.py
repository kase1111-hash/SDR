"""
Automatic Frequency Control (AFC) - Drift compensation.

Measures and corrects frequency drift in received signals,
keeping the signal centered in the passband.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import numpy as np


class AFCMode(Enum):
    """AFC operating mode."""

    OFF = "off"  # AFC disabled
    HOLD = "hold"  # Hold current correction, don't update
    SLOW = "slow"  # Slow tracking for stable signals
    MEDIUM = "medium"  # Medium tracking speed
    FAST = "fast"  # Fast tracking for drifting signals


class AFCMethod(Enum):
    """Frequency error detection method."""

    FFT_PEAK = "fft_peak"  # FFT peak detection
    PHASE_DIFF = "phase_diff"  # Phase difference method
    CORRELATION = "correlation"  # Cross-correlation method


@dataclass
class AFCStatus:
    """AFC status information."""

    mode: AFCMode  # Current mode
    enabled: bool  # AFC is active
    locked: bool  # AFC is locked to signal
    frequency_error_hz: float  # Measured frequency error
    correction_hz: float  # Applied correction
    drift_rate_hz_per_sec: float  # Estimated drift rate
    signal_present: bool  # Signal detected


@dataclass
class AFCConfig:
    """AFC configuration."""

    mode: AFCMode = AFCMode.MEDIUM
    method: AFCMethod = AFCMethod.FFT_PEAK

    # Loop parameters
    loop_bandwidth_hz: float = 100.0  # Control loop bandwidth
    damping_factor: float = 0.707  # Loop damping (0.707 = critically damped)

    # Limits
    max_correction_hz: float = 50000.0  # Maximum correction range
    max_rate_hz_per_sec: float = 50000.0  # Maximum correction rate

    # Detection
    min_signal_db: float = -120.0  # Absolute floor for "a signal is there"
    min_snr_db: float = 6.0  # ... and how far it must stand above the noise
    lock_threshold_hz: float = 100.0  # Error threshold for "locked" state

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

    def __init__(self, sample_rate: float, config: Optional[AFCConfig] = None):
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
        # Running NCO phase so the correction tone is continuous across blocks.
        self._nco_phase = 0.0

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
        """Natural frequency and damping of the second-order loop.

        The gains per update follow from these and the time between updates
        (see :meth:`_update_correction`): the previous continuous-time gains
        (2 zeta wn and wn^2, about 900 and 400000 for 100 Hz) were applied
        per update as they were, so the loop either slewed at its rate limit
        or oscillated, and never settled.
        """
        self._omega_n = 2 * np.pi * self._config.loop_bandwidth_hz
        self._zeta = self._config.damping_factor
        # Kept for callers that read them (continuous-time gains).
        self._kp = 2 * self._zeta * self._omega_n
        self._ki = self._omega_n**2
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

    def update(self, samples: np.ndarray, time_delta_ms: float = 10.0) -> AFCStatus:
        """
        Update AFC with new samples.

        The loop is closed: ``samples`` are expected *after* the correction
        (see :meth:`apply_correction`, or a receiver tuned by it), so what
        is measured is the error left, which the loop drives to zero; the
        correction then equals the signal's offset (and follows its drift).

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
        """Detect frequency error using FFT peak (averaged over the block's
        segments; present when the peak stands ``min_snr_db`` above the
        median bin)."""
        size = self._config.fft_size
        x = np.asarray(samples)
        n = len(x)
        if n == 0:
            return 0.0, False
        count = max(1, n // size)
        seg = min(n, size)
        window = np.hanning(seg)
        power = np.zeros(size)
        for k in range(count):
            part = x[k * seg : (k + 1) * seg] * window
            power += np.abs(np.fft.fft(part, size)) ** 2
        power /= count * float(np.sum(window)) ** 2

        # Check signal level
        peak_power_db = 10 * np.log10(np.max(power) + 1e-30)
        floor_db = 10 * np.log10(np.median(power) + 1e-30)

        if (
            peak_power_db < self._config.min_signal_db
            or peak_power_db - floor_db < self._config.min_snr_db
        ):
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
        """Detect frequency error with the power-weighted mean phase step
        (the mean instantaneous frequency: right for FM, whose modulation
        averages out, and for AM's carrier). Present when the steps are
        coherent enough to stand ``min_snr_db`` above noise."""
        x = np.asarray(samples, dtype=np.complex128)
        if len(x) < 2:
            return 0.0, False
        power = float(np.mean(np.abs(x) ** 2))
        power_db = 10 * np.log10(power + 1e-30)
        steps = x[1:] * np.conj(x[:-1])
        total = np.sum(steps)
        # |mean step| / mean power: 1 for a clean signal, ~0 for noise
        # (at per-sample SNR r it is about r / (1 + r)).
        coherence = float(np.abs(total)) / max(len(steps) * power, 1e-30)
        snr = coherence / max(1.0 - coherence, 1e-6)
        self._prev_samples = x[-100:].astype(np.complex64)
        if (
            power_db < self._config.min_signal_db
            or 10 * np.log10(snr + 1e-30) < self._config.min_snr_db - 6.0
        ):
            return 0.0, False
        error_hz = float(np.angle(total)) / (2 * np.pi) * self._sample_rate
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
        corr = np.correlate(samples[:n], self._prev_samples[:n], mode="full")

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
        """Second-order (PI) loop on the error left after the correction.

        Per update of ``dt`` seconds: the correction moves by
        ``2 zeta wn dt`` of the error plus the integrator, which gathers
        ``(wn dt)^2`` of it (and so learns a steady drift). ``wn dt`` is
        kept below 0.5, so a fast loop with slow updates stays stable.
        """
        speed = self._get_speed_factor()
        dt = max(time_delta_ms, 0.0) / 1000.0
        if speed <= 0 or dt <= 0:
            return
        w = min(self._omega_n * speed * dt, 0.5)
        kp = 2.0 * self._zeta * w
        ki = w * w

        self._integrator += ki * error_hz
        # Anti-windup: the integrator is a drift per update.
        limit = self._config.max_rate_hz_per_sec * dt
        self._integrator = float(np.clip(self._integrator, -limit, limit))

        change = kp * error_hz + self._integrator
        change = float(np.clip(change, -limit, limit))
        self._correction_hz = float(
            np.clip(
                self._correction_hz + change,
                -self._config.max_correction_hz,
                self._config.max_correction_hz,
            )
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
        self._nco_phase = 0.0

    def set_correction(self, correction_hz: float) -> None:
        """
        Manually set correction value.

        Args:
            correction_hz: Correction in Hz
        """
        self._correction_hz = float(
            np.clip(
                correction_hz,
                -self._config.max_correction_hz,
                self._config.max_correction_hz,
            )
        )
        self._integrator = 0.0  # (a correction, not a drift)

    def apply_correction(self, samples: np.ndarray) -> np.ndarray:
        """
        Apply frequency correction to samples.

        Args:
            samples: Complex I/Q samples

        Returns:
            Frequency-corrected samples
        """
        n = len(samples)
        if n == 0:
            return samples

        # Generate the correction tone from a running phase accumulator so the
        # tone is continuous from one block to the next. Rebuilding the time
        # vector from zero each call restarted the phase at every block
        # boundary, injecting a phase jump that corrupted phase-sensitive
        # downstream processing (e.g. PSK demodulation).
        # Positive correction shifts the signal down in frequency.
        phase = self._nco_phase - 2 * np.pi * self._correction_hz * (
            np.arange(n) / self._sample_rate
        )
        correction_tone = np.exp(1j * phase)

        # Advance and wrap the accumulator for the next block.
        self._nco_phase = float(
            (self._nco_phase - 2 * np.pi * self._correction_hz * n / self._sample_rate)
            % (2 * np.pi)
        )

        return samples * correction_tone.astype(np.complex64)


# --------------------------------------------------------------------------- #
# Keeping the tuned frequency on a drifting signal (Radio > Track Signal)
# --------------------------------------------------------------------------- #


@dataclass
class TrackStatus:
    """What :class:`SignalTracker` did with the last block."""

    nudge_hz: float  # move the tuned frequency (VFO) by this now
    correction_hz: float  # total since tracking started
    error_hz: float  # where the signal is, relative to the VFO (last)
    signal_present: bool
    locked: bool
    lost: bool  # nothing to follow for a while


#: Modes tracking works in, and how the signal's frequency is measured:
#: the carrier's peak (AM, CW) or the mean frequency (FM).
TRACK_METHODS = {
    "FM": AFCMethod.PHASE_DIFF,
    "AM": AFCMethod.FFT_PEAK,
    "CW": AFCMethod.FFT_PEAK,
}


class SignalTracker:
    """Keeps the tuned frequency (the VFO) on a drifting signal.

    Every block of the receiver goes through :meth:`process`: the channel
    around the VFO is cut out (a few channel widths, so a signal that
    drifted off the filter is still seen), the signal's frequency error
    measured (:class:`AutomaticFrequencyControl`, closed loop: the VFO is
    its oscillator) and the move returned, in steps of ``resolution_hz``
    (so the readout doesn't flicker). The window moves the VFO by it; the
    LO only retunes when the VFO leaves the span (the window's tuning
    rules). The loop learns a steady drift, so it follows it without lag.

    Args:
        sample_rate: The receiver's rate.
        offset_hz: The VFO's offset from the LO now.
        mode: "FM", "AM" or "CW" (see :data:`TRACK_METHODS`).
        channel_hz: The channel (filter) width.
        max_correction_hz: The farthest it follows from where it started.
    """

    #: Seconds without a signal before tracking says it lost it.
    LOST_S = 3.0

    def __init__(
        self,
        sample_rate: float,
        offset_hz: float,
        mode: str,
        channel_hz: float,
        max_correction_hz: Optional[float] = None,
    ):
        from .channelizer import Channelizer

        if mode not in TRACK_METHODS:
            raise ValueError(f"tracking works in FM, AM and CW, not {mode}")
        self.mode = mode
        channel = max(float(channel_hz), 100.0)
        if mode == "CW":
            half = max(1.5e3, 2.0 * channel)
            self.resolution_hz = 5.0
        elif mode == "AM":
            half = max(5e3, 1.0 * channel)
            self.resolution_hz = 10.0
        else:
            half = max(6e3, 1.0 * channel)
            self.resolution_hz = 25.0 if channel <= 50e3 else 250.0
        self.rate = float(sample_rate)
        self._vfo = float(offset_hz)
        self._channel = Channelizer(
            self.rate,
            self._vfo,
            pass_hz=half,
            stop_hz=1.25 * half,
            target_rate=3 * half,
        )
        method = TRACK_METHODS[mode]
        fft_size = 1 << int(
            np.clip(np.ceil(np.log2(self._channel.output_rate / 5.0)), 8, 14)
        )
        self._afc = AutomaticFrequencyControl(
            self._channel.output_rate,
            AFCConfig(
                mode=AFCMode.MEDIUM,
                method=method,
                loop_bandwidth_hz=1.5,
                damping_factor=0.9,
                max_correction_hz=(
                    float(max_correction_hz)
                    if max_correction_hz is not None
                    else max(10e3, 2.0 * channel)
                ),
                max_rate_hz_per_sec=max(5e3, channel),
                lock_threshold_hz=max(2.0 * self.resolution_hz, 20.0),
                fft_size=fft_size,
                min_snr_db=8.0,
            ),
        )
        self._applied = 0.0  # VFO moves handed out
        self._quiet_s = 0.0

    @property
    def correction_hz(self) -> float:
        """How far the VFO has been moved since tracking started."""
        return self._applied

    def process(self, block: np.ndarray, dt_s: float) -> TrackStatus:
        """Measure on a raw block (at the receiver's rate) and say how far
        to move the VFO now."""
        channel = self._channel.process(np.asarray(block, dtype=np.complex64))
        status = self._afc.update(channel, time_delta_ms=1000.0 * max(dt_s, 1e-3))
        if status.signal_present:
            self._quiet_s = 0.0
        else:
            self._quiet_s += max(dt_s, 0.0)
        # The loop's oscillator is the VFO, in whole steps.
        target = self._afc.correction
        steps = round((target - self._applied) / self.resolution_hz)
        nudge = steps * self.resolution_hz
        if nudge:
            self._applied += nudge
        # The channel follows the loop's own correction exactly.
        self._channel.set_offset(self._vfo + target)
        return TrackStatus(
            nudge_hz=float(nudge),
            correction_hz=float(self._applied),
            error_hz=float(status.frequency_error_hz),
            signal_present=bool(status.signal_present),
            locked=bool(status.locked and status.signal_present),
            lost=self._quiet_s >= self.LOST_S,
        )

    def retuned(self, offset_hz: float) -> None:
        """The VFO was moved by something else (the user tuned): start
        over from there."""
        self._vfo = float(offset_hz)
        self._channel.set_offset(self._vfo)
        self._afc.reset()
        self._applied = 0.0
        self._quiet_s = 0.0

    def lo_moved(self, new_offset_hz: float) -> None:
        """The LO was retuned with the VFO staying put: the channel is now
        at ``new_offset_hz`` from it (the correction carries on)."""
        target = self._afc.correction
        self._vfo = float(new_offset_hz) - self._applied
        self._channel.set_offset(self._vfo + target)

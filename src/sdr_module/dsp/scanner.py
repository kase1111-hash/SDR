"""
Frequency Scanner - Sweep across frequency ranges to detect activity.

Provides automated scanning across frequency bands with signal
detection, logging, and pause-on-signal functionality.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Callable
import numpy as np
import time


class ScanMode(Enum):
    """Scanning mode."""
    SINGLE = "single"           # Single sweep, then stop
    CONTINUOUS = "continuous"   # Continuous sweeping
    PAUSE_ON_SIGNAL = "pause"   # Pause when signal detected


class ScanState(Enum):
    """Scanner state."""
    IDLE = "idle"               # Not scanning
    SCANNING = "scanning"       # Actively scanning
    PAUSED = "paused"           # Paused on signal
    COMPLETED = "completed"     # Single scan completed


class ScanDirection(Enum):
    """Scan direction."""
    UP = "up"                   # Low to high frequency
    DOWN = "down"               # High to low frequency


@dataclass
class SignalHit:
    """Detected signal during scan."""
    frequency_hz: float         # Center frequency
    power_db: float             # Signal power
    bandwidth_hz: float         # Estimated bandwidth
    timestamp: float            # Unix timestamp
    snr_db: float = 0.0         # Signal-to-noise ratio


@dataclass
class ScanResult:
    """Result of a frequency scan."""
    start_freq_hz: float        # Scan start frequency
    end_freq_hz: float          # Scan end frequency
    step_hz: float              # Frequency step size
    hits: List[SignalHit] = field(default_factory=list)  # Detected signals
    spectrum_data: Optional[np.ndarray] = None  # Full spectrum if captured
    frequencies: Optional[np.ndarray] = None    # Frequency axis
    scan_time_sec: float = 0.0  # Total scan time


@dataclass
class ScanConfig:
    """Scanner configuration."""
    mode: ScanMode = ScanMode.CONTINUOUS
    direction: ScanDirection = ScanDirection.UP

    # Frequency range
    start_freq_hz: float = 88e6         # Start frequency
    end_freq_hz: float = 108e6          # End frequency
    step_hz: float = 100e3              # Step size

    # Detection thresholds
    threshold_db: float = -60.0         # Minimum signal level
    min_snr_db: float = 10.0            # Minimum SNR for detection

    # Timing
    dwell_time_ms: float = 50.0         # Time per step
    pause_time_ms: float = 2000.0       # Time to pause on signal

    # Options
    record_spectrum: bool = True        # Record full spectrum
    auto_resume: bool = True            # Auto-resume after pause


@dataclass
class ScanStatus:
    """Current scanner status."""
    state: ScanState                    # Current state
    current_freq_hz: float              # Current frequency
    progress_percent: float             # Scan progress (0-100)
    signals_found: int                  # Number of signals detected
    current_power_db: float             # Power at current frequency
    time_remaining_sec: float           # Estimated time remaining


class FrequencyScanner:
    """
    Frequency scanner for automated band sweeping.

    Features:
    - Configurable frequency range and step size
    - Multiple scan modes (single, continuous, pause-on-signal)
    - Signal detection and logging
    - Full spectrum recording
    - Progress tracking
    - Callback support for tuning hardware
    """

    def __init__(self, config: Optional[ScanConfig] = None):
        """
        Initialize frequency scanner.

        Args:
            config: Scanner configuration (uses defaults if None)
        """
        self._config = config or ScanConfig()

        # State
        self._state = ScanState.IDLE
        self._current_freq = self._config.start_freq_hz
        self._step_index = 0
        self._total_steps = 0
        self._scan_start_time = 0.0
        self._pause_start_time = 0.0

        # Results
        self._hits: List[SignalHit] = []
        self._spectrum_data: List[float] = []
        self._frequencies: List[float] = []

        # Current measurement
        self._current_power_db = -100.0
        self._noise_floor_db = -100.0

        # Callbacks
        self._on_tune: Optional[Callable[[float], None]] = None
        self._on_signal: Optional[Callable[[SignalHit], None]] = None
        self._on_complete: Optional[Callable[[ScanResult], None]] = None

        # Calculate total steps
        self._calculate_steps()

    def _calculate_steps(self) -> None:
        """Calculate total number of scan steps."""
        freq_range = abs(self._config.end_freq_hz - self._config.start_freq_hz)
        self._total_steps = max(1, int(freq_range / self._config.step_hz) + 1)

    @property
    def config(self) -> ScanConfig:
        """Get current configuration."""
        return self._config

    @config.setter
    def config(self, cfg: ScanConfig) -> None:
        """Set configuration."""
        self._config = cfg
        self._calculate_steps()

    @property
    def state(self) -> ScanState:
        """Get current state."""
        return self._state

    @property
    def current_frequency(self) -> float:
        """Get current scan frequency."""
        return self._current_freq

    @property
    def hits(self) -> List[SignalHit]:
        """Get list of detected signals."""
        return self._hits.copy()

    def set_tune_callback(self, callback: Callable[[float], None]) -> None:
        """
        Set callback for frequency tuning requests.

        Args:
            callback: Function called with frequency in Hz when scanner
                     needs to tune to a new frequency
        """
        self._on_tune = callback

    def set_signal_callback(self, callback: Callable[[SignalHit], None]) -> None:
        """
        Set callback for signal detection.

        Args:
            callback: Function called when a signal is detected
        """
        self._on_signal = callback

    def set_complete_callback(self, callback: Callable[[ScanResult], None]) -> None:
        """
        Set callback for scan completion.

        Args:
            callback: Function called when scan completes
        """
        self._on_complete = callback

    def start(self) -> None:
        """Start scanning."""
        if self._state == ScanState.SCANNING:
            return  # Already scanning

        self._state = ScanState.SCANNING
        self._step_index = 0
        self._hits.clear()
        self._spectrum_data.clear()
        self._frequencies.clear()
        self._scan_start_time = time.time()

        # Set initial frequency
        if self._config.direction == ScanDirection.UP:
            self._current_freq = self._config.start_freq_hz
        else:
            self._current_freq = self._config.end_freq_hz

        # Request tune
        if self._on_tune:
            self._on_tune(self._current_freq)

    def stop(self) -> None:
        """Stop scanning."""
        self._state = ScanState.IDLE

    def pause(self) -> None:
        """Pause scanning."""
        if self._state == ScanState.SCANNING:
            self._state = ScanState.PAUSED
            self._pause_start_time = time.time()

    def resume(self) -> None:
        """Resume scanning."""
        if self._state == ScanState.PAUSED:
            self._state = ScanState.SCANNING

    def update(
        self,
        spectrum_db: np.ndarray,
        center_freq: Optional[float] = None,
        sample_rate: Optional[float] = None
    ) -> ScanStatus:
        """
        Update scanner with new spectrum data.

        Args:
            spectrum_db: Power spectrum in dB
            center_freq: Center frequency of spectrum (uses current if None)
            sample_rate: Sample rate for bandwidth calculation

        Returns:
            ScanStatus with current scanner state
        """
        if center_freq is None:
            center_freq = self._current_freq

        if self._state == ScanState.IDLE:
            return self._make_status()

        # Update noise floor estimate
        self._noise_floor_db = float(np.percentile(spectrum_db, 10))

        # Get peak power at current frequency
        self._current_power_db = float(np.max(spectrum_db))

        # Record spectrum if enabled
        if self._config.record_spectrum:
            self._spectrum_data.append(self._current_power_db)
            self._frequencies.append(self._current_freq)

        # Check for signals
        if self._state == ScanState.SCANNING:
            self._check_for_signals(spectrum_db, center_freq, sample_rate)

        # Handle pause state
        if self._state == ScanState.PAUSED:
            return self._handle_pause()

        # Advance to next step
        if self._state == ScanState.SCANNING:
            self._advance_step()

        return self._make_status()

    def _check_for_signals(
        self,
        spectrum_db: np.ndarray,
        center_freq: float,
        sample_rate: Optional[float]
    ) -> None:
        """Check spectrum for signals above threshold."""
        peak_power = float(np.max(spectrum_db))
        snr = peak_power - self._noise_floor_db

        if peak_power >= self._config.threshold_db and snr >= self._config.min_snr_db:
            # Signal detected
            bandwidth = 0.0
            if sample_rate:
                # Estimate bandwidth from spectrum
                threshold = peak_power - 3.0  # 3 dB bandwidth
                above_threshold = spectrum_db >= threshold
                bandwidth = float(np.sum(above_threshold) * sample_rate / len(spectrum_db))

            hit = SignalHit(
                frequency_hz=center_freq,
                power_db=peak_power,
                bandwidth_hz=bandwidth,
                timestamp=time.time(),
                snr_db=snr,
            )
            self._hits.append(hit)

            # Callback
            if self._on_signal:
                self._on_signal(hit)

            # Pause if configured
            if self._config.mode == ScanMode.PAUSE_ON_SIGNAL:
                self._state = ScanState.PAUSED
                self._pause_start_time = time.time()

    def _handle_pause(self) -> ScanStatus:
        """Handle paused state."""
        if not self._config.auto_resume:
            return self._make_status()

        # Check if pause time has elapsed
        pause_elapsed_ms = (time.time() - self._pause_start_time) * 1000
        if pause_elapsed_ms >= self._config.pause_time_ms:
            self._state = ScanState.SCANNING
            self._advance_step()

        return self._make_status()

    def _advance_step(self) -> None:
        """Advance to next frequency step."""
        self._step_index += 1

        if self._step_index >= self._total_steps:
            # Scan complete
            if self._config.mode == ScanMode.SINGLE:
                self._state = ScanState.COMPLETED
                self._on_scan_complete()
            else:
                # Continuous mode - restart
                self._step_index = 0
                self._hits.clear()
                if self._config.record_spectrum:
                    self._spectrum_data.clear()
                    self._frequencies.clear()
            return

        # Calculate next frequency
        if self._config.direction == ScanDirection.UP:
            self._current_freq = self._config.start_freq_hz + \
                                 self._step_index * self._config.step_hz
        else:
            self._current_freq = self._config.end_freq_hz - \
                                 self._step_index * self._config.step_hz

        # Clamp to range
        self._current_freq = max(
            min(self._config.start_freq_hz, self._config.end_freq_hz),
            min(self._current_freq, max(self._config.start_freq_hz, self._config.end_freq_hz))
        )

        # Request tune
        if self._on_tune:
            self._on_tune(self._current_freq)

    def _on_scan_complete(self) -> None:
        """Handle scan completion."""
        if self._on_complete:
            result = self.get_result()
            self._on_complete(result)

    def _make_status(self) -> ScanStatus:
        """Create current status object."""
        progress = (self._step_index / self._total_steps) * 100 if self._total_steps > 0 else 0

        # Estimate remaining time
        if self._step_index > 0 and self._state == ScanState.SCANNING:
            elapsed = time.time() - self._scan_start_time
            time_per_step = elapsed / self._step_index
            remaining_steps = self._total_steps - self._step_index
            time_remaining = remaining_steps * time_per_step
        else:
            time_remaining = 0.0

        return ScanStatus(
            state=self._state,
            current_freq_hz=self._current_freq,
            progress_percent=progress,
            signals_found=len(self._hits),
            current_power_db=self._current_power_db,
            time_remaining_sec=time_remaining,
        )

    def get_result(self) -> ScanResult:
        """
        Get scan result with all collected data.

        Returns:
            ScanResult with hits and spectrum data
        """
        spectrum_arr = None
        freq_arr = None

        if self._config.record_spectrum and self._spectrum_data:
            spectrum_arr = np.array(self._spectrum_data)
            freq_arr = np.array(self._frequencies)

        return ScanResult(
            start_freq_hz=self._config.start_freq_hz,
            end_freq_hz=self._config.end_freq_hz,
            step_hz=self._config.step_hz,
            hits=self._hits.copy(),
            spectrum_data=spectrum_arr,
            frequencies=freq_arr,
            scan_time_sec=time.time() - self._scan_start_time if self._scan_start_time else 0.0,
        )

    def get_frequency_at_step(self, step: int) -> float:
        """
        Get frequency at a specific step.

        Args:
            step: Step index (0 to total_steps-1)

        Returns:
            Frequency in Hz at that step
        """
        if self._config.direction == ScanDirection.UP:
            return self._config.start_freq_hz + step * self._config.step_hz
        else:
            return self._config.end_freq_hz - step * self._config.step_hz

    def reset(self) -> None:
        """Reset scanner to initial state."""
        self._state = ScanState.IDLE
        self._step_index = 0
        self._current_freq = self._config.start_freq_hz
        self._hits.clear()
        self._spectrum_data.clear()
        self._frequencies.clear()
        self._scan_start_time = 0.0

    @property
    def total_steps(self) -> int:
        """Get total number of scan steps."""
        return self._total_steps

    @property
    def progress(self) -> float:
        """Get scan progress as percentage (0-100)."""
        return (self._step_index / self._total_steps) * 100 if self._total_steps > 0 else 0

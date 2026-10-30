"""
HAM Radio Signal Strength Meter.

Provides signal strength measurement in formats familiar to amateur radio operators:

1. S-Meter (S-Units): The classic S1-S9 scale
   - S1 = -121 dBm, S9 = -73 dBm (6 dB per S-unit)
   - Above S9: reported as "S9+10", "S9+20", etc.
   - Example: "You're 20 over nine" means S9+20dB

2. RST (Readability, Signal, Tone):
   - R = Readability (1-5): How well the signal can be understood
   - S = Signal strength (1-9): Corresponds to S-meter
   - T = Tone (1-9): CW signal quality (9 = perfect tone)
   - Example: "59" for phone, "599" for perfect CW

When you say "You're five-nine" or "twenty over" - any old HAM knows exactly what you mean.

Levels: the samples are measured in dBFS (relative to the SDR's full scale).
Until the meter is calibrated (:meth:`SignalMeter.set_calibration`) dBm is
only a rough guess (0 dBFS taken as -30 dBm), so S-units are relative too.
Signal-to-noise and the readability digit need a noise reference: pass the
channel's noise power to :meth:`SignalMeter.update` (the GUI measures it
from the spectrum around the channel), or the meter estimates it from the
samples' own spectrum.

Usage:
    meter = SignalMeter()
    meter.update(samples)

    print(meter.get_s_meter())      # "S7"
    print(meter.get_s_meter_full()) # "S7 (-85 dBm)"
    print(meter.get_rst())          # "57" or "579" for CW
    print(meter.get_report())       # "You're five and seven"
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SignalMode(Enum):
    """Signal mode for RST reporting."""

    PHONE = auto()  # Voice - uses RS (2 digits)
    CW = auto()  # Morse - uses RST (3 digits)
    DIGITAL = auto()  # Digital modes - uses RS (2 digits)


@dataclass
class SignalReading:
    """Complete signal reading with all HAM-style measurements."""

    # Raw measurements
    power_dbm: float  # Power in dBm
    power_dbfs: float  # Power in dB relative to full scale
    snr_db: float  # Signal-to-noise ratio

    # S-Meter
    s_units: float  # S-units (can be fractional, e.g., 7.5)
    s_meter: str  # S-meter string ("S7", "S9+20")
    s_value: int  # Integer S value (1-9)
    over_s9_db: float  # dB over S9 (0 if below S9)

    # RST
    readability: int  # R: 1-5
    strength: int  # S: 1-9
    tone: int  # T: 1-9 (for CW)

    # Timestamps
    timestamp: float
    peak_hold_dbm: float  # Peak hold value

    # Noise reference and calibration
    noise_dbm: Optional[float] = None  # channel noise power (None: unknown)
    calibrated: bool = False  # dBm calibrated (else dBFS - 30, a guess)
    signal_present: bool = True  # more than ~3 dB above the noise

    @property
    def peak_hold_dbfs(self) -> float:
        """Peak hold on the dBFS scale of ``power_dbfs``."""
        return self.peak_hold_dbm - (self.power_dbm - self.power_dbfs)


# S-Meter reference levels (IARU Region 1 standard for HF)
# Each S-unit = 6 dB, S9 = -73 dBm (50 ohm reference)
S_METER_REFERENCE = {
    1: -121,  # S1
    2: -115,  # S2
    3: -109,  # S3
    4: -103,  # S4
    5: -97,  # S5
    6: -91,  # S6
    7: -85,  # S7
    8: -79,  # S8
    9: -73,  # S9 (reference point)
}

S9_DBM = -73.0  # S9 reference level
DB_PER_S_UNIT = 6.0  # 6 dB per S-unit

# Uncalibrated, 0 dBFS is taken as this many dBm.
UNCALIBRATED_DBFS_TO_DBM = -30.0
# Below this SNR (dB) the channel holds no usable signal: R1, no report.
SIGNAL_PRESENT_SNR_DB = 3.0
# Peak hold: held this long (s), then falls at this rate (dB/s).
PEAK_HOLD_S = 2.0
PEAK_DECAY_DB_PER_S = 20.0
# The external noise reference counts as current for this long (s).
_NOISE_STALE_S = 2.0
# After a reset (retune) the first blocks can still hold the old channel
# (filter history, queued USB transfers): the peak follows the reading
# instead of holding for this long (s).
_PEAK_SETTLE_S = 0.5


class SignalMeter:
    """
    HAM Radio Signal Strength Meter.

    Measures signal strength and reports in formats familiar to amateur operators.

    Usage:
        meter = SignalMeter()

        # Process samples
        meter.update(iq_samples)

        # Get readings in various formats
        s_reading = meter.get_s_meter()        # "S7" or "S9+20"
        rst = meter.get_rst()                   # "59" or "599"
        report = meter.get_verbal_report()      # "Five and nine, twenty over"

        # Get full reading object
        reading = meter.get_reading()
    """

    def __init__(self, sample_rate: float = 2.4e6, impedance: float = 50.0):
        """
        Initialize signal meter.

        Args:
            sample_rate: Sample rate in Hz
            impedance: Reference impedance in ohms (default: 50)
        """
        self.sample_rate = sample_rate
        self.impedance = impedance

        # Current reading
        self._power_dbm: float = -130.0
        self._snr_db: float = 0.0
        self._peak_hold_dbm: float = -130.0
        self._peak_hold_time: float = 0.0
        self._last_update: float = 0.0

        # Averaging
        self._avg_window: List[float] = []
        self._avg_size = 10

        # Mode for RST
        self._mode = SignalMode.PHONE

        # Calibration: dB added to the uncalibrated estimate (dBFS - 30), at
        # the RF gain it was made at (None: not gain-compensated).
        self._cal_offset_db = 0.0
        self._calibrated = False
        self._cal_gain_db: Optional[float] = None
        self._gain_db: Optional[float] = None

        # Peak hold duration and decay
        self._peak_hold_duration = PEAK_HOLD_S
        self._peak_decay_db_per_s = PEAK_DECAY_DB_PER_S

        # Noise floor in the channel, in dBFS: given with each update (from
        # the spectrum around the channel) or, failing that, estimated from
        # the samples' own spectrum. None until known.
        self._noise_dbfs: Optional[float] = None
        self._noise_external_time = 0.0
        self._settle_until = 0.0
        # Kept for compatibility: the noise floor in dBm.
        self._noise_floor_dbm = -120.0

        logger.info(f"SignalMeter initialized, sample_rate={sample_rate}")

    def set_mode(self, mode: SignalMode) -> None:
        """Set signal mode for RST reporting."""
        self._mode = mode

    def get_mode(self) -> SignalMode:
        """The RST reporting mode."""
        return self._mode

    def set_calibration(
        self, offset_db: float, reference_gain_db: Optional[float] = None
    ) -> None:
        """Calibrate: add ``offset_db`` to the uncalibrated estimate.

        Args:
            offset_db: Correction in dB (dBm = dBFS - 30 + offset).
            reference_gain_db: RF gain the calibration was made at. When
                given, and the current gain is known (:meth:`set_gain`), a
                gain change moves the dBFS reading but not the dBm one.
        """
        self._cal_offset_db = float(offset_db)
        self._cal_gain_db = (
            None if reference_gain_db is None else float(reference_gain_db)
        )
        self._calibrated = True
        self._avg_window.clear()
        logger.info(f"Calibration offset set to {offset_db} dB")

    def clear_calibration(self) -> None:
        """Back to the uncalibrated estimate (dBm = dBFS - 30)."""
        self._cal_offset_db = 0.0
        self._cal_gain_db = None
        self._calibrated = False
        self._avg_window.clear()

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    @property
    def calibration_offset_db(self) -> float:
        return self._cal_offset_db

    @property
    def calibration_gain_db(self) -> Optional[float]:
        return self._cal_gain_db

    def set_gain(self, gain_db: Optional[float]) -> None:
        """The receiver's current RF gain (None: unknown, e.g. hardware AGC)."""
        self._gain_db = None if gain_db is None else float(gain_db)

    def _dbfs_to_dbm_offset(self) -> float:
        """dB to add to a dBFS level to get (calibrated or estimated) dBm."""
        offset = UNCALIBRATED_DBFS_TO_DBM
        if self._calibrated:
            offset += self._cal_offset_db
            if self._cal_gain_db is not None and self._gain_db is not None:
                offset -= self._gain_db - self._cal_gain_db
        return offset

    def offset_for_level(self, dbm: float) -> float:
        """Calibration offset that makes the current reading ``dbm``."""
        dbfs = self._power_dbm - self._dbfs_to_dbm_offset()
        return float(dbm) - (dbfs + UNCALIBRATED_DBFS_TO_DBM)

    def set_noise_floor(self, noise_dbm: float) -> None:
        """Set noise floor estimate for SNR calculation (channel noise, dBm)."""
        self._noise_floor_dbm = float(noise_dbm)
        self._noise_dbfs = float(noise_dbm) - self._dbfs_to_dbm_offset()
        self._noise_external_time = time.time()

    @property
    def settling(self) -> bool:
        """True just after a reset, while old samples may still arrive."""
        return time.time() < self._settle_until

    @property
    def has_noise_floor(self) -> bool:
        """True once a noise reference exists (SNR and R are meaningful)."""
        return self._noise_dbfs is not None

    def reset_peak(self) -> None:
        """Forget the peak hold, the averaging and the noise estimate (after
        a retune, a mode change, or Stop): the next reading starts afresh."""
        self._avg_window.clear()
        self._peak_hold_dbm = -130.0
        self._peak_hold_time = 0.0
        self._noise_dbfs = None
        self._noise_external_time = 0.0
        self._settle_until = time.time() + _PEAK_SETTLE_S

    def update(
        self, samples: np.ndarray, noise_dbfs: Optional[float] = None
    ) -> SignalReading:
        """
        Update meter with new samples.

        Args:
            samples: Complex I/Q samples
            noise_dbfs: Noise power in the same channel, in dBFS (as the
                samples' power would read with no signal). Without it the
                noise is estimated from the samples' spectrum.

        Returns:
            SignalReading with current measurements
        """
        now = time.time()
        previous = self._last_update
        self._last_update = now

        # Calculate power
        samples = np.asarray(samples)
        power_watts = float(np.mean(np.abs(samples) ** 2)) if len(samples) else 0.0
        power_dbfs = 10 * np.log10(power_watts + 1e-20)

        # dBm: calibrated, or the rough guess 0 dBFS = -30 dBm.
        to_dbm = self._dbfs_to_dbm_offset()
        power_dbm = power_dbfs + to_dbm

        # Apply averaging. The first reading after a reset's settling time
        # starts afresh: what came before may be the old channel's samples.
        settled = bool(self._settle_until) and now >= self._settle_until
        if settled:
            self._settle_until = 0.0
            self._avg_window.clear()
        self._avg_window.append(power_dbm)
        if len(self._avg_window) > self._avg_size:
            self._avg_window.pop(0)

        self._power_dbm = float(np.mean(self._avg_window))

        # Noise reference: the caller's, else the samples' own spectrum.
        if noise_dbfs is not None and math.isfinite(noise_dbfs):
            self._track_noise(float(noise_dbfs))
            self._noise_external_time = now
        elif now - self._noise_external_time > _NOISE_STALE_S:
            estimate = self._spectral_noise_dbfs(samples)
            if estimate is not None:
                self._track_noise(estimate)
        if self._noise_dbfs is not None:
            self._noise_floor_dbm = self._noise_dbfs + to_dbm

        # Calculate SNR: signal power above the noise, from total power.
        self._snr_db = self._compute_snr()

        # Peak hold: held for a while, then falls at a steady rate (and only
        # follows the reading while a reset settles).
        if settled or now < self._settle_until:
            self._peak_hold_dbm = self._power_dbm
            self._peak_hold_time = now
        elif self._power_dbm >= self._peak_hold_dbm:
            self._peak_hold_dbm = self._power_dbm
            self._peak_hold_time = now
        elif now - self._peak_hold_time > self._peak_hold_duration:
            elapsed = max(0.0, now - max(previous, self._peak_hold_time))
            self._peak_hold_dbm = max(
                self._power_dbm,
                self._peak_hold_dbm - self._peak_decay_db_per_s * elapsed,
            )

        return self.get_reading()

    def _track_noise(self, noise_dbfs: float) -> None:
        """Smooth the noise reference (it changes slowly)."""
        if self._noise_dbfs is None:
            self._noise_dbfs = noise_dbfs
        else:
            self._noise_dbfs += 0.2 * (noise_dbfs - self._noise_dbfs)

    @staticmethod
    def _spectral_noise_dbfs(samples: np.ndarray) -> Optional[float]:
        """Noise power of ``samples`` from their spectrum's median bin.

        For white noise each (Hann-windowed) bin holds total power * 1.5/N;
        the median of exponentially distributed bins is ln 2 of their mean.
        A narrow signal occupies few bins, so it barely moves the median.
        """
        n = min(len(samples), 4096)
        if n < 64:
            return None
        x = np.asarray(samples[-n:], dtype=np.complex128)
        window = np.hanning(n)
        bins = np.abs(np.fft.fft(x * window)) ** 2 / (np.sum(window) ** 2)
        median = float(np.median(bins))
        noise = median / math.log(2.0) * n / 1.5
        return 10.0 * math.log10(noise + 1e-20)

    def _compute_snr(self) -> float:
        if self._noise_dbfs is None:
            return 0.0
        signal = self._power_dbm - self._dbfs_to_dbm_offset()
        ratio = 10.0 ** ((signal - self._noise_dbfs) / 10.0) - 1.0
        return 10.0 * math.log10(max(ratio, 1e-2))

    def get_reading(self) -> SignalReading:
        """Get current signal reading."""
        # Calculate S-units
        s_units = self._dbm_to_s_units(self._power_dbm)
        s_value = max(1, min(9, int(round(s_units))))
        over_s9 = max(0.0, self._power_dbm - S9_DBM)

        # Format S-meter string
        if self._power_dbm >= S9_DBM:
            over_db = int(round(over_s9 / 10) * 10)  # Round to nearest 10
            if over_db > 0:
                s_meter = f"S9+{over_db}"
            else:
                s_meter = "S9"
        else:
            s_meter = f"S{s_value}"

        # Calculate RST
        readability = self._calculate_readability()
        strength = s_value
        tone = self._calculate_tone()
        to_dbm = self._dbfs_to_dbm_offset()
        noise_dbm = None if self._noise_dbfs is None else self._noise_dbfs + to_dbm

        return SignalReading(
            power_dbm=self._power_dbm,
            power_dbfs=self._power_dbm - to_dbm,
            snr_db=self._snr_db,
            s_units=s_units,
            s_meter=s_meter,
            s_value=s_value,
            over_s9_db=over_s9,
            readability=readability,
            strength=strength,
            tone=tone,
            timestamp=self._last_update,
            peak_hold_dbm=self._peak_hold_dbm,
            noise_dbm=noise_dbm,
            calibrated=self._calibrated,
            signal_present=(
                self._noise_dbfs is None or self._snr_db >= SIGNAL_PRESENT_SNR_DB
            ),
        )

    def _dbm_to_s_units(self, dbm: float) -> float:
        """Convert dBm to S-units (can be fractional)."""
        # S9 = -73 dBm, 6 dB per S-unit
        s_units = 9 + (dbm - S9_DBM) / DB_PER_S_UNIT
        return max(0.0, min(15.0, s_units))  # Cap at ~S9+36

    def _s_units_to_dbm(self, s_units: float) -> float:
        """Convert S-units to dBm."""
        return S9_DBM + (s_units - 9) * DB_PER_S_UNIT

    def _calculate_readability(self) -> int:
        """
        Calculate readability (R) for RST report.

        R1 = Unreadable
        R2 = Barely readable, occasional words distinguishable
        R3 = Readable with considerable difficulty
        R4 = Readable with practically no difficulty
        R5 = Perfectly readable
        """
        # Base on SNR
        if self._snr_db < 3:
            return 1
        elif self._snr_db < 6:
            return 2
        elif self._snr_db < 12:
            return 3
        elif self._snr_db < 20:
            return 4
        else:
            return 5

    def _calculate_tone(self) -> int:
        """
        Calculate tone quality (T) for RST report (CW only).

        T1 = Sixty cycle AC or less, very rough and broad
        T2 = Very rough AC, harsh and broad
        T3 = Rough AC tone, rectified but not filtered
        T4 = Rough note, some trace of filtering
        T5 = Filtered rectified AC, strongly ripple-modulated
        T6 = Filtered tone, definite trace of ripple modulation
        T7 = Near pure tone, trace of ripple modulation
        T8 = Near perfect tone, slight trace of modulation
        T9 = Perfect tone, no trace of ripple or modulation
        """
        # For SDR, assume good tone quality based on signal strength
        # Real implementation would analyze spectral purity
        if self._snr_db < 6:
            return 5
        elif self._snr_db < 12:
            return 7
        elif self._snr_db < 20:
            return 8
        else:
            return 9

    # =========================================================================
    # HAM-STYLE REPORTING METHODS
    # =========================================================================

    def get_s_meter(self) -> str:
        """
        Get S-meter reading.

        Returns:
            S-meter string like "S7" or "S9+20"
        """
        reading = self.get_reading()
        return reading.s_meter

    def get_s_meter_full(self) -> str:
        """
        Get S-meter with dBm.

        Returns:
            String like "S7 (-85 dBm)"
        """
        reading = self.get_reading()
        return f"{reading.s_meter} ({reading.power_dbm:.0f} dBm)"

    def get_rst(self) -> str:
        """
        Get RST report.

        Returns:
            "59" for phone, "599" for CW
        """
        reading = self.get_reading()
        if self._mode == SignalMode.CW:
            return f"{reading.readability}{reading.strength}{reading.tone}"
        else:
            return f"{reading.readability}{reading.strength}"

    def get_verbal_report(self) -> str:
        """
        Get verbal signal report the way HAMs say it.

        Returns:
            String like "Five nine" or "Five nine, twenty over"
        """
        reading = self.get_reading()

        # Number words for natural speech
        num_words = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
        }

        r_word = num_words.get(reading.readability, str(reading.readability))
        s_word = num_words.get(reading.strength, str(reading.strength))

        if self._mode == SignalMode.CW:
            t_word = num_words.get(reading.tone, str(reading.tone))
            report = f"{r_word} {s_word} {t_word}"
        else:
            report = f"{r_word} and {s_word}"

        # Add "over" if above S9
        if reading.over_s9_db >= 5:
            over = int(round(reading.over_s9_db / 10) * 10)
            if over > 0:
                report += f", {over} over"

        return report.capitalize()

    def get_short_report(self) -> str:
        """
        Get abbreviated report.

        Returns:
            "5x9" or "5x9+20"
        """
        reading = self.get_reading()

        base = f"{reading.readability}x{reading.strength}"

        if reading.over_s9_db >= 5:
            over = int(round(reading.over_s9_db / 10) * 10)
            if over > 0:
                base += f"+{over}"

        if self._mode == SignalMode.CW:
            base += f"/{reading.tone}"

        return base

    def get_qso_report(self) -> str:
        """
        Get report suitable for QSO logging.

        Returns:
            "RST: 59 | S9+20 | -53 dBm"
        """
        reading = self.get_reading()
        rst = self.get_rst()

        return f"RST: {rst} | {reading.s_meter} | {reading.power_dbm:.0f} dBm"

    def get_log_report(
        self, frequency_hz: Optional[float] = None, when: Optional[float] = None
    ) -> str:
        """One-line report to paste into a log or chat.

        Returns:
            "RST 59 · S9+20 · 14.250 MHz · 12:34 UTC" (frequency optional);
            "No signal above the noise (S4) · ..." on an empty channel.
        """
        reading = self.get_reading()
        if reading.signal_present:
            parts = [f"RST {self.get_rst()}", reading.s_meter]
        else:
            parts = [f"No signal above the noise ({reading.s_meter})"]
        if frequency_hz:
            parts.append(f"{float(frequency_hz) / 1e6:.3f} MHz")
        stamp = time.gmtime(when if when is not None else time.time())
        parts.append(time.strftime("%H:%M UTC", stamp))
        return " · ".join(parts)

    def get_contest_report(self) -> str:
        """
        Get contest-style report (always 59 or 599).

        In contests, everyone gives 59 regardless of actual signal!

        Returns:
            "59" or "599"
        """
        if self._mode == SignalMode.CW:
            return "599"
        else:
            return "59"

    # =========================================================================
    # DISPLAY HELPERS
    # =========================================================================

    def get_bar_graph(self, width: int = 20) -> str:
        """
        Get ASCII bar graph of signal strength.

        Args:
            width: Width in characters

        Returns:
            ASCII bar like "[████████░░░░░░░░░░░░] S7"
        """
        reading = self.get_reading()

        # Map S1-S9+60 to 0-100%
        percent = min(100, max(0, (reading.s_units / 15.0) * 100))
        filled = int(width * percent / 100)

        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {reading.s_meter}"

    def get_status_line(self) -> str:
        """
        Get one-line status for display.

        Returns:
            "S7 (-85 dBm) | RST: 57 | SNR: 24 dB"
        """
        reading = self.get_reading()
        rst = self.get_rst()

        return (
            f"{reading.s_meter} ({reading.power_dbm:.0f} dBm) | "
            f"RST: {rst} | SNR: {reading.snr_db:.0f} dB"
        )

    @staticmethod
    def get_s_meter_scale() -> str:
        """
        Get S-meter reference scale.

        Returns:
            Multi-line string showing S-unit to dBm mapping
        """
        lines = ["S-Meter Reference (50Ω):"]
        for s, dbm in sorted(S_METER_REFERENCE.items()):
            lines.append(f"  S{s} = {dbm} dBm")
        lines.append("  S9+10 = -63 dBm")
        lines.append("  S9+20 = -53 dBm")
        lines.append("  S9+40 = -33 dBm")
        return "\n".join(lines)


class SignalHistory:
    """
    Signal strength history for graphing.

    Keeps track of signal levels over time for display.
    """

    def __init__(self, max_samples: int = 300):
        """
        Initialize history.

        Args:
            max_samples: Maximum number of samples to keep
        """
        self._max_samples = max_samples
        self._history: List[Tuple[float, float]] = []  # (timestamp, dbm)
        self._s_history: List[Tuple[float, float]] = []  # (timestamp, s_units)

    def add(self, reading: SignalReading) -> None:
        """Add a reading to history."""
        self._history.append((reading.timestamp, reading.power_dbm))
        self._s_history.append((reading.timestamp, reading.s_units))

        # Trim to max size
        if len(self._history) > self._max_samples:
            self._history.pop(0)
            self._s_history.pop(0)

    def get_dbm_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get dBm history as numpy arrays."""
        if not self._history:
            return np.array([]), np.array([])
        times, values = zip(*self._history, strict=True)
        return np.array(times), np.array(values)

    def get_s_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get S-unit history as numpy arrays."""
        if not self._s_history:
            return np.array([]), np.array([])
        times, values = zip(*self._s_history, strict=True)
        return np.array(times), np.array(values)

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics over the history."""
        if not self._history:
            return {"min": 0, "max": 0, "avg": 0, "current": 0}

        values = [v for _, v in self._history]
        return {
            "min": min(values),
            "max": max(values),
            "avg": np.mean(values),
            "current": values[-1] if values else 0,
        }

    def clear(self) -> None:
        """Clear history."""
        self._history.clear()
        self._s_history.clear()


__all__ = [
    "PEAK_DECAY_DB_PER_S",
    "PEAK_HOLD_S",
    "SIGNAL_PRESENT_SNR_DB",
    "UNCALIBRATED_DBFS_TO_DBM",
    "SignalMode",
    "SignalReading",
    "SignalMeter",
    "SignalHistory",
    "S_METER_REFERENCE",
    "S9_DBM",
    "DB_PER_S_UNIT",
]

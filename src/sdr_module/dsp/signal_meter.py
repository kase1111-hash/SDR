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
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

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

        # Calibration offset (for real hardware)
        self._cal_offset_db = 0.0

        # Peak hold duration
        self._peak_hold_duration = 2.0  # seconds

        # Noise floor estimate
        self._noise_floor_dbm = -120.0

        logger.info(f"SignalMeter initialized, sample_rate={sample_rate}")

    def set_mode(self, mode: SignalMode) -> None:
        """Set signal mode for RST reporting."""
        self._mode = mode

    def set_calibration(self, offset_db: float) -> None:
        """Set calibration offset in dB."""
        self._cal_offset_db = offset_db
        logger.info(f"Calibration offset set to {offset_db} dB")

    def set_noise_floor(self, noise_dbm: float) -> None:
        """Set noise floor estimate for SNR calculation."""
        self._noise_floor_dbm = noise_dbm

    def update(self, samples: np.ndarray) -> SignalReading:
        """
        Update meter with new samples.

        Args:
            samples: Complex I/Q samples

        Returns:
            SignalReading with current measurements
        """
        now = time.time()
        self._last_update = now

        # Calculate power
        power_watts = np.mean(np.abs(samples) ** 2)
        power_dbfs = 10 * np.log10(power_watts + 1e-20)

        # Convert to dBm (assuming 0 dBFS = some reference)
        # For SDR, we estimate based on typical ADC full scale
        # This is approximate - real calibration needed for accuracy
        power_dbm = power_dbfs + self._cal_offset_db - 30  # Rough estimate

        # Apply averaging
        self._avg_window.append(power_dbm)
        if len(self._avg_window) > self._avg_size:
            self._avg_window.pop(0)

        self._power_dbm = float(np.mean(self._avg_window))

        # Calculate SNR
        self._snr_db = self._power_dbm - self._noise_floor_dbm

        # Peak hold
        if self._power_dbm > self._peak_hold_dbm:
            self._peak_hold_dbm = self._power_dbm
            self._peak_hold_time = now
        elif now - self._peak_hold_time > self._peak_hold_duration:
            # Decay peak hold
            self._peak_hold_dbm = max(self._power_dbm, self._peak_hold_dbm - 1.0)
            self._peak_hold_time = now

        return self.get_reading()

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

        return SignalReading(
            power_dbm=self._power_dbm,
            power_dbfs=self._power_dbm + 30 - self._cal_offset_db,
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
        times, values = zip(*self._history)
        return np.array(times), np.array(values)

    def get_s_history(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get S-unit history as numpy arrays."""
        if not self._s_history:
            return np.array([]), np.array([])
        times, values = zip(*self._s_history)
        return np.array(times), np.array(values)

    def get_statistics(self) -> dict:
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
    "SignalMode",
    "SignalReading",
    "SignalMeter",
    "SignalHistory",
    "S_METER_REFERENCE",
    "S9_DBM",
    "DB_PER_S_UNIT",
]

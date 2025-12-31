"""
QRP (Low Power) Operations Module.

QRP Definition:
- CW: 5 watts output or less
- SSB/Phone: 10 watts output or less
- QRPp: Milliwatt power (< 1W)

This module provides:
- Power conversion (dBm <-> watts)
- TX power limiting for QRP compliance
- Amplifier chain calculator
- QRP contest exchange formatting
- Miles-per-watt calculations

The HackRF outputs ~30mW max (+15 dBm), which is QRPp territory.
For true QRP (5W), you'll need an external amplifier.

Usage:
    from sdr_module.dsp.qrp import QRPController, dbm_to_watts

    qrp = QRPController()
    qrp.set_power_limit(5.0)  # 5 watts max

    # Check if power is QRP compliant
    if qrp.is_qrp_compliant(power_watts=3.0, mode="CW"):
        print("You're running QRP!")

    # Calculate amplifier output
    output = qrp.calculate_chain(
        input_dbm=0,           # HackRF output
        preamp_gain_db=10,     # Driver stage
        pa_gain_db=20          # Final amplifier
    )
    print(f"Output: {output['watts']:.1f}W")
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# POWER CONVERSION FUNCTIONS
# =============================================================================


def dbm_to_watts(dbm: float) -> float:
    """
    Convert dBm to watts.

    Args:
        dbm: Power in dBm

    Returns:
        Power in watts
    """
    return 10 ** ((dbm - 30) / 10)


def watts_to_dbm(watts: float) -> float:
    """
    Convert watts to dBm.

    Args:
        watts: Power in watts

    Returns:
        Power in dBm
    """
    if watts <= 0:
        return -float("inf")
    return 10 * math.log10(watts) + 30


def dbm_to_mw(dbm: float) -> float:
    """Convert dBm to milliwatts."""
    return 10 ** (dbm / 10)


def mw_to_dbm(mw: float) -> float:
    """Convert milliwatts to dBm."""
    if mw <= 0:
        return -float("inf")
    return 10 * math.log10(mw)


def format_power(watts: float) -> str:
    """
    Format power in human-readable units.

    Args:
        watts: Power in watts

    Returns:
        Formatted string like "5.0 W", "250 mW", "1.5 µW"
    """
    if watts >= 1.0:
        return f"{watts:.1f} W"
    elif watts >= 0.001:
        return f"{watts * 1000:.0f} mW"
    elif watts >= 0.000001:
        return f"{watts * 1000000:.1f} µW"
    else:
        return f"{watts * 1000000000:.2f} nW"


def format_power_verbose(watts: float, dbm: Optional[float] = None) -> str:
    """
    Format power with both watts and dBm.

    Args:
        watts: Power in watts
        dbm: Optional dBm value (calculated if not provided)

    Returns:
        Formatted string like "5.0 W (+37 dBm)"
    """
    if dbm is None:
        dbm = watts_to_dbm(watts)

    power_str = format_power(watts)
    return f"{power_str} ({dbm:+.0f} dBm)"


# =============================================================================
# QRP DEFINITIONS
# =============================================================================


class QRPClass(Enum):
    """QRP power classifications."""

    QRPP = auto()  # Milliwatt power (< 1W)
    QRP_CW = auto()  # ≤ 5W CW
    QRP_SSB = auto()  # ≤ 10W SSB
    LOW_POWER = auto()  # 10-100W (not QRP but still low)
    STANDARD = auto()  # > 100W


@dataclass
class QRPLimits:
    """Power limits for QRP operation."""

    qrp_cw_watts: float = 5.0  # Max CW power for QRP
    qrp_ssb_watts: float = 10.0  # Max SSB power for QRP
    qrpp_watts: float = 1.0  # Max power for QRPp
    low_power_watts: float = 100.0  # Max for "low power"


# Standard QRP limits
QRP_LIMITS = QRPLimits()


@dataclass
class AmplifierStage:
    """Represents an amplifier stage in the TX chain."""

    name: str
    gain_db: float
    max_output_dbm: float = 50.0  # 100W default max
    efficiency: float = 0.5  # 50% typical
    enabled: bool = True


@dataclass
class PowerChainResult:
    """Result of power chain calculation."""

    input_dbm: float
    output_dbm: float
    input_watts: float
    output_watts: float
    total_gain_db: float
    stages: List[Dict]
    is_qrp: bool
    qrp_class: QRPClass
    dc_power_watts: float  # Estimated DC power consumption


# =============================================================================
# QRP CONTROLLER
# =============================================================================


class QRPController:
    """
    QRP power management and compliance controller.

    Features:
    - TX power limiting
    - Power chain calculation
    - QRP compliance checking
    - Miles-per-watt tracking
    - Contest exchange formatting
    """

    def __init__(self):
        """Initialize QRP controller."""
        self._power_limit_watts: Optional[float] = None
        self._power_limit_enabled: bool = False
        self._limits = QRP_LIMITS
        self._amplifier_chain: List[AmplifierStage] = []

        # HackRF default output
        self._exciter_power_dbm: float = 0.0  # 1 mW typical

        # Statistics
        self._total_qsos: int = 0
        self._total_miles: float = 0.0
        self._best_mpw: float = 0.0  # Best miles per watt

        logger.info("QRPController initialized")

    # =========================================================================
    # POWER LIMITING
    # =========================================================================

    def set_power_limit(self, watts: float) -> None:
        """
        Set maximum TX power limit.

        Args:
            watts: Maximum power in watts
        """
        self._power_limit_watts = watts
        self._power_limit_enabled = True
        logger.info(f"TX power limit set to {format_power(watts)}")

    def disable_power_limit(self) -> None:
        """Disable TX power limiting."""
        self._power_limit_enabled = False
        logger.info("TX power limit disabled")

    def get_power_limit(self) -> Optional[float]:
        """Get current power limit in watts."""
        if self._power_limit_enabled:
            return self._power_limit_watts
        return None

    def is_within_limit(self, power_watts: float) -> Tuple[bool, str]:
        """
        Check if power is within configured limit.

        Args:
            power_watts: Power to check in watts

        Returns:
            Tuple of (is_ok, message)
        """
        if not self._power_limit_enabled or self._power_limit_watts is None:
            return True, "No power limit set"

        if power_watts <= self._power_limit_watts:
            return (
                True,
                f"OK: {format_power(power_watts)} ≤ {format_power(self._power_limit_watts)}",
            )
        else:
            return (
                False,
                f"EXCEEDS LIMIT: {format_power(power_watts)} > {format_power(self._power_limit_watts)}",
            )

    def limit_power(self, power_dbm: float) -> float:
        """
        Apply power limit, returning limited value.

        Args:
            power_dbm: Requested power in dBm

        Returns:
            Limited power in dBm
        """
        if not self._power_limit_enabled or self._power_limit_watts is None:
            return power_dbm

        limit_dbm = watts_to_dbm(self._power_limit_watts)
        if power_dbm > limit_dbm:
            logger.warning(
                f"Power limited from {power_dbm:.1f} dBm to {limit_dbm:.1f} dBm"
            )
            return limit_dbm
        return power_dbm

    # =========================================================================
    # QRP COMPLIANCE
    # =========================================================================

    def classify_power(self, power_watts: float) -> QRPClass:
        """
        Classify power level.

        Args:
            power_watts: Power in watts

        Returns:
            QRPClass classification
        """
        if power_watts <= self._limits.qrpp_watts:
            return QRPClass.QRPP
        elif power_watts <= self._limits.qrp_cw_watts:
            return QRPClass.QRP_CW
        elif power_watts <= self._limits.qrp_ssb_watts:
            return QRPClass.QRP_SSB
        elif power_watts <= self._limits.low_power_watts:
            return QRPClass.LOW_POWER
        else:
            return QRPClass.STANDARD

    def is_qrp_compliant(self, power_watts: float, mode: str = "CW") -> bool:
        """
        Check if power level is QRP compliant for given mode.

        Args:
            power_watts: Power in watts
            mode: Operating mode ("CW", "SSB", "FM", etc.)

        Returns:
            True if QRP compliant
        """
        mode = mode.upper()

        if mode in ("CW", "RTTY", "PSK", "FT8", "FT4", "JT65", "WSPR"):
            return power_watts <= self._limits.qrp_cw_watts
        else:  # SSB, FM, AM, etc.
            return power_watts <= self._limits.qrp_ssb_watts

    def get_qrp_status(self, power_watts: float, mode: str = "CW") -> str:
        """
        Get QRP status string.

        Args:
            power_watts: Power in watts
            mode: Operating mode

        Returns:
            Status string like "QRP (5W CW)" or "QRO"
        """
        qrp_class = self.classify_power(power_watts)

        if qrp_class == QRPClass.QRPP:
            return f"QRPp ({format_power(power_watts)})"
        elif qrp_class == QRPClass.QRP_CW:
            return f"QRP ({format_power(power_watts)})"
        elif qrp_class == QRPClass.QRP_SSB:
            if mode.upper() in ("CW", "RTTY", "PSK", "FT8"):
                return f"Low Power ({format_power(power_watts)})"  # Over 5W CW limit
            else:
                return f"QRP ({format_power(power_watts)})"
        elif qrp_class == QRPClass.LOW_POWER:
            return f"Low Power ({format_power(power_watts)})"
        else:
            return f"QRO ({format_power(power_watts)})"

    # =========================================================================
    # AMPLIFIER CHAIN
    # =========================================================================

    def set_exciter_power(self, dbm: float) -> None:
        """Set exciter (SDR) output power in dBm."""
        self._exciter_power_dbm = dbm

    def add_amplifier_stage(self, stage: AmplifierStage) -> None:
        """Add an amplifier stage to the chain."""
        self._amplifier_chain.append(stage)
        logger.info(f"Added amplifier stage: {stage.name} ({stage.gain_db} dB gain)")

    def clear_amplifier_chain(self) -> None:
        """Clear all amplifier stages."""
        self._amplifier_chain.clear()

    def calculate_chain(
        self,
        input_dbm: Optional[float] = None,
        preamp_gain_db: float = 0,
        pa_gain_db: float = 0,
        mode: str = "CW",
    ) -> PowerChainResult:
        """
        Calculate power through the amplifier chain.

        Args:
            input_dbm: Input power (uses exciter power if not specified)
            preamp_gain_db: Additional preamp/driver gain
            pa_gain_db: Final PA gain
            mode: Operating mode for QRP classification

        Returns:
            PowerChainResult with detailed breakdown
        """
        if input_dbm is None:
            input_dbm = self._exciter_power_dbm

        current_dbm = input_dbm
        total_gain = 0.0
        stages = []
        total_dc_power = 0.0

        # Add specified gains as stages
        if preamp_gain_db > 0:
            current_dbm += preamp_gain_db
            total_gain += preamp_gain_db
            stages.append(
                {
                    "name": "Preamp/Driver",
                    "gain_db": preamp_gain_db,
                    "output_dbm": current_dbm,
                    "output_watts": dbm_to_watts(current_dbm),
                }
            )

        if pa_gain_db > 0:
            current_dbm += pa_gain_db
            total_gain += pa_gain_db
            output_watts = dbm_to_watts(current_dbm)
            # Estimate DC power (assume 50% efficiency)
            dc_power = output_watts / 0.5
            total_dc_power += dc_power
            stages.append(
                {
                    "name": "Power Amplifier",
                    "gain_db": pa_gain_db,
                    "output_dbm": current_dbm,
                    "output_watts": output_watts,
                    "dc_power_watts": dc_power,
                }
            )

        # Process defined amplifier chain
        for amp in self._amplifier_chain:
            if not amp.enabled:
                continue

            # Apply gain but respect max output
            new_dbm = current_dbm + amp.gain_db
            if new_dbm > amp.max_output_dbm:
                new_dbm = amp.max_output_dbm  # Compression/limiting

            output_watts = dbm_to_watts(new_dbm)
            dc_power = output_watts / amp.efficiency
            total_dc_power += dc_power

            stages.append(
                {
                    "name": amp.name,
                    "gain_db": amp.gain_db,
                    "output_dbm": new_dbm,
                    "output_watts": output_watts,
                    "dc_power_watts": dc_power,
                }
            )

            total_gain += new_dbm - current_dbm
            current_dbm = new_dbm

        output_watts = dbm_to_watts(current_dbm)
        qrp_class = self.classify_power(output_watts)
        is_qrp = self.is_qrp_compliant(output_watts, mode)

        return PowerChainResult(
            input_dbm=input_dbm,
            output_dbm=current_dbm,
            input_watts=dbm_to_watts(input_dbm),
            output_watts=output_watts,
            total_gain_db=total_gain,
            stages=stages,
            is_qrp=is_qrp,
            qrp_class=qrp_class,
            dc_power_watts=total_dc_power,
        )

    # =========================================================================
    # QSO LOGGING & STATISTICS
    # =========================================================================

    def log_qso(self, distance_miles: float, power_watts: float) -> float:
        """
        Log a QSO for miles-per-watt calculation.

        Args:
            distance_miles: Distance to contacted station
            power_watts: TX power used

        Returns:
            Miles per watt for this QSO
        """
        self._total_qsos += 1
        self._total_miles += distance_miles

        mpw = distance_miles / power_watts if power_watts > 0 else 0
        if mpw > self._best_mpw:
            self._best_mpw = mpw
            logger.info(f"New best MPW: {mpw:.0f} miles/watt!")

        return mpw

    def get_statistics(self) -> Dict:
        """Get QRP statistics."""
        return {
            "total_qsos": self._total_qsos,
            "total_miles": self._total_miles,
            "best_mpw": self._best_mpw,
            "average_mpw": self._total_miles / max(1, self._total_qsos),
        }

    # =========================================================================
    # CONTEST EXCHANGE
    # =========================================================================

    def format_contest_exchange(
        self, rst: str, power_watts: float, state_province: str = "", name: str = ""
    ) -> str:
        """
        Format QRP contest exchange.

        Args:
            rst: RST report (e.g., "599")
            power_watts: TX power in watts
            state_province: State/province for exchange
            name: Operator name (for some contests)

        Returns:
            Formatted exchange string
        """
        # Round power for exchange
        if power_watts < 1:
            power_str = f"{int(power_watts * 1000)}mW"
        else:
            power_str = f"{power_watts:.0f}W"

        if state_province:
            return f"{rst} {state_province} {power_str}"
        elif name:
            return f"{rst} {name} {power_str}"
        else:
            return f"{rst} {power_str}"

    def parse_contest_exchange(self, exchange: str) -> Dict:
        """
        Parse a QRP contest exchange.

        Args:
            exchange: Exchange string like "599 CA 5W"

        Returns:
            Parsed components
        """
        parts = exchange.upper().split()
        result = {"rst": "", "location": "", "power": "", "power_watts": 0.0}

        for part in parts:
            if part.isdigit() and len(part) in (2, 3):
                result["rst"] = part
            elif part.endswith("W"):
                result["power"] = part
                try:
                    if "MW" in part:
                        result["power_watts"] = float(part.replace("MW", "")) / 1000
                    else:
                        result["power_watts"] = float(part.replace("W", ""))
                except ValueError:
                    pass
            elif len(part) == 2 and part.isalpha():
                result["location"] = part

        return result

    # =========================================================================
    # DISPLAY HELPERS
    # =========================================================================

    def get_power_display(self, power_dbm: float, mode: str = "CW") -> Dict:
        """
        Get comprehensive power display info.

        Args:
            power_dbm: Power in dBm
            mode: Operating mode

        Returns:
            Dict with display information
        """
        watts = dbm_to_watts(power_dbm)
        mw = dbm_to_mw(power_dbm)

        return {
            "dbm": power_dbm,
            "watts": watts,
            "milliwatts": mw,
            "formatted": format_power(watts),
            "verbose": format_power_verbose(watts, power_dbm),
            "qrp_status": self.get_qrp_status(watts, mode),
            "qrp_class": self.classify_power(watts).name,
            "is_qrp": self.is_qrp_compliant(watts, mode),
            "within_limit": self.is_within_limit(watts)[0],
        }

    def get_status_line(self, power_dbm: float, mode: str = "CW") -> str:
        """
        Get one-line status display.

        Args:
            power_dbm: Power in dBm
            mode: Operating mode

        Returns:
            Status line like "5.0 W (+37 dBm) - QRP CW"
        """
        info = self.get_power_display(power_dbm, mode)
        limit_str = ""
        if self._power_limit_enabled and self._power_limit_watts is not None:
            limit_str = f" [Limit: {format_power(self._power_limit_watts)}]"

        return f"{info['verbose']} - {info['qrp_status']}{limit_str}"


# =============================================================================
# COMMON AMPLIFIER CONFIGURATIONS
# =============================================================================


def get_qrp_amplifier_chain() -> List[AmplifierStage]:
    """Get a typical QRP amplifier chain (HackRF -> 5W)."""
    return [
        AmplifierStage(
            name="Driver", gain_db=20, max_output_dbm=30, efficiency=0.6  # 1W max
        ),
        AmplifierStage(
            name="QRP Final", gain_db=10, max_output_dbm=37, efficiency=0.5  # 5W max
        ),
    ]


def get_qro_amplifier_chain() -> List[AmplifierStage]:
    """Get a typical QRO amplifier chain (HackRF -> 100W)."""
    return [
        AmplifierStage(
            name="Driver", gain_db=20, max_output_dbm=33, efficiency=0.6  # 2W max
        ),
        AmplifierStage(
            name="Linear Amplifier",
            gain_db=20,
            max_output_dbm=50,  # 100W max
            efficiency=0.5,
        ),
    ]


__all__ = [
    # Power conversion
    "dbm_to_watts",
    "watts_to_dbm",
    "dbm_to_mw",
    "mw_to_dbm",
    "format_power",
    "format_power_verbose",
    # Classes
    "QRPClass",
    "QRPLimits",
    "QRP_LIMITS",
    "AmplifierStage",
    "PowerChainResult",
    "QRPController",
    # Helpers
    "get_qrp_amplifier_chain",
    "get_qro_amplifier_chain",
]

"""
Frequency management with TX lockouts, license profiles, and RX presets.

This module provides:
- TX frequency lockouts for protected bands (GPS, emergency, aviation, etc.)
- License-based TX permissions (None, Technician, General, Amateur Extra)
- RX frequency presets for common signals
- Frequency validation before transmission

SAFETY: GPS spoofing is extremely dangerous and illegal. Spoofed GPS signals
have caused aircraft navigation failures and could result in loss of life.
GPS frequencies are permanently locked out from transmission.

LICENSE PROFILES:
- None: Only license-free bands (CB, MURS, FRS)
- Technician: VHF/UHF full, limited HF (10m, some 80m/40m/15m CW)
- General: Most HF bands with sub-band restrictions
- Amateur Extra: Full amateur band privileges
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set

logger = logging.getLogger(__name__)

# Power headroom factor - allows 150% of legal limit for filtering/cable losses
# Real wattage is measured at antenna base, so we allow headroom for the TX chain
POWER_HEADROOM_FACTOR = 1.5

# Warning message for TX power testing
TX_POWER_WARNING = (
    "⚠️ IMPORTANT: Before transmitting, test your actual broadcast power "
    "with a 50Ω dummy load and power meter. The power shown here is the "
    "configured limit, not measured output. Cable losses, filtering, and "
    "amplifier efficiency affect actual radiated power."
)


class LockoutReason(Enum):
    """Reason for frequency lockout."""
    GPS = auto()              # GPS/GNSS - Aircraft/vehicle navigation
    AVIATION = auto()         # Aviation safety frequencies
    EMERGENCY = auto()        # Emergency/distress frequencies
    MILITARY = auto()         # Military bands
    CELLULAR = auto()         # Cellular/mobile bands
    SATELLITE = auto()        # Satellite communications
    GOVERNMENT = auto()       # Government/public safety
    BROADCAST = auto()        # Licensed broadcast bands
    LICENSE = auto()          # No license or insufficient license


class LicenseClass(Enum):
    """Amateur radio license class (US FCC)."""
    NONE = "none"                    # No license - only license-free bands
    TECHNICIAN = "technician"        # Entry level - VHF/UHF, limited HF
    GENERAL = "general"              # Most HF privileges
    AMATEUR_EXTRA = "amateur_extra"  # Full privileges

    @classmethod
    def from_string(cls, s: str) -> "LicenseClass":
        """Parse license class from string."""
        s = s.lower().strip().replace(" ", "_").replace("-", "_")
        for member in cls:
            if member.value == s or member.name.lower() == s:
                return member
        return cls.NONE


@dataclass
class BandPrivilege:
    """
    Defines TX privileges for a frequency band segment.

    Attributes:
        name: Human-readable band name
        start_hz: Lower edge of band segment
        end_hz: Upper edge of band segment
        modes: Allowed modes (CW, SSB, DATA, FM, AM) or empty for all
        max_power_watts: Maximum power for this segment (None = no limit)
        licenses: Set of license classes that can use this segment
    """
    name: str
    start_hz: float
    end_hz: float
    modes: Set[str] = field(default_factory=set)
    max_power_watts: Optional[float] = None
    licenses: Set[LicenseClass] = field(default_factory=set)

    def is_allowed(self, license_class: LicenseClass, mode: str = "") -> bool:
        """Check if the given license and mode are allowed."""
        if license_class not in self.licenses:
            return False
        if self.modes and mode and mode.upper() not in self.modes:
            return False
        return True

    def get_legal_power_limit(self) -> Optional[float]:
        """Get the legal power limit for this band (without headroom)."""
        return self.max_power_watts

    def get_effective_power_limit(self) -> Optional[float]:
        """
        Get the effective power limit with headroom for filtering/losses.

        Returns 150% of the legal limit to account for:
        - Cable/connector losses
        - Filter insertion loss
        - Amplifier efficiency variations
        - Measurement uncertainty

        Real radiated power should be verified with a 50Ω dummy load.
        """
        if self.max_power_watts is None:
            return None
        return self.max_power_watts * POWER_HEADROOM_FACTOR


@dataclass
class FrequencyBand:
    """A frequency band definition."""
    name: str
    start_hz: float
    end_hz: float
    description: str = ""
    rx_only: bool = False
    lockout_reason: Optional[LockoutReason] = None


@dataclass
class FrequencyPreset:
    """A frequency preset for quick tuning."""
    name: str
    frequency_hz: float
    bandwidth_hz: float = 200000  # 200 kHz default
    mode: str = "FM"  # Demodulation mode
    description: str = ""
    category: str = "General"


# =============================================================================
# TX LOCKOUT BANDS - These frequencies are NEVER allowed for transmission
# =============================================================================

TX_LOCKOUT_BANDS: List[FrequencyBand] = [
    # GPS/GNSS - CRITICAL SAFETY - NEVER TRANSMIT
    FrequencyBand(
        name="GPS L1",
        start_hz=1575.42e6 - 15e6,  # 1560.42 - 1590.42 MHz
        end_hz=1575.42e6 + 15e6,
        description="GPS L1 civilian signal - AIRCRAFT NAVIGATION",
        rx_only=True,
        lockout_reason=LockoutReason.GPS
    ),
    FrequencyBand(
        name="GPS L2",
        start_hz=1227.60e6 - 15e6,  # 1212.60 - 1242.60 MHz
        end_hz=1227.60e6 + 15e6,
        description="GPS L2 signal - AIRCRAFT NAVIGATION",
        rx_only=True,
        lockout_reason=LockoutReason.GPS
    ),
    FrequencyBand(
        name="GPS L5",
        start_hz=1176.45e6 - 15e6,  # 1161.45 - 1191.45 MHz
        end_hz=1176.45e6 + 15e6,
        description="GPS L5 safety-of-life signal - AIRCRAFT NAVIGATION",
        rx_only=True,
        lockout_reason=LockoutReason.GPS
    ),
    FrequencyBand(
        name="GLONASS L1",
        start_hz=1598e6 - 10e6,
        end_hz=1610e6,
        description="GLONASS navigation - AIRCRAFT NAVIGATION",
        rx_only=True,
        lockout_reason=LockoutReason.GPS
    ),
    FrequencyBand(
        name="GLONASS L2",
        start_hz=1242e6 - 10e6,
        end_hz=1249e6,
        description="GLONASS navigation - AIRCRAFT NAVIGATION",
        rx_only=True,
        lockout_reason=LockoutReason.GPS
    ),
    FrequencyBand(
        name="Galileo E1",
        start_hz=1575.42e6 - 15e6,
        end_hz=1575.42e6 + 15e6,
        description="Galileo E1 - AIRCRAFT NAVIGATION",
        rx_only=True,
        lockout_reason=LockoutReason.GPS
    ),
    FrequencyBand(
        name="Galileo E5",
        start_hz=1176.45e6 - 25e6,
        end_hz=1207.14e6 + 25e6,
        description="Galileo E5 - AIRCRAFT NAVIGATION",
        rx_only=True,
        lockout_reason=LockoutReason.GPS
    ),
    FrequencyBand(
        name="BeiDou B1",
        start_hz=1561.098e6 - 15e6,
        end_hz=1561.098e6 + 15e6,
        description="BeiDou B1 - NAVIGATION",
        rx_only=True,
        lockout_reason=LockoutReason.GPS
    ),

    # Aviation Emergency/Safety
    FrequencyBand(
        name="Aviation Emergency",
        start_hz=121.5e6 - 0.1e6,
        end_hz=121.5e6 + 0.1e6,
        description="International aviation emergency frequency",
        rx_only=True,
        lockout_reason=LockoutReason.EMERGENCY
    ),
    FrequencyBand(
        name="Aviation Emergency (UHF)",
        start_hz=243.0e6 - 0.1e6,
        end_hz=243.0e6 + 0.1e6,
        description="Military aviation emergency frequency",
        rx_only=True,
        lockout_reason=LockoutReason.EMERGENCY
    ),
    FrequencyBand(
        name="ELT/EPIRB",
        start_hz=406.0e6 - 0.5e6,
        end_hz=406.1e6 + 0.5e6,
        description="Emergency locator beacons - SEARCH AND RESCUE",
        rx_only=True,
        lockout_reason=LockoutReason.EMERGENCY
    ),

    # ADS-B / Mode S (Aviation safety)
    FrequencyBand(
        name="ADS-B/Mode S",
        start_hz=1090e6 - 2e6,
        end_hz=1090e6 + 2e6,
        description="Aircraft transponder - COLLISION AVOIDANCE",
        rx_only=True,
        lockout_reason=LockoutReason.AVIATION
    ),
    FrequencyBand(
        name="Mode S Interrogation",
        start_hz=1030e6 - 2e6,
        end_hz=1030e6 + 2e6,
        description="Radar interrogation frequency",
        rx_only=True,
        lockout_reason=LockoutReason.AVIATION
    ),

    # Marine Emergency
    FrequencyBand(
        name="Marine Distress",
        start_hz=156.8e6 - 0.05e6,
        end_hz=156.8e6 + 0.05e6,
        description="VHF Channel 16 - Marine distress and calling",
        rx_only=True,
        lockout_reason=LockoutReason.EMERGENCY
    ),

    # Cellular bands (illegal to transmit without license)
    FrequencyBand(
        name="Cellular 700 MHz",
        start_hz=698e6,
        end_hz=806e6,
        description="LTE cellular band",
        rx_only=True,
        lockout_reason=LockoutReason.CELLULAR
    ),
    FrequencyBand(
        name="Cellular 850 MHz",
        start_hz=824e6,
        end_hz=894e6,
        description="Cellular 850 band",
        rx_only=True,
        lockout_reason=LockoutReason.CELLULAR
    ),
    FrequencyBand(
        name="Cellular 1900 MHz",
        start_hz=1850e6,
        end_hz=1995e6,
        description="PCS cellular band",
        rx_only=True,
        lockout_reason=LockoutReason.CELLULAR
    ),
]


# =============================================================================
# LICENSE-FREE TX BANDS - Available without any license
# =============================================================================

# Shorthand for all amateur licenses
ALL_HAM = {LicenseClass.TECHNICIAN, LicenseClass.GENERAL, LicenseClass.AMATEUR_EXTRA}
GENERAL_EXTRA = {LicenseClass.GENERAL, LicenseClass.AMATEUR_EXTRA}
EXTRA_ONLY = {LicenseClass.AMATEUR_EXTRA}

LICENSE_FREE_BANDS: List[BandPrivilege] = [
    # CB Radio (Citizens Band) - 26.965-27.405 MHz
    # 4W AM, 12W PEP SSB, 40 channels
    BandPrivilege(
        name="CB Radio",
        start_hz=26.965e6,
        end_hz=27.405e6,
        modes={"AM", "SSB", "USB", "LSB"},
        max_power_watts=12.0,  # 12W PEP SSB, 4W AM carrier
        licenses={LicenseClass.NONE, LicenseClass.TECHNICIAN,
                  LicenseClass.GENERAL, LicenseClass.AMATEUR_EXTRA}
    ),

    # MURS (Multi-Use Radio Service) - 5 channels, 2W
    BandPrivilege(
        name="MURS Ch 1-3",
        start_hz=151.820e6,
        end_hz=151.940e6,
        modes={"FM"},
        max_power_watts=2.0,
        licenses={LicenseClass.NONE, LicenseClass.TECHNICIAN,
                  LicenseClass.GENERAL, LicenseClass.AMATEUR_EXTRA}
    ),
    BandPrivilege(
        name="MURS Ch 4-5",
        start_hz=154.570e6,
        end_hz=154.600e6,
        modes={"FM"},
        max_power_watts=2.0,
        licenses={LicenseClass.NONE, LicenseClass.TECHNICIAN,
                  LicenseClass.GENERAL, LicenseClass.AMATEUR_EXTRA}
    ),

    # FRS (Family Radio Service) - 22 channels
    # Channels 1-7, 15-22: 2W, Channels 8-14: 0.5W
    BandPrivilege(
        name="FRS Channels 1-7",
        start_hz=462.5625e6,
        end_hz=462.7125e6,
        modes={"FM"},
        max_power_watts=2.0,
        licenses={LicenseClass.NONE, LicenseClass.TECHNICIAN,
                  LicenseClass.GENERAL, LicenseClass.AMATEUR_EXTRA}
    ),
    BandPrivilege(
        name="FRS Channels 8-14",
        start_hz=467.5625e6,
        end_hz=467.7125e6,
        modes={"FM"},
        max_power_watts=0.5,
        licenses={LicenseClass.NONE, LicenseClass.TECHNICIAN,
                  LicenseClass.GENERAL, LicenseClass.AMATEUR_EXTRA}
    ),
    BandPrivilege(
        name="FRS Channels 15-22",
        start_hz=462.5500e6,
        end_hz=462.7250e6,
        modes={"FM"},
        max_power_watts=2.0,
        licenses={LicenseClass.NONE, LicenseClass.TECHNICIAN,
                  LicenseClass.GENERAL, LicenseClass.AMATEUR_EXTRA}
    ),
]


# =============================================================================
# AMATEUR RADIO BAND PRIVILEGES - Based on US FCC Part 97
# =============================================================================

AMATEUR_BAND_PRIVILEGES: List[BandPrivilege] = [
    # =========================================================================
    # 160 Meters (1.8-2.0 MHz) - All licenses, all modes
    # =========================================================================
    BandPrivilege(
        name="160m",
        start_hz=1.800e6,
        end_hz=2.000e6,
        modes=set(),  # All modes
        licenses=ALL_HAM
    ),

    # =========================================================================
    # 80 Meters (3.5-4.0 MHz)
    # =========================================================================
    # Extra: 3.500-3.600 CW/Data
    BandPrivilege(
        name="80m CW Extra",
        start_hz=3.500e6,
        end_hz=3.525e6,
        modes={"CW", "DATA"},
        licenses=EXTRA_ONLY
    ),
    # General + Extra: 3.525-3.600 CW/Data
    BandPrivilege(
        name="80m CW General",
        start_hz=3.525e6,
        end_hz=3.600e6,
        modes={"CW", "DATA"},
        licenses=GENERAL_EXTRA
    ),
    # Technician: 3.525-3.600 CW only (Novice portion)
    BandPrivilege(
        name="80m CW Tech",
        start_hz=3.525e6,
        end_hz=3.600e6,
        modes={"CW"},
        licenses={LicenseClass.TECHNICIAN}
    ),
    # Extra: 3.600-3.700 Phone/Image
    BandPrivilege(
        name="80m Phone Extra",
        start_hz=3.600e6,
        end_hz=3.700e6,
        modes={"SSB", "USB", "LSB", "AM", "DATA"},
        licenses=EXTRA_ONLY
    ),
    # General + Extra: 3.700-3.800 Phone/Image (unofficial DX window)
    BandPrivilege(
        name="80m Phone DX Window",
        start_hz=3.700e6,
        end_hz=3.800e6,
        modes={"SSB", "USB", "LSB", "AM", "DATA"},
        licenses=EXTRA_ONLY
    ),
    # General + Extra: 3.800-4.000 Phone
    BandPrivilege(
        name="80m Phone General",
        start_hz=3.800e6,
        end_hz=4.000e6,
        modes={"SSB", "USB", "LSB", "AM", "DATA"},
        licenses=GENERAL_EXTRA
    ),

    # =========================================================================
    # 60 Meters (5 MHz channels) - All licenses, USB only, 100W ERP
    # =========================================================================
    BandPrivilege(
        name="60m Ch 1",
        start_hz=5.3305e6,
        end_hz=5.3335e6,
        modes={"USB", "DATA", "CW"},
        max_power_watts=100.0,
        licenses=ALL_HAM
    ),
    BandPrivilege(
        name="60m Ch 2",
        start_hz=5.3465e6,
        end_hz=5.3495e6,
        modes={"USB", "DATA", "CW"},
        max_power_watts=100.0,
        licenses=ALL_HAM
    ),
    BandPrivilege(
        name="60m Ch 3",
        start_hz=5.3570e6,
        end_hz=5.3600e6,
        modes={"USB", "DATA", "CW"},
        max_power_watts=100.0,
        licenses=ALL_HAM
    ),
    BandPrivilege(
        name="60m Ch 4",
        start_hz=5.3715e6,
        end_hz=5.3745e6,
        modes={"USB", "DATA", "CW"},
        max_power_watts=100.0,
        licenses=ALL_HAM
    ),
    BandPrivilege(
        name="60m Ch 5",
        start_hz=5.4035e6,
        end_hz=5.4065e6,
        modes={"USB", "DATA", "CW"},
        max_power_watts=100.0,
        licenses=ALL_HAM
    ),

    # =========================================================================
    # 40 Meters (7.0-7.3 MHz)
    # =========================================================================
    # Extra: 7.000-7.025 CW
    BandPrivilege(
        name="40m CW Extra",
        start_hz=7.000e6,
        end_hz=7.025e6,
        modes={"CW", "DATA"},
        licenses=EXTRA_ONLY
    ),
    # General + Extra: 7.025-7.125 CW/Data
    BandPrivilege(
        name="40m CW General",
        start_hz=7.025e6,
        end_hz=7.125e6,
        modes={"CW", "DATA"},
        licenses=GENERAL_EXTRA
    ),
    # Technician: 7.025-7.125 CW only (Novice portion)
    BandPrivilege(
        name="40m CW Tech",
        start_hz=7.025e6,
        end_hz=7.125e6,
        modes={"CW"},
        licenses={LicenseClass.TECHNICIAN}
    ),
    # Extra: 7.125-7.175 Phone
    BandPrivilege(
        name="40m Phone Extra",
        start_hz=7.125e6,
        end_hz=7.175e6,
        modes={"SSB", "USB", "LSB", "AM", "DATA"},
        licenses=EXTRA_ONLY
    ),
    # General + Extra: 7.175-7.300 Phone
    BandPrivilege(
        name="40m Phone General",
        start_hz=7.175e6,
        end_hz=7.300e6,
        modes={"SSB", "USB", "LSB", "AM", "DATA"},
        licenses=GENERAL_EXTRA
    ),

    # =========================================================================
    # 30 Meters (10.1-10.15 MHz) - CW/Data only, 200W max
    # =========================================================================
    BandPrivilege(
        name="30m",
        start_hz=10.100e6,
        end_hz=10.150e6,
        modes={"CW", "DATA"},
        max_power_watts=200.0,
        licenses=GENERAL_EXTRA
    ),

    # =========================================================================
    # 20 Meters (14.0-14.35 MHz)
    # =========================================================================
    # Extra: 14.000-14.025 CW
    BandPrivilege(
        name="20m CW Extra",
        start_hz=14.000e6,
        end_hz=14.025e6,
        modes={"CW", "DATA"},
        licenses=EXTRA_ONLY
    ),
    # General + Extra: 14.025-14.150 CW/Data
    BandPrivilege(
        name="20m CW General",
        start_hz=14.025e6,
        end_hz=14.150e6,
        modes={"CW", "DATA"},
        licenses=GENERAL_EXTRA
    ),
    # Extra: 14.150-14.225 Phone
    BandPrivilege(
        name="20m Phone Extra",
        start_hz=14.150e6,
        end_hz=14.225e6,
        modes={"SSB", "USB", "LSB", "AM", "DATA"},
        licenses=EXTRA_ONLY
    ),
    # General + Extra: 14.225-14.350 Phone
    BandPrivilege(
        name="20m Phone General",
        start_hz=14.225e6,
        end_hz=14.350e6,
        modes={"SSB", "USB", "LSB", "AM", "DATA"},
        licenses=GENERAL_EXTRA
    ),

    # =========================================================================
    # 17 Meters (18.068-18.168 MHz)
    # =========================================================================
    # Extra: 18.068-18.110 CW/Data
    BandPrivilege(
        name="17m CW Extra",
        start_hz=18.068e6,
        end_hz=18.110e6,
        modes={"CW", "DATA"},
        licenses=EXTRA_ONLY
    ),
    # General + Extra: 18.110-18.168 All modes
    BandPrivilege(
        name="17m General",
        start_hz=18.110e6,
        end_hz=18.168e6,
        modes=set(),
        licenses=GENERAL_EXTRA
    ),

    # =========================================================================
    # 15 Meters (21.0-21.45 MHz)
    # =========================================================================
    # Extra: 21.000-21.025 CW
    BandPrivilege(
        name="15m CW Extra",
        start_hz=21.000e6,
        end_hz=21.025e6,
        modes={"CW", "DATA"},
        licenses=EXTRA_ONLY
    ),
    # General + Extra: 21.025-21.200 CW/Data
    BandPrivilege(
        name="15m CW General",
        start_hz=21.025e6,
        end_hz=21.200e6,
        modes={"CW", "DATA"},
        licenses=GENERAL_EXTRA
    ),
    # Technician: 21.025-21.200 CW only
    BandPrivilege(
        name="15m CW Tech",
        start_hz=21.025e6,
        end_hz=21.200e6,
        modes={"CW"},
        licenses={LicenseClass.TECHNICIAN}
    ),
    # Extra: 21.200-21.275 Phone
    BandPrivilege(
        name="15m Phone Extra",
        start_hz=21.200e6,
        end_hz=21.275e6,
        modes={"SSB", "USB", "LSB", "AM", "DATA"},
        licenses=EXTRA_ONLY
    ),
    # General + Extra: 21.275-21.450 Phone
    BandPrivilege(
        name="15m Phone General",
        start_hz=21.275e6,
        end_hz=21.450e6,
        modes={"SSB", "USB", "LSB", "AM", "DATA"},
        licenses=GENERAL_EXTRA
    ),

    # =========================================================================
    # 12 Meters (24.89-24.99 MHz)
    # =========================================================================
    BandPrivilege(
        name="12m CW",
        start_hz=24.890e6,
        end_hz=24.930e6,
        modes={"CW", "DATA"},
        licenses=GENERAL_EXTRA
    ),
    BandPrivilege(
        name="12m Phone",
        start_hz=24.930e6,
        end_hz=24.990e6,
        modes={"SSB", "USB", "LSB", "AM", "DATA"},
        licenses=GENERAL_EXTRA
    ),

    # =========================================================================
    # 10 Meters (28.0-29.7 MHz)
    # =========================================================================
    # CW/Data: 28.000-28.300
    BandPrivilege(
        name="10m CW",
        start_hz=28.000e6,
        end_hz=28.300e6,
        modes={"CW", "DATA"},
        licenses=ALL_HAM
    ),
    # Technician CW: 28.000-28.500 (200W max for Tech)
    BandPrivilege(
        name="10m CW Tech",
        start_hz=28.000e6,
        end_hz=28.500e6,
        modes={"CW", "DATA", "SSB", "USB", "LSB"},
        max_power_watts=200.0,
        licenses={LicenseClass.TECHNICIAN}
    ),
    # Phone: 28.300-29.700
    BandPrivilege(
        name="10m Phone",
        start_hz=28.300e6,
        end_hz=29.700e6,
        modes=set(),  # All modes
        licenses=GENERAL_EXTRA
    ),
    # FM Simplex: 29.600 (calling)
    BandPrivilege(
        name="10m FM",
        start_hz=29.500e6,
        end_hz=29.700e6,
        modes={"FM"},
        licenses=ALL_HAM
    ),

    # =========================================================================
    # 6 Meters (50-54 MHz) - All licenses full privileges
    # =========================================================================
    BandPrivilege(
        name="6m",
        start_hz=50.000e6,
        end_hz=54.000e6,
        modes=set(),
        licenses=ALL_HAM
    ),

    # =========================================================================
    # 2 Meters (144-148 MHz) - All licenses full privileges
    # =========================================================================
    BandPrivilege(
        name="2m",
        start_hz=144.000e6,
        end_hz=148.000e6,
        modes=set(),
        licenses=ALL_HAM
    ),

    # =========================================================================
    # 1.25 Meters (222-225 MHz) - All licenses full privileges
    # =========================================================================
    BandPrivilege(
        name="1.25m",
        start_hz=222.000e6,
        end_hz=225.000e6,
        modes=set(),
        licenses=ALL_HAM
    ),

    # =========================================================================
    # 70 Centimeters (420-450 MHz) - All licenses full privileges
    # =========================================================================
    BandPrivilege(
        name="70cm",
        start_hz=420.000e6,
        end_hz=450.000e6,
        modes=set(),
        licenses=ALL_HAM
    ),

    # =========================================================================
    # 33 Centimeters (902-928 MHz) - All licenses full privileges
    # =========================================================================
    BandPrivilege(
        name="33cm",
        start_hz=902.000e6,
        end_hz=928.000e6,
        modes=set(),
        licenses=ALL_HAM
    ),

    # =========================================================================
    # 23 Centimeters (1240-1300 MHz) - All licenses full privileges
    # =========================================================================
    BandPrivilege(
        name="23cm",
        start_hz=1240.000e6,
        end_hz=1300.000e6,
        modes=set(),
        licenses=ALL_HAM
    ),

    # =========================================================================
    # 13 Centimeters (2300-2450 MHz) - All licenses
    # =========================================================================
    BandPrivilege(
        name="13cm",
        start_hz=2300.000e6,
        end_hz=2450.000e6,
        modes=set(),
        licenses=ALL_HAM
    ),
]


# =============================================================================
# RX PRESETS - For receive only, useful for experimentation and learning
# =============================================================================

RX_PRESETS: List[FrequencyPreset] = [
    # GPS/GNSS (RX only for experimentation)
    FrequencyPreset(
        name="GPS L1 (C/A)",
        frequency_hz=1575.42e6,
        bandwidth_hz=2.4e6,
        mode="RAW",
        description="GPS civilian signal - requires special processing",
        category="GNSS"
    ),
    FrequencyPreset(
        name="GPS L2",
        frequency_hz=1227.60e6,
        bandwidth_hz=2.4e6,
        mode="RAW",
        description="GPS L2 signal",
        category="GNSS"
    ),
    FrequencyPreset(
        name="GPS L5",
        frequency_hz=1176.45e6,
        bandwidth_hz=2.4e6,
        mode="RAW",
        description="GPS L5 safety-of-life signal",
        category="GNSS"
    ),
    FrequencyPreset(
        name="GLONASS L1",
        frequency_hz=1602e6,
        bandwidth_hz=8e6,
        mode="RAW",
        description="GLONASS L1 band center",
        category="GNSS"
    ),

    # Aviation (RX only)
    FrequencyPreset(
        name="ADS-B",
        frequency_hz=1090e6,
        bandwidth_hz=2.4e6,
        mode="RAW",
        description="Aircraft transponders - use dump1090",
        category="Aviation"
    ),
    FrequencyPreset(
        name="ACARS",
        frequency_hz=131.55e6,
        bandwidth_hz=25e3,
        mode="AM",
        description="Aircraft communications",
        category="Aviation"
    ),
    FrequencyPreset(
        name="Aviation Emergency",
        frequency_hz=121.5e6,
        bandwidth_hz=25e3,
        mode="AM",
        description="International aviation emergency",
        category="Aviation"
    ),
    FrequencyPreset(
        name="Air Traffic Control",
        frequency_hz=127.85e6,
        bandwidth_hz=25e3,
        mode="AM",
        description="Example ATC frequency (varies by location)",
        category="Aviation"
    ),

    # Weather
    FrequencyPreset(
        name="NOAA Weather 1",
        frequency_hz=162.55e6,
        bandwidth_hz=25e3,
        mode="FM",
        description="NOAA Weather Radio",
        category="Weather"
    ),
    FrequencyPreset(
        name="NOAA Weather 2",
        frequency_hz=162.40e6,
        bandwidth_hz=25e3,
        mode="FM",
        description="NOAA Weather Radio",
        category="Weather"
    ),
    FrequencyPreset(
        name="NOAA APT (NOAA-15)",
        frequency_hz=137.62e6,
        bandwidth_hz=40e3,
        mode="FM",
        description="Weather satellite image downlink",
        category="Weather"
    ),
    FrequencyPreset(
        name="NOAA APT (NOAA-18)",
        frequency_hz=137.9125e6,
        bandwidth_hz=40e3,
        mode="FM",
        description="Weather satellite image downlink",
        category="Weather"
    ),
    FrequencyPreset(
        name="NOAA APT (NOAA-19)",
        frequency_hz=137.1e6,
        bandwidth_hz=40e3,
        mode="FM",
        description="Weather satellite image downlink",
        category="Weather"
    ),

    # Broadcast
    FrequencyPreset(
        name="FM Broadcast",
        frequency_hz=100e6,
        bandwidth_hz=200e3,
        mode="WFM",
        description="FM radio broadcast band",
        category="Broadcast"
    ),

    # Amateur Radio
    FrequencyPreset(
        name="2m Calling",
        frequency_hz=146.52e6,
        bandwidth_hz=15e3,
        mode="FM",
        description="2m FM simplex calling frequency",
        category="Amateur"
    ),
    FrequencyPreset(
        name="70cm Calling",
        frequency_hz=446.0e6,
        bandwidth_hz=15e3,
        mode="FM",
        description="70cm FM simplex calling frequency",
        category="Amateur"
    ),
    FrequencyPreset(
        name="APRS",
        frequency_hz=144.39e6,
        bandwidth_hz=15e3,
        mode="FM",
        description="APRS packet radio (North America)",
        category="Amateur"
    ),

    # ISM Bands
    FrequencyPreset(
        name="ISM 433 MHz",
        frequency_hz=433.92e6,
        bandwidth_hz=500e3,
        mode="RAW",
        description="ISM band - remotes, sensors, etc.",
        category="ISM"
    ),
    FrequencyPreset(
        name="ISM 915 MHz",
        frequency_hz=915e6,
        bandwidth_hz=2e6,
        mode="RAW",
        description="ISM band - LoRa, sensors (Americas)",
        category="ISM"
    ),

    # Paging
    FrequencyPreset(
        name="POCSAG/FLEX",
        frequency_hz=929.6e6,
        bandwidth_hz=25e3,
        mode="FM",
        description="Paging frequencies",
        category="Paging"
    ),

    # Marine
    FrequencyPreset(
        name="Marine Ch 16",
        frequency_hz=156.8e6,
        bandwidth_hz=25e3,
        mode="FM",
        description="Marine distress and calling",
        category="Marine"
    ),

    # Space / ISS (International Space Station)
    FrequencyPreset(
        name="ISS SSTV/Voice",
        frequency_hz=145.800e6,
        bandwidth_hz=15e3,
        mode="FM",
        description="ISS Slow Scan TV images & voice - use SSTV decoder",
        category="Space"
    ),
    FrequencyPreset(
        name="ISS APRS",
        frequency_hz=145.825e6,
        bandwidth_hz=15e3,
        mode="FM",
        description="ISS packet radio / APRS digipeater",
        category="Space"
    ),
    FrequencyPreset(
        name="ISS Packet (UHF)",
        frequency_hz=437.550e6,
        bandwidth_hz=15e3,
        mode="FM",
        description="ISS UHF packet downlink",
        category="Space"
    ),
    FrequencyPreset(
        name="ISS Repeater Output",
        frequency_hz=437.800e6,
        bandwidth_hz=15e3,
        mode="FM",
        description="ISS cross-band repeater output",
        category="Space"
    ),
    FrequencyPreset(
        name="Meteor-M2 LRPT",
        frequency_hz=137.1e6,
        bandwidth_hz=120e3,
        mode="RAW",
        description="Russian weather satellite images",
        category="Space"
    ),
    FrequencyPreset(
        name="SO-50 Downlink",
        frequency_hz=436.795e6,
        bandwidth_hz=15e3,
        mode="FM",
        description="SaudiSat-1C amateur satellite",
        category="Space"
    ),

    # QRP (Low Power) Calling Frequencies
    FrequencyPreset(
        name="QRP 80m CW",
        frequency_hz=3.560e6,
        bandwidth_hz=500,
        mode="CW",
        description="80m QRP CW calling frequency",
        category="QRP"
    ),
    FrequencyPreset(
        name="QRP 40m CW",
        frequency_hz=7.030e6,
        bandwidth_hz=500,
        mode="CW",
        description="40m QRP CW calling frequency",
        category="QRP"
    ),
    FrequencyPreset(
        name="QRP 40m SSB",
        frequency_hz=7.285e6,
        bandwidth_hz=2700,
        mode="LSB",
        description="40m QRP SSB calling frequency",
        category="QRP"
    ),
    FrequencyPreset(
        name="QRP 30m CW",
        frequency_hz=10.106e6,
        bandwidth_hz=500,
        mode="CW",
        description="30m QRP CW calling frequency",
        category="QRP"
    ),
    FrequencyPreset(
        name="QRP 20m CW",
        frequency_hz=14.060e6,
        bandwidth_hz=500,
        mode="CW",
        description="20m QRP CW calling frequency",
        category="QRP"
    ),
    FrequencyPreset(
        name="QRP 20m SSB",
        frequency_hz=14.285e6,
        bandwidth_hz=2700,
        mode="USB",
        description="20m QRP SSB calling frequency",
        category="QRP"
    ),
    FrequencyPreset(
        name="QRP 17m CW",
        frequency_hz=18.096e6,
        bandwidth_hz=500,
        mode="CW",
        description="17m QRP CW calling frequency",
        category="QRP"
    ),
    FrequencyPreset(
        name="QRP 15m CW",
        frequency_hz=21.060e6,
        bandwidth_hz=500,
        mode="CW",
        description="15m QRP CW calling frequency",
        category="QRP"
    ),
    FrequencyPreset(
        name="QRP 15m SSB",
        frequency_hz=21.385e6,
        bandwidth_hz=2700,
        mode="USB",
        description="15m QRP SSB calling frequency",
        category="QRP"
    ),
    FrequencyPreset(
        name="QRP 12m CW",
        frequency_hz=24.906e6,
        bandwidth_hz=500,
        mode="CW",
        description="12m QRP CW calling frequency",
        category="QRP"
    ),
    FrequencyPreset(
        name="QRP 10m CW",
        frequency_hz=28.060e6,
        bandwidth_hz=500,
        mode="CW",
        description="10m QRP CW calling frequency",
        category="QRP"
    ),
    FrequencyPreset(
        name="QRP 10m SSB",
        frequency_hz=28.360e6,
        bandwidth_hz=2700,
        mode="USB",
        description="10m QRP SSB calling frequency",
        category="QRP"
    ),
]


class FrequencyManager:
    """
    Manages frequency validation, lockouts, license profiles, and presets.

    This class ensures that:
    - Protected frequencies cannot be used for transmission
    - TX is only allowed on bands the operator is licensed for
    - Receive is allowed on any frequency
    """

    def __init__(self):
        """Initialize the frequency manager."""
        self._lockout_bands = list(TX_LOCKOUT_BANDS)
        self._rx_presets = list(RX_PRESETS)
        self._custom_lockouts: List[FrequencyBand] = []
        self._license_class = LicenseClass.NONE  # Default: no license
        self._license_free_bands = list(LICENSE_FREE_BANDS)
        self._amateur_bands = list(AMATEUR_BAND_PRIVILEGES)
        logger.info(f"FrequencyManager initialized with {len(self._lockout_bands)} lockout bands")
        logger.info(f"License class: {self._license_class.value}")

    # =========================================================================
    # License Management
    # =========================================================================

    def set_license_class(self, license_class: LicenseClass) -> None:
        """
        Set the operator's license class.

        Args:
            license_class: The license class (NONE, TECHNICIAN, GENERAL, AMATEUR_EXTRA)
        """
        old_class = self._license_class
        self._license_class = license_class
        logger.info(f"License class changed: {old_class.value} -> {license_class.value}")

    def get_license_class(self) -> LicenseClass:
        """Get the current license class."""
        return self._license_class

    def get_license_privileges(self) -> List[BandPrivilege]:
        """
        Get all band privileges available to the current license class.

        Returns:
            List of BandPrivilege objects the operator can use
        """
        privileges = []

        # Always include license-free bands
        for band in self._license_free_bands:
            if self._license_class in band.licenses:
                privileges.append(band)

        # Include amateur bands if licensed
        if self._license_class != LicenseClass.NONE:
            for band in self._amateur_bands:
                if self._license_class in band.licenses:
                    privileges.append(band)

        return privileges

    def _check_license_privilege(
        self,
        frequency_hz: float,
        mode: str = ""
    ) -> Tuple[bool, Optional[str], Optional[BandPrivilege]]:
        """
        Check if TX is allowed at the given frequency for the current license.

        Args:
            frequency_hz: Frequency in Hz
            mode: Operating mode (CW, SSB, FM, etc.)

        Returns:
            Tuple of (allowed, reason_if_blocked, matching_privilege)
        """
        # Check license-free bands first (available to everyone)
        for band in self._license_free_bands:
            if band.start_hz <= frequency_hz <= band.end_hz:
                if band.is_allowed(self._license_class, mode):
                    return True, None, band

        # No license = only license-free bands
        if self._license_class == LicenseClass.NONE:
            return False, "No amateur license - TX only allowed on license-free bands (CB, MURS, FRS)", None

        # Check amateur bands for licensed operators
        for band in self._amateur_bands:
            if band.start_hz <= frequency_hz <= band.end_hz:
                if band.is_allowed(self._license_class, mode):
                    return True, None, band
                else:
                    # Found band but wrong license or mode
                    if self._license_class not in band.licenses:
                        return False, f"Frequency {frequency_hz/1e6:.3f} MHz requires higher license class", None
                    if band.modes and mode:
                        return False, f"Mode '{mode}' not allowed in {band.name} - allowed: {band.modes}", None

        # Frequency not in any amateur or license-free band
        return False, f"Frequency {frequency_hz/1e6:.3f} MHz is not in any amateur or license-free band", None

    def get_power_limit(self, frequency_hz: float, mode: str = "") -> Optional[float]:
        """
        Get the legal TX power limit at the given frequency.

        Args:
            frequency_hz: Frequency in Hz
            mode: Operating mode

        Returns:
            Legal maximum power in watts, or None if no limit
        """
        allowed, _, privilege = self._check_license_privilege(frequency_hz, mode)
        if allowed and privilege:
            return privilege.get_legal_power_limit()
        return None

    def get_effective_power_limit(self, frequency_hz: float, mode: str = "") -> Optional[float]:
        """
        Get the effective TX power limit with headroom (150% of legal).

        This allows headroom for filtering, cable losses, and amplifier
        efficiency. Real radiated power should be verified with a 50Ω
        dummy load before transmitting.

        Args:
            frequency_hz: Frequency in Hz
            mode: Operating mode

        Returns:
            Effective maximum power in watts (150% of legal), or None if no limit
        """
        allowed, _, privilege = self._check_license_privilege(frequency_hz, mode)
        if allowed and privilege:
            return privilege.get_effective_power_limit()
        return None

    def get_tx_power_warning(self) -> str:
        """Get the TX power testing warning message."""
        return TX_POWER_WARNING

    def is_tx_allowed(
        self,
        frequency_hz: float,
        bandwidth_hz: float = 0,
        mode: str = ""
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if transmission is allowed at the given frequency.

        This checks both:
        1. Hardware/regulatory lockouts (GPS, aviation, etc.) - ALWAYS blocked
        2. License privileges - blocked if insufficient license

        Args:
            frequency_hz: Center frequency in Hz
            bandwidth_hz: Signal bandwidth in Hz (checks edges too)
            mode: Operating mode (CW, SSB, FM, etc.)

        Returns:
            Tuple of (allowed, reason_if_blocked)
        """
        # Check center frequency and band edges
        freqs_to_check = [frequency_hz]
        if bandwidth_hz > 0:
            freqs_to_check.extend([
                frequency_hz - bandwidth_hz / 2,
                frequency_hz + bandwidth_hz / 2
            ])

        # First check: Hardware/regulatory lockouts (always blocked regardless of license)
        for freq in freqs_to_check:
            for band in self._lockout_bands + self._custom_lockouts:
                if band.start_hz <= freq <= band.end_hz:
                    reason = f"TX BLOCKED: {band.name} - {band.description}"
                    if band.lockout_reason == LockoutReason.GPS:
                        reason += " [GPS SPOOFING IS DANGEROUS AND ILLEGAL]"
                    logger.warning(f"TX blocked at {frequency_hz/1e6:.3f} MHz: {band.name}")
                    return False, reason

        # Second check: License privileges
        for freq in freqs_to_check:
            allowed, reason, _ = self._check_license_privilege(freq, mode)
            if not allowed:
                logger.warning(f"TX blocked at {freq/1e6:.3f} MHz: {reason}")
                return False, f"LICENSE: {reason}"

        return True, None

    def validate_tx_frequency(self, frequency_hz: float, bandwidth_hz: float = 0) -> None:
        """
        Validate a TX frequency, raising an exception if blocked.

        Args:
            frequency_hz: Center frequency in Hz
            bandwidth_hz: Signal bandwidth in Hz

        Raises:
            ValueError: If transmission is not allowed at this frequency
        """
        allowed, reason = self.is_tx_allowed(frequency_hz, bandwidth_hz)
        if not allowed:
            raise ValueError(reason)

    def get_lockout_reason(self, frequency_hz: float) -> Optional[Tuple[str, LockoutReason]]:
        """
        Get the lockout reason for a frequency, if any.

        Args:
            frequency_hz: Frequency to check

        Returns:
            Tuple of (band_name, reason) or None if not locked
        """
        for band in self._lockout_bands + self._custom_lockouts:
            if band.start_hz <= frequency_hz <= band.end_hz:
                return band.name, band.lockout_reason
        return None

    def get_rx_presets(self, category: Optional[str] = None) -> List[FrequencyPreset]:
        """
        Get RX frequency presets, optionally filtered by category.

        Args:
            category: Filter by category (e.g., "GNSS", "Aviation")

        Returns:
            List of frequency presets
        """
        if category:
            return [p for p in self._rx_presets if p.category == category]
        return list(self._rx_presets)

    def get_preset_categories(self) -> List[str]:
        """Get list of preset categories."""
        return sorted(set(p.category for p in self._rx_presets))

    def get_preset_by_name(self, name: str) -> Optional[FrequencyPreset]:
        """Get a preset by name."""
        for preset in self._rx_presets:
            if preset.name == name:
                return preset
        return None

    def add_custom_lockout(self, band: FrequencyBand) -> None:
        """Add a custom TX lockout band."""
        self._custom_lockouts.append(band)
        logger.info(f"Added custom lockout: {band.name}")

    def get_all_lockouts(self) -> List[FrequencyBand]:
        """Get all TX lockout bands."""
        return self._lockout_bands + self._custom_lockouts

    def get_gps_lockouts(self) -> List[FrequencyBand]:
        """Get all GPS-related lockout bands."""
        return [b for b in self._lockout_bands if b.lockout_reason == LockoutReason.GPS]


# Global frequency manager instance
_frequency_manager: Optional[FrequencyManager] = None


def get_frequency_manager() -> FrequencyManager:
    """Get the global frequency manager instance."""
    global _frequency_manager
    if _frequency_manager is None:
        _frequency_manager = FrequencyManager()
    return _frequency_manager


def is_tx_allowed(
    frequency_hz: float,
    bandwidth_hz: float = 0,
    mode: str = ""
) -> Tuple[bool, Optional[str]]:
    """Check if TX is allowed at the given frequency."""
    return get_frequency_manager().is_tx_allowed(frequency_hz, bandwidth_hz, mode)


def validate_tx_frequency(frequency_hz: float, bandwidth_hz: float = 0) -> None:
    """Validate a TX frequency, raising ValueError if blocked."""
    get_frequency_manager().validate_tx_frequency(frequency_hz, bandwidth_hz)


def get_rx_presets(category: Optional[str] = None) -> List[FrequencyPreset]:
    """Get RX frequency presets."""
    return get_frequency_manager().get_rx_presets(category)


def set_license_class(license_class: LicenseClass) -> None:
    """Set the operator's license class."""
    get_frequency_manager().set_license_class(license_class)


def get_license_class() -> LicenseClass:
    """Get the current license class."""
    return get_frequency_manager().get_license_class()


def get_license_privileges() -> List[BandPrivilege]:
    """Get all band privileges available to the current license class."""
    return get_frequency_manager().get_license_privileges()


def get_power_limit(frequency_hz: float, mode: str = "") -> Optional[float]:
    """Get the legal TX power limit at the given frequency."""
    return get_frequency_manager().get_power_limit(frequency_hz, mode)


def get_effective_power_limit(frequency_hz: float, mode: str = "") -> Optional[float]:
    """Get the effective TX power limit with 150% headroom."""
    return get_frequency_manager().get_effective_power_limit(frequency_hz, mode)


def get_tx_power_warning() -> str:
    """Get the TX power testing warning message."""
    return TX_POWER_WARNING


__all__ = [
    # Enums
    'LockoutReason',
    'LicenseClass',
    # Dataclasses
    'FrequencyBand',
    'FrequencyPreset',
    'BandPrivilege',
    # Manager class
    'FrequencyManager',
    # Constants
    'TX_LOCKOUT_BANDS',
    'RX_PRESETS',
    'LICENSE_FREE_BANDS',
    'AMATEUR_BAND_PRIVILEGES',
    'POWER_HEADROOM_FACTOR',
    'TX_POWER_WARNING',
    # Singleton functions
    'get_frequency_manager',
    'is_tx_allowed',
    'validate_tx_frequency',
    'get_rx_presets',
    # License functions
    'set_license_class',
    'get_license_class',
    'get_license_privileges',
    # Power functions
    'get_power_limit',
    'get_effective_power_limit',
    'get_tx_power_warning',
]

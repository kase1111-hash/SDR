"""
Frequency management with TX lockouts and RX presets.

This module provides:
- TX frequency lockouts for protected bands (GPS, emergency, aviation, etc.)
- RX frequency presets for common signals
- Frequency validation before transmission

SAFETY: GPS spoofing is extremely dangerous and illegal. Spoofed GPS signals
have caused aircraft navigation failures and could result in loss of life.
GPS frequencies are permanently locked out from transmission.
"""

from __future__ import annotations

import logging
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

logger = logging.getLogger(__name__)


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
    Manages frequency validation, lockouts, and presets.

    This class ensures that protected frequencies cannot be used for
    transmission while allowing receive on any frequency.
    """

    def __init__(self):
        """Initialize the frequency manager."""
        self._lockout_bands = list(TX_LOCKOUT_BANDS)
        self._rx_presets = list(RX_PRESETS)
        self._custom_lockouts: List[FrequencyBand] = []
        logger.info(f"FrequencyManager initialized with {len(self._lockout_bands)} lockout bands")

    def is_tx_allowed(self, frequency_hz: float, bandwidth_hz: float = 0) -> Tuple[bool, Optional[str]]:
        """
        Check if transmission is allowed at the given frequency.

        Args:
            frequency_hz: Center frequency in Hz
            bandwidth_hz: Signal bandwidth in Hz (checks edges too)

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

        for freq in freqs_to_check:
            for band in self._lockout_bands + self._custom_lockouts:
                if band.start_hz <= freq <= band.end_hz:
                    reason = f"TX BLOCKED: {band.name} - {band.description}"
                    if band.lockout_reason == LockoutReason.GPS:
                        reason += " [GPS SPOOFING IS DANGEROUS AND ILLEGAL]"
                    logger.warning(f"TX blocked at {frequency_hz/1e6:.3f} MHz: {band.name}")
                    return False, reason

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


def is_tx_allowed(frequency_hz: float, bandwidth_hz: float = 0) -> Tuple[bool, Optional[str]]:
    """Check if TX is allowed at the given frequency."""
    return get_frequency_manager().is_tx_allowed(frequency_hz, bandwidth_hz)


def validate_tx_frequency(frequency_hz: float, bandwidth_hz: float = 0) -> None:
    """Validate a TX frequency, raising ValueError if blocked."""
    get_frequency_manager().validate_tx_frequency(frequency_hz, bandwidth_hz)


def get_rx_presets(category: Optional[str] = None) -> List[FrequencyPreset]:
    """Get RX frequency presets."""
    return get_frequency_manager().get_rx_presets(category)


__all__ = [
    'LockoutReason',
    'FrequencyBand',
    'FrequencyPreset',
    'FrequencyManager',
    'TX_LOCKOUT_BANDS',
    'RX_PRESETS',
    'get_frequency_manager',
    'is_tx_allowed',
    'validate_tx_frequency',
    'get_rx_presets',
]

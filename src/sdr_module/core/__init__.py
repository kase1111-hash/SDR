"""
Core module - Device management and sample handling.
"""

from .device_manager import DeviceManager
from .dual_sdr import DualSDRController
from .sample_buffer import SampleBuffer
from .config import SDRConfig
from .frequency_manager import (
    # Enums
    LockoutReason,
    LicenseClass,
    # Dataclasses
    FrequencyBand,
    FrequencyPreset,
    BandPrivilege,
    # Manager class
    FrequencyManager,
    # Constants
    TX_LOCKOUT_BANDS,
    RX_PRESETS,
    LICENSE_FREE_BANDS,
    AMATEUR_BAND_PRIVILEGES,
    # Singleton functions
    get_frequency_manager,
    is_tx_allowed,
    validate_tx_frequency,
    get_rx_presets,
    # License functions
    set_license_class,
    get_license_class,
    get_license_privileges,
    get_power_limit,
)

__all__ = [
    "DeviceManager",
    "DualSDRController",
    "SampleBuffer",
    "SDRConfig",
    # Frequency management - Enums
    "LockoutReason",
    "LicenseClass",
    # Frequency management - Dataclasses
    "FrequencyBand",
    "FrequencyPreset",
    "BandPrivilege",
    # Frequency management - Manager
    "FrequencyManager",
    # Frequency management - Constants
    "TX_LOCKOUT_BANDS",
    "RX_PRESETS",
    "LICENSE_FREE_BANDS",
    "AMATEUR_BAND_PRIVILEGES",
    # Frequency management - Functions
    "get_frequency_manager",
    "is_tx_allowed",
    "validate_tx_frequency",
    "get_rx_presets",
    # License functions
    "set_license_class",
    "get_license_class",
    "get_license_privileges",
    "get_power_limit",
]

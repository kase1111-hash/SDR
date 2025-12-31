"""
Core module - Device management and sample handling.
"""

from .config import SDRConfig
from .device_manager import DeviceManager
from .dual_sdr import DualSDRController
from .frequency_manager import (  # Enums; Dataclasses; Manager class; Constants; Singleton functions; License functions; Power functions
    AMATEUR_BAND_PRIVILEGES,
    LICENSE_FREE_BANDS,
    POWER_HEADROOM_FACTOR,
    RX_PRESETS,
    TX_LOCKOUT_BANDS,
    TX_POWER_WARNING,
    BandPrivilege,
    FrequencyBand,
    FrequencyManager,
    FrequencyPreset,
    LicenseClass,
    LockoutReason,
    get_effective_power_limit,
    get_frequency_manager,
    get_license_class,
    get_license_privileges,
    get_power_limit,
    get_rx_presets,
    get_tx_power_warning,
    is_tx_allowed,
    set_license_class,
    validate_tx_frequency,
)
from .sample_buffer import SampleBuffer

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
    "POWER_HEADROOM_FACTOR",
    "TX_POWER_WARNING",
    # Frequency management - Functions
    "get_frequency_manager",
    "is_tx_allowed",
    "validate_tx_frequency",
    "get_rx_presets",
    # License functions
    "set_license_class",
    "get_license_class",
    "get_license_privileges",
    # Power functions
    "get_power_limit",
    "get_effective_power_limit",
    "get_tx_power_warning",
]

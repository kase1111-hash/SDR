"""
Core module - Device management and sample handling.
"""

from .device_manager import DeviceManager
from .dual_sdr import DualSDRController
from .sample_buffer import SampleBuffer
from .config import SDRConfig
from .frequency_manager import (
    LockoutReason,
    FrequencyBand,
    FrequencyPreset,
    FrequencyManager,
    TX_LOCKOUT_BANDS,
    RX_PRESETS,
    get_frequency_manager,
    is_tx_allowed,
    validate_tx_frequency,
    get_rx_presets,
)

__all__ = [
    "DeviceManager",
    "DualSDRController",
    "SampleBuffer",
    "SDRConfig",
    # Frequency management
    "LockoutReason",
    "FrequencyBand",
    "FrequencyPreset",
    "FrequencyManager",
    "TX_LOCKOUT_BANDS",
    "RX_PRESETS",
    "get_frequency_manager",
    "is_tx_allowed",
    "validate_tx_frequency",
    "get_rx_presets",
]

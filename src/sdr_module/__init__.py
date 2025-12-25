"""
SDR Module - Software Defined Radio Framework

A dual-SDR system designed for RTL-SDR + HackRF One operation.
Provides signal visualization, frequency analysis, signal classification,
and protocol identification capabilities.

Supported Hardware:
    - RTL-SDR (RX only): 500 kHz - 1.7 GHz
    - HackRF One (TX/RX): 1 MHz - 6 GHz

Combined Capabilities:
    - Frequency Coverage: 500 kHz - 6 GHz
    - Full-Duplex: RTL-SDR RX + HackRF TX
    - Combined Bandwidth: 22.4 MHz
"""

__version__ = "0.1.0"
__author__ = "SDR Module Team"

from .core.device_manager import DeviceManager
from .core.dual_sdr import DualSDRController
from .core.sample_buffer import SampleBuffer

__all__ = [
    "DeviceManager",
    "DualSDRController",
    "SampleBuffer",
    "__version__",
]

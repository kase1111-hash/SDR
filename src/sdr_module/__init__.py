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

Antenna Array Support:
    The module supports multi-SDR antenna arrays for beamforming and
    direction finding. See sdr_module.antenna_array for configuration
    and controller classes.

Plugin System:
    The module supports extensibility through plugins. See sdr_module.plugins
    for the plugin API and base classes.

    Plugin Types:
        - ProtocolPlugin: Custom protocol decoders
        - DemodulatorPlugin: Custom demodulation algorithms
        - DevicePlugin: Custom SDR device drivers
        - ProcessorPlugin: Custom signal processing blocks
"""

__version__ = "0.1.0"
__author__ = "SDR Module Team"

from .core.device_manager import DeviceManager
from .core.dual_sdr import DualSDRController
from .core.sample_buffer import SampleBuffer

# Antenna array support
from .antenna_array import (
    AntennaArrayController,
    ArrayConfig,
    ArrayGeometry,
    TimestampedSampleBuffer,
    create_linear_2_element,
    create_linear_4_element,
)

# Plugin system
from .plugins import (
    DemodulatorPlugin,
    DevicePlugin,
    Plugin,
    PluginManager,
    PluginMetadata,
    PluginRegistry,
    PluginState,
    PluginType,
    ProcessorPlugin,
    ProtocolPlugin,
)

__all__ = [
    # Core
    "DeviceManager",
    "DualSDRController",
    "SampleBuffer",
    # Antenna array
    "AntennaArrayController",
    "ArrayConfig",
    "ArrayGeometry",
    "TimestampedSampleBuffer",
    "create_linear_2_element",
    "create_linear_4_element",
    # Plugin system
    "PluginManager",
    "PluginRegistry",
    "Plugin",
    "PluginMetadata",
    "PluginType",
    "PluginState",
    "ProtocolPlugin",
    "DemodulatorPlugin",
    "DevicePlugin",
    "ProcessorPlugin",
    # Version
    "__version__",
]

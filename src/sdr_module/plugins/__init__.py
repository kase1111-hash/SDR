"""
SDR Module Plugin System.

Provides a flexible plugin architecture for extending SDR functionality
with custom protocol decoders, demodulators, device drivers, and more.

Plugin Types:
    - ProtocolPlugin: Custom protocol decoders
    - DemodulatorPlugin: Custom demodulation algorithms
    - DevicePlugin: Custom SDR device drivers
    - ProcessorPlugin: Custom signal processing blocks

Usage:
    from sdr_module.plugins import PluginManager

    # Create manager and load plugins
    manager = PluginManager()
    manager.discover_plugins("~/.sdr_module/plugins")
    manager.load_all()

    # Get available plugins
    for plugin in manager.list_plugins():
        print(f"{plugin.name} v{plugin.version}")

    # Use a specific plugin
    decoder = manager.get_plugin("my_protocol_decoder")
"""

from .base import (
    DemodulatorPlugin,
    DevicePlugin,
    Plugin,
    PluginError,
    PluginMetadata,
    PluginState,
    PluginType,
    ProcessorPlugin,
    ProtocolPlugin,
)
from .manager import PluginManager
from .registry import PluginRegistry

__all__ = [
    # Base classes
    "Plugin",
    "PluginMetadata",
    "PluginType",
    "PluginState",
    "PluginError",
    # Plugin types
    "ProtocolPlugin",
    "DemodulatorPlugin",
    "DevicePlugin",
    "ProcessorPlugin",
    # Manager
    "PluginManager",
    "PluginRegistry",
]

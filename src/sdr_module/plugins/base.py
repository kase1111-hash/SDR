"""
Plugin base classes and metadata.

Provides abstract base classes for different plugin types
and metadata structures for plugin identification.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

import numpy as np


class PluginType(Enum):
    """Types of plugins supported by the system."""

    PROTOCOL = auto()  # Protocol decoder plugin
    DEMODULATOR = auto()  # Demodulator plugin
    DEVICE = auto()  # SDR device driver plugin
    PROCESSOR = auto()  # Signal processor plugin
    UI_WIDGET = auto()  # UI widget plugin


class PluginState(Enum):
    """Plugin lifecycle states."""

    DISCOVERED = auto()  # Plugin file found
    LOADED = auto()  # Plugin module loaded
    INITIALIZED = auto()  # Plugin instance created
    ENABLED = auto()  # Plugin active and usable
    DISABLED = auto()  # Plugin loaded but not active
    ERROR = auto()  # Plugin failed to load/initialize


class PluginError(Exception):
    """
    Base exception for plugin-related errors.

    All plugin exceptions inherit from this class, allowing callers to catch
    all plugin errors with a single except clause if desired.

    Attributes:
        plugin_name: Name of the plugin that caused the error
        cause: Original exception that caused this error (if any)
    """

    def __init__(
        self, message: str, plugin_name: str = "", cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.plugin_name = plugin_name
        self.cause = cause

    def __str__(self) -> str:
        msg = super().__str__()
        if self.plugin_name:
            msg = f"[{self.plugin_name}] {msg}"
        if self.cause:
            msg = f"{msg} (caused by: {self.cause})"
        return msg


class PluginLoadError(PluginError):
    """
    Raised when a plugin fails to load.

    This typically occurs when:
    - Plugin file cannot be found or read
    - Plugin module has syntax errors
    - Plugin dependencies are missing
    """

    pass


class PluginInitError(PluginError):
    """
    Raised when a plugin fails to initialize.

    This typically occurs when:
    - Plugin class cannot be instantiated
    - Plugin's __init__ raises an exception
    - Plugin configuration is invalid
    """

    pass


class PluginNotFoundError(PluginError):
    """
    Raised when a requested plugin is not found.

    This typically occurs when:
    - Plugin with the given name is not registered
    - Plugin was unloaded or disabled
    """

    pass


@dataclass
class PluginMetadata:
    """
    Plugin metadata and identification.

    Attributes:
        name: Unique plugin identifier
        version: Semantic version string
        author: Plugin author name
        description: Brief description of functionality
        plugin_type: Type of plugin (protocol, demodulator, etc.)
        dependencies: List of required plugin names
        min_api_version: Minimum plugin API version required
        tags: Searchable tags for categorization
        homepage: URL to plugin homepage/documentation
        license: License identifier (e.g., "MIT", "GPL-3.0")
    """

    name: str
    version: str
    plugin_type: PluginType
    author: str = ""
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    min_api_version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    homepage: str = ""
    license: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "plugin_type": self.plugin_type.name,
            "author": self.author,
            "description": self.description,
            "dependencies": self.dependencies,
            "min_api_version": self.min_api_version,
            "tags": self.tags,
            "homepage": self.homepage,
            "license": self.license,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PluginMetadata":
        """Create metadata from dictionary."""
        plugin_type = data.get("plugin_type", "PROTOCOL")
        if isinstance(plugin_type, str):
            plugin_type = PluginType[plugin_type]

        return cls(
            name=data["name"],
            version=data.get("version", "0.0.0"),
            plugin_type=plugin_type,
            author=data.get("author", ""),
            description=data.get("description", ""),
            dependencies=data.get("dependencies", []),
            min_api_version=data.get("min_api_version", "1.0.0"),
            tags=data.get("tags", []),
            homepage=data.get("homepage", ""),
            license=data.get("license", ""),
        )


class Plugin(ABC):
    """
    Abstract base class for all plugins.

    All plugin types must inherit from this class and implement
    the required abstract methods.

    Example:
        class MyProtocolPlugin(ProtocolPlugin):
            @classmethod
            def get_metadata(cls) -> PluginMetadata:
                return PluginMetadata(
                    name="my_protocol",
                    version="1.0.0",
                    plugin_type=PluginType.PROTOCOL,
                    author="Developer",
                    description="Decodes My Protocol"
                )

            def initialize(self, config: Dict[str, Any]) -> bool:
                self._sample_rate = config.get("sample_rate", 2.4e6)
                return True

            # ... implement other required methods
    """

    def __init__(self):
        self._state: PluginState = PluginState.LOADED
        self._config: Dict[str, Any] = {}
        self._error_message: str = ""

    @classmethod
    @abstractmethod
    def get_metadata(cls) -> PluginMetadata:
        """
        Get plugin metadata.

        Returns:
            PluginMetadata describing this plugin
        """
        pass

    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the plugin with configuration.

        Called after the plugin is loaded but before it's enabled.
        Use this to set up resources, validate config, etc.

        Args:
            config: Configuration dictionary

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    def enable(self) -> bool:
        """
        Enable the plugin.

        Called to activate the plugin after initialization.

        Returns:
            True if enabled successfully
        """
        if (
            self._state == PluginState.INITIALIZED
            or self._state == PluginState.DISABLED
        ):
            self._state = PluginState.ENABLED
            return True
        return False

    def disable(self) -> bool:
        """
        Disable the plugin.

        Called to deactivate the plugin without unloading.

        Returns:
            True if disabled successfully
        """
        if self._state == PluginState.ENABLED:
            self._state = PluginState.DISABLED
            return True
        return False

    def cleanup(self) -> None:
        """
        Clean up plugin resources.

        Called before the plugin is unloaded.
        Override to release resources, close files, etc.
        """
        pass

    @property
    def state(self) -> PluginState:
        """Get current plugin state."""
        return self._state

    @property
    def is_enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._state == PluginState.ENABLED

    @property
    def error_message(self) -> str:
        """Get error message if plugin is in error state."""
        return self._error_message

    def get_config_schema(self) -> Dict[str, Any]:
        """
        Get configuration schema for the plugin.

        Override to define configuration options.

        Returns:
            JSON Schema-like dictionary describing config options
        """
        return {}

    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration.

        Args:
            config: Configuration to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        return []


class ProtocolPlugin(Plugin):
    """
    Base class for protocol decoder plugins.

    Protocol plugins decode digital protocol data from I/Q samples.
    They integrate with the existing ProtocolDecoder framework.
    """

    @abstractmethod
    def get_protocol_info(self) -> Dict[str, Any]:
        """
        Get protocol information.

        Returns:
            Dictionary with protocol details:
                - name: Protocol name
                - protocol_type: Category (ism, amateur, aviation, etc.)
                - frequency_range: (min_hz, max_hz)
                - bandwidth_hz: Required bandwidth
                - modulation: Modulation type
                - symbol_rate: Symbol rate (optional)
                - description: Human-readable description
        """
        pass

    @abstractmethod
    def decode(self, samples: np.ndarray) -> List[Dict[str, Any]]:
        """
        Decode protocol data from samples.

        Args:
            samples: Complex I/Q samples

        Returns:
            List of decoded frames as dictionaries
        """
        pass

    @abstractmethod
    def can_decode(self, samples: np.ndarray) -> float:
        """
        Estimate probability that samples contain this protocol.

        Args:
            samples: Complex I/Q samples

        Returns:
            Confidence score from 0.0 to 1.0
        """
        pass

    def reset(self) -> None:
        """Reset decoder state between captures."""
        pass


class DemodulatorPlugin(Plugin):
    """
    Base class for demodulator plugins.

    Demodulator plugins implement custom demodulation algorithms
    for extracting baseband signals from modulated carriers.
    """

    @abstractmethod
    def get_modulation_info(self) -> Dict[str, Any]:
        """
        Get modulation information.

        Returns:
            Dictionary with modulation details:
                - name: Modulation name
                - modulation_type: Category (analog, digital)
                - supported_variants: List of variants (e.g., ["bpsk", "qpsk"])
                - description: Human-readable description
        """
        pass

    @abstractmethod
    def demodulate(self, samples: np.ndarray) -> np.ndarray:
        """
        Demodulate samples.

        Args:
            samples: Complex I/Q samples

        Returns:
            Demodulated output (real or complex depending on type)
        """
        pass

    def set_sample_rate(self, rate: float) -> None:
        """
        Set the sample rate.

        Args:
            rate: Sample rate in Hz
        """
        self._sample_rate = rate

    def get_sample_rate(self) -> float:
        """Get current sample rate."""
        return getattr(self, "_sample_rate", 0.0)

    def reset(self) -> None:
        """Reset demodulator state."""
        pass


class DevicePlugin(Plugin):
    """
    Base class for SDR device driver plugins.

    Device plugins provide support for additional SDR hardware
    beyond the built-in RTL-SDR and HackRF drivers.
    """

    @abstractmethod
    def get_device_info(self) -> Dict[str, Any]:
        """
        Get device information.

        Returns:
            Dictionary with device details:
                - name: Device name
                - manufacturer: Manufacturer name
                - capabilities: List of capabilities (rx, tx, etc.)
                - frequency_range: (min_hz, max_hz)
                - sample_rate_range: (min_hz, max_hz)
                - description: Human-readable description
        """
        pass

    @abstractmethod
    def enumerate_devices(self) -> List[Dict[str, Any]]:
        """
        Enumerate available devices of this type.

        Returns:
            List of device info dictionaries
        """
        pass

    @abstractmethod
    def open(self, device_index: int = 0) -> bool:
        """
        Open a device.

        Args:
            device_index: Index of device to open

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the device."""
        pass

    @abstractmethod
    def set_frequency(self, freq_hz: float) -> bool:
        """Set center frequency."""
        pass

    @abstractmethod
    def set_sample_rate(self, rate_hz: float) -> bool:
        """Set sample rate."""
        pass

    @abstractmethod
    def start_rx(self) -> bool:
        """Start receiving samples."""
        pass

    @abstractmethod
    def stop_rx(self) -> bool:
        """Stop receiving samples."""
        pass

    @abstractmethod
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """
        Read samples from device.

        Args:
            num_samples: Number of complex samples to read

        Returns:
            Complex numpy array or None on error
        """
        pass


class ProcessorPlugin(Plugin):
    """
    Base class for signal processor plugins.

    Processor plugins implement custom signal processing blocks
    that can be chained in a processing pipeline.
    """

    @abstractmethod
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get processor information.

        Returns:
            Dictionary with processor details:
                - name: Processor name
                - category: Category (filter, transform, analysis)
                - input_type: Expected input type
                - output_type: Output type
                - description: Human-readable description
        """
        pass

    @abstractmethod
    def process(self, samples: np.ndarray) -> np.ndarray:
        """
        Process samples.

        Args:
            samples: Input samples

        Returns:
            Processed samples
        """
        pass

    def set_parameter(self, name: str, value: Any) -> bool:
        """
        Set a processing parameter.

        Args:
            name: Parameter name
            value: Parameter value

        Returns:
            True if parameter was set successfully
        """
        return False

    def get_parameter(self, name: str) -> Optional[Any]:
        """
        Get a processing parameter.

        Args:
            name: Parameter name

        Returns:
            Parameter value or None if not found
        """
        return None

    def get_parameters(self) -> Dict[str, Any]:
        """Get all processing parameters."""
        return {}

    def reset(self) -> None:
        """Reset processor state."""
        pass


# Plugin API version for compatibility checking
PLUGIN_API_VERSION = "1.0.0"


def check_api_compatibility(required_version: str) -> bool:
    """
    Check if the required API version is compatible.

    Args:
        required_version: Minimum required API version

    Returns:
        True if compatible
    """
    required = [int(x) for x in required_version.split(".")]
    current = [int(x) for x in PLUGIN_API_VERSION.split(".")]

    # Major version must match, minor/patch can be higher
    if len(required) >= 1 and len(current) >= 1:
        if required[0] != current[0]:
            return False
        if len(required) >= 2 and len(current) >= 2:
            if required[1] > current[1]:
                return False

    return True

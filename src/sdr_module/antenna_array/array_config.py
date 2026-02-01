"""
Configuration for antenna array systems.

Defines array geometry, element positions, calibration data,
and synchronization parameters.
"""

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import ConfigValidationError, DeviceConfig

logger = logging.getLogger(__name__)

# Speed of light in m/s
SPEED_OF_LIGHT = 299792458.0


@dataclass
class ElementPosition:
    """
    Position of an antenna element in 3D space.

    Uses a right-handed coordinate system where:
    - X: positive to the right (East)
    - Y: positive forward (North)
    - Z: positive up

    All positions are in meters relative to the array center.
    """

    x: float = 0.0  # X position in meters
    y: float = 0.0  # Y position in meters
    z: float = 0.0  # Z position in meters

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [x, y, z]."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "ElementPosition":
        """Create from numpy array."""
        return cls(x=float(arr[0]), y=float(arr[1]), z=float(arr[2]))

    def distance_to(self, other: "ElementPosition") -> float:
        """Calculate Euclidean distance to another element."""
        return math.sqrt(
            (self.x - other.x) ** 2
            + (self.y - other.y) ** 2
            + (self.z - other.z) ** 2
        )


@dataclass
class ElementCalibration:
    """
    Calibration data for a single antenna element.

    Stores phase and amplitude corrections to compensate for
    hardware variations between array elements.
    """

    element_index: int = 0
    phase_offset: float = 0.0  # Phase offset in radians
    amplitude_scale: float = 1.0  # Amplitude correction factor
    delay_samples: float = 0.0  # Fractional sample delay correction
    frequency_offset: float = 0.0  # Frequency offset in Hz (for non-coherent SDRs)

    # Optional per-frequency calibration
    calibration_frequency: float = 0.0  # Frequency at which calibration was performed
    calibration_timestamp: float = 0.0  # Unix timestamp of calibration

    def get_correction_phasor(self) -> complex:
        """Get complex correction factor for this element."""
        return self.amplitude_scale * np.exp(-1j * self.phase_offset)


@dataclass
class ArrayElement:
    """
    Complete specification for one antenna array element.

    Combines device configuration, position, and calibration data.
    """

    index: int = 0
    device_type: str = "rtlsdr"
    device_index: int = 0
    position: ElementPosition = field(default_factory=ElementPosition)
    calibration: ElementCalibration = field(default_factory=ElementCalibration)
    enabled: bool = True

    # Optional device-specific config override
    device_config: Optional[DeviceConfig] = None

    def __post_init__(self) -> None:
        """Ensure calibration index matches element index."""
        self.calibration.element_index = self.index


@dataclass
class ArrayGeometry:
    """
    Predefined array geometry types with element positions.

    Provides factory methods for common array configurations.
    """

    geometry_type: str = "custom"  # "linear", "circular", "rectangular", "custom"
    num_elements: int = 2
    element_spacing: float = 0.5  # Spacing in wavelengths (for linear/rectangular)
    reference_frequency: float = 100e6  # Reference frequency for wavelength calculation
    elements: List[ElementPosition] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and generate element positions if needed."""
        if not self.elements and self.geometry_type != "custom":
            self.elements = self._generate_positions()

    def _generate_positions(self) -> List[ElementPosition]:
        """Generate element positions based on geometry type."""
        wavelength = SPEED_OF_LIGHT / self.reference_frequency
        spacing_m = self.element_spacing * wavelength

        if self.geometry_type == "linear":
            return self._generate_linear(spacing_m)
        elif self.geometry_type == "circular":
            return self._generate_circular(spacing_m)
        elif self.geometry_type == "rectangular":
            return self._generate_rectangular(spacing_m)
        else:
            return []

    def _generate_linear(self, spacing_m: float) -> List[ElementPosition]:
        """Generate linear array positions along X axis."""
        positions = []
        # Center the array at origin
        start_x = -spacing_m * (self.num_elements - 1) / 2
        for i in range(self.num_elements):
            positions.append(ElementPosition(x=start_x + i * spacing_m, y=0.0, z=0.0))
        return positions

    def _generate_circular(self, spacing_m: float) -> List[ElementPosition]:
        """Generate circular array positions in X-Y plane."""
        positions = []
        # Calculate radius from element spacing (chord length)
        if self.num_elements < 2:
            return [ElementPosition()]
        angle_step = 2 * math.pi / self.num_elements
        radius = spacing_m / (2 * math.sin(angle_step / 2))
        for i in range(self.num_elements):
            angle = i * angle_step
            positions.append(
                ElementPosition(
                    x=radius * math.cos(angle), y=radius * math.sin(angle), z=0.0
                )
            )
        return positions

    def _generate_rectangular(self, spacing_m: float) -> List[ElementPosition]:
        """Generate rectangular array positions in X-Y plane."""
        positions = []
        # Determine grid dimensions (prefer square)
        rows = int(math.sqrt(self.num_elements))
        cols = (self.num_elements + rows - 1) // rows

        # Center the array
        start_x = -spacing_m * (cols - 1) / 2
        start_y = -spacing_m * (rows - 1) / 2

        count = 0
        for row in range(rows):
            for col in range(cols):
                if count >= self.num_elements:
                    break
                positions.append(
                    ElementPosition(
                        x=start_x + col * spacing_m, y=start_y + row * spacing_m, z=0.0
                    )
                )
                count += 1
        return positions

    def get_wavelength(self) -> float:
        """Get wavelength at reference frequency in meters."""
        return SPEED_OF_LIGHT / self.reference_frequency

    def get_array_aperture(self) -> float:
        """Get maximum array aperture (largest element separation)."""
        if len(self.elements) < 2:
            return 0.0
        max_dist = 0.0
        for i, pos1 in enumerate(self.elements):
            for pos2 in self.elements[i + 1 :]:
                dist = pos1.distance_to(pos2)
                max_dist = max(max_dist, dist)
        return max_dist

    def get_position_matrix(self) -> np.ndarray:
        """Get element positions as Nx3 matrix."""
        return np.array([pos.to_array() for pos in self.elements], dtype=np.float64)


@dataclass
class SynchronizationConfig:
    """
    Configuration for array synchronization.

    Defines how multiple SDR devices are synchronized in time
    and frequency for coherent array processing.
    """

    method: str = "software"  # "software", "external_clock", "gpsdo", "correlation"
    reference_element: int = 0  # Index of reference element for phase alignment
    max_time_offset_us: float = 100.0  # Maximum allowed time offset in microseconds
    correlation_threshold: float = 0.7  # Minimum correlation for sync validation
    resync_interval_s: float = 10.0  # Re-synchronization interval in seconds

    # External clock settings
    external_clock_freq: float = 10e6  # External reference frequency (10 MHz typical)

    # Software sync settings
    calibration_tone_freq: float = 1e6  # Frequency offset for calibration tone
    calibration_tone_duration_s: float = 0.1  # Duration of calibration tone

    def __post_init__(self) -> None:
        """Validate configuration."""
        valid_methods = ("software", "external_clock", "gpsdo", "correlation")
        if self.method not in valid_methods:
            raise ConfigValidationError(
                f"sync method must be one of {valid_methods}, got {self.method}"
            )
        if self.max_time_offset_us <= 0:
            raise ConfigValidationError(
                f"max_time_offset_us must be positive, got {self.max_time_offset_us}"
            )
        if not (0.0 < self.correlation_threshold <= 1.0):
            raise ConfigValidationError(
                f"correlation_threshold must be in (0, 1], got {self.correlation_threshold}"
            )


@dataclass
class ArrayConfig:
    """
    Complete configuration for an antenna array system.

    Combines geometry, elements, synchronization settings,
    and common array parameters.
    """

    name: str = "default_array"
    geometry: ArrayGeometry = field(default_factory=ArrayGeometry)
    elements: List[ArrayElement] = field(default_factory=list)
    sync: SynchronizationConfig = field(default_factory=SynchronizationConfig)

    # Common settings applied to all elements (can be overridden per-element)
    common_frequency: float = 100e6  # Center frequency in Hz
    common_sample_rate: float = 2.4e6  # Sample rate in Hz
    common_bandwidth: float = 2.4e6  # Bandwidth in Hz
    common_gain: float = 30.0  # Gain in dB

    # Buffer settings
    buffer_capacity_chunks: int = 256
    buffer_capacity_samples: int = 4 * 1024 * 1024  # 4M samples per element

    # Processing settings
    enable_dc_removal: bool = True
    enable_iq_correction: bool = True

    def __post_init__(self) -> None:
        """Initialize elements from geometry if not provided."""
        if not self.elements and self.geometry.elements:
            self._create_elements_from_geometry()
        self._validate()

    def _create_elements_from_geometry(self) -> None:
        """Create ArrayElement instances from geometry positions."""
        for i, pos in enumerate(self.geometry.elements):
            element = ArrayElement(
                index=i,
                device_type="rtlsdr",
                device_index=i,
                position=pos,
                calibration=ElementCalibration(element_index=i),
            )
            self.elements.append(element)

    def _validate(self) -> None:
        """Validate array configuration."""
        if self.common_frequency <= 0:
            raise ConfigValidationError(
                f"common_frequency must be positive, got {self.common_frequency}"
            )
        if self.common_sample_rate <= 0:
            raise ConfigValidationError(
                f"common_sample_rate must be positive, got {self.common_sample_rate}"
            )

        # Validate element indices are unique
        indices = [e.index for e in self.elements]
        if len(indices) != len(set(indices)):
            raise ConfigValidationError("Element indices must be unique")

        # Validate reference element exists
        if self.elements and self.sync.reference_element >= len(self.elements):
            raise ConfigValidationError(
                f"Reference element {self.sync.reference_element} does not exist"
            )

    @property
    def num_elements(self) -> int:
        """Number of array elements."""
        return len(self.elements)

    @property
    def enabled_elements(self) -> List[ArrayElement]:
        """List of enabled elements."""
        return [e for e in self.elements if e.enabled]

    @property
    def wavelength(self) -> float:
        """Wavelength at common frequency in meters."""
        return SPEED_OF_LIGHT / self.common_frequency

    def get_element_by_index(self, index: int) -> Optional[ArrayElement]:
        """Get element by index."""
        for element in self.elements:
            if element.index == index:
                return element
        return None

    def get_element_by_device(
        self, device_type: str, device_index: int
    ) -> Optional[ArrayElement]:
        """Get element by device type and index."""
        for element in self.elements:
            if (
                element.device_type == device_type
                and element.device_index == device_index
            ):
                return element
        return None

    def get_position_matrix(self) -> np.ndarray:
        """Get element positions as Nx3 matrix (enabled elements only)."""
        return np.array(
            [e.position.to_array() for e in self.enabled_elements], dtype=np.float64
        )

    def get_calibration_vector(self) -> np.ndarray:
        """Get complex calibration corrections for all enabled elements."""
        return np.array(
            [e.calibration.get_correction_phasor() for e in self.enabled_elements],
            dtype=np.complex64,
        )

    def get_device_config(self, element_index: int) -> DeviceConfig:
        """
        Get DeviceConfig for a specific element.

        Uses element-specific config if available, otherwise common settings.
        """
        element = self.get_element_by_index(element_index)
        if element is None:
            raise ValueError(f"Element {element_index} not found")

        if element.device_config is not None:
            return element.device_config

        return DeviceConfig(
            device_type=element.device_type,
            device_index=element.device_index,
            frequency=self.common_frequency,
            sample_rate=self.common_sample_rate,
            bandwidth=self.common_bandwidth,
            gain=self.common_gain,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArrayConfig":
        """Create configuration from dictionary."""
        # Reconstruct nested dataclasses
        geometry_data = data.get("geometry", {})
        if "elements" in geometry_data:
            geometry_data["elements"] = [
                ElementPosition(**pos) for pos in geometry_data["elements"]
            ]
        geometry = ArrayGeometry(**geometry_data)

        elements_data = data.get("elements", [])
        elements = []
        for elem_data in elements_data:
            if "position" in elem_data:
                elem_data["position"] = ElementPosition(**elem_data["position"])
            if "calibration" in elem_data:
                elem_data["calibration"] = ElementCalibration(**elem_data["calibration"])
            if "device_config" in elem_data and elem_data["device_config"]:
                elem_data["device_config"] = DeviceConfig(**elem_data["device_config"])
            elements.append(ArrayElement(**elem_data))

        sync_data = data.get("sync", {})
        sync = SynchronizationConfig(**sync_data)

        return cls(
            name=data.get("name", "default_array"),
            geometry=geometry,
            elements=elements,
            sync=sync,
            common_frequency=data.get("common_frequency", 100e6),
            common_sample_rate=data.get("common_sample_rate", 2.4e6),
            common_bandwidth=data.get("common_bandwidth", 2.4e6),
            common_gain=data.get("common_gain", 30.0),
            buffer_capacity_chunks=data.get("buffer_capacity_chunks", 256),
            buffer_capacity_samples=data.get("buffer_capacity_samples", 4 * 1024 * 1024),
            enable_dc_removal=data.get("enable_dc_removal", True),
            enable_iq_correction=data.get("enable_iq_correction", True),
        )

    def save(self, path: str) -> bool:
        """Save configuration to JSON file."""
        try:
            with open(path, "w") as f:
                json.dump(self.to_dict(), f, indent=2)
            logger.info(f"Array configuration saved to {path}")
            return True
        except (OSError, IOError) as e:
            logger.error(f"Failed to save array configuration to {path}: {e}")
            return False

    @classmethod
    def load(cls, path: str) -> Optional["ArrayConfig"]:
        """Load configuration from JSON file."""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            config = cls.from_dict(data)
            logger.info(f"Array configuration loaded from {path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Array configuration file not found: {path}")
            return None
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to load array configuration from {path}: {e}")
            return None


# Preset array configurations
def create_linear_2_element(
    frequency: float = 433e6, spacing_wavelengths: float = 0.5
) -> ArrayConfig:
    """Create a 2-element linear array preset."""
    geometry = ArrayGeometry(
        geometry_type="linear",
        num_elements=2,
        element_spacing=spacing_wavelengths,
        reference_frequency=frequency,
    )
    config = ArrayConfig(
        name="linear_2_element",
        geometry=geometry,
        common_frequency=frequency,
        sync=SynchronizationConfig(method="software", reference_element=0),
    )
    return config


def create_linear_4_element(
    frequency: float = 433e6, spacing_wavelengths: float = 0.5
) -> ArrayConfig:
    """Create a 4-element linear array preset."""
    geometry = ArrayGeometry(
        geometry_type="linear",
        num_elements=4,
        element_spacing=spacing_wavelengths,
        reference_frequency=frequency,
    )
    config = ArrayConfig(
        name="linear_4_element",
        geometry=geometry,
        common_frequency=frequency,
        sync=SynchronizationConfig(method="software", reference_element=0),
    )
    return config


def create_circular_4_element(
    frequency: float = 433e6, spacing_wavelengths: float = 0.5
) -> ArrayConfig:
    """Create a 4-element circular array preset."""
    geometry = ArrayGeometry(
        geometry_type="circular",
        num_elements=4,
        element_spacing=spacing_wavelengths,
        reference_frequency=frequency,
    )
    config = ArrayConfig(
        name="circular_4_element",
        geometry=geometry,
        common_frequency=frequency,
        sync=SynchronizationConfig(method="software", reference_element=0),
    )
    return config


def create_rectangular_2x2(
    frequency: float = 433e6, spacing_wavelengths: float = 0.5
) -> ArrayConfig:
    """Create a 2x2 rectangular array preset."""
    geometry = ArrayGeometry(
        geometry_type="rectangular",
        num_elements=4,
        element_spacing=spacing_wavelengths,
        reference_frequency=frequency,
    )
    config = ArrayConfig(
        name="rectangular_2x2",
        geometry=geometry,
        common_frequency=frequency,
        sync=SynchronizationConfig(method="software", reference_element=0),
    )
    return config


ARRAY_PRESETS: Dict[str, ArrayConfig] = {
    "linear_2": create_linear_2_element(),
    "linear_4": create_linear_4_element(),
    "circular_4": create_circular_4_element(),
    "rectangular_2x2": create_rectangular_2x2(),
}


def get_array_preset(name: str) -> Optional[ArrayConfig]:
    """Get an array preset by name."""
    return ARRAY_PRESETS.get(name)


def list_array_presets() -> List[str]:
    """List available array preset names."""
    return list(ARRAY_PRESETS.keys())

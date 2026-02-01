"""
Antenna Array Module for SDR.

Provides support for multi-SDR antenna array operation including:
- N-device orchestration with AntennaArrayController
- Timestamped sample buffers for synchronization
- Array geometry and calibration configuration
- Support for beamforming and direction finding (Phase 2)

Example:
    from sdr_module.antenna_array import (
        AntennaArrayController,
        ArrayConfig,
        create_linear_2_element,
    )

    # Create a 2-element linear array configuration
    config = create_linear_2_element(frequency=433e6)

    # Initialize and use the array
    with AntennaArrayController(config) as array:
        array.start_receive(sample_callback=process_samples)
        # ... do processing ...

Phase 1 Components (Foundation):
    - TimestampedSampleBuffer: Sample buffer with timing metadata
    - ArrayConfig: Array geometry, elements, and calibration
    - AntennaArrayController: N-device orchestration

Phase 2 Components (Planned):
    - CrossCorrelator: Phase alignment between elements
    - BasicBeamformer: Delay-and-sum beamforming
    - PhaseDifferenceDoA: 2-element direction finding

Phase 3 Components (Planned):
    - MUSICDoA: Subspace-based direction finding
    - AdaptiveBeamformer: MVDR/Capon beamforming
    - ArrayCalibration: Automated phase offset correction
"""

from .array_config import (
    ARRAY_PRESETS,
    SPEED_OF_LIGHT,
    ArrayConfig,
    ArrayElement,
    ArrayGeometry,
    ElementCalibration,
    ElementPosition,
    SynchronizationConfig,
    create_circular_4_element,
    create_linear_2_element,
    create_linear_4_element,
    create_rectangular_2x2,
    get_array_preset,
    list_array_presets,
)
from .array_controller import (
    AntennaArrayController,
    ArrayOperationMode,
    ArrayState,
    ElementState,
    SyncState,
)
from .timestamped_buffer import (
    TimestampedChunk,
    TimestampedSampleBuffer,
    TimestampedBufferStats,
)

__all__ = [
    # Constants
    "SPEED_OF_LIGHT",
    # Configuration classes
    "ArrayConfig",
    "ArrayElement",
    "ArrayGeometry",
    "ElementCalibration",
    "ElementPosition",
    "SynchronizationConfig",
    # Controller classes
    "AntennaArrayController",
    "ArrayOperationMode",
    "ArrayState",
    "ElementState",
    "SyncState",
    # Buffer classes
    "TimestampedChunk",
    "TimestampedSampleBuffer",
    "TimestampedBufferStats",
    # Preset functions
    "create_linear_2_element",
    "create_linear_4_element",
    "create_circular_4_element",
    "create_rectangular_2x2",
    "get_array_preset",
    "list_array_presets",
    # Preset registry
    "ARRAY_PRESETS",
]

__version__ = "0.1.0"

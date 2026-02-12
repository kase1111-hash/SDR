"""
Antenna Array Module for SDR.

Provides support for multi-SDR antenna array operation including:
- N-device orchestration with AntennaArrayController
- Timestamped sample buffers for synchronization
- Array geometry and calibration configuration
- Beamforming and direction finding algorithms
- Adaptive beamforming with interference suppression
- Automated array calibration

Example:
    from sdr_module.antenna_array import (
        AntennaArrayController,
        ArrayConfig,
        Beamformer,
        AdaptiveBeamformer,
        ArrayCalibrator,
        PhaseDifferenceDoA,
        create_linear_2_element,
    )

    # Create a 2-element linear array configuration
    config = create_linear_2_element(frequency=433e6)

    # Initialize and use the array
    with AntennaArrayController(config) as array:
        array.start_receive(sample_callback=process_samples)
        # ... do processing ...

    # Adaptive beamforming with interference suppression
    adaptive = AdaptiveBeamformer(config)
    result = adaptive.mvdr(signals, desired_azimuth=np.radians(30))

    # Calibrate the array
    calibrator = ArrayCalibrator(config)
    cal_result = calibrator.calibrate_correlation(signals)
    if cal_result.success:
        calibrator.apply_calibration(cal_result)

Phase 1 Components (Foundation):
    - TimestampedSampleBuffer: Sample buffer with timing metadata
    - ArrayConfig: Array geometry, elements, and calibration
    - AntennaArrayController: N-device orchestration

Phase 2 Components (Spatial Processing):
    - CrossCorrelator: Phase alignment between elements
    - Beamformer: Delay-and-sum and phase-shift beamforming
    - PhaseDifferenceDoA: 2-element direction finding
    - BeamscanDoA: Conventional beamscan DoA
    - MUSICDoA: Subspace-based direction finding

Phase 3 Components (Advanced):
    - AdaptiveBeamformer: MVDR/Capon/LCMV/GSC beamforming
    - ArrayCalibrator: Automated phase offset calibration
"""

from .adaptive_beamformer import (
    AdaptiveBeamformer,
    AdaptiveBeamformerState,
    AdaptiveMethod,
    MVDRResult,
)
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
from .beamformer import (
    BeamformerOutput,
    BeamformingMethod,
    BeamPattern,
    Beamformer,
    SteeringVector,
)
from .calibration import (
    ArrayCalibrator,
    CalibrationConfig,
    CalibrationMeasurement,
    CalibrationMethod,
    CalibrationResult,
    CalibrationState,
)
from .cross_correlator import (
    ArrayAlignmentResult,
    CorrelationResult,
    CrossCorrelator,
)
from .doa import (
    BeamscanDoA,
    DoAMethod,
    DoAResult,
    MultiSourceDoAResult,
    MUSICDoA,
    PhaseDifferenceDoA,
)
from .timestamped_buffer import (
    TimestampedBufferStats,
    TimestampedChunk,
    TimestampedSampleBuffer,
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
    # Cross-correlator
    "CrossCorrelator",
    "CorrelationResult",
    "ArrayAlignmentResult",
    # Beamformer
    "Beamformer",
    "BeamformerOutput",
    "BeamformingMethod",
    "BeamPattern",
    "SteeringVector",
    # Adaptive Beamformer
    "AdaptiveBeamformer",
    "AdaptiveBeamformerState",
    "AdaptiveMethod",
    "MVDRResult",
    # Calibration
    "ArrayCalibrator",
    "CalibrationConfig",
    "CalibrationMeasurement",
    "CalibrationMethod",
    "CalibrationResult",
    "CalibrationState",
    # Direction of Arrival
    "DoAMethod",
    "DoAResult",
    "MultiSourceDoAResult",
    "PhaseDifferenceDoA",
    "BeamscanDoA",
    "MUSICDoA",
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

__version__ = "0.3.0"
